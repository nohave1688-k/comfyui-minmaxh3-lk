from __future__ import annotations

import math
import os
import re
import json
import logging
import gc
import shutil
import subprocess
import tempfile
import threading
import time

import folder_paths
import nodes
import torch
import comfy.model_management as model_management
from comfy_api.latest import io
from comfy_api.latest._input_impl import VideoFromFile
from comfy_extras import nodes_audio

from .conditioning import build_conditioning
from .core import RUNTIME_TYPE, align_frame_count_down
from . import bundle as gh_bundle
from .long_video import (
    MAX_SEGMENTS,
    SEGMENT_TYPE,
    build_empty_plan_latent,
    build_plan_from_shot_nodes,
)


NODE_CATEGORY = "LK/MiniMax H3 Integration"
MAX_RESOLUTION = 16384
ASPECTS = {
    "adaptive": None,
    "16:9": 16 / 9,
    "9:16": 9 / 16,
    "3:2": 3 / 2,
    "2:3": 2 / 3,
    "4:3": 4 / 3,
    "3:4": 3 / 4,
    "1:1": 1.0,
    "21:9": 21 / 9,
}
DEFAULT_CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
DEFAULT_VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
DEFAULT_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"


def _uploaded_media(name, tooltip=None, hidden=False):
    """Store a custom-uploaded filename without Combo enum validation.

    The DOM panel owns the upload control.  A fixed upload Combo cannot
    represent an unused slot: ComfyUI validates every submitted Combo value,
    so the old ``(none)`` sentinel made the last empty media slots invalid.
    Empty strings are valid optional STRING values and are converted to None
    by the existing file loaders.
    """
    return io.String.Input(
        name,
        default="",
        optional=True,
        tooltip=tooltip,
        extra_dict={"hidden": True} if hidden else None,
    )


def _hidden_combo(name, options, default=None):
    return io.Combo.Input(
        name,
        options=options,
        default=default,
        extra_dict={"hidden": True},
    )


def _hidden_boolean(name, default):
    return io.Boolean.Input(name, default=default, extra_dict={"hidden": True})


def _hidden_float(name, default, minimum, maximum, step, round_value=None):
    return io.Float.Input(
        name,
        default=default,
        min=minimum,
        max=maximum,
        step=step,
        round=round_value,
        extra_dict={"hidden": True},
    )


def _hidden_int(name, default, minimum, maximum, step=1):
    return io.Int.Input(
        name,
        default=default,
        min=minimum,
        max=maximum,
        step=step,
        extra_dict={"hidden": True},
    )


def _default_model(options, preferred):
    return preferred if preferred in options else (options[0] if options else None)


def _release_text_encoder(clip) -> None:
    """Best-effort CLIP-only VRAM release across ComfyUI versions."""
    patcher = getattr(clip, "patcher", None)
    unloaded = False
    try:
        if patcher is not None:
            unload = getattr(model_management, "unload_model_and_clones", None)
            if not callable(unload):
                # Compatibility with ComfyUI builds that used the older name.
                unload = getattr(model_management, "unload_model_clones", None)
            if callable(unload):
                unload(patcher)
                unloaded = True
    except Exception as exc:
        logging.warning("MiniMax H3 could not explicitly unload the text encoder: %s", exc)
    finally:
        if patcher is not None and not unloaded:
            try:
                # Last-resort targeted offload. Never unload every model here:
                # both VAEs are still required by downstream nodes.
                detach = getattr(patcher, "detach", None)
                if callable(detach):
                    detach()
            except Exception as exc:
                logging.warning("MiniMax H3 could not detach the text encoder: %s", exc)
        try:
            empty_cache = getattr(model_management, "soft_empty_cache", None)
            if callable(empty_cache):
                empty_cache()
        except Exception as exc:
            logging.warning("MiniMax H3 could not clear released text-encoder cache: %s", exc)


def _round32(value: float) -> int:
    return max(32, int(value / 32 + 0.5) * 32)


def calculate_canvas(aspect: str, megapixels: float, source_width=None, source_height=None):
    ratio = ASPECTS.get(aspect, ASPECTS["16:9"])
    if ratio is None:
        try:
            ratio = float(source_width) / float(source_height)
        except (TypeError, ValueError, ZeroDivisionError):
            ratio = ASPECTS["16:9"]
    area = max(0.05, float(megapixels)) * 1024 * 1024
    # Round both axes independently from the requested aspect ratio. Using
    # area / rounded_width makes even 1:1 drift after the second rounding.
    width = _round32((area * ratio) ** 0.5)
    height = _round32((area / ratio) ** 0.5)
    if ratio == 1.0:
        height = width
    return min(MAX_RESOLUTION, width), min(MAX_RESOLUTION, height)


def calculate_length(duration_seconds: float) -> int:
    requested_frames = max(5, round(float(duration_seconds) * 24))
    return requested_frames + (5 - (requested_frames % 17)) % 17


def _load_image_file(value):
    if not value or value == "(none)":
        return None
    image, _mask = nodes.LoadImage().load_image(value)
    return image


def _load_audio_file(value):
    if not value or value == "(none)":
        return None
    try:
        return nodes_audio.LoadAudio.load(value)[0]
    except Exception as primary_error:
        # ComfyUI's PyAV loader can reject otherwise usable files (notably
        # some FLAC encoders or files containing a damaged metadata/frame
        # block).  Decode through FFmpeg to a plain PCM WAV as a compatibility
        # fallback.  This also broadens support to common formats such as
        # M4A/AAC, OGG/Opus, WMA and AIFF without changing H3's AUDIO contract.
        source_path = folder_paths.get_annotated_filepath(value)
        try:
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                import imageio_ffmpeg
                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

            with tempfile.TemporaryDirectory(prefix="minimax_h3_audio_") as temp_dir:
                wav_path = os.path.join(temp_dir, "decoded.wav")
                process = subprocess.run(
                    [
                        ffmpeg,
                        "-hide_banner", "-loglevel", "error", "-y",
                        "-fflags", "+discardcorrupt", "-err_detect", "ignore_err",
                        "-i", source_path,
                        "-vn", "-map", "0:a:0",
                        "-c:a", "pcm_s16le", wav_path,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                if process.returncode != 0 or not os.path.isfile(wav_path):
                    details = (process.stderr or "").strip()[-1200:]
                    raise RuntimeError(details or f"FFmpeg exited with code {process.returncode}")
                waveform, sample_rate = nodes_audio.load(wav_path)
                logging.info(
                    "MiniMax H3 decoded audio through FFmpeg compatibility fallback: %s",
                    os.path.basename(source_path),
                )
                return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        except Exception as fallback_error:
            raise RuntimeError(
                f"无法解码音频 {os.path.basename(str(source_path))}。"
                "节点支持常见音频格式，但该文件可能已损坏、扩展名与实际编码不符，"
                "或当前环境缺少可用的 FFmpeg。"
                f"\nComfyUI/PyAV: {primary_error}"
                f"\nFFmpeg fallback: {fallback_error}"
            ) from fallback_error


def _trim_audio_to_duration(audio, duration_seconds):
    """Return a same-rate AUDIO value exactly matching the requested duration."""
    if audio is None:
        return None
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    sample_count = max(1, round(float(duration_seconds) * sample_rate))
    trimmed = waveform[..., :sample_count]
    if trimmed.shape[-1] < sample_count:
        trimmed = torch.nn.functional.pad(trimmed, (0, sample_count - trimmed.shape[-1]))
    return {"waveform": trimmed, "sample_rate": sample_rate}


def _apply_audio_selection(audio, selection):
    """Slice an AUDIO value by a saved UI range without changing its rate."""
    if audio is None or not selection:
        return audio
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    start_seconds, end_seconds = selection
    start_sample = max(0, min(waveform.shape[-1], round(float(start_seconds) * sample_rate)))
    end_sample = max(start_sample + 1, min(waveform.shape[-1], round(float(end_seconds) * sample_rate)))
    return {"waveform": waveform[..., start_sample:end_sample], "sample_rate": sample_rate}


def _sample_video_frames(frames, source_fps, target_frames):
    """Convert the source to 24fps without changing playback speed.

    The caller supplies H3's maximum target frame count. Longer references are
    clipped to that window, while shorter references keep their own duration
    instead of repeating their final frame to fill the generated-video length.
    """
    if frames is None or frames.shape[0] <= 1:
        return frames
    source_fps = float(source_fps or 0)
    requested = max(1, int(target_frames))
    # Keep the old helper contract for direct callers that pass a legacy FPS
    # value. The node itself passes the calculated target frame count.
    if requested <= 30:
        target_fps = requested
        duration = min(frames.shape[0] / source_fps, 360 / 24) if source_fps > 0 else 0
        target_frames = min(360, max(48, round(duration * target_fps)))
        if source_fps <= 0:
            sampled = frames[:target_frames]
        else:
            source_end = min(frames.shape[0] - 1, max(0, math.ceil(duration * source_fps) - 1))
            indices = torch.linspace(0, source_end, target_frames).round().long()
            return frames[indices]
    else:
        if source_fps > 0:
            source_duration_frames = max(1, round(frames.shape[0] * 24 / source_fps))
            target_frames = min(requested, source_duration_frames)
        else:
            target_frames = min(requested, frames.shape[0])
    if source_fps <= 0:
        source_indices = torch.arange(target_frames, dtype=torch.long)
    else:
        # Preserve source playback rate: 60fps -> 24fps drops frames at the
        # 24fps cadence; 12fps -> 24fps repeats source frames at that cadence.
        source_indices = torch.floor(torch.arange(target_frames, dtype=torch.float32) * source_fps / 24).long()
    source_indices = source_indices.clamp(max=frames.shape[0] - 1)
    return frames[source_indices][:target_frames]


def _load_video_frames(value, target_frames):
    if not value or value == "(none)":
        return None, None
    path = folder_paths.get_annotated_filepath(value)
    components = VideoFromFile(path).get_components()
    frames = components.images
    frames = _sample_video_frames(frames, components.frame_rate, target_frames)
    return frames, components.audio


def _load_video_object(video, target_frames):
    """Decode a connected io.Video socket object into (frames, soundtrack)."""
    if video is None:
        return None, None
    try:
        components = video.get_components()
    except Exception:
        return None, None
    frames = components.images
    if frames is None:
        return None, None
    frames = _sample_video_frames(frames, components.frame_rate, target_frames)
    return frames, components.audio


def _as_audio_value(audio):
    """Normalize an io.Audio socket value into the internal dict shape."""
    if audio is None:
        return None
    if isinstance(audio, dict):
        return audio if audio.get("waveform") is not None else None
    waveform = getattr(audio, "waveform", None)
    if waveform is None:
        return None
    return {"waveform": waveform, "sample_rate": getattr(audio, "sample_rate", None)}


_CONNECTED_MEDIA_KINDS = ("first_frame", "last_frame", "image", "video", "audio")
_CONNECTED_APPLY_MODES = ("all segments", "per segment (in order)")


def _connected_media_inputs():
    """Native connectable media sockets (V3 Autogrow).

    Five socket groups — first/last frame, images, videos, audios — rendered
    by ComfyUI as regular input sockets: native drag, snap and serialization.
    A connected socket OVERRIDES the panel's same-kind upload.  Declared after
    every legacy widget so existing workflow widget positions remain unchanged.
    """
    inputs = [
        io.Combo.Input(
            "conn_apply",
            options=list(_CONNECTED_APPLY_MODES),
            default="all segments",
            tooltip=(
                "How connected media map to segments: 'all segments' applies the "
                "same references to every segment in range; 'per segment (in "
                "order)' gives the first/last frame to the range's first/last "
                "segment and distributes images, videos and audios one segment "
                "each in order."
            ),
        ),
    ]
    for group_id, factory, label, count in (
        ("conn_first_frame", io.Image.Input, "首帧（连线）", 1),
        ("conn_last_frame", io.Image.Input, "尾帧（连线）", 1),
        # Per-segment groups follow the segment count: the frontend materializes
        # one slot per segment, so the pool must cover every possible segment.
        ("conn_image", io.Image.Input, "参考图片（连线）", MAX_SEGMENTS),
        ("conn_video", io.Video.Input, "参考视频（连线）", MAX_SEGMENTS),
        ("conn_audio", io.Audio.Input, "参考音频（连线）", MAX_SEGMENTS),
    ):
        inputs.append(io.Autogrow.Input(
            group_id,
            optional=True,
            template=io.Autogrow.TemplatePrefix(
                input=factory(
                    group_id,
                    optional=True,
                    display_name=label,
                    tooltip=f"{label}：连接后覆盖面板上传的同槽位素材。",
                ),
                prefix=f"{group_id}_", min=0, max=count),
            tooltip=f"{label}：连接后覆盖面板上传的同槽位素材。",
        ))
    return inputs


def _collect_connected_media(conn_media):
    """Flatten the Autogrow socket values into one media set.

    Each Autogrow input arrives as a dict keyed by its socket names
    (``conn_image_0`` ...), ordered by ordinal.  Image batches are split into
    one reference per frame.  The result is distributed across segments by
    ``build_multi_plan`` according to the ``conn_apply`` mode.  The pool holds
    one reference per segment (MAX_SEGMENTS); per-segment kind limits are
    enforced downstream when each segment bundle is built.
    """
    collected = {"first_frame": None, "last_frame": None, "images": [], "videos": [], "audios": []}
    for key, bundle in (conn_media or {}).items():
        if not isinstance(bundle, dict):
            continue
        kind = key[5:] if key.startswith("conn_") else None
        if kind not in _CONNECTED_MEDIA_KINDS:
            continue
        values = []
        for slot in sorted(bundle, key=lambda name: int(name.rsplit("_", 1)[-1]) if name.rsplit("_", 1)[-1].isdigit() else 0):
            value = bundle.get(slot)
            if value is None:
                continue
            if kind == "audio":
                value = _as_audio_value(value)
            if value is None:
                continue
            values.append(value)
        if not values:
            continue
        if kind in ("first_frame", "last_frame"):
            collected[kind] = values[0]
        elif kind == "image":
            for tensor in values:
                for frame_index in range(int(tensor.shape[0])):
                    if len(collected["images"]) >= MAX_SEGMENTS:
                        break
                    collected["images"].append(tensor[frame_index:frame_index + 1])
        elif kind == "video":
            collected["videos"].extend(values[:MAX_SEGMENTS])
        elif kind == "audio":
            collected["audios"].extend(values[:MAX_SEGMENTS])
    collected["images"] = collected["images"][:MAX_SEGMENTS]
    collected["videos"] = collected["videos"][:MAX_SEGMENTS]
    collected["audios"] = collected["audios"][:MAX_SEGMENTS]
    return collected


class MiniMaxH3ComposerGH(io.ComfyNode):
    # The composer bakes the loaded CLIP/VAE objects and the shot plan into
    # its outputs; downstream conditioning embeds those objects.  Re-execute
    # every queue (lanxi semantics) so the conditioning chain can never ride
    # a stale cached bundle across runs.
    NOT_IDEMPOTENT = True
    """One-stop film composer: whole-film prompt + model selection + shot
    collection + the empty-AV template, merged into a single node.

    Everything downstream (分段引导 / 分段采样 / 二跑分段精修 / T8 contract
    nodes / AV 解码) pulls from this node's outputs; the template output also
    removes the cycle-breaker node from T8-style primitive chains.  The older
    LKH3Settings / LKH3FilmPrompt / LKH3Shots / LKH3AVTemplate nodes stay
    registered for existing workflows.
    """

    @classmethod
    def define_schema(cls):
        schema_inputs = [
            io.String.Input(
                "film_prompt",
                multiline=True,
                dynamic_prompts=True,
                default="",
                tooltip="拼接到每个镜头提示词前面的全片内容（主体定义、风格块、<Picture n> 标签）。留空则不追加。",
            ),
            io.Combo.Input(
                "clip_name",
                options=folder_paths.get_filename_list("text_encoders"),
                default=_default_model(folder_paths.get_filename_list("text_encoders"), DEFAULT_CLIP),
            ),
            io.Combo.Input(
                "video_vae_name",
                options=folder_paths.get_filename_list("vae"),
                default=_default_model(folder_paths.get_filename_list("vae"), DEFAULT_VIDEO_VAE),
            ),
            io.Combo.Input(
                "audio_vae_name",
                options=folder_paths.get_filename_list("vae"),
                default=_default_model(folder_paths.get_filename_list("vae"), DEFAULT_AUDIO_VAE),
            ),
            io.Combo.Input("aspect", options=list(ASPECTS), default="adaptive"),
            io.Float.Input("megapixels", default=0.5, min=0.2, max=2.0, step=0.1, round=0.1),
            io.Combo.Input(
                "ref_image_size",
                options=["match", "1.2x", "1.5x", "2x", "max"],
                default="match",
            ),
            io.Combo.Input(
                "audio_mode",
                options=["lock_source", "reference_only", "native"],
                default="lock_source",
            ),
            io.Boolean.Input(
                "latent_continue",
                default=True,
                tooltip="Carry the previous shot's latent tail into the next shot.",
            ),
            io.Int.Input(
                "range_start", default=1, min=1, max=256, step=1,
                tooltip="First shot to run (1-based, in connection order: 链上各镜依序展开)。",
            ),
            io.Int.Input(
                "range_end", default=1, min=1, max=256, step=1,
                tooltip="Last shot to run (inclusive).  超出实际分镜数时自动收窄。",
            ),

            io.String.Input(
                "project_dir", default="", optional=True,
                tooltip="逐镜缓存目录：留空 = 自动使用 output/LK_H3_projects/ 下的按片子目录，按片自动建束（改哪镜只重跑哪镜），超限自动整束清理（LK_BUNDLE_CACHE_GB 默认16GB，0=关；LK_BUNDLE_CACHE_DAYS 按龄，默认关）。也可填自定义路径。LK_SEGMENT_CACHE=0 完全关闭缓存。",
            ),
        ]
        # v3 起：散点/组的接入统一走 分镜序列（链式串联），原 分镜 1-8 口
        # 已移除。旧工作流请改用各节点的 分镜序列 输出串接后接入此口。
        schema_inputs.append(io.Custom("LK_H3_SHOT_GROUP").Input(
            "shot_group_chain",
            display_name="分镜序列",
            optional=True,
            tooltip="分镜序列：接 分镜组/散点分镜 的「分镜序列」输出（多节点首尾相接）。顺序 = 链上各镜依序展开，无镜数上限。",
        ))
        return io.Schema(
            node_id="LKH3Composer",
            display_name="MiniMaxH3 长视频编排（LK）",
            category=NODE_CATEGORY,
            description=(
                "One-stop long-video composer: whole-film prompt, model selection, "
                "shot collection, and the empty-AV template in one node.  Feed its "
                "outputs to the samplers / 分段引导 / 二跑分段精修 / AV 解码."
            ),
            inputs=schema_inputs,
            outputs=[
                io.Custom(SEGMENT_TYPE).Output(
                    display_name="segments",
                    tooltip="Long-video segment plan; connect to the samplers / 分段引导 / 二跑分段精修.",
                ),
                io.Latent.Output(
                    "av_template", display_name="空白AV模板",
                    tooltip="整段空白 AV 模板（含 lk_layout）— 接 双钟采样/T8契约 的 av_latent 和 SamplerCustomAdvanced 的 latent_image。",
                ),
                io.Custom("CLIP").Output(
                    "clip", display_name="CLIP",
                    tooltip="裸 CLIP 输出 — 给本包之外的节点（如 T8/lanxi 的采样、解码节点）单独拉线用。",
                ),
                io.Vae.Output(
                    "video_vae", display_name="视频 VAE",
                    tooltip="裸视频 VAE 输出 — 给本包之外的节点单独拉线用。",
                ),
                io.Vae.Output(
                    "audio_vae", display_name="音频 VAE",
                    tooltip="裸音频 VAE 输出 — 给本包之外的节点单独拉线用。",
                ),
                # Appended LAST so pre-bundle canvases keep every bare wire on
                # its original slot - the bundle is purely additive.
                io.Custom(RUNTIME_TYPE).Output(
                    "runtime", display_name="解码设置",
                    tooltip="CLIP + 视频 VAE + 音频 VAE 捆绑 — 一根线接本包 分段引导/二跑分段精修/AV 解码 的 解码设置 输入。",
                ),
            ],
        )

    @classmethod
    def execute(cls, film_prompt, clip_name, video_vae_name, audio_vae_name,
                aspect, megapixels, ref_image_size, audio_mode,
                latent_continue=True, range_start=1, range_end=1,
                project_dir="", shot_group_chain=None):
        # Loaders cache by name — the same objects the samplers would load.
        clip = nodes.CLIPLoader().load_clip(clip_name, "minimax")[0] if clip_name else None
        video_vae = nodes.VAELoader().load_vae(video_vae_name)[0] if video_vae_name else None
        audio_vae = nodes.VAELoader().load_vae(audio_vae_name)[0] if audio_vae_name else None
        # 分镜序列展平：链值本身就是有序 shot dict 列表（前序分镜 + 各组
        # 依序累积），即全片顺序；参考上下文补全在此之后执行。
        # is_input_list 下 shot_group_chain 是"链的列表"（每条连线一个有序
        # shot dict 列表）；多分支全部展平，连线顺序即片内顺序。兼容单连线
        # 旧形态（list of shot dicts）。
        chain_shots = []
        if shot_group_chain is not None:
            if not isinstance(shot_group_chain, list):
                raise ValueError("分镜序列 输入必须是 分镜组/散点分镜 的「分镜序列」输出。")
            if any(isinstance(item, list) for item in shot_group_chain):
                for chain in shot_group_chain:
                    if isinstance(chain, list):
                        chain_shots.extend(s for s in chain if isinstance(s, dict))
            else:
                chain_shots = [shot for shot in shot_group_chain if isinstance(shot, dict)]
        shots = [shot for shot in chain_shots if isinstance(shot, dict)]
        fill_inherited_refs(shots)
        if shots:
            logging.info("[LKH3] 分镜序列：全片 %d 镜", len(shots))
        plan = build_plan_from_shot_nodes(
            settings={"clip_name": clip_name, "video_vae_name": video_vae_name,
                      "audio_vae_name": audio_vae_name},
            film_settings={"film_prompt": film_prompt, "aspect": aspect,
                           "megapixels": megapixels, "ref_image_size": ref_image_size,
                           "audio_mode": audio_mode},
            shots=shots, latent_continue=latent_continue,
            range_start=range_start, range_end=range_end,
            clip=clip, video_vae=video_vae, audio_vae=audio_vae,
            max_segments=max(MAX_SEGMENTS, len(shots)),
        )
        # 项目束 = 纯逐镜性能缓存：目录留空时自动在 output/LK_H3_projects/
        # 下按片身份（film_key）开**子束**——多部片各占一个文件夹，互不串
        # 缓存；束内容全部可再生，sweep 按阈值整束清理。身份刻意不含逐镜
        # 内容——改某一镜的提示词不变更束，只有结构变更（模型/画布/镜数）
        # 才换束。LK_SEGMENT_CACHE=0 = 完全关闭（不建目录、不读写缓存）。
        plan["bundle_dir"] = plan["film_key"] = None
        plan["cache_enabled"] = False
        if os.environ.get("LK_SEGMENT_CACHE", "1") != "0":
            identity = gh_bundle.film_identity(
                clip_name=clip_name, video_vae_name=video_vae_name,
                audio_vae_name=audio_vae_name,
                width=plan["width"], height=plan["height"],
                segment_count=len(plan["segments"]),
            )
            bundle_root = (project_dir or "").strip() or os.path.join(
                folder_paths.get_output_directory(), "LK_H3_projects",
                gh_bundle.canonical_hash(identity)[:8])
            resolved = gh_bundle.resolve_bundle(bundle_root, identity)
            if resolved is not None:
                # 参考媒体只算内容 hash（喂缓存键），不落盘副本。
                plan["media_hashes"] = gh_bundle.shot_media_digests(shots)
                plan["bundle_dir"] = resolved["dir"]
                plan["film_key"] = resolved["film_key"]
                plan["cache_enabled"] = True
        start = int(plan["range"]["start"]) - 1
        end = int(plan["range"]["end"]) - 1
        template = build_empty_plan_latent(plan, start, end)
        runtime = {"clip": clip, "video_vae": video_vae, "audio_vae": audio_vae}
        # The bundle feeds THIS pack's nodes (one wire per consumer); the bare
        # clip/VAE outputs stay so third-party nodes (T8/lanxi samplers,
        # decoders, generic VAE consumers) keep working unchanged.  Order MUST
        # mirror the outputs list: segments, template, CLIP, video VAE,
        # audio VAE, runtime - a stray wire got the CLIP object as its "VAE"
        # when this tuple lagged the definition reorder.
        return io.NodeOutput(plan, template, clip, video_vae, audio_vae, runtime)

class MiniMaxH3ShotGH(io.ComfyNode):
    """One shot of the long video: duration, prompt, reference media and
    per-shot LoRA rows (managed by the panel through ``shot_json``).
    Output chains through 前序分镜/分镜序列 into the composer.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LKH3Shot",
            display_name="MiniMaxH3分镜（LK）",
            category=NODE_CATEGORY,
            description="One long-video shot. Media arrives through the link sockets; LoRA rows are managed by the panel.",
            inputs=[
                io.Int.Input(
                    "duration", default=5, min=2, max=30, step=1,
                    tooltip="Shot length in seconds (snapped to the model's frame grid).",
                ),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True, default=""),
                io.Image.Input("first_frame", optional=True, tooltip="Optional first frame."),
                io.Image.Input("last_frame", optional=True, tooltip="Optional last frame."),
                io.Image.Input("ref_image_1", optional=True),
                io.Image.Input("ref_image_2", optional=True),
                io.Image.Input("ref_image_3", optional=True),
                io.Image.Input("ref_image_4", optional=True),
                io.Image.Input("ref_image_5", optional=True),
                io.Image.Input("ref_image_6", optional=True),
                io.Video.Input("ref_video_1", optional=True),
                io.Video.Input("ref_video_2", optional=True),
                io.Audio.Input("ref_audio_1", optional=True),
                io.Audio.Input("ref_audio_2", optional=True),
                io.Audio.Input("ref_audio_3", optional=True),
                io.String.Input(
                    "shot_json", default="", optional=True, extra_dict={"hidden": True},
                    tooltip="Per-shot LoRA rows (managed by the panel)",
                ),
                # Appended LAST so legacy canvases keep every socket on its
                # original index - the chain input is purely additive.
                io.Custom("LK_H3_SHOT_GROUP").Input(
                    "group_in", optional=True,
                    display_name="前序分镜",
                    tooltip="接上一个 分镜 或 分镜组 的「分镜序列」输出——本镜排在序列尾。",
                ),
            ],
            outputs=[
                io.Custom("LK_H3_SHOT_GROUP").Output(
                    display_name="分镜序列",
                    tooltip="以本镜结尾的分镜序列。接下一镜/组的 前序分镜，或接 Composer 的 分镜序列。",
                ),
            ],
        )

    @classmethod
    def execute(cls, duration, prompt, first_frame=None, last_frame=None,
                ref_image_1=None, ref_image_2=None, ref_image_3=None,
                ref_image_4=None, ref_image_5=None, ref_image_6=None,
                ref_video_1=None, ref_video_2=None,
                ref_audio_1=None, ref_audio_2=None, ref_audio_3=None,
                shot_json="", group_in=None):
        if group_in is not None and not isinstance(group_in, list):
            raise ValueError("前序分镜 输入必须是分镜序列（请连接上一个 分镜/分镜组 的「分镜序列」输出）。")
        loras = []
        audio_mode = ""
        try:
            parsed = json.loads(shot_json) if isinstance(shot_json, str) and shot_json.strip() else {}
            if isinstance(parsed, dict):
                loras = parsed.get("loras") or []
                shot_audio = str(parsed.get("audio_mode") or "").strip().lower()
                if shot_audio in ("lock_source", "reference_only", "native"):
                    audio_mode = shot_audio
        except (TypeError, ValueError):
            pass
        images = [t for t in (ref_image_1, ref_image_2, ref_image_3,
                              ref_image_4, ref_image_5, ref_image_6) if t is not None]
        videos = [t for t in (ref_video_1, ref_video_2) if t is not None]
        audios = [t for t in (ref_audio_1, ref_audio_2, ref_audio_3) if t is not None]
        shot_dict = {
            "duration": duration,
            "prompt": prompt or "",
            "first_frame": first_frame,
            "last_frame": last_frame,
            "images": images,
            "videos": videos,
            "audios": audios,
            "loras": loras,
            "audio_mode": audio_mode,
        }
        # 组链输出：上游链 + 本镜（散点直插组链；无上游时为单镜 1 元素链）。
        chain = [item for item in (group_in or []) if isinstance(item, dict)]
        chain.append(shot_dict)
        return io.NodeOutput(chain)


GROUP_MAX_SHOTS = 3
MAX_SHOT_LORAS = 4


def _load_video_by_name(name):
    """按 input 目录文件名加载 V3 Video 对象（分镜组弹出镜上传通道）。"""
    if not name or name == "(none)":
        return None
    try:
        return VideoFromFile(folder_paths.get_annotated_filepath(name))
    except Exception as exc:
        logging.info("[LKH3] 分镜组：参考视频 %s 加载失败（跳过）：%s", name, exc)
        return None


def fill_inherited_refs(shots) -> None:
    """Composer 展平后的参考上下文补全（原地修改）。

    组节点只能看到组链里的上游——组链之前的散点分镜对它是不可见的，所以
    「继承」标记为空参考的第一组镜头会在这里兜底：从展平序里最近的已声明
    镜（含散点）填入参考。散点分镜不带 refs_source 标记，永不被填充
    （散点语义 = 接了才算）。填充后的镜自身成为新的上下文，后续继承镜
    依次取到同一套参考。"""
    context = None
    for shot in shots:
        if not isinstance(shot, dict):
            continue
        if shot.get("images") or shot.get("videos") or shot.get("audios"):
            context = shot
            continue
        if shot.get("refs_source") == "inherit" and context is not None:
            shot["images"] = list(context.get("images") or [])
            shot["videos"] = list(context.get("videos") or [])
            shot["audios"] = list(context.get("audios") or [])
            context = shot


def _group_pool_values(pool):
    """Flatten one Autogrow pool dict into an ordered list of non-None values
    (slot ordinal order)."""
    values = []
    if isinstance(pool, dict):
        for key in sorted(pool, key=lambda k: int(k.rsplit("_", 1)[-1])
                          if k.rsplit("_", 1)[-1].isdigit() else 0):
            if pool.get(key) is not None:
                values.append(pool[key])
    return values


class MiniMaxH3ShotGroupGH(io.ComfyNode):
    """分镜组：一个节点收纳 ≤3 个分镜，链式串联定顺序。

    分镜1（主分镜）保留全套参考 socket；分镜2/3 在面板卡片内编辑（提示词/
    时长/音频模式/LoRA），参考默认「继承主分镜」，需要不同的参考时
    用卡片里的 ＋图片/＋视频/＋音频/＋首帧/＋尾帧 按钮上传（lanxi 操作台
    同款按钮驱动方式）。输出 = 上游链 + 本组活动分镜（prompt 非空或有媒体
    才算活动），接到 Composer 的「分镜序列」口；多节点首尾相接即完整片顺序。
    """

    @classmethod
    def define_schema(cls):
        inputs = [
            io.Custom("LK_H3_SHOT_GROUP").Input(
                "group_in", optional=True, display_name="前序分镜",
                tooltip="链接上一个分镜组的输出（多组首尾相接）。组链传递顺序与参考上下文：各镜「参考」开关默认「继承」，无自己连线/上传时自动取链上就近声明的参考；提示词/LoRA/音频模式/首尾帧等参数永不跨组。",
            ),
            io.String.Input(
                "group_json", default="", optional=True, extra_dict={"hidden": True},
                tooltip="Per-shot group state (managed by the panel)",
            ),
        ]
        return io.Schema(
            node_id="LKH3ShotGroup",
            display_name="MiniMaxH3 分镜组（LK）",
            category=NODE_CATEGORY,
            description=(
                "One shot group: up to 3 shots in one node, all reference media "
                "uploaded via card buttons (lanxi 操作台 style — no reference "
                "lines).  Each shot can 跟随上游链上下文 (default) or set its "
                "own uploads.  Chain groups through group_in/group_out; connect "
                "the last group to the composer's 分镜序列 input."
            ),
            inputs=inputs,
            outputs=[
                io.Custom("LK_H3_SHOT_GROUP").Output(
                    display_name="分镜序列",
                    tooltip="本组活动分镜 + 前序分镜 组成的分镜序列。接下一镜/组的 前序分镜，或接 Composer 的 分镜序列。",
                ),
            ],
        )

    @classmethod
    def execute(cls, group_in=None, group_json=""):
        if group_in is not None and not isinstance(group_in, list):
            raise ValueError("前序分镜 输入必须是分镜序列（请连接上一个 分镜/分镜组 的「分镜序列」输出）。")

        # 面板状态：三行（缺行用默认值容错，风格同 LKH3Shot 的解析）。弹出镜
        # 的参考媒体是卡片按钮上传的文件名（lanxi 操作台同款按钮驱动）。
        rows = []
        try:
            parsed = json.loads(group_json) if isinstance(group_json, str) and group_json.strip() else {}
            if isinstance(parsed, dict):
                rows = parsed.get("shots") or []
            elif isinstance(parsed, list):
                rows = parsed
        except (TypeError, ValueError):
            rows = []

        def _name_list(value):
            if not isinstance(value, list):
                return []
            return [str(item).strip() for item in value if str(item or "").strip()]

        def _name_or_none(value):
            name = str(value or "").strip()
            return name or None

        states = []
        for i in range(GROUP_MAX_SHOTS):
            row = rows[i] if i < len(rows) and isinstance(rows[i], dict) else {}
            shot_audio = str(row.get("audio_mode") or "").strip().lower()
            if shot_audio not in ("lock_source", "reference_only", "native"):
                shot_audio = ""
            try:
                duration = max(2, min(30, int(row.get("duration") or 5)))
            except (TypeError, ValueError):
                duration = 5
            states.append({
                "duration": duration,
                "prompt": str(row.get("prompt") or ""),
                "audio_mode": shot_audio,
                "loras": list(row.get("loras") or [])[:MAX_SHOT_LORAS] if isinstance(row.get("loras"), list) else [],
                "inherit_refs": bool(row.get("inherit_refs", True)),
                # enabled = 用户在面板里展开/添加过该弹出镜（意图标记）。没有
                # 它，从未触碰的分镜3空行会因「继承」凭空激活成多余的一镜。
                "enabled": bool(row.get("enabled", False)),
                "images": _name_list(row.get("images")),
                "videos": _name_list(row.get("videos")),
                "audios": _name_list(row.get("audios")),
                "first_frame": _name_or_none(row.get("first_frame")),
                "last_frame": _name_or_none(row.get("last_frame")),
            })

        # 主镜参考 = 卡片按钮上传（lanxi 按钮驱动；组节点无参考连线）
        main_images, main_videos, main_audios = [], [], []
        main_first = main_last = None
        if states[0]["images"] or states[0]["videos"] or states[0]["audios"] \
                or states[0]["first_frame"] or states[0]["last_frame"]:
            main_images = [img for img in (_load_image_file(name) for name in states[0]["images"])
                           if img is not None]
            main_videos = [vid for vid in (_load_video_by_name(name) for name in states[0]["videos"])
                           if vid is not None]
            main_audios = [aud for aud in (_load_audio_file(name) for name in states[0]["audios"])
                           if aud is not None]
            main_first = _load_image_file(states[0]["first_frame"]) if states[0]["first_frame"] else None
            main_last = _load_image_file(states[0]["last_frame"]) if states[0]["last_frame"] else None

        shots_out = [item for item in (group_in or []) if isinstance(item, dict)]
        kept, skipped = [], []

        def _nearest_declared_refs():
            """组链内的参考上下文：最近一个自带参考媒体的上游组镜。
            「独立且空」的镜透明——不截断上下文流。散点分镜不在组链里，
            由 Composer 的展平补全遍历兜底（refs_source="inherit" 标记）。"""
            for shot in reversed(upstream):
                if shot.get("images") or shot.get("videos") or shot.get("audios"):
                    return (shot.get("images"), shot.get("videos"), shot.get("audios"))
            return (None, None, None)

        upstream = shots_out

        # —— 分镜1（主）：own socket > 继承上游就近声明 > 空。继承只填参考，
        #    永不激活镜头（激活只看自身内容——空主镜不凭继承出片）。——
        state1 = states[0]
        if main_images or main_videos or main_audios:
            main_refs = (main_images, main_videos, main_audios)
            main_source = "own"
        elif state1["inherit_refs"]:
            main_refs = _nearest_declared_refs()
            main_source = "inherit"
        else:
            main_refs = (None, None, None)
            main_source = "none"
        main_active = bool(state1["prompt"].strip()) or bool(
            main_images or main_videos or main_audios
        ) or main_first is not None or main_last is not None
        if main_active:
            kept.append("分镜1")
            shots_out.append({
                "duration": state1["duration"],
                "prompt": state1["prompt"],
                "first_frame": main_first,
                "last_frame": main_last,
                "images": main_refs[0] or [],
                "videos": main_refs[1] or [],
                "audios": main_refs[2] or [],
                "loras": state1["loras"],
                "audio_mode": state1["audio_mode"],
                "refs_source": main_source,
            })
        else:
            skipped.append("分镜1")

        # —— 分镜2/3：参考来源 own上传 > 继承主镜解析结果（主镜未参与时退
        #    到就近上游声明）> 空。首尾帧不参与继承——逐镜锚定，不是参考。——
        for n in (2, 3):
            state = states[n - 1]
            gated = state["enabled"] or bool(state["prompt"].strip())                 or state["images"] or state["videos"] or state["audios"]                 or state["first_frame"] or state["last_frame"]
            # 弹出镜参与门：面板展开/添加过（enabled）或本身有内容（含上传
            # 的独立首尾帧）。默认空行永不凭继承激活。
            if not gated:
                skipped.append(f"分镜{n}")
                continue

            # 加载弹出镜自己上传的参考（继承模式下为空列表，不产生 IO）。
            loaded_images = [img for img in (_load_image_file(name) for name in state["images"])
                             if img is not None]
            loaded_videos = [vid for vid in (_load_video_by_name(name) for name in state["videos"])
                             if vid is not None]
            loaded_audios = [aud for aud in (_load_audio_file(name) for name in state["audios"])
                             if aud is not None]
            loaded_first = _load_image_file(state["first_frame"]) if state["first_frame"] else None
            loaded_last = _load_image_file(state["last_frame"]) if state["last_frame"] else None
            has_own = bool(loaded_images or loaded_videos or loaded_audios
                           or loaded_first is not None or loaded_last is not None)

            if has_own:
                images, videos, audios = loaded_images, loaded_videos, loaded_audios
                refs_source = "own"
            elif state["inherit_refs"] and main_active:
                # 主镜已参与 → 继承主镜解析结果（同一对象即可：refs/ 按
                # 内容去重，connected 通道共享张量）。
                images, videos, audios = main_refs
                refs_source = "inherit"
            elif state["inherit_refs"]:
                images, videos, audios = _nearest_declared_refs()
                refs_source = "inherit"
            else:
                images, videos, audios = ([], [], [])
                refs_source = "none"
            active = bool(state["prompt"].strip()) or images or videos or audios                 or loaded_first is not None or loaded_last is not None
            label = f"分镜{n}"
            if not active:
                skipped.append(label)
                continue
            kept.append(label)
            shots_out.append({
                "duration": state["duration"],
                "prompt": state["prompt"],
                "first_frame": loaded_first,
                "last_frame": loaded_last,
                "images": images or [],
                "videos": videos or [],
                "audios": audios or [],
                "loras": state["loras"],
                "audio_mode": state["audio_mode"],
                "refs_source": refs_source,
            })
        if skipped:
            logging.info("[LKH3] 分镜组：跳过无内容的 %s", "、".join(skipped))
        if kept or shots_out:
            logging.info(
                "[LKH3] 分镜组：活动 %s%s",
                "、".join(kept) or "（无）",
                f"（含上游 {len(shots_out) - len(kept)} 镜）" if len(shots_out) > len(kept) else "",
            )
        return io.NodeOutput(shots_out)