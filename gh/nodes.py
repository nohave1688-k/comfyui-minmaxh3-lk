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
from ..refmod.nodes import load_visual_mods
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
                "range_start", default=1, min=1, max=MAX_SEGMENTS, step=1,
                tooltip="First shot to run (1-based, in connection order).",
            ),
            io.Int.Input(
                "range_end", default=1, min=1, max=MAX_SEGMENTS, step=1,
                tooltip="Last shot to run (inclusive).",
            ),
        ]
        for i in range(1, MAX_SEGMENTS + 1):
            schema_inputs.append(io.Custom("LK_H3_SHOT").Input(
                f"shot_{i}",
                display_name=f"分镜 {i}",
                optional=True,
                tooltip=f"Shot {i} in the running order.",
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
                shot_1=None, shot_2=None, shot_3=None, shot_4=None,
                shot_5=None, shot_6=None, shot_7=None, shot_8=None):
        # Loaders cache by name — the same objects the samplers would load.
        clip = nodes.CLIPLoader().load_clip(clip_name, "minimax")[0] if clip_name else None
        video_vae = nodes.VAELoader().load_vae(video_vae_name)[0] if video_vae_name else None
        audio_vae = nodes.VAELoader().load_vae(audio_vae_name)[0] if audio_vae_name else None
        shots = [shot for shot in (shot_1, shot_2, shot_3, shot_4,
                                   shot_5, shot_6, shot_7, shot_8)
                 if shot is not None]
        plan = build_plan_from_shot_nodes(
            settings={"clip_name": clip_name, "video_vae_name": video_vae_name,
                      "audio_vae_name": audio_vae_name},
            film_settings={"film_prompt": film_prompt, "aspect": aspect,
                           "megapixels": megapixels, "ref_image_size": ref_image_size,
                           "audio_mode": audio_mode},
            shots=shots, latent_continue=latent_continue,
            range_start=range_start, range_end=range_end,
            clip=clip, video_vae=video_vae, audio_vae=audio_vae,
        )
        start = int(plan["range"]["start"]) - 1
        end = int(plan["range"]["end"]) - 1
        template = build_empty_plan_latent(plan, start, end)
        runtime = {"clip": clip, "video_vae": video_vae, "audio_vae": audio_vae}
        # The bundle feeds THIS pack's nodes (one wire per consumer); the bare
        # clip/VAE outputs stay so third-party nodes (T8/lanxi samplers,
        # decoders, generic VAE consumers) keep working unchanged.
        return io.NodeOutput(plan, template, runtime, clip, video_vae, audio_vae)

class MiniMaxH3ShotGH(io.ComfyNode):
    """One shot of the long video: duration, prompt, reference media and
    per-shot LoRA/RefMod rows (managed by the panel through ``shot_json``).
    Output connects to the MiniMaxH3分镜合成 node.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LKH3Shot",
            display_name="MiniMaxH3分镜（LK）",
            category=NODE_CATEGORY,
            description="One long-video shot. Media arrives through the link sockets; LoRA/RefMod rows are managed by the panel.",
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
                    tooltip="Per-shot LoRA/RefMod rows (managed by the panel)",
                ),
            ],
            outputs=[
                io.Custom("LK_H3_SHOT").Output(
                    display_name="shot",
                    tooltip="Connect to a MiniMaxH3分镜合成 slot (分镜 1-8).",
                ),
            ],
        )

    @classmethod
    def execute(cls, duration, prompt, first_frame=None, last_frame=None,
                ref_image_1=None, ref_image_2=None, ref_image_3=None,
                ref_image_4=None, ref_image_5=None, ref_image_6=None,
                ref_video_1=None, ref_video_2=None,
                ref_audio_1=None, ref_audio_2=None, ref_audio_3=None,
                shot_json=""):
        mods, loras = [], []
        try:
            parsed = json.loads(shot_json) if isinstance(shot_json, str) and shot_json.strip() else {}
            if isinstance(parsed, dict):
                mods = parsed.get("mods") or []
                loras = parsed.get("loras") or []
        except (TypeError, ValueError):
            pass
        images = [t for t in (ref_image_1, ref_image_2, ref_image_3,
                              ref_image_4, ref_image_5, ref_image_6) if t is not None]
        videos = [t for t in (ref_video_1, ref_video_2) if t is not None]
        audios = [t for t in (ref_audio_1, ref_audio_2, ref_audio_3) if t is not None]
        return io.NodeOutput({
            "duration": duration,
            "prompt": prompt or "",
            "first_frame": first_frame,
            "last_frame": last_frame,
            "images": images,
            "videos": videos,
            "audios": audios,
            "mods": mods,
            "loras": loras,
        })