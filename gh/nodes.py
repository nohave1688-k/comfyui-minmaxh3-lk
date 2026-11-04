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
from .core import align_frame_count_down
from ..refmod.nodes import load_visual_mods
from .long_video import (
    MAX_AUDIOS,
    MAX_IMAGES,
    MAX_SEGMENTS,
    MAX_VIDEOS,
    SEGMENT_TYPE,
    build_multi_plan,
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
        ("conn_image", io.Image.Input, "参考图片（连线）", MAX_IMAGES),
        ("conn_video", io.Video.Input, "参考视频（连线）", MAX_VIDEOS),
        ("conn_audio", io.Audio.Input, "参考音频（连线）", MAX_AUDIOS),
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
    ``build_multi_plan`` according to the ``conn_apply`` mode.
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
                    if len(collected["images"]) >= MAX_IMAGES:
                        break
                    collected["images"].append(tensor[frame_index:frame_index + 1])
        elif kind == "video":
            collected["videos"].extend(values[:MAX_VIDEOS])
        elif kind == "audio":
            collected["audios"].extend(values[:MAX_AUDIOS])
    collected["images"] = collected["images"][:MAX_IMAGES]
    collected["videos"] = collected["videos"][:MAX_VIDEOS]
    collected["audios"] = collected["audios"][:MAX_AUDIOS]
    return collected


class MiniMaxH3IntegrationGH(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        schema_inputs = [
            _hidden_combo("main_mode", ["text_keyframes", "all_reference", "multi_segment"], "text_keyframes"),
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
                io.String.Input("prompt", multiline=True, dynamic_prompts=True, default="", extra_dict={"hidden": True}),
                _uploaded_media("first_frame", tooltip="Optional first frame", hidden=True),
                _uploaded_media("last_frame", tooltip="Optional last frame", hidden=True),
                _uploaded_media("hybrid_audio", tooltip="Optional audio for keyframe Hybrid", hidden=True),
                _uploaded_media("ref_image_1", hidden=True),
                _uploaded_media("ref_image_2", hidden=True),
                _uploaded_media("ref_image_3", hidden=True),
                _uploaded_media("ref_image_4", hidden=True),
                _uploaded_media("ref_image_5", hidden=True),
                _uploaded_media("ref_image_6", hidden=True),
                _uploaded_media("ref_image_7", hidden=True),
                _uploaded_media("ref_image_8", hidden=True),
                _uploaded_media("ref_image_9", hidden=True),
                _uploaded_media("ref_video_1", hidden=True),
                _uploaded_media("ref_video_2", hidden=True),
                _uploaded_media("ref_video_3", hidden=True),
                _uploaded_media("ref_audio_1", hidden=True),
                _uploaded_media("ref_audio_2", hidden=True),
                _uploaded_media("ref_audio_3", hidden=True),
                _hidden_combo("task_type", ["auto", "t2va", "i2va", "l2va", "fl2va", "ref2va", "hybrid"], "auto"),
                _hidden_float("audio_denoise_strength", 0.0, 0.0, 1.0, 0.01, 0.01),
                # Match T8: 1 selects the first audio; 0 disables remapping.
                _hidden_int("drive_audio_ordinal", 1, 0, 6),
                _hidden_boolean("strict_prompt_tags", True),
                # Keep this last so adding persistence does not shift legacy
                # widgets_values positions for any existing input.
                io.String.Input("gh_state_json", default="", optional=True, extra_dict={"hidden": True}),
                # Ordered RefMod rows written by the panel.  mod_1 -> <Video 1>,
                # mod_2 -> <Video 2>, ...; the panel is the source of truth and
                # this hidden widget is what gets serialized into the prompt.
                io.String.Input(
                    "refmods_json",
                    default="",
                    optional=True,
                    extra_dict={"hidden": True},
                    tooltip="Ordered RefMod list bound to <Video n> tags (managed by the panel)",
                ),
                io.String.Input(
                    "segments_json",
                    default="",
                    optional=True,
                    extra_dict={"hidden": True},
                    tooltip="Long-video segment table (managed by the panel; 1-8 segments)",
                ),
                # Keep new inputs after every legacy widget so old workflow
                # widget-value positions remain unchanged.
                io.String.Input(
                    "prompt_override",
                    display_name="Prompt",
                    optional=True,
                    force_input=True,
                    tooltip="Optional upstream prompt; overrides the local prompt editor when connected",
                ),
        ]
        # Per-segment connectable media sockets, appended after every legacy
        # widget so existing workflow widget positions remain unchanged.
        schema_inputs.extend(_connected_media_inputs())
        # The adapter stays the only expansion node for the private
        # "MiniMax" socket; the merged RefMod list is exposed separately so
        # it can also drive Apply/TextEncode/Save nodes.
        return io.Schema(
            node_id="LKH3Studio",
            display_name="MiniMaxH3长视频操作台（LK)",
            category=NODE_CATEGORY,
            description="Integrated MiniMax H3 conditioning input. The diffusion model remains external.",
            inputs=schema_inputs,
            outputs=[
                io.Custom(SEGMENT_TYPE).Output(
                    display_name="segments",
                    tooltip="Long-video segment plan; connect to the long-video sampler",
                ),
            ],
        )

    @classmethod
    def execute(cls, main_mode, aspect, megapixels, clip_name, video_vae_name, audio_vae_name,
                prompt, first_frame, last_frame, hybrid_audio, ref_image_1, ref_image_2, ref_image_3,
                ref_image_4, ref_image_5, ref_image_6, ref_image_7, ref_image_8, ref_image_9,
                ref_video_1, ref_video_2, ref_video_3, ref_audio_1, ref_audio_2, ref_audio_3,
                task_type, audio_mode, audio_denoise_strength,
                drive_audio_ordinal, strict_prompt_tags, ref_image_size,
                gh_state_json="", refmods_json="", segments_json="", prompt_override=None,
                conn_apply="all segments", **conn_media):
        preview, plan_mods, plan = build_multi_plan(
            aspect=aspect, megapixels=megapixels, segments_json=segments_json,
            clip_name=clip_name, video_vae_name=video_vae_name,
            audio_vae_name=audio_vae_name, audio_mode=audio_mode,
            audio_denoise_strength=audio_denoise_strength,
            strict_prompt_tags=strict_prompt_tags, ref_image_size=ref_image_size,
            connected_media=_collect_connected_media(conn_media),
            connected_apply=conn_apply,
        )
        return io.NodeOutput(plan)