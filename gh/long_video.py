"""Long-video multi-segment continuation for the unified MiniMax H3 node.

The integrated node owns the segment table (UI) and emits a lightweight
``LK_H3_SEGMENTS`` plan.  `run_long_video` walks the selected
range, rebuilds that segment's conditioning, carries the previous segment's
latent tail into the next segment's head (``_carry_latent``), applies the
segment's LoRAs, samples with the dual clock, decodes, trims the overlap and
concatenates the result.  It also emits the whole-video packed AV latent
(``av_latent``) with ``lk_layout`` metadata.

Feeding the packed AV latent (via ``LKH3AVUpscale``) into downstream chains
switches the same loop into the HD second pass: per-segment slices are
re-sampled from a truncated sigma schedule instead of being generated fresh.

Only ``MODEL`` is patched per segment; CLIP stays untouched (the UI only exposes
a model strength for each LoRA slot), so prompts keep encoding with the base
text encoder.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil

import folder_paths
import nodes as comfy_nodes
import torch
from comfy_api.latest import io

import comfy.nested_tensor
import comfy.utils

from .conditioning import build_conditioning
from .core import empty_av_latent, nested_av_parts
from .sampling import (
    DEFAULT_SAMPLER_NAME,
    DEFAULT_SCHEDULER_NAME,
    SAMPLER_OPTIONS,
    SCHEDULER_OPTIONS,
    _pixel_frames_from_latent_t,
    _scheduler_sigmas,
    setup_dual_clock_sampling_gh,
    shift_sigma,
    time_shift_sigma,
)
from ..refmod.nodes import load_visual_mods

MAX_SEGMENTS = 8
MAX_IMAGES = 6
MAX_VIDEOS = 2
MAX_AUDIOS = 3
MAX_MODS = 8
MAX_LORAS = 4
MIN_DURATION = 2
MAX_DURATION = 30
DEFAULT_DURATION = 5
MIN_OVERLAP = 0
MAX_OVERLAP = 20
DEFAULT_OVERLAP = 20
# Latent continuation (the 'latent_continue' toggle): carry the previous
# segment's last 7 latent tokens (22 pixel frames on H3's 17n+5 grid) as
# the next segment's first tokens, held at sigma 0 via noise_mask=0.
CARRY_TOKENS = 7
CARRY_FRAMES = 22
CARRY_AUDIO_STEPS = 36
# Shared with the lanxi pack on purpose: the storyboard node's plan must stay
# connectable to BOTH packs' long-video samplers (and the lanxi studio's plan
# to ours), so the socket type keeps the original LANXI id.
SEGMENT_TYPE = "LANXI_H3_SEGMENTS"


def _clamp_int(value, default, low, high):
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = int(default)
    return max(int(low), min(int(high), number))


def _clamp_float(value, default, low, high):
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(default)
    if not math.isfinite(number):
        number = float(default)
    return max(float(low), min(float(high), number))


def _media():
    """Resolve shared media helpers lazily (``nodes.py`` imports this module)."""
    from . import nodes as gh_nodes
    return gh_nodes


def _clean_name(value) -> str:
    return str(value or "").strip().replace("\\", "/")


def _normalize_segment(raw, index):
    raw = raw if isinstance(raw, dict) else {}
    images = []
    for item in raw.get("images") or []:
        name = _clean_name(item)
        if name and name not in images and len(images) < MAX_IMAGES:
            images.append(name)
    videos = []
    for item in raw.get("videos") or []:
        if isinstance(item, str):
            item = {"name": item}
        if not isinstance(item, dict):
            continue
        name = _clean_name(item.get("name"))
        if name and len(videos) < MAX_VIDEOS:
            videos.append({"name": name, "muted": bool(item.get("muted"))})
    audios = []
    for item in raw.get("audios") or []:
        if isinstance(item, str):
            item = {"name": item}
        if not isinstance(item, dict):
            continue
        name = _clean_name(item.get("name"))
        if not name or len(audios) >= MAX_AUDIOS:
            continue
        try:
            start = max(0.0, float(item.get("trimStart", 0) or 0))
        except (TypeError, ValueError):
            start = 0.0
        try:
            raw_end = item.get("trimEnd")
            end = float(raw_end) if raw_end is not None else None
        except (TypeError, ValueError):
            end = None
        audios.append({
            "name": name,
            "trimStart": start,
            "trimEnd": end if end is not None and end > start else None,
        })
    mods = []
    for item in raw.get("mods") or []:
        if isinstance(item, str):
            item = {"name": item}
        if not isinstance(item, dict):
            continue
        name = _clean_name(item.get("name"))
        if not name or name in ("(none)", "None"):
            continue
        strength = _clamp_float(item.get("strength", 1.0), 1.0, 0.05, 1.0)
        count = _clamp_int(item.get("count", item.get("visuals", 1)), 1, 1, 256)
        if len(mods) < MAX_MODS:
            mods.append({"name": name, "strength": strength, "count": count})
    loras = []
    for item in raw.get("loras") or []:
        if isinstance(item, str):
            item = {"name": item}
        if not isinstance(item, dict):
            continue
        name = _clean_name(item.get("name"))
        if not name or name in ("(none)", "None"):
            continue
        strength = _clamp_float(item.get("strength_model", item.get("strength", 1.0)), 1.0, -10.0, 10.0)
        if len(loras) < MAX_LORAS:
            loras.append({"name": name, "strength_model": strength})
    return {
        "index": index,
        "duration": _clamp_int(raw.get("duration", DEFAULT_DURATION), DEFAULT_DURATION, MIN_DURATION, MAX_DURATION),
        "prompt": str(raw.get("prompt") or ""),
        "first_frame": _clean_name(raw.get("first_frame")),
        "last_frame": _clean_name(raw.get("last_frame")),
        "images": images,
        "videos": videos,
        "audios": audios,
        "mods": mods,
        "loras": loras,
    }


def parse_segments(segments_json) -> dict:
    """Validate the panel's segment config into a plain plan dictionary."""
    data = segments_json
    if isinstance(data, str):
        try:
            data = json.loads(data) if data.strip() else {}
        except (TypeError, ValueError):
            data = {}
    if not isinstance(data, dict):
        data = {}
    raw_segments = data.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raw_segments = [{}]
    segments = [
        _normalize_segment(raw, index)
        for index, raw in enumerate(raw_segments[:MAX_SEGMENTS])
    ]
    count = len(segments)
    raw_range = data.get("range") if isinstance(data.get("range"), dict) else {}
    start = _clamp_int(raw_range.get("start", 1), 1, 1, count)
    end = _clamp_int(raw_range.get("end", count), count, 1, count)
    if start > end:
        start, end = end, start
    overlap = _clamp_int(data.get("overlap_frames", DEFAULT_OVERLAP), DEFAULT_OVERLAP, MIN_OVERLAP, MAX_OVERLAP)
    return {
        "version": 1,
        "segments": segments,
        "count": count,
        "range": {"start": start, "end": end},
        "overlap_frames": overlap,
        "latent_continue": bool(data.get("latent_continue", True)),
    }




def _segment_source_size(segment):
    """Best-effort aspect source for the adaptive canvas."""
    loader = _media()._load_image_file
    if segment.get("first_frame"):
        image = loader(segment["first_frame"])
        if image is not None:
            return int(image.shape[2]), int(image.shape[1])
    for name in segment.get("images") or []:
        image = loader(name)
        if image is not None:
            return int(image.shape[2]), int(image.shape[1])
    return None, None


def _video_soundtrack(frames, soundtrack, muted, duration_seconds):
    if soundtrack is None or muted:
        return None
    from .core import align_frame_count_down
    reference_duration = align_frame_count_down(int(frames.shape[0])) / 24.0
    return _media()._trim_audio_to_duration(soundtrack, min(duration_seconds, reference_duration))


def build_segment_bundle(plan, index, first_frame_override=None, clip=None, connected=None):
    """Build one segment's conditioning bundle (also used for the ``>`` preview)."""
    segments = plan["segments"]
    segment = segments[index]
    width, height = int(plan["width"]), int(plan["height"])
    length = int(segment.get("length") or _media().calculate_length(segment["duration"]))
    effective_duration = length / 24.0
    loader = _media()
    if clip is None:
        clip = plan.get("clip") or _load_clip(plan)
    # Connected media sockets override the panel's same-kind upload for this
    # segment (replace, not append).
    connected = connected or {}

    if first_frame_override is not None:
        first = first_frame_override[:1]
    elif connected.get("first_frame") is not None:
        first = connected["first_frame"][:1]
    else:
        first = loader._load_image_file(segment.get("first_frame") or "")
    last = connected.get("last_frame")
    if last is None:
        last = loader._load_image_file(segment.get("last_frame") or "")

    refs = {}
    connected_images = connected.get("images")
    if connected_images:
        for image_index, image in enumerate(connected_images, 1):
            refs[f"ref_image_{image_index}"] = image
    else:
        for image_index, name in enumerate(segment.get("images") or [], 1):
            image = loader._load_image_file(name)
            if image is not None:
                refs[f"ref_image_{image_index}"] = image

    ref_videos = {}
    video_audio = {}
    connected_videos = connected.get("videos")
    if connected_videos:
        for video_index, video in enumerate(connected_videos, 1):
            frames, soundtrack = loader._load_video_object(video, length)
            if frames is None:
                continue
            ref_videos[f"ref_video_{video_index}"] = frames
            audio = _video_soundtrack(frames, soundtrack, False, effective_duration)
            if audio is not None:
                video_audio[f"ref_video_audio_{video_index}"] = audio
    else:
        for video_index, item in enumerate(segment.get("videos") or [], 1):
            frames, soundtrack = loader._load_video_frames(item["name"], length)
            if frames is None:
                continue
            ref_videos[f"ref_video_{video_index}"] = frames
            audio = _video_soundtrack(frames, soundtrack, bool(item.get("muted")), effective_duration)
            if audio is not None:
                video_audio[f"ref_video_audio_{video_index}"] = audio

    audios = {}
    connected_audios = connected.get("audios")
    if connected_audios:
        for audio_index, audio in enumerate(connected_audios, 1):
            audio = loader._trim_audio_to_duration(audio, effective_duration)
            if audio is not None:
                audios[f"ref_audio_{audio_index}"] = audio
    else:
        for audio_index, item in enumerate(segment.get("audios") or [], 1):
            audio = loader._load_audio_file(item["name"])
            selection = None
            if item.get("trimEnd") is not None:
                selection = (float(item.get("trimStart") or 0.0), float(item["trimEnd"]))
            audio = loader._apply_audio_selection(audio, selection)
            audio = loader._trim_audio_to_duration(audio, effective_duration)
            if audio is not None:
                audios[f"ref_audio_{audio_index}"] = audio

    mods = load_visual_mods(segment.get("mods") or [])
    ordered_drive = list(video_audio.values()) + list(audios.values())
    drive_audio = ordered_drive[0] if ordered_drive else None
    # The panel's global Audio reference mode controls every segment: the
    # speaker only decides whether that reference video contributes Audio 1.
    audio_mode = str(plan.get("audio_mode") or "lock_source")
    if audio_mode not in {"lock_source", "reference_only", "native"}:
        audio_mode = "lock_source"
    if drive_audio is None:
        audio_mode = "native"
    add_source_as_reference = audio_mode != "native"
    final_audio = loader._trim_audio_to_duration(drive_audio, effective_duration)

    result = build_conditioning(
        clip, plan["video_vae"], plan["audio_vae"],
        segment.get("prompt") or "", width, height, length,
        "auto", audio_mode, 0.0, add_source_as_reference,
        False, plan.get("ref_image_size") or "match",
        drive_audio, final_audio,
        first, last, refs, ref_videos, video_audio, audios,
        ref_mods=mods,
    )
    bundle = {
        "positive": result[0], "av_latent": result[1], "mux_audio": result[2],
        "conditioned_prompt": result[3], "media_map_json": result[4], "report": result[5],
        "video_vae": plan["video_vae"], "audio_vae": plan["audio_vae"],
        "width": width, "height": height, "length": length,
        "duration_seconds": segment["duration"], "resolved_mode": "all_reference",
        "is_original_audio": audio_mode == "lock_source",
        "refmods": mods, "segment_index": index + 1, "prompt": segment.get("prompt") or "",
    }
    report = bundle.get("report") or ""
    bundle["report"] = f"segment={index + 1}\n{report}".strip()
    # Console visibility: what actually got packed for this shot. A prompt tag
    # pointing at an unconnected slot (e.g. <Video 1> with no video reference)
    # shows up here as a warning line — the model cannot reference media that
    # was never packed.
    logging.info(
        "[LKH3] 镜头%d 条件打包: pictures=%d videos=%d audios=%d | %s",
        index + 1,
        len(refs),
        len(ref_videos),
        len(audios),
        " | ".join(report.splitlines()[:10]),
    )
    return bundle


def _connected_source_size(connected):
    """Aspect source from connected socket tensors (overrides panel files)."""
    first = connected.get("first_frame")
    if first is not None:
        return int(first.shape[2]), int(first.shape[1])
    for image in connected.get("images") or []:
        return int(image.shape[2]), int(image.shape[1])
    return None, None




def _clamp_shot_mods(raw_mods):
    mods = []
    for item in raw_mods or []:
        if isinstance(item, str):
            item = {"name": item}
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name or name in {"(none)", "None"}:
            continue
        mods.append({
            "name": name,
            "strength": max(0.05, min(1.0, float(item.get("strength") or 1) or 1)),
            "count": max(1, min(256, int(item.get("count") or 1))),
        })
    return mods[:MAX_MODS]


def _clamp_shot_loras(raw_loras):
    loras = []
    for item in raw_loras or []:
        if isinstance(item, str):
            item = {"name": item}
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name or name in {"(none)", "None"}:
            continue
        loras.append({
            "name": name,
            "strength_model": max(-10.0, min(10.0, float(item.get("strength_model") or 1) or 1)),
        })
    return loras[:MAX_LORAS]


def _save_image_temp(tensor, name) -> str:
    """Save an IMAGE tensor [B,H,W,C] into ComfyUI's temp directory and return
    the annotated name ([temp] suffix) that both packs' loaders resolve."""
    from PIL import Image as PILImage
    image = (
        tensor[0].detach().to("cpu").clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
    )
    path = os.path.join(folder_paths.get_temp_directory(), name)
    PILImage.fromarray(image).save(path, format="PNG")
    return f"{name} [temp]"


def _save_audio_temp(audio, name) -> str:
    """Save an AUDIO dict to the temp directory.  torchaudio.save may route to
    the (often missing) TorchCodec backend on newer builds, so fall back to
    soundfile and then to a plain stdlib WAV writer."""
    path = os.path.join(folder_paths.get_temp_directory(), name)
    waveform = audio["waveform"][0].detach().to("cpu").float()
    sample_rate = int(audio.get("sample_rate") or 24000)
    try:
        import torchaudio
        torchaudio.save(path, waveform, sample_rate)
        return f"{name} [temp]"
    except Exception:
        pass
    try:
        import soundfile as sf
        sf.write(path, waveform.numpy().T, sample_rate)
        return f"{name} [temp]"
    except Exception:
        pass
    import wave
    import numpy as np
    data = waveform.numpy()
    if data.ndim == 1:
        data = data[None, :]
    interleaved = np.transpose(data).reshape(-1)
    pcm = (np.clip(interleaved, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as handle:
        handle.setnchannels(int(data.shape[0]))
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return f"{name} [temp]"


def _video_source_path(video):
    """Best-effort source path of a V3 VIDEO input (files only; in-memory
    components cannot be referenced by name)."""
    source = getattr(video, "get_stream_source", None)
    if callable(source):
        try:
            source = source()
        except Exception:
            return None
    return source if isinstance(source, str) and os.path.exists(source) else None


def _persist_shot_media(shot, idx):
    """Materialize a shot's linked tensor media as temp-directory files and
    return the filename-keyed segment entries the lanxi sampler consumes —
    its loaders resolve media by name through
    ``folder_paths.get_annotated_filepath`` and cannot read tensors.  Our own
    sampler ignores these entries while the tensor channel in
    ``connected_by_segment`` carries content, so both samplers stay fed by the
    same sockets."""
    media = {}
    first_frame = shot.get("first_frame")
    if first_frame is not None:
        try:
            media["first_frame"] = _save_image_temp(first_frame, f"lk_shot{idx}_first.png")
        except Exception as error:
            logging.warning("[LKH3] 镜头%d first_frame 落盘失败: %s", idx, error)
    last_frame = shot.get("last_frame")
    if last_frame is not None:
        try:
            media["last_frame"] = _save_image_temp(last_frame, f"lk_shot{idx}_last.png")
        except Exception as error:
            logging.warning("[LKH3] 镜头%d last_frame 落盘失败: %s", idx, error)
    images = []
    for slot, image in enumerate(shot.get("images") or [], 1):
        try:
            images.append(_save_image_temp(image, f"lk_shot{idx}_img{slot}.png"))
        except Exception as error:
            logging.warning("[LKH3] 镜头%d 参考图%d 落盘失败: %s", idx, slot, error)
    if images:
        media["images"] = images
    videos = []
    for slot, video in enumerate(shot.get("videos") or [], 1):
        source = _video_source_path(video)
        if not source:
            logging.warning(
                "[LKH3] 镜头%d 参考视频%d 无法落盘（内存型视频没有源文件），lanxi 采样器将看不到它",
                idx, slot,
            )
            continue
        name = f"lk_shot{idx}_vid{slot}{os.path.splitext(source)[1] or '.mp4'}"
        try:
            shutil.copyfile(source, os.path.join(folder_paths.get_temp_directory(), name))
            videos.append({"name": f"{name} [temp]", "muted": False})
        except Exception as error:
            logging.warning("[LKH3] 镜头%d 参考视频%d 拷贝失败: %s", idx, slot, error)
    if videos:
        media["videos"] = videos
    audios = []
    for slot, audio in enumerate(shot.get("audios") or [], 1):
        try:
            audios.append({"name": _save_audio_temp(audio, f"lk_shot{idx}_aud{slot}.wav")})
        except Exception as error:
            logging.warning("[LKH3] 镜头%d 参考音频%d 落盘失败: %s", idx, slot, error)
    if audios:
        media["audios"] = audios
    return media


def build_plan_from_shot_nodes(*, settings=None, shots, film_settings=None,
                               latent_continue=True, range_start=1, range_end=1,
                               clip=None, video_vae=None, audio_vae=None):
    """Assemble the long-video plan from the composer + ordered LKH3Shot nodes.

    ``settings`` (模型设置) is optional legacy fallback: it only contributes the
    CLIP name plus canvas/audio knobs when the 全片提示词设置 node is absent.
    ``film_settings`` (全片提示词设置) supplies the whole-film prompt block —
    prepended to every shot's prompt — and overrides the canvas/audio knobs.
    Linked clip/VAEs (from the 模型设置 node) are baked into the plan as
    objects, which is the only channel the lanxi sampler can consume.
    Each shot dict carries its own media as tensors (first_frame, last_frame,
    images/videos/audios lists) which are routed through the
    ``connected_by_segment`` tensor channel for this pack's samplers; the same
    media is also materialized as temp-directory files referenced by name in
    the segment entries, which is the channel the lanxi pack's sampler reads
    (its loaders go through ``folder_paths.get_annotated_filepath``).  CLIP
    stays None when no link and no name is available (this pack's samplers
    take it as a link or lazy-load it).
    """
    settings = settings if isinstance(settings, dict) else {}
    film = film_settings if isinstance(film_settings, dict) else {}
    if not isinstance(shots, (list, tuple)) or not shots:
        raise ValueError("至少连接一个「MiniMaxH3分镜」节点到分镜合成节点。")

    try:
        start = max(1, min(MAX_SEGMENTS, int(range_start)))
        end = max(1, min(MAX_SEGMENTS, int(range_end)))
    except (TypeError, ValueError):
        start, end = 1, 1
    if start > end:
        start, end = end, start
    # The range auto-narrows to the shots actually connected: a 结束镜头 set
    # past the last wired shot can never execute and should not error out.
    if end > len(shots):
        logging.info(
            "[LKH3] 结束镜头 %d 超出已连接的分镜数，自动收窄为 %d（分镜 1-%d）",
            end, len(shots), len(shots),
        )
        end = len(shots)
    if start > end:
        logging.info(
            "[LKH3] 起始镜头 %d 超出已连接的分镜数，自动收窄为 %d",
            start, end,
        )
        start = end

    aspect = film.get("aspect") or settings.get("aspect", "adaptive")
    megapixels = film.get("megapixels", settings.get("megapixels", 0.5))
    clip_name = settings.get("clip_name")
    video_vae_name = settings.get("video_vae_name")
    audio_vae_name = settings.get("audio_vae_name")
    film_prompt = str(film.get("film_prompt") or "").strip()

    segments, connected_by_segment = [], {}
    for i, shot in enumerate(shots[:MAX_SEGMENTS]):
        idx = i + 1
        shot = shot if isinstance(shot, dict) else {}
        try:
            duration = max(MIN_DURATION, min(MAX_DURATION, int(shot.get("duration") or 5)))
        except (TypeError, ValueError):
            duration = 5
        shot_prompt = str(shot.get("prompt") or "").strip()
        prompt = f"{film_prompt}\n\n{shot_prompt}".strip() if film_prompt else shot_prompt
        segments.append({
            "duration": duration,
            "prompt": prompt,
            "length": _media().calculate_length(duration),
            "mods": _clamp_shot_mods(shot.get("mods")),
            "loras": _clamp_shot_loras(shot.get("loras")),
            "index": idx,
            **_persist_shot_media(shot, idx),
        })
        connected_by_segment[idx] = {
            "first_frame": shot.get("first_frame"),
            "last_frame": shot.get("last_frame"),
            "images": list(shot.get("images") or []),
            "videos": list(shot.get("videos") or []),
            "audios": list(shot.get("audios") or []),
        }

    first_conn = connected_by_segment.get(start) or {}
    source_width, source_height = _connected_source_size(first_conn)
    width, height = _media().calculate_canvas(aspect, megapixels, source_width, source_height)

    # Linked clip/VAEs win; names are the legacy fallback.  The lanxi sampler
    # can only consume objects baked into the plan (it has no link inputs).
    video_vae = video_vae if video_vae is not None else (
        comfy_nodes.VAELoader().load_vae(video_vae_name)[0] if video_vae_name else None
    )
    audio_vae = audio_vae if audio_vae is not None else (
        comfy_nodes.VAELoader().load_vae(audio_vae_name)[0] if audio_vae_name else None
    )
    if clip is None and not clip_name:
        logging.warning(
            "[LKH3] 计划未携带 CLIP（未连模型设置、也没接 CLIP 连线）："
            "喂给 lanxi 长视频分段采样会失败；本包采样器请把 CLIP 直接连到采样节点。"
        )
    if video_vae is None or audio_vae is None:
        logging.warning(
            "[LKH3] 计划未携带 视频/音频 VAE：lanxi 采样器解码需要它们；"
            "本包采样器可直接把 VAE 连到采样节点。"
        )

    return {
        "version": 1,
        "segments": segments,
        "count": len(segments),
        "range": {"start": start, "end": end},
        "latent_continue": bool(latent_continue),
        "width": width,
        "height": height,
        "clip_name": clip_name,
        "video_vae_name": video_vae_name,
        "audio_vae_name": audio_vae_name,
        "aspect": aspect,
        "megapixels": megapixels,
        "audio_mode": film.get("audio_mode") or settings.get("audio_mode", "lock_source"),
        "audio_denoise_strength": 0.0,
        "strict_prompt_tags": False,
        "ref_image_size": film.get("ref_image_size") or settings.get("ref_image_size", "match"),
        "clip": clip,
        "video_vae": video_vae,
        "audio_vae": audio_vae,
        "connected_by_segment": connected_by_segment,
        "preview": None,
    }


def _load_clip(plan):
    if not plan.get("clip_name"):
        raise ValueError(
            "CLIP 未就绪：请把模型设置节点的 CLIP 连线到采样节点，或在模型设置里选择 CLIP 名称。"
        )
    clip = comfy_nodes.CLIPLoader().load_clip(plan["clip_name"], "minimax")[0]
    plan["clip"] = clip
    return clip


def _release_clip(plan, clip):
    from .nodes import _release_text_encoder
    _release_text_encoder(clip)
    plan["clip"] = None


def _lora_cache():
    return {}


def _apply_model_loras(model, loras, cache):
    if not loras:
        return model
    import comfy.sd
    import comfy.utils

    patched = model.clone()
    for entry in loras:
        name = entry.get("name")
        path = folder_paths.get_full_path("loras", name)
        if not path or not os.path.isfile(path):
            try:
                searched = ", ".join(str(item) for item in folder_paths.get_folder_paths("loras"))
            except Exception:
                searched = "ComfyUI/models/loras"
            raise ValueError(
                f"LoRA '{name}' not found. Searched loras folders: {searched}. "
                "Refresh the LoRA list in the long-video panel after adding files."
            )
        key = os.path.normcase(os.path.abspath(path))
        weights = cache.get(key)
        if weights is None:
            weights = comfy.utils.load_torch_file(path)
            cache[key] = weights
        patched, _clip = comfy.sd.load_lora_for_models(
            patched, None, weights, float(entry.get("strength_model", 1.0)), 0.0
        )
    return patched


def _make_guider(model, positive, cfg):
    cfg = float(cfg)
    if abs(cfg - 1.0) < 1e-6:
        # H3 is distilled and normally runs without CFG; Guider_Basic also
        # avoids evaluating the negative branch for every step.
        #
        # Current ComfyUI exposes the guiders in comfy_extras as V3 nodes
        # (io.ComfyNode subclasses): instantiating them with a model raises
        # "_ComfyNodeBaseInternal.__init__() takes 1 positional argument".
        # Use the runtime class Guider_Basic from the same module instead;
        # the CFG path below uses comfy.samplers.CFGGuider.
        try:
            from comfy_extras.nodes_custom_sampler import Guider_Basic
            guider = Guider_Basic(model)
            guider.set_conds(positive)
            return guider
        except Exception:
            pass
    try:
        import comfy.samplers
        guider = comfy.samplers.CFGGuider(model)
    except Exception as exc:  # pragma: no cover - depends on ComfyUI version
        raise RuntimeError("This ComfyUI build has no CFGGuider; update ComfyUI to use long video.") from exc
    guider.set_conds(positive, positive)
    guider.set_cfg(cfg)
    return guider


def _randn_like(part, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)
    dtype = part.dtype if getattr(part.dtype, "is_floating_point", False) else torch.float32
    return torch.randn(
        part.size(), dtype=dtype, layout=part.layout, generator=generator, device="cpu"
    )


def _make_noise(samples, seed):
    if getattr(samples, "is_nested", False):
        from comfy.nested_tensor import NestedTensor
        parts = [
            _randn_like(part, int(seed) + index * 104729)
            for index, part in enumerate(samples.unbind())
        ]
        return NestedTensor(parts)
    return _randn_like(samples, int(seed))


def _carry_latent(previous_samples, av_latent, context_noise=0.0):
    """Place the previous segment's latent tail at the start of the next one.

    The carried video tokens use ``context_noise`` as their denoise mask: 0
    keeps them exactly, while 0.1-0.3 lets the sampler re-noise and clean the
    carried face without breaking motion.  Audio stays at 0 so speech stays
    continuous.  ``CARRY_TOKENS=7`` and ``CARRY_FRAMES=22`` are the same point
    on H3's 5n+2 latent / 17n+5 pixel grids, so the duplicated pixel span can
    be trimmed 1:1 after decoding.

    Returns ``(latent, carried_tokens, carried_audio_steps)`` so the runner can
    trim the exact duplicated span from the whole-video latent output.
    """
    if not getattr(previous_samples, "is_nested", False):
        return av_latent, 0, 0
    samples = av_latent.get("samples")
    if not getattr(samples, "is_nested", False):
        return av_latent, 0, 0
    previous_video, previous_audio = previous_samples.unbind()
    video, audio = samples.unbind()
    tokens = min(CARRY_TOKENS, int(previous_video.shape[2]), int(video.shape[2]))
    audio_steps = min(CARRY_AUDIO_STEPS, int(previous_audio.shape[-1]), int(audio.shape[-1]))
    if tokens <= 0:
        return av_latent, 0, 0
    video[..., :tokens, :, :] = previous_video[..., -tokens:, :, :].to(video)
    if audio_steps > 0:
        audio[..., :audio_steps] = previous_audio[..., -audio_steps:].to(audio)

    existing = av_latent.get("noise_mask")
    if getattr(existing, "is_nested", False):
        video_mask, audio_mask = existing.unbind()
        # H3's own masks are channel-shaped, but the generic sampler's
        # denoise_mask is a single-channel mask.  Keeping 24 channels here
        # made prepare_mask/KSamplerX0Inpaint reinterpret the mask and the
        # carried rows were effectively not held, which is why the seams
        # still jumped.  Collapse to [B, 1, T, H, W] / [B, 1, 2, T].
        if video_mask.shape[1] != 1:
            video_mask = video_mask.amax(dim=1, keepdim=True)
        if audio_mask.shape[1] != 1:
            audio_mask = audio_mask.amax(dim=1, keepdim=True)
        video_mask = video_mask.clone()
        audio_mask = audio_mask.clone()
    else:
        video_mask = torch.ones(
            (video.shape[0], 1, video.shape[2], video.shape[3], video.shape[4]),
            device=video.device, dtype=torch.float32,
        )
        audio_mask = torch.ones(
            (audio.shape[0], 1, audio.shape[2], audio.shape[3]),
            device=audio.device, dtype=torch.float32,
        )
    video_mask[..., :tokens, :, :] = max(0.0, min(1.0, float(context_noise)))
    if audio_steps > 0:
        audio_mask[..., :audio_steps] = 0.0
    out = dict(av_latent)
    out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
    return out, tokens, audio_steps


def _collapse_av_mask(av_latent):
    """Normalize the AV noise_mask to the sampler's single-channel layout.

    H3's own masks are channel-shaped ([B,24,T,H,W] video / [B,32,2,T] audio),
    but the generic denoise_mask path is single-channel.  Collapse via amax so
    the whole-video latent output and the refine input share one layout.
    """
    masks = av_latent.get("noise_mask")
    if not getattr(masks, "is_nested", False):
        return None, None
    video_mask, audio_mask = masks.unbind()
    if video_mask.shape[1] != 1:
        video_mask = video_mask.amax(dim=1, keepdim=True)
    if audio_mask.shape[1] != 1:
        audio_mask = audio_mask.amax(dim=1, keepdim=True)
    return video_mask, audio_mask


def _assemble_whole_av_latent(collected, audio_mode, latent_continue, start, end):
    """Concatenate the executed segments into one whole-video AV latent.

    Carried heads are already trimmed by the caller, so the packed latent
    decodes 1:1 into the concatenated output frames.  ``lk_layout`` records the
    per-segment token counts (the refine pass's slicing map).
    """
    video = torch.cat([item["video"] for item in collected], dim=2)
    audio = torch.cat([item["audio"] for item in collected], dim=-1)
    entries = []
    video_offset = 0
    audio_offset = 0
    for item in collected:
        tokens = int(item["video"].shape[2])
        audio_steps = int(item["audio"].shape[-1])
        entries.append({
            "segment": item["segment"],
            "video_offset": video_offset,
            "video_tokens": tokens,
            "audio_offset": audio_offset,
            "audio_steps": audio_steps,
            "frames": _pixel_frames_from_latent_t(tokens),
        })
        video_offset += tokens
        audio_offset += audio_steps
    out = {
        "samples": comfy.nested_tensor.NestedTensor((video, audio)),
        "gh_h3_audio_mode": audio_mode or "native",
        "lk_layout": {
            "version": 1,
            "pass": "first",
            "latent_continue": bool(latent_continue),
            "range": {"start": start + 1, "end": end + 1},
            "video_hw": [int(video.shape[3]), int(video.shape[4])],
            "segments": entries,
        },
    }
    if any(item["video_mask"] is not None or item["audio_mask"] is not None for item in collected):
        video_masks = []
        audio_masks = []
        for item in collected:
            v, a = item["video"], item["audio"]
            video_masks.append(
                item["video_mask"] if item["video_mask"] is not None
                else torch.ones((v.shape[0], 1) + tuple(v.shape[2:]), device=v.device, dtype=v.dtype)
            )
            audio_masks.append(
                item["audio_mask"] if item["audio_mask"] is not None
                else torch.ones((a.shape[0], 1, a.shape[2], a.shape[3]), device=a.device, dtype=a.dtype)
            )
        out["noise_mask"] = comfy.nested_tensor.NestedTensor(
            (torch.cat(video_masks, dim=2), torch.cat(audio_masks, dim=-1))
        )
    return out


def _external_sampler_for(av_latent, external_sampler):
    """Validate an externally handed-in dual-clock KSAMPLER against THIS latent.

    The dual-clock euler splits the packed AV tensor at ``packed - audio_values``
    where ``audio_values`` was captured from the latent the sampler was BUILT
    on.  Contract samplers are built on the whole-film template, so their split
    is wrong for every individual segment: each segment's tail (plus its real
    audio) silently took the AUDIO clock instead of the video clock, and since
    that tail is exactly what the next shot carries, the film degraded hop over
    hop.  The original pack never hit this because it rebuilds the sampler per
    segment.  Reject mismatched samplers and fall back to the per-segment
    builtin.
    """
    if external_sampler is None:
        return None
    want = getattr(external_sampler, "_goohai_audio_values", None)
    if want is None:
        return external_sampler
    _video, audio = nested_av_parts(av_latent)
    have = int(math.prod(audio.shape[1:]))
    if int(want) != have:
        logging.info(
            "[LKH3] 外接双钟采样器的音频切分(%d)与本镜(%d)不符 —— 本镜改用按本镜音频长度构建的内置双钟采样器",
            int(want), have,
        )
        return None
    return external_sampler


def _sample_bundle(model, bundle, *, steps, shift_video, shift_audio,
                   sampler_name, scheduler, seed, cfg, progress=None,
                   external_noise=None, external_sigmas=None, external_sampler=None):
    av_latent = bundle["av_latent"]
    # An external plain standard sampler requires the native FLOW_AV model
    # -sampling patch (any sampler name other than dual_clock_euler selects it)
    # so the sampler advances the audio clock correctly. Our own dual-clock
    # KSAMPLER (marker attr) pairs with the dual-clock patch instead — the
    # segmented guider path hands it in as an external sampler.
    external_sampler = _external_sampler_for(av_latent, external_sampler)
    effective_sampler_name = sampler_name
    if external_sampler is not None and not getattr(external_sampler, "_goohai_dual_clock", False):
        effective_sampler_name = "euler"
    patched_model, _sampler_builtin, default_sigmas = setup_dual_clock_sampling_gh(
        model, av_latent, steps, shift_video, shift_audio, effective_sampler_name, scheduler
    )
    sampler = external_sampler if external_sampler is not None else _sampler_builtin
    if external_sigmas is not None:
        # User-owned schedule: used as-is for every pass.
        sigmas = external_sigmas
    else:
        sigmas = default_sigmas
    guider = _make_guider(patched_model, bundle["positive"], cfg)
    samples = av_latent["samples"]
    if external_noise is not None:
        try:
            noise = external_noise.generate_noise(samples, int(seed) & 0x7FFFFFFFFFFFFFFF)
        except TypeError:
            # comfy-core noise objects (Noise_RandomNoise) bake their own seed
            # and only accept the latent — fall back to this pack's per-seed
            # generator so every shot still gets its own noise.
            noise = _make_noise(samples, seed)
    else:
        noise = _make_noise(samples, seed)
    disable_pbar = not getattr(comfy.utils, "PROGRESS_BAR_ENABLED", True)
    previewer = None
    try:
        import latent_preview
        previewer = latent_preview.get_previewer(
            patched_model.load_device, patched_model.model.latent_format
        )
    except Exception:
        previewer = None
    progress_callback = None
    if progress is not None:
        def progress_callback(step, x0, x, total):
            preview_bytes = None
            if previewer is not None:
                try:
                    latent = x0.tensors[0] if getattr(x0, "is_nested", False) else x0
                    preview_bytes = previewer.decode_latent_to_preview_image("JPEG", latent)
                except Exception:
                    preview_bytes = None
            progress.update_absolute(progress.current + 1, progress.total, preview_bytes)
    sampled = guider.sample(
        noise, samples, sampler, sigmas,
        denoise_mask=av_latent.get("noise_mask"),
        seed=int(seed), disable_pbar=disable_pbar,
        callback=progress_callback,
    )
    out = dict(av_latent)
    out["samples"] = sampled
    return out


def run_long_video(model, segments, *, steps, shift_video, shift_audio,
                   sampler_name, scheduler, seed, seed_policy, cfg, release_clip,
                   free_between_segments, context_noise=0.0,
                   video_vae=None, audio_vae=None, clip=None, prebuilt=None,
                   external_noise=None, external_sigmas=None, external_sampler=None):
    plan = segments
    if not isinstance(plan, dict) or not plan.get("segments"):
        raise ValueError("Connect the storyboard node's 'segments' output to the long video sampler.")
    # Linked CLIP/VAEs override the name-based loading the plan builder
    # performed, so conditioning sees exactly these objects.
    if clip is not None:
        plan["clip"] = clip
    if video_vae is not None:
        plan["video_vae"] = video_vae
    if audio_vae is not None:
        plan["audio_vae"] = audio_vae
    start = int(plan["range"]["start"]) - 1
    end = int(plan["range"]["end"]) - 1
    latent_continue = bool(plan.get("latent_continue", True))
    lora_cache = _lora_cache()
    clip = plan.get("clip")
    prebuilt_bundle = None
    prebuilt_clip = None
    if prebuilt is not None:
        prebuilt_bundle, prebuilt_clip = prebuilt
        if clip is None:
            clip = prebuilt_clip
    preview = plan.get("preview") or {}
    progress = None
    if getattr(comfy.utils, "PROGRESS_BAR_ENABLED", True):
        progress = comfy.utils.ProgressBar(max(1, int(steps)) * max(1, end - start + 1))
    previous_samples = None
    collected = []
    audio_mode = None
    first_positive = None

    for index in range(start, end + 1):
        segment = plan["segments"][index]
        if preview.get("index") == index and preview.get("bundle") is not None:
            bundle = preview["bundle"]
        elif index == start and prebuilt_bundle is not None:
            bundle = prebuilt_bundle
        else:
            if clip is None:
                clip = _load_clip(plan)
            # No automatic previous-tail keyframe: continuation is carried in
            # latent space below, not as a first-frame conditioning image.
            bundle = build_segment_bundle(
                plan, index, first_frame_override=None, clip=clip,
                connected=(plan.get("connected_by_segment") or {}).get(index + 1),
            )
        if first_positive is None:
            # The conditioning output for downstream sampler chains: the first
            # executed shot's positive bundle (prompt + reference encoding).
            first_positive = bundle["positive"]

        if release_clip and clip is not None:
            _release_clip(plan, clip)
            clip = None

        seg_seed = int(seed) if seed_policy == "fixed" else int(seed) + (index - start)
        seg_model = _apply_model_loras(model, segment.get("loras") or [], lora_cache)

        carry_t = carry_a = 0
        carried = bool(latent_continue and previous_samples is not None)
        if carried:
            bundle = dict(bundle)
            bundle["av_latent"], carry_t, carry_a = _carry_latent(
                previous_samples, bundle["av_latent"], context_noise=float(context_noise)
            )
            carried = carry_t > 0
        samples = _sample_bundle(
            seg_model, bundle,
            steps=int(steps), shift_video=float(shift_video), shift_audio=float(shift_audio),
            sampler_name=sampler_name, scheduler=scheduler, seed=seg_seed, cfg=float(cfg),
            progress=progress,
            external_noise=external_noise, external_sigmas=external_sigmas,
            external_sampler=external_sampler,
        )
        if latent_continue:
            # ``samples`` is the segment's output dict; carry its nested latent
            # tensor into the next segment.
            previous_samples = samples["samples"]
        seg_video, seg_audio = nested_av_parts(samples)
        seg_vmask, seg_amask = _collapse_av_mask(samples)
        absolute_start = None
        if carried:
            seg_video = seg_video[..., carry_t:, :, :]
            seg_audio = seg_audio[..., carry_a:]
            if seg_vmask is not None:
                seg_vmask = seg_vmask[..., carry_t:, :, :]
            if seg_amask is not None:
                seg_amask = seg_amask[..., carry_a:]

        if audio_mode is None:
            audio_mode = samples.get("gh_h3_audio_mode") or plan.get("audio_mode") or "native"
        # All-ones masks carry no information (an absent mask already means
        # "denoise everywhere" downstream), so drop the trivial ones instead of
        # storing a redundant full-size mask in the whole-video latent.
        if (
            seg_vmask is not None
            and not bool(torch.any(seg_vmask < 1.0).item())
            and (seg_amask is None or not bool(torch.any(seg_amask < 1.0).item()))
        ):
            seg_vmask = seg_amask = None
        # Keep CPU copies (slicing shares storage; copies release the GPU latent
        # at ``del samples``) for the whole-video latent output.  First-pass
        # copies are already head-trimmed for carried segments, so the packed
        # latent decodes 1:1 into the concatenated frames.
        collected.append({
            "segment": index + 1,
            "video": seg_video.detach().to("cpu"),
            "audio": seg_audio.detach().to("cpu"),
            "video_mask": None if seg_vmask is None else seg_vmask.detach().to("cpu"),
            "audio_mask": None if seg_amask is None else seg_amask.detach().to("cpu"),
        })
        # No per-segment decode here: the packed AV latent is the node's output
        # and decoding happens in the downstream AV 解码 node.
        del samples, seg_model

        if free_between_segments:
            try:
                import comfy.model_management as model_management
                model_management.soft_empty_cache()
            except Exception:
                pass

    if not collected:
        raise RuntimeError("Long video sampler produced no segments.")
    av_latent_out = _assemble_whole_av_latent(collected, audio_mode, latent_continue, start, end)
    return av_latent_out, first_positive




def build_empty_plan_latent(plan, start, end):
    """Whole-range empty AV template with the layout the segmented sampler
    assembles (per-shot own-token spans, carry heads excluded).

    Feeds contract nodes (双钟采样 / T8 sampler contracts) and
    SamplerCustomAdvanced's ``latent_image`` without depending on the
    sampler's own output, so external sampling recipes stay acyclic.
    """
    width, height = int(plan["width"]), int(plan["height"])
    videos = []
    audios = []
    entries = []
    video_offset = 0
    audio_offset = 0
    for index in range(start, end + 1):
        segment = plan["segments"][index]
        length = int(segment.get("length") or _media().calculate_length(segment["duration"]))
        latent, _frames = empty_av_latent(width, height, length)
        video, audio = nested_av_parts(latent)
        tokens = int(video.shape[2])
        audio_steps = int(audio.shape[-1])
        videos.append(video.detach().to("cpu"))
        audios.append(audio.detach().to("cpu"))
        entries.append({
            "segment": index + 1,
            "video_offset": video_offset,
            "video_tokens": tokens,
            "audio_offset": audio_offset,
            "audio_steps": audio_steps,
            "frames": _pixel_frames_from_latent_t(tokens),
        })
        video_offset += tokens
        audio_offset += audio_steps
    video = torch.cat(videos, dim=2)
    audio = torch.cat(audios, dim=-1)
    return {
        "samples": comfy.nested_tensor.NestedTensor((video, audio)),
        "gh_h3_audio_mode": str(plan.get("audio_mode") or "native"),
        "lk_layout": {
            "version": 1,
            "pass": "template",
            "latent_continue": bool(plan.get("latent_continue", True)),
            "range": {"start": start + 1, "end": end + 1},
            "video_hw": [int(video.shape[3]), int(video.shape[4])],
            "segments": entries,
        },
    }




class _SegmentedAVGuider:
    """GUIDER object for SamplerCustomAdvanced: owns the segmented long-video
    loop (per-shot conditioning + LoRA, latent continuation, seam noise) while
    the schedule and the seed come from the surrounding standard sampling
    primitives — the T8-style decomposition of the first pass.

    ``noise``/``latent_image`` handed in by SamplerCustomAdvanced are the
    whole-template artifacts of that composition; the loop builds its own
    per-shot templates and noise. The primitives that matter are the sigmas
    and the seed.  The SAMPLER deliberately does NOT flow in: the dual-clock
    euler bakes the audio-element count of the latent it was built on, so a
    whole-film contract sampler mis-splits every per-segment packed tensor —
    each segment rebuilds its own sampler instead (exactly what the original
    pack's per-segment loop does).  The contract sampler this node emits stays
    a wiring requirement of SamplerCustomAdvanced, nothing more.
    """

    def __init__(self, model_patcher, plan, *, steps, shift_video, shift_audio,
                 sampler_name, scheduler, seed_policy, context_noise,
                 release_clip, free_between_segments, video_vae, audio_vae, prebuilt):
        self.model_patcher = model_patcher
        self._plan = plan
        self._steps = int(steps)
        self._shift_video = float(shift_video)
        self._shift_audio = float(shift_audio)
        self._sampler_name = sampler_name or DEFAULT_SAMPLER_NAME
        self._scheduler = scheduler or DEFAULT_SCHEDULER_NAME
        self._seed_policy = seed_policy
        self._cfg = 1.0  # H3 is distillation-trained; cfg > 1 breaks the trajectory
        self._context_noise = float(context_noise)
        self._release_clip = bool(release_clip)
        self._free_between_segments = bool(free_between_segments)
        self._video_vae = video_vae
        self._audio_vae = audio_vae
        self._prebuilt = prebuilt

    def sample(self, noise, latent_image, sampler, sigmas,
               denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        av_latent, _first_positive = run_long_video(
            self.model_patcher, self._plan,
            steps=self._steps, shift_video=self._shift_video, shift_audio=self._shift_audio,
            sampler_name=self._sampler_name, scheduler=self._scheduler,
            seed=int(seed) if seed is not None else 0,
            seed_policy=self._seed_policy, cfg=self._cfg,
            release_clip=self._release_clip, free_between_segments=self._free_between_segments,
            context_noise=self._context_noise,
            video_vae=self._video_vae, audio_vae=self._audio_vae,
            prebuilt=self._prebuilt,
            # The SCA sampler is the whole-film contract object; per-segment
            # dual-clock samplers must be rebuilt against each segment's own
            # audio length (see _external_sampler_for), so only the schedule
            # flows in from the primitives.
            external_noise=None, external_sigmas=sigmas, external_sampler=None,
        )
        return av_latent["samples"]


class MiniMaxH3SegmentGuiderGH(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LKH3SegmentGuider",
            display_name="MiniMax-H3 分段引导（LK）",
            category="LK/MiniMax H3 Integration",
            description=(
                "T8-style GUIDER for the first pass: owns the segmented loop "
                "(per-shot conditioning + LoRA, latent continuation) and emits its own "
                "SAMPLER / SIGMAS contract from the same steps/shifts recipe — one "
                "source of truth for the first-pass sampling.  Wire its four outputs "
                "into SamplerCustomAdvanced (sampler/sigmas from this node, noise from "
                "any NOISE primitive, latent_image from the 空白AV模板)."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Custom(SEGMENT_TYPE).Input("segments", display_name="segments"),
                io.Int.Input("steps", default=4, min=1, max=1000,
                             tooltip="一采步数。步数蒸馏 LoRA（4step/step600/加速6步 等）只在各自的短窗口内服从 conditioning——超过蒸馏步数后参考身份会被 LoRA 先验冲掉（实测 6 步参考失效、4 步恢复）。与二跑分段精修的 refine 调度合计对齐 T8 发布日程 4+5。"),
                io.Float.Input("shift_video", default=12.0, min=0.01, max=100.0, step=0.01),
                io.Float.Input("shift_audio", default=3.0, min=0.01, max=100.0, step=0.01),
                io.Combo.Input("sampler_name", options=SAMPLER_OPTIONS, default=DEFAULT_SAMPLER_NAME,
                               tooltip="dual_clock_euler = 内置双钟欧拉；选标准采样器名则切 FLOW_AV 补丁。"),
                io.Combo.Input("scheduler", options=SCHEDULER_OPTIONS, default=DEFAULT_SCHEDULER_NAME),
                io.Combo.Input("seed_policy", options=["increment", "fixed"], default="fixed",
                               tooltip="increment = 每镜种子+1；fixed = 全部镜头同一种子。基础种子来自 SamplerCustomAdvanced 的噪声原语（RandomNoise 的种子）。"),
                io.Float.Input("context_noise", default=0.0, min=0.0, max=1.0, step=0.05,
                               tooltip="接缝处对续接上下文帧加噪。0 完全保留上一镜尾帧。"),
                io.Boolean.Input(
                    "release_clip", default=True, advanced=True,
                    tooltip="每镜采样前卸载文本编码器，下镜条件构建时再加载。显存紧张保持开启。",
                ),
                io.Boolean.Input("free_between_segments", default=True, advanced=True),
                io.Vae.Input(
                    "video_vae", optional=True, display_name="视频 VAE",
                    tooltip="视频 VAE（从 长视频编排 节点连线），用于参考图编码。未连接时按模型设置名字加载。",
                ),
                io.Vae.Input(
                    "audio_vae", optional=True, display_name="音频 VAE",
                    tooltip="音频 VAE（从 长视频编排 节点连线）。未连接时按模型设置名字加载。",
                ),
                io.Custom("CLIP").Input(
                    "clip", optional=True, display_name="CLIP",
                    tooltip="CLIP（从 长视频编排 节点的 CLIP 输出连线），用于逐镜条件构建。未连接时按名字加载。",
                ),
            ],
            outputs=[
                io.Guider.Output(
                    display_name="GUIDER",
                    tooltip="分段引导对象 — 接 SamplerCustomAdvanced 的 guider。",
                ),
                io.Sampler.Output(
                    "sampler", display_name="sampler",
                    tooltip="与本节点 steps/shifts 配套的一采 SAMPLER — 接 SamplerCustomAdvanced 的 sampler。",
                ),
                io.Sigmas.Output(
                    "sigmas", display_name="sigmas",
                    tooltip="与本节点 steps/shifts/scheduler 配套的一采调度表 — 接 SamplerCustomAdvanced 的 sigmas。",
                ),
            ],
        )

    @classmethod
    def execute(cls, model, segments, steps, shift_video, shift_audio,
                sampler_name, scheduler,
                seed_policy, context_noise,
                release_clip=True, free_between_segments=True,
                video_vae=None, audio_vae=None, clip=None):
        plan = segments
        if not isinstance(plan, dict) or not plan.get("segments"):
            raise ValueError("Connect the storyboard node's 'segments' output to the segment guider.")
        if clip is not None:
            plan["clip"] = clip
        if video_vae is not None:
            plan["video_vae"] = video_vae
        if audio_vae is not None:
            plan["audio_vae"] = audio_vae
        start = int(plan["range"]["start"]) - 1
        end = int(plan["range"]["end"]) - 1
        # Pre-build the first executed shot's conditioning once: the loop
        # reuses it (no double work) and it doubles as the conditioning output
        # for a downstream second-pass BasicGuider.
        clip = plan.get("clip") or _load_clip(plan)
        first_bundle = build_segment_bundle(
            plan, start, first_frame_override=None, clip=clip,
            connected=(plan.get("connected_by_segment") or {}).get(start + 1),
        )
        guider = _SegmentedAVGuider(
            model, plan,
            steps=steps, shift_video=shift_video, shift_audio=shift_audio,
            sampler_name=sampler_name, scheduler=scheduler,
            seed_policy=seed_policy, context_noise=context_noise,
            release_clip=release_clip, free_between_segments=free_between_segments,
            video_vae=video_vae, audio_vae=audio_vae,
            prebuilt=(first_bundle, clip),
        )
        # The sampler contract is built HERE from the same steps/shifts recipe
        # the loop uses — one source of truth, no separate contract node whose
        # widgets could drift out of sync.
        _contract_model, contract_sampler, contract_sigmas = setup_dual_clock_sampling_gh(
            model, build_empty_plan_latent(plan, start, end),
            int(steps), float(shift_video), float(shift_audio),
            sampler_name or DEFAULT_SAMPLER_NAME, scheduler or DEFAULT_SCHEDULER_NAME,
        )
        return io.NodeOutput(guider, contract_sampler, contract_sigmas)




def run_refine_segments(model, segments, refined_latent, *, base_steps, coarse_steps,
                        shift_video, shift_audio, sampler_name, scheduler, seed,
                        seed_policy, cfg, release_clip, free_between_segments,
                        audio_policy="keep", clip=None, video_vae=None,
                        audio_vae=None, external_sigmas=None, external_noise=None,
                        external_sampler=None, carry_source="anchor",
                        context_noise=0.15):
    """Segmented HD second pass: refine an upscaled whole-film AV latent one
    shot at a time.

    This is the long-video equivalent of the T8 two-pass HIGH stage.  The
    first pass already made the film continuous, so there is no carry here —
    each shot's span is sliced from the upscaled latent by ``lk_layout``,
    re-conditioned at the UPSCALED canvas (references re-encoded in HD, the
    HIGH-conditioning equivalent), sampled with the per-shot LoRA set and the
    refine schedule, and the spans are concatenated back.  With
    ``audio_policy="keep"`` the first pass's audio is composited back
    verbatim (the T8 ``legacy_policy`` equivalent) — audio is
    resolution-independent, so the second pass must not rewrite it.

    The refine schedule is computed internally as the tail of the full
    ``base_steps`` schedule split at ``coarse_steps`` (9/4 -> 5 refine steps,
    matching the T8 ParityPlan) when no external sigmas are wired.
    """
    plan = segments
    if not isinstance(plan, dict) or not plan.get("segments"):
        raise ValueError("Connect the storyboard node's 'segments' output to the refine node.")
    if not getattr(refined_latent.get("samples", None), "is_nested", False):
        raise ValueError("二跑精修需要放大后的嵌套 AV 潜空间（来自 AV潜空间放大 / 学习型放大器）。")
    video, audio = nested_av_parts(refined_latent)
    layout = refined_latent.get("lk_layout") or {}
    entries = layout.get("segments") or []
    start = int(plan["range"]["start"]) - 1
    end = int(plan["range"]["end"]) - 1
    span_count = end - start + 1
    if len(entries) < span_count:
        raise ValueError(
            f"放大后的潜空间布局只有 {len(entries)} 个镜头段，但计划范围需要 {span_count} 段——"
            "av_latent 必须来自同一分段计划的一采输出放大。"
        )

    # Per-shot conditioning must be built at the UPSCALED canvas so the
    # reference images re-encode in HD (the HIGH-conditioning equivalent).
    height = int(video.shape[3]) * 16
    width = int(video.shape[4]) * 16
    plan = dict(plan)
    plan["width"] = width
    plan["height"] = height
    if clip is not None:
        plan["clip"] = clip
    if video_vae is not None:
        plan["video_vae"] = video_vae
    if audio_vae is not None:
        plan["audio_vae"] = audio_vae
    if external_noise is not None:
        # comfy-core noise objects bake their own seed; honor it as the base
        # seed for the per-shot noise split instead of the widget value.
        obj_seed = getattr(external_noise, "seed", None)
        if isinstance(obj_seed, int):
            seed = obj_seed

    if external_sigmas is None:
        # Self-computed refine schedule: the full base schedule split at the
        # coarse cutoff (9/4 -> the remaining 5 sigmas down to 0), matching
        # the T8 ParityPlan without the external contract/plan nodes.
        coarse = max(1, min(int(coarse_steps), int(base_steps) - 1))
        _patched, _sampler, full_schedule = setup_dual_clock_sampling_gh(
            model, refined_latent, int(base_steps), float(shift_video),
            float(shift_audio), sampler_name or DEFAULT_SAMPLER_NAME,
            scheduler or DEFAULT_SCHEDULER_NAME,
        )
        external_sigmas = full_schedule[int(coarse):]

    lora_cache = _lora_cache()
    clip_obj = plan.get("clip")
    progress = None
    if getattr(comfy.utils, "PROGRESS_BAR_ENABLED", True):
        progress = comfy.utils.ProgressBar(max(1, int(base_steps) - max(1, int(coarse_steps))) * max(1, span_count))
    collected = []
    audio_mode = None
    first_positive = None
    previous_samples = None

    for index in range(start, end + 1):
        entry = entries[index - start]
        v0, vt = int(entry["video_offset"]), int(entry["video_tokens"])
        a0, at = int(entry["audio_offset"]), int(entry["audio_steps"])
        seg_video = video[..., v0:v0 + vt, :, :]
        seg_audio = audio[..., a0:a0 + at]

        if clip_obj is None:
            clip_obj = _load_clip(plan)
        bundle = build_segment_bundle(
            plan, index, first_frame_override=None, clip=clip_obj,
            connected=(plan.get("connected_by_segment") or {}).get(index + 1),
        )
        if first_positive is None:
            first_positive = bundle["positive"]
        if release_clip and clip_obj is not None:
            _release_clip(plan, clip_obj)
            clip_obj = None

        # Swap the template's samples for this shot's slice of the upscaled
        # film; the template's metadata (audio mode, keyframe bookkeeping)
        # stays so the sampler contract sees a consistent latent.
        seg_latent = dict(bundle["av_latent"])
        seg_latent["samples"] = comfy.nested_tensor.NestedTensor((seg_video, seg_audio))

        # Carry the previous shot's tail as an EXTENDED window head
        # ([prev tail | own span], head locked at denoise-mask 0).  Unlike
        # pass 1's overwrite-style carry this must not consume the shot's own
        # tokens - the upscaled layout's span sizes are fixed and the assembled
        # film has to match them 1:1.  Per-shot windows otherwise refine blind,
        # and a high refine sigma re-asserts each shot's own prompt, breaking
        # the cross-shot continuity pass 1 established.
        #
        # ``carry_source`` picks where the head comes from:
        # * "anchor"  - the previous shot's tail sliced from the UPSCALED
        #   first-pass latent (fixed quality level).  The carried CONTENT is
        #   identical in both modes - the previous shot's last frames; per-shot
        #   references never enter the carry - so multi-reference films are
        #   unaffected.  What changes is that the context quality no longer
        #   depends on how the previous REFINED shots went, so pass-2 quality
        #   cannot drift hop over hop (an 8-shot film refines every shot
        #   against the same anchor instead of a 7-link quality chain).
        # * "chained" - the previous shot's REFINED output (pre-anchor
        #   behavior): the seam continues from exactly what the final film
        #   shows there, at the cost of chaining every shot's quality to the
        #   accumulated state of all previous refine hops.
        carry_t = carry_a = 0
        head_video = head_audio = None
        tokens = audio_steps = 0
        if carry_source != "chained" and index > start:
            prev_entry = entries[index - 1 - start]
            tokens = min(CARRY_TOKENS, int(prev_entry["video_tokens"]))
            # The audio head is carried EVEN with audio_policy="keep": H3
            # generates video and audio jointly and aligns them positionally
            # within the window, so a video head without the matching audio
            # head desyncs the streams by ~CARRY_FRAMES for every shot after
            # the first (the model then denoises video against soundtrack that
            # belongs ~1s later - lip/eye jitter and flicker).  The carried
            # audio is locked at denoise-mask 0 and the output audio is still
            # replaced by the first pass's own span below, so the film's
            # soundtrack stays verbatim.
            audio_steps = min(CARRY_AUDIO_STEPS, int(prev_entry["audio_steps"]))
            if tokens > 0:
                pv0, pvt = int(prev_entry["video_offset"]), int(prev_entry["video_tokens"])
                pa0, pat = int(prev_entry["audio_offset"]), int(prev_entry["audio_steps"])
                # video is [B, C, T, H, W] - time is dim 2, so the tail slice
                # needs the explicit trailing dims (unlike the audio part,
                # whose [B, C, F, T] layout keeps time in the LAST dim).
                head_video = video[..., pv0 + pvt - tokens:pv0 + pvt, :, :]
                head_audio = (
                    audio[..., pa0 + pat - audio_steps:pa0 + pat]
                    if audio_steps > 0 else None
                )
        elif carry_source == "chained" and previous_samples is not None:
            prev_video, prev_audio = previous_samples.unbind()
            tokens = min(CARRY_TOKENS, int(prev_video.shape[2]))
            # See the anchor branch: the audio head is required for model-side
            # A/V alignment even when the output audio is kept verbatim.
            audio_steps = min(CARRY_AUDIO_STEPS, int(prev_audio.shape[-1]))
            if tokens > 0:
                head_video = prev_video[..., -tokens:, :, :]
                head_audio = prev_audio[..., -audio_steps:] if audio_steps > 0 else None
        if head_video is not None:
            window_video = torch.cat([head_video.to(seg_video), seg_video], dim=2)
            window_audio = (
                torch.cat([head_audio.to(seg_audio), seg_audio], dim=-1)
                if head_audio is not None else seg_audio
            )
            video_mask = torch.ones(
                (1, 1, window_video.shape[2], window_video.shape[3], window_video.shape[4]),
                device=window_video.device, dtype=torch.float32,
            )
            # The head must NOT be hard-locked at 0: the refine pass starts at
            # sigma[coarse] (mid-schedule), and a zero-noise head amid a
            # half-noisy span is an out-of-distribution input for the temporal
            # attention - the span's early frames get pulled toward the head's
            # clean statistics and the trajectory oscillates (visible flicker
            # in every shot after the first, in BOTH carry modes, while shot 1
            # - which has no head - stays clean; pass 1 shows the same knob
            # fixes it).  context_noise keeps the head lightly noised so it
            # stays in-distribution; the head is trimmed from the output, so
            # this only affects what the model sees.
            video_mask[..., :tokens, :, :] = max(0.0, min(1.0, float(context_noise)))
            audio_mask = torch.ones(
                (1, 1, window_audio.shape[2], window_audio.shape[3]),
                device=window_audio.device, dtype=torch.float32,
            )
            if audio_steps > 0:
                audio_mask[..., :audio_steps] = 0.0
            seg_latent = dict(seg_latent)
            seg_latent["samples"] = comfy.nested_tensor.NestedTensor((window_video, window_audio))
            seg_latent["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
            carry_t, carry_a = tokens, audio_steps
        bundle = dict(bundle)
        bundle["av_latent"] = seg_latent

        segment = plan["segments"][index]
        seg_seed = int(seed) if seed_policy == "fixed" else int(seed) + (index - start)
        seg_model = _apply_model_loras(model, segment.get("loras") or [], lora_cache)
        samples = _sample_bundle(
            seg_model, bundle,
            steps=int(base_steps), shift_video=float(shift_video), shift_audio=float(shift_audio),
            sampler_name=sampler_name, scheduler=scheduler, seed=seg_seed, cfg=float(cfg),
            progress=progress,
            external_noise=external_noise, external_sigmas=external_sigmas,
            external_sampler=external_sampler,
        )
        out_video, out_audio = nested_av_parts(samples)
        if audio_mode is None:
            audio_mode = samples.get("gh_h3_audio_mode") or plan.get("audio_mode") or "native"
        if carry_t > 0:
            # Trim the duplicated carried span so the assembled film keeps the
            # upscaled layout's offsets 1:1.
            out_video = out_video[..., carry_t:, :, :]
            out_audio = out_audio[..., carry_a:]
        if audio_policy == "keep":
            # Composite the first pass's audio back verbatim AFTER the trim:
            # the sampled window carries the locked previous-tail audio only
            # for model-side A/V alignment - the film's soundtrack stays the
            # first pass's own span, untouched.
            out_audio = seg_audio
        previous_samples = samples["samples"]
        collected.append({
            "segment": index + 1,
            "video": out_video.detach().to("cpu"),
            "audio": out_audio.detach().to("cpu"),
        })
        del samples, seg_model
        if free_between_segments:
            try:
                import comfy.model_management as model_management
                model_management.soft_empty_cache()
            except Exception:
                pass

    out_video = torch.cat([item["video"] for item in collected], dim=2)
    out_audio = torch.cat([item["audio"] for item in collected], dim=-1)
    out_entries = []
    video_offset = 0
    audio_offset = 0
    for item, entry in zip(collected, entries[start:end + 1]):
        tokens = int(item["video"].shape[2])
        out_entries.append({
            "segment": item["segment"],
            "video_offset": video_offset,
            "video_tokens": tokens,
            "audio_offset": audio_offset,
            "audio_steps": int(item["audio"].shape[-1]),
            "frames": _pixel_frames_from_latent_t(tokens),
        })
        video_offset += tokens
        audio_offset += int(item["audio"].shape[-1])
    out = {
        "samples": comfy.nested_tensor.NestedTensor((out_video, out_audio)),
        "gh_h3_audio_mode": audio_mode or "native",
        "lk_layout": {
            "version": 1,
            "pass": "second",
            "latent_continue": bool(plan.get("latent_continue", True)),
            "range": {"start": start + 1, "end": end + 1},
            "video_hw": [int(out_video.shape[3]), int(out_video.shape[4])],
            "segments": out_entries,
        },
    }
    return out, first_positive




class MiniMaxH3RefineSegmentsGH(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LKH3RefineSegments",
            display_name="MiniMax-H3 二跑分段精修（LK）",
            category="LK/MiniMax H3 Integration",
            description=(
                "Segmented HD second pass: refines the upscaled whole-film AV latent one "
                "shot at a time — per-shot conditioning re-encoded at the upscaled canvas "
                "(the HIGH-conditioning equivalent), per-shot LoRA, external refine "
                "schedule, first-pass audio kept verbatim."
            ),
            inputs=[
                io.Latent.Input(
                    "av_latent",
                    tooltip="放大后的整条 AV 潜空间（含 lk_layout）— 来自 AV潜空间放大 或学习型放大器，其布局必须与分段计划一致。",
                ),
                io.Custom(SEGMENT_TYPE).Input("segments", display_name="segments"),
                io.Model.Input("model"),
                io.Float.Input("shift_video", default=12.0, min=0.01, max=100.0, step=0.01),
                io.Float.Input("shift_audio", default=3.0, min=0.01, max=100.0, step=0.01),
                io.Int.Input("base_steps", default=9, min=2, max=1000,
                             tooltip="完整基础调度的步数（对应 T8 ParityPlan 的 base）。精修步数 = base - coarse。"),
                io.Int.Input("coarse_steps", default=4, min=1, max=1000,
                             tooltip="粗采 cutoff：基础调度从第 N 个 σ 起进入精修段（对应 T8 ParityPlan 的 coarse）。"),
                io.Combo.Input(
                    "audio_policy", options=["keep", "regenerate"], default="keep",
                    tooltip="keep = 二跑完全保留一采生成的音频（等价 T8 的 legacy_policy）；regenerate = 音频随视频一起精修。",
                ),
                io.Boolean.Input(
                    "release_clip", default=True, advanced=True,
                    tooltip="每镜条件构建后卸载文本编码器，下镜再加载。显存紧张保持开启。",
                ),
                io.Boolean.Input("free_between_segments", default=True, advanced=True),
                io.Combo.Input(
                    "carry_source", options=["anchor", "chained"], default="anchor",
                    tooltip="每镜续接上下文的来源。anchor = 取自放大的首采潜空间（固定质量锚点，逐镜不复利，8 镜以上全片质量均匀）；chained = 取上一镜的精修输出（接缝与精修结果严格连续，但逐镜链式累积）。两种模式的续接内容都是上一镜末帧——分镜各自参考不同图片不受影响，参考从不进入续接。",
                ),
                io.Float.Input(
                    "context_noise", default=0.15, min=0.0, max=1.0, step=0.05,
                    tooltip="续接头的重加噪强度（等价一采的 Context noise）。精修从中段 σ 起跑，头部若完全锁死（0）属于分布外输入，镜 2 起会闪烁；0.1-0.2 让采样器轻度重净头部。头部会被裁掉、不进成片，调它只影响模型所见。",
                ),
                io.Custom("CLIP").Input(
                    "clip", optional=True, display_name="CLIP",
                    tooltip="CLIP（从 MiniMaxH3 长视频编排 连线）。未连接时按模型设置名字加载。",
                ),
                io.Vae.Input(
                    "video_vae", optional=True, display_name="视频 VAE",
                    tooltip="视频 VAE（从 MiniMaxH3 长视频编排 连线）。未连接时按模型设置名字加载。",
                ),
                io.Vae.Input(
                    "audio_vae", optional=True, display_name="音频 VAE",
                    tooltip="音频 VAE（从 MiniMaxH3 长视频编排 连线）。未连接时按模型设置名字加载。",
                ),
                io.Noise.Input(
                    "noise", optional=True, display_name="噪声",
                    tooltip="外接 NOISE 原语（贡献每镜种子与噪声生成）。未连接时内置生成。",
                ),
            ],
            outputs=[
                io.Latent.Output(
                    "av_latent", display_name="AV 潜空间",
                    tooltip="整条高清 AV 潜空间（含 lk_layout）— 接 AV解码 出片。",
                ),
            ],
        )

    @classmethod
    def execute(cls, av_latent, segments, model, shift_video, shift_audio,
                base_steps, coarse_steps, audio_policy,
                release_clip=True, free_between_segments=True,
                carry_source="anchor", context_noise=0.15,
                clip=None, video_vae=None, audio_vae=None,
                noise=None):
        return io.NodeOutput(*run_refine_segments(
            model, segments, av_latent,
            base_steps=base_steps, coarse_steps=coarse_steps,
            shift_video=shift_video, shift_audio=shift_audio,
            sampler_name=DEFAULT_SAMPLER_NAME, scheduler=DEFAULT_SCHEDULER_NAME,
            seed=0, seed_policy="fixed", cfg=1.0,
            release_clip=release_clip, free_between_segments=free_between_segments,
            audio_policy=audio_policy, carry_source=carry_source,
            context_noise=context_noise,
            clip=clip, video_vae=video_vae, audio_vae=audio_vae,
            external_sigmas=None, external_noise=noise, external_sampler=None,
        ))
