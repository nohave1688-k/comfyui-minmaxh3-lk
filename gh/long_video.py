"""Long-video multi-segment continuation for the unified MiniMax H3 node.

The integrated node owns the segment table (UI) and emits a lightweight
``LK_H3_SEGMENTS`` plan.  ``MiniMaxH3LongVideoRunnerGH`` walks the selected
range, rebuilds that segment's conditioning, carries the previous segment's
latent tail into the next segment's head (``_carry_latent``), applies the
segment's LoRAs, samples with the dual clock, decodes, trims the overlap and
concatenates the result.  It also emits the whole-video packed AV latent
(``av_latent``) with ``lk_layout`` metadata.

Connecting an upscaled AV latent (via ``LKH3AVUpscale``) to ``refine_latent``
switches the same loop into the HD second pass: per-segment slices are
re-sampled from a truncated sigma schedule instead of being generated fresh.

Only ``MODEL`` is patched per segment; CLIP stays untouched (the UI only exposes
a model strength for each LoRA slot), so prompts keep encoding with the base
text encoder.
"""

from __future__ import annotations

import json
import math
import os

import folder_paths
import nodes as comfy_nodes
import torch
from comfy_api.latest import io

import comfy.nested_tensor
import comfy.utils

from .audio_ops import decode_av_latent
from .conditioning import build_conditioning
from .core import nested_av_parts, temporal_shape
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
SEGMENT_TYPE = "LK_H3_SEGMENTS"


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
    return bundle


def _connected_source_size(connected):
    """Aspect source from connected socket tensors (overrides panel files)."""
    first = connected.get("first_frame")
    if first is not None:
        return int(first.shape[2]), int(first.shape[1])
    for image in connected.get("images") or []:
        return int(image.shape[2]), int(image.shape[1])
    return None, None


def build_multi_plan(*, aspect, megapixels, segments_json, clip_name, video_vae_name,
                     audio_vae_name, audio_mode, audio_denoise_strength,
                     strict_prompt_tags, ref_image_size, connected_media=None):
    """Load shared models, compute the global canvas and build the start preview."""
    connected_media = connected_media or {}
    config = parse_segments(segments_json)
    clip = comfy_nodes.CLIPLoader().load_clip(clip_name, "minimax")[0]
    video_vae = comfy_nodes.VAELoader().load_vae(video_vae_name)[0]
    audio_vae = comfy_nodes.VAELoader().load_vae(audio_vae_name)[0]

    connected_first = connected_media.get(1) or {}
    source_width, source_height = _connected_source_size(connected_first)
    if source_width is None:
        source_width, source_height = _segment_source_size(config["segments"][0])
    width, height = _media().calculate_canvas(aspect, megapixels, source_width, source_height)
    for segment in config["segments"]:
        segment["length"] = _media().calculate_length(segment["duration"])

    plan = {
        **config,
        "width": width,
        "height": height,
        "clip_name": clip_name,
        "video_vae_name": video_vae_name,
        "audio_vae_name": audio_vae_name,
        "aspect": aspect,
        "megapixels": megapixels,
        "audio_mode": audio_mode,
        "audio_denoise_strength": audio_denoise_strength,
        "strict_prompt_tags": strict_prompt_tags,
        "ref_image_size": ref_image_size,
        "clip": clip,
        "video_vae": video_vae,
        "audio_vae": audio_vae,
        "connected_media": connected_media,
        "preview": None,
    }
    start_index = config["range"]["start"] - 1
    preview = build_segment_bundle(plan, start_index, clip=clip,
                                   connected=connected_media.get(start_index + 1))
    plan["preview"] = {"index": start_index, "bundle": preview}
    return preview, preview["refmods"], plan


def _load_clip(plan):
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


def _refine_sigmas(patched_model, scheduler, steps, sigma_start, shift_video):
    """Truncated dual-clock sigma schedule starting at ``sigma_start``.

    For native_flow the truncation is exact: the shifted schedule is a
    monotone map of the base linspace, so starting the base linspace at the
    inverse-shifted ``sigma_start`` reproduces the full schedule's shape below
    the cut.  Other schedulers keep their native tail and get the exact
    ``sigma_start`` prepended.
    """
    sigma_start = max(1e-4, min(1.0, float(sigma_start)))
    shift_video = float(shift_video)
    if scheduler == DEFAULT_SCHEDULER_NAME:
        base_start = sigma_start / (shift_video - sigma_start * (shift_video - 1.0))
        base = torch.linspace(base_start, 0.0, int(steps) + 1, dtype=torch.float32)
        return shift_sigma(base, shift_video)
    model_sampling = patched_model.get_model_object("model_sampling")
    full = _scheduler_sigmas(model_sampling, scheduler, int(steps), shift_video).cpu().float()
    tail = full[full <= sigma_start + 1e-6]
    if tail.numel() < 2:
        tail = torch.tensor([sigma_start, 0.0], dtype=torch.float32)
    elif float(tail[0]) < sigma_start - 1e-6:
        tail = torch.cat([torch.tensor([sigma_start], dtype=torch.float32), tail])
    return tail


def _fix_regenerate_audio_noise(noise, av_latent, sigma_video, shift_video, shift_audio,
                                patched_model):
    """Re-anchor the audio part of the packed noise to the audio clock.

    ``CONST.noise_scaling`` mixes the whole packed latent at the video sigma,
    but the dual-clock trajectory starts the audio stream at
    ``time_shift(sigma_video)``.  Pre-distorting the audio noise makes the
    post-scaling audio part land on its own clock:
    ``sigma_v * eps' + (1 - sigma_v) * x0 == sigma_a * eps + (1 - sigma_a) * x0``.
    Only needed when the second pass regenerates audio; a locked audio mask
    (refine_audio="keep") restores the clean latent every step regardless.
    """
    sigma_video = float(sigma_video)
    if sigma_video >= 1.0 - 1e-6:
        return noise
    if not getattr(noise, "is_nested", False):
        return noise
    parts = noise.unbind()
    if len(parts) != 2:
        return noise
    noise_video, noise_audio = parts
    _, clean_audio = nested_av_parts(av_latent)
    sigma_audio = time_shift_sigma(sigma_video, float(shift_video), float(shift_audio))
    sampling_obj = patched_model.get_model_object("model_sampling")
    noise_scale = float(getattr(sampling_obj, "noise_scale", 1.0) or 1.0)
    corrected = (
        sigma_audio * noise_audio.float()
        + (sigma_video - sigma_audio) * clean_audio.float()
    ) / max(1e-6, sigma_video * noise_scale)
    return comfy.nested_tensor.NestedTensor((noise_video, corrected.to(noise_audio.dtype)))


def _resolve_refine_layout(refine_latent, plan):
    """Resolve per-segment token layout for slicing the whole-video latent.

    The layout metadata (``lk_layout``) written by the first pass is the source
    of truth: it records what was actually trimmed at each seam.  Without it
    (hand-built latents) the current plan's semantics are derived and strongly
    validated against the packed length by the caller.  Requires full-segment
    coverage so absolute offsets exist for every segment.
    """
    segments_plan = plan["segments"]
    shapes = []
    for segment in segments_plan:
        length = int(segment.get("length") or _media().calculate_length(segment["duration"]))
        shapes.append(temporal_shape(length))
    raw = refine_latent.get("lk_layout")
    raw_entries = raw.get("segments") if isinstance(raw, dict) else None
    if not isinstance(raw_entries, list) or not raw_entries:
        latent_continue = bool(plan.get("latent_continue", True))
        raw_entries = [
            {
                "segment": position + 1,
                "video_tokens": video_t - (CARRY_TOKENS if latent_continue and position > 0 else 0),
                "audio_steps": audio_t - (CARRY_AUDIO_STEPS if latent_continue and position > 0 else 0),
            }
            for position, (_frame_count, video_t, audio_t) in enumerate(shapes)
        ]
    if len(raw_entries) != len(shapes):
        raise ValueError(
            "The refine latent's layout does not match the connected plan "
            f"(layout has {len(raw_entries)} segment entries, plan has {len(shapes)}). "
            "Run the first pass over the full segment range with this plan, then upscale its "
            "'av_latent' output for the second pass."
        )
    normalized = []
    for position, (entry, (_frame_count, video_t, audio_t)) in enumerate(zip(raw_entries, shapes)):
        segment = position + 1
        tokens = int(entry.get("video_tokens", video_t))
        if tokens not in (video_t, video_t - CARRY_TOKENS):
            raise ValueError(
                f"Segment {segment}: latent length {tokens} tokens does not match the plan "
                f"(expected {video_t} or {video_t - CARRY_TOKENS}). Segment durations changed "
                "after the latent was produced; re-run the first pass to refresh it."
            )
        audio_steps = int(entry.get("audio_steps", audio_t))
        if audio_steps not in (audio_t, audio_t - CARRY_AUDIO_STEPS):
            raise ValueError(
                f"Segment {segment}: latent audio length {audio_steps} steps does not match the "
                f"plan (expected {audio_t} or {audio_t - CARRY_AUDIO_STEPS}). Re-run the first "
                "pass to refresh the latent."
            )
        normalized.append({
            "segment": segment,
            "video_tokens": tokens,
            "audio_steps": audio_steps,
            "frames": _pixel_frames_from_latent_t(tokens),
        })
    video_offset = 0
    audio_offset = 0
    frames_before = 0
    for entry in normalized:
        entry["video_offset"] = video_offset
        entry["audio_offset"] = audio_offset
        entry["frames_before"] = frames_before
        video_offset += entry["video_tokens"]
        audio_offset += entry["audio_steps"]
        frames_before += entry["frames"]
    return {
        "version": 1,
        "pass": str(raw.get("pass")) if isinstance(raw, dict) and raw.get("pass") else "first",
        "latent_continue": bool(raw.get("latent_continue")) if isinstance(raw, dict)
        else bool(plan.get("latent_continue", True)),
        "segments": normalized,
    }


def _assemble_whole_av_latent(collected, audio_mode, latent_continue, start, end):
    """Concatenate the executed segments into one whole-video AV latent.

    Carried heads are already trimmed by the caller, so the packed latent
    decodes 1:1 into the concatenated output frames.  ``lk_layout`` records the
    per-segment token counts (the refine pass's slicing map).
    """
    video = torch.cat([item["video"] for item in collected], dim=2)
    audio = torch.cat([item["audio"] for item in collected], dim=-1)
    entries = []
    for item in collected:
        tokens = int(item["video"].shape[2])
        entries.append({
            "segment": item["segment"],
            "video_tokens": tokens,
            "audio_steps": int(item["audio"].shape[-1]),
            "frames": _pixel_frames_from_latent_t(tokens),
        })
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


def _assemble_refined_latent(refine_latent, layout, collected, start, end):
    """Splice the refined segment latents back into the untouched ones.

    The result covers every segment again (layout unchanged), so the output can
    feed further refine rounds or the upscale node without re-deriving offsets.
    """
    video, audio = nested_av_parts(refine_latent)
    ref_vmask, ref_amask = _collapse_av_mask(refine_latent)
    refined_by_segment = {item["segment"]: item for item in collected}
    video_parts = []
    audio_parts = []
    video_masks = []
    audio_masks = []
    for entry in layout["segments"]:
        refined = refined_by_segment.get(entry["segment"])
        if refined is not None:
            video_parts.append(refined["video"])
            audio_parts.append(refined["audio"])
            video_masks.append(refined["video_mask"])
            audio_masks.append(refined["audio_mask"])
            continue
        v_slice = video[..., entry["video_offset"]:entry["video_offset"] + entry["video_tokens"], :, :]
        a_slice = audio[..., entry["audio_offset"]:entry["audio_offset"] + entry["audio_steps"]]
        video_parts.append(v_slice)
        audio_parts.append(a_slice)
        video_masks.append(
            None if ref_vmask is None
            else ref_vmask[..., entry["video_offset"]:entry["video_offset"] + entry["video_tokens"], :, :]
        )
        audio_masks.append(
            None if ref_amask is None
            else ref_amask[..., entry["audio_offset"]:entry["audio_offset"] + entry["audio_steps"]]
        )
    out = {
        "samples": comfy.nested_tensor.NestedTensor(
            (torch.cat(video_parts, dim=2), torch.cat(audio_parts, dim=-1))
        ),
        "gh_h3_audio_mode": refine_latent.get("gh_h3_audio_mode") or "native",
        "lk_layout": {
            "version": 1,
            "pass": "refine",
            "latent_continue": bool(layout.get("latent_continue")),
            "range": {"start": start + 1, "end": end + 1},
            "refined": [item["segment"] for item in collected],
            "segments": [dict(entry) for entry in layout["segments"]],
        },
    }
    if any(mask is not None for mask in video_masks) or any(mask is not None for mask in audio_masks):
        filled_video_masks = []
        filled_audio_masks = []
        for entry, vmask, amask in zip(layout["segments"], video_masks, audio_masks):
            tokens = entry["video_tokens"]
            steps = entry["audio_steps"]
            filled_video_masks.append(
                vmask if vmask is not None
                else torch.ones((video.shape[0], 1, tokens, video.shape[3], video.shape[4]),
                                device=video.device, dtype=video.dtype)
            )
            filled_audio_masks.append(
                amask if amask is not None
                else torch.ones((audio.shape[0], 1, audio.shape[2], steps),
                                device=audio.device, dtype=audio.dtype)
            )
        out["noise_mask"] = comfy.nested_tensor.NestedTensor(
            (torch.cat(filled_video_masks, dim=2), torch.cat(filled_audio_masks, dim=-1))
        )
    return out


def _sample_bundle(model, bundle, *, steps, shift_video, shift_audio,
                   sampler_name, scheduler, seed, cfg, progress=None,
                   refine_sigma_start=None, refine_audio_regen=False):
    av_latent = bundle["av_latent"]
    patched_model, sampler, default_sigmas = setup_dual_clock_sampling_gh(
        model, av_latent, steps, shift_video, shift_audio, sampler_name, scheduler
    )
    if refine_sigma_start is not None:
        sigmas = _refine_sigmas(patched_model, scheduler, steps, refine_sigma_start, shift_video)
    else:
        sigmas = default_sigmas
    guider = _make_guider(patched_model, bundle["positive"], cfg)
    samples = av_latent["samples"]
    noise = _make_noise(samples, seed)
    if refine_audio_regen:
        noise = _fix_regenerate_audio_noise(
            noise, av_latent, float(sigmas[0]), shift_video, shift_audio, patched_model
        )
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


def _audio_samples(audio, seconds):
    sample_rate = int(audio.get("sample_rate") or 0)
    return max(0, int(round(float(seconds) * sample_rate))) if sample_rate else 0


def _concat_audio(clips):
    if not clips:
        return None
    sample_rate = int(clips[0].get("sample_rate") or 0)
    waveforms = []
    for clip in clips:
        waveform = clip.get("waveform")
        if waveform is None:
            continue
        if sample_rate and int(clip.get("sample_rate") or sample_rate) != sample_rate:
            waveform = torchaudio_resample(waveform, int(clip["sample_rate"]), sample_rate)
        waveforms.append(waveform)
    if not waveforms:
        return None
    return {"waveform": torch.cat(waveforms, dim=-1), "sample_rate": sample_rate}


def torchaudio_resample(waveform, source_rate, target_rate):
    try:
        import torchaudio
        return torchaudio.functional.resample(waveform, source_rate, target_rate)
    except Exception:
        # Linear fallback keeps the loop alive on installs without torchaudio.
        ratio = target_rate / float(source_rate)
        length = max(1, int(round(waveform.shape[-1] * ratio)))
        indices = torch.linspace(0, waveform.shape[-1] - 1, length).round().long()
        return waveform[..., indices]


def run_long_video(model, segments, *, steps, shift_video, shift_audio,
                   sampler_name, scheduler, seed, seed_policy, cfg,
                   release_clip, free_between_segments, context_noise=0.0,
                   refine_latent=None, refine_denoise=0.45, refine_audio="keep"):
    plan = segments
    if not isinstance(plan, dict) or not plan.get("segments"):
        raise ValueError("Connect the integrated node's 'segments' output to the long video sampler.")
    start = int(plan["range"]["start"]) - 1
    end = int(plan["range"]["end"]) - 1
    latent_continue = bool(plan.get("latent_continue", True))
    refine_mode = refine_latent is not None
    refine_denoise = max(0.05, min(1.0, float(refine_denoise)))
    refine_audio_regen = refine_mode and str(refine_audio).lower() == "regenerate"
    lora_cache = _lora_cache()
    clip = plan.get("clip")
    frames_out = []
    audio_out = []
    reports = []
    infos = []
    cursor = 0
    preview = plan.get("preview") or {}
    progress = None
    if getattr(comfy.utils, "PROGRESS_BAR_ENABLED", True):
        progress = comfy.utils.ProgressBar(max(1, int(steps)) * max(1, end - start + 1))
    previous_samples = None
    collected = []
    audio_mode = None
    refine_layout = None
    refine_video = refine_audio_latent = None
    if refine_mode:
        refine_layout = _resolve_refine_layout(refine_latent, plan)
        refine_video, refine_audio_latent = nested_av_parts(refine_latent)
        total_tokens = sum(entry["video_tokens"] for entry in refine_layout["segments"])
        total_steps = sum(entry["audio_steps"] for entry in refine_layout["segments"])
        if total_tokens != int(refine_video.shape[2]) or total_steps != int(refine_audio_latent.shape[-1]):
            raise ValueError(
                "The refine latent's packed length does not match its layout metadata "
                f"(video {int(refine_video.shape[2])} vs {total_tokens} tokens, "
                f"audio {int(refine_audio_latent.shape[-1])} vs {total_steps} steps). "
                "Re-run the first pass with the current plan to refresh the latent."
            )

    for index in range(start, end + 1):
        segment = plan["segments"][index]
        if preview.get("index") == index and preview.get("bundle") is not None:
            bundle = preview["bundle"]
        else:
            if clip is None:
                clip = _load_clip(plan)
            # No automatic previous-tail keyframe: continuation is carried in
            # latent space below, not as a first-frame conditioning image.
            bundle = build_segment_bundle(
                plan, index, first_frame_override=None, clip=clip,
                connected=(plan.get("connected_media") or {}).get(index + 1),
            )

        if release_clip and clip is not None:
            _release_clip(plan, clip)
            clip = None

        seg_seed = int(seed) if seed_policy == "fixed" else int(seed) + (index - start)
        seg_model = _apply_model_loras(model, segment.get("loras") or [], lora_cache)

        carry_t = carry_a = 0
        carried = False
        seg_vmask = seg_amask = None
        absolute_start = None
        if refine_mode:
            entry = refine_layout["segments"][index]
            video_slice = refine_video[
                ..., entry["video_offset"]:entry["video_offset"] + entry["video_tokens"], :, :
            ]
            audio_slice = refine_audio_latent[
                ..., entry["audio_offset"]:entry["audio_offset"] + entry["audio_steps"]
            ]
            video_mask = torch.ones(
                (video_slice.shape[0], 1) + tuple(video_slice.shape[2:]),
                device=video_slice.device, dtype=torch.float32,
            )
            # The video stream refines everywhere (pinned seam rows would keep
            # their upscaled first-pass blur in an otherwise sharp pass).
            audio_shape = (audio_slice.shape[0], 1, audio_slice.shape[2], audio_slice.shape[3])
            if refine_audio_regen:
                audio_mask = torch.ones(audio_shape, device=audio_slice.device, dtype=torch.float32)
            else:
                # Hold the first-pass audio bit-exact through the second pass.
                audio_mask = torch.zeros(audio_shape, device=audio_slice.device, dtype=torch.float32)
            bundle = dict(bundle)
            bundle["av_latent"] = {
                "samples": comfy.nested_tensor.NestedTensor((video_slice, audio_slice)),
                "noise_mask": comfy.nested_tensor.NestedTensor((video_mask, audio_mask)),
                "gh_h3_audio_mode": audio_mode or plan.get("audio_mode") or "native",
            }
            seg_vmask, seg_amask = video_mask, audio_mask
            absolute_start = entry["frames_before"]
            samples = _sample_bundle(
                seg_model, bundle,
                steps=int(steps), shift_video=float(shift_video), shift_audio=float(shift_audio),
                sampler_name=sampler_name, scheduler=scheduler, seed=seg_seed, cfg=float(cfg),
                progress=progress, refine_sigma_start=refine_denoise,
                refine_audio_regen=refine_audio_regen,
            )
            seg_video, seg_audio = nested_av_parts(samples)
        else:
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
            )
            if latent_continue:
                # ``samples`` is the segment's output dict; carry its nested latent
                # tensor into the next segment.
                previous_samples = samples["samples"]
            seg_video, seg_audio = nested_av_parts(samples)
            seg_vmask, seg_amask = _collapse_av_mask(samples)
            if carried:
                seg_video = seg_video[..., carry_t:, :, :]
                seg_audio = seg_audio[..., carry_a:]
                if seg_vmask is not None:
                    seg_vmask = seg_vmask[..., carry_t:, :, :]
                if seg_amask is not None:
                    seg_amask = seg_amask[..., carry_a:]

        if audio_mode is None:
            audio_mode = samples.get("gh_h3_audio_mode") or plan.get("audio_mode") or "native"
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
        frames, audio, _video_latent, _audio_latent = decode_av_latent(
            samples, plan["video_vae"], plan["audio_vae"]
        )
        del samples, seg_model
        frames = frames.detach().to("cpu")
        if audio is not None and audio.get("waveform") is not None:
            audio = dict(audio)
            audio["waveform"] = audio["waveform"].detach().to("cpu")

        if carried:
            if int(frames.shape[0]) > CARRY_FRAMES:
                frames = frames[CARRY_FRAMES:]
            trim_samples = _audio_samples(audio, CARRY_FRAMES / 24.0)
            if trim_samples and audio.get("waveform") is not None:
                audio = dict(audio)
                audio["waveform"] = audio["waveform"][..., trim_samples:]

        infos.append({
            "segment": index + 1,
            "seed": seg_seed,
            "start_frame": cursor if absolute_start is None else absolute_start,
            "frames": int(frames.shape[0]),
            "carried_frames": CARRY_FRAMES if carried else 0,
            "latent_continue": bool(latent_continue),
            "refined": bool(refine_mode),
            "duration_seconds": int(frames.shape[0]) / 24.0,
        })
        frames_out.append(frames)
        if audio is not None:
            audio_out.append(audio)
        cursor += int(frames.shape[0])
        if refine_mode:
            reports.append(
                f"[segment {index + 1}] pass=refine seed={seg_seed} frames={int(frames.shape[0])} "
                f"denoise={refine_denoise:.2f} audio={refine_audio} "
                f"loras={len(segment.get('loras') or [])}"
            )
        else:
            reports.append(
                f"[segment {index + 1}] seed={seg_seed} frames={int(frames.shape[0])} "
                f"carried={CARRY_FRAMES if carried else 0} loras={len(segment.get('loras') or [])}"
            )

        if free_between_segments:
            try:
                import comfy.model_management as model_management
                model_management.soft_empty_cache()
            except Exception:
                pass

    if not frames_out:
        raise RuntimeError("Long video sampler produced no frames.")
    all_frames = torch.cat(frames_out, dim=0)
    all_audio = _concat_audio(audio_out)
    last_frame = all_frames[-1:]
    if refine_mode:
        av_latent_out = _assemble_refined_latent(refine_latent, refine_layout, collected, start, end)
    else:
        av_latent_out = _assemble_whole_av_latent(collected, audio_mode, latent_continue, start, end)
    segment_info = json.dumps(
        {
            "pass": "refine" if refine_mode else "first",
            "refine_denoise": float(refine_denoise) if refine_mode else None,
            "refine_audio": str(refine_audio) if refine_mode else None,
            "range": {"start": start + 1, "end": end + 1},
            "latent_continue": latent_continue,
            "context_noise": float(context_noise),
            "width": int(collected[0]["video"].shape[4]) * 16,
            "height": int(collected[0]["video"].shape[3]) * 16,
            "total_frames": int(all_frames.shape[0]),
            "segments": infos,
        },
        ensure_ascii=False,
        indent=2,
    )
    return all_frames, all_audio, last_frame, "\n".join(reports), segment_info, av_latent_out


class MiniMaxH3LongVideoRunnerGH(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LKH3LongVideo",
            display_name="MiniMax-H3 长视频分段采样（LK）",
            category="LK/MiniMax H3 Integration",
            description=(
                "Runs the integrated node's 1-8 segment plan: per-segment LoRA, "
                "previous-tail continuation, dual-clock sampling, decode and concatenation."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Custom(SEGMENT_TYPE).Input("segments", display_name="segments"),
                io.Int.Input("steps", default=4, min=1, max=1000),
                io.Float.Input("shift_video", default=12.0, min=0.01, max=100.0, step=0.01, advanced=True),
                io.Float.Input("shift_audio", default=3.0, min=0.01, max=100.0, step=0.01, advanced=True),
                io.Combo.Input("sampler_name", options=SAMPLER_OPTIONS, default=DEFAULT_SAMPLER_NAME, optional=True),
                io.Combo.Input("scheduler", options=SCHEDULER_OPTIONS, default=DEFAULT_SCHEDULER_NAME, optional=True),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Combo.Input("seed_policy", options=["increment", "fixed"], default="increment"),
                io.Float.Input(
                    "context_noise", default=0.0, min=0.0, max=1.0, step=0.05,
                    tooltip=(
                        "Noise applied to the carried context frames at each seam. "
                        "0 keeps the previous tail exactly; 0.1-0.3 lets the sampler "
                        "clean the carried face instead of compounding texture. "
                        "Audio remains held for speech continuity."
                    ),
                ),
                io.Float.Input("cfg", default=1.0, min=0.0, max=20.0, step=0.1, advanced=True),
                io.Boolean.Input(
                    "release_clip", default=True, advanced=True,
                    tooltip="Unload the H3 text encoder before each segment's sampling and reload it for the next conditioning. Disable only if VRAM allows keeping CLIP and the DiT together.",
                ),
                io.Boolean.Input("free_between_segments", default=True, advanced=True),
                io.Latent.Input(
                    "refine_latent", optional=True,
                    tooltip=(
                        "Second-pass AV latent (chain: this node's av_latent -> LKH3AVUpscale -> "
                        "here). When connected, the node re-samples the upscaled latent with a "
                        "truncated schedule (HD second pass) instead of generating fresh latents. "
                        "The first pass must have run the full segment range so the latent covers "
                        "every segment."
                    ),
                ),
                io.Float.Input(
                    "refine_denoise", default=0.45, min=0.05, max=1.0, step=0.05,
                    tooltip=(
                        "Starting sigma of the truncated second-pass schedule (= denoise "
                        "strength). 0.3-0.5 sharpens upscaled latents; 1.0 regenerates fully. "
                        "Ignored by the first pass."
                    ),
                ),
                io.Combo.Input(
                    "refine_audio", options=["keep", "regenerate"], default="keep",
                    tooltip=(
                        "Second-pass audio policy. keep: hold the first-pass audio bit-exact "
                        "(locked mask). regenerate: re-denoise the audio stream as well. "
                        "Ignored by the first pass."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output("frames", display_name="frames"),
                io.Audio.Output("audio", display_name="audio"),
                io.Image.Output("last_frame", display_name="last_frame"),
                io.String.Output("report", display_name="report"),
                io.String.Output("segment_info", display_name="segment_info"),
                io.Latent.Output(
                    "av_latent", display_name="av_latent",
                    tooltip=(
                        "Packed AV latent of the executed range with lk_layout metadata. "
                        "Feed through LKH3AVUpscale and back into refine_latent for the HD "
                        "second pass."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, model, segments, steps, shift_video, shift_audio,
                sampler_name=DEFAULT_SAMPLER_NAME, scheduler=DEFAULT_SCHEDULER_NAME,
                seed=0, seed_policy="increment", cfg=1.0, context_noise=0.0,
                release_clip=True, free_between_segments=True,
                refine_latent=None, refine_denoise=0.45, refine_audio="keep"):
        return io.NodeOutput(*run_long_video(
            model, segments,
            steps=steps, shift_video=shift_video, shift_audio=shift_audio,
            sampler_name=sampler_name or DEFAULT_SAMPLER_NAME,
            scheduler=scheduler or DEFAULT_SCHEDULER_NAME,
            seed=seed, seed_policy=seed_policy, cfg=cfg,
            release_clip=release_clip, free_between_segments=free_between_segments,
            context_noise=context_noise,
            refine_latent=refine_latent, refine_denoise=refine_denoise, refine_audio=refine_audio,
        ))

