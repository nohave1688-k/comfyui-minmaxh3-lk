"""Spatial upscale for the packed MiniMax H3 AV latent (HD second pass).

Only the video part is interpolated; the audio stream, its mask half and every
metadata key (``gh_h3_audio_mode``, ``lk_layout``) pass through untouched so the
upscaled latent can re-enter a downstream HD sampler chain (双钟采样 +
BasicGuider + SamplerCustomAdvanced) or the sampler's primitive inputs.  Latent
H/W are rounded to even values because H3's pixel canvas is a multiple of 32
while the latent grid is pixels / 16.
"""

from __future__ import annotations

import logging

import torch

import comfy.nested_tensor
import comfy.utils
from comfy_api.latest import io

from .core import nested_av_parts

UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]


def _round_even(value: float) -> int:
    return max(2, int(round(float(value) / 2.0)) * 2)


def _upscale_video_chunk(chunk, width: int, height: int, upscale_method: str):
    # common_upscale reshapes 5D input internally (extra dims move to batch)
    # and restores the shape, so the packed video latent goes in as-is.
    if (height, width) == tuple(chunk.shape[-2:]):
        return chunk
    return comfy.utils.common_upscale(
        chunk.contiguous(), width, height, upscale_method, "disabled"
    )


def upscale_av_latent(
    av_latent: dict, upscale_method: str, scale_by: float,
    upscale_in_segments: bool = True,
) -> dict:
    video, audio = nested_av_parts(av_latent)
    scale_by = max(1.0, min(4.0, float(scale_by)))
    height = _round_even(int(video.shape[3]) * scale_by)
    width = _round_even(int(video.shape[4]) * scale_by)

    layout_segments = (av_latent.get("lk_layout") or {}).get("segments") or []

    def _chunked_upscale(stream):
        parts = []
        for entry in layout_segments:
            seg_start = int(entry["video_offset"])
            seg_end = seg_start + int(entry["video_tokens"])
            parts.append(
                _upscale_video_chunk(stream[..., seg_start:seg_end, :, :], width, height, upscale_method)
            )
        return torch.cat(parts, dim=2)

    if upscale_in_segments and len(layout_segments) > 1:
        # Per-shot spatial upscaling: one layout segment's video tokens are in
        # the GPU at a time, bounding the VRAM peak on long videos. Spans are
        # temporal and non-overlapping and the interpolation is frame-local, so
        # the assembled result is identical to a whole-tensor upscale.
        upscaled = _chunked_upscale(video)
    elif (height, width) == tuple(video.shape[-2:]):
        upscaled = video
    else:
        upscaled = _upscale_video_chunk(video, width, height, upscale_method)

    out = {key: value for key, value in av_latent.items() if key not in {"samples", "noise_mask"}}
    masks = av_latent.get("noise_mask")
    if getattr(masks, "is_nested", False):
        video_mask, audio_mask = masks.unbind()
        if video_mask.shape[-2:] != (height, width):
            if upscale_in_segments and len(layout_segments) > 1:
                video_mask = _chunked_upscale(video_mask)
            else:
                video_mask = comfy.utils.common_upscale(
                    video_mask, width, height, upscale_method, "disabled"
                )
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
    elif masks is not None:
        logging.debug(
            "LKH3AVUpscale dropped a non-nested noise_mask (%s); "
            "expected the nested AV mask from the long video sampler.",
            type(masks).__name__,
        )
    out["samples"] = comfy.nested_tensor.NestedTensor((upscaled, audio))
    return out


class MiniMaxH3AVUpscaleGH(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LKH3AVUpscale",
            display_name="MiniMax-H3 AV潜空间放大（LK）",
            category="LK/MiniMax H3 Integration",
            description=(
                "Spatially upscales the video part of the packed AV latent for the HD second "
                "pass; audio and layout metadata pass through unchanged. Chain: long video "
                "sampler av_latent -> this node -> 双钟采样 / downstream sampler chain."
            ),
            inputs=[
                io.Latent.Input(
                    "av_latent",
                    tooltip="Packed AV latent from the long video sampler's av_latent output.",
                ),
                io.Combo.Input(
                    "upscale_method",
                    options=UPSCALE_METHODS,
                    default="bilinear",
                    tooltip="Interpolation applied to the video latent (and its mask).",
                ),
                io.Float.Input(
                    "scale_by",
                    default=2.0,
                    min=1.0,
                    max=4.0,
                    step=0.25,
                    round=0.25,
                    tooltip=(
                        "Spatial scale factor. The default 2.0 takes the 0.5MP base canvas to "
                        "the H3 trained maximum (~1920x1088). The second pass costs roughly "
                        "scale^2 more compute per token than the first pass."
                    ),
                ),
                io.Boolean.Input(
                    "upscale_in_segments",
                    default=True,
                    tooltip=(
                        "逐镜放大：一次只在显存里放一个镜头的时间片，长视频显存峰值不随镜头数上涨。"
                        "时间片互不重叠、空间插值逐帧独立，拼回的结果与整条张量放大完全一致。"
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output(
                    "av_latent", display_name="av_latent",
                    tooltip="Upscaled packed AV latent; feed the downstream HD sampler chain (双钟采样 + BasicGuider + SamplerCustomAdvanced).",
                ),
            ],
        )

    @classmethod
    def execute(cls, av_latent, upscale_method, scale_by, upscale_in_segments=True):
        return io.NodeOutput(
            upscale_av_latent(av_latent, upscale_method, scale_by, upscale_in_segments)
        )
