"""Spatial upscale for the packed MiniMax H3 AV latent (HD second pass).

Only the video part is interpolated; the audio stream, its mask half and every
metadata key (``gh_h3_audio_mode``, ``lk_layout``) pass through untouched so the
upscaled latent re-enters ``LKH3LongVideo``'s ``refine_latent`` input for the
low-sigma second pass.  Latent H/W are rounded to even values because H3's
pixel canvas is a multiple of 32 while the latent grid is pixels / 16.
"""

from __future__ import annotations

import logging

import comfy.nested_tensor
import comfy.utils
from comfy_api.latest import io

from .core import nested_av_parts

UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]


def _round_even(value: float) -> int:
    return max(2, int(round(float(value) / 2.0)) * 2)


def upscale_av_latent(av_latent: dict, upscale_method: str, scale_by: float) -> dict:
    video, audio = nested_av_parts(av_latent)
    scale_by = max(1.0, min(4.0, float(scale_by)))
    height = _round_even(int(video.shape[3]) * scale_by)
    width = _round_even(int(video.shape[4]) * scale_by)
    if (height, width) == tuple(video.shape[-2:]):
        upscaled = video
    else:
        # common_upscale reshapes 5D input internally (extra dims move to batch)
        # and restores the shape, so the packed video latent goes in as-is.
        upscaled = comfy.utils.common_upscale(video, width, height, upscale_method, "disabled")

    out = {key: value for key, value in av_latent.items() if key not in {"samples", "noise_mask"}}
    masks = av_latent.get("noise_mask")
    if getattr(masks, "is_nested", False):
        video_mask, audio_mask = masks.unbind()
        if video_mask.shape[-2:] != (height, width):
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
                "sampler av_latent -> this node -> long video sampler refine_latent."
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
            ],
            outputs=[
                io.Latent.Output(
                    "av_latent", display_name="av_latent",
                    tooltip="Upscaled packed AV latent; connect to the long video sampler's refine_latent.",
                ),
            ],
        )

    @classmethod
    def execute(cls, av_latent, upscale_method, scale_by):
        return io.NodeOutput(upscale_av_latent(av_latent, upscale_method, scale_by))
