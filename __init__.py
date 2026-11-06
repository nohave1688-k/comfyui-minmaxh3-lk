"""ComfyUI MiniMax H3 Long Video — the long-video focused successor pack.

Slimmed-down rebuild of ComfyUI-MiniMaxH3-lanxi (the GH integration + RefMod
merge) that keeps only the long-video core:

* ``LKH3Studio`` — the segment panel / conditioning builder (1-8
  segments, per-segment prompts, first/last frames, picture/video/audio
  references, RefMod rows, per-segment LoRA slots, and connectable media
  sockets: drag upstream links onto the panel buttons).
* ``LKH3LongVideo`` — the segmented sampler: per-segment LoRA, latent
  continuation across seams, dual-clock sampling.  It outputs the whole-video
  packed AV latent plus the positive conditioning; decoding and HD second
  passes are composed downstream from standard sampling primitives.
* ``LKH3AVUpscale`` — spatially upscales the video part of the packed AV
  latent (audio/layout pass through, per-shot chunks to bound VRAM) for the
  HD second-pass chain.
* ``LKH3DualClockSampler`` — the T8-style sampler contract: patches the model
  with the AV dual-clock / FLOW_AV model sampling and emits MODEL / SAMPLER /
  SIGMAS for comfy-core BasicGuider + SamplerCustomAdvanced chains.
* ``LKH3SegmentGuider`` — the T8-style GUIDER for the first pass: owns the
  segmented loop (per-shot conditioning + LoRA, latent continuation) while
  sampler/sigmas/seed/cfg come from the standard primitives.
* ``LKH3RefineSegments`` — the segmented HD second pass: refines the
  upscaled whole-film latent one shot at a time (per-shot HD conditioning,
  external refine schedule, first-pass audio kept verbatim).
* ``LKH3ReferenceFocus`` — reference-photo preprocessing (YuNet face crops
  into an identity sheet) for stronger identity adherence.
* ``LKH3Composer`` — one-stop film composer (whole-film prompt + model
  selection + shot collection + empty-AV template output).
* ``LKH3AVDecode`` — decodes the packed AV latent back to frames + audio.

Removed from the lanxi pack during the rebuild: the single-mode pipeline
(adapter + standalone dual-clock sampler + AV decode), the V1 dual-track node
registration and the 16 never-registered RefMod graph nodes (the RefMod
*library* remains as the panel's backing store under ``models/refmods/``).

Registered purely through ``NODE_CLASS_MAPPINGS``; the classes are V3
``io.ComfyNode`` implementations that ComfyUI executes natively.
"""

from __future__ import annotations

WEB_DIRECTORY = "./web"

from .gh import nodes as _gh_nodes
from .gh import long_video as _gh_long_video
from .gh import latent_upscale as _gh_latent_upscale
from .gh import sampling as _gh_sampling
from .gh import audio_ops as _gh_audio_ops
from .gh import reference_tools as _gh_reference_tools
from .gh import prompt_optimizer as _gh_prompt_optimizer  # noqa: F401  (registers prompt-optimizer routes)

NODE_CLASS_MAPPINGS = {
    "LKH3Studio": _gh_nodes.MiniMaxH3IntegrationGH,
    "LKH3Shot": _gh_nodes.MiniMaxH3ShotGH,
    "LKH3SamplerSettings": _gh_long_video.MiniMaxH3SamplerSettingsGH,
    "LKH3LongVideo": _gh_long_video.MiniMaxH3LongVideoRunnerGH,
    "LKH3AVUpscale": _gh_latent_upscale.MiniMaxH3AVUpscaleGH,
    "LKH3DualClockSampler": _gh_sampling.MiniMaxH3DualClockSamplerGH,
    "LKH3AVDecode": _gh_audio_ops.MiniMaxH3AVDecodeGH,
    "LKH3SegmentGuider": _gh_long_video.MiniMaxH3SegmentGuiderGH,
    "LKH3RefineSegments": _gh_long_video.MiniMaxH3RefineSegmentsGH,
    "LKH3ReferenceFocus": _gh_reference_tools.MiniMaxH3ReferenceFocusGH,
    "LKH3Composer": _gh_nodes.MiniMaxH3ComposerGH,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LKH3Studio": "MiniMaxH3长视频操作台（LK)",
    # REMOVED: LKH3{gone}
    # "LKH3{gone}"模型设置（LK)",
    "LKH3Shot": "MiniMaxH3分镜（LK）",
    # REMOVED: LKH3{gone}
    # "LKH3{gone}"分镜合成（LK）",
    "LKH3SamplerSettings": "MiniMaxH3采样设置（LK）",
    "LKH3LongVideo": "MiniMax-H3 长视频分段采样（LK）",
    "LKH3AVUpscale": "MiniMax-H3 AV潜空间放大（LK）",
    "LKH3DualClockSampler": "MiniMax-H3 双钟采样（LK）",
    "LKH3AVDecode": "MiniMax-H3 AV解码（LK）",
    "LKH3SegmentGuider": "MiniMax-H3 分段引导（LK）",
    "LKH3RefineSegments": "MiniMax-H3 二跑分段精修（LK）",
    "LKH3ReferenceFocus": "MiniMax-H3 参考聚焦（LK）",
    "LKH3Composer": "MiniMaxH3 长视频编排（LK）",
    # REMOVED: LKH3{gone}
    # "LKH3{gone}"全片提示词设置（LK）",
}

# LoRA list for the per-segment LoRA slots.  ``folder_paths`` is resolved at
# request time so extra_model_paths.yaml changes are picked up on refresh.
def _register_loras_route() -> None:
    try:
        from aiohttp import web
        import folder_paths
        from server import PromptServer

        instance = getattr(PromptServer, "instance", None)
        if instance is None:
            return

        @instance.routes.get("/lk/minimax_h3/loras")
        async def _list_loras(_request):
            try:
                names = list(folder_paths.get_filename_list("loras"))
            except Exception:
                names = []
            return web.json_response({"loras": names})
    except Exception:
        # API-only / CLI contexts: the panel simply falls back to its cache.
        pass


_register_loras_route()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
