"""ComfyUI MiniMax H3 Long Video (LK) — segmented multi-shot audio-video
generation for the MiniMax H3 hybrid Ref2VA model.

Node roster:

* LKH3Composer — one-stop film composer: whole-film prompt, model
  selection, shot collection, and the empty-AV template output.
* LKH3Shot — one shot: duration, prompt, reference media, LoRA rows.
* LKH3ShotGroup — up to 3 shots in one node (shot 1 keeps full reference
  sockets; shots 2-3 are panel-edited and inherit shot 1's refs by default).
  Groups chain through group_in/group_out and feed the composer's
  分镜组链 input, lifting the 8-shot ceiling.
* LKH3SegmentGuider — first pass: owns the segmented loop (per-shot
  conditioning + LoRA, latent continuation) and emits its own SAMPLER/SIGMAS.
* LKH3RefineSegments — second pass: refines the upscaled whole-film latent
  one shot at a time (per-shot HD conditioning, internal refine schedule split
  base/coarse, first-pass audio kept verbatim).
* LKH3AVUpscale — per-shot spatial upscale of the packed AV latent.
* LKH3AVLearnedUpscale — checkpoint-driven dense latent resizer; the 3D
  forward runs per shot (temporal chunking with context heads), so the VRAM
  peak stays O(one shot) no matter how many shots the film has.
* LKH3AVDecode — decodes the packed AV latent back to frames + audio.

The plan socket type is LANXI_H3_SEGMENTS on purpose: plans stay
interchangeable with the predecessor pack's long-video samplers.
"""

from __future__ import annotations

import logging

WEB_DIRECTORY = "./web"

from .gh import nodes as _gh_nodes
from .gh import long_video as _gh_long_video
from .gh import latent_upscale as _gh_latent_upscale
from .gh import learned_upscale as _gh_learned_upscale
from .gh import sampling as _gh_sampling
from .gh import audio_ops as _gh_audio_ops
from .gh import prompt_optimizer as _gh_prompt_optimizer  # noqa: F401  (registers prompt-optimizer routes)

__version__ = "2.0.0"

NODE_CLASS_MAPPINGS = {
    "LKH3Shot": _gh_nodes.MiniMaxH3ShotGH,
    "LKH3ShotGroup": _gh_nodes.MiniMaxH3ShotGroupGH,
    "LKH3AVUpscale": _gh_latent_upscale.MiniMaxH3AVUpscaleGH,
    "LKH3AVLearnedUpscale": _gh_learned_upscale.MiniMaxH3AVLearnedUpscaleGH,
    "LKH3AVDecode": _gh_audio_ops.MiniMaxH3AVDecodeGH,
    "LKH3SegmentGuider": _gh_long_video.MiniMaxH3SegmentGuiderGH,
    "LKH3RefineSegments": _gh_long_video.MiniMaxH3RefineSegmentsGH,
    "LKH3Composer": _gh_nodes.MiniMaxH3ComposerGH,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LKH3Shot": "MiniMaxH3分镜（LK）",
    "LKH3ShotGroup": "MiniMaxH3 分镜组（LK）",
    "LKH3AVUpscale": "MiniMax-H3 AV潜空间放大（LK）",
    "LKH3AVLearnedUpscale": "MiniMax-H3 AV学习型放大（LK）",
    "LKH3AVDecode": "MiniMax-H3 AV解码（LK）",
    "LKH3SegmentGuider": "MiniMax-H3 分段引导（LK）",
    "LKH3RefineSegments": "MiniMax-H3 二跑分段精修（LK）",
    "LKH3Composer": "MiniMaxH3 长视频编排（LK）",
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

logging.basicConfig(level=logging.INFO)
logging.info(
    "[LKH3] LK 包 v%s 已加载: %d 个节点（Composer/Shot/ShotGroup/SegmentGuider/RefineSegments/"
    "AVUpscale/AVLearnedUpscale/AVDecode）",
    __version__, len(NODE_CLASS_MAPPINGS),
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
