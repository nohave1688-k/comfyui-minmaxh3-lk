"""ComfyUI MiniMax H3 Long Video — the long-video focused successor pack.

Slimmed-down rebuild of ComfyUI-MiniMaxH3-lanxi (the GH integration + RefMod
merge) that keeps only the long-video core:

* ``LongvideoH3Studio`` — the segment panel / conditioning builder (1-8
  segments, per-segment prompts, first/last frames, picture/video/audio
  references, RefMod rows, per-segment LoRA slots, and connectable media
  sockets: drag upstream links onto the panel buttons).
* ``LongvideoH3LongVideo`` — the segmented sampler: per-segment LoRA, latent
  continuation across seams, dual-clock sampling, joint AV decode.

Removed from the lanxi pack during the rebuild: the single-mode pipeline
(adapter + standalone dual-clock sampler + AV decode), the second-pass tiled
sampler, the latent upscaler, the V1 dual-track node registration and the 16
never-registered RefMod graph nodes (the RefMod *library* remains as the
panel's backing store under ``models/refmods/``).

Registered purely through ``NODE_CLASS_MAPPINGS``; the two classes are V3
``io.ComfyNode`` implementations that ComfyUI executes natively.
"""

from __future__ import annotations

WEB_DIRECTORY = "./web"

from .gh import nodes as _gh_nodes
from .gh import long_video as _gh_long_video
from .gh import prompt_optimizer as _gh_prompt_optimizer  # noqa: F401  (registers prompt-optimizer routes)

NODE_CLASS_MAPPINGS = {
    "LongvideoH3Studio": _gh_nodes.MiniMaxH3IntegrationGH,
    "LongvideoH3LongVideo": _gh_long_video.MiniMaxH3LongVideoRunnerGH,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LongvideoH3Studio": "MiniMaxH3长视频操作台（longvideo)",
    "LongvideoH3LongVideo": "MiniMax-H3 长视频分段采样（longvideo）",
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

        @instance.routes.get("/longvideo/minimax_h3/loras")
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
