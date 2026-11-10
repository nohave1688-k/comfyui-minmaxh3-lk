"""Goohai MiniMax H3 integration modules (merged into the unified pack).

The original pack exposed these nodes through ``comfy_entrypoint``.  The
unified pack registers the very same ``io.ComfyNode`` classes through
``NODE_CLASS_MAPPINGS`` so that ComfyUI can load the nodes (which use
the V1 node API) from one single extension.
"""

from . import nodes  # noqa: F401  (registers the V3 node classes)
from . import prompt_optimizer  # noqa: F401  (registers the prompt-optimizer routes)

__all__ = ["nodes", "prompt_optimizer"]
