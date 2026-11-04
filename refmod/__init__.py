"""
RefMod (no-training reference adapters for MiniMax H3) - merged subpackage.

The original ComfyUI-MiniMaxH3Mod pack.  Nodes are described in ``nodes.py``;
storage lives in ``models/refmods/``.  Importing this subpackage also registers
the ``/refmods/library`` metadata endpoint.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
