"""Vision foundation models (VFMs): feature extractors for building the retrieval index.

Supports radiov3 / radiov2.5 / dinov2 / dinov3 / pe (all ViT-L scale, 1024-dim).
The vendored subdirectories dinov3/ and pe/ rely on the sys.path insertion below
to stay importable under their original package names.
"""
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from .factory import VFM_REGISTRY, build_vfm  # noqa: E402,F401
