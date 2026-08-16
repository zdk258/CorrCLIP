"""Instance mask generators: entityseg / eomt / sam / sam2.

The vendored subdirectories CropFormer/, eomt/ and sam2/ rely on the sys.path
insertion below to stay importable under their original package names (sam2's
hydra config resolution also requires `import sam2` to resolve to this directory).
"""
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from .factory import MASK_GENERATOR_CHOICES, build_mask_generator  # noqa: E402,F401
