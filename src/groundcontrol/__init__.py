"""groundcontrol: fetch ground control points for any AOI and assess DEM accuracy,
with rigorous 3D CRS/datum/epoch handling.

See docs/plan.md and docs/crs_implementation.md in the repository for the design.
"""

import warnings
from importlib.metadata import PackageNotFoundError, version as _version

# Single-sourced from the installed distribution metadata (pyproject `version`),
# so this can never drift from the released tag again (v0.1.1 shipped a stale
# literal). io.py stamps this into output provenance, so a wrong value is a
# data bug — fail loud and fall back to a string that is obviously not a
# version rather than a plausible one.
try:
    __version__ = _version("groundcontrol")
except PackageNotFoundError:  # source tree with no install
    warnings.warn(
        "groundcontrol is not installed (no distribution metadata found); "
        "__version__ set to 'unknown' and output provenance will record it. "
        "Install with `pip install -e .` to fix.",
        stacklevel=2,
    )
    __version__ = "unknown"
