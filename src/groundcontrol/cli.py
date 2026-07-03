"""Console-script entry points (plan: CLI wrapper scripts).

Entry points live inside the package (a console script cannot live outside
``src/``); ``scripts/*.py`` may later wrap these. Implemented in Increment 1
step 4 (fetch) and Increment 2 (assess).
"""

from __future__ import annotations

import sys


def fetch_control_main(argv=None) -> int:
    """``groundcontrol-fetch`` — AOI -> control GeoParquet/CSV/KML."""
    sys.exit("groundcontrol-fetch: not yet implemented (Increment 1, step 4 — see docs/plan.md)")


def assess_dem_main(argv=None) -> int:
    """``groundcontrol-assess`` — DEM(+AOI) -> fetch -> sample -> stats + figures."""
    sys.exit("groundcontrol-assess: not yet implemented (Increment 2 — see docs/plan.md)")
