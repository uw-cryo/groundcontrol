"""Control-point source providers + the fetch_control dispatcher.

Dispatcher contract (docs/plan.md): each provider is wrapped in try/except and
returns a schema-shaped frame (possibly zero rows); the dispatcher concats
non-empty frames and returns ``(combined_gdf, status)`` where ``status`` maps
source -> {n_rows, error}. On total failure: an empty schema frame.

Transform placement: providers return native-frame, schema-shaped frames; the
dispatcher performs the CRS landing. **Interim MVP landing:** all current
sources are NAD83(2011)-family CONUS products and are landed horizontally in
``EPSG:6318`` with NAVD88 heights in ``height`` — a near-no-op consistent with
the plan's plate-fixed "simple path". Full user-chosen target 3D CRS + epoch
landing (docs/crs_implementation.md §1-§5) is TODO and requesting it raises.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import pandas as pd

from groundcontrol import schema
from groundcontrol.sources import checkpoints_3dep, ngs

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#: source name -> (fetch, parse) callables
PROVIDERS = {
    "3dep": (checkpoints_3dep.fetch, checkpoints_3dep.parse),
    "ngs": (ngs.fetch_nde, ngs.parse_nde),
    "opus": (ngs.fetch_opus, ngs.parse_opus),
}

#: Interim MVP landing frame (see module docstring).
_INTERIM_LANDING_CRS = "EPSG:6318"


def _aoi_bounds(aoi) -> tuple[float, float, float, float]:
    """Accept (minx, miny, maxx, maxy) EPSG:4326, a GeoDataFrame, or a vector-file path."""
    if isinstance(aoi, (tuple, list)) and len(aoi) == 4:
        return tuple(float(v) for v in aoi)
    if isinstance(aoi, str):
        aoi = gpd.read_file(aoi)
    if isinstance(aoi, gpd.GeoDataFrame):
        return tuple(aoi.to_crs(4326).total_bounds)
    raise TypeError(f"unsupported AOI type: {type(aoi)!r}")


def fetch_control(aoi, sources=("3dep", "ngs", "opus"), target_crs=None, target_epoch=None):
    """Fetch control points for an AOI from the requested sources.

    Returns ``(GeoDataFrame, status)``. See the dispatcher contract in the
    module docstring; per-source failures degrade gracefully into ``status``.
    """
    if target_crs is not None or target_epoch is not None:
        raise NotImplementedError(
            "user-chosen target CRS/epoch landing is not implemented yet "
            "(docs/crs_implementation.md §1-§5); current output is the interim "
            f"{_INTERIM_LANDING_CRS} + NAVD88 landing."
        )
    bounds = _aoi_bounds(aoi)
    frames: list[gpd.GeoDataFrame] = []
    status: dict[str, dict] = {}
    for name in sources:
        if name not in PROVIDERS:
            status[name] = {"n_rows": 0, "error": f"unknown source {name!r}"}
            continue
        fetch, parse = PROVIDERS[name]
        try:
            gdf = schema.normalize(parse(fetch(bounds)), source=name)
            status[name] = {"n_rows": len(gdf), "error": None}
            if len(gdf):
                frames.append(gdf)
        except Exception as e:  # degrade gracefully per-source (dispatcher contract)
            logger.exception("source %s failed", name)
            status[name] = {"n_rows": 0, "error": f"{type(e).__name__}: {e}"}
    if not frames:
        return schema.empty(crs=_INTERIM_LANDING_CRS), status
    for f in frames:
        if f.crs is None or f.crs.to_epsg() != 6318:
            raise NotImplementedError(
                f"source produced CRS {f.crs}; only the interim {_INTERIM_LANDING_CRS} "
                "landing is implemented (full target landing TODO)"
            )
    combined = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    schema.validate(combined)
    logger.info("fetch_control: %d points | %s", len(combined),
                {k: v["n_rows"] for k, v in status.items()})
    return combined, status
