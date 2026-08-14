"""Control-point source providers + the fetch_control dispatcher.

Dispatcher contract (docs/plan.md): each provider is wrapped in try/except and
returns a schema-shaped frame (possibly zero rows); the dispatcher concats
non-empty frames and returns ``(combined_gdf, status)`` where ``status`` maps
source -> {n_rows, error}. On total failure: an empty schema frame.

Transform placement: providers return native-frame, schema-shaped frames; the
dispatcher performs the CRS landing. **Interim MVP landing:** all sources are
landed horizontally in ``EPSG:6318``. For the NAD83(2011)-family CONUS
products this is a near-no-op (plate-fixed "simple path") with NAVD88 heights
in ``height``. The NGL GNSS source is dynamic-frame (ITRF-aliased):
``crs.land_horizontal`` evaluates its time-dependent Helmert at each row's
``coord_epoch`` (the provisional D6 tt rule, docs/crs_implementation.md §1)
and its ellipsoidal heights ride through with honest provenance labels. Full
user-chosen target 3D CRS + epoch landing (§1-§5) is TODO and requesting it
raises.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import pandas as pd

from groundcontrol import crs, schema
from groundcontrol.sources import checkpoints_3dep, faa, ngl, ngs

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#: source name -> (fetch, parse) callables
PROVIDERS = {
    "3dep": (checkpoints_3dep.fetch, checkpoints_3dep.parse),
    "ngs": (ngs.fetch_nde, ngs.parse_nde),
    "opus": (ngs.fetch_opus, ngs.parse_opus),
    "ngl": (ngl.fetch, ngl.parse),
    "faa": (faa.fetch, faa.parse),
}

#: Interim MVP landing frame (see module docstring).
_INTERIM_LANDING_CRS = "EPSG:6318"


def _aoi_bounds_and_poly(aoi):
    """Accept (minx, miny, maxx, maxy) EPSG:4326, a GeoDataFrame, or a vector-file path.

    Returns ``(bounds_4326, polygon_or_None)`` — sources fetch by bbox; the
    dispatcher clips the combined result to the polygon when one was given.
    """
    if isinstance(aoi, (tuple, list)) and len(aoi) == 4:
        return tuple(float(v) for v in aoi), None
    if isinstance(aoi, (str, bytes)) or hasattr(aoi, "__fspath__"):
        aoi = gpd.read_file(aoi)
    if isinstance(aoi, gpd.GeoDataFrame):
        aoi4326 = aoi.to_crs(4326)
        return tuple(aoi4326.total_bounds), aoi4326.union_all()
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
    bounds, poly = _aoi_bounds_and_poly(aoi)
    frames: list[gpd.GeoDataFrame] = []
    status: dict[str, dict] = {}
    for name in sources:
        if name not in PROVIDERS:
            status[name] = {"n_rows": 0, "error": f"unknown source {name!r}"}
            continue
        fetch, parse = PROVIDERS[name]
        try:
            gdf = parse(fetch(bounds))
            # per-row quarantine report (e.g. #21 unmapped NGS realizations)
            # — read BEFORE landing/normalize (pandas ops may drop .attrs)
            skipped = dict(getattr(gdf, "attrs", {}).get("skipped") or {})
            # B7: per-datum horizontal landing into the interim frame (subset
            # AOIs, fail-loud on missing grids/unknown realizations).
            gdf = crs.land_horizontal(gdf, target=_INTERIM_LANDING_CRS)
            gdf = schema.normalize(gdf, source=name)
            status[name] = {"n_rows": len(gdf), "error": None}
            if skipped:
                status[name]["n_skipped"] = skipped.get("n")
                status[name]["skip_reasons"] = skipped.get("reasons")
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
    if poly is not None:
        # polygon AOI: keep only points inside (sources fetched by bbox).
        # NAD83(2011) vs WGS84 polygon frames differ at the ~1 m level —
        # negligible for AOI membership at these scales.
        n0 = len(combined)
        combined = combined[combined.geometry.within(poly)].reset_index(drop=True)
        logger.info("polygon clip: %d -> %d points", n0, len(combined))
    schema.validate(combined)
    logger.info("fetch_control: %d points | %s", len(combined),
                {k: v["n_rows"] for k, v in status.items()})
    return combined, status
