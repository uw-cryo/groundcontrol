"""Control-point source providers + the fetch_control dispatcher.

Dispatcher contract (docs/plan.md): each provider is wrapped in try/except and
returns a schema-shaped frame (possibly zero rows); the dispatcher concats
non-empty frames and returns ``(combined_gdf, status)`` where ``status`` maps
source -> {n_rows, error}. On total failure: an empty schema frame.

Transform placement: providers return native-frame, schema-shaped frames; the
dispatcher performs the CRS landing. **Interim MVP landing:** all sources are
landed horizontally in ``EPSG:6318`` unless ``landing_crs`` overrides the
frame (required outside the NAD83 area of use — see :func:`fetch_control`;
a source that cannot land in the requested frame degrades into
``status``). For the NAD83(2011)-family CONUS
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
import pyproj

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


def _short(spec, n=80):
    """Refusal messages echo the input; a WKT input is thousands of chars."""
    s = str(spec)
    return repr(s if len(s) <= n else s[:n] + "...")


def validate_landing_crs(landing_crs) -> str:
    """Gate a fetch landing override; returns the normalized 2D ``EPSG:nnnn``.

    The landing is HORIZONTAL-ONLY: heights stay in the ``height`` column
    with per-row ``vertical_crs`` provenance, so the frame-level CRS must be
    2D geographic like the ``EPSG:6318`` default — a compound OR a
    geographic-3D landing would stamp a vertical axis/datum the rows do not
    carry (re-audit rounds 1-2, 2026-08-29: ``EPSG:6318+5703`` exported
    ellipsoidal heights under a NAVD88 file CRS, and ``EPSG:7912`` — itself
    3D — stamped an ellipsoidal-height axis over per-row orthometric
    provenance). A 3D geographic input is therefore DEMOTED to its 2D
    counterpart (7912 -> 9000, 9989 -> 9990, 6319 -> 6318; logged). Gates,
    applied to the RESOLVED EPSG entry so no spelling bypasses them
    (round 2: ``+datum=WGS84`` resolved to the refused 4326 ensemble):
    no BoundCRS (a declared ``+towgs84`` moved points ~950 m at unknown
    accuracy), no derived geographic (rotated-pole axes are not lon/lat),
    no datum ENSEMBLE (meter-class ambiguity — pass a realization), no
    projected/geocentric, and the frame must resolve to an EPSG code (an
    unidentifiable frame has no place in provenance). Idempotent on its
    own output.
    """
    lc = pyproj.CRS.from_user_input(landing_crs)
    if lc.is_derived and lc.is_geographic:
        raise ValueError(
            f"landing_crs {_short(landing_crs)} ({lc.name}) is a DERIVED "
            "geographic CRS (e.g. rotated pole): its axes are not lon/lat. "
            "Pass a plain geographic realization.")
    if lc.is_bound:
        raise ValueError(
            f"landing_crs {_short(landing_crs)} ({lc.name}) is a BoundCRS: the declared "
            "transformation (+towgs84/TOWGS84) would move every point at "
            "unstated accuracy. Pass the bare frame's EPSG code.")
    code = lc.to_epsg()
    if code is None:
        raise ValueError(
            f"landing_crs {_short(landing_crs)} ({lc.name}) does not resolve to an EPSG "
            "entry: an unidentifiable landing frame has no place in the transform "
            "provenance. Pass the frame's EPSG code (EPSG:7912 ITRF2014, "
            "EPSG:9989 ITRF2020, EPSG:6318 NAD83(2011), ...).")
    lc = pyproj.CRS.from_epsg(code)  # gate the RESOLVED entry, not the spelling
    if lc.is_compound or not lc.is_geographic:
        raise ValueError(
            f"landing_crs {_short(landing_crs)} ({lc.name}) must be a plain geographic "
            "CRS: the fetch landing is horizontal-only — heights stay in the "
            "height column with per-row vertical_crs provenance, and a compound/"
            "projected/geocentric/derived landing would stamp a meaning the data "
            "do not have. Compose vertical datums in the assess step "
            "(--target-crs).")
    # scoped to WGS84's ensemble, matching geodesy.is_wgs84_ensemble
    # (round-2 audit: the broad datum.type_name test here refused the
    # EPSG:4937 landing the CLI auto-derives for an ETRS89 target, with
    # a message about a --landing-crs the user never passed — ETRS89's
    # ~0.1 m intra-ensemble ambiguity is fine to land in)
    from groundcontrol.geodesy import is_wgs84_ensemble
    if is_wgs84_ensemble(lc):
        raise ValueError(
            f"landing_crs {_short(landing_crs)} ({lc.name}) is a datum ENSEMBLE "
            "(meter-class ambiguity by definition; the transform provenance "
            "would honestly record acc ~2-5 m). Pass a specific realization: "
            "EPSG:7912 (ITRF2014), EPSG:9989 (ITRF2020), EPSG:6318 "
            "(NAD83(2011)), ...")
    flat = lc.to_2d()
    code2d = flat.to_epsg()
    if code2d is None:  # pragma: no cover - defensive (all 241 EPSG geographic
        # 3D entries demote to a registered 2D code; round-3 audit)
        raise ValueError(
            f"landing_crs {_short(landing_crs)} ({lc.name}) is 3D and its 2D "
            "counterpart has no EPSG entry; pass the 2D geographic code directly.")
    if code2d != code:
        logger.info("landing_crs %s is geographic 3D; landing at its 2D "
                    "counterpart EPSG:%d (the landing is horizontal-only)",
                    landing_crs, code2d)
    return f"EPSG:{code2d}"


def _aoi_bounds_and_poly(aoi):
    """Any AOI form -> ``(bounds_4326, polygon_or_None)``; see
    :func:`groundcontrol.aoi.resolve_aoi` (bbox, vector file, raster
    footprint, GeoDataFrame). Sources fetch by bbox; the dispatcher clips
    the combined result to the polygon when one was given."""
    from groundcontrol.aoi import resolve_aoi
    return resolve_aoi(aoi)


def fetch_control(aoi, sources=("3dep", "ngs", "opus", "ngl", "faa"),
                  target_crs=None, target_epoch=None, landing_crs=None):
    """Fetch control points for an AOI from the requested sources.

    Returns ``(GeoDataFrame, status)``. See the dispatcher contract in the
    module docstring; per-source failures degrade gracefully into ``status``.

    ``landing_crs`` overrides the interim HORIZONTAL landing frame (default
    ``EPSG:6318``, the CONUS contract). Required for non-CONUS AOIs: the
    ITRF->NAD83(2011) operations carry a North America area of use, so the
    default landing correctly fail-louds outside it (rasuwa, 2026-08-29) —
    e.g. Nepal NGL control lands with ``landing_crs="EPSG:7912"``, which
    lands at the 2D counterpart ``EPSG:9000`` via the registered null
    geographic offset (accuracy 0.0 m, coordinates unchanged; as a
    dynamic-frame source every row must carry ``coord_epoch``). Must be a
    geographic CRS: the
    landing is horizontal-only, heights stay in the ``height`` column with
    their per-row ``vertical_crs`` provenance. A plate-fixed source cannot
    land into a dynamic frame (needs the D6 target-epoch semantics) — that
    source degrades into ``status`` with the error. This is NOT the full
    user-chosen 3D target landing (``target_crs``/``target_epoch``, §1-§5),
    which still raises below.
    """
    if target_crs is not None or target_epoch is not None:
        raise NotImplementedError(
            "user-chosen target CRS/epoch landing is not implemented yet "
            "(docs/crs_implementation.md §1-§5); current output is the interim "
            f"{_INTERIM_LANDING_CRS} + NAVD88 landing."
        )
    landing = _INTERIM_LANDING_CRS
    if landing_crs is not None:
        landing = validate_landing_crs(landing_crs)
    bounds, poly = _aoi_bounds_and_poly(aoi)
    frames: list[gpd.GeoDataFrame] = []
    status: dict[str, dict] = {}
    # network fetches run CONCURRENTLY (owner 2026-08-30: five serial
    # providers were pure wall-clock; the slow parts are downloads).
    # Parse + landing stay SERIAL in this thread: the landing path shares
    # cached pyproj transformers, and provider fetches are the only part
    # that is trivially independent.
    from concurrent.futures import ThreadPoolExecutor
    known = [n for n in sources if n in PROVIDERS]
    raw: dict = {}
    fetch_err: dict = {}
    if known:
        import time as _time
        from concurrent.futures import as_completed
        t0 = _time.monotonic()
        with ThreadPoolExecutor(max_workers=len(known)) as pool:
            futs = {}
            for name in known:
                logger.info("querying %s ...", name)
                futs[pool.submit(PROVIDERS[name][0], bounds)] = name
            for fut in as_completed(futs):
                name = futs[fut]
                try:
                    raw[name] = fut.result()
                    logger.info("%s fetched (%.1f s)", name,
                                _time.monotonic() - t0)
                except Exception as e:  # logged once, in the main handler
                    fetch_err[name] = e
                    logger.info("%s failed after %.1f s (details below)",
                                name, _time.monotonic() - t0)
    for name in sources:
        if name not in PROVIDERS:
            status[name] = {"n_rows": 0, "error": f"unknown source {name!r}"}
            continue
        _, parse = PROVIDERS[name]
        try:
            if name in fetch_err:
                raise fetch_err[name]
            gdf = parse(raw[name])
            # per-row quarantine report (e.g. #21 unmapped NGS realizations)
            # — read BEFORE landing/normalize (pandas ops may drop .attrs)
            skipped = dict(getattr(gdf, "attrs", {}).get("skipped") or {})
            # B7: per-datum horizontal landing into the landing frame (subset
            # AOIs, fail-loud on missing grids/unknown realizations).
            gdf = crs.land_horizontal(gdf, target=landing)
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
        return schema.empty(crs=landing), status
    landing_obj = pyproj.CRS.from_user_input(landing)
    for f in frames:
        if f.crs is None or not pyproj.CRS(f.crs).equals(landing_obj):
            raise NotImplementedError(
                f"source produced CRS {f.crs}; expected the {landing} landing "
                "(full target landing TODO)"
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
