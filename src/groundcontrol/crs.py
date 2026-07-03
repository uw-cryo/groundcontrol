"""AOI-aware datum + epoch transforms (thin pyproj wrapper).

Implements the verified directives in docs/crs_implementation.md — the
fail-loud transformer pattern (§2), decimal-year helpers (plan B9), and the
two-stage epoch model (§1). Recipes follow uw-cryo/3D_CRS_Transformation_Resources;
theory is cited there, not re-documented here.

TODO(D6): the tt rule (which epoch feeds the 4D time argument) is PROVISIONAL —
see docs/crs_implementation.md §1. Validate numerically against the CRS fixtures
(CORS coord_14/coord_20 pairs, closure matrix) before trusting it.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import pyproj
from pyproj.transformer import TransformerGroup

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class NoTransformPathError(RuntimeError):
    """No usable coordinate operation exists (e.g. required grid missing).

    Raised instead of pyproj's bare ``IndexError`` when a ``TransformerGroup``
    has no available transformers under ``allow_ballpark=False`` (plan B6).
    Carries the diagnostic payload that is also logged (crs_implementation §7.5).
    """

    def __init__(self, message: str, diagnostics: dict | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


# ---------------------------------------------------------------------------
# Decimal-year helpers (plan Appendix B9 — hardened spec)
# ---------------------------------------------------------------------------

def _to_utc_naive(values) -> pd.Series:
    """Parse to datetimes; tz-aware -> UTC, naive assumed UTC; NaT preserved."""
    s = pd.Series(values)
    dt = pd.to_datetime(s, utc=True, errors="coerce")
    return dt.dt.tz_localize(None)


def _year_bounds(year: pd.Series) -> tuple[pd.Series, pd.Series]:
    y = year.astype("Int64")
    start = pd.to_datetime(y.astype("string"), format="%Y", errors="coerce")
    nxt = pd.to_datetime((y + 1).astype("string"), format="%Y", errors="coerce")
    return start, nxt


def decyear(dt):
    """Datetime -> decimal year. Leap-exact and sub-daily precise.

    Polymorphic: str / datetime / pd.Timestamp scalars, or Series /
    DatetimeIndex / array-likes (returns a float Series aligned to the input
    index where one exists). tz-aware input is converted to UTC; naive input
    is assumed UTC. NaT -> NaN.

    Formula: ``year + (t - year_start) / (next_year_start - year_start)`` —
    2010-01-01T00:00 -> 2010.0 exactly; 2010-07-02T12:00 -> 2010.5 exactly
    (non-leap); 2020-07-02T00:00 -> 2020.5 exactly (leap).
    """
    scalar = not isinstance(dt, (pd.Series, pd.DatetimeIndex, np.ndarray, list, tuple))
    index = dt.index if isinstance(dt, pd.Series) else None
    t = _to_utc_naive([dt] if scalar else dt)
    year = t.dt.year
    start, nxt = _year_bounds(year)
    frac = (t - start) / (nxt - start)
    out = year.astype("float64") + frac
    if scalar:
        return float(out.iloc[0])
    return pd.Series(out.to_numpy(dtype="float64"), index=index, name="decyear")


def decyear_inv(dy):
    """Decimal year -> pd.Timestamp (UTC-naive). Inverse of :func:`decyear`.

    Round-trip accuracy: better than 1 second. NaN -> NaT.
    """
    scalar = np.isscalar(dy)
    s = pd.Series([dy] if scalar else dy, dtype="float64")
    year = np.floor(s)
    start, nxt = _year_bounds(year)
    out = start + (s - year) * (nxt - start)
    if scalar:
        return out.iloc[0]
    index = dy.index if isinstance(dy, pd.Series) else None
    return pd.Series(out.to_numpy(), index=index, name="datetime")


# ---------------------------------------------------------------------------
# Fail-loud transformer selection (crs_implementation.md §2)
# ---------------------------------------------------------------------------

def get_transformer(source_crs, target_crs, aoi_bounds_4326=None) -> pyproj.Transformer:
    """Select the best available coordinate operation, loudly.

    Parameters
    ----------
    source_crs, target_crs : anything ``pyproj.CRS`` accepts (EPSG string,
        WKT2, ``pyproj.CRS``). Compound sources like ``"EPSG:6318+5703"`` are
        supported.
    aoi_bounds_4326 : optional (west, south, east, north) in **degrees**
        (EPSG:4326). ``pyproj.aoi.AreaOfInterest`` takes degrees — never pass
        projected bounds (crs_implementation §2). Per plan B7: use the bounds
        of the datum *subset* being transformed.

    Returns a single-operation transformer from the ``TransformerGroup`` (its
    ``.definition``/``.accuracy`` are real — a multi-candidate ``from_crs``
    transformer returns placeholders; crs_implementation §7.1).

    Raises :class:`NoTransformPathError` when nothing is available under
    ``allow_ballpark=False`` (never a silent ballpark/degraded result), and
    warns when better operations exist but are unavailable (missing grids).
    """
    aoi = pyproj.aoi.AreaOfInterest(*aoi_bounds_4326) if aoi_bounds_4326 is not None else None
    tg = TransformerGroup(
        source_crs, target_crs,
        area_of_interest=aoi, allow_ballpark=False, always_xy=True,
    )
    unavailable = [
        {"name": op.name, "grids": [g.short_name for g in op.grids if not g.available]}
        for op in tg.unavailable_operations
    ]
    if not tg.transformers:
        diag = {
            "source_crs": str(source_crs),
            "target_crs": str(target_crs),
            "aoi_bounds_4326": aoi_bounds_4326,
            "unavailable_operations": unavailable,
        }
        msg = (
            f"No usable transform {source_crs!r} -> {target_crs!r} with allow_ballpark=False. "
            f"{len(unavailable)} operation(s) unavailable (missing grids: "
            f"{sorted({g for u in unavailable for g in u['grids']})}). "
            "Remedies: TransformerGroup(...).download_grids(), `pyproj sync`, or PROJ_NETWORK=ON."
        )
        logger.error(msg, extra={"diagnostics": diag})
        raise NoTransformPathError(msg, diag)
    if not tg.best_available:
        missing = sorted({g for u in unavailable for g in u["grids"]})
        warnings.warn(
            f"Best transform {source_crs!r} -> {target_crs!r} unavailable (missing grids: "
            f"{missing}); using an available lower-accuracy operation. "
            "Provision grids for full accuracy.",
            stacklevel=2,
        )
    chosen = tg.transformers[0]
    logger.info(
        "transform selected: %s -> %s | %s | accuracy %s m | candidates %d, unavailable %d",
        source_crs, target_crs, chosen.description, chosen.accuracy,
        len(tg.transformers), len(unavailable),
    )
    return chosen


# ---------------------------------------------------------------------------
# NGS realization mapping + per-datum horizontal landing (plan B7)
# ---------------------------------------------------------------------------

def transform_points(gdf, target_crs, tt, height_col: str = "height",
                     source_crs=None, aoi_bounds_4326=None):
    """Transform 2D-geometry + height-column points into a target 3D frame.

    The packaged form of the canonical pattern in docs/quickstart.md §2: ONE
    3D/4D transformer applied to (x, y, h, tt) arrays — heights are never
    routed through ``to_crs`` (crs_implementation §5).

    Parameters
    ----------
    gdf : GeoDataFrame with 2D point geometry and a numeric ``height_col``.
    target_crs : the DEM/working frame — EPSG string, WKT2, or ``pyproj.CRS``
        (3D or compound; a declared frame overriding a lying embedded WKT is
        the caller's responsibility — see quickstart).
    tt : scalar decimal year or per-row array/Series — the 4D time coordinate.
        REQUIRED and explicit: per the provisional D6 rule (TODO(D6)), pass the
        target/product epoch for plate-fixed sources going to a dynamic frame,
        or per-row ``coord_epoch`` for dynamic-frame sources.
    source_crs : source 3D/compound CRS. Default: ``gdf.crs`` if it is already
        compound/3D; else built from ``gdf.crs`` + a single uniform non-null
        ``vertical_crs`` column value. Anything ambiguous raises (fail-loud —
        never guess a vertical datum).
    aoi_bounds_4326 : optional degrees bounds for operation selection;
        defaults to the gdf's bounds in EPSG:4326.

    Returns a copy: geometry in ``target_crs`` (2D points), ``height_col``
    replaced by the transformed height, and ``transform_id`` stamped when the
    column exists. Other provenance columns (``native_*``) are untouched.
    """
    import geopandas as gpd

    if source_crs is None:
        if gdf.crs is None:
            raise ValueError("gdf has no CRS and source_crs was not given")
        if gdf.crs.is_compound or len(gdf.crs.axis_info) == 3:
            source_crs = gdf.crs
        elif "vertical_crs" in gdf.columns:
            vcodes = gdf["vertical_crs"].dropna().unique()
            if len(vcodes) != 1 or gdf["vertical_crs"].isna().any():
                raise ValueError(
                    f"vertical_crs is not a single uniform value ({list(vcodes)!r}); "
                    "pass source_crs explicitly — the vertical datum is never guessed")
            auth = gdf.crs.to_authority()
            source_crs = f"{auth[0]}:{auth[1]}+{str(vcodes[0]).split(':')[-1]}"
        else:
            raise ValueError("cannot infer the source vertical datum; pass source_crs")
    if aoi_bounds_4326 is None:
        aoi_bounds_4326 = tuple(gdf.geometry.to_crs(4326).total_bounds)  # AOI use only
    t = get_transformer(source_crs, target_crs, aoi_bounds_4326=aoi_bounds_4326)
    h = pd.to_numeric(gdf[height_col], errors="raise").to_numpy(dtype="float64")
    tt_arr = np.full(len(gdf), float(tt)) if np.isscalar(tt) else np.asarray(tt, dtype="float64")
    if np.isnan(tt_arr).any():
        raise ValueError("tt contains NaN — every point needs an epoch (D6 rule)")
    x, y, h2, _ = t.transform(gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy(),
                              h, tt_arr, errcheck=True)
    out = gdf.copy()
    out["geometry"] = gpd.points_from_xy(x, y)
    out = out.set_crs(target_crs, allow_override=True)
    out[height_col] = h2
    if "transform_id" in out.columns:
        out["transform_id"] = (
            f"transform_points:{source_crs}->{target_crs}|acc={t.accuracy}m")
    logger.info("transform_points: %d pts %s -> %s | %s | accuracy %s m",
                len(out), source_crs, target_crs, t.description, t.accuracy)
    return out


#: NGS datasheet ``posDatum`` / OPUS ``refFrame`` strings -> EPSG geographic 2D CRS.
#: State HPGN/HARN readjustments are labelled by year (1991-1999) on datasheets;
#: they are all NAD83(HARN) EPSG:4152 for transformation purposes (NADCON5).
_NGS_DATUM_RULES = (
    ("2011", "EPSG:6318"),
    ("NSRS2007", "EPSG:4759"),
    ("2007", "EPSG:4759"),
    ("CORS96", "EPSG:6783"),
    ("FBN", "EPSG:8860"),
    ("HARN", "EPSG:4152"),
    ("1991", "EPSG:4152"), ("1992", "EPSG:4152"), ("1993", "EPSG:4152"),
    ("1994", "EPSG:4152"), ("1995", "EPSG:4152"), ("1996", "EPSG:4152"),
    ("1997", "EPSG:4152"), ("1998", "EPSG:4152"), ("1999", "EPSG:4152"),
    ("1986", "EPSG:4269"),
)


def ngs_datum_to_epsg(datum: str) -> str:
    """Map an NGS realization string (e.g. ``'NAD 83(1992)'``) to its EPSG CRS.

    Raises ``ValueError`` listing the supported realizations for anything
    unrecognized (fail-loud: an unmapped realization must never be silently
    carried as if it were NAD83(2011)).
    """
    s = (datum or "").upper().replace(" ", "").replace("_", "")
    if s.startswith("NAD83") or s.startswith("NAD_83"):
        for token, epsg in _NGS_DATUM_RULES:
            if token in s:
                return epsg
    raise ValueError(
        f"unrecognized NGS realization {datum!r}; supported: NAD83 "
        "(1986/1991-1999 state HARN/HARN/FBN/CORS96/NSRS2007/2007/2011)"
    )


def is_dynamic_frame(crs_like) -> bool:
    """True when the CRS's datum is a *dynamic* reference frame (ITRF/IGS...).

    Dynamic-frame coordinates are only meaningful with a coordinate epoch —
    time-dependent transforms must be evaluated at each point's
    ``coord_epoch`` (crs_implementation §1).
    """
    datum = pyproj.CRS(crs_like).datum
    return datum is not None and "dynamic" in (datum.type_name or "").lower()


def land_horizontal(gdf, target: str = "EPSG:6318", datum_col: str = "horizontal_crs"):
    """Land a mixed-realization frame horizontally into one target CRS (plan B7).

    Groups rows by ``datum_col`` (per-row EPSG strings) and transforms each
    subset with its own transformer built from the **subset's** bounds (B7a).

    **Epoch handling — TODO(D6), provisional tt rule (crs_implementation §1):**
    when a subset's source CRS is a *dynamic* frame (ITRF/IGS — e.g. NGL's
    aliased ``EPSG:7912``/``EPSG:9989``), the per-row ``coord_epoch`` is
    passed as the 4th transform argument ``tt`` so time-dependent Helmerts
    (ITRF2014->NAD83(2011), EPSG:8970) are evaluated at each point's own
    coordinate epoch. An omitted ``tt`` silently evaluates at the operation's
    fixed ``t_epoch`` (2010.0) — plate-velocity x delta-t wrong (cm-dm) for
    2015-2025 coordinates (verified: ~1.8 cm/yr horizontal for CONUS). A
    dynamic-frame subset without a usable ``coord_epoch`` therefore raises —
    silently landing it at 2010.0 is exactly the failure mode this library
    exists to prevent. Static operations ignore ``tt``, so plate-fixed
    subsets keep the 2D path.

    Heights are untouched — valid for the plate-fixed sources ONLY because
    they carry *published orthometric* heights (vertical-datum quantities
    that no horizontal realization transform operates on). **Ellipsoidal
    heights are NOT invariant** — NADCON5 carries explicit eht-shift grids
    (empirically: NCAT HARN->2011 shifts eht by −0.072 m at Casa Grande) — so
    any future ellipsoidal-height path must transform h here as part of
    landing (B7b), and must never recompute H = h − N across a realization
    change. Dynamic-frame sources (NGL) carry *ellipsoidal* heights: those
    ride through as native-frame values with honest provenance labels
    (``height_datum='ellipsoidal'`` + frame code in ``vertical_crs``) until
    the full 3D landing (crs_implementation §5) lands.

    Sets a per-subset ``transform_id`` and the single target CRS on return.
    Original per-row provenance (``horizontal_crs``/``native_*``) is preserved.
    """
    import geopandas as gpd  # local import to keep crs.py light for non-geo use

    out = gdf.copy()
    if not len(out):
        return out.set_crs(target, allow_override=True)
    tgt_norm = str(target).upper().replace(" ", "")
    for datum, idx in out.groupby(datum_col, dropna=False).groups.items():
        sub = out.loc[idx]
        if pd.isna(datum) or str(datum).upper().replace(" ", "") == tgt_norm:
            out.loc[idx, "transform_id"] = f"land:identity:{target}"
            continue
        bounds = tuple(sub.geometry.total_bounds)  # subset AOI, in degrees (B7a)
        t = get_transformer(datum, target, aoi_bounds_4326=bounds)
        xs = sub.geometry.x.to_numpy()
        ys = sub.geometry.y.to_numpy()
        if is_dynamic_frame(datum):
            # TODO(D6): provisional — dynamic source frame -> tt = coord_epoch
            if "coord_epoch" not in sub.columns:
                raise ValueError(
                    f"dynamic-frame subset {datum!r} has no coord_epoch column; "
                    "cannot evaluate the time-dependent transform (an omitted tt "
                    "silently lands at the operation's t_epoch — see "
                    "docs/crs_implementation.md §1)"
                )
            tt = sub["coord_epoch"].to_numpy(dtype="float64")
            if np.isnan(tt).any():
                raise ValueError(
                    f"dynamic-frame subset {datum!r} has NaN coord_epoch for "
                    f"{int(np.isnan(tt).sum())}/{len(tt)} rows; refusing to land "
                    "(NaN tt propagates to NaN coordinates)"
                )
            # tt passed -> pyproj returns (x, y, tt); tt is echoed, not consumed
            x, y, _ = t.transform(xs, ys, tt=tt, errcheck=True)
            tid = f"land:{datum}->{target}|acc={t.accuracy}m|tt=coord_epoch"
        else:
            x, y = t.transform(xs, ys, errcheck=True)
            tid = f"land:{datum}->{target}|acc={t.accuracy}m"
        out.loc[idx, "geometry"] = gpd.points_from_xy(x, y)
        out.loc[idx, "transform_id"] = tid
        logger.info("landed %d rows %s -> %s (%s, accuracy %s m)",
                    len(idx), datum, target, t.description, t.accuracy)
    return out.set_crs(target, allow_override=True)
