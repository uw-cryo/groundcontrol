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


def land_horizontal(gdf, target: str = "EPSG:6318", datum_col: str = "horizontal_crs"):
    """Land a mixed-realization frame horizontally into one target CRS (plan B7).

    Groups rows by ``datum_col`` (per-row EPSG strings) and transforms each
    subset with its own transformer built from the **subset's** bounds (B7a).
    Heights are untouched: the current sources carry NAVD88 *orthometric*
    heights, which are invariant under NAD83 horizontal realization changes —
    ellipsoidal heights would not be (B7b); those remain provenance in ``raw``.

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
        x, y = t.transform(sub.geometry.x.to_numpy(), sub.geometry.y.to_numpy(),
                           errcheck=True)
        out.loc[idx, "geometry"] = gpd.points_from_xy(x, y)
        out.loc[idx, "transform_id"] = f"land:{datum}->{target}|acc={t.accuracy}m"
        logger.info("landed %d rows %s -> %s (%s, accuracy %s m)",
                    len(idx), datum, target, t.description, t.accuracy)
    return out.set_crs(target, allow_override=True)
