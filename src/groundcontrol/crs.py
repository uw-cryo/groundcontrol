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
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
import pyproj
from pyproj.transformer import TransformerGroup

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#: Row count above which a report stops carrying its PER-ROW array (the aggregate fields and
#: the durable per-row COLUMN still cover it). Mirrors ``velocity.PER_ROW_PROVENANCE_MAX``.
#:
#: Not a memory nicety — a correctness-adjacent performance trap. pandas ``__finalize__``
#: runs ``deepcopy(other.attrs)`` on essentially every DataFrame operation, so a per-row list
#: parked in ``attrs`` is deep-copied, element by element, for the rest of the frame's life.
#: Measured 2026-08-08 on a 13.3 M-point ICESat-2 set: **1.98 billion** ``deepcopy`` calls
#: (~2,270 s) across one preprocessing run — writing a CSV took 129 s instead of 3 s, and one
#: diagnostic PNG took 241 s instead of 16 s. Per-row provenance belongs in a column
#: (``epoch_residual_m``), never in ``attrs``.
PER_ROW_PROVENANCE_MAX = 100_000


def _per_row(values):
    """Per-row report payload: a plain list up to :data:`PER_ROW_PROVENANCE_MAX`, else None."""
    return [float(v) for v in values] if len(values) <= PER_ROW_PROVENANCE_MAX else None


#: Public API (Phase D). Consumers keep the submodule-qualified convention
#: (``from groundcontrol.crs import propagate_epoch``); nothing is re-exported
#: from the package ``__init__``.
__all__ = [
    # errors
    "NoTransformPathError",
    # decimal-year helpers (plan B9)
    "decyear",
    "decyear_inv",
    # stage 1 — frame transforms
    "get_transformer",
    "transform_points",
    "land_horizontal",
    "ngs_datum_to_epsg",
    "is_dynamic_frame",
    # stage 2 — intra-frame epoch propagation
    "PLATE_MOTION_RATE_BOUND",
    "PlateMotionModel",
    "EulerPoleModel",
    "ITRF2020PMM",
    "ITRF2020_PMM_DEG_PER_MYR",
    "ITRF2020_ORB_MM_PER_YR",
    "assign_plate",
    "enu_to_ecef",
    "ecef_to_enu",
    "check_frame_epoch_reduced",
    "propagate_epoch",
]


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


def _plate_fixed_datum(crs_obj) -> bool:
    """True when the CRS's datum is plate-fixed (static, non-ITRF-aliased).

    Datum *type* metadata does not encode plate-fixity: ETRS89 is an EPSG
    datum ensemble whose members are all Eurasia-fixed, while the WGS84
    ensemble (and WGS84-named single frames from PROJ strings) are
    ITRF-aliased in practice. Rule: dynamic frames are never plate-fixed;
    WGS84 by *name* is treated as dynamic-aliased; every other geodetic
    frame or ensemble (NAD83*, ETRS89, GDA*, ...) is treated as plate-fixed
    — fail-loud rather than fabricate plate motion.
    """
    d = pyproj.CRS(crs_obj).datum
    tname = ((d.type_name or "") if d is not None else "").lower()
    if "dynamic" in tname:
        return False
    name = ((d.name or "") if d is not None else "").upper()
    if "WGS" in name or "WORLD GEODETIC" in name:
        return False
    return True


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


# ---------------------------------------------------------------------------
# Stage 2 — intra-frame epoch propagation (crs_implementation.md §1 stage 2)
# ---------------------------------------------------------------------------
# Stage 1 (transform_points / land_horizontal above) changes *frame* at each
# point's coordinate epoch — PROJ then relabels, not propagates, the output
# epoch (§1). Stage 2 here is the missing intra-frame move from each point's
# coord_epoch to a user-chosen target_epoch:
#     x += vel_enu · (target_epoch − coord_epoch)
# using per-point ENU velocities (NGL/MIDAS) where they exist, else a
# plate-motion model, else a no-op with the velocity·Δt bound surfaced (§1,
# §7.1 op_kind "velocity propagation"). It composes with stage 1 by running in
# the (geographic) dynamic source frame either before or after the frame
# transform — the coord_20 truth test asserts the two orders commute to mm
# (§8 test 2). An ITRF-at-fixed-epoch target *requires* stage 2 for every
# ITRF-native source; plate-fixed targets make it ≈ 0 by construction.

#: Coarse global upper bound on horizontal plate speed (m/yr) used only to
#: *report* the velocity·Δt uncertainty of rows left un-propagated (no
#: per-point velocity, no plate model). Owner decision (dshean 2026-07-05):
#: the bound must cover the fastest plate motion anywhere (~0.1 up to
#: ~0.16 m/yr), not a typical CONUS rate (~0.02 m/yr). This is a reporting
#: bound only — it is never applied to a coordinate. Replace with a real
#: per-row rate once a plate table is bundled.
PLATE_MOTION_RATE_BOUND = 0.16


@runtime_checkable
class PlateMotionModel(Protocol):
    """Interface for a plate-motion velocity source (the stage-2 PMM fallback).

    A model returns the per-point ENU velocity (m/yr) at a location, used to
    propagate rows that carry no per-point (MIDAS) velocity.
    :class:`EulerPoleModel` is the generic concrete implementation (caller
    supplies the pole); :class:`ITRF2020PMM` bundles the published ITRF2020
    plate poles (Altamimi et al. 2023), e.g. ``ITRF2020PMM("NOAM")``.
    """

    name: str

    def velocity_enu(self, lon, lat, h=0.0):  # -> (v_e, v_n, v_u) arrays, m/yr
        ...


def enu_to_ecef(e, n, u, lon, lat):
    """Rotate a local ENU vector to ECEF at geodetic ``(lon, lat)`` (degrees).

    Columns of the rotation are the ECEF components of the East/North/Up unit
    vectors. The inverse is :func:`ecef_to_enu`. Applying an ENU velocity as if
    it were ECEF (skipping this rotation) is the classic bug the §8-test-6
    ENU→ECEF check catches.
    """
    lam = np.radians(lon)
    phi = np.radians(lat)
    sl, cl = np.sin(lam), np.cos(lam)
    sp, cp = np.sin(phi), np.cos(phi)
    x = -sl * e - sp * cl * n + cp * cl * u
    y = cl * e - sp * sl * n + cp * sl * u
    z = cp * n + sp * u
    return x, y, z


def ecef_to_enu(x, y, z, lon, lat):
    """Rotate an ECEF vector to local ENU at geodetic ``(lon, lat)`` (degrees).

    Transpose of :func:`enu_to_ecef` (an orthonormal rotation).
    """
    lam = np.radians(lon)
    phi = np.radians(lat)
    sl, cl = np.sin(lam), np.cos(lam)
    sp, cp = np.sin(phi), np.cos(phi)
    e = -sl * x + cl * y
    n = -sp * cl * x - sp * sl * y + cp * z
    u = cp * cl * x + cp * sl * y + sp * z
    return e, n, u


def _ellipsoid_params(crs_like) -> tuple[float, float]:
    """(semi-major axis a [m], first eccentricity squared e²) for a CRS's ellipsoid."""
    ell = pyproj.CRS(crs_like).ellipsoid
    a = float(ell.semi_major_metre)
    rf = ell.inverse_flattening
    f = 0.0 if not rf else 1.0 / float(rf)
    return a, f * (2.0 - f)


def _geodetic_to_ecef(lon, lat, h, a, e2):
    """Geodetic (lon, lat deg; h m) -> ECEF (X, Y, Z m). Closed form."""
    lam = np.radians(lon)
    phi = np.radians(lat)
    sphi, cphi = np.sin(phi), np.cos(phi)
    n = a / np.sqrt(1.0 - e2 * sphi ** 2)
    x = (n + h) * cphi * np.cos(lam)
    y = (n + h) * cphi * np.sin(lam)
    z = (n * (1.0 - e2) + h) * sphi
    return x, y, z


def _apply_enu_displacement(lon, lat, h, d_e, d_n, d_u, a, e2):
    """Move geodetic (lon, lat deg; h m) by an ENU displacement (m).

    Uses the local radii of curvature (meridian ``M``, prime-vertical ``N``).
    Exact to well below 1 µm for the decimetre-scale displacements stage 2
    produces (linearisation error ~ (d/R)²·d ≈ 1e-15 m); the §8-test-6 ENU→ECEF
    rotation is the independent oracle the tests cross-check this against.
    """
    lat_rad = np.radians(lat)
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    w = np.sqrt(1.0 - e2 * sin_lat ** 2)
    m = a * (1.0 - e2) / w ** 3        # meridian radius of curvature
    n = a / w                          # prime-vertical radius of curvature
    dlat = np.degrees(d_n / (m + h))
    dlon = np.degrees(d_e / ((n + h) * cos_lat))
    return lon + dlon, lat + dlat, h + d_u


class EulerPoleModel:
    """Rigid-plate ENU velocities from an Euler pole (a concrete PMM fallback).

    The caller supplies the rotation pole (geographic lat/lon of the pole and
    the angular rate in degrees/Myr) or a cartesian angular velocity via
    :meth:`from_angular_velocity`; for the published ITRF2020 plate poles use
    :class:`ITRF2020PMM` instead. Velocity at a point is the rigid-body
    ``v = Ω × r`` in ECEF, rotated to local ENU, so a pure-horizontal plate
    rotation yields ~zero vertical velocity.
    """

    def __init__(self, pole_lat_deg, pole_lon_deg, rate_deg_per_myr,
                 name="euler", ellipsoid="EPSG:7912"):
        self.pole_lat_deg = float(pole_lat_deg)
        self.pole_lon_deg = float(pole_lon_deg)
        self.rate_deg_per_myr = float(rate_deg_per_myr)
        self.name = str(name)
        self._a, self._e2 = _ellipsoid_params(ellipsoid)
        omega = np.radians(self.rate_deg_per_myr) * 1e-6  # rad/yr
        plam = np.radians(self.pole_lon_deg)
        pphi = np.radians(self.pole_lat_deg)
        self._omega_ecef = omega * np.array([
            np.cos(pphi) * np.cos(plam),
            np.cos(pphi) * np.sin(plam),
            np.sin(pphi),
        ])

    @classmethod
    def from_angular_velocity(cls, wx, wy, wz, *, unit, name="euler",
                              ellipsoid="EPSG:7912"):
        """Build from a cartesian angular velocity (ECEF x/y/z components).

        ``unit`` is explicit — no default — because published tables use both
        conventions: ``'deg/Myr'`` (the ITRF2020-PMM.dat product file) and
        ``'mas/yr'`` (Altamimi et al. 2023 Table 1; 1 deg/Myr = 3.6 mas/yr).
        Converts to the pole lat/lon + rate form and reuses ``__init__`` (the
        ``_omega_ecef`` path), so both construction routes are one code path.
        """
        scale = {"deg/Myr": 1.0, "mas/yr": 1.0 / 3.6}.get(unit)
        if scale is None:
            raise ValueError(f"unit must be 'deg/Myr' or 'mas/yr', got {unit!r}")
        wx, wy, wz = float(wx) * scale, float(wy) * scale, float(wz) * scale
        rate = float(np.sqrt(wx ** 2 + wy ** 2 + wz ** 2))  # deg/Myr
        if rate == 0.0:
            raise ValueError("zero angular velocity (wx = wy = wz = 0)")
        pole_lat = float(np.degrees(np.arctan2(wz, np.hypot(wx, wy))))
        pole_lon = float(np.degrees(np.arctan2(wy, wx)))
        return cls(pole_lat, pole_lon, rate, name=name, ellipsoid=ellipsoid)

    def velocity_enu(self, lon, lat, h=0.0):
        """ENU velocity (m/yr) at geodetic ``(lon, lat)`` (degrees). Vectorized."""
        lon = np.asarray(lon, dtype="float64")
        lat = np.asarray(lat, dtype="float64")
        h = np.broadcast_to(np.asarray(h, dtype="float64"), lon.shape)
        x, y, z = _geodetic_to_ecef(lon, lat, h, self._a, self._e2)
        wx, wy, wz = self._omega_ecef
        vx = wy * z - wz * y
        vy = wz * x - wx * z
        vz = wx * y - wy * x
        return ecef_to_enu(vx, vy, vz, lon, lat)


# ---------------------------------------------------------------------------
# ITRF2020 plate motion model (Altamimi et al. 2023) — bundled pole table (C1)
# ---------------------------------------------------------------------------

#: ITRF2020-PMM rotation-pole cartesian angular velocities, **deg/Myr**
#: (ECEF x/y/z), transcribed verbatim from the ITRF product file (Table S1)
#: https://itrf.ign.fr/docs/solutions/itrf2020/ITRF2020-PMM.dat
#: (Altamimi, Métivier, Rebischung, Kreemer & Parsons 2023, ITRF2020 Plate
#: Motion Model, Geophys. Res. Lett. 50, e2023GL106373,
#: doi:10.1029/2023GL106373). Paper Table 1 lists the same poles in mas/yr;
#: 1 deg/Myr = 3.6 mas/yr (NOAM cross-check: 0.0126*3.6 = 0.045 ✓).
ITRF2020_PMM_DEG_PER_MYR: dict[str, tuple[float, float, float]] = {
    "AMUR": (-0.0364, -0.1532, 0.2325),
    "ANTA": (-0.0746, -0.0866, 0.1882),
    "ARAB": (0.3135, -0.0406, 0.3994),
    "AUST": (0.4132, 0.3265, 0.3398),
    "CARB": (0.0576, -0.3949, 0.2017),
    "EURA": (-0.0237, -0.1442, 0.2091),
    "INDI": (0.3159, 0.0037, 0.4010),
    "NAZC": (-0.0907, -0.4336, 0.4459),
    "NOAM": (0.0126, -0.1849, -0.0272),
    "NUBI": (0.0250, -0.1625, 0.1991),
    "PCFC": (-0.1122, 0.2836, -0.5984),
    "SOAM": (-0.0725, -0.0784, -0.0437),
    "SOMA": (-0.0225, -0.1997, 0.2401),
}

#: ITRF2020-PMM origin rate bias (mm/yr, ECEF Tx/Ty/Tz), same product file.
#: ITRF advisory: "Users are advised to add the estimated ORB to the
#: horizontal velocities predicted by the ITRF2020 PMM rotation poles."
ITRF2020_ORB_MM_PER_YR: tuple[float, float, float] = (0.37, 0.35, 0.74)

#: PB2002 (Bird 2003) two-letter plate codes -> the ITRF2020-PMM plates that
#: model them. Points on any other PB2002 plate (e.g. JF Juan de Fuca, CO
#: Cocos) get NaN velocity, so propagate_epoch leaves them un-moved and
#: surfaces the epoch_residual_m bound instead of inventing motion.
_PB2002_TO_ITRF: dict[str, str] = {
    "AF": "NUBI", "AM": "AMUR", "AN": "ANTA", "AR": "ARAB", "AU": "AUST",
    "CA": "CARB", "EU": "EURA", "IN": "INDI", "NA": "NOAM", "NZ": "NAZC",
    "PA": "PCFC", "SA": "SOAM", "SO": "SOMA",
}

_pb2002_plates_cache = None


def _pb2002_plates():
    """Bundled decimated PB2002 plate polygons (Bird 2003), lazy + cached.

    Sourced via fraxen/tectonicplates (ODC-By 1.0) and decimated for
    packaging — see ``src/groundcontrol/data/README.md`` for attribution and
    the decimation parameters.
    """
    global _pb2002_plates_cache
    if _pb2002_plates_cache is None:
        from importlib.resources import as_file, files

        import geopandas as gpd
        src = files("groundcontrol").joinpath("data/pb2002_plates_decimated.geojson")
        with as_file(src) as path:
            _pb2002_plates_cache = gpd.read_file(path)[["Code", "geometry"]]
    return _pb2002_plates_cache


def assign_plate(lon, lat):
    """ITRF2020-PMM plate name for each geodetic ``(lon, lat)`` (degrees).

    Vectorized point-in-polygon assignment against the bundled PB2002
    boundaries, mapped to the 13 ITRF2020-PMM plates. Returns an object array
    of plate names with ``None`` for points on plates the PMM does not model
    (e.g. Juan de Fuca) or inside boundary-zone slivers. Longitudes are
    wrapped to [-180, 180), so 0..360 inputs are handled.
    """
    import geopandas as gpd

    lon = np.atleast_1d(np.asarray(lon, dtype="float64"))
    lat = np.atleast_1d(np.asarray(lat, dtype="float64"))
    lon_w = ((lon + 180.0) % 360.0) - 180.0
    plates = _pb2002_plates()
    pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(lon_w.ravel(), lat.ravel()), crs=plates.crs)
    # "intersects", not "within": lon = ±180 wraps exactly onto the seam edge
    # of the antimeridian-split polygons and "within" excludes boundaries —
    # every such point would silently get no plate.
    joined = gpd.sjoin(pts, plates, how="left", predicate="intersects")
    # The decimated polygons overlap slightly along boundaries (64 pairs,
    # worst ~1 deg²), and boundary points intersect both neighbors. Keep a
    # deterministic (alphabetical-code) choice — arbitrary either way inside
    # the ~0.05° boundary blur, but stable across runs and geopandas versions.
    joined = joined.sort_values("Code", kind="stable")
    joined = joined[~joined.index.duplicated(keep="first")].sort_index()
    mapped = joined["Code"].map(_PB2002_TO_ITRF)  # unmapped/no-hit -> NaN
    return mapped.astype(object).where(mapped.notna(), None).to_numpy().reshape(lon.shape)


class ITRF2020PMM:
    """ITRF2020 plate-motion velocities (Altamimi et al. 2023) — a stage-2 PMM.

    ``plate`` (e.g. ``'NOAM'``) selects a fixed plate from
    :data:`ITRF2020_PMM_DEG_PER_MYR` — correct for single-plate AOIs (Las
    Vegas, Casa Grande) and needs no boundary data. ``plate=None`` assigns
    each point to a plate via the bundled PB2002 boundaries
    (:func:`assign_plate`) — needed where an AOI straddles a boundary (San
    Francisco: NOAM/PCFC across the San Andreas). Points on plates the PMM
    does not model get NaN velocity, so :func:`propagate_epoch` leaves them
    un-moved and surfaces the ``epoch_residual_m`` bound.

    ``apply_orb=True`` (default, per the ITRF advisory) adds the origin-rate
    bias translation to the **horizontal** (E/N) velocity components only: a
    rigid plate rotation predicts ~zero vertical motion, and the ORB's
    vertical projection is origin drift, not plate motion.
    """

    def __init__(self, plate, *, apply_orb=True, ellipsoid="EPSG:7912"):
        self.plate = None if plate is None else str(plate).upper()
        self.apply_orb = bool(apply_orb)
        self._ellipsoid = ellipsoid
        self._poles: dict[str, EulerPoleModel] = {}
        if self.plate is not None:
            self._pole_for(self.plate)  # validate + cache eagerly
        orb = "+ORB" if self.apply_orb else ""
        self.name = f"ITRF2020-PMM[{self.plate or 'auto'}]{orb}"

    def _pole_for(self, code) -> EulerPoleModel:
        """The single pole-construction path (fixed-plate and boundary cases)."""
        pole = self._poles.get(code)
        if pole is None:
            if code not in ITRF2020_PMM_DEG_PER_MYR:
                raise KeyError(
                    f"unknown ITRF2020-PMM plate {code!r}; available: "
                    f"{sorted(ITRF2020_PMM_DEG_PER_MYR)}")
            wx, wy, wz = ITRF2020_PMM_DEG_PER_MYR[code]
            pole = self._poles[code] = EulerPoleModel.from_angular_velocity(
                wx, wy, wz, unit="deg/Myr", name=f"ITRF2020:{code}",
                ellipsoid=self._ellipsoid)
        return pole

    def velocity_enu(self, lon, lat, h=0.0):
        """ENU velocity (m/yr) at geodetic ``(lon, lat)`` (degrees). Vectorized."""
        lon = np.asarray(lon, dtype="float64")
        lat = np.asarray(lat, dtype="float64")
        h = np.broadcast_to(np.asarray(h, dtype="float64"), lon.shape)
        if self.plate is None:
            ve, vn, vu = self._velocity_by_boundary(lon, lat, h)
        else:
            ve, vn, vu = self._pole_for(self.plate).velocity_enu(lon, lat, h)
        if self.apply_orb:
            tx, ty, tz = (t * 1e-3 for t in ITRF2020_ORB_MM_PER_YR)  # -> m/yr
            oe, on, _ou = ecef_to_enu(tx, ty, tz, lon, lat)
            ve = ve + oe
            vn = vn + on
        return ve, vn, vu

    def _velocity_by_boundary(self, lon, lat, h):
        """plate=None path: per-point PB2002 assignment, NaN off-model."""
        shape = lon.shape
        lon, lat, h = np.atleast_1d(lon), np.atleast_1d(lat), np.atleast_1d(h)
        codes = assign_plate(lon, lat)
        ve = np.full(lon.shape, np.nan)
        vn = np.full(lon.shape, np.nan)
        vu = np.full(lon.shape, np.nan)
        for code in {c for c in codes.ravel() if c}:
            m = codes == code
            ve[m], vn[m], vu[m] = self._pole_for(code).velocity_enu(
                lon[m], lat[m], h[m])
        return ve.reshape(shape), vn.reshape(shape), vu.reshape(shape)


def check_frame_epoch_reduced(gdf, *, tol_yr: float = 1e-6, on_violation: str = "warn"):
    """QC (§1): plate-fixed rows must be reduced to their ``frame_epoch``.

    A plate-fixed realization publishes positions *reduced to* its reference
    epoch, so a finite ``frame_epoch`` with ``coord_epoch != frame_epoch``
    mechanically flags an unreduced position (e.g. raw RTK). Dynamic frames
    carry NaN ``frame_epoch`` and are skipped. Returns the boolean violation
    mask; ``on_violation='raise'`` raises instead of warning.
    """
    if "frame_epoch" not in gdf.columns or "coord_epoch" not in gdf.columns:
        return np.zeros(len(gdf), dtype=bool)
    fe = pd.to_numeric(gdf["frame_epoch"], errors="coerce").to_numpy(dtype="float64")
    ce = pd.to_numeric(gdf["coord_epoch"], errors="coerce").to_numpy(dtype="float64")
    plate_fixed = np.isfinite(fe)
    bad = plate_fixed & np.isfinite(ce) & (np.abs(ce - fe) > tol_yr)
    if bad.any():
        msg = (f"{int(bad.sum())} plate-fixed row(s) have coord_epoch != frame_epoch "
               f"(unreduced position; crs_implementation.md §1 QC)")
        if on_violation == "raise":
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=2)
    return bad


def propagate_epoch(gdf, target_epoch, *, source_crs=None, height_col: str = "height",
                    coord_epoch_col: str = "coord_epoch",
                    vel_cols=("vel_e", "vel_n", "vel_u"), plate_model=None,
                    on_nan_epoch: str = "raise",
                    residual_rate_m_per_yr: float = PLATE_MOTION_RATE_BOUND,
                    qc_frame_epoch: bool = True,
                    allow_static_frame: bool = False):
    """Stage 2: propagate each point from its ``coord_epoch`` to ``target_epoch``.

    ``x += vel_enu · (target_epoch − coord_epoch)`` (crs_implementation.md §1).
    Operates in the point's own (geographic) dynamic frame — run it before or
    after the stage-1 frame transform (:func:`transform_points` /
    :func:`land_horizontal`); the two orders commute to mm (§8 test 2). Only the
    geometry x/y and ``height_col`` move; ``coord_epoch`` is advanced to
    ``target_epoch`` for every propagated row.

    Velocity source, in priority order (§1):

    1. **per-point** ENU velocities from ``vel_cols`` (all three finite) — MIDAS
       for GNSS;
    2. **plate-motion model** ``plate_model`` (a :class:`PlateMotionModel`, e.g.
       :class:`EulerPoleModel`) for rows lacking per-point velocities;
    3. **none** — the row is left at its own epoch and its ``velocity·Δt`` bound
       is surfaced (WARNING log + ``gdf.attrs['epoch_propagation']`` report +
       the per-row ``epoch_residual_m`` column).

    Parameters
    ----------
    gdf : GeoDataFrame with 2D geographic point geometry (lon/lat degrees) and a
        numeric ``height_col`` (ellipsoidal). Its CRS (or ``source_crs``) must be
        **geographic** — projected input raises (project *after* stage 2).
    target_epoch : scalar decimal year to propagate every row to.
    source_crs : geographic CRS to use when ``gdf.crs`` is None (e.g. NGL native
        frames land un-set); supplies the ellipsoid for the ENU→geodetic step.
    plate_model : optional :class:`PlateMotionModel` fallback (priority 2).
    on_nan_epoch : ``'raise'`` (default) — a row with a usable velocity but NaN
        ``coord_epoch`` raises (a NaN Δt would propagate to NaN coordinates);
        ``'skip'`` leaves it un-propagated with a warning (§8 test 9 policy flag).
    residual_rate_m_per_yr : reporting-only rate for the un-propagated
        ``velocity·Δt`` bound (see :data:`PLATE_MOTION_RATE_BOUND`).
    qc_frame_epoch : run :func:`check_frame_epoch_reduced` (warn) first.
    allow_static_frame : stage 2 is intra-DYNAMIC-frame motion; in a
        plate-fixed frame (NAD83(2011), ...) plate motion is ~zero by
        construction, so moving points there fabricates displacement while
        reporting ``epoch_residual_m=0`` — the guard raises instead. Pass
        True only when the velocities deliberately express intra-frame motion
        (e.g. local subsidence rates in NAD83(2011)). WGS84-named frames and
        ensembles pass the guard (ITRF-aliased); plate-fixed ensembles like
        ETRS89 are guarded. No-op rows (zero Δt, zero velocity, or a plate
        model that would never be consulted) never trip it.

    Returns a copy: propagated geometry/height, advanced ``coord_epoch``,
    ``transform_id`` appended with a ``prop:`` tag, a durable per-row
    ``epoch_residual_m`` column (the schema column that survives export/concat;
    0.0 = epoch-reconciled, >0 = velocity·Δt bound for rows left un-propagated,
    NaN = not assessable — NaN ``coord_epoch`` under ``on_nan_epoch='skip'``),
    and a ``gdf.attrs['epoch_propagation']`` report (counts, models used, max
    applied displacement, and the surfaced residual-bound array).
    """
    import geopandas as gpd

    target_epoch = float(target_epoch)
    if not np.isfinite(target_epoch):
        raise ValueError(f"target_epoch must be finite, got {target_epoch!r}")

    out = gdf.copy()
    crs_like = out.crs if out.crs is not None else source_crs
    if crs_like is None:
        raise ValueError(
            "propagate_epoch needs a geographic CRS: gdf.crs is None and "
            "source_crs was not given (the ENU→geodetic step needs the ellipsoid)")
    crs_obj = pyproj.CRS(crs_like)
    if not crs_obj.is_geographic:
        raise ValueError(
            f"propagate_epoch expects geographic lon/lat coordinates; got "
            f"{crs_obj.name!r} (projected). Run stage 2 before projecting.")
    a, e2 = _ellipsoid_params(crs_obj)
    static_frame = _plate_fixed_datum(crs_obj)

    if qc_frame_epoch:
        check_frame_epoch_reduced(out)

    n = len(out)
    report = {
        "target_epoch": target_epoch, "crs": str(crs_obj.to_authority() or crs_obj.name),
        "n_total": n, "n_propagated": 0, "n_noop": n, "n_unassessable": 0,
        "models": {"per_point": 0, "plate": 0, "none": n},
        "plate_model": getattr(plate_model, "name", None),
        "max_applied_displacement_m": 0.0,
        "residual_rate_m_per_yr": float(residual_rate_m_per_yr),
        "max_residual_bound_m": 0.0,
        "residual_bound_m": [0.0] * n if n <= PER_ROW_PROVENANCE_MAX else None,
    }
    if n == 0:
        out["epoch_residual_m"] = np.zeros(0)
        out.attrs["epoch_propagation"] = report
        return out

    def _col(name):
        if name in out.columns:
            # copy=True: pandas>=3 CoW returns read-only zero-copy views and
            # ve/vn/vu are filled in place at the plate-model step below
            return pd.to_numeric(out[name], errors="coerce").to_numpy(
                dtype="float64", copy=True)
        return np.full(n, np.nan)

    lon = out.geometry.x.to_numpy(dtype="float64")
    lat = out.geometry.y.to_numpy(dtype="float64")
    h = pd.to_numeric(out[height_col], errors="raise").to_numpy(dtype="float64")
    ve, vn, vu = (_col(c) for c in vel_cols)

    has_vel = np.isfinite(ve) & np.isfinite(vn) & np.isfinite(vu)
    partial = (np.isfinite(ve) | np.isfinite(vn) | np.isfinite(vu)) & ~has_vel
    if partial.any():
        warnings.warn(
            f"{int(partial.sum())} row(s) have a partial ENU velocity (some but not "
            "all of vel_e/n/u finite); treating them as having no per-point velocity",
            stacklevel=2)

    ce = _col(coord_epoch_col)
    dt = target_epoch - ce
    if static_frame and not allow_static_frame:
        # mover-aware: rows that would actually be displaced. Zero Δt, zero
        # velocity, and a plate model that would never be consulted are no-ops
        # and must not trip the guard.
        would_move = np.isfinite(dt) & (dt != 0.0)
        mover = has_vel & ((ve != 0.0) | (vn != 0.0) | (vu != 0.0)) & would_move
        if mover.any():
            raise ValueError(
                f"{int(mover.sum())} row(s) carry per-point velocities and a "
                f"nonzero Δt in the plate-fixed frame {crs_obj.name!r}. "
                "MIDAS/ITRF velocities are expressed in a dynamic frame — "
                "propagating them here double-counts plate motion. If the "
                "velocities deliberately express intra-frame motion (e.g. local "
                "subsidence rates in NAD83(2011)), pass allow_static_frame=True.")
        if plate_model is not None and (~has_vel & would_move).any():
            raise ValueError(
                f"plate_model={getattr(plate_model, 'name', plate_model)!r} would "
                f"be consulted for {int((~has_vel & would_move).sum())} row(s) in "
                f"the plate-fixed frame {crs_obj.name!r}: plate motion is ~zero "
                "by construction in a static frame, so an ITRF plate-motion "
                "model here fabricates cm/yr displacement while reporting "
                "epoch_residual_m=0. Run stage 2 in the dynamic frame (before "
                "land_horizontal), or pass allow_static_frame=True only for a "
                "model that deliberately expresses intra-frame motion.")

    model_label = np.where(has_vel, "per_point", "none").astype(object)
    if plate_model is not None:
        need = ~has_vel
        if static_frame and not allow_static_frame:
            # the guard above passed only because no consulted row would move
            # (zero/NaN Δt) — skip the fill entirely so durable provenance
            # (transform_id, models counts) never records an ITRF plate model
            # applied inside a plate-fixed frame; rows stay prop:noop
            need &= np.isfinite(dt) & (dt != 0.0)
        if need.any():
            pe, pn, pu = plate_model.velocity_enu(lon[need], lat[need], h[need])
            ve[need], vn[need], vu[need] = pe, pn, pu
            filled = need.copy()
            filled[need] = np.isfinite(pe) & np.isfinite(pn) & np.isfinite(pu)
            model_label[filled] = "plate"

    movable = np.isfinite(ve) & np.isfinite(vn) & np.isfinite(vu)

    nan_epoch = movable & ~np.isfinite(ce)
    if nan_epoch.any():
        if on_nan_epoch == "raise":
            raise ValueError(
                f"{int(nan_epoch.sum())} row(s) have a usable velocity but NaN "
                f"{coord_epoch_col!r}; a NaN Δt would propagate to NaN coordinates. "
                "Fix the epoch or pass on_nan_epoch='skip' to leave them un-propagated.")
        if on_nan_epoch != "skip":
            raise ValueError(f"on_nan_epoch must be 'raise' or 'skip', got {on_nan_epoch!r}")
        warnings.warn(
            f"{int(nan_epoch.sum())} row(s) with velocity but NaN {coord_epoch_col!r} "
            "left un-propagated (on_nan_epoch='skip')", stacklevel=2)
        movable = movable & ~nan_epoch
        model_label[nan_epoch] = "none"

    moved = movable
    lon2, lat2, h2 = lon.copy(), lat.copy(), h.copy()
    disp = np.zeros(n)
    if moved.any():
        d_e = ve[moved] * dt[moved]
        d_n = vn[moved] * dt[moved]
        d_u = vu[moved] * dt[moved]
        lo, la, hh = _apply_enu_displacement(
            lon[moved], lat[moved], h[moved], d_e, d_n, d_u, a, e2)
        lon2[moved], lat2[moved], h2[moved] = lo, la, hh
        disp[moved] = np.sqrt(d_e ** 2 + d_n ** 2 + d_u ** 2)

    ce_out = ce.copy()
    ce_out[moved] = target_epoch

    # velocity·Δt bound for the rows we could NOT propagate (surface, don't hide)
    noop = ~moved
    bound = np.zeros(n)
    finite_dt_noop = noop & np.isfinite(dt)
    bound[finite_dt_noop] = np.abs(dt[finite_dt_noop]) * float(residual_rate_m_per_yr)

    # write coordinates / epoch back
    out["geometry"] = gpd.points_from_xy(lon2, lat2)
    out[height_col] = h2
    if coord_epoch_col in out.columns:
        out[coord_epoch_col] = ce_out
    # durable per-row residual (schema column) — survives export/concat where
    # attrs evaporate. Tri-state: 0.0 = epoch-reconciled, >0 = bounded
    # un-propagated velocity·Δt, NaN = not assessable (NaN coord_epoch under
    # on_nan_epoch='skip' — never claim 0.0 for an unknowable Δt; matches
    # schema.normalize's NaN for never-propagated frames)
    col_bound = bound.copy()
    col_bound[noop & ~np.isfinite(dt)] = np.nan
    out["epoch_residual_m"] = col_bound

    # transform_id: append a compact provenance tag (chain-ordered)
    tag = np.where(
        moved,
        np.where(model_label == "plate",
                 f"prop:plate[{getattr(plate_model, 'name', 'plate')}]->{target_epoch:g}",
                 f"prop:per_point->{target_epoch:g}"),
        "prop:noop",
    )
    if "transform_id" in out.columns:
        existing = out["transform_id"].tolist()
        chained = [t if pd.isna(e) else f"{e}+{t}" for e, t in zip(existing, tag)]
        out["transform_id"] = pd.array(chained, dtype="string")

    # report + fail-loud-adjacent surfacing
    report.update(
        n_propagated=int(moved.sum()), n_noop=int(noop.sum()),
        models={
            "per_point": int((model_label == "per_point").sum()),
            "plate": int((model_label == "plate").sum()),
            "none": int((model_label == "none").sum()),
        },
        max_applied_displacement_m=float(disp.max()),
        # report mirrors the durable column exactly: NaN = unassessable Δt
        # (on_nan_epoch='skip'), never a claimed-zero bound for unknown Δt
        max_residual_bound_m=(float(np.nanmax(col_bound))
                              if np.isfinite(col_bound).any() else float("nan")),
        residual_bound_m=_per_row(col_bound),
        n_unassessable=int(np.isnan(col_bound).sum()),
    )
    out.attrs["epoch_propagation"] = report

    logger.info(
        "propagate_epoch -> %g: %d/%d propagated (per_point=%d, plate=%d), "
        "%d no-op | max applied %.4f m | max residual bound %.4f m",
        target_epoch, report["n_propagated"], n, report["models"]["per_point"],
        report["models"]["plate"], report["n_noop"],
        report["max_applied_displacement_m"], report["max_residual_bound_m"])
    if report["max_residual_bound_m"] > 1e-4 or report["n_unassessable"]:
        warnings.warn(
            f"{report['n_noop']} row(s) left at their own epoch (no usable velocity); "
            f"un-propagated velocity·Δt bound up to {report['max_residual_bound_m']:.3f} m "
            f"(at rate {residual_rate_m_per_yr} m/yr), {report['n_unassessable']} row(s) "
            "unassessable (NaN Δt) — see attrs['epoch_propagation']",
            stacklevel=2)
    return out
