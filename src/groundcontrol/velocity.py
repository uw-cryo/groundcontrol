"""General observed-velocity lookup — the tier-1 source for stage-2 epoch propagation.

``groundcontrol.crs.propagate_epoch`` moves a point from its ``coord_epoch`` to a
user ``target_epoch`` by ``x += vel_enu · Δt``, drawing the velocity from a three-tier
ladder (crs_implementation.md §1):

1. **observed** per-point ENU velocities (this module) — interpolated from a network
   of GNSS station velocities (NGL/MIDAS via :mod:`groundcontrol.sources.ngl`);
2. **plate-motion model** (:class:`groundcontrol.crs.EulerPoleModel`) for rows with no
   nearby stations;
3. **none** — leave the point at its own epoch and surface the ``velocity·Δt`` bound.

This module implements tier 1 and is **source-agnostic by construction**: the core
:func:`interpolate_velocity` takes a plain station DataFrame carrying ``lon``/``lat`` +
``vel_e``/``vel_n``/``vel_u`` (m/yr) and never references MIDAS/NGL — any velocity
network (MIDAS, UNAVCO/GAGE, EPN, a bundled ITRF-PMM grid) can feed it.

**Global by design — no site-specific or fault geometry is hardcoded.** Rather than
encoding fault locations (e.g. the Hayward or San Andreas), fault/block boundaries are
detected *empirically*: the station-to-station velocity spread inside the search radius
is measured, and a selection whose spread exceeds a threshold — the signature of pulling
stations from two sides of a discontinuity — is flagged (``quality='spread_warning'``)
rather than silently averaged across the break. Where too few stations fall inside the
radius the velocity is returned as NaN with ``quality='low_density'`` so the caller falls
back to the tier-2 plate model. Nothing here assumes CONUS, NAD83(2011), or any site.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#: Mean Earth radius (IUGG), km — for the haversine station-distance metric.
_EARTH_RADIUS_KM = 6371.0088

#: Default nearest-station search radius (km). Inside a rigid plate this captures
#: many mutually-consistent stations; near a plate-boundary/deforming zone it will
#: also pull cross-boundary stations, which the spread gate then flags.
DEFAULT_RADIUS_KM = 75.0
#: Fewer than this many stations inside the radius -> NaN + ``low_density`` (fall back
#: to the plate model). Keeps a lone station from masquerading as a network solution.
DEFAULT_MIN_STATIONS = 3
#: At most this many nearest stations are combined (caps cost + over-smoothing).
DEFAULT_MAX_STATIONS = 15
#: Horizontal station-velocity spread (std, mm/yr) above which the selection is treated as
#: straddling a fault/block boundary. Rationale, calibrated against live MIDAS: a rigid-
#: plate interior agrees to well under ~1 mm/yr over ~75 km (Las Vegas: 0.5 mm/yr); a
#: coherent block that still carries intraplate deformation + isolated noisy station pairs
#: reaches ~3 mm/yr (the San Francisco peninsula, Bay block); a selection that spans a
#: major locked fault (across the Hayward, ~9 mm/yr slip) jumps to 5-15+ mm/yr as two
#: blocks with different secular motion enter the set. 4 mm/yr sits above coherent-block
#: noise and below a genuine cross-fault signal, so uniform/single-block networks pass and
#: block-straddling selections trip. Configurable per call.
DEFAULT_SPREAD_THRESHOLD_MM_YR = 4.0
#: IDW distance floor (km) so a station coincident with the target (e.g. a GNSS mark
#: looking up its own velocity) yields a finite, dominant weight instead of div-by-zero.
_IDW_EPS_KM = 1e-3

#: Quality-flag vocabulary (string column ``quality``).
QUALITY_OK = "ok"                       # enough stations, spread within threshold
QUALITY_SPREAD_WARNING = "spread_warning"   # enough stations but spread too large
QUALITY_SINGLE_STATION = "single_station"   # only one station (no spread estimate)
QUALITY_LOW_DENSITY = "low_density"     # too few stations in radius -> velocity NaN

#: Columns :func:`interpolate_velocity` returns, one row per target point.
RESULT_COLUMNS = (
    "vel_e", "vel_n", "vel_u",          # combined ENU velocity (m/yr)
    "vel_spread_h", "vel_spread_u",     # station-velocity spread (m/yr; NaN if <2 used)
    "n_stations_used", "nearest_dist_km", "nearest_sta", "quality",
)

#: Station-DataFrame id columns tried (in order) to label the nearest station.
_ID_COL_CANDIDATES = ("sta", "id", "station", "site")


def _haversine_km(lon1, lat1, lon2, lat2):
    """Great-circle distance (km) on a sphere; vectorized, antimeridian/pole-safe."""
    lon1, lat1, lon2, lat2 = (np.radians(np.asarray(v, dtype="float64"))
                              for v in (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _pick_id_column(stations: pd.DataFrame) -> str | None:
    for col in _ID_COL_CANDIDATES:
        if col in stations.columns:
            return col
    return None


def _spread(values: np.ndarray) -> float:
    """Sample std (ddof=1) of a velocity component; NaN for fewer than 2 values."""
    return float(np.std(values, ddof=1)) if values.size > 1 else np.nan


def _combine_one(sel_vel: np.ndarray, dist_km: np.ndarray, method: str,
                 idw_power: float) -> np.ndarray:
    """Combine an (n_sel, 3) ENU velocity block into one (3,) vector.

    ``median`` — component-wise median (robust to a single cross-boundary outlier).
    ``idw`` — inverse-distance-weighted mean (nearest station dominates).
    """
    if method == "median":
        return np.median(sel_vel, axis=0)
    if method == "idw":
        w = 1.0 / np.maximum(dist_km, _IDW_EPS_KM) ** float(idw_power)
        return (w[:, None] * sel_vel).sum(axis=0) / w.sum()
    raise ValueError(f"unknown combine method {method!r}; use 'median' or 'idw'")


def interpolate_velocity(lon, lat, stations: pd.DataFrame, *,
                         radius_km: float = DEFAULT_RADIUS_KM,
                         min_stations: int = DEFAULT_MIN_STATIONS,
                         max_stations: int = DEFAULT_MAX_STATIONS,
                         method: str = "median",
                         spread_threshold_mm_yr: float = DEFAULT_SPREAD_THRESHOLD_MM_YR,
                         idw_power: float = 1.0,
                         lon_col: str = "lon", lat_col: str = "lat",
                         vel_cols=("vel_e", "vel_n", "vel_u")) -> pd.DataFrame:
    """Interpolate ENU velocity (m/yr) at target ``lon``/``lat`` from a station network.

    Source-agnostic: ``stations`` is any DataFrame with ``lon_col``/``lat_col`` (degrees)
    and ``vel_cols`` (m/yr) — MIDAS is the intended feed but nothing here requires it.

    For each target point the nearest ``<= max_stations`` stations inside ``radius_km``
    (haversine) are selected and robustly combined (``method='median'`` default, or
    ``'idw'``). The station-velocity spread inside the selection is measured; if the
    horizontal spread exceeds ``spread_threshold_mm_yr`` the selection likely straddles a
    fault/block boundary and ``quality`` is set to ``spread_warning`` (the velocity is
    still returned — the caller decides). Fewer than ``min_stations`` inside the radius
    yields NaN velocity + ``quality='low_density'`` so the caller falls back to the plate
    model.

    Parameters
    ----------
    lon, lat : scalar or 1-D array-like target coordinates (EPSG:4326 degrees).
    stations : station-velocity DataFrame (see above).
    radius_km, min_stations, max_stations : nearest-N-within-radius selection knobs.
    method : ``'median'`` (robust, default) or ``'idw'`` (inverse-distance weighting).
    spread_threshold_mm_yr : horizontal spread gate (see module constant).
    idw_power : IDW exponent (only used for ``method='idw'``).
    lon_col, lat_col, vel_cols : column names in ``stations``.

    Returns a DataFrame with :data:`RESULT_COLUMNS`, one row per target point (a 1-row
    frame for scalar input). ``vel_e/n/u`` are NaN where ``quality='low_density'``.
    """
    if method not in ("median", "idw"):
        raise ValueError(f"unknown combine method {method!r}; use 'median' or 'idw'")
    needed = (lon_col, lat_col, *vel_cols)
    missing = [c for c in needed if c not in stations.columns]
    if missing:
        raise ValueError(f"stations DataFrame missing column(s) {missing}; has {list(stations.columns)}")

    lon_arr = np.atleast_1d(np.asarray(lon, dtype="float64"))
    lat_arr = np.atleast_1d(np.asarray(lat, dtype="float64"))
    if lon_arr.shape != lat_arr.shape:
        raise ValueError(f"lon/lat shapes differ: {lon_arr.shape} vs {lat_arr.shape}")

    # Drop stations with any non-finite coordinate/velocity up front (fail-quiet on data
    # gaps — a station absent from the velocity file simply cannot contribute).
    st = stations.reset_index(drop=True)
    coords = st[[lon_col, lat_col, *vel_cols]].apply(pd.to_numeric, errors="coerce")
    st = st.loc[np.isfinite(coords.to_numpy()).all(axis=1)]
    slon = pd.to_numeric(st[lon_col], errors="coerce").to_numpy(dtype="float64")
    slat = pd.to_numeric(st[lat_col], errors="coerce").to_numpy(dtype="float64")
    svel = np.column_stack([pd.to_numeric(st[c], errors="coerce").to_numpy(dtype="float64")
                            for c in vel_cols])
    id_col = _pick_id_column(st)
    sids = st[id_col].astype("string").to_numpy() if id_col is not None else None

    rows = []
    for tlon, tlat in zip(lon_arr, lat_arr):
        rows.append(_lookup_one(tlon, tlat, slon, slat, svel, sids, radius_km,
                                min_stations, max_stations, method, idw_power,
                                spread_threshold_mm_yr))
    out = pd.DataFrame.from_records(rows, columns=RESULT_COLUMNS)
    out["n_stations_used"] = out["n_stations_used"].astype("int64")
    out["quality"] = out["quality"].astype("string")
    out["nearest_sta"] = out["nearest_sta"].astype("string")
    return out


def _lookup_one(tlon, tlat, slon, slat, svel, sids, radius_km, min_stations,
                max_stations, method, idw_power, spread_threshold_mm_yr) -> dict:
    """Single-target selection + combine (memory-safe: one M-vector, no N×M matrix)."""
    empty = dict(vel_e=np.nan, vel_n=np.nan, vel_u=np.nan, vel_spread_h=np.nan,
                 vel_spread_u=np.nan, n_stations_used=0, nearest_dist_km=np.nan,
                 nearest_sta=pd.NA, quality=QUALITY_LOW_DENSITY)
    if slon.size == 0:
        return empty
    dist = _haversine_km(tlon, tlat, slon, slat)
    within = np.flatnonzero(dist <= radius_km)
    if within.size == 0:
        return empty
    order = within[np.argsort(dist[within], kind="stable")]
    sel = order[:max_stations]
    n_used = int(sel.size)
    nearest = sel[0]
    nearest_dist = float(dist[nearest])
    nearest_sta = sids[nearest] if sids is not None else pd.NA

    if n_used < min_stations:
        e = dict(empty)
        e.update(n_stations_used=n_used, nearest_dist_km=nearest_dist, nearest_sta=nearest_sta)
        return e

    sel_vel = svel[sel]
    ve, vn, vu = _combine_one(sel_vel, dist[sel], method, idw_power)
    spread_e = _spread(sel_vel[:, 0])
    spread_n = _spread(sel_vel[:, 1])
    spread_u = _spread(sel_vel[:, 2])
    spread_h = float(np.hypot(spread_e, spread_n)) if np.isfinite(spread_e) else np.nan

    if n_used == 1:
        quality = QUALITY_SINGLE_STATION
    elif np.isfinite(spread_h) and spread_h * 1e3 > spread_threshold_mm_yr:
        quality = QUALITY_SPREAD_WARNING
    else:
        quality = QUALITY_OK

    return dict(vel_e=float(ve), vel_n=float(vn), vel_u=float(vu),
                vel_spread_h=spread_h, vel_spread_u=spread_u, n_stations_used=n_used,
                nearest_dist_km=nearest_dist, nearest_sta=nearest_sta, quality=quality)


def fill_velocities(gdf, stations: pd.DataFrame, *,
                    vel_cols=("vel_e", "vel_n", "vel_u"),
                    quality_col: str | None = None, **kwargs):
    """Populate ``gdf``'s ENU velocity columns from an observed station network.

    The tier-1 entry point a downstream application (or the dispatcher) calls **before**
    :func:`groundcontrol.crs.propagate_epoch`: it looks up each point's velocity with
    :func:`interpolate_velocity` and writes ``vel_cols``. Rows flagged ``low_density``
    stay NaN, so ``propagate_epoch`` transparently falls to its tier-2 plate model there.

    ``gdf`` must carry geographic (lon/lat degrees) point geometry — the same requirement
    ``propagate_epoch`` enforces. ``**kwargs`` pass straight through to
    :func:`interpolate_velocity` (radius/min/max/method/spread threshold, etc.).

    ``vel_cols`` names the three OUTPUT columns written on ``gdf`` (default the schema's
    ``vel_e``/``vel_n``/``vel_u``); the station-side column names are configured on
    :func:`interpolate_velocity` via ``**kwargs`` (``vel_cols`` there, defaulting to the
    same trio). Returns a copy with ``vel_cols`` filled and the full per-row lookup table
    plus a summary attached at ``out.attrs['velocity_fill']`` (provenance without widening
    the schema). If ``quality_col`` is given, the per-row quality flag is also written to
    that column. Original rows/columns are otherwise untouched.
    """
    out = gdf.copy()
    if len(out) == 0:
        out.attrs["velocity_fill"] = {"n_total": 0, "quality_counts": {}}
        return out
    lon = out.geometry.x.to_numpy(dtype="float64")
    lat = out.geometry.y.to_numpy(dtype="float64")
    res = interpolate_velocity(lon, lat, stations, **kwargs)
    for col, src in zip(vel_cols, ("vel_e", "vel_n", "vel_u")):
        out[col] = res[src].to_numpy(dtype="float64")
    if quality_col is not None:
        out[quality_col] = res["quality"].to_numpy()
    counts = res["quality"].value_counts().to_dict()
    out.attrs["velocity_fill"] = {
        "n_total": int(len(res)),
        "n_filled": int(res["vel_e"].notna().sum()),
        "quality_counts": {str(k): int(v) for k, v in counts.items()},
        "per_row": res.to_dict("records"),
    }
    logger.info("fill_velocities: %d/%d points filled | quality %s",
                out.attrs["velocity_fill"]["n_filled"], len(res),
                out.attrs["velocity_fill"]["quality_counts"])
    return out
