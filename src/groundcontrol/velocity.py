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
import warnings

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
        raise ValueError(f"stations DataFrame missing column(s) {missing}; "
                         f"has {list(stations.columns)}")

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

    # Drop stations that CANNOT reach any target (triangle inequality about an anchor
    # target): a site-scale lookup against a continental network is otherwise dominated by
    # stations thousands of km away. Order-preserving, so the index tie-break is unchanged.
    flat_lon, flat_lat = lon_arr.ravel(), lat_arr.ravel()
    # NaN targets reach nothing
    finite = np.flatnonzero(np.isfinite(flat_lon) & np.isfinite(flat_lat))
    if slon.size and finite.size:
        alon, alat = flat_lon[finite[0]], flat_lat[finite[0]]
        # nanmax over the targets as-is: NaN targets yield NaN distance and are ignored,
        # so no boolean-index copy of a multi-million-point array is needed.
        reach = radius_km + float(np.nanmax(_haversine_km(alon, alat, flat_lon, flat_lat)))
        near = _haversine_km(alon, alat, slon, slat) <= reach
        if not near.all():
            slon, slat, svel = slon[near], slat[near], svel[near]
            sids = sids[near] if sids is not None else None

    # Block-vectorized selection + combine. Identical results to the per-point reference
    # (:func:`_lookup_one`, kept as the readable definition and the tie-edge fallback);
    # the block path exists because altimetry callers look up millions of points at once
    # (a per-point Python loop costs ~60 us/pt -> ~15 min for a 14 M-point granule set).
    frames = [_lookup_block(flat_lon[s], flat_lat[s], slon, slat, sids, svel,
                            radius_km, min_stations, max_stations, method, idw_power,
                            spread_threshold_mm_yr)
              for s in _blocks(flat_lon.size, slon.size)]
    out = (pd.concat(frames, ignore_index=True) if frames
           else pd.DataFrame.from_records([], columns=RESULT_COLUMNS))
    out["n_stations_used"] = out["n_stations_used"].astype("int64")
    out["quality"] = out["quality"].astype("string")
    out["nearest_sta"] = out["nearest_sta"].astype("string")
    return out


#: Target-point block size is chosen so the (n_points x n_stations) distance matrix stays
#: around this many elements (~64 MB float64) regardless of network size.
_BLOCK_ELEMENTS = 8_000_000
_BLOCK_MAX_POINTS = 200_000


def _blocks(n_points: int, n_stations: int):
    """Slices partitioning ``n_points`` into memory-bounded vectorized blocks."""
    if n_points == 0:
        return []
    step = min(_BLOCK_MAX_POINTS, max(1, _BLOCK_ELEMENTS // max(n_stations, 1)))
    return [slice(i, min(i + step, n_points)) for i in range(0, n_points, step)]


def _lookup_block(tlon, tlat, slon, slat, sids, svel, radius_km, min_stations,
                  max_stations, method, idw_power, spread_threshold_mm_yr) -> pd.DataFrame:
    """Vectorized :func:`_lookup_one` over a block of target points.

    Mirrors the reference semantics exactly: nearest ``<= max_stations`` inside
    ``radius_km``, ties broken by station order (what the reference's stable argsort
    does), component-wise median/IDW combine, ddof=1 spread, and the same quality
    ladder (``low_density`` > ``single_station`` > ``spread_warning`` > ``ok``).
    """
    n, m = tlon.size, slon.size
    if m == 0 or n == 0:            # no usable station anywhere -> the reference's `empty` row
        return pd.DataFrame({
            "vel_e": np.full(n, np.nan), "vel_n": np.full(n, np.nan),
            "vel_u": np.full(n, np.nan), "vel_spread_h": np.full(n, np.nan),
            "vel_spread_u": np.full(n, np.nan),
            "n_stations_used": np.zeros(n, dtype="int64"),
            "nearest_dist_km": np.full(n, np.nan),
            "nearest_sta": pd.array([pd.NA] * n, dtype="string"),
            "quality": pd.array([QUALITY_LOW_DENSITY] * n, dtype="string"),
        }, columns=list(RESULT_COLUMNS))

    dist = _haversine_km(tlon[:, None], tlat[:, None], slon[None, :], slat[None, :])
    inside = np.where(dist <= radius_km, dist, np.inf)     # outside radius -> never selected
    k = int(min(max_stations, m))

    if m > k:  # noqa: SIM108 - ternary here is a 100-column one-liner
        cand = np.argpartition(inside, k - 1, axis=1)[:, :k]
    else:
        cand = np.tile(np.arange(m), (n, 1))
    d_cand = np.take_along_axis(inside, cand, axis=1)
    order = np.lexsort((cand, d_cand), axis=1)             # distance, then station index
    idx = np.take_along_axis(cand, order, axis=1)
    d = np.take_along_axis(d_cand, order, axis=1)
    valid = np.isfinite(d)
    n_used = valid.sum(axis=1).astype("int64")

    # argpartition is not stable, so a distance tie straddling the k-th slot could pick a
    # different (equally near) station than the reference. Vanishingly rare with float
    # haversine distances, but resolve those rows exactly rather than assume.
    if m > k:
        kth = d[:, -1]
        amb = np.isfinite(kth) & ((inside <= kth[:, None]).sum(axis=1) > k)
        if amb.any():
            for i in np.flatnonzero(amb):
                ref = np.argsort(inside[i], kind="stable")[:k]
                idx[i], d[i] = ref, inside[i][ref]
            valid = np.isfinite(d)
            n_used = valid.sum(axis=1).astype("int64")

    sel_vel = np.where(valid[:, :, None], svel[idx], np.nan)      # (n, k, 3)
    with np.errstate(invalid="ignore", divide="ignore"), \
            warnings.catch_warnings():                            # all-NaN / ddof>=count rows
        warnings.simplefilter("ignore", RuntimeWarning)
        if method == "median":
            comb = np.nanmedian(sel_vel, axis=1)
        else:
            w = np.where(valid, 1.0 / np.maximum(d, _IDW_EPS_KM) ** float(idw_power), 0.0)
            comb = (w[:, :, None] * np.nan_to_num(sel_vel)).sum(axis=1) / w.sum(axis=1)[:, None]
        spread = np.nanstd(sel_vel, axis=1, ddof=1)               # (n, 3); NaN if < 2 used
    spread_h = np.hypot(spread[:, 0], spread[:, 1])
    spread_u = spread[:, 2]

    quality = np.full(n, QUALITY_OK, dtype=object)
    over_spread = np.isfinite(spread_h) & (spread_h * 1e3 > spread_threshold_mm_yr)
    quality[over_spread] = QUALITY_SPREAD_WARNING
    quality[n_used == 1] = QUALITY_SINGLE_STATION
    # `| n_used == 0` matters only when a caller passes min_stations=0: the reference returns
    # its `empty` (NaN + low_density) whenever NO station is inside the radius, before the
    # min_stations test — without this that row would come back quality='ok'.
    dropped = (n_used < min_stations) | (n_used == 0)             # -> NaN velocity, tier-2 fallback
    quality[dropped] = QUALITY_LOW_DENSITY
    comb[dropped] = np.nan
    spread_h[dropped] = np.nan
    spread_u[dropped] = np.nan

    found = n_used > 0
    nearest_dist = np.where(found, d[:, 0], np.nan)
    nearest_sta = np.array([pd.NA] * n, dtype=object)
    if sids is not None:
        nearest_sta[found] = sids[idx[found, 0]]

    return pd.DataFrame({
        "vel_e": comb[:, 0], "vel_n": comb[:, 1], "vel_u": comb[:, 2],
        "vel_spread_h": spread_h, "vel_spread_u": spread_u,
        "n_stations_used": n_used,
        "nearest_dist_km": nearest_dist,
        "nearest_sta": pd.array(nearest_sta, dtype="string"),
        "quality": pd.array(quality, dtype="string"),
    }, columns=list(RESULT_COLUMNS))


def _lookup_one(tlon, tlat, slon, slat, svel, sids, radius_km, min_stations,
                max_stations, method, idw_power, spread_threshold_mm_yr) -> dict:
    """Single-target selection + combine (memory-safe: one M-vector, no N×M matrix).

    The readable REFERENCE definition of the lookup semantics. :func:`interpolate_velocity`
    runs the block-vectorized :func:`_lookup_block` instead; ``tests/test_velocity.py``
    asserts the two agree exactly, so this stays the specification.
    """
    empty = {"vel_e": np.nan, "vel_n": np.nan, "vel_u": np.nan, "vel_spread_h": np.nan,
             "vel_spread_u": np.nan, "n_stations_used": 0, "nearest_dist_km": np.nan,
             "nearest_sta": pd.NA, "quality": QUALITY_LOW_DENSITY}
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

    return {"vel_e": float(ve), "vel_n": float(vn), "vel_u": float(vu),
            "vel_spread_h": spread_h, "vel_spread_u": spread_u, "n_stations_used": n_used,
            "nearest_dist_km": nearest_dist, "nearest_sta": nearest_sta, "quality": quality}


#: ``fill_velocities(per_row='auto')`` attaches the per-row lookup table up to this many
#: rows. Above it the table is ~1 kB/row of Python objects (a 14 M-point altimetry set
#: would cost >10 GB) and nothing consumes it, so only the summary is kept.
PER_ROW_PROVENANCE_MAX = 100_000


def fill_velocities(gdf, stations: pd.DataFrame, *,
                    vel_cols=("vel_e", "vel_n", "vel_u"),
                    quality_col: str | None = None, per_row="auto", **kwargs):
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
    same trio). Returns a copy with ``vel_cols`` filled and a summary — plus, for modest
    inputs, the full per-row lookup table — attached at ``out.attrs['velocity_fill']``
    (provenance without widening the schema). If ``quality_col`` is given, the per-row
    quality flag is also written to that column. Original rows/columns are otherwise
    untouched.

    ``per_row`` controls that per-row table: ``'auto'`` (default) keeps it up to
    :data:`PER_ROW_PROVENANCE_MAX` rows, ``True`` always, ``False`` never. The cap keeps a
    million-point altimetry lookup from spending more memory on unread provenance than on
    the data; ``quality_col`` gives the per-point flag at full scale for free.
    """
    # Validate rather than fall back on truthiness: a stray value ('false', 'always', 1)
    # would silently re-enable the very payload the cap exists to prevent.
    if per_row != "auto" and not isinstance(per_row, bool):
        raise ValueError(f"per_row must be 'auto', True or False; got {per_row!r}")
    out = gdf.copy()
    if len(out) == 0:
        out.attrs["velocity_fill"] = {"n_total": 0, "quality_counts": {}}
        return out
    lon = out.geometry.x.to_numpy(dtype="float64")
    lat = out.geometry.y.to_numpy(dtype="float64")
    res = interpolate_velocity(lon, lat, stations, **kwargs)
    for col, src in zip(vel_cols, ("vel_e", "vel_n", "vel_u"), strict=True):
        out[col] = res[src].to_numpy(dtype="float64")
    if quality_col is not None:
        out[quality_col] = res["quality"].to_numpy()
    counts = res["quality"].value_counts().to_dict()
    keep_rows = len(res) <= PER_ROW_PROVENANCE_MAX if per_row == "auto" else per_row
    out.attrs["velocity_fill"] = {
        "n_total": len(res),
        "n_filled": int(res["vel_e"].notna().sum()),
        "quality_counts": {str(k): int(v) for k, v in counts.items()},
        "per_row": res.to_dict("records") if keep_rows else None,
    }
    logger.info("fill_velocities: %d/%d points filled | quality %s",
                out.attrs["velocity_fill"]["n_filled"], len(res),
                out.attrs["velocity_fill"]["quality_counts"])
    return out
