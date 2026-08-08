"""Observed-velocity lookup tests (groundcontrol.velocity) — all offline/synthetic.

The lookup is source-agnostic (a plain station DataFrame with lon/lat/vel_e/n/u), so
these tests use fabricated stations with exactly-known velocities and distances rather
than the live MIDAS network. The fault-awareness is exercised with a synthetic
two-block "fault" (no real geometry hardcoded anywhere) and the tier-1 -> propagate_epoch
hand-off is checked end-to-end against the production kernel.
"""

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

import geopandas as gpd

from groundcontrol import velocity as V
from groundcontrol.crs import propagate_epoch


def _stations(records):
    """records: list of dicts with sta, lon, lat, vel_e, vel_n, vel_u (m/yr)."""
    return pd.DataFrame(records)


def _uniform(lon0, lat0, dlats, ve, vn, vu, prefix="S"):
    """Stations strung north of (lon0, lat0) at the given dlat offsets, same velocity."""
    return _stations([
        {"sta": f"{prefix}{i}", "lon": lon0, "lat": lat0 + dl,
         "vel_e": ve, "vel_n": vn, "vel_u": vu}
        for i, dl in enumerate(dlats)
    ])


# ---------------------------------------------------------------------------
# distance metric
# ---------------------------------------------------------------------------

def test_haversine_meridian_distance():
    # 0.1 deg of latitude along a meridian ~ 11.12 km (R_mean * dlat_rad)
    d = V._haversine_km(-115.15, 36.10, -115.15, 36.20)
    assert d == pytest.approx(11.12, abs=0.05)
    # a full degree ~ 111.2 km
    assert V._haversine_km(-115.15, 36.10, -115.15, 37.10) == pytest.approx(111.19, abs=0.1)


# ---------------------------------------------------------------------------
# core interpolation
# ---------------------------------------------------------------------------

def test_known_velocity_synthetic_exact():
    """Uniform network -> the exact velocity, zero spread, quality ok (no injected error)."""
    st = _uniform(-115.15, 36.10, [-0.1, -0.05, 0.05, 0.1], 0.020, -0.010, 0.001)
    r = V.interpolate_velocity(-115.15, 36.10, st).iloc[0]
    assert (r["vel_e"], r["vel_n"], r["vel_u"]) == pytest.approx((0.020, -0.010, 0.001))
    assert r["vel_spread_h"] == pytest.approx(0.0, abs=1e-15)
    assert r["n_stations_used"] == 4
    assert r["quality"] == V.QUALITY_OK


def test_nearest_station_selection_within_radius():
    """Radius is a hard cutoff; selection is the nearest-N inside it, sorted by distance."""
    st = _stations([
        {"sta": "AT0", "lon": -115.15, "lat": 36.10, "vel_e": 0.02, "vel_n": 0.0, "vel_u": 0.0},   # 0 km
        {"sta": "NR1", "lon": -115.15, "lat": 36.20, "vel_e": 0.02, "vel_n": 0.0, "vel_u": 0.0},   # ~11 km
        {"sta": "MD2", "lon": -115.15, "lat": 36.55, "vel_e": 0.02, "vel_n": 0.0, "vel_u": 0.0},   # ~50 km
        {"sta": "FAR", "lon": -115.15, "lat": 37.10, "vel_e": 0.99, "vel_n": 0.0, "vel_u": 0.0},   # ~111 km (out)
    ])
    r = V.interpolate_velocity(-115.15, 36.10, st, radius_km=75.0).iloc[0]
    assert r["n_stations_used"] == 3            # FAR excluded by radius
    assert r["nearest_sta"] == "AT0"
    assert r["nearest_dist_km"] == pytest.approx(0.0, abs=1e-6)
    assert r["vel_e"] == pytest.approx(0.02)    # FAR's 0.99 never enters


def test_max_stations_caps_selection():
    st = _uniform(-115.15, 36.10, [0.0, 0.05, 0.1, 0.15, 0.2, 0.25], 0.02, -0.01, 0.0)
    r = V.interpolate_velocity(-115.15, 36.10, st, max_stations=2).iloc[0]
    assert r["n_stations_used"] == 2


def test_median_vs_idw_combine():
    """Median = component median; IDW pulls toward the nearer station (hand-checked)."""
    st = _stations([
        {"sta": "A", "lon": -115.15, "lat": 36.15, "vel_e": 0.030, "vel_n": 0.0, "vel_u": 0.0},  # ~5.56 km
        {"sta": "B", "lon": -115.15, "lat": 36.30, "vel_e": 0.010, "vel_n": 0.0, "vel_u": 0.0},  # ~22.24 km
    ])
    med = V.interpolate_velocity(-115.15, 36.10, st, min_stations=2, method="median").iloc[0]
    idw = V.interpolate_velocity(-115.15, 36.10, st, min_stations=2, method="idw").iloc[0]
    assert med["vel_e"] == pytest.approx(0.020)          # (0.030 + 0.010) / 2
    dA = V._haversine_km(-115.15, 36.10, -115.15, 36.15)
    dB = V._haversine_km(-115.15, 36.10, -115.15, 36.30)
    wA, wB = 1.0 / dA, 1.0 / dB
    expect = (wA * 0.030 + wB * 0.010) / (wA + wB)
    assert idw["vel_e"] == pytest.approx(expect, rel=1e-6)
    assert idw["vel_e"] > med["vel_e"]                   # weighted toward nearer A


def test_spread_gate_flags_boundary_straddle():
    """A selection spanning a synthetic velocity discontinuity trips the spread gate;
    a coherent selection does not. No fault geometry is encoded — only the spread."""
    coherent = _stations([
        {"sta": "C0", "lon": -115.15, "lat": 36.10, "vel_e": -0.0150, "vel_n": -0.0084, "vel_u": 0.0},
        {"sta": "C1", "lon": -115.15, "lat": 36.20, "vel_e": -0.0155, "vel_n": -0.0088, "vel_u": 0.0},
        {"sta": "C2", "lon": -115.05, "lat": 36.15, "vel_e": -0.0149, "vel_n": -0.0089, "vel_u": 0.0},
    ])
    r_ok = V.interpolate_velocity(-115.10, 36.15, coherent).iloc[0]
    assert r_ok["quality"] == V.QUALITY_OK
    assert r_ok["vel_spread_h"] * 1e3 < V.DEFAULT_SPREAD_THRESHOLD_MM_YR

    # two blocks (~2.5 cm/yr apart) — like sampling both sides of a locked fault
    straddle = _stations([
        {"sta": "W0", "lon": -115.15, "lat": 36.12, "vel_e": -0.030, "vel_n": 0.010, "vel_u": 0.0},
        {"sta": "W1", "lon": -115.15, "lat": 36.18, "vel_e": -0.030, "vel_n": 0.010, "vel_u": 0.0},
        {"sta": "E0", "lon": -115.05, "lat": 36.12, "vel_e": -0.005, "vel_n": -0.010, "vel_u": 0.0},
        {"sta": "E1", "lon": -115.05, "lat": 36.18, "vel_e": -0.005, "vel_n": -0.010, "vel_u": 0.0},
    ])
    r_bad = V.interpolate_velocity(-115.10, 36.15, straddle).iloc[0]
    assert r_bad["quality"] == V.QUALITY_SPREAD_WARNING
    assert r_bad["vel_spread_h"] * 1e3 > V.DEFAULT_SPREAD_THRESHOLD_MM_YR
    assert np.isfinite(r_bad["vel_e"])  # still returned — caller decides


def test_low_density_returns_nan_for_plate_fallback():
    """Too few stations in radius -> NaN velocity + low_density (caller -> tier-2 model)."""
    st = _uniform(-115.15, 36.10, [0.0, 0.05], 0.02, -0.01, 0.0)  # only 2 stations
    r = V.interpolate_velocity(-115.15, 36.10, st, min_stations=3).iloc[0]
    assert r["quality"] == V.QUALITY_LOW_DENSITY
    assert np.isnan(r["vel_e"]) and np.isnan(r["vel_n"]) and np.isnan(r["vel_u"])
    assert r["n_stations_used"] == 2  # reported even though below the minimum

    # nothing at all within radius
    r2 = V.interpolate_velocity(0.0, 0.0, st).iloc[0]
    assert r2["quality"] == V.QUALITY_LOW_DENSITY and r2["n_stations_used"] == 0


def test_single_station_flag():
    st = _uniform(-115.15, 36.10, [0.0], 0.02, -0.01, 0.0)
    r = V.interpolate_velocity(-115.15, 36.10, st, min_stations=1).iloc[0]
    assert r["quality"] == V.QUALITY_SINGLE_STATION
    assert r["vel_e"] == pytest.approx(0.02)
    assert np.isnan(r["vel_spread_h"])  # no spread from one station


def test_array_input_one_row_per_target():
    st = _uniform(-115.15, 36.10, [-0.1, -0.05, 0.05, 0.1], 0.02, -0.01, 0.0)
    res = V.interpolate_velocity([-115.15, 0.0], [36.10, 0.0], st)
    assert len(res) == 2
    assert res.iloc[0]["quality"] == V.QUALITY_OK
    assert res.iloc[1]["quality"] == V.QUALITY_LOW_DENSITY  # (0,0) far from Nevada


def test_missing_station_columns_raises():
    with pytest.raises(ValueError, match="missing column"):
        V.interpolate_velocity(-115.15, 36.10, pd.DataFrame({"lon": [-115.15], "lat": [36.1]}))


def test_unknown_method_raises():
    st = _uniform(-115.15, 36.10, [0.0, 0.05, 0.1], 0.02, -0.01, 0.0)
    with pytest.raises(ValueError, match="method"):
        V.interpolate_velocity(-115.15, 36.10, st, method="kriging")


# ---------------------------------------------------------------------------
# fill_velocities -> propagate_epoch hand-off (the tier-1 wiring)
# ---------------------------------------------------------------------------

def _gdf(points, coord_epoch=2022.5, h=600.0):
    n = len(points)
    return gpd.GeoDataFrame(
        {"height": [h] * n, "coord_epoch": [coord_epoch] * n,
         "vel_e": [np.nan] * n, "vel_n": [np.nan] * n, "vel_u": [np.nan] * n},
        geometry=[Point(lon, lat) for lon, lat in points],
        crs="EPSG:7912",  # geographic ITRF2014
    )


def test_fill_velocities_populates_columns_and_attrs():
    st = _uniform(-115.15, 36.10, [-0.1, -0.05, 0.05, 0.1], 0.020, -0.010, 0.001)
    gdf = _gdf([(-115.15, 36.10), (0.0, 0.0)])  # one near stations, one far
    out = V.fill_velocities(gdf, st)
    assert out["vel_e"].iloc[0] == pytest.approx(0.020)
    assert np.isnan(out["vel_e"].iloc[1])  # far point stays NaN -> plate fallback
    rep = out.attrs["velocity_fill"]
    assert rep["n_total"] == 2 and rep["n_filled"] == 1
    assert rep["quality_counts"].get(V.QUALITY_LOW_DENSITY) == 1


def test_fill_velocities_writes_quality_column_when_requested():
    st = _uniform(-115.15, 36.10, [-0.1, 0.0, 0.1], 0.02, -0.01, 0.0)
    out = V.fill_velocities(_gdf([(-115.15, 36.10)]), st, quality_col="vel_quality")
    assert out["vel_quality"].iloc[0] == V.QUALITY_OK


def test_filled_velocity_drives_propagate_epoch_tier1():
    """The end-to-end reason this exists: filled per-point velocity is consumed as
    propagate_epoch's tier-1 (per_point) source; unfilled rows fall through as no-op."""
    st = _uniform(-115.15, 36.10, [-0.1, -0.05, 0.05, 0.1], 0.020, -0.010, 0.0)
    gdf = V.fill_velocities(_gdf([(-115.15, 36.10), (0.0, 0.0)]), st)
    out = propagate_epoch(gdf, target_epoch=2010.0)  # no plate_model -> tier 1 or no-op
    rep = out.attrs["epoch_propagation"]
    assert rep["models"]["per_point"] == 1   # the filled LV point
    assert rep["models"]["none"] == 1        # the far (NaN) point -> no-op
    # displacement ~ |v_h| * dt : hypot(0.02, 0.01) * (2022.5 - 2010.0) ~ 0.28 m
    assert rep["max_applied_displacement_m"] == pytest.approx(
        np.hypot(0.020, 0.010) * 12.5, rel=1e-3)


def test_fill_velocities_empty_gdf():
    st = _uniform(-115.15, 36.10, [0.0, 0.05, 0.1], 0.02, -0.01, 0.0)
    out = V.fill_velocities(_gdf([]), st)
    assert len(out) == 0 and out.attrs["velocity_fill"]["n_total"] == 0


# ---------------------------------------------------------------------------
# block-vectorized path == per-point reference (_lookup_one is the specification)
# ---------------------------------------------------------------------------

def _reference_interpolate(lon, lat, stations, **kw):
    """The pre-vectorization implementation: _lookup_one per target point."""
    lon_arr = np.atleast_1d(np.asarray(lon, dtype="float64")).ravel()
    lat_arr = np.atleast_1d(np.asarray(lat, dtype="float64")).ravel()
    st = stations.reset_index(drop=True)
    cols = ("lon", "lat", "vel_e", "vel_n", "vel_u")
    finite = np.isfinite(st[list(cols)].apply(pd.to_numeric, errors="coerce").to_numpy()).all(axis=1)
    st = st.loc[finite]
    slon = st["lon"].to_numpy(dtype="float64")
    slat = st["lat"].to_numpy(dtype="float64")
    svel = np.column_stack([st[c].to_numpy(dtype="float64") for c in ("vel_e", "vel_n", "vel_u")])
    sids = st["sta"].astype("string").to_numpy() if "sta" in st.columns else None
    rows = [V._lookup_one(a, b, slon, slat, svel, sids,
                          kw.get("radius_km", V.DEFAULT_RADIUS_KM),
                          kw.get("min_stations", V.DEFAULT_MIN_STATIONS),
                          kw.get("max_stations", V.DEFAULT_MAX_STATIONS),
                          kw.get("method", "median"), kw.get("idw_power", 1.0),
                          kw.get("spread_threshold_mm_yr", V.DEFAULT_SPREAD_THRESHOLD_MM_YR))
            for a, b in zip(lon_arr, lat_arr)]
    out = pd.DataFrame.from_records(rows, columns=V.RESULT_COLUMNS)
    out["n_stations_used"] = out["n_stations_used"].astype("int64")
    out["quality"] = out["quality"].astype("string")
    out["nearest_sta"] = out["nearest_sta"].astype("string")
    return out


def _random_network(rng, n_sta, lon0=-115.15, lat0=36.10, spread_deg=1.5):
    """Stations scattered around a center, velocities in two blocks (a synthetic fault)."""
    lon = lon0 + rng.uniform(-spread_deg, spread_deg, n_sta)
    lat = lat0 + rng.uniform(-spread_deg, spread_deg, n_sta)
    east_block = lon > lon0
    return _stations([
        {"sta": f"R{i:03d}", "lon": lon[i], "lat": lat[i],
         "vel_e": (0.020 if east_block[i] else 0.012) + rng.normal(0, 5e-4),
         "vel_n": (-0.010 if east_block[i] else -0.004) + rng.normal(0, 5e-4),
         "vel_u": rng.normal(0, 1e-3)}
        for i in range(n_sta)
    ])


@pytest.mark.parametrize("n_sta,kw", [
    (60, {}),                                   # dense network, defaults
    (60, {"method": "idw"}),                    # IDW combine
    (60, {"max_stations": 3, "radius_km": 40}), # tight selection (partition boundary busy)
    (12, {"radius_km": 25}),                    # sparse -> low_density rows
    (4, {"min_stations": 1, "radius_km": 15}),  # single_station rows
    (1, {"min_stations": 1}),                   # one station
    (12, {"min_stations": 0, "radius_km": 20}), # min_stations=0: empty selections stay low_density
])
def test_block_path_matches_reference_loop(n_sta, kw):
    rng = np.random.default_rng(11)
    st = _random_network(rng, n_sta)
    lon = -115.15 + rng.uniform(-2.0, 2.0, 500)   # some targets fall outside every radius
    lat = 36.10 + rng.uniform(-2.0, 2.0, 500)
    pd.testing.assert_frame_equal(V.interpolate_velocity(lon, lat, st, **kw),
                                  _reference_interpolate(lon, lat, st, **kw))


def test_block_path_matches_reference_with_duplicate_stations():
    """Co-located stations make distance ties real — the argpartition tie-break guard."""
    rng = np.random.default_rng(3)
    st = _random_network(rng, 20)
    st = pd.concat([st, st.assign(sta=st["sta"] + "b")], ignore_index=True)  # exact ties
    lon = -115.15 + rng.uniform(-1.0, 1.0, 300)
    lat = 36.10 + rng.uniform(-1.0, 1.0, 300)
    for kw in ({}, {"max_stations": 5}, {"max_stations": 2, "min_stations": 1}):
        pd.testing.assert_frame_equal(V.interpolate_velocity(lon, lat, st, **kw),
                                      _reference_interpolate(lon, lat, st, **kw))


def test_block_path_matches_reference_across_block_boundary(monkeypatch):
    """Blocking is an implementation detail: results must not depend on the block size."""
    rng = np.random.default_rng(7)
    st = _random_network(rng, 30)
    lon = -115.15 + rng.uniform(-1.0, 1.0, 250)
    lat = 36.10 + rng.uniform(-1.0, 1.0, 250)
    ref = _reference_interpolate(lon, lat, st)
    for block in (1, 7, 250, 10_000):
        monkeypatch.setattr(V, "_BLOCK_MAX_POINTS", block)
        monkeypatch.setattr(V, "_BLOCK_ELEMENTS", block)
        pd.testing.assert_frame_equal(V.interpolate_velocity(lon, lat, st), ref)


def test_no_usable_stations_matches_reference():
    st = _stations([{"sta": "N0", "lon": np.nan, "lat": 36.1,
                     "vel_e": 0.02, "vel_n": -0.01, "vel_u": 0.0}])  # dropped as non-finite
    lon, lat = np.array([-115.15, -115.10]), np.array([36.10, 36.05])
    pd.testing.assert_frame_equal(V.interpolate_velocity(lon, lat, st),
                                  _reference_interpolate(lon, lat, st))


def test_per_row_provenance_capped_by_default(monkeypatch):
    st = _uniform(-115.15, 36.10, [-0.1, -0.05, 0.05, 0.1], 0.020, -0.010, 0.001)
    gdf = _gdf([(-115.15, 36.10)] * 5)
    assert V.fill_velocities(gdf, st).attrs["velocity_fill"]["per_row"] is not None
    monkeypatch.setattr(V, "PER_ROW_PROVENANCE_MAX", 2)
    rep = V.fill_velocities(gdf, st).attrs["velocity_fill"]
    assert rep["per_row"] is None and rep["n_total"] == 5     # summary still exact
    assert V.fill_velocities(gdf, st, per_row=True).attrs["velocity_fill"]["per_row"] is not None


def test_station_prefilter_keeps_nan_targets_and_far_points_exact():
    """The reach prefilter must not change any answer: NaN targets, far targets, and a
    network with distant stations that can never be selected."""
    rng = np.random.default_rng(19)
    st = pd.concat([_random_network(rng, 25),                              # local
                    _random_network(rng, 25, lon0=10.0, lat0=50.0)],       # continent away
                   ignore_index=True)
    lon = np.concatenate([-115.15 + rng.uniform(-0.5, 0.5, 200), [np.nan, 10.0, np.nan]])
    lat = np.concatenate([36.10 + rng.uniform(-0.5, 0.5, 200), [36.1, 50.0, np.nan]])
    pd.testing.assert_frame_equal(V.interpolate_velocity(lon, lat, st),
                                  _reference_interpolate(lon, lat, st))


@pytest.mark.parametrize("bad", ["false", "always", 1, 0, None])
def test_per_row_rejects_unsupported_values(bad):
    """A stray truthy value must not silently re-enable the per-row payload (PR #15 review)."""
    st = _uniform(-115.15, 36.10, [-0.1, 0.0, 0.1], 0.02, -0.01, 0.0)
    with pytest.raises(ValueError, match="per_row must be"):
        V.fill_velocities(_gdf([(-115.15, 36.10)]), st, per_row=bad)
