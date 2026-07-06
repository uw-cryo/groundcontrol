"""Stage-2 epoch-propagation kernel tests (crs.py; crs_implementation.md §1/§8).

All offline. The numerically-verifiable core (§8 test 6): a synthetic point with a
known ENU velocity over a known Δt gives an exact displacement, cross-checked
against an *independent* ENU→ECEF rotation oracle (pyproj EPSG:4979↔EPSG:4978,
pure cartesian conversion — no grids, no network), and a round-trip A→B→A returns
the original. Also covers the velocity priority ladder (per-point → plate model →
none+bound), the fail-loud guards, and the frame_epoch QC.
"""

import numpy as np
import pandas as pd
import pyproj
import pytest
from shapely.geometry import Point

import geopandas as gpd

from groundcontrol.crs import (
    EulerPoleModel,
    PLATE_MOTION_RATE_BOUND,
    check_frame_epoch_reduced,
    ecef_to_enu,
    enu_to_ecef,
    propagate_epoch,
)

# WGS84 geographic-3D <-> geocentric: a pure conversion (no datum shift/grids).
_TO_ECEF = pyproj.Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)


def _gdf(lon=-115.1, lat=36.1, h=700.0, ve=np.nan, vn=np.nan, vu=np.nan,
         coord_epoch=2015.0, crs="EPSG:4979", frame_epoch=np.nan, n=1,
         transform_id=True):
    data = {
        "height": np.full(n, float(h)),
        "coord_epoch": np.full(n, float(coord_epoch)),
        "frame_epoch": np.full(n, float(frame_epoch)),
        "vel_e": np.full(n, float(ve)),
        "vel_n": np.full(n, float(vn)),
        "vel_u": np.full(n, float(vu)),
    }
    if transform_id:
        data["transform_id"] = pd.array([pd.NA] * n, dtype="string")
    return gpd.GeoDataFrame(data, geometry=[Point(lon, lat)] * n, crs=crs)


def _ecef(lon, lat, h):
    lon, lat, h = (np.atleast_1d(np.asarray(v, dtype="float64")) for v in (lon, lat, h))
    x, y, z = _TO_ECEF.transform(lon, lat, h)
    return np.stack([np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(z)], axis=-1)


def _sep_m(gdf_row_lon, gdf_row_lat, gdf_row_h, lon2, lat2, h2):
    """3D separation (m) between two geodetic positions, via ECEF."""
    p = _ecef(gdf_row_lon, gdf_row_lat, gdf_row_h)
    q = _ecef(lon2, lat2, h2)
    return float(np.linalg.norm(p - q, axis=-1).max())


def _oracle(lon, lat, h, d_e, d_n, d_u):
    """Independent displaced position: rotate ENU displacement to ECEF, add, invert."""
    X, Y, Z = _TO_ECEF.transform(lon, lat, h)
    dX, dY, dZ = enu_to_ecef(d_e, d_n, d_u, lon, lat)
    to_geo = pyproj.Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)
    return to_geo.transform(X + dX, Y + dY, Z + dZ)


# ---------------------------------------------------------------------------
# ENU<->ECEF rotation helpers (the §8-test-6 oracle machinery)
# ---------------------------------------------------------------------------

def test_enu_ecef_rotation_roundtrips():
    e, n, u, lon, lat = 0.2, -0.1, 0.05, -115.1, 36.1
    x, y, z = enu_to_ecef(e, n, u, lon, lat)
    e2, n2, u2 = ecef_to_enu(x, y, z, lon, lat)
    assert (e2, n2, u2) == pytest.approx((e, n, u), abs=1e-15)
    # ENU-applied-as-ECEF (skipping the rotation) is a *different* vector (the caught bug)
    assert np.hypot.reduce([x - e, y - n, z - u]) > 0.1


# ---------------------------------------------------------------------------
# core kernel: exact displacement vs the ENU->ECEF oracle (§8 test 6)
# ---------------------------------------------------------------------------

def test_known_velocity_exact_displacement_matches_ecef_oracle():
    ve, vn, vu, dt = 0.020, -0.010, 0.005, 10.0  # -> 0.20, -0.10, 0.05 m
    g = _gdf(ve=ve, vn=vn, vu=vu, coord_epoch=2015.0)
    out = propagate_epoch(g, target_epoch=2015.0 + dt)
    lon_o, lat_o, h_o = _oracle(-115.1, 36.1, 700.0, ve * dt, vn * dt, vu * dt)
    sep = _sep_m(out.geometry.x.iloc[0], out.geometry.y.iloc[0], out["height"].iloc[0],
                 lon_o, lat_o, h_o)
    assert sep < 1e-4, f"kernel vs ENU->ECEF oracle differ by {sep*1000:.4f} mm"
    # applied displacement magnitude is exactly ||vel*dt||
    rep = out.attrs["epoch_propagation"]
    assert rep["max_applied_displacement_m"] == pytest.approx(
        np.sqrt((ve * dt) ** 2 + (vn * dt) ** 2 + (vu * dt) ** 2), abs=1e-9)
    assert rep["n_propagated"] == 1 and rep["models"]["per_point"] == 1


@pytest.mark.parametrize("axis,ve,vn,vu", [
    ("east", 0.03, 0.0, 0.0),
    ("north", 0.0, 0.03, 0.0),
    ("up", 0.0, 0.0, 0.03),
])
def test_single_axis_velocity_moves_only_that_axis(axis, ve, vn, vu):
    g = _gdf(ve=ve, vn=vn, vu=vu, coord_epoch=2010.0)
    out = propagate_epoch(g, target_epoch=2020.0)  # dt = 10 yr -> 0.3 m
    dlon = out.geometry.x.iloc[0] - (-115.1)
    dlat = out.geometry.y.iloc[0] - 36.1
    dh = out["height"].iloc[0] - 700.0
    if axis == "east":
        assert abs(dlat) < 1e-12 and abs(dh) < 1e-12 and dlon > 0
    elif axis == "north":
        assert abs(dlon) < 1e-12 and abs(dh) < 1e-12 and dlat > 0
    else:  # up is exact — no curvature involved
        assert abs(dlon) < 1e-12 and abs(dlat) < 1e-12
        assert dh == pytest.approx(0.3, abs=1e-12)


def test_zero_velocity_control_no_move():
    g = _gdf(ve=0.0, vn=0.0, vu=0.0, coord_epoch=2015.0)
    out = propagate_epoch(g, target_epoch=2025.0)
    assert out.geometry.x.iloc[0] == pytest.approx(-115.1, abs=1e-15)
    assert out.geometry.y.iloc[0] == pytest.approx(36.1, abs=1e-15)
    assert out["height"].iloc[0] == 700.0
    # a moved row (even by zero) still advances its coord_epoch
    assert out["coord_epoch"].iloc[0] == 2025.0


def test_roundtrip_a_b_a_returns_original():
    g = _gdf(ve=0.021, vn=-0.013, vu=0.004, coord_epoch=2013.0)
    fwd = propagate_epoch(g, target_epoch=2027.0)
    back = propagate_epoch(fwd, target_epoch=2013.0)  # velocities carry through
    sep = _sep_m(-115.1, 36.1, 700.0,
                 back.geometry.x.iloc[0], back.geometry.y.iloc[0], back["height"].iloc[0])
    assert sep < 1e-4, f"A->B->A closure {sep*1000:.5f} mm"
    assert back["coord_epoch"].iloc[0] == 2013.0


def test_coord_epoch_advanced_and_transform_id_chained():
    g = _gdf(ve=0.01, vn=0.0, vu=0.0, coord_epoch=2016.5)
    g["transform_id"] = pd.array(["land:EPSG:7912->EPSG:9989"], dtype="string")
    out = propagate_epoch(g, target_epoch=2020.0)
    assert out["coord_epoch"].iloc[0] == 2020.0
    tid = out["transform_id"].iloc[0]
    assert tid == "land:EPSG:7912->EPSG:9989+prop:per_point->2020"


# ---------------------------------------------------------------------------
# velocity priority ladder: per-point -> plate model -> none + bound
# ---------------------------------------------------------------------------

def test_no_velocity_no_model_is_noop_and_surfaces_bound():
    g = _gdf(coord_epoch=2015.0)  # all vel NaN
    with pytest.warns(UserWarning, match="velocity"):
        out = propagate_epoch(g, target_epoch=2025.0)
    # coordinates and coord_epoch untouched
    assert out.geometry.x.iloc[0] == -115.1 and out.geometry.y.iloc[0] == 36.1
    assert out["coord_epoch"].iloc[0] == 2015.0
    rep = out.attrs["epoch_propagation"]
    assert rep["n_propagated"] == 0 and rep["n_noop"] == 1 and rep["models"]["none"] == 1
    # velocity·Δt bound = |Δt| * PLATE_MOTION_RATE_BOUND (reporting-only)
    assert rep["max_residual_bound_m"] == pytest.approx(10.0 * PLATE_MOTION_RATE_BOUND)
    assert rep["residual_bound_m"][0] == pytest.approx(1.6)
    assert out["transform_id"].iloc[0] == "prop:noop"


def test_plate_model_fills_missing_velocity():
    # north-pole Euler pole -> purely eastward plate velocity
    model = EulerPoleModel(pole_lat_deg=90.0, pole_lon_deg=0.0,
                           rate_deg_per_myr=0.2, name="northpole")
    g = _gdf(coord_epoch=2015.0)  # no per-point velocity
    out = propagate_epoch(g, target_epoch=2025.0, plate_model=model)
    rep = out.attrs["epoch_propagation"]
    assert rep["n_propagated"] == 1 and rep["models"]["plate"] == 1
    assert rep["plate_model"] == "northpole"
    # moved eastward only (north-pole rotation), by ~v_e * 10 yr
    ve, vn, vu = (float(v[0]) for v in model.velocity_enu([-115.1], [36.1], [700.0]))
    assert abs(vn) < 1e-9 and abs(vu) < 1e-9 and ve > 0
    lon_o, lat_o, h_o = _oracle(-115.1, 36.1, 700.0, ve * 10, vn * 10, vu * 10)
    sep = _sep_m(out.geometry.x.iloc[0], out.geometry.y.iloc[0], out["height"].iloc[0],
                 lon_o, lat_o, h_o)
    assert sep < 1e-4
    assert out["transform_id"].iloc[0] == "prop:plate[northpole]->2025"


def test_per_point_velocity_takes_priority_over_plate_model():
    model = EulerPoleModel(90.0, 0.0, 5.0, name="fast")  # would move a lot
    g = _gdf(ve=0.0, vn=0.0, vu=0.0, coord_epoch=2015.0)  # explicit zero per-point
    out = propagate_epoch(g, target_epoch=2025.0, plate_model=model)
    assert out.geometry.x.iloc[0] == pytest.approx(-115.1, abs=1e-15)  # plate NOT used
    assert out.attrs["epoch_propagation"]["models"]["per_point"] == 1


# ---------------------------------------------------------------------------
# EulerPoleModel closed form (north pole -> zonal v_e = omega*N*cos(lat))
# ---------------------------------------------------------------------------

def test_euler_pole_northpole_closed_form():
    rate = 0.2  # deg/Myr
    model = EulerPoleModel(90.0, 0.0, rate)
    ve, vn, vu = (float(v[0]) for v in model.velocity_enu([-115.1], [36.1], [0.0]))
    a, rf = 6378137.0, 298.257222101  # GRS80 (EPSG:7912 default ellipsoid)
    f = 1.0 / rf
    e2 = f * (2.0 - f)
    phi = np.radians(36.1)
    N = a / np.sqrt(1.0 - e2 * np.sin(phi) ** 2)
    omega = np.radians(rate) * 1e-6
    assert ve == pytest.approx(omega * N * np.cos(phi), rel=1e-9)
    assert vn == pytest.approx(0.0, abs=1e-12)
    assert vu == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# fail-loud guards (§8 test 9)
# ---------------------------------------------------------------------------

def test_nan_coord_epoch_with_velocity_raises():
    g = _gdf(ve=0.02, vn=0.0, vu=0.0, coord_epoch=np.nan)
    with pytest.raises(ValueError, match="NaN"):
        propagate_epoch(g, target_epoch=2025.0)


def test_nan_coord_epoch_skip_policy_leaves_unpropagated():
    g = _gdf(ve=0.02, vn=0.0, vu=0.0, coord_epoch=np.nan)
    with pytest.warns(UserWarning):
        out = propagate_epoch(g, target_epoch=2025.0, on_nan_epoch="skip")
    assert out.geometry.x.iloc[0] == -115.1
    assert out.attrs["epoch_propagation"]["n_propagated"] == 0


def test_projected_crs_raises():
    g = _gdf(ve=0.02, vn=0.0, vu=0.0).to_crs("EPSG:32611")  # UTM 11N (projected)
    with pytest.raises(ValueError, match="projected"):
        propagate_epoch(g, target_epoch=2025.0)


def test_no_crs_and_no_source_crs_raises():
    g = _gdf(ve=0.02, vn=0.0, vu=0.0)
    g = g.set_crs(None, allow_override=True)
    with pytest.raises(ValueError, match="geographic CRS"):
        propagate_epoch(g, target_epoch=2025.0)


def test_no_crs_uses_source_crs():
    g = _gdf(ve=0.03, vn=0.0, vu=0.0, coord_epoch=2015.0)
    g = g.set_crs(None, allow_override=True)
    out = propagate_epoch(g, target_epoch=2025.0, source_crs="EPSG:7912")
    assert out.geometry.x.iloc[0] > -115.1  # moved east


def test_target_epoch_must_be_finite():
    with pytest.raises(ValueError, match="finite"):
        propagate_epoch(_gdf(ve=0.02, vn=0.0, vu=0.0), target_epoch=np.nan)


def test_partial_velocity_warns_and_is_treated_as_none():
    g = _gdf(ve=0.02, vn=np.nan, vu=np.nan, coord_epoch=2015.0)
    with pytest.warns(UserWarning, match="partial"):
        out = propagate_epoch(g, target_epoch=2025.0)
    assert out.attrs["epoch_propagation"]["n_propagated"] == 0


def test_empty_gdf():
    g = _gdf(n=0)
    out = propagate_epoch(g, target_epoch=2025.0)
    assert len(out) == 0
    assert out.attrs["epoch_propagation"]["n_total"] == 0


# ---------------------------------------------------------------------------
# frame_epoch QC (§1: plate-fixed positions must be reduced to frame_epoch)
# ---------------------------------------------------------------------------

def test_check_frame_epoch_reduced_flags_unreduced_platefixed():
    g = _gdf(frame_epoch=2010.0, coord_epoch=2012.0, crs="EPSG:6318")
    with pytest.warns(UserWarning, match="unreduced"):
        bad = check_frame_epoch_reduced(g)
    assert bool(bad[0]) is True


def test_check_frame_epoch_reduced_passes_reduced_and_dynamic():
    reduced = _gdf(frame_epoch=2010.0, coord_epoch=2010.0, crs="EPSG:6318")
    assert not check_frame_epoch_reduced(reduced).any()  # coord==frame: OK
    dynamic = _gdf(frame_epoch=np.nan, coord_epoch=2020.0)
    assert not check_frame_epoch_reduced(dynamic).any()  # NaN frame_epoch skipped


def test_check_frame_epoch_reduced_raise_mode():
    g = _gdf(frame_epoch=2010.0, coord_epoch=2012.0, crs="EPSG:6318")
    with pytest.raises(ValueError, match="unreduced"):
        check_frame_epoch_reduced(g, on_violation="raise")


# ---------------------------------------------------------------------------
# composition with stage 1 (offline: ITRF2014 -> propagate -> NAD83(2011) land)
# ---------------------------------------------------------------------------

def test_composes_with_land_horizontal_offline():
    from groundcontrol.crs import land_horizontal
    g = _gdf(ve=0.018, vn=-0.006, vu=0.002, coord_epoch=2018.0, crs="EPSG:7912")
    g["horizontal_crs"] = pd.array(["EPSG:7912"], dtype="string")
    # stage 2 first (still in ITRF2014), then stage 1 frame transform (coord_20 order)
    prop = propagate_epoch(g, target_epoch=2010.0)
    assert prop["coord_epoch"].iloc[0] == 2010.0
    # stage-2 moved the point by ~||vel||*8yr (a few cm)
    moved_m = _sep_m(-115.1, 36.1, 700.0,
                     prop.geometry.x.iloc[0], prop.geometry.y.iloc[0], prop["height"].iloc[0])
    assert 0.1 < moved_m < 0.3
    landed = land_horizontal(prop, target="EPSG:6318")  # tt = coord_epoch = 2010
    assert landed.crs.to_epsg() == 6318
    assert np.isfinite(landed.geometry.x.iloc[0]) and np.isfinite(landed.geometry.y.iloc[0])
    assert landed["transform_id"].iloc[0].endswith("tt=coord_epoch")
