"""ITRF2020 plate motion model (crs.ITRF2020PMM) — Phase C1.

Pole table transcribed from https://itrf.ign.fr/docs/solutions/itrf2020/ITRF2020-PMM.dat
(Altamimi et al. 2023, GRL 50, e2023GL106373). Cross-checks here pin the
transcription against the paper's independent mas/yr form (Table 1) and the
kernel's pole/rate construction path.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from groundcontrol.crs import (
    ITRF2020_ORB_MM_PER_YR,
    ITRF2020_PMM_DEG_PER_MYR,
    ITRF2020PMM,
    EulerPoleModel,
    assign_plate,
    propagate_epoch,
)

# Las Vegas — the single-plate (NOAM) AOI the stage-2 plan calls out
LV_LON, LV_LAT, LV_H = -115.1, 36.1, 700.0


def test_from_angular_velocity_reconstructs_omega_exactly():
    """Pole/rate round-trip: cartesian -> (pole, rate) -> _omega_ecef matches."""
    wx, wy, wz = ITRF2020_PMM_DEG_PER_MYR["NOAM"]
    m = EulerPoleModel.from_angular_velocity(wx, wy, wz, unit="deg/Myr")
    w_expect = np.radians(np.array([wx, wy, wz])) * 1e-6  # deg/Myr -> rad/yr
    assert m._omega_ecef == pytest.approx(w_expect, rel=1e-12)


def test_from_angular_velocity_mas_yr_matches_paper_within_0p1_mm_yr():
    """Published-pole check: paper Table 1 (mas/yr, 3 decimals) vs the .dat
    (deg/Myr) agree to <=0.1 mm/yr predicted velocity at a CONUS point."""
    m_dat = EulerPoleModel.from_angular_velocity(
        *ITRF2020_PMM_DEG_PER_MYR["NOAM"], unit="deg/Myr")
    m_paper = EulerPoleModel.from_angular_velocity(
        0.045, -0.666, -0.098, unit="mas/yr")  # Altamimi et al. 2023 Table 1
    v1 = np.array(m_dat.velocity_enu([LV_LON], [LV_LAT], [LV_H]))
    v2 = np.array(m_paper.velocity_enu([LV_LON], [LV_LAT], [LV_H]))
    assert float(np.linalg.norm(v1 - v2)) <= 1e-4  # 0.1 mm/yr


def test_from_angular_velocity_requires_explicit_unit():
    with pytest.raises(TypeError):
        EulerPoleModel.from_angular_velocity(0.1, 0.2, 0.3)  # no unit kwarg
    with pytest.raises(ValueError, match="unit"):
        EulerPoleModel.from_angular_velocity(0.1, 0.2, 0.3, unit="rad/s")
    with pytest.raises(ValueError, match="zero angular velocity"):
        EulerPoleModel.from_angular_velocity(0.0, 0.0, 0.0, unit="deg/Myr")


def test_pole_table_matches_bundled_dat_file():
    """The transcribed dict and the bundled provenance .dat must never drift."""
    import re
    from importlib.resources import files

    text = (files("groundcontrol") / "data" / "ITRF2020-PMM.dat").read_text()
    rows = re.findall(
        r"^\s+([A-Z]{4})\s+(-?[\d.]+),\s+(-?[\d.]+),\s+(-?[\d.]+),", text, re.MULTILINE)
    table = {k: (float(a), float(b), float(c)) for k, a, b, c in rows}
    assert table == ITRF2020_PMM_DEG_PER_MYR
    orb = re.findall(r"^T[XYZ] = (-?[\d.]+)", text, re.MULTILINE)
    assert tuple(float(v) for v in orb) == ITRF2020_ORB_MM_PER_YR


def test_noam_conus_velocity_magnitude_and_direction():
    """NOAM ITRF velocity in the SW US: ~1.5-2 cm/yr horizontal (the ~1.65
    cm/yr §1 note), pointing W-SW, with ~zero vertical."""
    pmm = ITRF2020PMM("NOAM")
    ve, vn, vu = pmm.velocity_enu(np.array([LV_LON]), np.array([LV_LAT]),
                                  np.array([LV_H]))
    speed_h = float(np.hypot(ve, vn)[0])
    assert 0.012 < speed_h < 0.022
    assert abs(float(vu[0])) < 5e-4
    assert float(ve[0]) < 0  # westward component


def test_orb_shifts_horizontal_only_and_is_submm():
    v_on = np.array(ITRF2020PMM("NOAM")
                    .velocity_enu([LV_LON], [LV_LAT], [LV_H])).ravel()
    v_off = np.array(ITRF2020PMM("NOAM", apply_orb=False)
                     .velocity_enu([LV_LON], [LV_LAT], [LV_H])).ravel()
    dh = float(np.hypot(*(v_on[:2] - v_off[:2])))
    assert 0.0 < dh <= np.linalg.norm(ITRF2020_ORB_MM_PER_YR) * 1e-3  # sub-mm/yr
    assert float(v_on[2]) == pytest.approx(float(v_off[2]))  # vertical untouched


def test_unknown_plate_raises_with_available_list():
    with pytest.raises(KeyError, match="JFDF"):
        ITRF2020PMM("JFDF")


def test_plate_none_matches_fixed_plate_on_plate_interior():
    """plate=None (bundled PB2002 assignment) reproduces the fixed-plate
    velocity exactly for a point in the NOAM interior."""
    v_auto = np.array(ITRF2020PMM(None)
                      .velocity_enu([LV_LON], [LV_LAT], [LV_H])).ravel()
    v_fixed = np.array(ITRF2020PMM("NOAM")
                       .velocity_enu([LV_LON], [LV_LAT], [LV_H])).ravel()
    assert v_auto == pytest.approx(v_fixed, abs=1e-15)


def test_assign_plate_known_points_and_antimeridian():
    codes = assign_plate(
        [LV_LON, -157.86, 2.35, 179.5, -179.5],  # LV, Honolulu, Paris, mid-PCFC
        [LV_LAT, 21.31, 48.85, 5.0, 5.0])
    assert list(codes) == ["NOAM", "PCFC", "EURA", "PCFC", "PCFC"]


def test_san_andreas_straddle_resolves_two_plates():
    """The C2 motivating case: an AOI straddling the San Andreas gets NOAM on
    one side, PCFC on the other, with the ~5 cm/yr relative motion between."""
    lons, lats = np.array([-122.42, -123.8]), np.array([37.77, 37.5])
    assert list(assign_plate(lons, lats)) == ["NOAM", "PCFC"]
    ve, vn, _vu = ITRF2020PMM(None, apply_orb=False).velocity_enu(lons, lats)
    rel = float(np.hypot(ve[0] - ve[1], vn[0] - vn[1]))
    assert 0.03 < rel < 0.06  # published PA-NA relative motion ~4.6-5 cm/yr


def test_off_model_plate_is_noop_with_residual_bound():
    """Juan de Fuca: PB2002 assigns JF, which the ITRF2020 PMM does not model
    -> NaN velocity -> propagate_epoch no-op + epoch_residual_m bound."""
    assert assign_plate([-127.5], [46.0])[0] is None
    g = gpd.GeoDataFrame(
        {"height": [0.0], "coord_epoch": [2015.0]},
        geometry=[Point(-127.5, 46.0)], crs="EPSG:9989")
    with pytest.warns(UserWarning, match="velocity"):
        out = propagate_epoch(g, target_epoch=2025.0, plate_model=ITRF2020PMM(None))
    rep = out.attrs["epoch_propagation"]
    assert rep["models"] == {"per_point": 0, "plate": 0, "none": 1}
    assert out["epoch_residual_m"].iloc[0] == pytest.approx(1.6)
    assert out.geometry.x.iloc[0] == -127.5  # un-moved


def test_propagate_epoch_composes_and_midas_wins():
    """propagate_epoch(plate_model=ITRF2020PMM('NOAM')): per-point MIDAS
    velocity takes priority; velocity-less rows get the plate model."""
    g = gpd.GeoDataFrame(
        {
            "height": [LV_H, LV_H],
            "vel_e": [0.0, np.nan],  # row 0: explicit zero per-point velocity
            "vel_n": [0.0, np.nan],
            "vel_u": [0.0, np.nan],
            "coord_epoch": [2015.0, 2015.0],
            "transform_id": pd.array(["a", "b"], dtype="string"),
        },
        geometry=[Point(LV_LON, LV_LAT), Point(LV_LON + 0.1, LV_LAT + 0.1)],
        crs="EPSG:9989",  # ITRF2020 3D... geographic; ellipsoid GRS80
    )
    out = propagate_epoch(g, target_epoch=2025.0, plate_model=ITRF2020PMM("NOAM"))
    rep = out.attrs["epoch_propagation"]
    assert rep["models"] == {"per_point": 1, "plate": 1, "none": 0}
    assert rep["plate_model"] == "ITRF2020-PMM[NOAM]+ORB"
    # row 0 pinned by its zero per-point velocity
    assert out.geometry.x.iloc[0] == pytest.approx(LV_LON, abs=1e-15)
    # row 1 moved ~10 yr * ~1.6 cm/yr = ~16 cm horizontal
    dlon = (out.geometry.x.iloc[1] - (LV_LON + 0.1)) * 111e3 * np.cos(np.radians(LV_LAT))
    dlat = (out.geometry.y.iloc[1] - (LV_LAT + 0.1)) * 111e3
    moved = float(np.hypot(dlon, dlat))
    assert 0.05 < moved < 0.5
    # both rows fully propagated -> zero residual bound
    assert (out["epoch_residual_m"] == 0.0).all()


def test_assign_plate_exact_antimeridian_not_orphaned():
    """lon = ±180 lies exactly on the seam edge of the antimeridian-split
    polygons; 'within' orphaned every such point (adversarial audit)."""
    for lon in (180.0, -180.0):
        codes = assign_plate([lon, lon], [-30.0, 60.0])
        assert codes[0] is not None and codes[1] is not None
