"""crs.realization_shift: predicted NAD83 realization shifts through PROJ's best
route (NADCON5 over CONUS). Grid-dependent tests carry the network marker."""

import numpy as np
import pytest

from groundcontrol.crs import (
    NAD83_GEOGRAPHIC_CRS,
    NAD83_REALIZATIONS,
    NoTransformPathError,
    RealizationShift,
    nad83_realization,
    realization_shift,
)

LV = (-115.17, 36.10)          # Las Vegas NV
KS = (-98.0, 38.5)             # central Kansas


def test_every_token_has_a_geographic_crs_that_classifies_back():
    assert set(NAD83_GEOGRAPHIC_CRS) == set(NAD83_REALIZATIONS)
    for tok, code in NAD83_GEOGRAPHIC_CRS.items():
        assert nad83_realization(code).realization == tok, tok


def test_identity_and_shapes():
    r = realization_shift([-115.1, -115.2], [36.1, 36.2], "NAD83(2011)", "NAD83(2011)")
    assert isinstance(r, RealizationShift) and r.route.startswith("identity")
    assert r.dE.shape == (2,) and not r.dE.any() and not r.dN.any() and r.accuracy == 0.0
    # tokens may be given as CRS-like inputs
    assert realization_shift(LV[0], LV[1], 6340, 6318).route.startswith("identity")
    with pytest.raises(ValueError, match="shapes differ"):
        realization_shift([1.0, 2.0], [1.0], "NAD83(HARN)")
    with pytest.raises(ValueError, match="not a NAD83 realization"):
        realization_shift(LV[0], LV[1], "EPSG:4326")


def test_pair_without_any_operation_raises():
    """EPSG defines no NAD83(CORS96) -> NAD83(2011) operation: no fallback."""
    with pytest.raises(NoTransformPathError, match="no operation"):
        realization_shift(LV[0], LV[1], "NAD83(CORS96)")


@pytest.mark.network
def test_las_vegas_harn_matches_the_published_table():
    """The 3DEP datum provenance page tabulates Las Vegas HARN -> NAD83(2011)
    as -0.069 / +0.032 m (PROJ 9.7.1, NADCON5)."""
    r = realization_shift(LV[0], LV[1], "NAD83(HARN)")
    assert "NADCON5" in r.route and r.definition and r.accuracy is not None
    assert r.dE == pytest.approx(-0.069, abs=0.005) and r.dN == pytest.approx(0.032, abs=0.005)


@pytest.mark.network
@pytest.mark.parametrize("src, lo, hi", [
    ("NAD83(1986)", 0.20, 0.45),      # bare NAD83 -> 2011 over CONUS, the page's 0.2 to 0.45 m
    ("NAD83(HARN)", 0.005, 0.20),     # HARN -> 2011: 2 cm in Kansas, 7 cm at Las Vegas, 15 cm at Seattle
    ("NAD83(NSRS2007)", 0.0, 0.10),   # NSRS2007 -> 2011, about 1 cm at Las Vegas, 6 cm at Seattle
])
def test_nadcon5_magnitudes_in_the_documented_range(src, lo, hi):
    lon = np.array([LV[0], KS[0], -122.3])
    lat = np.array([LV[1], KS[1], 47.6])
    r = realization_shift(lon, lat, src)
    assert r.dE.shape == (3,) and "NADCON5" in r.route or src == "NAD83(NSRS2007)"
    assert (r.magnitude >= lo).all() and (r.magnitude <= hi).all(), r.magnitude


@pytest.mark.network
def test_shift_agrees_with_an_independent_ecef_enu_recomputation():
    """dE/dN through Geod.inv against the ECEF difference rotated into the
    local ENU frame at the source point (crs.ecef_to_enu), to 1 mm."""
    import pyproj
    from groundcontrol.crs import ecef_to_enu
    r = realization_shift(LV[0], LV[1], "NAD83(1986)")
    t = pyproj.Transformer.from_crs("EPSG:4269", "EPSG:6318", always_xy=True)
    lon2, lat2 = t.transform(LV[0], LV[1])
    to_ecef = pyproj.Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    x1, y1, z1 = to_ecef.transform(LV[0], LV[1], 0.0)
    x2, y2, z2 = to_ecef.transform(lon2, lat2, 0.0)
    e, n, u = ecef_to_enu(np.array([x2 - x1]), np.array([y2 - y1]), np.array([z2 - z1]),
                          np.array([LV[0]]), np.array([LV[1]]))
    assert float(r.dE) == pytest.approx(e.item(), abs=1e-3)
    assert float(r.dN) == pytest.approx(n.item(), abs=1e-3)
    assert 0.3 < float(r.magnitude) < 0.45


@pytest.mark.network
def test_reverse_direction_negates():
    a = realization_shift(LV[0], LV[1], "NAD83(HARN)", "NAD83(2011)")
    b = realization_shift(LV[0], LV[1], "NAD83(2011)", "NAD83(HARN)")
    assert a.dE == pytest.approx(-b.dE, abs=1e-3) and a.dN == pytest.approx(-b.dN, abs=1e-3)
