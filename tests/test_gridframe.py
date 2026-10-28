"""Projected GRID frame <-> true ENU (crs.grid_jacobian and friends).

Two independent oracles, neither using proj_factors:
- geodesic: step 1 m true east / north on the ellipsoid with pyproj.Geod, project, difference;
- ECEF: displace a point at height h along the local ENU axes in ECEF (crs.enu_to_ecef),
  convert back to geodetic with PROJ geocent, project, difference.
All offline (no grids, no datum shifts: the lon/lat are in each projection's own datum).
"""

import numpy as np
import pyproj
import pytest

from groundcontrol.crs import (
    _geodetic_to_ecef,
    enu_to_ecef,
    enu_to_grid_gradient,
    enu_to_grid_vector,
    grid_convergence,
    grid_jacobian,
    grid_to_enu_gradient,
    grid_to_enu_vector,
    grid_to_true_azimuth,
)

# (crs, lon, lat): UTM 14N west of its CM (Kansas), UTM 11N east of its CM (Las Vegas),
# UTM 10N near a zone edge (Washington), CONUS Albers far from its CM (non-conformal, gamma ~ -14)
CASES = [
    ("EPSG:6343", -101.0, 37.8),
    ("EPSG:6340", -115.1, 36.1),
    ("EPSG:6339", -120.2, 47.6),
    ("EPSG:5070", -120.0, 45.0),
    ("EPSG:32733", 17.0, -22.5),      # southern hemisphere UTM 33S (Namibia)
    ("EPSG:3031", 160.0, -77.5),      # Antarctic polar stereographic, away from the pole
    ("EPSG:3413", -50.0, 72.0),       # NSIDC north polar stereographic
    ("EPSG:2193", 172.0, -43.5),      # NZTM: declared axis order northing, easting
    ("EPSG:27572", 5.0, 46.5),        # Lambert zone II, Paris prime meridian, grads base
]
# non-meter grids, checked by the same oracles in native units: Washington North (US survey
# foot) and Arizona Central (international foot), NAD83(2011) State Plane
FEET_CASES = [
    ("EPSG:6597", -120.8, 48.2),
    ("EPSG:6405", -111.9, 33.4),
]


def _greenwich_longlat(crs):
    """Greenwich-degree geographic CRS on crs's own ellipsoid (the oracles' input frame; PROJ
    applies the prime-meridian shift inside the transformer, so EPSG:27572's Paris base works)."""
    ell = crs.ellipsoid
    return pyproj.CRS.from_proj4(f"+proj=longlat +a={ell.semi_major_metre} +rf={ell.inverse_flattening} +type=crs")


def _geod_jacobian(crs, lon, lat, d=1.0):
    """Brute-force A (h = 0): central differences of 1 m geodesic steps east and north."""
    gg = _greenwich_longlat(crs)
    proj = pyproj.Transformer.from_crs(gg, crs, always_xy=True)
    geod = gg.get_geod()
    cols = []
    for az in (90.0, 0.0):
        lo1, la1, _ = geod.fwd(lon, lat, az, d)
        lo0, la0, _ = geod.fwd(lon, lat, az + 180.0, d)
        x1, y1 = proj.transform(lo1, la1)
        x0, y0 = proj.transform(lo0, la0)
        cols.append([(x1 - x0) / (2 * d), (y1 - y0) / (2 * d)])
    return np.array(cols).T


@pytest.mark.parametrize("code,lon,lat", CASES + FEET_CASES)
def test_jacobian_matches_geodesic_steps(code, lon, lat):
    crs = pyproj.CRS(code)
    np.testing.assert_allclose(grid_jacobian(code, lon, lat), _geod_jacobian(crs, lon, lat), atol=1e-8)


@pytest.mark.parametrize("code,lon,lat", CASES + FEET_CASES)
@pytest.mark.parametrize("h", [0.0, 1500.0, 4000.0])
def test_vector_matches_ecef_displacement(code, lon, lat, h):
    crs = pyproj.CRS(code)
    ell = crs.ellipsoid
    a = ell.semi_major_metre
    f = 1.0 / ell.inverse_flattening
    e2 = f * (2 - f)
    gg = _greenwich_longlat(crs)
    to_geod = pyproj.Transformer.from_crs(
        pyproj.CRS.from_proj4(f"+proj=geocent +a={a} +rf={ell.inverse_flattening}"), gg, always_xy=True)
    proj = pyproj.Transformer.from_crs(gg, crs, always_xy=True)
    x0, y0, z0 = _geodetic_to_ecef(lon, lat, h, a, e2)
    d_e, d_n = 0.7, -0.4                                  # a ground displacement at height h
    dX, dY, dZ = enu_to_ecef(d_e, d_n, 0.0, lon, lat)
    out = []
    for s in (1.0, -1.0):
        lo, la, _ = to_geod.transform(x0 + s * dX, y0 + s * dY, z0 + s * dZ)
        out.append(proj.transform(lo, la))
    want = (np.array(out[0]) - np.array(out[1])) / 2
    got = np.array(enu_to_grid_vector(d_e, d_n, code, lon, lat, h))
    np.testing.assert_allclose(got, want, atol=1e-8)
    back = np.array(grid_to_enu_vector(*want, code, lon, lat, h))
    np.testing.assert_allclose(back, [d_e, d_n], atol=1e-8)


@pytest.mark.parametrize("code,lon,lat", CASES[:3])
def test_conformal_reduces_to_convergence_and_scale(code, lon, lat):
    fac = pyproj.Proj(code).get_factors(lon, lat)
    gamma, k = np.radians(fac.meridian_convergence), fac.meridional_scale
    # grid axes sit gamma clockwise of true, so true east lands at +gamma counterclockwise in the grid
    rot = np.array([[np.cos(gamma), -np.sin(gamma)], [np.sin(gamma), np.cos(gamma)]])
    np.testing.assert_allclose(grid_jacobian(code, lon, lat), k * rot, atol=1e-9)
    for az in (0.0, 30.0, 137.0, 300.0):
        want = (az + fac.meridian_convergence) % 360.0
        assert abs(((grid_to_true_azimuth(az, code, lon, lat) - want) + 180) % 360 - 180) < 1e-7


def test_convergence_sign_east_and_west_of_cm():
    # northern hemisphere: grid north is clockwise of true north (gamma > 0) east of the CM
    assert grid_to_true_azimuth(0.0, "EPSG:6343", -97.0, 37.8) > 0.5         # CM -99, point east
    assert grid_to_true_azimuth(0.0, "EPSG:6343", -101.0, 37.8) > 358.0      # point west: just below 360


@pytest.mark.parametrize("code,lon,lat", CASES + FEET_CASES)
def test_grid_convergence_is_the_true_azimuth_of_grid_north(code, lon, lat):
    # brute force, the geocalval definition: unproject a step along grid +y, take its geodesic azimuth
    crs = pyproj.CRS(code)
    gg = _greenwich_longlat(crs)
    fwd = pyproj.Transformer.from_crs(gg, crs, always_xy=True)
    inv = pyproj.Transformer.from_crs(crs, gg, always_xy=True)
    x, y = fwd.transform(lon, lat)
    lon2, lat2 = inv.transform(x, y + 1.0)
    want, _, _ = gg.get_geod().inv(lon, lat, lon2, lat2)
    got = grid_convergence(code, lon, lat)
    assert -180.0 < got <= 180.0
    assert abs(got - want) < 1e-6
    conformal = crs.coordinate_operation.method_name != "Albers Equal Area"
    if crs.geodetic_crs.prime_meridian.longitude != 0:
        return                                       # PROJ's factors are Paris-relative; compared above
    proj = pyproj.Proj(code).get_factors(lon, lat).meridian_convergence
    # PROJ's convergence is the GRID angle of the true-north direction: the same angle only
    # for a conformal projection (Albers at (-120, 45): 0.037 deg apart)
    assert (abs(got - proj) < 1e-7) == conformal


def test_grid_convergence_sign_and_range():
    assert grid_convergence("EPSG:6343", -97.0, 37.8) > 0.5          # east of CM: positive
    assert -3.0 < grid_convergence("EPSG:6343", -101.0, 37.8) < -0.5  # west of CM: negative, not ~358
    assert abs(grid_convergence("EPSG:6343", -99.0, 37.8)) < 1e-9     # on the CM


def test_albers_is_not_a_pure_rotation():
    # EPSG:5070 at (-120, 45): azimuth conversion is not az + gamma for every direction (shear)
    gamma = pyproj.Proj("EPSG:5070").get_factors(-120.0, 45.0).meridian_convergence
    d = [((grid_to_true_azimuth(az, "EPSG:5070", -120.0, 45.0) - az - gamma + 180) % 360) - 180 for az in (0, 45, 90, 135)]
    assert np.ptp(d) > 0.1


@pytest.mark.parametrize("code,lon,lat", CASES)
def test_gradient_is_a_covector(code, lon, lat):
    rng = np.random.default_rng(0)
    gx, gy, dx, dy = rng.normal(size=(4, 50))
    ge, gn = grid_to_enu_gradient(gx, gy, code, lon, lat, 900.0)
    de, dn = grid_to_enu_vector(dx, dy, code, lon, lat, 900.0)
    np.testing.assert_allclose(gx * dx + gy * dy, ge * de + gn * dn, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.array(enu_to_grid_gradient(ge, gn, code, lon, lat, 900.0)), [gx, gy], atol=1e-12)


def test_vectorized_shapes_and_broadcast():
    lon = np.linspace(-102, -96, 7)
    lat = np.full(7, 38.0)
    A = grid_jacobian("EPSG:6343", lon, lat)
    assert A.shape == (7, 2, 2)
    e, n = grid_to_enu_vector(1.0, 0.0, "EPSG:6343", lon, lat)
    assert e.shape == (7,) and n.shape == (7,)
    for i in range(7):
        np.testing.assert_allclose(A[i], grid_jacobian("EPSG:6343", lon[i], lat[i]), atol=0)


@pytest.mark.parametrize("feet,meters,lon,lat,ft", [
    ("EPSG:6597", "EPSG:6596", -120.8, 48.2, 1200.0 / 3937.0),
    ("EPSG:6405", "EPSG:6404", -111.9, 33.4, 0.3048),
])
def test_feet_grid_is_the_meter_grid_scaled(feet, meters, lon, lat, ft):
    # same projection, different linear unit: A differs by exactly meters-per-foot
    np.testing.assert_allclose(grid_jacobian(feet, lon, lat) * ft, grid_jacobian(meters, lon, lat), rtol=1e-12)
    # azimuth is unit-free
    assert abs(grid_to_true_azimuth(30.0, feet, lon, lat) - grid_to_true_azimuth(30.0, meters, lon, lat)) < 1e-9


def test_compound_crs_uses_its_projected_component():
    np.testing.assert_allclose(grid_jacobian("EPSG:6340+5703", -115.1, 36.1), grid_jacobian("EPSG:6340", -115.1, 36.1))


def test_paris_prime_meridian_is_not_ignored():
    # ignoring the 2.337 deg prime meridian gives +3.645 deg here; the truth is +1.941
    assert abs(grid_convergence("EPSG:27572", 5.0, 46.5) - 1.94108) < 2e-5


def test_pole_raises():
    for code, lat in (("EPSG:3031", -90.0), ("EPSG:3413", 90.0)):
        with pytest.raises(ValueError, match="pole"):
            grid_jacobian(code, 0.0, lat)
    with pytest.raises(ValueError, match="pole"):
        grid_jacobian("EPSG:3413", np.array([-50.0, 0.0]), np.array([72.0, 90.0]))


def test_nan_input_propagates_and_outside_domain_raises():
    A = grid_jacobian("EPSG:6343", np.array([-99.0, np.nan]), np.array([38.0, 38.0]))
    assert np.isfinite(A[0]).all() and np.isnan(A[1]).all()
    with pytest.raises(pyproj.exceptions.ProjError):   # errcheck=True: the far side of an orthographic
        grid_jacobian("+proj=ortho +lat_0=0 +lon_0=0 +type=crs", 120.0, 0.0)


def test_northing_first_axis_order_is_still_easting_northing():
    # NZTM declares (northing, easting); dx is still PROJ easting
    A = grid_jacobian("EPSG:2193", 172.0, -43.5)
    assert A[0, 0] > 0.99 and A[1, 1] > 0.99


def test_geographic_crs_raises():
    with pytest.raises(ValueError, match="projected"):
        grid_jacobian("EPSG:6318", -100.0, 38.0)
