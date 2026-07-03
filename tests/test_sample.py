"""sample_raster tests — plan A5 port with the B3/B4/B11 fixes.

All offline; synthetic GeoTIFF fixtures are built in tmp_path via rasterio.
The known-value fixtures use a planar field z = 2x + 3y: bilinear interpolation
reproduces a plane exactly, so 'linear' results are checkable anywhere, and
pixel-center values check 'nearest'. North-up and south-up rasters encode the
SAME georeferenced field (B4: results must agree).
"""

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import rioxarray
import xarray as xr
from rasterio.transform import Affine

from groundcontrol.sample import sample_raster

CRS = "EPSG:32611"
NX, NY = 12, 8         # raster size (pixel size 1 m — exact FP arithmetic)
X0, Y0 = 50.0, 40.0    # grid origin (nonzero: exercises the transform math)


def _plane(x, y):
    return 2.0 * x + 3.0 * y


def _exp(rx, ry):
    """Expected plane value at raster-relative coords (rx in 0..NX, ry in 0..NY)."""
    return _plane(X0 + rx, Y0 + ry)


def _write_tif(path, arr, transform, crs=CRS, nodata=None):
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
        count=1, dtype=str(arr.dtype), crs=crs, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(arr, 1)
    return str(path)


@pytest.fixture
def north_up_tif(tmp_path):
    """North-up (descending y): row 0 is the TOP of the raster."""
    j, i = np.meshgrid(np.arange(NX), np.arange(NY))
    arr = _plane(X0 + j + 0.5, Y0 + NY - i - 0.5)  # y = Y0 + NY - (row + 0.5)
    t = Affine(1.0, 0.0, X0, 0.0, -1.0, Y0 + NY)
    return _write_tif(tmp_path / "north_up.tif", arr, t)


@pytest.fixture
def south_up_tif(tmp_path):
    """South-up (ascending y): same georeferenced plane as north_up_tif."""
    j, i = np.meshgrid(np.arange(NX), np.arange(NY))
    arr = _plane(X0 + j + 0.5, Y0 + i + 0.5)  # y = Y0 + row + 0.5
    t = Affine(1.0, 0.0, X0, 0.0, 1.0, Y0)
    return _write_tif(tmp_path / "south_up.tif", arr, t)


def _points(xy, crs=CRS, **cols):
    """Points from raster-relative (rx, ry) pairs (see _exp)."""
    xs, ys = zip(*xy)
    return gpd.GeoDataFrame(
        cols, geometry=gpd.points_from_xy(np.asarray(xs) + X0, np.asarray(ys) + Y0), crs=crs)


# ---------------------------------------------------------------------------
# known-value sampling: nearest + linear, north-up AND south-up (B4)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", ["north_up_tif", "south_up_tif"])
def test_linear_known_values_on_plane(fixture, request):
    tif = request.getfixturevalue(fixture)
    pts = [(2.3, 5.4), (0.5, 0.5), (11.49, 7.49), (6.0, 4.0)]  # interior points
    out = sample_raster(_points(pts), tif, method="linear")
    expected = [_exp(x, y) for x, y in pts]
    np.testing.assert_allclose(out[out.columns[-1]].to_numpy(), expected, rtol=1e-12)


@pytest.mark.parametrize("fixture", ["north_up_tif", "south_up_tif"])
def test_nearest_known_values_pixel_centers(fixture, request):
    tif = request.getfixturevalue(fixture)
    # (2.3, 5.4) is inside the pixel centered at (2.5, 5.5)
    out = sample_raster(_points([(2.3, 5.4), (7.9, 1.1)]), tif, method="nearest")
    np.testing.assert_allclose(
        out[out.columns[-1]].to_numpy(),
        [_exp(2.5, 5.5), _exp(7.5, 1.5)], rtol=1e-12)


@pytest.mark.parametrize("method", ["nearest", "linear"])
def test_north_up_and_south_up_agree(north_up_tif, south_up_tif, method):
    """B4: the two encodings of the same field must sample identically."""
    rng = np.random.default_rng(4)
    pts = list(zip(rng.uniform(0.6, NX - 0.6, 50), rng.uniform(0.6, NY - 0.6, 50)))
    a = sample_raster(_points(pts), north_up_tif, method=method)
    b = sample_raster(_points(pts), south_up_tif, method=method)
    np.testing.assert_allclose(a[a.columns[-1]].to_numpy(), b[b.columns[-1]].to_numpy(),
                               rtol=1e-12)


def test_point_outside_raster_is_nan(north_up_tif):
    out = sample_raster(_points([(-5.0, 4.0), (2.3, 5.4)]), north_up_tif, method="nearest")
    v = out[out.columns[-1]].to_numpy()
    assert np.isnan(v[0]) and np.isfinite(v[1])


# ---------------------------------------------------------------------------
# nodata masking (B3): NaN-nodata float raster and -9999 sentinel
# ---------------------------------------------------------------------------

@pytest.fixture
def nan_nodata_tif(tmp_path):
    j, i = np.meshgrid(np.arange(NX), np.arange(NY))
    arr = _plane(X0 + j + 0.5, Y0 + NY - i - 0.5)
    arr[2, 3] = np.nan  # pixel centered at relative x=3.5, y=NY-2.5=5.5
    t = Affine(1.0, 0.0, X0, 0.0, -1.0, Y0 + NY)
    return _write_tif(tmp_path / "nan_nodata.tif", arr, t, nodata=np.nan)


@pytest.fixture
def sentinel_nodata_tif(tmp_path):
    j, i = np.meshgrid(np.arange(NX), np.arange(NY))
    arr = _plane(X0 + j + 0.5, Y0 + NY - i - 0.5)
    arr[2, 3] = -9999.0
    t = Affine(1.0, 0.0, X0, 0.0, -1.0, Y0 + NY)
    return _write_tif(tmp_path / "sentinel.tif", arr, t, nodata=-9999.0)


@pytest.mark.parametrize("fixture", ["nan_nodata_tif", "sentinel_nodata_tif"])
def test_nodata_masked_never_returned_as_height(fixture, request):
    """B3: `arr == nodata` is always False for NaN nodata — must still mask."""
    tif = request.getfixturevalue(fixture)
    # on the nodata pixel (nearest), adjacent to it (linear), and far from it
    pts = _points([(3.5, 5.5), (3.9, 5.9), (9.5, 2.5)])
    near = sample_raster(pts, tif, method="nearest")[
        "nan_nodata" if "nan" in fixture else "sentinel"].to_numpy()
    lin = sample_raster(pts, tif, method="linear")[
        "nan_nodata" if "nan" in fixture else "sentinel"].to_numpy()
    assert np.isnan(near[0]) and np.isnan(lin[0])
    assert np.isnan(lin[1])  # linear neighborhood touches the nodata pixel
    assert near[2] == pytest.approx(_exp(9.5, 2.5))
    assert lin[2] == pytest.approx(_exp(9.5, 2.5))
    # the sentinel must never leak through as a plausible height
    for v in (near, lin):
        assert not np.any(v == -9999.0)


# ---------------------------------------------------------------------------
# tiling (B11): tiled-vs-single-window identity at seams for linear
# ---------------------------------------------------------------------------

def test_tiled_matches_single_window_bit_identical(tmp_path):
    """Points straddling tile seams must be unaffected by the tiling (halo=2)."""
    nx, ny = 64, 48
    rng = np.random.default_rng(11)
    arr = rng.normal(1000.0, 50.0, size=(ny, nx))
    t = Affine(1.0, 0.0, X0, 0.0, -1.0, Y0 + ny)  # exact FP grid arithmetic
    tif = _write_tif(tmp_path / "tiles.tif", arr, t)
    xs = rng.uniform(0.0, nx, 300)
    ys = rng.uniform(0.0, ny, 300)
    # add points ON pixel seams near tile boundaries
    xs = np.concatenate([xs, [15.0, 16.0, 17.0, 31.5, 32.5]])
    ys = np.concatenate([ys, [16.0, 15.5, 32.0, 31.0, 16.5]])
    pts = _points(list(zip(xs, ys)))
    single = sample_raster(pts, tif, method="linear", block=4096)
    tiled = sample_raster(pts, tif, method="linear", block=16)
    np.testing.assert_array_equal(single["tiles"].to_numpy(), tiled["tiles"].to_numpy())


def test_tiled_matches_single_window_nearest(tmp_path):
    nx, ny = 40, 40
    rng = np.random.default_rng(7)
    arr = rng.normal(0.0, 1.0, size=(ny, nx))
    tif = _write_tif(tmp_path / "tiles_n.tif", arr,
                     Affine(1.0, 0.0, X0, 0.0, -1.0, Y0 + ny))
    pts = _points(list(zip(rng.uniform(0, nx, 200), rng.uniform(0, ny, 200))))
    single = sample_raster(pts, tif, method="nearest", block=4096)
    tiled = sample_raster(pts, tif, method="nearest", block=8)
    np.testing.assert_array_equal(single["tiles_n"].to_numpy(), tiled["tiles_n"].to_numpy())


# ---------------------------------------------------------------------------
# API contract: CRS check, no mutation, method restriction, diff, inputs
# ---------------------------------------------------------------------------

def test_crs_mismatch_raises(north_up_tif):
    pts = _points([(2.3, 5.4)], crs="EPSG:4326")
    with pytest.raises(ValueError, match="different CRS"):
        sample_raster(pts, north_up_tif)


def test_missing_points_crs_raises(north_up_tif):
    pts = _points([(2.3, 5.4)], crs=None)
    with pytest.raises(ValueError, match="cannot verify CRS"):
        sample_raster(pts, north_up_tif)


def test_check_crs_false_escape_hatch_warns(north_up_tif):
    pts = _points([(2.3, 5.4)], crs="EPSG:4326")
    with pytest.warns(UserWarning, match="check_crs=False"):
        out = sample_raster(pts, north_up_tif, method="nearest", check_crs=False)
    assert out["north_up"].iloc[0] == pytest.approx(_exp(2.5, 5.5))


def test_input_gdf_not_mutated(north_up_tif):
    pts = _points([(2.3, 5.4)], height=[10.0])
    cols_before = list(pts.columns)
    out = sample_raster(pts, north_up_tif, diff=True)
    assert list(pts.columns) == cols_before  # no new columns on the input
    assert out is not pts
    assert "north_up" in out.columns and "north_up minus height" in out.columns


@pytest.mark.parametrize("method", ["cubic", "quintic", "bilinear", "spline"])
def test_unsupported_method_rejected(north_up_tif, method):
    with pytest.raises(ValueError, match="unsupported method"):
        sample_raster(_points([(2.3, 5.4)]), north_up_tif, method=method)


def test_diff_column_values(north_up_tif):
    pts = _points([(6.0, 4.0)], height=[20.0])
    out = sample_raster(pts, north_up_tif, method="linear", diff=True)
    assert out["north_up minus height"].iloc[0] == pytest.approx(_exp(6.0, 4.0) - 20.0)


def test_diff_without_height_column_raises(north_up_tif):
    with pytest.raises(ValueError, match="height"):
        sample_raster(_points([(6.0, 4.0)]), north_up_tif, diff=True)


def test_dataarray_input_matches_path_input(north_up_tif):
    pts = _points([(2.3, 5.4), (6.0, 4.0)])
    r = rioxarray.open_rasterio(north_up_tif)
    a = sample_raster(pts, r, method="linear")
    b = sample_raster(pts, north_up_tif, method="linear")
    np.testing.assert_array_equal(a[a.columns[-1]].to_numpy(), b["north_up"].to_numpy())


def test_in_memory_dataarray_no_source(north_up_tif):
    """Computed arrays (no encoding['source']) use the xarray fallback (B4 there too)."""
    r = rioxarray.open_rasterio(north_up_tif).squeeze("band", drop=True)
    mem = xr.DataArray(r.values, coords={"y": r.y.values, "x": r.x.values},
                       dims=("y", "x"), name="mem")
    mem = mem.rio.write_crs(CRS)
    assert "source" not in mem.encoding
    pts = _points([(2.3, 5.4), (6.0, 4.0), (-5.0, 4.0)])
    lin = sample_raster(pts, mem, method="linear")["mem"].to_numpy()
    near = sample_raster(pts, mem, method="nearest")["mem"].to_numpy()
    np.testing.assert_allclose(lin[:2], [_exp(2.3, 5.4), _exp(6.0, 4.0)], rtol=1e-12)
    np.testing.assert_allclose(near[:2], [_exp(2.5, 5.5), _exp(6.5, 4.5)], rtol=1e-12)
    assert np.isnan(lin[2]) and np.isnan(near[2])  # outside -> NaN, not edge-snapped


def test_multiband_raster_rejected(tmp_path):
    arr = np.zeros((2, 4, 4), dtype="float64")
    with rasterio.open(
        tmp_path / "mb.tif", "w", driver="GTiff", height=4, width=4, count=2,
        dtype="float64", crs=CRS, transform=Affine(1.0, 0.0, X0, 0.0, -1.0, Y0 + 4.0),
    ) as dst:
        dst.write(arr)
    with pytest.raises(ValueError, match="single-band"):
        sample_raster(_points([(1.0, 1.0)]), str(tmp_path / "mb.tif"))
