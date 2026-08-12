"""point_context_gallery: opt-in per-point QA contact sheet (offline,
synthetic rasters)."""

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from groundcontrol.figures import point_context_gallery

CRS_A = "EPSG:32610"  # WGS84 UTM 10N
CRS_B = "EPSG:6339"   # NAD83(2011) UTM 10N (reprojection path)


def _write(tmp, name, bands, crs=CRS_A, res=1.0, size=200, nodata=-9999.0,
           dtype="float32"):
    fn = tmp / name
    rng = np.random.default_rng(42)
    data = rng.uniform(10, 20, (bands, size, size)).astype(dtype)
    with rasterio.open(
            fn, "w", driver="GTiff", width=size, height=size, count=bands,
            dtype=dtype, crs=crs, nodata=nodata,
            transform=from_origin(500000, 4000000, res, res)) as dst:
        dst.write(data)
    return fn


@pytest.fixture
def layers(tmp_path):
    return [
        ("ortho", _write(tmp_path, "rgb.tif", 3), "rgb"),
        ("intensity", _write(tmp_path, "int.tif", 1), "gray"),
        ("relief", _write(tmp_path, "dem.tif", 1), "relief"),
    ]


@pytest.fixture
def points():
    # two points inside the 200 m synthetic footprint, one far outside
    return gpd.GeoDataFrame(
        {"id": ["P1", "P2", "FAR"],
         "cls": ["mast", "building", "mast"]},
        geometry=gpd.points_from_xy([500050, 500150, 900000],
                                    [3999950, 3999850, 3000000]),
        crs=CRS_A)


class TestPointContextGallery:
    def test_writes_sheet(self, layers, points, tmp_path):
        fp = point_context_gallery(points, layers, tmp_path, "TEST",
                                   half_m=20, scale_len=10)
        assert fp.exists() and fp.name == "TEST_station_gallery_40m.png"

    def test_out_of_footprint_point_survives(self, layers, points, tmp_path):
        # FAR is outside every raster: sheet still writes (panel renders
        # "unavailable" or an all-nodata window, never raises)
        fp = point_context_gallery(points, layers, tmp_path, "TEST",
                                   half_m=20)
        assert fp.exists()

    def test_reprojects_mismatched_layer(self, points, tmp_path):
        # layer in a different CRS: per-panel reprojection path must run
        # without error (NAD83 vs WGS84 UTM here is a ~m-level shift)
        lyr = [("gray-b", _write(tmp_path, "b.tif", 1, crs=CRS_B), "gray")]
        fp = point_context_gallery(points.iloc[:2], lyr, tmp_path, "TEST",
                                   half_m=20)
        assert fp.exists()

    def test_class_colors_and_tier_tag(self, layers, points, tmp_path):
        fp = point_context_gallery(
            points.iloc[:2], layers, tmp_path, "TEST", half_m=15,
            interp="nearest", tier_tag="30m", subset_tag="opus",
            class_col="cls", class_colors={"mast": "#111111",
                                           "building": "#8B4E00"})
        assert fp.name == "TEST_opus_gallery_30m.png"

    def test_unknown_kind_is_loud(self, points, tmp_path):
        lyr = [("bad", _write(tmp_path, "x.tif", 1), "slope")]
        fp = point_context_gallery(points.iloc[:1], lyr, tmp_path, "TEST")
        # the bad kind lands in the per-panel guard -> "unavailable" panel,
        # sheet still written; nothing silently mis-rendered
        assert fp.exists()
