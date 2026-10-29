"""crs.is_dynamic_frame when the WKT2 DYNAMIC property is lost (GeoTIFF keys
cannot carry it): the datum identity decides."""

import numpy as np
import pyproj
import pytest
import rasterio
from rasterio.transform import from_origin

from groundcontrol.crs import DatasetFrame, compare_frames, dataset_frame, is_dynamic_frame

G2139_UTM10_2D = pyproj.CRS.from_user_input(
    'PROJCRS["WGS 84 (G2139) / UTM zone 10N",BASEGEOGCRS["WGS 84 (G2139)",'
    'DYNAMIC[FRAMEEPOCH[2016]],DATUM["World Geodetic System 1984 (G2139)",'
    'ELLIPSOID["WGS 84",6378137,298.257223563]],ID["EPSG",9755]],'
    'CONVERSION["UTM zone 10N",METHOD["Transverse Mercator"],'
    'PARAMETER["Latitude of natural origin",0],PARAMETER["Longitude of natural origin",-123],'
    'PARAMETER["Scale factor at natural origin",0.9996],PARAMETER["False easting",500000],'
    'PARAMETER["False northing",0]],CS[Cartesian,2],AXIS["(E)",east],AXIS["(N)",north],'
    'LENGTHUNIT["metre",1]]')


def _write(path, crs):
    with rasterio.open(path, "w", driver="GTiff", width=2, height=2, count=1, dtype="float32",
                       crs=crs.to_wkt(), transform=from_origin(5e5, 5e6, 1, 1)) as d:
        d.write(np.zeros((1, 2, 2), np.float32))
    return str(path)


def test_two_d_geotiff_in_a_dynamic_frame_reads_back_dynamic(tmp_path):
    """The written CRS has DYNAMIC; the GeoTIFF keys drop it; the datum id
    (EPSG 1309) and name survive, and the registry says it is dynamic."""
    p = _write(tmp_path / "g2139.tif", G2139_UTM10_2D)
    with rasterio.open(p) as r:
        back = pyproj.CRS.from_wkt(r.crs.to_wkt())
    assert "DYNAMIC" not in back.to_wkt()                 # the loss this guards against
    assert is_dynamic_frame(back) and is_dynamic_frame(G2139_UTM10_2D)
    f = dataset_frame(p)
    assert f.dynamic is True and f.crs_source == "dataset"
    assert compare_frames(f, f).dynamic_a is True


@pytest.mark.parametrize("crs_like, expected", [
    ("EPSG:7912", True), ("EPSG:9755", True), ("EPSG:9989", True), ("EPSG:8401", True),   # ITRF2014, G2139, ITRF2020, ETRF2014
    ("EPSG:4326", False), ("EPSG:32610", False), ("EPSG:4258", False),                      # ensembles
    ("EPSG:6318", False), ("EPSG:6340", False), ("EPSG:4269", False), ("EPSG:4267", False),
])
def test_registry_identity(crs_like, expected):
    assert is_dynamic_frame(crs_like) is expected


@pytest.mark.parametrize("datum_name, expected", [
    ("World Geodetic System 1984 (G2139)", True), ("WGS 84 (G1674)", True),
    ("International Terrestrial Reference Frame 2014", True), ("ITRF2020", True), ("IGb14", True),
    ("European Terrestrial Reference Frame 2014", True),
    ("World Geodetic System 1984", False), ("NAD83 (National Spatial Reference System 2011)", False),
    ("North American Datum 1983", False),
])
def test_name_fallback_for_id_less_datums(datum_name, expected):
    wkt = (f'GEOGCS["x",DATUM["{datum_name}",SPHEROID["WGS 84",6378137,298.257223563]],'
           'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]')
    crs = pyproj.CRS(wkt)
    if (crs.datum.to_json_dict().get("id") or {}).get("code") is not None:
        pytest.skip("PROJ attached a registry id; the registry path is tested above")
    assert is_dynamic_frame(crs) is expected
    assert DatasetFrame(crs, None, "undeclared", "x.tif").dynamic is expected
