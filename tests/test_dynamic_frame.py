"""crs.is_dynamic_frame when the WKT2 DYNAMIC property is lost (GeoTIFF keys
cannot carry it): the datum identity decides."""

import numpy as np
import pyproj
import pytest
import rasterio
from rasterio.transform import from_origin

from groundcontrol import crs as gc_crs
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


def _need(epsg):
    """Skip when the runner's PROJ database predates this EPSG code."""
    try:
        return pyproj.CRS.from_epsg(epsg)
    except pyproj.exceptions.CRSError:
        pytest.skip(f"EPSG:{epsg} not in this PROJ database ({pyproj.proj_version_str})")


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
    ("EPSG:7912", True), ("EPSG:9755", True), ("EPSG:9989", True),   # ITRF2014, G2139, ITRF2020
    ("EPSG:4326", False), ("EPSG:32610", False), ("EPSG:4258", False),                      # ensembles
    ("EPSG:6318", False), ("EPSG:6340", False), ("EPSG:4269", False), ("EPSG:4267", False),
])
def test_registry_identity(crs_like, expected):
    assert is_dynamic_frame(crs_like) is expected


@pytest.mark.parametrize("datum_name, expected", [
    ("World Geodetic System 1984 (G2139)", True), ("WGS 84 (G1674)", True),
    ("International Terrestrial Reference Frame 2014", True), ("ITRF2020", True), ("IGb14", True),
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


def _lost_dynamic(epsg):
    """A geographic CRS re-parsed from WKT1_GDAL: the DYNAMIC property is gone,
    the datum's EPSG id (AUTHORITY) survives, as in a GeoTIFF read-back."""
    src = _need(epsg)
    if len(src.axis_info) == 3:
        src = src.to_2d()                       # WKT1 cannot carry geographic 3D
    crs = pyproj.CRS.from_wkt(src.to_wkt("WKT1_GDAL"))
    assert "DYNAMIC" not in crs.to_wkt()
    assert (crs.datum.to_json_dict().get("id") or {}).get("code") is not None
    return crs


@pytest.mark.parametrize("epsg, expected", [
    (9309, True),    # ATRF2014: dynamic in EPSG, name misses the plain regex families
    (9475, True),    # PZ-90.11: dynamic, name has no ITRF/WGS/ETRF token at all
    (8986, True),    # SIRGAS-CON SIR17P01: dynamic, name misses the regex
    (10952, False),  # California SRN epoch 2025.0 (ITRF2020): STATIC despite "ITRF2020" in the name
    (4483, False),   # Mexico ITRF92: static despite the name
])
def test_registry_decides_for_a_crs_object_with_an_id(epsg, expected, monkeypatch):
    """Review HIGH: a CRS OBJECT input must reach the registry, not only a
    string; the regex alone is wrong on these. The spy proves the path."""
    calls = []
    real = gc_crs._registry_dynamic_datum
    monkeypatch.setattr(gc_crs, "_registry_dynamic_datum",
                        lambda auth, code: calls.append(int(code)) or real(auth, code))
    crs = _lost_dynamic(epsg)
    assert is_dynamic_frame(crs) is expected
    assert calls, "registry was not consulted for a CRS object with a datum id"
    assert DatasetFrame(crs, None, "undeclared", "x.tif").dynamic is expected


def test_two_d_geotiff_reaches_the_registry(tmp_path, monkeypatch):
    calls = []
    real = gc_crs._registry_dynamic_datum
    monkeypatch.setattr(gc_crs, "_registry_dynamic_datum",
                        lambda auth, code: calls.append(int(code)) or real(auth, code))
    f = dataset_frame(_write(tmp_path / "g.tif", G2139_UTM10_2D))
    assert f.dynamic is True and 1309 in calls


def test_plate_fixed_gate_agrees_with_is_dynamic_frame():
    """propagate_epoch's raise is gated by _plate_fixed_datum; it must not call
    an ITRF frame plate-fixed just because a GeoTIFF dropped DYNAMIC."""
    for epsg in (9000, 9755, 9309):          # ITRF2014 2D, WGS 84 (G2139) 3D->to_2d below, ATRF2014
        crs = _lost_dynamic(epsg)
        assert is_dynamic_frame(crs) and not gc_crs._plate_fixed_datum(crs)
    for epsg in (6318, 4269, 10952):
        crs = _lost_dynamic(epsg)
        assert not is_dynamic_frame(crs) and gc_crs._plate_fixed_datum(crs)
    assert not gc_crs._plate_fixed_datum("EPSG:4326")     # WGS 84 ensemble: ITRF-aliased by name


def test_registry_query_failure_falls_back_to_the_name(monkeypatch, tmp_path):
    """A proj.db that cannot be queried (foreign schema, corrupt) must not
    raise out of is_dynamic_frame; the name decides and a warning is logged."""
    bad = tmp_path / "proj.db"
    bad.write_bytes(b"not a database")
    gc_crs._registry_dynamic_datum_cached.cache_clear()
    assert gc_crs._registry_dynamic_datum_cached("EPSG", "1165", str(tmp_path)) is None
    gc_crs._registry_dynamic_datum_cached.cache_clear()


@pytest.mark.parametrize("epsg, expected", [(6365, False), (10952, False), (4483, False),
                                            (9000, True), (9475, True)])
def test_wkt2_string_round_trip_resolves_the_datum_through_the_crs_id(epsg, expected):
    """A WKT2 export omits the nested datum id under the CRS id; the CRS id
    survives and names the datum in the registry (round 2 M1)."""
    wkt2 = _need(epsg).to_wkt()
    assert is_dynamic_frame(wkt2) is expected
    assert is_dynamic_frame(pyproj.CRS.from_wkt(wkt2)) is expected


def _points(crs, ve=np.nan, vn=np.nan, vu=np.nan):
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point
    data = {"height": [700.0], "coord_epoch": [2015.0], "frame_epoch": [np.nan],
            "vel_e": [ve], "vel_n": [vn], "vel_u": [vu],
            "transform_id": pd.array([pd.NA], dtype="string")}
    return gpd.GeoDataFrame(data, geometry=[Point(-115.1, 36.1)], crs=crs)


def test_propagate_epoch_gate_sees_the_datum_identity():
    """The gate that matters is propagate_epoch's own (round 2 HIGH): a static
    frame with ITRF in its name must still refuse velocities as plate-fixed,
    and a dynamic frame that lost DYNAMIC must propagate or raise on no
    velocity, never silently no-op."""
    from groundcontrol.crs import propagate_epoch
    mexico = _need(6365)                               # Mexico ITRF2008: static
    with pytest.raises(ValueError, match="plate-fixed"):
        propagate_epoch(_points(mexico, 0.01, 0.01, 0.0), target_epoch=2025.0)
    pz = _lost_dynamic(9475)                          # PZ-90.11: dynamic, DYNAMIC lost
    out = propagate_epoch(_points(pz, 0.01, 0.01, 0.0), target_epoch=2025.0)
    assert out.attrs["epoch_propagation"]["n_propagated"] == 1
    with pytest.raises(ValueError, match="no usable velocity"):
        propagate_epoch(_points(pz), target_epoch=2025.0)   # no velocity: raise, not no-op


@pytest.mark.parametrize("epsg, expected", [(6365, False), (10952, False), (9475, True), (9000, True)])
def test_bound_crs_through_geoparquet_keeps_the_datum_identity(tmp_path, epsg, expected):
    """A TOWGS84-bound CRS (GDAL-2-era sources) serialized by geopandas to
    GeoParquet (PROJJSON) loses the nested datum id under its source CRS;
    the source CRS id resolves it (round 3 MEDIUM)."""
    import geopandas as gpd
    from shapely.geometry import Point
    wkt1 = _need(epsg).to_wkt("WKT1_GDAL")
    if "TOWGS84" not in wkt1:   # a GDAL-2-era header: the tie sits inside DATUM after SPHEROID
        i = wkt1.index("SPHEROID[") + len("SPHEROID[")
        depth = 1
        while depth:              # walk to the SPHEROID node's closing bracket (AUTHORITY nests)
            depth += {"[": 1, "]": -1}.get(wkt1[i], 0)
            i += 1
        wkt1 = wkt1[:i] + ",TOWGS84[0,0,0,0,0,0,0]" + wkt1[i:]
    bound = pyproj.CRS.from_wkt(wkt1)
    assert bound.is_bound, wkt1
    g = gpd.GeoDataFrame({"a": [1]}, geometry=[Point(-100.0, 20.0)], crs=bound)
    g.to_parquet(tmp_path / "b.parquet")
    back = gpd.read_parquet(tmp_path / "b.parquet").crs
    assert back.is_bound and (back.datum.to_json_dict().get("id") or {}).get("code") is None
    assert is_dynamic_frame(back) is expected
    assert gc_crs._plate_fixed_datum(back) is (not expected)


def test_towgs84_proj_string_is_not_treated_as_wgs84():
    """PROJ names a +towgs84 proj-string datum 'Unknown based on ... using
    towgs84=...'; the substring WGS must not pass the plate-fixed guard."""
    crs = pyproj.CRS("+proj=longlat +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +no_defs")
    assert crs.datum.name.lower().startswith("unknown")
    assert gc_crs._plate_fixed_datum(crs) is True
    assert gc_crs._plate_fixed_datum("EPSG:4326") is False      # the real WGS 84 ensemble still passes


@pytest.mark.parametrize("compound, expected", [("EPSG:9755+3855", True), ("EPSG:9000+5703", True),
                                                ("EPSG:6340+5703", False), ("EPSG:32611+5773", False)])
def test_compound_crs_classifies_by_its_horizontal_member(compound, expected):
    """PR #43 bot: a compound CRS, as an object and as a WKT2 string (ids
    dropped), classifies by the horizontal member's datum."""
    crs = pyproj.CRS(compound)
    assert is_dynamic_frame(crs) is expected
    assert is_dynamic_frame(crs.to_wkt()) is expected
    assert DatasetFrame(crs, None, "undeclared", "x.tif").dynamic is expected


@pytest.mark.parametrize("form", ["object", "wkt2", "wkt1", "idless"])
def test_etrf_realizations_are_plate_fixed_for_propagation(form):
    """PR #43 bot: ETRF is ITRF rotated onto the Eurasian plate; ITRF-frame
    velocities must not be applied to it, however the datum arrives."""
    from groundcontrol.crs import propagate_epoch
    base = _need(8401)                                    # ETRF2014 geographic 2D
    crs = {"object": base, "wkt2": pyproj.CRS(base.to_wkt()),
           "wkt1": pyproj.CRS.from_wkt(base.to_wkt("WKT1_GDAL")),
           "idless": pyproj.CRS('GEOGCS["x",DATUM["European Terrestrial Reference Frame 2014",'
                                'SPHEROID["GRS 1980",6378137,298.257222101]],PRIMEM["Greenwich",0],'
                                'UNIT["degree",0.0174532925199433]]')}[form]
    assert gc_crs._plate_fixed_datum(crs) is True
    if crs.is_geographic:   # pyproj reports some WKT1 re-parses of ETRF2014 as projected
        with pytest.raises(ValueError, match="plate-fixed"):
            propagate_epoch(_points(crs, 0.02, 0.01, 0.0), target_epoch=2025.0)
    assert gc_crs._plate_fixed_datum("EPSG:4258") is True   # the ETRS89 ensemble, as before


@pytest.mark.parametrize("compound, plate_fixed", [
    ("EPSG:32611+5773", False),   # WGS 84 ensemble member: ITRF-aliased by name, never plate-fixed
    ("EPSG:9000+5703", False),    # ITRF2014: dynamic
    ("EPSG:6340+5703", True),     # NAD83(2011): plate-fixed
])
def test_plate_fixed_gate_sees_the_compound_member(compound, plate_fixed):
    """PR #43 bot: the gate must derive the member's datum for a compound CRS
    too, as an object and as a WKT2 string."""
    crs = pyproj.CRS(compound)
    assert gc_crs._plate_fixed_datum(crs) is plate_fixed
    assert gc_crs._plate_fixed_datum(crs.to_wkt()) is plate_fixed
    assert gc_crs._horizontal_datum_of(crs) is not None
