"""AOI contract (aoi.py): bbox, vector file, raster footprint, GeoDataFrame all
resolve to the same (bounds_4326, polygon) pair; the BYOD assess path derives
its AOI, site name, target CRS and figure hillshade from the product alone.
Offline; every raster is synthetic.
"""

import numpy as np
import geopandas as gpd
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from groundcontrol import aoi as aoi_mod

CRS = "EPSG:32612"


def _dem(path, *, crs=CRS, nodata=-9999.0, hole=True, n=40, wkt=None):
    """40x40 30 m plane at UTM 12N (Casa Grande-ish); left 25% nodata."""
    arr = np.full((n, n), 400.0, dtype="float32")
    if hole:
        arr[:, : n // 4] = nodata
    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=1,
                       dtype="float32", crs=wkt or crs, nodata=nodata,
                       transform=from_origin(400000.0, 3650000.0, 30.0, 30.0)) as d:
        d.write(arr, 1)
    return str(path)


def test_resolve_bbox_and_geodataframe():
    b, poly = aoi_mod.resolve_aoi((-112, 32.6, -111.5, 33.0))
    assert b == (-112.0, 32.6, -111.5, 33.0) and poly is None
    g = gpd.GeoDataFrame(geometry=[box(-112, 32.6, -111.5, 33.0)], crs=4326)
    b2, poly2 = aoi_mod.resolve_aoi(g)
    assert b2 == pytest.approx(b) and poly2.equals(g.geometry.iloc[0])
    # projected input is reprojected, GeoSeries accepted, shapely taken as lon/lat
    b3, _ = aoi_mod.resolve_aoi(g.to_crs(CRS).geometry)
    assert b3 == pytest.approx(b, abs=1e-6)
    b4, _ = aoi_mod.resolve_aoi(box(-112, 32.6, -111.5, 33.0))
    assert b4 == b


def test_resolve_vector_file_formats(tmp_path):
    g = gpd.GeoDataFrame(geometry=[box(-112, 32.6, -111.5, 33.0)], crs=4326)
    for name, drv in (("a.geojson", "GeoJSON"), ("a.gpkg", "GPKG"), ("a.shp", None)):
        g.to_file(tmp_path / name, driver=drv)
        b, poly = aoi_mod.resolve_aoi(tmp_path / name)
        assert b == pytest.approx((-112, 32.6, -111.5, 33.0)) and poly is not None


def test_raster_footprint_is_valid_data_only(tmp_path):
    dem = _dem(tmp_path / "dem.tif")
    fp = aoi_mod.raster_footprint(dem)
    assert fp.crs.to_epsg() == 4326 and len(fp) == 1
    # valid data = right 75% of the grid: compare areas in the raster CRS
    poly_utm = fp.to_crs(CRS).geometry.iloc[0]
    assert poly_utm.area == pytest.approx((40 * 30) ** 2 * 0.75, rel=1e-6)
    b = poly_utm.bounds
    assert b[0] == pytest.approx(400000 + 10 * 30, abs=1e-3) and b[2] == pytest.approx(401200)
    # no nodata tag -> the whole grid is the footprint
    full = _dem(tmp_path / "full.tif", hole=False)
    assert aoi_mod.raster_footprint(full).to_crs(CRS).geometry.iloc[0].area \
        == pytest.approx(1200 * 1200, rel=1e-6)


def test_raster_footprint_decimated_read_matches_full(tmp_path):
    dem = _dem(tmp_path / "dem.tif", n=200)
    coarse = aoi_mod.raster_footprint(dem, max_px=50).to_crs(CRS).geometry.iloc[0]
    fine = aoi_mod.raster_footprint(dem, max_px=200).to_crs(CRS).geometry.iloc[0]
    # 1/4 decimation: the nodata edge lands on a coarse-cell boundary (<= 1
    # coarse column = 2% of the width); the footprint is a fetch extent, not
    # a survey boundary, and the docs say so
    assert coarse.area == pytest.approx(fine.area, rel=0.03)


def test_raster_footprint_refuses_no_crs_and_no_data(tmp_path):
    arr = np.ones((4, 4), "float32")
    p = tmp_path / "nocrs.tif"
    with rasterio.open(p, "w", driver="GTiff", height=4, width=4, count=1, dtype="float32",
                       transform=from_origin(0, 4, 1, 1)) as d:
        d.write(arr, 1)
    with pytest.raises(ValueError, match="no CRS"):
        aoi_mod.raster_footprint(p)
    p2 = tmp_path / "empty.tif"
    with rasterio.open(p2, "w", driver="GTiff", height=4, width=4, count=1, dtype="float32",
                       crs=CRS, nodata=0.0, transform=from_origin(0, 4, 1, 1)) as d:
        d.write(np.zeros((4, 4), "float32"), 1)
    with pytest.raises(ValueError, match="no valid pixels"):
        aoi_mod.raster_footprint(p2)


def test_read_aoi_dispatches_and_reports_both_readers(tmp_path):
    dem = _dem(tmp_path / "dem.tif")
    assert "source_raster" in aoi_mod.read_aoi(dem).columns
    (tmp_path / "junk.geojson").write_text("not json")   # vector suffix: vector error
    with pytest.raises(ValueError, match="not a readable vector AOI"):
        aoi_mod.read_aoi(tmp_path / "junk.geojson")
    (tmp_path / "junk.dat").write_text("neither")           # unknown: both readers named
    with pytest.raises(ValueError, match="not a readable vector AOI .* nor a raster"):
        aoi_mod.read_aoi(tmp_path / "junk.dat")
    g = gpd.GeoDataFrame(geometry=[box(-112, 32.6, -111.5, 33.0)], crs=4326)
    g.to_parquet(tmp_path / "aoi.parquet")                  # GeoParquet AOI
    assert aoi_mod.read_aoi(tmp_path / "aoi.parquet").total_bounds[0] == pytest.approx(-112)


def test_union_footprints_covers_every_product(tmp_path):
    a = _dem(tmp_path / "a.tif")               # right 75%
    b = _dem(tmp_path / "b.tif", hole=False)   # all
    u = aoi_mod.union_footprints([a, b]).to_crs(CRS).geometry.iloc[0]
    assert u.area == pytest.approx(1200 * 1200, rel=1e-6)
    assert len(aoi_mod.union_footprints([a])) == 1


def test_fetch_control_accepts_raster_aoi(tmp_path, monkeypatch):
    """The dispatcher resolves a DEM path to its footprint and clips to it."""
    from groundcontrol import sources
    dem = _dem(tmp_path / "dem.tif")
    seen = {}

    def fake_fetch(bounds):
        seen["bounds"] = bounds
        return {}

    def fake_parse(raw):
        from groundcontrol import schema
        return schema.empty(crs="EPSG:6318")

    monkeypatch.setitem(sources.PROVIDERS, "fake", (fake_fetch, fake_parse))
    gdf, status = sources.fetch_control(dem, sources=("fake",))
    fp = aoi_mod.raster_footprint(dem).total_bounds
    assert seen["bounds"] == pytest.approx(tuple(fp))
    assert status["fake"]["n_rows"] == 0 and status["fake"]["error"] is None


# ------------------------------------------------------------- BYOD assess CLI

def _compound_wkt():
    import pyproj
    return pyproj.CRS("EPSG:6341+5703").to_wkt()


def _control_on(dem, n=4):
    """Landed control inside the DEM's valid data (heights on the plane)."""
    with rasterio.open(dem) as src:
        b = src.bounds
    xs = np.linspace(b.left + 400, b.right - 100, n)
    ys = np.linspace(b.bottom + 100, b.top - 100, n)
    return gpd.GeoDataFrame(
        {"source": ["3dep", "3dep", "opus", "ngs"][:n],
         "point_type": ["NVA", "VVA", "gnss_campaign", "monument"][:n],
         "height": np.full(n, 400.0)},
        geometry=gpd.points_from_xy(xs, ys), crs=CRS)


def test_assess_byod_needs_only_the_product(tmp_path, monkeypatch):
    """No --aoi, --site-name, --target-crs, --hs: AOI = product footprint,
    site name = product stem, target CRS = the product's compound CRS,
    hillshade computed from the product; fetch sees the footprint."""
    from groundcontrol.cli import assess_dem_main
    dem = _dem(tmp_path / "my_site_dsm.tif", wkt=_compound_wkt())
    seen = {}

    def fake_fetch_control(aoi, sources=(), **kw):
        seen["aoi"] = aoi
        return _control_on(dem), {s: {"n_rows": 4, "error": None} for s in sources}

    monkeypatch.setattr("groundcontrol.sources.fetch_control", fake_fetch_control)
    rc = assess_dem_main(["--product", f"DSM={dem}", "--source-crs", CRS,
                          "--outdir", str(tmp_path / "out")])
    assert rc == 0
    assert isinstance(seen["aoi"], gpd.GeoDataFrame)  # the footprint, not a bbox
    fp = aoi_mod.raster_footprint(dem)
    assert seen["aoi"].geometry.iloc[0].equals(fp.geometry.iloc[0])
    out = tmp_path / "out"
    assert (out / "my_site_dsm_control.parquet").exists()
    assert (out / "my_site_dsm_assessed.parquet").exists()
    assert (out / "my_site_dsm_dz_stats.csv").exists()
    assert (out / "my_site_dsm_validation_dz_DSM.png").exists()


def test_assess_refuses_2d_embedded_crs_without_target_crs(tmp_path, monkeypatch):
    from groundcontrol.cli import assess_dem_main
    dem = _dem(tmp_path / "dsm.tif")  # plain EPSG:32612, 2D
    monkeypatch.setattr("groundcontrol.sources.fetch_control",
                        lambda *a, **k: pytest.fail("fetch must not run"))
    with pytest.raises(SystemExit,
                       match=r"--target-crs or --vdatum is required"
                             r"[\s\S]*without a height axis"
                             r"[\s\S]*--vdatum ellipsoid"):
        assess_dem_main(["--product", f"DSM={dem}", "--outdir", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_assess_refuses_disagreeing_embedded_crs(tmp_path, monkeypatch):
    import pyproj
    from groundcontrol.cli import assess_dem_main
    a = _dem(tmp_path / "a.tif", wkt=_compound_wkt())
    b = _dem(tmp_path / "b.tif", wkt=pyproj.CRS("EPSG:6341+6360").to_wkt())  # NAVD88 ft
    monkeypatch.setattr("groundcontrol.sources.fetch_control",
                        lambda *a, **k: pytest.fail("fetch must not run"))
    with pytest.raises(SystemExit, match="products declare different CRSs"):
        assess_dem_main(["--product", f"DSM={a}", "--product", f"DTM={b}",
                         "--outdir", str(tmp_path / "out")])


def test_assess_explicit_aoi_raster_and_site_name_still_win(tmp_path, monkeypatch):
    from groundcontrol.cli import assess_dem_main
    dem = _dem(tmp_path / "dsm.tif", wkt=_compound_wkt())
    other = _dem(tmp_path / "other.tif", hole=False)
    seen = {}

    def fake_fetch_control(aoi, sources=(), **kw):
        seen["aoi"] = aoi
        return _control_on(dem), {s: {"n_rows": 4, "error": None} for s in sources}

    monkeypatch.setattr("groundcontrol.sources.fetch_control", fake_fetch_control)
    rc = assess_dem_main(["--aoi", other, "--product", f"DSM={dem}", "--source-crs", CRS,
                          "--site-name", "named", "--outdir", str(tmp_path / "out"),
                          "--no-figures"])
    assert rc == 0
    assert seen["aoi"].geometry.iloc[0].equals(
        aoi_mod.raster_footprint(other).geometry.iloc[0])
    assert (tmp_path / "out" / "named_assessed.parquet").exists()


def test_fetch_cli_accepts_raster_aoi(tmp_path, monkeypatch):
    from groundcontrol.cli import fetch_control_main
    dem = _dem(tmp_path / "dsm.tif")
    seen = {}

    def fake_fetch_control(aoi, sources=(), **kw):
        seen["aoi"] = aoi
        return _control_on(dem), {s: {"n_rows": 4, "error": None} for s in sources}

    monkeypatch.setattr("groundcontrol.sources.fetch_control", fake_fetch_control)
    rc = fetch_control_main(["--aoi", dem, "--sources", "ngs",
                             "--out", str(tmp_path / "c.parquet")])
    assert rc == 0 and isinstance(seen["aoi"], gpd.GeoDataFrame)


def test_fetch_cli_reports_unreadable_aoi_before_fetch(tmp_path, monkeypatch):
    from groundcontrol.cli import fetch_control_main
    monkeypatch.setattr("groundcontrol.sources.fetch_control",
                        lambda *a, **k: pytest.fail("fetch must not run"))
    (tmp_path / "junk.geojson").write_text("{}")
    with pytest.raises(SystemExit, match=r"--aoi .*junk.geojson: not a readable vector AOI"):
        fetch_control_main(["--aoi", str(tmp_path / "junk.geojson"), "--sources", "ngs",
                            "--out", str(tmp_path / "c.parquet")])


# ------------------------------------------------------------- auto hillshade

def test_hillshade_from_raster_shape_extent_and_nodata(tmp_path):
    from groundcontrol.figures import hillshade_from_raster
    dem = _dem(tmp_path / "dem.tif", n=100)
    hs, ext = hillshade_from_raster(dem, max_px=25)
    assert hs.shape == (25, 25)
    assert ext == [400000.0, 403000.0, 3647000.0, 3650000.0]
    assert np.isnan(hs[:, :6]).all()          # nodata quarter stays transparent
    fin = hs[np.isfinite(hs)]
    assert fin.size and fin.min() >= 0 and fin.max() <= 1


def test_validation_figure_accepts_hillshade_tuple(tmp_path):
    from groundcontrol.assess import assess_products
    dem = _dem(tmp_path / "dsm.tif", hole=False)
    pts = _control_on(dem)
    sampled, stats, art = assess_products(
        pts, {"DSM": dem}, CRS, source_crs=CRS, outdir=tmp_path / "out",
        site_name="s", aoi=None, hs=None,          # hs derived from the product
        midas_velocities=False)                    # offline test: no network
    assert [p.name for p in art["figures"]] == ["s_validation_dz_DSM.png"]


# ------------------------------------------------------- review round 1 fixes

def test_resolve_bbox_duck_typed_like_the_original_dispatcher():
    """numpy scalars / Decimal / str bboxes worked on main; a 4-list of
    geometries is not a bbox and must fall through to the type error."""
    from decimal import Decimal
    from shapely.geometry import Point
    for seq in ([np.float32(-112), np.float32(32.6), np.float32(-111.5), np.float32(33)],
                [np.int32(-112), np.int32(32), np.int32(-111), np.int32(33)],
                (Decimal("-112"), Decimal("32.6"), Decimal("-111.5"), Decimal("33")),
                ["-112", "32.6", "-111.5", "33"]):
        b, poly = aoi_mod.resolve_aoi(seq)
        assert poly is None and b[0] == pytest.approx(-112) and b[3] == pytest.approx(33)
    with pytest.raises(TypeError, match="unsupported AOI type"):
        aoi_mod.resolve_aoi([Point(0, 0)] * 4)


def test_raster_footprint_rotated_and_untagged_nan(tmp_path):
    """Rotated (incl. exactly 90 deg) grids must not blow up the densifier;
    untagged NaN is documented as VALID (footprint = bounds)."""
    from rasterio.transform import Affine
    n = 40
    for deg in (30.0, 90.0, 180.0):
        p = tmp_path / f"rot{deg:g}.tif"
        t = Affine.translation(400000, 3650000) * Affine.rotation(deg) * Affine.scale(30, -30)
        with rasterio.open(p, "w", driver="GTiff", height=n, width=n, count=1,
                           dtype="float32", crs=CRS, transform=t) as d:
            d.write(np.ones((n, n), "float32"), 1)
        fp = aoi_mod.raster_footprint(p, max_px=20).to_crs(CRS).geometry.iloc[0]
        assert fp.area == pytest.approx((n * 30) ** 2, rel=1e-6)
        assert len(fp.exterior.coords) < 2000
    p = tmp_path / "nan_untagged.tif"
    arr = np.full((n, n), 400.0, "float32")
    arr[:, : n // 2] = np.nan
    with rasterio.open(p, "w", driver="GTiff", height=n, width=n, count=1, dtype="float32",
                       crs=CRS, transform=from_origin(400000.0, 3650000.0, 30.0, 30.0)) as d:
        d.write(arr, 1)
    assert aoi_mod.raster_footprint(p).to_crs(CRS).geometry.iloc[0].area \
        == pytest.approx((n * 30) ** 2, rel=1e-6)
    with rasterio.open(p, "r+") as d:  # the tag is what excludes the hole
        d.nodata = np.nan
    assert aoi_mod.raster_footprint(p).to_crs(CRS).geometry.iloc[0].area \
        == pytest.approx((n * 30) ** 2 / 2, rel=1e-6)


def test_raster_footprint_uses_band_1_mask(tmp_path):
    n = 40
    p = tmp_path / "two_band.tif"
    with rasterio.open(p, "w", driver="GTiff", height=n, width=n, count=2, dtype="float32",
                       crs=CRS, nodata=-9999.0,
                       transform=from_origin(400000.0, 3650000.0, 30.0, 30.0)) as d:
        b1 = np.full((n, n), 400.0, "float32")
        b1[:, : n // 2] = -9999.0
        d.write(b1, 1)
        d.write(np.full((n, n), 1.0, "float32"), 2)
    assert aoi_mod.raster_footprint(p).to_crs(CRS).geometry.iloc[0].area \
        == pytest.approx((n * 30) ** 2 / 2, rel=1e-6)


def test_has_vertical_axis_rule():
    """The ONE rule behind the CLI gate and transform_control's guard."""
    import pyproj
    from groundcontrol.assess import has_vertical_axis
    assert has_vertical_axis("EPSG:6341+5703")
    assert has_vertical_axis(pyproj.CRS("EPSG:32612").to_3d())
    assert has_vertical_axis("EPSG:4979")
    assert not has_vertical_axis("EPSG:32612")          # 2D
    assert not has_vertical_axis("EPSG:4978")           # geocentric XYZ
    time_wkt = pyproj.CRS.from_user_input(
        'COMPOUNDCRS["UTM12N + time",' + pyproj.CRS("EPSG:32612").to_wkt() +
        ',TIMECRS["Time",TDATUM["epoch",CALENDAR["proleptic Gregorian"],'
        'TIMEORIGIN[2000-01-01]],CS[TemporalDateTime,1],AXIS["time (T)",future]]]')
    assert not has_vertical_axis(time_wkt)              # compound, no height member
    assert not has_vertical_axis("EPSG:5703")           # vertical ALONE (round 2)
    assert not has_vertical_axis("EPSG:5715")


def test_embedded_gate_and_transform_guard_refuse_heightless_3axis_crs(tmp_path, monkeypatch):
    import pyproj
    from groundcontrol.assess import transform_control
    from groundcontrol.cli import assess_dem_main
    monkeypatch.setattr("groundcontrol.sources.fetch_control",
                        lambda *a, **k: pytest.fail("fetch must not run"))
    time_wkt = pyproj.CRS.from_user_input(
        'COMPOUNDCRS["UTM12N + time",' + pyproj.CRS("EPSG:32612").to_wkt() +
        ',TIMECRS["Time",TDATUM["epoch",CALENDAR["proleptic Gregorian"],'
        'TIMEORIGIN[2000-01-01]],CS[TemporalDateTime,1],AXIS["time (T)",future]]]').to_wkt()
    dem = _dem(tmp_path / "t.tif", wkt=time_wkt)
    with pytest.raises(SystemExit, match="without a height axis"):
        assess_dem_main(["--product", f"DSM={dem}", "--outdir", str(tmp_path / "out")])
    pts = _control_on(dem).to_crs("EPSG:6318")
    with pytest.raises(ValueError, match="heights would"):
        transform_control(pts, time_wkt)
    with pytest.raises(ValueError, match="heights would"):
        transform_control(pts, "EPSG:4978")
    with pytest.raises(ValueError, match="heights would"):   # bare vertical CRS
        transform_control(pts, "EPSG:5703")


def test_validation_map_without_aoi_zooms_to_points_not_product(tmp_path):
    """Auto hillshade covers the whole product; a bbox / no-AOI map must
    still frame the control points (review round 1)."""
    import matplotlib
    matplotlib.use("Agg")
    from groundcontrol.figures import _finish_map, hillshade_from_raster
    import matplotlib.pyplot as plt
    dem = _dem(tmp_path / "big.tif", hole=False, n=400)      # 12 km
    pts = _control_on(dem).to_crs(CRS)
    pts = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(
        [400100, 400400, 400700, 401000], [3649100, 3649300, 3649600, 3649900]), crs=CRS)
    hs, ext = hillshade_from_raster(dem, max_px=50)
    fig, ax = plt.subplots()
    ax.imshow(hs, extent=ext)
    _finish_map(ax, None, points=pts)
    x0, x1 = ax.get_xlim()
    assert x1 - x0 < 2000                                   # ~1 km + margin, not 12 km
    plt.close(fig)


def test_assess_bbox_aoi_clips_figures_to_the_bbox(tmp_path, monkeypatch):
    from groundcontrol import assess as assess_mod
    from groundcontrol.cli import assess_dem_main
    dem = _dem(tmp_path / "dsm.tif", wkt=_compound_wkt(), hole=False)
    monkeypatch.setattr("groundcontrol.sources.fetch_control",
                        lambda aoi, sources=(), **k: (_control_on(dem),
                                                      {s: {"n_rows": 4, "error": None}
                                                       for s in sources}))
    seen = {}
    real = assess_mod.assess_products

    def spy(*a, **k):
        seen["aoi"] = k.get("aoi")
        return real(*a, **k)

    monkeypatch.setattr(assess_mod, "assess_products", spy)
    rc = assess_dem_main(["--aoi=-112,32.6,-111.5,33.0", "--product", f"DSM={dem}",
                          "--source-crs", CRS, "--outdir", str(tmp_path / "out"),
                          "--no-figures"])
    assert rc == 0
    assert isinstance(seen["aoi"], gpd.GeoDataFrame) and seen["aoi"].crs.to_epsg() == 4326
    assert tuple(seen["aoi"].total_bounds) == pytest.approx((-112, 32.6, -111.5, 33.0))


def test_parse_aoi_path_with_comma(tmp_path):
    from groundcontrol.cli import _parse_aoi
    d = tmp_path / "site,2024"
    d.mkdir()
    dem = _dem(d / "dem.tif")
    assert _parse_aoi(dem) == dem
    with pytest.raises(SystemExit, match="file not found"):      # a typo'd path, not a bbox
        _parse_aoi("site,2024/missing.tif")
    with pytest.raises(SystemExit, match="expected minx,miny,maxx,maxy"):
        _parse_aoi("1,2,x,4")
    for typo in ("-112.0,32.6,-111.5,33.0x", "1,2,3,4.5.6", "-112.0,32.6,-111.5,33.0N"):
        with pytest.raises(SystemExit, match="expected minx,miny,maxx,maxy.*could not convert"):
            _parse_aoi(typo)   # a decimal bbox typo keeps the float diagnostic (round 3)
    # tilde and remote comma paths reach the path branch too (round 2)
    assert _parse_aoi("https://example.com/tiles/a,b.tif") == "https://example.com/tiles/a,b.tif"
    assert _parse_aoi("/vsicurl/https://x/a,b.tif") == "/vsicurl/https://x/a,b.tif"


def test_parse_aoi_tilde_path_with_comma(tmp_path, monkeypatch):
    from groundcontrol.cli import _parse_aoi
    monkeypatch.setenv("HOME", str(tmp_path))
    d = tmp_path / "site,2024"
    d.mkdir()
    dem = _dem(d / "dem.tif")
    assert _parse_aoi("~/site,2024/dem.tif") == dem


def test_figure_helpers_accept_raster_aoi_path(tmp_path):
    """The three figure entry points read a DEM path as an AOI (footprint)
    like every other entry point (round 1 finding 5; round 2: untested)."""
    from groundcontrol.assess import sample_products
    from groundcontrol.figures import (family_dz_figures, standard_control_figures,
                                       validation_dz_figures)
    dem = _dem(tmp_path / "dsm.tif", hole=False)
    pts = _control_on(dem)
    pts["h_ell"] = pts["height"]
    sampled = sample_products(pts, {"DSM": dem})
    out = validation_dz_figures(sampled, dem, tmp_path / "v", "s", products=("DSM",))
    assert out and out[0].exists()
    out = family_dz_figures(sampled, dem, tmp_path / "f", "s", products=("DSM",),
                            families=("3dep",))
    assert out and out[0].exists()
    pts["raw"] = "{}"                                 # datasheet facets read raw
    out = standard_control_figures(pts, dem, tmp_path / "c", "s",
                                   midas_velocities=False,  # offline test
                                   map_basemap=None)
    assert out and all(p.exists() for p in out)


def caplog_at_info():
    import contextlib
    import logging as _lg

    @contextlib.contextmanager
    def _cm():
        _lg.getLogger("groundcontrol.figures").setLevel(_lg.INFO)
        yield
    return _cm()


def test_family_panels_share_one_frame_including_all_gap_panel(tmp_path, monkeypatch):
    """Round 2: per-panel framing drew side-by-side maps at different
    scales; since 2026-08-30 an all-NaN (mosaic gap / vertically
    unassessable) subclass panel is omitted entirely."""
    import matplotlib.pyplot as plt
    from groundcontrol.assess import sample_products
    from groundcontrol.figures import family_dz_figures
    dem = _dem(tmp_path / "dsm.tif", hole=False, n=400)      # 12 km product
    pts = _control_on(dem)
    pts = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(         # 1 km cluster
        [400100, 400400, 400700, 401000], [3649100, 3649300, 3649600, 3649900]), crs=CRS)
    pts["h_ell"] = pts["height"]
    sampled = sample_products(pts, {"DSM": dem})
    sampled.loc[sampled["source"] == "ngs", "dh_DSM_before"] = np.nan   # the gap panel
    fams = {"two": ("TWO", [("A", lambda d: d["source"] == "ngs", "monument", "o"),
                           ("B", lambda d: d["source"] != "ngs", "NVA", "o")])}
    seen = {}
    real = plt.Figure.savefig

    def spy(fig, *a, **k):
        seen["xlims"] = [ax.get_xlim() for ax in fig.axes[:-1]
                         if ax.get_title().startswith(("A", "B"))]
        return real(fig, *a, **k)

    monkeypatch.setattr(plt.Figure, "savefig", spy)
    with caplog_at_info():
        family_dz_figures(sampled, None, tmp_path / "f", "s", products=("DSM",),
                          families=("two",), extra_families=fams,
                          hs_tif={"DSM": (np.ones((10, 10)),
                                          [390000, 420000, 3620000, 3660000])})
    assert "xlims" in seen, "family_dz_figures wrote no figure"
    xl = seen["xlims"]
    # the all-NaN (mosaic-gap) subclass panel is OMITTED with a log line
    # (owner 2026-08-30: never render a blank map); the surviving panel
    # still frames the points, not the product
    assert len(xl) == 1
    assert xl[0][1] - xl[0][0] < 2000                     # points (~1 km), not the product


def test_finish_map_edge_cases_do_not_raise():
    import matplotlib.pyplot as plt
    from groundcontrol.figures import _finish_map
    one_geo = gpd.GeoDataFrame(geometry=gpd.points_from_xy([-111.9], [33.0]), crs=4326)
    none_geo = gpd.GeoDataFrame(geometry=[None, None], crs=CRS)
    for pts, span in ((one_geo, 0.002), (none_geo, None)):
        fig, ax = plt.subplots()
        ax.imshow(np.ones((5, 5)), extent=[390000, 420000, 3620000, 3660000])
        _finish_map(ax, None, points=pts)
        if span is not None:
            x0, x1 = ax.get_xlim()
            assert x1 - x0 == pytest.approx(span)
        plt.close(fig)


def test_hillshade_from_raster_declines_south_up_and_degenerate(tmp_path):
    from rasterio.transform import Affine
    from groundcontrol.figures import hillshade_from_raster
    for name, t in (("south_up", Affine(30, 0, 400000, 0, 30, 3650000)),
                    ("flat", Affine(30, 0, 400000, 0, 0, 3650000))):
        p = tmp_path / f"{name}.tif"
        with rasterio.open(p, "w", driver="GTiff", height=20, width=20, count=1,
                           dtype="float32", crs=CRS, transform=t) as d:
            d.write(np.ones((20, 20), "float32"), 1)
        assert hillshade_from_raster(p) is None


def test_raster_footprint_small_patch_survives_decimation(tmp_path):
    """A nodata-tagged raster with a few valid pixels keeps a footprint at
    coarse decimation (average read; nearest dropped it) and the error for a
    truly empty raster names the decimation."""
    n = 256
    p = tmp_path / "sparse.tif"
    arr = np.full((n, n), -9999.0, "float32")
    arr[100:103, 100:103] = 1.0
    with rasterio.open(p, "w", driver="GTiff", height=n, width=n, count=1, dtype="float32",
                       crs=CRS, nodata=-9999.0,
                       transform=from_origin(400000.0, 3650000.0, 30.0, 30.0)) as d:
        d.write(arr, 1)
    fp = aoi_mod.raster_footprint(p, max_px=8).to_crs(CRS).geometry.iloc[0]
    assert fp.area > 0 and fp.covers(aoi_mod.raster_footprint(p, max_px=256)
                                     .to_crs(CRS).geometry.iloc[0])
    arr[:] = -9999.0
    with rasterio.open(p, "r+") as d:
        d.write(arr, 1)
    with pytest.raises(ValueError, match="no valid pixels at 1/32 decimation"):
        aoi_mod.raster_footprint(p, max_px=8)


def test_raster_footprint_warns_when_mask_comes_from_overviews(tmp_path, caplog):
    """Round 4: nearest-built overviews (gdaladdo default) can drop valid
    patches before the decimated read; the footprint says so."""
    from rasterio.enums import Resampling
    n = 256
    p = tmp_path / "ovr.tif"
    arr = np.full((n, n), -9999.0, "float32")
    arr[:128] = 1.0
    with rasterio.open(p, "w", driver="GTiff", height=n, width=n, count=1, dtype="float32",
                       crs=CRS, nodata=-9999.0,
                       transform=from_origin(400000.0, 3650000.0, 30.0, 30.0)) as d:
        d.write(arr, 1)
    with rasterio.open(p, "r+") as d:
        d.build_overviews([2, 4, 8], Resampling.nearest)
    with caplog.at_level("WARNING", logger="groundcontrol.aoi"):
        fp = aoi_mod.raster_footprint(p, max_px=32)
    assert "overview pyramid" in caplog.text
    assert fp.to_crs(CRS).geometry.iloc[0].area == pytest.approx((n * 30) ** 2 / 2, rel=0.05)


def test_fetch_cli_reads_aoi_only_after_output_checks(tmp_path, monkeypatch):
    from groundcontrol import aoi as aoi_pkg
    from groundcontrol.cli import fetch_control_main
    dem = _dem(tmp_path / "dsm.tif")
    monkeypatch.setattr(aoi_pkg, "raster_footprint",
                        lambda *a, **k: pytest.fail("AOI read before the --out check"))
    with pytest.raises(SystemExit, match="unsupported export format"):
        fetch_control_main(["--aoi", dem, "--sources", "ngs", "--out", str(tmp_path / "c.shp")])


def test_hillshade_from_raster_declines_rotated_grid(tmp_path, caplog):
    from rasterio.transform import Affine
    from groundcontrol.figures import hillshade_from_raster
    p = tmp_path / "rot.tif"
    t = Affine.translation(400000, 3650000) * Affine.rotation(30) * Affine.scale(30, -30)
    with rasterio.open(p, "w", driver="GTiff", height=20, width=20, count=1, dtype="float32",
                       crs=CRS, transform=t) as d:
        d.write(np.ones((20, 20), "float32"), 1)
    with caplog.at_level("WARNING", logger="groundcontrol.figures"):
        assert hillshade_from_raster(p) is None
    assert "rotated" in caplog.text


# --------------------------------------------- standard GNSS/FAA contact sheets

def test_assess_writes_context_sheets_for_gnss_and_faa(tmp_path):
    """assess_products' figure bundle includes contact sheets for the GNSS
    and FAA subsets (owner 2026-08-29: standard, not opt-in); none for a
    3DEP/NGS-only control set."""
    import json
    from groundcontrol.assess import assess_products
    dem = _dem(tmp_path / "dsm.tif", hole=False)
    pts = _control_on(dem)                                   # has one gnss_campaign
    faa_row = pts.iloc[[0]].copy()
    faa_row["source"] = "faa"
    faa_row["point_type"] = "runway_end"
    faa_row["raw"] = json.dumps({"pos_class": "surveyed"})
    pts = gpd.GeoDataFrame(pd.concat([pts, faa_row], ignore_index=True), crs=CRS)
    pts["id"] = [f"P{i}" for i in range(len(pts))]
    _, _, art = assess_products(pts, {"DSM": dem}, CRS, source_crs=CRS,
                                outdir=tmp_path / "out", site_name="s",
                                basemap=None,      # offline test: no tile fetch
                                midas_velocities=False)
    names = sorted(p.name for p in art["context_sheets"])
    # per-source subsets x the two standard tiers (recovered spec 2026-08-29)
    assert names == [f"s_{sub}_gallery_{tier}.png"
                     for sub in ("3dep_nva", "3dep_vva", "faa_runway", "opus")
                     for tier in ("120m", "30m")]
    # ... routed into per-SOURCE subdirs (owner layout 2026-08-30)
    assert all(p.exists() for p in art["context_sheets"])
    dirs = {p.parent.name for p in art["context_sheets"]}
    assert dirs == {"3dep", "gnss", "faa"}

    dense = pts[pts["point_type"] == "monument"]   # NGS monuments: never sheeted
    _, _, art2 = assess_products(dense, {"DSM": dem}, CRS, source_crs=CRS,
                                 outdir=tmp_path / "out2", site_name="d",
                                 basemap=None, midas_velocities=False)
    assert "context_sheets" not in art2
    assert not list((tmp_path / "out2").glob("*gallery*"))


def test_context_sheets_layer_stack_order(tmp_path, monkeypatch):
    """Owner 2026-08-29: relief alone is not enough — the standard stack is
    RGB | intensity (when given) | one relief per product, in that order."""
    from groundcontrol import figures
    dem = _dem(tmp_path / "dsm.tif", hole=False)
    dtm = _dem(tmp_path / "dtm.tif", hole=False)
    ortho = _dem(tmp_path / "ortho.tif", hole=False)
    inten = _dem(tmp_path / "intensity.tif", hole=False)
    pts = _control_on(dem)
    pts["id"] = [f"P{i}" for i in range(len(pts))]
    seen = {}

    def spy(points, layers, *a, **k):
        seen["layers"] = layers
        return []

    monkeypatch.setattr(figures, "point_context_gallery", spy)
    figures.context_sheets(pts, {"DSM": dem, "DTM": dtm}, tmp_path, "s",
                           rgb=ortho, intensity=inten, basemap=None)
    assert [(t, k) for t, _, k in seen["layers"]] == [
        ("RGB ortho", "rgb"), ("intensity", "gray"),
        ("DSM relief", "relief"), ("DTM relief", "relief")]
    assert seen["layers"][0][1] == ortho          # single path, no chain
    figures.context_sheets(pts, {"DSM": dem}, tmp_path, "s", basemap=None)
    assert [(t, k) for t, _, k in seen["layers"]] == [("DSM relief", "relief")]


def test_point_context_gallery_leaves_preopen_datasets_open(tmp_path):
    """A chain entry that is an already-open dataset (the web-basemap
    WarpedVRT pattern) is read in place and NOT closed by the gallery."""
    import rasterio
    from groundcontrol.figures import point_context_gallery
    dem = _dem(tmp_path / "dsm.tif", hole=False)
    pts = _control_on(dem).iloc[:2]
    pts["id"] = ["A", "B"]
    with rasterio.open(dem) as pre:
        pages = point_context_gallery(
            pts, [("rgb-ish", [str(tmp_path / "dsm.tif"), pre], "gray"),
                  ("relief", dem, "relief")],
            tmp_path, "s", subset_tag="pre")
        assert pages and pages[0].exists()
        assert not pre.closed                     # caller still owns it
        pre.read(1)                               # and it still reads


@pytest.mark.network
def test_open_web_basemap_reads_real_tiles():
    """Esri World Imagery through GDAL_WMS + WarpedVRT: nonzero pixels in
    the product frame at the native z19 ground resolution."""
    from groundcontrol.figures import open_web_basemap
    got = open_web_basemap("EPSG:32612", (423000, 3628800, 423400, 3629200),
                           margin_m=50)
    assert got is not None
    base, vrt = got
    try:
        arr = vrt.read(window=((0, 200), (0, 200)))
        assert arr.shape[0] == 3 and (arr != 0).mean() > 0.5
    finally:
        vrt.close()
        base.close()


def test_gallery_reads_warpedvrt_layers(tmp_path, caplog):
    """WarpedVRT (the web-basemap panel) refuses boundless reads; the window
    reader must fall back to a clamped read instead of an 'unavailable'
    panel — including for a point near the dataset edge."""
    import rasterio
    from rasterio.vrt import WarpedVRT
    from groundcontrol.figures import point_context_gallery
    dem = _dem(tmp_path / "dsm.tif", hole=False)
    with rasterio.open(dem) as base:
        b = base.bounds
    pts = gpd.GeoDataFrame(
        {"id": ["mid", "edge"]},
        geometry=gpd.points_from_xy([(b.left + b.right) / 2, b.left + 40.0],
                                    [(b.bottom + b.top) / 2, b.top - 40.0]), crs=CRS)
    with rasterio.open(dem) as base, WarpedVRT(base, crs=CRS) as vrt:
        with caplog.at_level("WARNING", logger="groundcontrol.figures"):
            pages = point_context_gallery(pts, [("web", vrt, "gray")],
                                          tmp_path, "s", subset_tag="w")
    assert pages and pages[0].exists()
    assert "panel failed" not in caplog.text


def test_fetch_context_sheets_from_aoi_only(tmp_path, monkeypatch):
    """AOI-only workflow (no DEM, no intensity, geographic landing): the
    fetch CLI's --context-sheets renders RGB-basemap-only sheets, with the
    basemap built in the estimated UTM (owner 2026-08-29)."""
    import rasterio
    from rasterio.vrt import WarpedVRT
    from groundcontrol import figures
    from groundcontrol.cli import fetch_control_main
    dem = _dem(tmp_path / "fakeweb.tif", hole=False)
    pts = _control_on(dem).to_crs("EPSG:6318")     # the fetch landing (geographic)
    pts["id"] = [f"P{i}" for i in range(len(pts))]
    monkeypatch.setattr("groundcontrol.sources.fetch_control",
                        lambda aoi, sources=(), **k:
                        (pts, {s: {"n_rows": len(pts), "error": None} for s in sources}))
    seen = {}

    def fake_web(crs, bounds, **kw):
        import pyproj
        seen["crs"] = pyproj.CRS.from_user_input(crs)
        base = rasterio.open(dem)
        return base, WarpedVRT(base, crs=crs if hasattr(crs, "to_wkt")
                               and crs.is_projected else "EPSG:32612")

    monkeypatch.setattr(figures, "open_web_basemap", fake_web)

    def _no_net(*a, **k):   # offline test: the MIDAS block takes its skip path
        raise OSError("offline test")
    monkeypatch.setattr("groundcontrol.sources.ngl.read_midas", _no_net)
    rc = fetch_control_main(["--aoi=-112,32.6,-111.5,33.0", "--sources", "ngs",
                             "--out", str(tmp_path / "ctl.parquet"),
                             "--context-sheets"])
    assert rc == 0
    assert seen["crs"].is_projected                # UTM, not the 6318 landing
    pages = sorted(p.name for p in tmp_path.rglob("ctl_*_gallery_*.png"))
    assert pages == sorted(f"ctl_{sub}_gallery_{tier}.png"
                           for sub in ("3dep_nva", "3dep_vva", "opus")
                           for tier in ("120m", "30m"))
    assert (tmp_path / "gnss" / "ctl_opus_gallery_120m.png").exists()
    # the labeled all-sources control map accompanies the sheets (2026-08-30)
    assert (tmp_path / "ctl_control_map.png").exists()


def test_web_placeholder_chroma_rule_scoped_to_basemap_sources(tmp_path, caplog):
    """A pure-achromatic window from a WEB-BASEMAP source is a provider
    placeholder (measured chroma 0.0 vs >=24 real; Nepal 2026-08-30) and
    falls through the chain — but a grayscale USER ortho (KH-9) is
    legitimate imagery and must render."""
    import rasterio
    from groundcontrol.figures import point_context_gallery
    n = 40
    gray = np.full((3, n, n), 204, dtype="uint8")          # placeholder-like
    color = np.random.default_rng(0).integers(0, 255, (3, n, n), dtype="uint8")
    for name, arr in (("gray.tif", gray), ("color.tif", color)):
        with rasterio.open(tmp_path / name, "w", driver="GTiff", height=n, width=n,
                           count=3, dtype="uint8", crs=CRS,
                           transform=from_origin(400000.0, 3650000.0, 3.0, 3.0)) as d:
            d.write(arr)
    pts = gpd.GeoDataFrame({"id": ["A"]},
                           geometry=gpd.points_from_xy([400060.0], [3649940.0]), crs=CRS)
    with rasterio.open(tmp_path / "gray.tif") as fake_web, \
            rasterio.open(tmp_path / "color.tif") as real:
        fake_web.gc_web_basemap = True                     # the basemap marker
        with caplog.at_level("INFO", logger="groundcontrol.figures"):
            pages = point_context_gallery(
                pts, [("rgb", [fake_web, real], "rgb")], tmp_path, "s",
                subset_tag="w")
        assert pages and "fallback source 1 used" in caplog.text  # placeholder skipped
    with rasterio.open(tmp_path / "gray.tif") as user_ortho:      # NO marker
        caplog.clear()
        with caplog.at_level("INFO", logger="groundcontrol.figures"):
            point_context_gallery(pts, [("rgb", [user_ortho, str(tmp_path / "color.tif")],
                                         "rgb")], tmp_path, "s2", subset_tag="w")
        assert "fallback source 1 used" not in caplog.text        # grayscale renders


def test_gnss_timeseries_figures_from_fixture(tmp_path, monkeypatch):
    """Standard TOP-LEVEL NGL series figures (owner 2026-08-30): the E/N/U
    common panels and the per-station vertical small multiples (MIDAS-rate
    fit, earthquake + equipment steps marked); both skip cleanly when a
    station's series is unavailable."""
    import json
    from pathlib import Path as _P

    from groundcontrol.figures import gnss_station_series, gnss_timeseries
    from groundcontrol.sources import ngl as ngl_mod
    fixture = _P("tests/data/ngl_CLV1_IGS14_sample.tenv3").read_text()

    def fake_read(sid, frame="IGS14", **kw):
        if sid == "GONE":
            raise OSError("no cache, no network")
        return ngl_mod.parse_tenv3(fixture)

    monkeypatch.setattr(ngl_mod, "read_tenv3", fake_read)
    st = gpd.GeoDataFrame({
        "source": ["ngl", "ngl", "ngs"],
        "id": ["CLV1", "GONE", "XX"],
        "point_type": ["gnss_cont", "gnss_cont", "monument"],
        "vel_e": [0.002, np.nan, np.nan],
        "vel_n": [-0.003, np.nan, np.nan],
        "vel_u": [-0.004, -0.001, np.nan],
        "raw": [json.dumps({"eq_steps": [2017.9], "equip_steps": [2017.95]}),
                json.dumps({}), None],
    }, geometry=gpd.points_from_xy([-115.25, -115.2, -115.1], [36.2, 36.3, 36.1]),
        crs="EPSG:9000")
    fp = gnss_timeseries(st, tmp_path, "s")
    assert fp is not None and fp.exists()
    assert fp.name == "s_gnss_timeseries.png"
    fp2 = gnss_station_series(st, tmp_path, "s")
    assert fp2 is not None and fp2.exists()
    assert fp2.name == "s_gnss_station_series.png"
    # no NGL rows -> None, no file
    assert gnss_timeseries(st[st.source == "ngs"], tmp_path, "t") is None
    assert gnss_station_series(st[st.source == "ngs"], tmp_path, "t") is None


def test_raster_footprint_caps_fragmented_mosaics(tmp_path, monkeypatch, caplog):
    """A strip/tile mosaic that polygonizes into thousands of pieces falls
    back to the valid-data bounds box with a warning (rasuwa corridor
    finding 2026-08-31) instead of an unbounded unary_union."""
    import rasterio
    from affine import Affine

    from groundcontrol import aoi as A
    arr = np.zeros((16, 16), dtype="float32")
    arr[::2, ::2] = 1.0                       # 64 disjoint single-pixel patches
    path = tmp_path / "checker.tif"
    with rasterio.open(path, "w", driver="GTiff", height=16, width=16, count=1,
                       dtype="float32", crs="EPSG:32612", nodata=0.0,
                       transform=Affine(1.0, 0.0, 500000, 0.0, -1.0, 3800016)) as dst:
        dst.write(arr, 1)
    monkeypatch.setattr(A, "FOOTPRINT_MAX_PIECES", 8)
    with caplog.at_level("WARNING", logger="groundcontrol.aoi"):
        gdf = A.raster_footprint(path)
    assert "simplified to the valid-data bounds box" in caplog.text
    geom = gdf.to_crs("EPSG:32612").geometry.iloc[0]
    # one simple box spanning the valid block (rows/cols 0..14 inclusive)
    assert geom.geom_type == "Polygon"
    np.testing.assert_allclose(geom.bounds,                # 4326 round-trip
                               (500000.0, 3800001.0, 500015.0, 3800016.0),
                               atol=1e-4)
    # under the cap the exact multipart footprint is kept
    monkeypatch.setattr(A, "FOOTPRINT_MAX_PIECES", 2000)
    exact = A.raster_footprint(path).to_crs("EPSG:32612").geometry.iloc[0]
    assert exact.geom_type == "MultiPolygon"
    assert len(exact.geoms) == 64
