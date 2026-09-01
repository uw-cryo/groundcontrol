"""Regression tests for the 2026-08-30 adversarial-audit fixes (H1-H10).

Each test pins the exact failure scenario its finding was proven with,
kept offline: ellipsoidal-only transform chains (no geoid grids), synthetic
rasters, and monkeypatched caches. The findings live in the audit report
(gitignored review/); the commit messages on this branch carry the
per-finding failure scenarios.
"""

import numpy as np
import pandas as pd
import pytest

pyproj = pytest.importorskip("pyproj")
gpd = pytest.importorskip("geopandas")

from pyproj import CRS  # noqa: E402
from shapely.geometry import Point  # noqa: E402

from groundcontrol.geodesy import (  # noqa: E402
    is_wgs84_ensemble,
    rebase_projection_2d,
    rebase_projection_3d,
    with_vdatum,
)

# ---------------------------------------------------------------------------
# H3 — is_wgs84_ensemble must match ONLY the WGS 84 ensemble
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code,expected", [
    (4326, True),     # WGS 84 ensemble geographic
    (32645, True),    # WGS 84 ensemble UTM
    (25833, False),   # ETRS89 ensemble — 0.76 m from ITRF2014 at Berlin
    (4258, False),    # ETRS89 geographic
    (4747, False),    # GR96 (Greenland) ensemble
    (6318, False),    # NAD83(2011)
    (7912, False),    # ITRF2014 (realization, 3D)
])
def test_is_wgs84_ensemble_scoped_to_wgs84(code, expected):
    assert is_wgs84_ensemble(CRS.from_epsg(code)) is expected


def test_etrs89_horizontal_never_rebased_to_itrf():
    # the audit's probed consequence: ETRS89 + EVRF2000 silently became
    # ITRF2014 / UTM 33N (dE +0.594, dN +0.465 m at Berlin, epoch 2020)
    out = with_vdatum(CRS.from_epsg(25833), "EPSG:5730")
    assert "ETRS89" in out.name
    assert "ITRF" not in out.name


# ---------------------------------------------------------------------------
# H1 — rebase_projection_2d/3d must keep the source Cartesian CS (units)
# ---------------------------------------------------------------------------


def test_rebase_preserves_ftus_axes_and_coordinates():
    # EPSG:32664 (BLM 14N, US survey foot) is both ftUS and WGS84-ensemble
    c2 = rebase_projection_2d(CRS.from_epsg(32664), 9000, "ITRF2014")
    assert c2.axis_info[0].unit_name == "US survey foot"
    c3 = rebase_projection_3d(CRS.from_epsg(32664), 9000, "ITRF2014")
    assert c3.axis_info[0].unit_name == "US survey foot"
    # native coordinates survive the rebase (the bug was a 3.28x scale)
    from pyproj import Transformer
    src = CRS.from_epsg(32664)
    lon, lat = Transformer.from_crs(src, src.geodetic_crs,
                                    always_xy=True).transform(1640416.67, 1e6)
    e2, n2 = Transformer.from_crs(c2.geodetic_crs, c2,
                                  always_xy=True).transform(lon, lat)
    assert e2 == pytest.approx(1640416.67, abs=0.01)
    assert n2 == pytest.approx(1e6, abs=0.01)


def test_rebase_metre_grids_unchanged():
    for code in (32610, 3413, 3031):
        assert rebase_projection_2d(
            CRS.from_epsg(code), 9000,
            "ITRF2014").axis_info[0].unit_name == "metre"


# ---------------------------------------------------------------------------
# MED — with_vdatum('ellipsoidal') recursed forever appending ':itrf2014'
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("horizontal", [32610, 32645])  # realized + ensemble
def test_with_vdatum_bad_ellipsoid_token_raises_once(horizontal):
    with pytest.raises(ValueError, match="ellipsoid"):
        with_vdatum(CRS.from_epsg(horizontal), "ellipsoidal")


def test_with_vdatum_orthometric_on_ensemble_still_recurses_once():
    out = with_vdatum(CRS.from_epsg(32645), "EPSG:3855")
    assert "ITRF2014" in out.name and "EGM2008" in out.name


# ---------------------------------------------------------------------------
# H2 — compound orthometric product CRS through the CLI resolvers
# ---------------------------------------------------------------------------


def _write_tif(path, crs_input):
    import rasterio
    from rasterio.transform import from_origin

    crs = pyproj.CRS.from_user_input(crs_input)
    with rasterio.open(path, "w", driver="GTiff", width=4, height=4,
                       count=1, dtype="float32", crs=crs.to_wkt(),
                       transform=from_origin(340000, 3120000, 30, 30)) as dst:
        dst.write(np.full((1, 4, 4), 1500.0, np.float32))
    return path


def test_cli_vdatum_refuses_compound_orthometric_product(tmp_path):
    from groundcontrol.cli import _vdatum_target_crs
    p = _write_tif(tmp_path / "cop30.tif", "EPSG:32645+3855")
    # following the OLD refusal's advice stripped EGM2008 (-37.35 m at
    # Rasuwa); the product declares its heights, so --vdatum must refuse
    with pytest.raises(ValueError, match="declares its heights"):
        _vdatum_target_crs({"cop30": str(p)}, "ellipsoid:itrf2014")


def test_cli_embedded_accepts_compound_orthometric_on_ensemble(tmp_path):
    from groundcontrol.cli import _embedded_target_crs
    p = _write_tif(tmp_path / "cop30.tif", "EPSG:32645+3855")
    wkt = _embedded_target_crs({"cop30": str(p)})
    out = pyproj.CRS.from_wkt(wkt)
    # EGM2008 preserved; ambiguous ensemble legs rebased to ITRF2014
    assert "EGM2008" in out.name
    assert "ITRF2014" in out.name
    vert = pyproj.CRS(out.sub_crs_list[1])
    assert vert.is_vertical and "EGM2008" in vert.name


def test_cli_embedded_still_refuses_3d_ellipsoidal_ensemble(tmp_path):
    from groundcontrol.cli import _embedded_target_crs
    p = _write_tif(tmp_path / "e.tif",
                   pyproj.CRS.from_epsg(32645).to_3d())
    with pytest.raises(SystemExit, match="ENSEMBLE"):
        _embedded_target_crs({"e": str(p)})


# ---------------------------------------------------------------------------
# H4 — per-row vertical guard for 3D (non-compound) sources; NA refused
# ---------------------------------------------------------------------------

UTM12_3D = pyproj.CRS.from_epsg(32612).to_3d()  # ellipsoidal target, no grids


def _ctl(vertical_values, crs="EPSG:6319"):
    n = len(vertical_values)
    return gpd.GeoDataFrame({
        "height": [500.0] * n,
        "vertical_crs": pd.array(vertical_values, dtype="string"),
    }, geometry=[Point(-111.5, 34.5)] * n, crs=crs)


def test_vertical_guard_covers_3d_noncompound_source():
    from groundcontrol.assess import transform_control
    ctl = _ctl(["EPSG:5703", "EPSG:5703", "EPSG:6319"])
    out, info = transform_control(ctl, UTM12_3D, source_crs="EPSG:6319")
    # orthometric rows are masked under the ellipsoidal chain (the guard
    # was silently inert here: ~30 m geoid applied, counted valid)
    assert info["n_vertical_excluded"] == 2
    h = out["h_ell"].to_numpy()
    assert np.isnan(h[:2]).all() and np.isfinite(h[2])


def test_vertical_guard_refuses_na():
    from groundcontrol.assess import transform_control
    ctl = _ctl(["EPSG:6319", pd.NA])
    out, info = transform_control(ctl, UTM12_3D, source_crs="EPSG:6319")
    assert info["n_vertical_excluded"] == 1
    assert np.isnan(out["h_ell"].to_numpy()[1])


def test_vertical_note_survives_mixed_na_and_codes():
    # caught on a real cache (COP30_E rerun): >= 2 distinct incompatible
    # values including NA made the vertical_note's sorted() raise
    # "boolean value of NA is ambiguous"
    from groundcontrol.assess import transform_control
    ctl = _ctl([pd.NA, "EPSG:5703", "EPSG:6319"])
    out, info = transform_control(ctl, UTM12_3D, source_crs="EPSG:6319")
    assert info["n_vertical_excluded"] == 2
    assert "<NA>" in info["vertical_note"]


def test_vertical_guard_accepts_unit_variant_code_same_datum():
    # EPSG:6360 (NAVD88 ftUS code) was falsely excluded by code-string
    # equality; datum identity accepts it. NO scaling: schema says the
    # height column is always metres — vertical_crs records the ORIGINAL
    # datum/unit provenance (round-2 audit: scaling silently divided
    # schema-compliant metre heights by 3.28)
    from groundcontrol.assess import transform_control
    ctl = gpd.GeoDataFrame({
        "height": [500.0, 500.0],  # both metres, per schema
        "vertical_crs": pd.array(["EPSG:5703", "EPSG:6360"], dtype="string"),
    }, geometry=[Point(-111.5, 34.5)] * 2, crs="EPSG:6318")
    # identity compound chain: no geoid grid needed, h_ell == height in m
    out, info = transform_control(ctl, "EPSG:6318+5703",
                                  source_crs="EPSG:6318+5703")
    h = out["h_ell"].to_numpy()
    assert info["n_vertical_excluded"] == 0
    assert h[0] == pytest.approx(500.0) and h[1] == pytest.approx(500.0)


def test_transform_control_refuses_depth_target():
    # round-5 audit: a depth-type vertical target landed +1500 m control
    # at h_ell = -1500 silently on every non-ensemble path
    from groundcontrol.assess import transform_control
    ctl = gpd.GeoDataFrame({"height": [1500.0]},
                           geometry=[Point(-111.5, 34.5)], crs="EPSG:26911")
    with pytest.raises(ValueError, match="DEPTH"):
        transform_control(ctl, "EPSG:26911+5715", source_crs="EPSG:26911+5715")


def test_compound_vertical_discrimination():
    # _gravity_height was never exercised by the suite through four audit
    # rounds (round-5 INFO); pin the verified matrix
    from pyproj.crs import CompoundCRS

    from groundcontrol.cli import _compound_vertical
    utm = pyproj.CRS.from_epsg(32645)

    def comp(vwkt_or_code):
        v = (pyproj.CRS.from_epsg(vwkt_or_code)
             if isinstance(vwkt_or_code, int)
             else pyproj.CRS(vwkt_or_code))
        return pyproj.CRS(CompoundCRS(name="x", components=[utm, v]))

    def vert(name):
        return (f'VERTCRS["{name}",VDATUM["{name} datum"],'
                'CS[vertical,1],AXIS["Up",up],LENGTHUNIT["metre",1]]')

    # ellipsoidal-family names (incl. post-WKT1 'Up' axes): excluded
    for nm in ("Height above ellipsoid", "Ellipsoid height",
               "Ellipsoidal height (unrealized)", "WGS84 ellipsoid"):
        assert _compound_vertical(comp(vert(nm))) is None, nm
    # depth: excluded even when WKT1 erased the direction
    assert _compound_vertical(comp(5715)) is None
    assert _compound_vertical(comp(vert("Local chart depth"))) is None
    # genuine geoids, registered or not: datum-defining
    for nm in ("Nepal Geoid 2020 height", "EGM96 height", "NAVD88 height"):
        assert _compound_vertical(comp(vert(nm))) is not None, nm
    assert _compound_vertical(comp(3855)) is not None


def test_transform_control_refuses_empty_frame():
    from groundcontrol.assess import transform_control
    ctl = _ctl(["EPSG:6319"]).iloc[0:0]
    with pytest.raises(ValueError, match="empty"):
        transform_control(ctl, UTM12_3D, source_crs="EPSG:6319")


def test_transform_control_duplicate_index_labels():
    # rows must be INCOMPATIBLE with usable natives so the native
    # re-target loop (the code that crashed on duplicate labels via
    # get_indexer) actually runs (round-2 audit: a compatible-rows
    # version was vacuous)
    from groundcontrol.assess import transform_control
    ctl = gpd.GeoDataFrame({
        "height": [500.0, 501.0],
        # ITRF2014 rows under a NAD83(2011) 3D source: incompatible
        "vertical_crs": pd.array(["EPSG:7912"] * 2, dtype="string"),
        "native_x": [-111.5, -111.501], "native_y": [34.5, 34.501],
        "native_h": [500.0, 501.0],
        "native_crs": pd.array(["EPSG:7912"] * 2, dtype="string"),
        "coord_epoch": [2020.0, 2020.0],
    }, geometry=[Point(-111.5, 34.5), Point(-111.501, 34.501)],
        crs="EPSG:6319", index=[0, 0])  # two caches concatenated
    out, info = transform_control(ctl, UTM12_3D, source_crs="EPSG:6319")
    assert info["n_vertical_native"] == 2
    assert np.isfinite(out["h_ell"].to_numpy()).all()


# ---------------------------------------------------------------------------
# H6 — an empty steps cache must read as NOT checked; atomic cache writes
# ---------------------------------------------------------------------------


def test_empty_steps_cache_is_not_checked(tmp_path, monkeypatch):
    monkeypatch.setenv("GROUNDCONTROL_CACHE_DIR", str(tmp_path))
    (tmp_path / "ngl_steps.txt").write_text("")  # interrupted download
    from groundcontrol.sources import ngl
    stations = [{"meta": {"sta": "CHLM"}}]
    ngl._attach_steps(stations)
    # eq_steps ABSENT = not checked -> propagate_epoch counts it
    # step_unchecked; [] would read as "checked, no Gorkha steps"
    assert "eq_steps" not in stations[0]["meta"]


def test_cache_write_atomic_and_typed(tmp_path):
    from groundcontrol.sources.checkpoints_3dep import cache_write
    p = tmp_path / "c.txt"
    cache_write(p, "text")
    assert p.read_text() == "text"
    cache_write(p, b"bytes")
    assert p.read_bytes() == b"bytes"
    assert list(tmp_path.iterdir()) == [p]  # no temp litter


def test_refresh_env_zero_is_not_a_refresh(tmp_path, monkeypatch):
    from groundcontrol.sources.checkpoints_3dep import cache_stale
    p = tmp_path / "c.txt"
    p.write_text("x")
    monkeypatch.setenv("GROUNDCONTROL_REFRESH", "0")
    assert cache_stale(p, None) is False
    monkeypatch.setenv("GROUNDCONTROL_REFRESH", "1")
    assert cache_stale(p, None) is True


# ---------------------------------------------------------------------------
# H7/H8 — antimeridian / polar footprints fail loud, never a global AOI
# ---------------------------------------------------------------------------


def _mk_raster(path, crs, x0, y0, px, n=50):
    import rasterio
    from rasterio.transform import from_origin
    with rasterio.open(path, "w", driver="GTiff", width=n, height=n,
                       count=1, dtype="float32", crs=crs, nodata=-9999.0,
                       transform=from_origin(x0, y0, px, px)) as dst:
        dst.write(np.full((1, n, n), 100.0, np.float32))
    return str(path)


def test_antimeridian_raster_footprint_raises(tmp_path):
    from groundcontrol.aoi import raster_footprint
    fiji = _mk_raster(tmp_path / "fiji.tif", "EPSG:32760",
                      680000, 7910000, 4000)
    with pytest.raises(ValueError, match="antimeridian"):
        raster_footprint(fiji)


def test_pole_raster_footprint_raises(tmp_path):
    from groundcontrol.aoi import raster_footprint
    pole = _mk_raster(tmp_path / "pole.tif", "EPSG:3031",
                      -100000, 100000, 4000)
    with pytest.raises(ValueError, match="pole|antimeridian"):
        raster_footprint(pole)


def test_global_web_mercator_raster_still_resolves(tmp_path):
    # round-2: a LEGITIMATELY global projected raster spans ~360 deg with
    # a valid ring that contains its own centre — must not be refused
    from groundcontrol.aoi import resolve_aoi
    world = _mk_raster(tmp_path / "world.tif", "EPSG:3857",
                       -20037508.34, 20037508.34, 2 * 20037508.34 / 50)
    b, _poly = resolve_aoi(world)
    assert (b[2] - b[0]) > 350.0


def test_off_meridian_antarctic_raster_still_resolves(tmp_path):
    # MDV/Taylor Valley regression: EPSG:3031 near 162.5E must keep working
    from groundcontrol.aoi import resolve_aoi
    mdv = _mk_raster(tmp_path / "mdv.tif", "EPSG:3031",
                     350000, -1300000, 200)
    b, _poly = resolve_aoi(mdv)
    assert (b[2] - b[0]) < 5.0


def test_union_footprints_names_offending_raster(tmp_path):
    from groundcontrol.aoi import union_footprints
    fiji = _mk_raster(tmp_path / "fiji.tif", "EPSG:32760",
                      680000, 7910000, 4000)
    ok = _mk_raster(tmp_path / "ok.tif", "EPSG:32610",
                    550000, 5280000, 30)
    with pytest.raises(ValueError, match="fiji"):
        union_footprints([fiji, ok])


def test_resolve_aoi_accepts_ndarray_bbox():
    from groundcontrol.aoi import resolve_aoi
    b, poly = resolve_aoi(np.array([-115.5, 35.8, -114.8, 36.5]))
    assert b == (-115.5, 35.8, -114.8, 36.5) and poly is None


# ---------------------------------------------------------------------------
# H9 — the military datum check must return NaN outside NAVD88 coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lon,lat", [
    (-157.92, 21.32),   # Hickam AFB, Hawaii — was +14.28 m of fiction
    (85.3, 28.1),       # Nepal — was -38.89
    (-30.0, 30.0),      # mid-Atlantic — was +59.81
])
def test_egm96_navd88_delta_nan_outside_coverage(lon, lat):
    from groundcontrol.figures import _egm96_navd88_delta
    assert np.isnan(_egm96_navd88_delta(lon, lat, 10.0))


# ---------------------------------------------------------------------------
# MED — _grid_signature must see prime meridian and axis units
# ---------------------------------------------------------------------------


def test_grid_signature_prime_meridian_and_units():
    from groundcontrol.sample import _grid_signature
    sig = _grid_signature
    assert sig(CRS.from_epsg(4326)) != sig(CRS.from_epsg(4807))   # Paris PM
    assert sig(CRS.from_epsg(2230)) != sig(CRS.from_epsg(26946))  # ftUS vs m
    # genuine same-grid datum reinterpretation still matches
    assert sig(CRS.from_epsg(32611)) == sig(CRS.from_epsg(6340))


# ---------------------------------------------------------------------------
# MED — FAA military rows must not assert a definite vertical EPSG code
# ---------------------------------------------------------------------------


def test_faa_pos_class_mil_ownership():
    from groundcontrol.sources.faa import pos_class
    assert pos_class("NGS", "MA") == "mil"   # ownership wins
    assert pos_class("NGS", "PU") == "surveyed"


def test_faa_mil_rows_never_assert_navd88():
    import json
    from pathlib import Path

    from groundcontrol.sources import faa
    with open(Path(__file__).parent / "data" / "faa_apt_sample.txt",
              encoding="latin-1") as f:
        lines = f.readlines()
    out = faa.parse({"cycle": "2026-08-06",
                     "aoi_bounds_4326": (-180.0, -90.0, 180.0, 90.0),
                     "lines": lines})
    mil = np.array([json.loads(r).get("pos_class") == "mil"
                    for r in out["raw"]])
    assert mil.any() and (~mil).any()
    # ambiguous datum: NA code, explicit non-committal height_datum;
    # natives kept (distribution frame) so the EGM96 diagnostic survives
    assert out.loc[mil, "vertical_crs"].isna().all()
    assert (out.loc[mil, "height_datum"].str.contains("EGM96")).all()
    assert out.loc[mil, "native_crs"].eq("EPSG:6349").all()
    assert out.loc[~mil, "vertical_crs"].eq("EPSG:5703").all()


def test_ngl_zero_candidate_exit_before_catalog_pool(monkeypatch):
    # round-2: the 0-station exit sat after the catalog-warming pool, so
    # an empty AOI still downloaded the ~40 MB steps + MIDAS catalogs
    from pathlib import Path

    from groundcontrol.sources import ngl
    idx = ngl.parse_dataholdings(
        (Path(__file__).parent / "data" / "ngl_dataholdings_sample.txt")
        .read_text())
    monkeypatch.setattr(ngl, "_load_index", lambda *a, **k: idx)

    def _no_network(*a, **k):
        raise AssertionError("catalog fetch attempted for empty AOI")

    for name in ("_midas_text", "_steps_text", "_tenv3_text"):
        monkeypatch.setattr(ngl, name, _no_network)
    out = ngl.fetch((10.0, 10.0, 10.1, 10.1))  # no stations here
    assert out["stations"] == []
