"""assess.py pipeline tests (Increment 2): transform -> sample -> stats + CLI.

Offline; synthetic plane rasters as in test_sample.py. The pure-frame
transform test promotes EPSG:6341 (NAD83(2011)/UTM 12N) to 3D so no geoid
grid is needed; the compound-source (GEOID18) path is exercised only when the
local PROJ can resolve it (skipped otherwise, never ballpark).
"""

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import pytest
import rasterio
from rasterio.transform import Affine

from groundcontrol.assess import (SEGMENTS, assess_products, sample_products,
                                  summarize_dz, transform_control)
from groundcontrol.crs import NoTransformPathError, get_transformer

UTM12_3D = pyproj.CRS("EPSG:6341").to_3d()  # NAD83(2011) / UTM 12N, ellipsoidal h
AOI_AZ = (-112.0, 32.4, -111.3, 33.3)


def _control_6319(n=4, h=400.0):
    """Schema-ish control in NAD83(2011) 3D geographic (ellipsoidal heights)."""
    lon = np.linspace(-111.9, -111.6, n)
    lat = np.linspace(32.6, 32.9, n)
    return gpd.GeoDataFrame(
        {"source": ["3dep", "3dep", "opus", "ngs"][:n],
         "point_type": ["NVA", "VVA", "gnss", "monument"][:n],
         "id": [f"P{i}" for i in range(n)],
         "height": np.full(n, h)},
        geometry=gpd.points_from_xy(lon, lat), crs="EPSG:6319")


def test_transform_control_pure_frame_heights_pass_through():
    ctl = _control_6319()
    out, info = transform_control(ctl, UTM12_3D, source_crs="EPSG:6319",
                                  aoi_bounds_4326=AOI_AZ)
    # ellipsoidal in, ellipsoidal out: heights unchanged, geometry reprojected
    np.testing.assert_allclose(out["h_ell"], ctl["height"], atol=1e-9)
    assert pyproj.CRS(out.crs).equals(UTM12_3D)
    t = get_transformer("EPSG:6319", UTM12_3D, aoi_bounds_4326=AOI_AZ)
    E, N = t.transform(ctl.geometry.x.to_numpy(), ctl.geometry.y.to_numpy(),
                       ctl["height"].to_numpy(), np.full(len(ctl), 2010.0))[:2]
    np.testing.assert_allclose(out.geometry.x, E)
    np.testing.assert_allclose(out.geometry.y, N)
    # original frame untouched (copy semantics), info block populated
    assert pyproj.CRS(ctl.crs).to_epsg() == 6319
    assert info["n_points"] == len(ctl) and info["pipeline"].startswith("proj=")
    assert info["dh_stats"]["median"] == pytest.approx(0.0, abs=1e-9)


def test_transform_control_navd88_chain_if_available():
    ctl = _control_6319().set_crs("EPSG:6318", allow_override=True)
    try:
        out, info = transform_control(ctl, UTM12_3D, aoi_bounds_4326=AOI_AZ)
    except NoTransformPathError:
        pytest.skip("GEOID18 grid unavailable in this PROJ install")
    # Sonoran Desert geoid undulation: h_ell = H + N with N ~ -30 m
    assert -35 < info["dh_stats"]["median"] < -25
    assert "vgridshift" in info["pipeline"]


# ---------------------------------------------------------------------------
# sampling + stats on a known plane (z = 2x + 3y), CRS-consistent throughout
# ---------------------------------------------------------------------------

CRS = "EPSG:32611"
NX, NY, X0, Y0 = 12, 8, 50.0, 40.0


def _plane_tif(tmp_path, name="plane.tif"):
    j, i = np.meshgrid(np.arange(NX), np.arange(NY))
    arr = (2.0 * (X0 + j + 0.5) + 3.0 * (Y0 + NY - i - 0.5)).astype("float64")
    t = Affine(1.0, 0.0, X0, 0.0, -1.0, Y0 + NY)
    path = tmp_path / name
    with rasterio.open(path, "w", driver="GTiff", height=NY, width=NX, count=1,
                       dtype="float64", crs=CRS, transform=t) as dst:
        dst.write(arr, 1)
    return str(path)


def _landed(offsets, outside=0):
    """Points on the plane with h_ell = plane - offset (so dh_before = offset)."""
    n = len(offsets)
    xs = np.linspace(X0 + 2.5, X0 + 8.5, n)
    ys = np.linspace(Y0 + 2.5, Y0 + 5.5, n)
    if outside:
        xs = np.append(xs, X0 - 100.0)   # off-raster -> NaN sample
        ys = np.append(ys, Y0 - 100.0)
    m = len(xs)
    plane = 2.0 * xs + 3.0 * ys
    return gpd.GeoDataFrame(
        {"source": (["3dep", "3dep", "opus", "ngs"] * m)[:m],
         "point_type": (["NVA", "VVA", "gnss", "monument"] * m)[:m],
         "h_ell": plane - np.append(np.asarray(offsets, dtype="float64"),
                                    np.zeros(outside))},
        geometry=gpd.points_from_xy(xs, ys), crs=CRS)


def test_sample_products_standard_columns_and_values(tmp_path):
    dsm = _plane_tif(tmp_path, "a-DSM_mos.tif")
    dtm = _plane_tif(tmp_path, "b-DTM_no_fill_mos.tif")
    pts = _landed([0.10, -0.20, 0.30, 0.40])
    out = sample_products(pts, {"DSM": dsm, "DTM": dtm})
    for prod in ("DSM", "DTM"):
        assert {f"h_{prod}", f"dh_{prod}_before"} <= set(out.columns)
        np.testing.assert_allclose(out[f"dh_{prod}_before"],
                                   [0.10, -0.20, 0.30, 0.40], atol=1e-9)
    assert not any(" minus " in c for c in out.columns)
    assert "h_ell" in out.columns  # input columns ride along


def test_sample_products_radius_columns(tmp_path):
    dsm = _plane_tif(tmp_path)
    pts = _landed([0.0, 0.0, 0.0, 0.0])
    out = sample_products(pts, {"DSM": dsm}, radius=1.5)
    assert {"h_DSM", "h_DSM_nmad", "h_DSM_n", "dh_DSM_before"} <= set(out.columns)
    assert (out["h_DSM_n"] > 0).all()


def test_summarize_dz_segments_nodata_and_applies(tmp_path):
    dsm = _plane_tif(tmp_path, "a-DSM_mos.tif")
    dtm = _plane_tif(tmp_path, "b-DTM_no_fill_mos.tif")
    pts = _landed([0.10, -0.20, 0.30, 0.40], outside=1)  # 5th point off-raster
    out = sample_products(pts, {"DSM": dsm, "DTM": dtm})
    stats = summarize_dz(out)
    assert set(stats["product"]) == {"DSM", "DTM"}
    alls = stats[(stats["product"] == "DSM") & (stats.segment == "ALL")].iloc[0]
    assert alls["n"] == 5 and alls["n_valid"] == 4  # gap point reported, not dropped
    vva_dsm = stats[(stats["product"] == "DSM") & (stats.segment == "3DEP VVA")].iloc[0]
    vva_dtm = stats[(stats["product"] == "DTM") & (stats.segment == "3DEP VVA")].iloc[0]
    assert not vva_dsm["applies"] and vva_dtm["applies"]
    nva = stats[(stats["product"] == "DSM") & (stats.segment == "3DEP NVA")].iloc[0]
    assert nva["median_m"] == pytest.approx(0.10, abs=1e-9)
    assert set(SEGMENTS) == {"3DEP NVA", "3DEP VVA", "GNSS/OPUS", "NGS monument"}


def test_assess_products_end_to_end_writes_artifacts(tmp_path):
    """Whole orchestration in one frame (no vertical leg): CRS = raster CRS."""
    dsm = _plane_tif(tmp_path, "a-DSM_mos.tif")
    pts = _landed([0.10, -0.20, 0.30, 0.40]).rename(columns={"h_ell": "height"})
    # height column is ellipsoidal already; source == target -> pure identity
    sampled, stats, artifacts = assess_products(
        pts, {"DSM": dsm}, CRS, source_crs=CRS,
        outdir=tmp_path / "out", site_name="synthsite", figures=False)
    assert (tmp_path / "out" / "synthsite_assessed.parquet").exists()
    assert (tmp_path / "out" / "synthsite_dz_stats.csv").exists()
    # pure-frame op: PROJ reports 0/None ("exact"), a grid chain reports >0 —
    # either way the info block must carry the key and never a negative
    acc = artifacts["transform"]["accuracy_m"]
    assert acc is None or acc >= 0
    rt = gpd.read_parquet(tmp_path / "out" / "synthsite_assessed.parquet")
    np.testing.assert_allclose(rt["dh_DSM_before"], [0.10, -0.20, 0.30, 0.40],
                               atol=1e-9)


def test_cli_assess_smoke(tmp_path):
    """CLI wiring: cached control -> stats + parquet, no figures, no network."""
    from groundcontrol.cli import assess_dem_main
    dsm = _plane_tif(tmp_path, "a-DSM_mos.tif")
    pts = _landed([0.05, 0.05, 0.05, 0.05]).rename(columns={"h_ell": "height"})
    cache = tmp_path / "ctl.parquet"
    pts.to_parquet(cache)
    aoi = tmp_path / "aoi.geojson"
    pts.to_crs("EPSG:4326")[["geometry"]].to_file(aoi, driver="GeoJSON")
    rc = assess_dem_main([
        "--aoi", str(aoi), "--product", f"DSM={dsm}", "--target-crs", CRS,
        "--source-crs", CRS, "--control", str(cache),
        "--outdir", str(tmp_path / "out"),
        "--site-name", "clisite", "--no-figures"])
    assert rc == 0
    assert (tmp_path / "out" / "clisite_assessed.parquet").exists()
    assert (tmp_path / "out" / "clisite_dz_stats.csv").exists()


# expand_attributes lives in sources.ngs but is exercised here with assess-side
# usage (systematic monument isolation by stamping).
def test_expand_attributes_lifts_raw_fields():
    from groundcontrol.sources.ngs import expand_attributes
    raw = [json.dumps({"name": "AG 45", "stamping": "AG 45 1967 ARMY MAP SERVICE",
                       "vertSource": " VERTCON3 "}),
           json.dumps({"name": "LARK"}),
           "not json"]
    g = gpd.GeoDataFrame({"raw": raw},
                         geometry=gpd.points_from_xy([0, 1, 2], [0, 0, 0]),
                         crs="EPSG:6318")
    out = expand_attributes(g, fields=("name", "stamping", "vertSource"))
    assert out["ngs_name"].tolist()[:2] == ["AG 45", "LARK"]
    assert out["ngs_vertSource"][0] == "VERTCON3"          # stripped
    assert pd.isna(out["ngs_stamping"][1])                 # absent -> NA
    assert pd.isna(out["ngs_name"][2])                     # bad JSON -> NA
    assert g.columns.tolist() == ["raw", "geometry"]       # input not mutated
    sel = out["ngs_stamping"].str.contains("ARMY MAP", na=False)
    assert sel.tolist() == [True, False, False]


def test_family_dz_figures_smoke(tmp_path):
    from groundcontrol.figures import default_ngs_best, family_dz_figures
    n = 12
    src = (["3dep"] * 4 + ["opus"] * 2 + ["ngs"] * 6)
    ptype = (["NVA", "NVA", "VVA", "VVA"] + ["gnss"] * 2 + ["monument"] * 6)
    raw = [None] * 6 + [json.dumps({"posSource": "ADJUSTED", "vertSource": "GPS OBS"})] * 3 \
        + [json.dumps({"posSource": "SCALED", "vertSource": "VERTCON3"})] * 3
    g = gpd.GeoDataFrame(
        {"source": src, "point_type": ptype, "raw": raw,
         "ref_frame": ["NAD83(2011)"] * 6 + ["NAD 83(2011)"] * 3 + ["NAD 83(1986)"] * 3,
         "dh_DSM_before": np.linspace(-0.1, 0.1, n),
         "dh_DTM_before": np.append(np.linspace(-0.1, 0.1, n - 1), np.nan)},
        geometry=gpd.points_from_xy(np.linspace(0, 100, n), np.linspace(0, 80, n)),
        crs="EPSG:32611")
    best = default_ngs_best(g)
    assert best.sum() == 3          # ADJUSTED + GPS OBS rows only
    out = family_dz_figures(g, None, tmp_path, "syn")
    names = sorted(p.name for p in out)
    assert names == sorted(f"syn_dz_{fam}_{prod}.png"
                           for fam in ("3dep", "gnss", "ngs_best")
                           for prod in ("DSM", "DTM"))
    assert all(p.exists() for p in out)


def test_transform_control_xform_acc_column():
    ctl = _control_6319()
    out, info = transform_control(ctl, UTM12_3D, source_crs="EPSG:6319",
                                  aoi_bounds_4326=AOI_AZ)
    assert "xform_acc_m" in out.columns
    a = out["xform_acc_m"].iloc[0]
    # pure-frame promotion: PROJ reports 0 (exact) -> NaN (unknown/exact,
    # never a fake positive); grid-based chains yield real positive values
    assert np.isnan(a) or a > 0


# ---------------------------------------------------------------------------
# adversarial-audit fixes (2026-07-16)
# ---------------------------------------------------------------------------

def test_transform_control_rejects_mismatched_declared_crs():
    """A 3D-tagged (ellipsoidal-height) frame under the default NAVD88 landing
    would get the geoid undulation applied to already-ellipsoidal heights —
    and the dh_stats tripwire would read like a plausible geoid signal."""
    ctl = _control_6319()  # declares EPSG:6319
    with pytest.raises(ValueError, match="refusing to reinterpret"):
        transform_control(ctl, UTM12_3D, aoi_bounds_4326=AOI_AZ)


def test_sample_products_rejects_already_sampled(tmp_path):
    dsm = _plane_tif(tmp_path, "a-DSM_mos.tif")
    pts = _landed([0.10, -0.20, 0.30, 0.40])
    once = sample_products(pts, {"DSM": dsm})
    with pytest.raises(ValueError, match="already present"):
        sample_products(once, {"DSM": dsm})  # duplicate labels -> mixed stats


def test_summarize_dz_tolerates_na_point_type():
    df = gpd.GeoDataFrame(
        {"source": pd.array(["3dep", "3dep", None], dtype="string"),
         "point_type": pd.array(["NVA", None, "VVA"], dtype="string"),
         "dh_DSM_before": [0.1, 0.2, 0.3]},
        geometry=gpd.points_from_xy([0, 1, 2], [0, 0, 0]), crs=CRS)
    stats = summarize_dz(df)  # NA rows are excluded, never a bool-cast crash
    nva = stats[(stats["product"] == "DSM") & (stats.segment == "3DEP NVA")].iloc[0]
    assert nva["n"] == 1


def test_error_report_zero_nmad_skips_gate():
    """Quantized residuals (>=50% identical) collapse NMAD to 0; the gate must
    then keep everything, never report fake-perfect stats."""
    from groundcontrol.accuracy import error_report
    r = error_report([0.0] * 10 + [0.01] * 5)
    assert r["n_outliers"] == 0 and r["n_used"] == 15
    assert r["rmse"] > 0
    r2 = error_report([0.05] * 8 + [0.06] * 4 + [0.30])
    assert r2["n_outliers"] == 0 and r2["mean"] > 0.05


def test_datum_tag_compound_names_vertical_datum():
    from groundcontrol.figures import _datum_tag
    assert _datum_tag("EPSG:6318+5703") == "NAVD88 height"
    assert "ellipsoid" in _datum_tag("EPSG:6319")


def test_parse_kv_duplicate_name_raises():
    from groundcontrol.cli import _parse_kv
    with pytest.raises(SystemExit, match="twice"):
        _parse_kv(["DSM=a.tif", "DSM=b.tif"], "--product")


def test_expand_attributes_numeric_format_stable():
    """'vertOrder: 2' must expand to "2" regardless of whether OTHER rows are
    missing the field (apply's inference floated int columns with a None)."""
    from groundcontrol.sources.ngs import expand_attributes
    for raws in ([json.dumps({"vertOrder": 2}), json.dumps({})],
                 [json.dumps({"vertOrder": 2}), json.dumps({"vertOrder": 1})]):
        g = gpd.GeoDataFrame({"raw": raws},
                             geometry=gpd.points_from_xy([0, 1], [0, 0]),
                             crs="EPSG:6318")
        out = expand_attributes(g, fields=("vertOrder",))
        assert out["ngs_vertOrder"][0] == "2"


def test_family_dz_ngs_best_na_mask(tmp_path):
    """default_ngs_best yields Kleene-NA for an ADJUSTED mark with missing
    ref_frame and non-GPS vertical — exclude the row, don't crash the figure."""
    from groundcontrol.figures import default_ngs_best, family_dz_figures
    g = gpd.GeoDataFrame(
        {"source": ["ngs", "ngs"], "point_type": ["monument"] * 2,
         "raw": [json.dumps({"posSource": "ADJUSTED", "vertSource": "RESET"}),
                 json.dumps({"posSource": "ADJUSTED", "vertSource": "GPS OBS"})],
         "ref_frame": pd.array([None, "NAD 83(2011)"], dtype="string"),
         "dh_DSM_before": [0.05, -0.02]},
        geometry=gpd.points_from_xy([0, 50], [0, 40]), crs=CRS)
    assert default_ngs_best(g).isna().any()  # the trap this test pins
    out = family_dz_figures(g, None, tmp_path, "na",
                            products=("DSM",), families=("ngs_best",))
    assert len(out) == 1 and out[0].exists()
