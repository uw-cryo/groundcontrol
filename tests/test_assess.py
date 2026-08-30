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
         "point_type": ["NVA", "VVA", "gnss_campaign", "monument"][:n],
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
         "point_type": (["NVA", "VVA", "gnss_campaign", "monument"] * m)[:m],
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
    assert set(SEGMENTS) == {
        "3DEP NVA", "3DEP VVA", "GNSS continuous", "GNSS semi-continuous",
        "GNSS campaign (OPUS)", "GNSS campaign (NGL)",
        "GNSS campaign (other)", "GNSS (pre-split)", "NGS monument",
        "FAA runway surveyed", "FAA other",
        "OTHER (unsegmented)"}


def test_gnss_taxonomy_exhaustive_and_styled():
    # every GNSS-class row from ANY source lands in exactly one GNSS
    # segment (incl. the "GNSS campaign (other)" backstop for future
    # sources — audit finding: third-source campaign rows silently fell
    # out), and every class is styled + visible in the gnss DZ family
    # (audit finding: the family was the one consumer that dropped the
    # legacy pre-split label)
    import pandas as pd
    from groundcontrol.figures import DZ_FAMILIES, POINT_STYLE
    classes = ["gnss_cont", "gnss_semicont", "gnss_campaign", "gnss"]
    rows = [(pt, src) for pt in classes for src in ("ngl", "opus", "newsrc")]
    g = gpd.GeoDataFrame(
        {"source": pd.Series([s for _, s in rows], dtype="string"),
         "point_type": pd.Series([p for p, _ in rows], dtype="string"),
         "dh_DSM_before": [0.0] * len(rows)},
        geometry=gpd.points_from_xy(range(len(rows)), [0.0] * len(rows)),
        crs="EPSG:32611")
    gnss_segs = [fn for lbl, (fn, _, _) in SEGMENTS.items()
                 if lbl.startswith("GNSS")]
    counts = sum(pd.Series(fn(g)).fillna(False).to_numpy(dtype=bool).astype(int)
                 for fn in gnss_segs)
    assert (counts == 1).all()
    fam_masks = [sub[1] for sub in DZ_FAMILIES["gnss"][1]]
    for pt in classes:
        assert pt in POINT_STYLE
        sel = (g["point_type"] == pt).fillna(False)
        assert any(pd.Series(m(g)).fillna(False)[sel].any()
                   for m in fam_masks), f"{pt} invisible in gnss DZ family"


def test_summarize_dz_gnss_routes_by_point_type():
    # per-row taxonomy (2026-08-22): routing is by each row's OWN class
    # (ngl.occupation_class), not its source. The legacy pre-split "gnss"
    # label gets its own segment so old parquets stay visible in the stats
    # table instead of silently dropping out; campaign keeps the NGL/OPUS
    # split (ARP vs ground-mark heights are not comparable).
    g = gpd.GeoDataFrame(
        {"source": ["ngl", "ngl", "ngl", "opus", "ngl", "opus"],
         "point_type": ["gnss_cont", "gnss_semicont", "gnss_campaign",
                        "gnss_campaign", "gnss", "gnss"],
         "dh_DSM_before": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]},
        geometry=gpd.points_from_xy([0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                                    [0.0] * 6),
        crs="EPSG:32611")
    stats = summarize_dz(g)
    seg_n = dict(zip(stats.segment, stats.n))
    assert seg_n["GNSS continuous"] == 1
    assert seg_n["GNSS semi-continuous"] == 1
    assert seg_n["GNSS campaign (NGL)"] == 1
    # legacy pre-split OPUS rows fold into the validating campaign segment
    # (they are the same ground marks — audit round 3: routing them to the
    # context row flipped applies on every existing parquet)
    assert seg_n["GNSS campaign (OPUS)"] == 2
    assert seg_n["GNSS (pre-split)"] == 1  # the non-OPUS legacy row only
    assert seg_n["OTHER (unsegmented)"] == 0
    # only OPUS campaign validates (ground-mark heights); NGL-fed classes
    # are context-only until an ant_m correction lands (audit round 2:
    # ARP heights must not claim applies=True)
    assert stats[stats.segment == "GNSS campaign (OPUS)"].iloc[0]["applies"]
    for seg in ("GNSS continuous", "GNSS semi-continuous",
                "GNSS campaign (NGL)", "GNSS (pre-split)"):
        assert not stats[stats.segment == seg].iloc[0]["applies"]


def test_summarize_dz_unsegmented_row_surfaces():
    # audit round 4: an NA point_type is schema-legal, and such a row must
    # surface in the residual segment, never silently vanish from the table
    import pandas as pd
    g = gpd.GeoDataFrame(
        {"source": pd.Series(["opus", "opus"], dtype="string"),
         "point_type": pd.Series([pd.NA, "gnss_campaign"], dtype="string"),
         "dh_DSM_before": [0.1, 0.2]},
        geometry=gpd.points_from_xy([0.0, 1.0], [0.0, 0.0]), crs="EPSG:32611")
    stats = summarize_dz(g)
    seg_n = dict(zip(stats.segment, stats.n))
    assert seg_n["OTHER (unsegmented)"] == 1
    assert seg_n["GNSS campaign (OPUS)"] == 1
    total_named = sum(v for k, v in seg_n.items() if k != "ALL")
    assert total_named == seg_n["ALL"] == 2  # nothing lost, nothing doubled


def test_seg_style_covers_segments():
    # the figure style map must stay in lockstep with the taxonomy — a
    # label add/rename fails HERE, not mid-run at figure time (audit rd 4)
    from groundcontrol.figures import _SEG_STYLE
    assert set(_SEG_STYLE) == set(SEGMENTS)


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
    # info carries PROJ's RAW accuracy (None/0/-1 sentinels allowed); the
    # honest per-point mapping (sentinels -> NaN) is what xform_acc_m asserts
    acc = artifacts["transform"]["accuracy_m"]
    assert acc is None or isinstance(acc, float)
    xa = sampled["xform_acc_m"].iloc[0]
    assert np.isnan(xa) or xa > 0
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
    ptype = (["NVA", "NVA", "VVA", "VVA"] + ["gnss_campaign"] * 2
             + ["monument"] * 6)
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


def test_validation_dz_figures_accepts_path_aoi(tmp_path):
    # validation_dz_figures must accept a path aoi in any CRS (read + reproject),
    # like its sibling figure functions — not only a pre-reprojected GeoDataFrame
    from shapely.geometry import box

    from groundcontrol.figures import validation_dz_figures
    n = 10
    g = gpd.GeoDataFrame(
        {"source": ["3dep"] * 4 + ["opus"] * 2 + ["ngs"] * 4,
         "point_type": ["NVA", "NVA", "VVA", "VVA"] + ["gnss_campaign"] * 2
                       + ["monument"] * 4,
         "dh_DSM_before": np.linspace(-0.1, 0.1, n),
         "dh_DTM_before": np.linspace(-0.1, 0.1, n)},
        geometry=gpd.points_from_xy(np.linspace(0, 100, n),
                                    np.linspace(0, 80, n)),
        crs="EPSG:32611")
    aoi = tmp_path / "aoi.geojson"  # supplied as a path in a DIFFERENT CRS
    gpd.GeoDataFrame(geometry=[box(*g.total_bounds)], crs=g.crs) \
        .to_crs("EPSG:4326").to_file(aoi, driver="GeoJSON")
    out = validation_dz_figures(g, aoi, tmp_path, "syn",
                                products=("DSM", "DTM"))
    assert sorted(p.name for p in out) == ["syn_validation_dz_DSM.png",
                                           "syn_validation_dz_DTM.png"]
    assert all(p.exists() for p in out)


def test_aspect_panel_w_degenerate_aoi():
    # zero-height (or zero-width) AOI bounds must fall back to a square
    # panel, not divide by zero (Python float -> ZeroDivisionError)
    from groundcontrol.figures import _aspect_panel_w
    flat = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([0.0, 100.0], [50.0, 50.0]),
        crs="EPSG:32611")  # dy == 0
    assert _aspect_panel_w(flat, 5.7) == 5.7
    tall = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([50.0, 50.0], [0.0, 100.0]),
        crs="EPSG:32611")  # dx == 0
    assert _aspect_panel_w(tall, 5.7) == 5.7
    assert _aspect_panel_w(None, 5.7) == 5.7


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


def test_transform_control_2d_tag_under_3d_source_passes():
    """The universal 2D EPSG:6318 cache tag must not be rejected under an
    explicit 3D geographic source (round-2 audit: to_2d demotion)."""
    ctl = _control_6319().set_crs("EPSG:6318", allow_override=True)
    out, info = transform_control(ctl, UTM12_3D, source_crs="EPSG:6319",
                                  aoi_bounds_4326=AOI_AZ)
    np.testing.assert_allclose(out["h_ell"], ctl["height"], atol=1e-9)


def test_transform_control_accepts_esri_wkt_tag():
    """An ESRI-WKT1 (.prj-derived) NAD83(2011) tag is the same CRS —
    matched via resolved EPSG code, not string equality."""
    esri = pyproj.CRS("EPSG:6318").to_wkt(version="WKT1_ESRI")
    ctl = _control_6319().set_crs(esri, allow_override=True)
    out, _ = transform_control(ctl, UTM12_3D, source_crs="EPSG:6319",
                               aoi_bounds_4326=AOI_AZ)
    assert len(out) == len(ctl)


def test_sample_products_radius_resample_also_guarded(tmp_path):
    """Dropping only h_/dh_ then re-sampling in radius mode must still raise
    (h_*_nmad/h_*_n would otherwise duplicate silently)."""
    dsm = _plane_tif(tmp_path, "a-DSM_mos.tif")
    pts = _landed([0.10, -0.20, 0.30, 0.40])
    once = sample_products(pts, {"DSM": dsm}, radius=1.5)
    stripped = once.drop(columns=["h_DSM", "dh_DSM_before"])
    with pytest.raises(ValueError, match="already present"):
        sample_products(stripped, {"DSM": dsm}, radius=1.5)


def test_family_dz_misaligned_mask_series_raises(tmp_path):
    """A positional mask built on a RangeIndex against a .loc-filtered frame
    would label-align to all-False — raise instead of an empty figure."""
    from groundcontrol.figures import family_dz_figures
    g = gpd.GeoDataFrame(
        {"source": ["ngs"] * 4, "point_type": ["monument"] * 4,
         "raw": [None] * 4, "ref_frame": ["NAD 83(2011)"] * 4,
         "dh_DSM_before": [0.1, 0.2, 0.3, 0.4]},
        geometry=gpd.points_from_xy(range(4), range(4)), crs=CRS,
        index=[2, 5, 7, 9])
    bad = pd.Series([True, True, False, False])  # RangeIndex
    with pytest.raises(ValueError, match="mask index"):
        family_dz_figures(g, None, tmp_path, "mis", products=("DSM",),
                          families=("ngs_best",), ngs_best=bad)
    ok = family_dz_figures(g, None, tmp_path, "ok", products=("DSM",),
                           families=("ngs_best",),
                           ngs_best=bad.to_numpy())  # positional array: fine
    assert len(ok) == 1


def test_expand_attributes_json_nan_becomes_na():
    """json.loads accepts bare NaN — it must expand to NA, never "nan"."""
    from groundcontrol.sources.ngs import expand_attributes
    g = gpd.GeoDataFrame({"raw": ['{"name": NaN}', '{"name": "AG 45"}']},
                         geometry=gpd.points_from_xy([0, 1], [0, 0]),
                         crs="EPSG:6318")
    out = expand_attributes(g, fields=("name",))
    assert pd.isna(out["ngs_name"][0]) and out["ngs_name"][1] == "AG 45"


def test_figures_ngs_gate_skipped_when_nmad_zero():
    """The NGS histogram gate mirrors error_report: quantized residuals
    (NMAD=0) keep everything — including the genuine outlier (round-3 pin)."""
    from groundcontrol.figures import _ngs_gate
    v = np.array([0.30] * 8 + [45.0])
    out = _ngs_gate(v, 3.0)
    assert len(out) == 9 and 45.0 in out          # no gate at NMAD=0
    v2 = np.array([0.0, 0.01, -0.01, 0.02, -0.02, 45.0])
    assert 45.0 not in _ngs_gate(v2, 3.0)          # normal spread still gates


def test_transform_control_2d_target_raises_issue22():
    """#22: a 2D non-compound target must fail loud, not skip the vertical."""
    ctl = _control_6319()
    with pytest.raises(ValueError, match="2D"):
        transform_control(ctl, "EPSG:32612", source_crs="EPSG:6319",
                          aoi_bounds_4326=AOI_AZ)


def test_transform_control_2d_identity_still_allowed():
    """source == target 2D identity is height-inert and stays legal
    (the end-to-end orchestration tests rely on it)."""
    n = 3
    g = gpd.GeoDataFrame(
        {"height": np.full(n, 100.0)},
        geometry=gpd.points_from_xy(np.linspace(400000.0, 401000.0, n),
                                    np.linspace(3600000.0, 3601000.0, n)),
        crs="EPSG:32611")
    out, _ = transform_control(g, "EPSG:32611", source_crs="EPSG:32611",
                               aoi_bounds_4326=(-120.0, 32.0, -119.0, 33.0))
    np.testing.assert_allclose(out["h_ell"], g["height"])


def test_faa_segments_route_by_pos_class(tmp_path):
    """Owner figure review 2026-08-30: FAA rows previously fell to OTHER.
    Surveyed validates both product classes; estimated is context-only."""
    import json
    dsm = _plane_tif(tmp_path, "b-DSM_mos.tif")
    pts = _landed([0.05, 0.05, 0.05, 0.05]).rename(columns={"h_ell": "height"})
    pts["source"] = "faa"
    pts["point_type"] = ["runway_end", "runway_end", "helipad", "displaced_threshold"]
    # the helipad is SURVEYED-class on paper — it must still route to
    # context (the CG +0.33 m military-helipad finding)
    pts["raw"] = [json.dumps({"pos_class": c})
                  for c in ("surveyed", "surveyed", "surveyed", "surveyed")]
    pts["h_ell"] = pts["height"]
    sampled = sample_products(pts, {"DSM": dsm})
    stats = summarize_dz(sampled, products=["DSM"]).set_index("segment")
    # survey-grade = PAINTED runway features only: the surveyed HELIPAD is
    # context (CG 2026-08-30: 8 MILITARY-source helipads measured +0.33 m —
    # a different accuracy class, some hand-held GNSS per the owner)
    assert stats.loc["FAA runway surveyed", "n"] == 3   # 2 ends + 1 displaced
    assert bool(stats.loc["FAA runway surveyed", "applies"])
    assert stats.loc["FAA other", "n"] == 1             # the surveyed helipad
    assert not bool(stats.loc["FAA other", "applies"])
    assert stats.loc["OTHER (unsegmented)", "n"] == 0


def test_assess_bundle_includes_labeled_control_map(tmp_path):
    """The standard figure bundle carries the all-sources labeled control
    map (owner 2026-08-30: prototyped in July, never formally included)."""
    dsm = _plane_tif(tmp_path, "a-DSM_mos.tif")
    pts = _landed([0.1, -0.1, 0.2, 0.0]).rename(columns={"h_ell": "height"})
    _, _, art = assess_products(pts, {"DSM": dsm}, CRS, source_crs=CRS,
                                outdir=tmp_path / "out", site_name="cm",
                                basemap=None, midas_velocities=False)
    names = [p.name for p in art["control_figures"]]
    assert "cm_control_map.png" in names
    assert (tmp_path / "out" / "cm_control_map.png").exists()


def test_transform_control_masks_mismatched_vertical_rows():
    """Mixed-vertical cache (ngl in the defaults, 2026-08-30): rows whose
    vertical_crs disagrees with the declared source vertical get h_ell=NaN
    (positions keep the horizontal leg) — never a silently mis-applied
    geoid. Compatible rows are byte-identical to a compatible-only run."""
    import warnings as _w
    pts = _control_6319(4)  # helper frame; retag as a NAVD88-landed cache
    pts = pts.rename(columns={"h_ell": "height"}) if "h_ell" in pts.columns else pts
    pts = pts.set_crs("EPSG:6318", allow_override=True)
    pts["vertical_crs"] = ["EPSG:5703", "EPSG:5703", "EPSG:7912", "EPSG:5703"]
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        out, info = transform_control(pts, "EPSG:6341+5703",
                                      source_crs="EPSG:6318+5703")
        ref, _ = transform_control(pts.drop(columns=["vertical_crs"]),
                                   "EPSG:6341+5703", source_crs="EPSG:6318+5703")
    assert info["n_vertical_excluded"] == 1
    assert "EPSG:7912" in info["vertical_note"]
    assert np.isnan(out["h_ell"].iloc[2])
    ok = [0, 1, 3]
    np.testing.assert_allclose(out["h_ell"].iloc[ok], ref["h_ell"].iloc[ok])
    # the horizontal leg still lands the excluded row (maps/sheets valid)
    assert out.geometry.iloc[2].x == ref.geometry.iloc[2].x


def test_transform_control_native_retarget_for_mismatched_vertical():
    """Owner 2026-08-30 ("I need to see the dz values"): a vertically-
    mismatched row WITH native 3D coordinates is re-targeted through its
    NATIVE frame's chain (per-row coord_epoch tt for a dynamic frame);
    rows without natives stay masked. Routing proven via a recording
    transformer fake."""
    import warnings as _w

    import groundcontrol.assess as A
    calls = []

    class _T:
        def __init__(self, src):
            self.src = src
            self.accuracy = 0.02
            self.description = f"fake {src}"
            self.definition = "fake"

        def transform(self, x, y, z, t, errcheck=True):
            calls.append((self.src, np.asarray(t).copy()))
            return (np.asarray(x) + 1.0, np.asarray(y) + 1.0,
                    np.asarray(z) + 100.0, np.asarray(t))

    real = A.get_transformer
    A.get_transformer = lambda src, tgt, aoi_bounds_4326=None: _T(str(src))
    try:
        pts = _control_6319(4).rename(columns={"h_ell": "height"},
                                      errors="ignore")
        pts = pts.set_crs("EPSG:6318", allow_override=True)
        pts["vertical_crs"] = ["EPSG:5703", "EPSG:7912", "EPSG:7912", "EPSG:5703"]
        pts["native_x"] = pts.geometry.x
        pts["native_y"] = pts.geometry.y
        pts["native_h"] = [np.nan, 500.0, 510.0, np.nan]
        pts["native_crs"] = [None, "EPSG:7912", "EPSG:7912", None]
        pts["coord_epoch"] = [np.nan, 2022.3, np.nan, np.nan]  # row 2: no epoch
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            out, info = A.transform_control(pts, "EPSG:6341+5703",
                                            source_crs="EPSG:6318+5703")
    finally:
        A.get_transformer = real
    # row 1: dynamic native chain, tt = its coord_epoch, h from native+100
    assert info["n_vertical_native"] == 1
    assert info["n_vertical_excluded"] == 1        # row 2: no coord_epoch
    assert out["h_ell"].iloc[1] == 600.0
    assert np.isnan(out["h_ell"].iloc[2])
    assert out["xform_acc_m"].iloc[1] == 0.02
    srcs = [c[0] for c in calls]
    assert "EPSG:7912" in srcs                     # the native chain ran
    tt_native = calls[[i for i, s in enumerate(srcs)
                       if s == "EPSG:7912"][0]][1]
    assert tt_native.tolist() == [2022.3]          # per-row epoch, not 2010
