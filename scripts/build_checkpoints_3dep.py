#!/usr/bin/env python
"""Build ``data/checkpoints-3dep-2.0-qc.parquet`` from the USGS GeoPackage.

Source: "Consolidated Standardized Survey-Grade Checkpoints 3DEP 2004 to 2025"
(ver. 2.0, December 2025), https://doi.org/10.5066/P13NOW9E, ScienceBase item
67075e6bd34e969edc59c3e7. See ``docs/checkpoints_3dep.md`` for what this changes and why.

Usage (from the repo root, in an environment with groundcontrol installed)::

    python scripts/build_checkpoints_3dep.py                 # download, build
    python scripts/build_checkpoints_3dep.py --gpkg Checkpoints_3DEP_2004_2025.gpkg

What it does, in order:

1. Downloads the release zip and checks its MD5 against ScienceBase's
   published checksum *and* against the one this script was written for, so
   a new USGS release stops the build instead of passing silently.
2. Keeps every USGS attribute value unchanged. It renames
   ``unique_identifier`` -> ``id`` and ``collection_date`` -> ``datetime``
   (the STAC-style names stac-map and groundcontrol read), and drops the
   GeoPackage row counter ``OBJECTID``.
3. Writes 2D points in EPSG:6318 (the horizontal part of the GeoPackage's
   compound EPSG:6349). The height lives in columns, not in geometry Z.
4. Adds ``z_meter_qc`` and ``qc_height_flag``: known errors in the published heights
   (see ``QC_RULES``), corrected where the fix is certain, set to NaN where
   the height is unusable, and flagged but left unchanged where the published
   height disagrees with its own source fields for an unknown reason.
   ``z_meter_vdatum_update`` remains the USGS value. The VDatum step is
   re-run with PROJ and the NGS geoid grids (fetched from the PROJ CDN) to
   find the latter.
5. Adds ``qc_horizontal_flag`` (see ``apply_horizontal_qc``): rows whose
   source coordinates do not fit their recorded coordinate system, and rows
   whose source datum is not NAD83(2011), so the published lon,lat position
   carries a realization label that cannot be checked. Geometry and source
   fields stay the USGS values.
6. Sorts rows along a Hilbert curve and writes ~10k-row row groups with a
   ``bbox`` covering column (GeoParquet 1.1), so bbox reads skip row groups.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

ITEM_ID = "67075e6bd34e969edc59c3e7"
ITEM_URL = f"https://www.sciencebase.gov/catalog/item/{ITEM_ID}"
DOI = "https://doi.org/10.5066/P13NOW9E"
ZIP_NAME = "Checkpoints_3DEP_2004_2025.zip"
GPKG_NAME = "Checkpoints_3DEP_2004_2025.gpkg"
LAYER = "Consolidated_Standardized_Checkpoints_3DEP_Version2_20250930"
# ver. 2.0 (uploaded 2025-12-17). A different checksum means a new release:
# rerun scripts/screen_checkpoints_3dep.py (docs/checkpoints_3dep.md) and update QC_RULES before bumping this.
EXPECTED_MD5 = "5bd22479b1429087227d8bb4af0ebce0"
EXPECTED_ROWS = 145_299

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "checkpoints-3dep-2.0-qc.parquet"
ROW_GROUP_SIZE = 10_000
#: published as s3://tacolab/groundcontrol/<DATA_VERSION>.parquet on Kopah;
#: matches PARQUET_NAME in the source
DATA_VERSION = "checkpoints-3dep-2.0-qc"

US_FT = 1200 / 3937  # U.S. survey foot, m
INTL_FT = 0.3048     # international foot, m
UNIT_TO_M = {"meter": 1.0, "U.S. Survey Feet": US_FT, "International Feet": INTL_FT}

# Re-running the VDatum step (source geoid -> GEOID18, CONUS) with the NGS geoid
# grids in PROJ: H18 = H_src + N_src - N_GEOID18. GEOID12A has no grid on the
# PROJ CDN; GEOID12B, which replaced it, stands in. Reproduces VDatum to ~1 mm.
GEOID_GRID = {"GEOID03": "us_noaa_geoid03_conus.tif", "GEOID09": "us_noaa_geoid09_conus.tif",
              "GEOID12A": "us_noaa_g2012bu0.tif", "GEOID12B": "us_noaa_g2012bu0.tif",
              "GEOID18": "us_noaa_g2018u0.tif"}
CONUS = (-125.0, 24.0, -66.0, 50.0)
VDATUM_TOL_M = 0.02   # published vs re-run beyond this: flagged

# ME_SouthernArea_2012 (project 1067): every source_elevation lies in [1, 10),
# as if the exponent of a scientific-notation value was dropped. The true
# factor (10 or 100) per point comes from the project's own source DEM
# (scripts/screen_checkpoints_3dep.py): DEM / checkpoint = 9.88-10.0 or
# ~100, residual after the fix within +/-0.16 m. Key: id suffix.
_ME_1067_FACTOR = {
    1: 100, 2: 10, 3: 100, 4: 10, 5: 10, 6: 10, 7: 10, 8: 10, 9: 100, 10: 100,
    11: 100, 12: 10, 13: 100, 14: 10, 15: 10, 16: 10, 17: 10, 18: 10, 19: 10,
    20: 100, 21: 100, 22: 10, 23: 100, 24: 100, 25: 100, 26: 10, 27: 100, 28: 10,
    29: 10, 30: 10, 31: 10, 32: 100, 33: 100, 34: 100,
}

# Each rule selects rows and sets ``z_meter_qc`` one of four ways: a unit repair
# (``true_src_to_m``: the true metres per source unit, scalar or per-row
# Series), NaN (``true_src_to_m=None``), the re-run VDatum height
# (``z_qc="rerun"``), or the published height unchanged (``z_qc="keep"``).
# Selectors see the re-run height as ``_z_rerun``. ``n`` is the expected match
# count: a mismatch stops the build. Evidence: docs/checkpoints_quality_assessment.ipynb.
QC_RULES = [
    dict(
        flag="m_labelled_usft",
        note="CA_Sonoma_2013 (project 12384): source heights are metres labelled U.S. survey feet",
        select=lambda g: g["project_id"] == 12384,
        true_src_to_m=1.0,
        n=411,
    ),
    dict(
        flag="m_labelled_usft",
        note="CA_Sacramento_2017 subset of project 58946: metres labelled U.S. survey feet",
        select=lambda g: g["id"].str.contains("_58946_Sacramento", regex=False),
        true_src_to_m=1.0,
        n=143,
    ),
    dict(
        flag="m_labelled_ft",
        note="OR_HarneySilver_2020_A20 (project 191399) VVA points: metres labelled international feet",
        select=lambda g: (g["project_id"] == 191399) & (g["point_type"] == "VVA"),
        true_src_to_m=1.0,
        n=57,
    ),
    dict(
        flag="ft_labelled_m",
        note="Fork_Salt_River_NRCS (project 11903): U.S. survey feet labelled metres "
             "(source_geoid=UNK: still not VDatum-harmonized)",
        select=lambda g: g["project_id"] == 11903,
        true_src_to_m=US_FT,
        n=39,
    ),
    dict(
        flag="decimal_shift",
        note="ME_SouthernArea_2012 (project 1067): decimal point misplaced by 10x or 100x",
        select=lambda g: g["project_id"] == 1067,
        true_src_to_m=lambda g: g["id"].str.extract(r"_(\d+)$", expand=False)
                                 .astype(int).map(_ME_1067_FACTOR).astype(float),
        n=34,
    ),
    dict(
        flag="ellipsoid_height_labelled_navd88",
        note="HI_Oahu_2012 (project 4259): heights sit a GEOID12B undulation (~15.4 m) above the "
             "DEM, i.e. ellipsoid heights; Hawaii has no NAVD88",
        select=lambda g: g["project_id"] == 4259,
        true_src_to_m=None,
        n=100,
    ),
    dict(
        flag="vdatum_scaled_not_rerun",
        note="NRCS_Lower_and_MiddleGreenPondRough (project 11309): heights corrected by multiplying "
             "by 0.3048 instead of re-running VDatum (version history item 8), which also scaled "
             "the geoid shift; that rule reproduces every published height. z_meter_qc = re-run",
        select=lambda g: g["project_id"] == 11309,
        z_qc="rerun",
        n=1130,
    ),
    dict(
        flag="vdatum_mismatch",
        note="published height differs by more than 2 cm from re-running VDatum on its own source "
             "fields (one-row offsets and unexplained); cannot tell which is right. z_meter_qc unchanged",
        select=lambda g: (g["project_id"] != 11309)
        & ((g["_z_rerun"] - g["z_meter_vdatum_update"]).abs() > VDATUM_TOL_M),
        z_qc="keep",
        n=88,
    ),
]


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_gpkg(workdir: Path, allow_new_release: bool) -> tuple[Path, dict]:
    item = requests.get(ITEM_URL, params={"format": "json"}, timeout=60).json()
    entry = next(f for f in item["files"] if f["name"] == ZIP_NAME)
    published_md5 = entry["checksum"]["value"]
    print(f"{item['title']}\n  {ZIP_NAME}: uploaded {entry['dateUploaded']}, md5 {published_md5}")
    if published_md5 != EXPECTED_MD5 and not allow_new_release:
        sys.exit(f"ScienceBase now publishes md5 {published_md5}, not {EXPECTED_MD5}: a new "
                 "release. Rerun scripts/screen_checkpoints_3dep.py, update QC_RULES and "
                 "EXPECTED_MD5, or pass --allow-new-release to build anyway.")
    zpath = workdir / ZIP_NAME
    with requests.get(entry["url"], stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(zpath, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    got = md5sum(zpath)
    if got != published_md5:
        sys.exit(f"download md5 {got} != published {published_md5}")
    with zipfile.ZipFile(zpath) as z:
        z.extract(GPKG_NAME, workdir)
    src = dict(item=ITEM_URL, doi=DOI, title=item["title"], zip=ZIP_NAME,
               zip_md5=got, uploaded=entry["dateUploaded"])
    return workdir / GPKG_NAME, src


def rerun_vdatum(gdf: gpd.GeoDataFrame) -> pd.Series:
    """Re-run the published VDatum step from the source fields (CONUS rows with a
    known source geoid; NaN elsewhere)."""
    import pyproj
    pyproj.network.set_network_enabled(True)

    def undulation(grid, lat, lon):
        t = pyproj.Transformer.from_pipeline(
            "+proj=pipeline +step +proj=axisswap +order=2,1 +step +proj=unitconvert +xy_in=deg "
            f"+xy_out=rad +step +proj=vgridshift +grids={grid} +multiplier=1 +step "
            "+proj=unitconvert +xy_in=rad +xy_out=deg +step +proj=axisswap +order=2,1")
        return t.transform(lat, lon, np.zeros(len(lat)))[2]

    x, y = gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()
    w, sth, e, n = CONUS
    use = (x > w) & (x < e) & (y > sth) & (y < n) & gdf["source_geoid"].isin(GEOID_GRID).to_numpy()
    z = pd.Series(np.nan, index=gdf.index)
    n18 = undulation(GEOID_GRID["GEOID18"], y[use], x[use])
    sub = gdf[use]
    src_m = (sub["source_elevation"] * sub["source_vertical_unit"].map(UNIT_TO_M)).to_numpy()
    out = np.full(len(sub), np.nan)
    for geoid, g in sub.groupby("source_geoid"):
        i = sub.index.get_indexer(g.index)
        out[i] = src_m[i] + undulation(GEOID_GRID[geoid], y[use][i], x[use][i]) - n18[i]
    z[use] = out
    return z


def apply_qc(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add ``z_meter_qc`` / ``qc_height_flag``; USGS columns stay untouched.

    The published height is ``z = src * labelled_unit + geoid_shift``. When
    only the unit (or decimal point) of ``src`` was wrong, the VDatum shift
    to GEOID18 is still right, so the fix adds back the unit error alone:
    ``z_qc = z + (src * true_unit - src * labelled_unit)``.
    """
    labelled = gdf["source_vertical_unit"].map(UNIT_TO_M)
    if labelled.isna().any():
        bad = sorted(gdf.loc[labelled.isna(), "source_vertical_unit"].unique())
        raise ValueError(f"unknown source_vertical_unit values: {bad}")
    src_m = gdf["source_elevation"] * labelled

    z_qc = gdf["z_meter_vdatum_update"].astype("float64").copy()
    flag = pd.Series(pd.NA, index=gdf.index, dtype="string")
    gdf["_z_rerun"] = rerun_vdatum(gdf)
    for rule in QC_RULES:
        m = rule["select"](gdf).fillna(False).to_numpy(bool)
        if m.sum() != rule["n"]:
            raise AssertionError(f"QC rule {rule['flag']!r} ({rule['note']}) matched "
                                 f"{m.sum()} rows, expected {rule['n']}")
        if flag[m].notna().any():
            raise AssertionError(f"QC rule {rule['flag']!r} overlaps an earlier rule")
        flag[m] = rule["flag"]
        if rule.get("z_qc") == "keep":
            continue
        if rule.get("z_qc") == "rerun":
            if gdf.loc[m, "_z_rerun"].isna().any():
                raise AssertionError(f"QC rule {rule['flag']!r}: rows without a re-run height")
            z_qc[m] = gdf.loc[m, "_z_rerun"]
            continue
        true = rule["true_src_to_m"]
        if true is None:
            z_qc[m] = np.nan
            continue
        true_m = true(gdf[m]) if callable(true) else true
        if np.any(pd.isna(true_m)):
            raise AssertionError(f"QC rule {rule['flag']!r}: missing correction factor")
        z_qc[m] = gdf.loc[m, "z_meter_vdatum_update"] + (
            gdf.loc[m, "source_elevation"] * true_m - src_m[m])
    gdf["z_meter_qc"] = z_qc
    gdf["qc_height_flag"] = flag
    return gdf.drop(columns="_z_rerun")


# Horizontal QC (docs/checkpoints_quality_assessment.ipynb, Parts 3 and 4).
# Source coordinates are unprojected in their recorded CRS's own datum and
# compared with the published lon,lat position. NAD83 realizations differ by
# metres at most, so more than 1 km apart means the recorded code is wrong.
CRS_MISMATCH_M = 1000.0
H_RULES = [
    dict(flag="source_crs_mismatch",
         note="source_easting/northing land more than 1 km from the published lon,lat position "
              "when unprojected in source_horizontal_epsg: the recorded code is wrong for these "
              "values (wrong zone, state plane recorded as UTM, foot code for metres, axes "
              "swapped). The published lon,lat position agrees with the project DEMs",
         n=4_748),
    dict(flag="source_datum_not_nad83_2011",
         note="source datum is not NAD83(2011) (NAD83(1986), NSRS2007, HARN, CORS96, WGS 84): "
              "the published lon,lat position is labelled NAD83(2011) but was mostly re-labelled, "
              "not transformed, and the source realization label itself cannot be verified; "
              "expect 0.2-1.2 m horizontal uncertainty",
         n=29_591),
]


def recorded_crs(code):
    """source_horizontal_epsg as a CRS: an EPSG code, or an Esri code recorded as one."""
    import pyproj
    for auth in ("EPSG", "ESRI"):
        try:
            return pyproj.CRS.from_authority(auth, str(int(code)))
        except pyproj.exceptions.CRSError:
            pass
    raise ValueError(f"source_horizontal_epsg {code} is neither an EPSG nor an Esri code")


def apply_horizontal_qc(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add ``qc_horizontal_flag``; geometry and source fields stay untouched."""
    import pyproj
    geod = pyproj.Geod(ellps="GRS80")
    dist = pd.Series(np.nan, index=gdf.index)
    datum = pd.Series("", index=gdf.index, dtype=object)
    for code, g in gdf.groupby("source_horizontal_epsg"):
        crs = recorded_crs(code)
        lon, lat = pyproj.Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True).transform(
            g["source_easting"].to_numpy(), g["source_northing"].to_numpy())
        dist[g.index] = geod.inv(lon, lat, g.geometry.x.to_numpy(), g.geometry.y.to_numpy())[2]
        datum[g.index] = crs.geodetic_crs.name
    if dist.isna().any():
        raise AssertionError(f"{dist.isna().sum()} rows without a source-coordinate distance")
    masks = {"source_crs_mismatch": dist >= CRS_MISMATCH_M,
             "source_datum_not_nad83_2011": (dist < CRS_MISMATCH_M) & (datum != "NAD83(2011)")}
    flag = pd.Series(pd.NA, index=gdf.index, dtype="string")
    for rule in H_RULES:
        m = masks[rule["flag"]].to_numpy()
        if m.sum() != rule["n"]:
            raise AssertionError(f"horizontal rule {rule['flag']!r} matched {m.sum()} rows, "
                                 f"expected {rule['n']}")
        flag[m] = rule["flag"]
    gdf["qc_horizontal_flag"] = flag
    return gdf


def build(gpkg: Path, out: Path, source: dict) -> None:
    gdf = gpd.read_file(gpkg, layer=LAYER, engine="pyogrio", use_arrow=True)
    if len(gdf) != EXPECTED_ROWS:
        raise AssertionError(f"{len(gdf)} rows, expected {EXPECTED_ROWS}")
    if gdf.crs is None or gdf.crs.to_epsg() != 6349:
        raise AssertionError(f"expected EPSG:6349, GeoPackage says {gdf.crs}")
    if not gdf["unique_identifier"].is_unique:
        raise AssertionError("unique_identifier is not unique")

    gdf = gdf.drop(columns="OBJECTID").rename(
        columns={"unique_identifier": "id", "collection_date": "datetime"})
    # 2D geometry + the matching 2D CRS. Geometry Z duplicated
    # z_meter_vdatum_update (to 1 mm), and the one compound CRS is not true for
    # every row (Puerto Rico heights are PRVD02; Hawaii has no NAVD88).
    gdf = gdf.set_geometry(gdf.geometry.force_2d()).set_crs("EPSG:6318", allow_override=True)
    gdf = apply_qc(gdf)
    gdf = apply_horizontal_qc(gdf)

    # Hilbert order => compact row groups => bbox reads skip most of the file
    gdf = gdf.iloc[np.argsort(gdf.hilbert_distance().to_numpy(), kind="stable")]
    gdf = gdf.reset_index(drop=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.parquet")
    gdf.to_parquet(tmp, index=False, schema_version="1.1.0", geometry_encoding="WKB",
                   write_covering_bbox=True)
    # rewrite once with provenance metadata, zstd and ~10k-row row groups
    table = pq.read_table(tmp)
    tmp.unlink()
    provenance = dict(
        source=source,
        # no build timestamp: the same inputs and library versions give the
        # same bytes, so anyone can rebuild and check PARQUET_SHA256
        release=DATA_VERSION,
        libraries={m.__name__: m.__version__ for m in (gpd, pd, np, pa)},
        script="scripts/build_checkpoints_3dep.py (github.com/uw-cryo/groundcontrol)",
        changes=["renamed unique_identifier->id, collection_date->datetime",
                 "dropped OBJECTID", "geometry forced 2D, CRS EPSG:6349 -> EPSG:6318",
                 "added z_meter_qc and qc_height_flag (see qc_rules)",
                 "added qc_horizontal_flag (see horizontal_qc_rules)", "Hilbert-sorted rows"],
        vdatum_rerun=dict(method="H18 = H_src + N_src - N_GEOID18 (PROJ vgridshift)",
                          grids=GEOID_GRID, region="CONUS", tolerance_m=VDATUM_TOL_M),
        qc_rules=[dict(flag=r["flag"], note=r["note"], n=r["n"]) for r in QC_RULES],
        horizontal_qc_rules=[dict(r, threshold_m=CRS_MISMATCH_M) if i == 0 else r
                             for i, r in enumerate(H_RULES)],
    )
    md = dict(table.schema.metadata or {})
    md.pop(b"pandas", None)
    md[b"groundcontrol"] = json.dumps(provenance).encode()
    table = table.replace_schema_metadata(md)
    pq.write_table(table, out, row_group_size=ROW_GROUP_SIZE, compression="zstd",
                   compression_level=9)

    # read back and check
    chk = gpd.read_parquet(out)
    assert len(chk) == EXPECTED_ROWS and chk.crs.to_epsg() == 6318
    assert not chk.geometry.has_z.any()
    nrg = pq.ParquetFile(out).metadata.num_row_groups
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(chk)} rows, {nrg} row groups)")
    print(chk["qc_height_flag"].value_counts().to_string())
    print(chk["qc_horizontal_flag"].value_counts().to_string())


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--gpkg", type=Path,
                   help="use this local GeoPackage instead of downloading the release zip")
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT,
                   help="output parquet (default: data/checkpoints-3dep-2.0-qc.parquet)")
    p.add_argument("--allow-new-release", action="store_true",
                   help="build even if ScienceBase publishes a different zip checksum")
    args = p.parse_args(argv)
    if args.gpkg:
        build(args.gpkg, args.output,
              dict(item=ITEM_URL, doi=DOI, gpkg=args.gpkg.name, gpkg_md5=md5sum(args.gpkg)))
        return
    with tempfile.TemporaryDirectory() as d:
        gpkg, source = download_gpkg(Path(d), args.allow_new_release)
        build(gpkg, args.output, source)


if __name__ == "__main__":
    main()
