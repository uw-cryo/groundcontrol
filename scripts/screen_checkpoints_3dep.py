#!/usr/bin/env python
"""Screen the 3DEP checkpoint heights for unit, decimal and datum errors.

Each checkpoint was surveyed to assess one lidar project (``project_id``).
This samples that project's own source DEM (the OPR DEM USGS stages per work
unit, read through the work unit's VRT) at the checkpoint, and compares the
two heights per project.

Why a ratio: a unit error *multiplies* the height, while terrain change,
vegetation and survey error *shift* it. DEM / checkpoint stays near 1 for a
good point and lands on a fixed value for each kind of error:

    3.2808 (1/0.3048)  metres converted as if they were feet
    0.3048             feet treated as metres
    10, 100            decimal point misplaced (0.1, 0.01 the other way)

A datum error is an offset, not a ratio: a large, consistent DEM - checkpoint
difference (e.g. ellipsoid heights labelled NAVD88 sit one geoid undulation
high). Geoid-model and horizontal-realization differences between the DEM
and the checkpoint are centimetres to a couple of metres, far below what the
screen tests for, so points are only transformed horizontally.

Work units come from WESM (USGS Work Unit Extent Spatial Metadata,
downloaded fresh unless ``--wesm`` is given). The DEM height unit comes from,
in order: the VRT's compound CRS, its band unit, WESM ``vert_crs``. That
label can carry the same error as the checkpoints (then the two agree), so
each work unit's unit is checked against the Copernicus GLO-30 DEM at the
same points (see ``check_units``); GLO-30 decides only feet vs metres.

Usage (from the repo root, groundcontrol installed)::

    python scripts/screen_checkpoints_3dep.py                  # screen USGS heights
    python scripts/screen_checkpoints_3dep.py --height z_meter_qc --check

``--height z_meter_qc --check`` re-screens the repaired heights and exits
non-zero if any project is still flagged. DEM samples are cached in
``--samples`` (and each finished task in ``<samples>_parts/``, so an
interrupted run resumes). A full run reads ~1,800 work-unit VRTs and ~1,100
GLO-30 tiles, roughly 30 minutes with 16 threads.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import rasterio
import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "data" / "checkpoints-3dep-2.0-qc.parquet"
DEFAULT_SAMPLES = ROOT / "data" / "checkpoints_3dep_opr_samples.parquet"
WESM_URL = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/metadata/WESM.csv"
BUCKET = "https://prd-tnm.s3.amazonaws.com"
GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIRANGE="YES",
                CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.vrt", GDAL_HTTP_MAX_RETRY="4",
                GDAL_HTTP_RETRY_DELAY="2", GDAL_MAX_DATASET_POOL_SIZE="450",
                GDAL_HTTP_TIMEOUT="60", GDAL_HTTP_CONNECTTIMEOUT="20")
CHUNK = 300          # points per sampling task (each task opens the VRT itself)

FOOT_US, FOOT_INTL = 1200 / 3937, 0.3048
SIGNATURES = {"m_as_ft": 1 / FOOT_INTL, "ft_as_m": FOOT_INTL, "x10": 10.0, "x100": 100.0,
              "div10": 0.1, "div100": 0.01}
MIN_Z = 5.0          # |height| below this makes the ratio unstable
RATIO_TOL = 0.10     # project median ratio further than this from 1 is flagged
SIG_TOL = 0.03       # a point "matches" a signature within this relative tolerance
SIG_MIN_POINTS = 5   # ... and a project is flagged when this many points match one
OFFSET_M = 5.0       # |median DEM - checkpoint| beyond this flags a datum offset


# ----------------------------------------------------------------- DEM lookup

def s3_keys(prefix: str) -> list[str]:
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    keys, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix, "delimiter": "/"}
        if token:
            params["continuation-token"] = token
        r = requests.get(BUCKET + "/", params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        keys += [c.find("s:Key", ns).text for c in root.findall("s:Contents", ns)]
        if root.find("s:IsTruncated", ns).text != "true":
            return keys
        token = root.find("s:NextContinuationToken", ns).text


def work_unit_vrt(sourcedem_link: str) -> str | None:
    """The work unit's own VRT: ``<prefix>/<workunit>.vrt``, else the single
    ``.vrt`` at the top of the prefix, else None (not screened)."""
    m = re.search(r"prefix=(StagedProducts/Elevation/OPR/Projects/[^&]+)", sourcedem_link)
    if not m:
        return None
    prefix = m.group(1).rstrip("/")
    def quoted(key: str) -> str:   # some project folders contain spaces
        return f"{BUCKET}/{urllib.parse.quote(key)}"
    url = quoted(f"{prefix}/{prefix.split('/')[-1]}.vrt")
    if requests.head(url, timeout=60).status_code == 200:
        return url
    vrts = [k for k in s3_keys(prefix + "/") if k.endswith(".vrt")]
    return quoted(vrts[0]) if len(vrts) == 1 else None


def vertical_unit(src, wesm_vert_crs) -> tuple[float | None, str]:
    """Metres per DEM height unit, and where that came from."""
    crs = pyproj.CRS(src.crs.to_wkt()) if src.crs else None
    if crs is not None and crs.is_compound:
        v = crs.sub_crs_list[1]
        return v.axis_info[0].unit_conversion_factor, f"compound CRS ({v.name})"
    u = (src.units[0] or "").strip().lower()
    if u in ("m", "metre", "meter", "meters", "metres"):
        return 1.0, f"band unit {u!r}"
    if "foot" in u or "feet" in u or u in ("ft", "ftus"):
        return (FOOT_US if ("us" in u or "survey" in u) else FOOT_INTL), f"band unit {u!r}"
    try:
        v = pyproj.CRS.from_epsg(int(wesm_vert_crs))
        return v.axis_info[0].unit_conversion_factor, f"WESM vert_crs EPSG:{int(wesm_vert_crs)}"
    except (TypeError, ValueError, pyproj.exceptions.CRSError):
        pass
    t = str(wesm_vert_crs).lower()   # free text, e.g. "Local Mean Sea Level in Meters"
    if "meter" in t or "metre" in t:
        return 1.0, f"WESM vert_crs text {wesm_vert_crs!r}"
    if "feet" in t or "foot" in t:
        return (FOOT_INTL if "international" in t else FOOT_US), \
            f"WESM vert_crs text {wesm_vert_crs!r}"
    return None, f"unknown (no compound CRS, band unit or WESM vert_crs {wesm_vert_crs!r})"


def _sample(src, xs, ys) -> list[float]:
    """Sample points; when one read fails (a broken tile behind the VRT),
    redo them one at a time so only the unreadable points become NaN."""
    try:
        return [v[0] for v in src.sample(zip(xs, ys), masked=False)]
    except rasterio.errors.RasterioIOError:
        out = []
        for x, y in zip(xs, ys):
            try:
                out.append(next(src.sample([(x, y)], masked=False))[0])
            except rasterio.errors.RasterioIOError:
                out.append(np.nan)
        if all(np.isnan(v) for v in out):
            raise   # nothing readable: let the caller's retry treat it as transient
        return out


def sample_chunk(vrt: str, wu: dict, ids, lon, lat) -> pd.DataFrame:
    with rasterio.Env(**GDAL_ENV), rasterio.open("/vsicurl/" + vrt) as src:
        crs = pyproj.CRS(src.crs.to_wkt())
        h = crs.sub_crs_list[0] if crs.is_compound else crs
        x, y = pyproj.Transformer.from_crs("EPSG:6318", h, always_xy=True).transform(lon, lat)
        left, bottom, right, top = src.bounds
        inb = (x >= left) & (x <= right) & (y >= bottom) & (y <= top)
        vals = np.full(len(ids), np.nan)
        if inb.any():
            vals[inb] = _sample(src, x[inb], y[inb])
        if src.nodata is not None:
            vals[vals == src.nodata] = np.nan
        vals[(vals < -1e4) | (vals > 1e5)] = np.nan   # fill values without a nodata tag
        f, why = vertical_unit(src, wu["vert_crs"])
    return pd.DataFrame({"id": ids, "workunit": wu["workunit"], "dem_raw": vals,
                         "dem_unit_m": f, "dem_unit_from": why, "dem_crs": crs.name})


def sample_chunk_retry(*a, attempts: int = 4) -> pd.DataFrame:
    """Transient network errors (DNS, resets) under many threads: back off, retry."""
    for i in range(attempts):
        try:
            return sample_chunk(*a)
        except rasterio.errors.RasterioIOError:
            if i == attempts - 1:
                raise
            time.sleep(5 * 2 ** i)


def sample_all(cp: pd.DataFrame, wesm: pd.DataFrame, threads: int,
               parts: Path) -> pd.DataFrame:
    """Sample every checkpoint in each of its project's work-unit VRTs.

    Each finished task is written to ``parts/`` (and the VRT lookup to
    ``parts/vrts.csv``), so an interrupted run resumes where it stopped."""
    parts.mkdir(parents=True, exist_ok=True)
    wus = wesm[wesm["project_id"].isin(cp["project_id"]) & wesm["sourcedem_link"].notna()]
    wus = wus.drop_duplicates(["project_id", "sourcedem_link"])
    vrt_cache = parts / "vrts.csv"
    known = (pd.read_csv(vrt_cache).set_index("sourcedem_link")["vrt"].to_dict()
             if vrt_cache.exists() else {})
    todo = [s for s in wus["sourcedem_link"].unique() if s not in known]
    print(f"{len(wus)} work units; finding {len(todo)} VRTs", file=sys.stderr)
    with ThreadPoolExecutor(threads) as pool:
        known.update(zip(todo, pool.map(work_unit_vrt, todo)))
    pd.DataFrame({"sourcedem_link": list(known), "vrt": list(known.values())}).to_csv(
        vrt_cache, index=False)
    wus = wus.assign(vrt=wus["sourcedem_link"].map(known))
    print(f"  {wus['vrt'].notna().sum()} with a VRT", file=sys.stderr)

    tasks = []
    by_project = {k: g for k, g in cp.groupby("project_id")}
    for wu in wus[wus["vrt"].notna()].to_dict("records"):
        pts = by_project[wu["project_id"]]
        for i in range(0, len(pts), CHUNK):
            part = parts / f"{wu['project_id']}_{re.sub(r'[^A-Za-z0-9_.-]', '_', wu['workunit'])}_{i}.parquet"
            c = pts.iloc[i:i + CHUNK]
            tasks.append((part, (wu["vrt"], wu, c["id"].to_numpy(), c["lon"].to_numpy(),
                                 c["lat"].to_numpy())))
    todo = [t for t in tasks if not t[0].exists()]
    print(f"sampling: {len(todo)} of {len(tasks)} tasks left ({threads} threads)",
          file=sys.stderr)
    failed, t0 = [], time.perf_counter()

    def run(task):
        part, a = task
        df = sample_chunk_retry(*a)
        df.to_parquet(part.with_suffix(".tmp"), index=False)
        part.with_suffix(".tmp").replace(part)

    with ThreadPoolExecutor(threads) as pool:
        futs = {pool.submit(run, t): t for t in todo}
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                fut.result()
            except Exception as e:  # one bad VRT must not sink the run
                failed.append((futs[fut][1][1]["workunit"], repr(e)[:200]))
            if k % 200 == 0:
                print(f"  {k}/{len(todo)} ({time.perf_counter() - t0:.0f} s)", file=sys.stderr)
    for wu, err in sorted(set(failed)):
        print(f"  failed: {wu}: {err}", file=sys.stderr)
    if failed:
        sys.exit(f"{len(failed)} tasks failed; rerun to retry just those")
    s = pd.concat([pd.read_parquet(t[0]) for t in tasks], ignore_index=True)
    # one row per checkpoint: the first work unit with a valid value
    s = s.sort_values(["id", "workunit"])
    valid = s[s["dem_raw"].notna()].drop_duplicates("id")
    return pd.concat([valid, s[~s["id"].isin(valid["id"])].drop_duplicates("id")],
                     ignore_index=True)


# ------------------------------------------------- DEM unit check (GLO-30)
# A DEM's unit label can carry the same error as its checkpoints (feet
# labelled metres in both), and then the two agree. So each work unit's DEM
# unit is checked against an independent reference, the Copernicus GLO-30 DEM
# (TanDEM-X, metres, EGM2008), at the same checkpoints: median(GLO-30 / DEM
# value) is ~1 for metres and ~0.3048 for feet. GLO-30 is a 30 m surface
# model, so it decides only the unit, by a median over many points; it never
# judges a checkpoint.

GLO = "https://copernicus-dem-30m.s3.amazonaws.com/{n}/{n}.tif"
UNIT_CHECK_MIN_RAW = 100.0  # DEM value (its own unit) below which a point does not vote:
                            # a wrong vote needs GLO-30 to overshoot by 0.7x the value
                            # (canopy, buildings), plausible at 20, not at 100+
UNIT_CHECK_MEDIAN_N = 10    # with no strong voter: median GLO-30 / DEM over >= this many
                            # points, decided when 2x nearer (log scale) one unit
UNIT_CHECK_AGREE = 0.8   # share of a work unit's point votes that must agree
UNIT_CHECK_TOL = 0.15     # relative tolerance around 1 and 0.3048


def glo_tile(lon: float, lat: float) -> str:
    la, lo = math.floor(lat), math.floor(lon)
    return (f"Copernicus_DSM_COG_10_{'N' if la >= 0 else 'S'}{abs(la):02d}_00_"
            f"{'E' if lo >= 0 else 'W'}{abs(lo):03d}_00_DEM")


def sample_glo_tile(name: str, ids, lon, lat) -> pd.DataFrame:
    try:
        with rasterio.Env(**GDAL_ENV), rasterio.open("/vsicurl/" + GLO.format(n=name)) as src:
            v = np.array([x[0] for x in src.sample(zip(lon, lat))], "float64")
            if src.nodata is not None:
                v[v == src.nodata] = np.nan
    except rasterio.errors.RasterioIOError:   # no tile (open water)
        v = np.full(len(ids), np.nan)
    return pd.DataFrame({"id": ids, "glo_h": v})


def add_glo(samples: pd.DataFrame, cp: pd.DataFrame, threads: int) -> pd.DataFrame:
    pts = cp[cp["id"].isin(samples.loc[samples["dem_raw"].notna(), "id"])]
    tiles = [glo_tile(x, y) for x, y in zip(pts["lon"], pts["lat"])]
    groups = [(t, g["id"].to_numpy(), g["lon"].to_numpy(), g["lat"].to_numpy())
              for t, g in pts.groupby(tiles)]
    print(f"GLO-30 unit check: {len(pts)} points in {len(groups)} tiles", file=sys.stderr)
    with ThreadPoolExecutor(threads) as pool:
        glo = pd.concat(pool.map(lambda a: sample_glo_tile(*a), groups), ignore_index=True)
    return samples.drop(columns="glo_h", errors="ignore").merge(glo, on="id", how="left")


def _vote(r: np.ndarray) -> np.ndarray:
    """Per point: metres per DEM unit implied by GLO-30 / DEM value (1.0 or
    the foot), or NaN when the ratio is near neither."""
    v = np.full(r.shape, np.nan)
    v[np.abs(r - 1) <= UNIT_CHECK_TOL] = 1.0
    v[np.abs(r / FOOT_INTL - 1) <= UNIT_CHECK_TOL] = FOOT_US
    return v


def check_units(samples: pd.DataFrame, project_of: pd.Series
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Settle each work unit's DEM unit against GLO-30; set ``dem_h`` (m).

    Every checkpoint whose DEM value is at least UNIT_CHECK_MIN_RAW votes
    metres or feet (at 250 m the two differ by 570 m, far beyond GLO-30
    error, so one point can decide); a work unit is decided when >= UNIT_CHECK_AGREE of its votes
    agree. A work unit with no such point (low terrain) is decided by the
    median GLO-30 / DEM over at least UNIT_CHECK_MEDIAN_N points, when it lies
    at least twice as near (log scale) to one unit as to the other. A work unit with no usable vote (low terrain) takes the unit of
    its project's work units with the same unit metadata only if at least two
    of them are decided and all agree; otherwise it stays unverified and keeps
    its metadata unit. ``project_of`` maps checkpoint id -> project_id.
    Returns the samples and a per-work-unit table.
    """
    s = samples.copy()
    s["project_id"] = s["id"].map(project_of)
    s["meta_key"] = s["dem_unit_m"].round(6).astype(str) + "|" + s["dem_unit_from"].astype(str)
    have = s["dem_raw"].notna() & s["glo_h"].notna()
    s["glo_ratio"] = np.where(have, s["glo_h"] / s["dem_raw"], np.nan)
    s["vote"] = _vote(s["glo_ratio"].to_numpy())
    strong = have & (s["dem_raw"] >= UNIT_CHECK_MIN_RAW)
    pos = have & (s["glo_ratio"] > 0)
    rows = []
    for wu, g in s[s["workunit"].notna()].groupby("workunit"):
        decided, how = np.nan, ""
        if strong[g.index].any():
            votes = g.loc[strong[g.index], "vote"].dropna()
            if len(votes):
                top = votes.value_counts()
                if top.iloc[0] / len(votes) >= UNIT_CHECK_AGREE:
                    decided, how = float(top.index[0]), "votes"
        elif pos[g.index].sum() >= UNIT_CHECK_MEDIAN_N:
            # low terrain: GLO-30 (a surface model) is noisy point by point, but
            # its median ratio still sits far nearer one unit than the other
            lr = np.log(np.median(g.loc[pos[g.index], "glo_ratio"]))
            d_m, d_ft = abs(lr), abs(lr - np.log(FOOT_INTL))
            if d_m * 2 <= d_ft:
                decided, how = 1.0, "median"
            elif d_ft * 2 <= d_m:
                decided, how = FOOT_US, "median"
        votes = g.loc[strong[g.index], "vote"].dropna()
        rows.append(dict(workunit=wu, project_id=g["project_id"].iloc[0],
                         meta_key=g["meta_key"].iloc[0], n_votes=len(votes),
                         glo_over_dem=float(g["glo_ratio"].median()),
                         metadata_unit_m=g["dem_unit_m"].iloc[0],
                         metadata_from=g["dem_unit_from"].iloc[0], checked=decided,
                         decided_by=how, basis=""))
    units = pd.DataFrame(rows)
    for _, g in units.groupby(["project_id", "meta_key"]):
        dec = g["checked"].dropna()
        if len(dec) >= 2 and dec.nunique() == 1:
            fill = g.index[g["checked"].isna()]
            units.loc[fill, "checked"] = dec.iloc[0]
            units.loc[fill, "basis"] = " (project pool)"
            units.loc[fill, "decided_by"] = "pool"
    verdicts, used = [], []
    for r in units.itertuples():
        meta = r.metadata_unit_m
        meta_is_ft = np.isfinite(meta) and abs(meta - FOOT_INTL) < 1e-3
        meta_is_m = np.isfinite(meta) and abs(meta - 1) < 1e-9
        if not np.isfinite(r.checked):
            unit, verdict = meta, "unverified"
        elif not (meta_is_ft or meta_is_m):
            unit, verdict = r.checked, "from GLO-30 (no metadata unit)" + r.basis
        elif (r.checked == 1.0) == meta_is_m:
            unit, verdict = meta, "agrees" + r.basis
        else:
            unit, verdict = r.checked, "metadata unit WRONG" + r.basis
        verdicts.append(verdict)
        used.append(unit)
    units = units.assign(unit_m=used, verdict=verdicts).drop(
        columns=["meta_key", "basis", "checked"])
    s = s.drop(columns=["dem_unit_check", "dem_h"], errors="ignore").merge(
        units[["workunit", "unit_m", "verdict"]].rename(
            columns={"unit_m": "dem_unit_used", "verdict": "dem_unit_check"}),
        on="workunit", how="left")
    s["dem_h"] = s["dem_raw"] * s["dem_unit_used"]
    s = s.drop(columns=["dem_unit_used", "meta_key", "project_id", "vote", "glo_ratio"])
    return s, units


# ---------------------------------------------------------------------- screen

def nearest_signature(r: np.ndarray) -> np.ndarray:
    names = np.full(r.shape, "", dtype=object)
    for name, v in SIGNATURES.items():
        names[np.abs(r / v - 1) <= SIG_TOL] = name
    return names


def screen(cp: pd.DataFrame, samples: pd.DataFrame, height: str) -> pd.DataFrame:
    d = cp[["id", "project_id", "point_type", height]].merge(
        samples[["id", "dem_h", "dem_unit_from"]], on="id", how="left")
    d = d.rename(columns={height: "z"})
    d = d[d["z"].notna() & d["dem_h"].notna()]
    d["dz"] = d["dem_h"] - d["z"]
    big = d[d["z"].abs() > MIN_Z].copy()
    big["ratio"] = big["dem_h"] / big["z"]
    big["sig"] = nearest_signature(big["ratio"].to_numpy())
    rows = []
    for pid, g in d.groupby("project_id"):
        b = big[big["project_id"] == pid]
        med_r = float(b["ratio"].median()) if len(b) else np.nan
        sig_counts = b.loc[b["sig"] != "", "sig"].value_counts()
        top_sig = sig_counts.index[0] if len(sig_counts) else ""
        n_sig = int(sig_counts.iloc[0]) if len(sig_counts) else 0
        med_dz = float(g["dz"].median())
        nmad = float(1.4826 * (g["dz"] - med_dz).abs().median())
        reasons = []
        if len(b) >= SIG_MIN_POINTS and abs(med_r - 1) > RATIO_TOL:
            reasons.append("median_ratio")
        if n_sig >= SIG_MIN_POINTS:
            reasons.append(f"{n_sig}_pts_{top_sig}")
        if len(g) >= SIG_MIN_POINTS and abs(med_dz) > OFFSET_M and nmad < abs(med_dz) / 3:
            reasons.append("offset")
        rows.append(dict(project_id=pid, n=len(g), n_ratio=len(b), median_ratio=med_r,
                         top_signature=top_sig, n_signature=n_sig, median_dz=med_dz,
                         nmad_dz=nmad, types=",".join(sorted(g["point_type"].unique())),
                         dem_unit_from=";".join(sorted(g["dem_unit_from"].dropna().unique())),
                         flagged=";".join(reasons)))
    return pd.DataFrame(rows)


def breakdown(cp: pd.DataFrame, samples: pd.DataFrame, height: str, pids) -> pd.DataFrame:
    """Flagged projects split by point_type and id prefix (the id minus its
    trailing number, which tracks the source file): where a rule's edges are."""
    d = cp[cp["project_id"].isin(pids)][["id", "project_id", "point_type", height]].merge(
        samples[["id", "dem_h"]], on="id")
    d = d[d["dem_h"].notna() & (d[height].abs() > 1)]
    d["prefix"] = d["id"].str.replace(r"_?\d+$", "", regex=True)
    d["ratio"] = d["dem_h"] / d[height]
    d["dz"] = d["dem_h"] - d[height]
    return (d.groupby(["project_id", "point_type", "prefix"])
             .agg(n=("id", "size"), median_ratio=("ratio", "median"),
                  median_dz=("dz", "median"), z_median=(height, "median"))
             .reset_index())


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("parquet", nargs="?", type=Path, default=DEFAULT_IN)
    p.add_argument("--height", default="z_meter_vdatum_update",
                   help="height column to screen (z_meter_qc to check the repairs)")
    p.add_argument("--wesm", default=WESM_URL, help="WESM.csv path or URL")
    p.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES,
                   help="DEM sample cache (read if present, written if not)")
    p.add_argument("--resample", action="store_true", help="ignore the sample cache")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--report", type=Path, help="write the per-project table as CSV")
    p.add_argument("--units-report", type=Path,
                   help="write the per-work-unit DEM unit check as CSV")
    p.add_argument("--check", action="store_true", help="exit 1 if any project is flagged")
    p.add_argument("--projects", help="comma-separated project_ids to screen (default: all)")
    args = p.parse_args(argv)

    import geopandas as gpd
    gdf = gpd.read_parquet(args.parquet)
    cp = pd.DataFrame(gdf.drop(columns=["geometry", "bbox"], errors="ignore"))
    cp["lon"], cp["lat"] = gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()
    if args.projects:
        cp = cp[cp["project_id"].isin([int(v) for v in args.projects.split(",")])]

    if args.samples.exists() and not args.resample:
        samples = pd.read_parquet(args.samples)
    else:
        wesm = pd.read_csv(args.wesm, usecols=["workunit", "project_id", "vert_crs",
                                               "sourcedem_link"])
        samples = sample_all(cp, wesm, args.threads,
                             args.samples.with_name(args.samples.stem + "_parts"))
        args.samples.parent.mkdir(parents=True, exist_ok=True)
        samples.to_parquet(args.samples, index=False)
    if "glo_h" not in samples.columns:
        samples = add_glo(samples, cp, args.threads)
        samples.to_parquet(args.samples, index=False)
    samples, units = check_units(samples, cp.set_index("id")["project_id"])
    if args.units_report:
        units.to_csv(args.units_report, index=False)
    n_dem = int(samples["dem_h"].notna().sum())
    print(f"{n_dem} of {len(cp)} checkpoints have a height from their own project's DEM",
          file=sys.stderr)
    with pd.option_context("display.width", 250, "display.max_rows", 500,
                           "display.max_colwidth", 60, "display.float_format", "{:.4f}".format):
        print("DEM unit check against GLO-30, work units by verdict:")
        print(units["verdict"].value_counts().to_string())
        bad = units[units["verdict"] != "agrees"].drop(columns="unit_m")
        bad = bad[bad["verdict"] != "unverified"]
        if len(bad):
            print(bad.to_string(index=False))
        print()

    rep = screen(cp, samples, args.height)
    if args.report:
        rep.to_csv(args.report, index=False)
    flagged = rep[rep["flagged"] != ""].sort_values("median_ratio")
    with pd.option_context("display.width", 250, "display.max_rows", 500,
                           "display.max_colwidth", 60, "display.float_format", "{:.3f}".format):
        print(f"screening {args.height}: {len(flagged)} of {len(rep)} projects flagged")
        if len(flagged):
            print(flagged.to_string(index=False))
            print("\nflagged projects by point_type and id prefix (|z| > 1 m):")
            print(breakdown(cp, samples, args.height, flagged["project_id"]).to_string(index=False))
    if args.check and len(flagged):
        sys.exit(1)


if __name__ == "__main__":
    main()
