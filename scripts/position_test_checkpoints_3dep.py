#!/usr/bin/env python
"""Which NAD83 realization are the 3DEP checkpoints' source coordinates really in?

Exploratory (see docs/checkpoints_position_investigation.ipynb). For checkpoints
whose source coordinates and their own project's OPR DEM carry *different*
realization labels (e.g. NAD83(HARN) checkpoints over a DEM labelled NAD83), the
two labels make different predictions for where the checkpoint sits on the DEM:

- **same frame**: the labels are two names for one frame, so the source
  easting/northing go onto the DEM grid by a projection change alone;
- **labels true**: PROJ's realization transformation applies (NADCON5 and
  friends: typically 0.05-1 m).

On sloping ground the right position gives tighter DEM - checkpoint residuals.
For each work unit this reads a small DEM patch around every checkpoint once,
then scores both positions and a grid of offsets around them by the NMAD of
DEM - checkpoint (a constant vertical datum offset drops out). When both labels
are the same, the two positions coincide and the DEM cannot tell them apart;
those checkpoints are not tested here.

Inputs: the parquet built on this branch (``source_horizontal_crs_qc``), WESM,
and the screen's DEM sample cache, which says which work unit gave each
checkpoint a DEM value and where its VRT is.

    python scripts/position_test_checkpoints_3dep.py \\
        --samples data/checkpoints_3dep_opr_samples.parquet --out data/position_test.csv
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parent))
from screen_checkpoints_3dep import GDAL_ENV, WESM_URL  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "data" / "checkpoints-3dep-2.0-qc.parquet"
DEFAULT_SAMPLES = ROOT / "data" / "checkpoints_3dep_opr_samples.parquet"
SEARCH_M = 2.5        # offset grid half-width (m)
STEP_M = 0.1          # offset grid step (m)
MAX_POINTS = 400      # per work unit, seeded sample: statistics need hundreds
SLOPE_MIN_DEG = 5.0   # points flatter than this barely constrain a horizontal shift
MIN_SLOPED = 15       # fewer sloped points than this: undecided
BOOT = 1000           # bootstrap resamples for the decision interval
MIN_SHIFT_M = 0.05    # hypotheses closer than this cannot be told apart


def nmad(a) -> float:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(1.4826 * np.median(np.abs(a - np.median(a)))) if len(a) else np.nan


def horizontal(crs: pyproj.CRS) -> pyproj.CRS:
    return crs.sub_crs_list[0] if crs.is_compound else crs


def datum_name(crs: pyproj.CRS) -> str:
    return horizontal(crs).geodetic_crs.name


def positions(src_crs: pyproj.CRS, dem_crs: pyproj.CRS, e, n):
    """Source coordinates on the DEM grid under both hypotheses, and the
    operation PROJ used for the second (None if only a ballpark exists)."""
    # same frame: unproject in the source datum, re-project with the DEM's
    # projection, no datum operation
    lon, lat = pyproj.Transformer.from_crs(src_crs, src_crs.geodetic_crs,
                                           always_xy=True).transform(e, n)
    xs, ys = pyproj.Transformer.from_crs(dem_crs.geodetic_crs, dem_crs,
                                         always_xy=True).transform(lon, lat)
    try:
        t = pyproj.Transformer.from_crs(src_crs, dem_crs, always_xy=True,
                                        allow_ballpark=False)
    except pyproj.exceptions.ProjError:
        return xs, ys, None, None, None
    xt, yt = t.transform(e, n)
    try:   # the operation PROJ actually used (pyproj >= 3.4)
        op = t.get_last_used_operation().description
    except (AttributeError, pyproj.exceptions.ProjError):
        op = t.description
    return xs, ys, xt, yt, op


def read_patches(vrt: str, xs, ys, half_m: float):
    """DEM patches (own unit) centred on (xs, ys), with their pixel-centre
    origin, plus the DEM's horizontal CRS, pixel size and nodata."""
    with rasterio.Env(**GDAL_ENV), rasterio.open("/vsicurl/" + vrt) as src:
        T = src.transform
        res = abs(T.a)
        k = int(np.ceil(half_m / res)) + 2
        out = []
        for x, y in zip(xs, ys):
            c, r = (x - T.c) / T.a - 0.5, (y - T.f) / T.e - 0.5
            c0, r0 = int(np.floor(c)) - k, int(np.floor(r)) - k
            try:
                a = src.read(1, window=Window(c0, r0, 2 * k + 2, 2 * k + 2),
                             boundless=True, masked=True).astype("float64").filled(np.nan)
            except rasterio.errors.RasterioIOError:
                a = np.full((2 * k + 2, 2 * k + 2), np.nan)
            if src.nodata is not None:
                a[a == src.nodata] = np.nan
            a[(a < -1e4) | (a > 1e5)] = np.nan
            out.append((a, c0, r0))
        return out, T, src.crs.to_wkt()


def bilinear(patches, T, xs, ys) -> np.ndarray:
    v = np.full(len(xs), np.nan)
    for i, ((a, c0, r0), x, y) in enumerate(zip(patches, xs, ys)):
        c, r = (x - T.c) / T.a - 0.5 - c0, (y - T.f) / T.e - 0.5 - r0
        ci, ri = int(np.floor(c)), int(np.floor(r))
        if ci < 0 or ri < 0 or ci + 1 >= a.shape[1] or ri + 1 >= a.shape[0]:
            continue
        fc, fr = c - ci, r - ri
        w = a[ri:ri + 2, ci:ci + 2]
        v[i] = (w[0, 0] * (1 - fc) * (1 - fr) + w[0, 1] * fc * (1 - fr)
                + w[1, 0] * (1 - fc) * fr + w[1, 1] * fc * fr)
    return v


def slope_deg(patches, T, xs, ys) -> np.ndarray:
    """Slope at each point from the DEM (central differences over +-1 pixel)."""
    res = abs(T.a)
    zx = (bilinear(patches, T, xs + res, ys) - bilinear(patches, T, xs - res, ys)) / (2 * res)
    zy = (bilinear(patches, T, xs, ys + res) - bilinear(patches, T, xs, ys - res)) / (2 * res)
    return np.degrees(np.arctan(np.hypot(zx, zy)))


def sample_workunit(task) -> dict:
    """Both candidate positions for a work unit's checkpoints on its DEM grid,
    and a DEM patch around each (read once; scoring happens per project)."""
    wu, vrt, pts = task
    src_crs = pyproj.CRS(pts["source_horizontal_crs_qc"].iloc[0])
    with rasterio.Env(**GDAL_ENV), rasterio.open("/vsicurl/" + vrt) as src:
        dem_crs = horizontal(pyproj.CRS(src.crs.to_wkt()))
    e, n = pts["e"].to_numpy(), pts["n"].to_numpy()
    xs, ys, xt, yt, op = positions(src_crs, dem_crs, e, n)
    rec = dict(workunit=wu, project_id=int(pts["project_id"].iloc[0]), ids=pts["id"].to_numpy(),
               src_crs=src_crs.name, dem_crs=dem_crs.name, src_datum=datum_name(src_crs),
               dem_datum=datum_name(dem_crs), operation=op)
    if xt is None:
        return {**rec, "status": "no transformation"}
    dx, dy = xt - xs, yt - ys
    rec.update(shift_e=float(np.median(dx)), shift_n=float(np.median(dy)))
    rec["shift_m"] = float(np.hypot(rec["shift_e"], rec["shift_n"]))
    if rec["shift_m"] < MIN_SHIFT_M:   # the labels name one frame (or PROJ has a null op)
        return {**rec, "status": "labels agree (not testable)"}
    half = SEARCH_M + float(np.max(np.hypot(dx, dy))) + 1
    patches, T, _ = read_patches(vrt, xs, ys, half)
    return {**rec, "status": "sampled", "xs": xs, "ys": ys, "dx": dx, "dy": dy,
            "patches": patches, "T": T, "unit": float(pts["dem_unit_m"].iloc[0]),
            "z": pts["z_meter_qc"].to_numpy(), "slope": slope_deg(patches, T, xs, ys)}


def score_group(recs: list[dict], slope_min: float = SLOPE_MIN_DEG) -> dict:
    """Decide one (project, source label, DEM label) group from its work units.

    Offsets are applied relative to each point's own 'same frame' position, so
    work units in different projections pool cleanly: 0 = same frame,
    (dx, dy) = labels true (per point), and a grid of common offsets around them."""
    recs = [r for r in recs if r["status"] == "sampled"]
    sl = np.concatenate([r["slope"] for r in recs])
    m = sl >= slope_min

    def dz(r, ox, oy):
        return bilinear(r["patches"], r["T"], r["xs"] + ox, r["ys"] + oy) * r["unit"] - r["z"]

    def nmad_of(f, mask=m):
        """NMAD of dz after removing each work unit's median (work units can
        differ by a vertical datum offset)."""
        parts, k = [], 0
        for r in recs:
            d = f(r)
            mm = mask[k:k + len(d)]
            k += len(d)
            d = d[mm & np.isfinite(d)]
            if len(d):
                parts.append(d - np.median(d))
        return nmad(np.concatenate(parts)) if parts else np.nan

    s_same = nmad_of(lambda r: dz(r, 0, 0))
    s_shift = nmad_of(lambda r: dz(r, r["dx"], r["dy"]))
    se = float(np.median(np.concatenate([r["dx"] for r in recs])))
    sn = float(np.median(np.concatenate([r["dy"] for r in recs])))
    out = dict(project_id=recs[0]["project_id"], src_datum=recs[0]["src_datum"],
               dem_datum=recs[0]["dem_datum"], src_crs=recs[0]["src_crs"],
               dem_crs=recs[0]["dem_crs"], operation=recs[0]["operation"],
               workunits=len(recs), n=int(len(sl)), n_sloped=int(m.sum()),
               median_slope_deg=float(np.nanmedian(sl)), shift_e=se, shift_n=sn,
               shift_m=float(np.hypot(se, sn)), nmad_same=s_same, nmad_shift=s_shift,
               nmad_all_same=nmad_of(lambda r: dz(r, 0, 0), np.isfinite(sl)),
               nmad_all_shift=nmad_of(lambda r: dz(r, r["dx"], r["dy"]), np.isfinite(sl)))
    if m.sum() >= MIN_SLOPED:
        g = np.arange(-SEARCH_M, SEARCH_M + 1e-9, STEP_M)
        cx, cy = se / 2, sn / 2   # grid centred between the two hypotheses
        grid = np.array([[nmad_of(lambda r: dz(r, cx + a, cy + b)) for a in g] for b in g])
        bi, bj = np.unravel_index(np.nanargmin(grid), grid.shape)
        out.update(best_e=float(cx + g[bj]), best_n=float(cy + g[bi]),
                   nmad_best=float(grid[bi, bj]))
    # paired bootstrap over sloped points: NMAD(shift) - NMAD(same), 95% interval
    d_same, d_shift = [], []
    k = 0
    for r in recs:
        a, b = dz(r, 0, 0), dz(r, r["dx"], r["dy"])
        mm = m[k:k + len(a)] & np.isfinite(a) & np.isfinite(b)
        k += len(a)
        if mm.any():
            d_same.append(a[mm] - np.median(a[mm]))
            d_shift.append(b[mm] - np.median(b[mm]))
    lo = hi = np.nan
    if d_same and sum(len(x) for x in d_same) >= MIN_SLOPED:
        a, b = np.concatenate(d_same), np.concatenate(d_shift)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(a), (BOOT, len(a)))
        diff = [nmad(b[i]) - nmad(a[i]) for i in idx]
        lo, hi = np.percentile(diff, [2.5, 97.5])
    out.update(diff_lo=float(lo), diff_hi=float(hi))
    if "best_e" in out:   # where the free offset search landed, relative to the two answers
        out["best_to_same_m"] = float(np.hypot(out["best_e"], out["best_n"]))
        out["best_to_shift_m"] = float(np.hypot(out["best_e"] - se, out["best_n"] - sn))
    if m.sum() < MIN_SLOPED or not np.isfinite([s_same, s_shift, lo, hi]).all():
        out["verdict"] = "undecided (too few sloped points)"
    elif hi < 0:      # shifting is better in 97.5% of resamples
        out["verdict"] = "labels true"
    elif lo > 0:
        out["verdict"] = "same frame"
    else:
        out["verdict"] = "undecided"
    return out


def build_tasks(cp, samples, wesm, vrts, seed=0):
    """One task per (work unit, source CRS) whose labels differ from the DEM's."""
    s = samples[samples["dem_raw"].notna()][["id", "workunit", "dem_unit_m", "dem_crs"]]
    cp = cp.merge(s, on="id")
    cp = cp[cp["source_horizontal_crs_qc"].notna() & cp["z_meter_qc"].notna()]
    swap = cp["source_axes_swapped_qc"].fillna(False).to_numpy(bool)
    cp = cp.assign(e=np.where(swap, cp["source_northing"], cp["source_easting"]),
                   n=np.where(swap, cp["source_easting"], cp["source_northing"]))
    # cheap label comparison from the cached DEM CRS name; work units whose
    # name is missing are opened to find out
    cp["dem_datum_guess"] = cp["dem_crs"].str.split(" \\+ ").str[0].str.split(" / ").str[0]
    src_datum = {c: datum_name(pyproj.CRS(c)) for c in cp["source_horizontal_crs_qc"].unique()}
    cp["src_datum"] = cp["source_horizontal_crs_qc"].map(src_datum)
    named = cp["dem_crs"].str.contains(" / ", na=False)
    cp = cp[~named | (cp["src_datum"] != cp["dem_datum_guess"])]
    link = wesm.drop_duplicates("workunit").set_index("workunit")["sourcedem_link"]
    cp["vrt"] = cp["workunit"].map(link).map(vrts)
    tasks = []
    for (wu, _crs), g in cp[cp["vrt"].notna()].groupby(["workunit", "source_horizontal_crs_qc"]):
        if len(g) > MAX_POINTS:
            g = g.sample(MAX_POINTS, random_state=seed)
        tasks.append((wu, g["vrt"].iloc[0], g))
    return tasks


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input", type=Path, default=DEFAULT_IN)
    p.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES,
                   help="screen_checkpoints_3dep.py sample cache (its *_parts/vrts.csv too)")
    p.add_argument("--wesm", default=WESM_URL)
    p.add_argument("--out", type=Path, required=True, help="per-work-unit results (CSV)")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--projects", type=int, nargs="*", help="only these project_ids")
    p.add_argument("--slope", type=float, default=SLOPE_MIN_DEG,
                   help="minimum slope (deg) of a point used to decide")
    args = p.parse_args(argv)
    pyproj.network.set_network_enabled(True)

    cp = pd.read_parquet(args.input, columns=[
        "id", "project_id", "source_easting", "source_northing", "source_horizontal_crs_qc",
        "source_axes_swapped_qc", "z_meter_qc", "qc_horizontal_flag"])
    if args.projects:
        cp = cp[cp["project_id"].isin(args.projects)]
    samples = pd.read_parquet(args.samples)
    vrts = pd.read_csv(args.samples.with_name(args.samples.stem + "_parts") / "vrts.csv"
                       ).set_index("sourcedem_link")["vrt"]
    wesm = pd.read_csv(args.wesm, usecols=["workunit", "sourcedem_link"])
    tasks = build_tasks(cp, samples, wesm, vrts)
    print(f"{len(tasks)} work-unit tasks, {sum(len(t[2]) for t in tasks)} checkpoints",
          file=sys.stderr)

    cache = args.out.with_suffix(".parts")
    cache.mkdir(parents=True, exist_ok=True)
    recs, failed, t0 = [], [], time.perf_counter()

    def run(task):
        key = cache / (re.sub(r"[^A-Za-z0-9_.-]", "_", f"{task[0]}_"
                              f"{task[2]['source_horizontal_crs_qc'].iloc[0]}") + ".pkl")
        if key.exists():
            return pickle.loads(key.read_bytes())
        rec = sample_workunit(task)
        key.write_bytes(pickle.dumps(rec))
        return rec

    with ThreadPoolExecutor(args.threads) as pool:
        futs = {pool.submit(run, t): t[0] for t in tasks}
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                recs.append(fut.result())
            except Exception as e:  # one bad VRT must not sink the run
                failed.append((futs[fut], repr(e)[:200]))
            if k % 20 == 0:
                print(f"  {k}/{len(tasks)} ({time.perf_counter() - t0:.0f} s)", file=sys.stderr)
    for wu, err in failed:
        print(f"  failed: {wu}: {err}", file=sys.stderr)

    status = pd.DataFrame([{k: r.get(k) for k in ("workunit", "project_id", "src_datum",
                                                  "dem_datum", "status", "shift_m")}
                           for r in recs])
    print(status["status"].value_counts().to_string(), file=sys.stderr)
    groups = {}
    for r in recs:
        if r["status"] == "sampled":
            groups.setdefault((r["project_id"], r["src_datum"], r["dem_datum"]), []).append(r)
    out = pd.DataFrame([score_group(g, args.slope) for g in groups.values()])
    out = out.sort_values(["verdict", "project_id"])
    out.to_csv(args.out, index=False)
    status.to_csv(args.out.with_suffix(".workunits.csv"), index=False)
    print(out["verdict"].value_counts().to_string())


if __name__ == "__main__":
    main()
