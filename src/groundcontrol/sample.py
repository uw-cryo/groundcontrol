"""Point sampling of rasters — memory-safe block-wise windowed reads.

Port of the ``sample_raster`` snapshot in docs/plan.md Appendix A5 with the
Appendix B sampler fixes applied:

- B3: NaN-nodata float rasters are masked correctly (``arr == nodata`` is
  always ``False`` when nodata is NaN); non-finite values are always masked.
- B4: coordinate monotonicity is normalized before xarray ``sel``/``interp``
  (north-up rasters carry a *descending* y — an interpolation footgun); both
  north-up and south-up rasters are supported.
- B11: methods are restricted to {'nearest', 'linear'} with a per-method tile
  halo ({nearest: 0, linear: 2}); anything else raises (cubic/quintic need a
  larger halo and seam testing before they can be offered).

API differences from the snapshot (Increment-1 port contract): the caller's
GeoDataFrame is never mutated — a copy with the new column(s) is returned;
CRS agreement between points and raster is asserted (the plan's
vertical-datum-reconciliation step 3: never silently mis-sample) with a
loudly-logged ``check_crs=False`` escape hatch; the parallel ThreadPool
variant is deferred.

Beyond the snapshot: ``radius=`` neighborhood sampling (median/NMAD/n over all
pixels whose cell centers fall within a radius of each point) for assessed
products with inherent geolocation uncertainty and for pre/post co-registration
comparisons — see :func:`sample_raster`.

Resampling rule (every repo that measures from a DEM): point samples are
native-resolution windowed reads of the source file — never an overview level
or a decimated ``out_shape`` read. Overviews are a half-cell lattice away from
the source and their content depends on how they were built: ``nearest``
(gdaladdo's default) and ``mode`` shift content half an input pixel per level
toward E and S, ``gauss`` the same amounts toward W and N, cumulative with the
pyramid (measured on a 0.5 m DTM: ±0.25 m at x2, ±0.75 m at x4); only
``average`` and ``cubic`` overviews are shift-free, and a file does not say
which it carries. Any geometry-bearing resample of a DEM (co-registration
input, dh map) is ``gdalwarp -r average -ovr NONE`` from full resolution;
``nearest`` only for classified rasters, and then via ``gdalwarp`` at an ODD
integer ratio (shift-free on classes); at an even ratio ``near`` shifts half a
source pixel toward W and N and ``mode`` toward E and S, and ``gdaladdo -r
nearest`` shifts (ratio-1)/2 source pixels at odd ratios too, so never through
overviews either. A planted-shift test goes through the same read path as the
data. Overview or
averaged reads are confined to footprint masks (``aoi``) and display (``plot``);
nothing here reads them.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pyproj
import rasterio
import rioxarray  # noqa: F401  (registers the .rio accessor on xarray objects)
import xarray as xr
from rasterio.windows import Window

from groundcontrol.accuracy import med_nmad
from groundcontrol.crs import compare_frames, grid_signature, horizontal_2d, vertical_kind

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#: sampling method -> pixel halo added around each tile read (plan B11).
METHOD_HALO = {"nearest": 0, "linear": 2}


def _open_dataarray(r):
    """Normalize input: a path becomes a rioxarray DataArray named after the file."""
    if isinstance(r, (str, Path)):
        da = rioxarray.open_rasterio(r)
        da.name = Path(r).stem
        return da
    if isinstance(r, xr.DataArray):
        return r
    raise TypeError(f"r must be a rioxarray DataArray or a raster path, got {type(r)!r}")


def _squeeze_band(da: xr.DataArray) -> xr.DataArray:
    if "band" in da.dims:
        if da.sizes["band"] != 1:
            raise ValueError(
                f"sample_raster expects a single-band raster; got {da.sizes['band']} bands "
                "(select one, e.g. r.isel(band=0))"
            )
        da = da.isel(band=0, drop=True)
    return da


# the frame-identity helpers live in crs (one definition for every consumer);
# the private names stay importable for existing callers (assess, tests)
_horizontal_2d = horizontal_2d
_grid_signature = grid_signature


def _check_vertical(points_crs, raster_crs, height_kind, allow_vertical_mismatch) -> None:
    """Refuse a DECLARED vertical conflict between the points' heights and the
    raster's (a NAVD88 compound raster against ellipsoidal points is ~25 m at
    a CONUS site and passed silently before v0.3). ``height_kind`` declares
    the points' heights when their CRS is 2D; an undeclared side (the common
    2D DEM header) is logged, never refused — it cannot be checked, only
    trusted."""
    declared = vertical_kind(points_crs)
    if height_kind is not None and declared not in ("undeclared", height_kind):
        raise ValueError(f"height_kind={height_kind!r} contradicts the points' own CRS, which "
                         f"declares {declared} heights ({pyproj.CRS.from_user_input(points_crs).name!r})")
    # a declaring points CRS wins over a (consistent) height_kind so that two gravity
    # verticals still get compared as datums (NAVD88 vs NGVD29), not just as kinds
    pk = declared if declared != "undeclared" else (height_kind or declared)
    rk = vertical_kind(raster_crs)
    if pk == "depth" or rk == "depth":
        raise ValueError(f"depth-type vertical axis (points {pk}, raster {rk}): heights are "
                         "assumed up everywhere downstream")
    if pk == "undeclared" or rk == "undeclared":
        conflict = False
    elif pk != rk:
        conflict = True
    elif declared != "undeclared":
        # both declared the same kind: two gravity verticals can still differ
        conflict = compare_frames(points_crs, raster_crs).vertical_conflict
    else:
        conflict = False
    if conflict:
        msg = (f"points carry {pk} heights ({pyproj.CRS.from_user_input(points_crs).name!r}), "
               f"the raster {rk} heights ({pyproj.CRS.from_user_input(raster_crs).name!r}): "
               "refusing to difference a vertical-datum separation into dz. Transform the "
               "heights first (groundcontrol.crs), or pass allow_vertical_mismatch=True "
               "(expert use, logged).")
        if not allow_vertical_mismatch:
            raise ValueError(msg)
        warnings.warn("sample_raster(allow_vertical_mismatch=True): " + msg, stacklevel=4)
        logger.warning("sample_raster(allow_vertical_mismatch=True): %s", msg)
        return
    if pk == "undeclared" or rk == "undeclared":
        logger.info("sample_raster: vertical datum not checked (points %s, raster %s); the "
                    "heights are assumed to share one", pk, rk)


def _check_crs(gdf, da: xr.DataArray, check_crs: bool,
               declared_crs=None, height_kind=None, allow_vertical_mismatch=False) -> None:
    """Fail loud on point/raster frame disagreement (never silently
    mis-sample). ``declared_crs``: the frame the caller DECLARED the
    raster to actually be in (--target-crs / --vdatum semantics — the
    header's datum is overridden, the grid is not): points matching the
    declaration are accepted when the declaration addresses the same
    grid as the header (same projection, datum-blind); a different grid
    still refuses. Then the vertical: see :func:`_check_vertical`."""
    raster_crs = da.rio.crs
    if not check_crs:
        msg = (
            "sample_raster(check_crs=False): skipping CRS agreement check "
            f"(points: {gdf.crs}; raster: {raster_crs}). Results are only valid if the "
            "coordinates genuinely share a frame — expert use only."
        )
        warnings.warn(msg, stacklevel=3)
        logger.warning(msg)
        return
    if gdf.crs is None or raster_crs is None:
        raise ValueError(
            f"cannot verify CRS agreement: points CRS is {gdf.crs!r}, raster CRS is "
            f"{raster_crs!r}. Set both (e.g. r.rio.write_crs(...)) or pass check_crs=False "
            "(expert use) to override."
        )
    if not _horizontal_2d(gdf.crs).equals(_horizontal_2d(raster_crs)):
        if (declared_crs is not None
                and _horizontal_2d(gdf.crs).equals(_horizontal_2d(declared_crs))
                and _grid_signature(declared_crs) == _grid_signature(raster_crs)):
            logger.warning(
                "sampling under the declared frame %r; the raster header "
                "says %r — same grid, datum reinterpretation (the "
                "--target-crs/--vdatum contract), not a transform",
                pyproj.CRS.from_user_input(declared_crs).name,
                pyproj.CRS.from_user_input(raster_crs).name)
            _check_vertical(gdf.crs, declared_crs, height_kind, allow_vertical_mismatch)
            return
        raise ValueError(
            "points and raster are in different CRSs — refusing to silently mis-sample. "
            f"points: {pyproj.CRS.from_user_input(gdf.crs).name!r}; "
            f"raster: {pyproj.CRS.from_user_input(raster_crs).name!r}. "
            "Transform the points into the raster's CRS first (see groundcontrol.crs), "
            "or pass check_crs=False (expert use) to override."
        )
    # the declaration overrides the header's vertical too (--vdatum: "the heights ARE NAVD88"),
    # but only a declaration OF THIS RASTER counts: same grid as the header, same horizontal
    # frame as the points (the mismatch branch above checks the same two things)
    if declared_crs is not None and not (
            horizontal_2d(declared_crs).equals(horizontal_2d(gdf.crs))
            and grid_signature(declared_crs) == grid_signature(raster_crs)):
        raise ValueError(
            f"declared_crs {pyproj.CRS.from_user_input(declared_crs).name!r} does not describe "
            f"this raster ({pyproj.CRS.from_user_input(raster_crs).name!r}) in the points' frame "
            f"({pyproj.CRS.from_user_input(gdf.crs).name!r}): a declaration must address the "
            "header's grid and the points' horizontal frame")
    _check_vertical(gdf.crs, declared_crs if declared_crs is not None else raster_crs,
                    height_kind, allow_vertical_mismatch)


def _ascending(arr: np.ndarray, xc: np.ndarray, yc: np.ndarray):
    """Normalize coordinate monotonicity (plan B4): flip to ascending x and y."""
    if yc.size > 1 and yc[0] > yc[-1]:
        yc = yc[::-1]
        arr = arr[::-1, :]
    if xc.size > 1 and xc[0] > xc[-1]:
        xc = xc[::-1]
        arr = arr[:, ::-1]
    return arr, xc, yc


def _mask_nodata(arr: np.ndarray, nodata) -> np.ndarray:
    """Nodata -> NaN, robust to NaN-nodata float rasters (plan B3)."""
    if nodata is not None and not np.isnan(nodata):
        arr[arr == nodata] = np.nan
    arr[~np.isfinite(arr)] = np.nan  # always: NaN nodata, inf, unmasked sentinels
    return arr


def _radius_stats(arr, xc, yc, xq, yq, radius):
    """Per-point neighborhood stats over pixels whose CELL CENTERS fall within
    ``radius`` of the point: (median, nmad, n) arrays. ``xc``/``yc`` ascending;
    ``arr`` already nodata-masked (non-finite pixels are excluded and not counted).
    """
    npt = len(xq)
    med = np.full(npt, np.nan, dtype="float64")
    nmad = np.full(npt, np.nan, dtype="float64")
    cnt = np.zeros(npt, dtype="int64")
    r2 = radius * radius
    for k in range(npt):
        x, y = xq[k], yq[k]
        i0 = np.searchsorted(yc, y - radius, side="left")
        i1 = np.searchsorted(yc, y + radius, side="right")
        j0 = np.searchsorted(xc, x - radius, side="left")
        j1 = np.searchsorted(xc, x + radius, side="right")
        if i0 >= i1 or j0 >= j1:
            continue
        dy = yc[i0:i1] - y
        dx = xc[j0:j1] - x
        within = (dx[None, :] ** 2 + dy[:, None] ** 2) <= r2
        v = arr[i0:i1, j0:j1][within]
        v = v[np.isfinite(v)]
        if v.size:
            med[k], nmad[k] = med_nmad(v)  # single source of the 1.4826 constant
            cnt[k] = v.size
    return med, nmad, cnt



#: max wasted pixels a group read may spend per point before it is split
#: (owner 2026-08-30, Las Vegas 0.5 m VRT: fixed 4096-px blocks read 1.5 GB
#: — 241 windows up to 13 Mpx — to sample 671 scattered points needing ~6 kpx;
#: ~30 s per product)
_PER_POINT_PX = 256 * 256


def _point_windows(rows, cols, idx, halo_r, halo_c, H, W, block):
    """Yield ``(idx, rr0, rr1, cc0, cc1)`` read groups for the point set.

    Recursive spatial split: a group is read in one window when the tight
    bbox (+halo) fits the ``block``-squared memory cap AND wastes at most
    ``_PER_POINT_PX`` pixels per point; otherwise it splits at the midpoint
    of the longer point-span axis (both halves provably non-empty) and
    recurses. Dense clusters still amortize into one read; isolated points
    read a few-pixel window instead of a multi-megapixel tile. A group
    whose points share one pixel (or a lone point with a big radius halo)
    can no longer split and is yielded regardless of the waste cap.
    """
    sr, sc = rows[idx], cols[idx]
    rr0, rr1 = max(0, int(sr.min()) - halo_r), min(H, int(sr.max()) + halo_r + 1)
    cc0, cc1 = max(0, int(sc.min()) - halo_c), min(W, int(sc.max()) + halo_c + 1)
    area = (rr1 - rr0) * (cc1 - cc0)
    rspan, cspan = int(sr.max() - sr.min()), int(sc.max() - sc.min())
    if (area <= block * block and area <= len(idx) * _PER_POINT_PX) \
            or (rspan == 0 and cspan == 0):
        yield idx, rr0, rr1, cc0, cc1
        return
    v = sr if rspan >= cspan else sc
    cut = (int(v.min()) + int(v.max())) // 2  # span > 0: both halves non-empty
    lo = v <= cut
    yield from _point_windows(rows, cols, idx[lo], halo_r, halo_c, H, W, block)
    yield from _point_windows(rows, cols, idx[~lo], halo_r, halo_c, H, W, block)


def _sample_windowed(src_fn: str, xs_pt, ys_pt, method: str, block: int, radius=None):
    """Block-wise windowed sampling from the raster's source file (memory-safe).

    Returns a values array (interpolation mode), or ``(median, nmad, n)``
    arrays when ``radius`` is given (neighborhood mode).
    """
    n_pt = len(xs_pt)
    vals = np.full(n_pt, np.nan, dtype="float64")
    r_med = np.full(n_pt, np.nan, dtype="float64")
    r_nmad = np.full(n_pt, np.nan, dtype="float64")
    r_cnt = np.zeros(n_pt, dtype="int64")
    with rasterio.open(src_fn) as ds:
        T = ds.transform
        if T.b or T.d:
            raise NotImplementedError("rotated/sheared rasters are not supported")
        nodata = ds.nodata
        H, W = ds.height, ds.width
        if radius is not None:
            # every cell center within radius of a point inside its containing
            # pixel is <= radius + one pixel away from that pixel's center
            halo_c = int(np.ceil(radius / abs(T.a))) + 1
            halo_r = int(np.ceil(radius / abs(T.e))) + 1
        else:
            halo_c = halo_r = METHOD_HALO[method]
        finite = np.isfinite(xs_pt) & np.isfinite(ys_pt)
        rows = np.full(n_pt, -1, dtype="int64")
        cols = np.full(n_pt, -1, dtype="int64")
        # containing-pixel indices; T.e < 0 (north-up) and T.e > 0 (south-up) both work
        cols[finite] = np.floor((xs_pt[finite] - T.c) / T.a).astype("int64")
        rows[finite] = np.floor((ys_pt[finite] - T.f) / T.e).astype("int64")
        inb = finite & (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
        if not inb.any():
            return vals if radius is None else (r_med, r_nmad, r_cnt)
        for idx, rr0, rr1, cc0, cc1 in _point_windows(
                rows, cols, np.where(inb)[0], halo_r, halo_c, H, W, block):
                win = Window(cc0, rr0, cc1 - cc0, rr1 - rr0)
                arr = ds.read(1, window=win).astype("float64")
                arr = _mask_nodata(arr, nodata)
                wt = ds.window_transform(win)
                ny, nx = arr.shape
                xc = wt.c + (np.arange(nx) + 0.5) * wt.a
                yc = wt.f + (np.arange(ny) + 0.5) * wt.e
                arr, xc, yc = _ascending(arr, xc, yc)
                if radius is not None:
                    m, s, c = _radius_stats(arr, xc, yc, xs_pt[idx], ys_pt[idx], radius)
                    r_med[idx], r_nmad[idx], r_cnt[idx] = m, s, c
                    continue
                da = xr.DataArray(arr, coords={"y": yc, "x": xc}, dims=("y", "x"))
                xq = xr.DataArray(xs_pt[idx], dims="z")
                yq = xr.DataArray(ys_pt[idx], dims="z")
                s = (da.sel(x=xq, y=yq, method="nearest") if method == "nearest"
                     else da.interp(x=xq, y=yq, method=method))
                vals[idx] = s.values
    return vals if radius is None else (r_med, r_nmad, r_cnt)


def _sample_in_memory(da: xr.DataArray, xs_pt, ys_pt, method: str, radius=None):
    """Fallback for computed arrays with no source file (loads via xarray directly)."""
    nodata = da.rio.nodata
    if nodata is not None and not np.isnan(nodata):
        da = da.where(da != nodata)
    da = da.where(np.isfinite(da))
    # B4: normalize to ascending coords before sel/interp
    if da.sizes["y"] > 1 and da.y.values[0] > da.y.values[-1]:
        da = da.isel(y=slice(None, None, -1))
    if da.sizes["x"] > 1 and da.x.values[0] > da.x.values[-1]:
        da = da.isel(x=slice(None, None, -1))
    if radius is not None:
        return _radius_stats(da.values.astype("float64"), da.x.values, da.y.values,
                             xs_pt, ys_pt, radius)
    xq = xr.DataArray(np.asarray(xs_pt, dtype="float64"), dims="z")
    yq = xr.DataArray(np.asarray(ys_pt, dtype="float64"), dims="z")
    if method == "nearest":
        s = da.sel(x=xq, y=yq, method="nearest").values.astype("float64")
        # sel(nearest) snaps out-of-bounds points to the edge pixel — mask beyond
        # the half-pixel margin (the windowed path excludes them via the inb mask)
        xv, yv = da.x.values, da.y.values
        hx = float(np.median(np.diff(xv)) / 2) if xv.size > 1 else np.inf
        hy = float(np.median(np.diff(yv)) / 2) if yv.size > 1 else np.inf
        out = (
            (xq.values >= xv[0] - hx) & (xq.values <= xv[-1] + hx)
            & (yq.values >= yv[0] - hy) & (yq.values <= yv[-1] + hy)
        )
        s[~out] = np.nan
        return s
    return da.interp(x=xq, y=yq, method=method).values.astype("float64")


def sample_raster(gdf, r, col: str = "height", method: str = "linear", diff: bool = False,
                  block: int = 4096, check_crs: bool = True, radius=None,
                  declared_crs=None, height_kind: str | None = None,
                  allow_vertical_mismatch: bool = False):
    """Sample raster ``r`` at the points of ``gdf``; return a copy with new column(s).

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Point geometries **in the raster's CRS** (asserted; see ``check_crs``).
        Never mutated — the sampled values come back on a copy.
    r : xarray.DataArray or str/Path
        Single-band raster opened with rioxarray, or a path to one. When the
        DataArray carries a source file (``r.encoding['source']``), sampling
        is block-wise via windowed rasterio reads (memory-safe for huge
        rasters) — note the *file contents* are sampled, not any in-memory
        modifications; computed arrays without a source use xarray directly.
    col : str
        Column holding point heights, used only when ``diff=True``.
    method : {'nearest', 'linear'}
        Sampling method (plan B11: other methods are rejected). Mutually
        exclusive with ``radius`` — leave at the default when ``radius`` is set.
    diff : bool
        Also add ``'<name> minus <col>'`` (raster minus point heights; the
        neighborhood median in radius mode).
    block : int
        Tile size in pixels for the windowed path.
    check_crs : bool
        Require point/raster CRS agreement (default). ``False`` skips the
        check with a loud warning — expert use only.
    height_kind : {'ellipsoidal', 'gravity'}, optional
        What the points' heights are when their CRS does not say (a 2D
        ``gdf.crs``); a 3D or compound ``gdf.crs`` declares it already. Checked
        against the raster's vertical (``declared_crs`` when given, else the
        header): a declared conflict (NAVD88 raster, ellipsoidal points) refuses;
        an undeclared side (the usual 2D DEM header) is logged and trusted.
    allow_vertical_mismatch : bool
        Accept a declared vertical conflict with a loud warning — expert use.
    radius : float, optional
        Neighborhood sampling radius in raster CRS units (meters for projected
        CRSs). Instead of interpolating at the exact point, gather every pixel
        whose CELL CENTER falls within ``radius`` of the point and report
        robust per-point stats: ``<name>`` = neighborhood median,
        ``<name>_nmad`` (via :func:`groundcontrol.accuracy.med_nmad` — the
        single 1.4826 source) and ``<name>_n`` = count of finite pixels used.
        Use case: the assessed product can carry inherent geolocation error
        (e.g. a few meters for satellite-derived DEMs), so the pixel at the
        nominal control-point location may not be the actual point; also for
        before/after co-registration comparisons. Fail-honest: if zero cell
        centers fall within ``radius`` the result is NaN with ``n=0`` — choose
        ``radius >= ~0.71 * resolution`` (half the pixel diagonal) to guarantee
        at least one center. Points outside the raster extent are NaN/0 even
        if the radius overlaps the edge (consistent with interpolation mode).

        Choosing a radius (owner guidance + empirical notes, 2026-07): scale
        it to the ASSESSED product's geolocation class — near-zero (prefer
        bilinear) for accurately-georeferenced lidar, ~3-4 m for ~3 m-class
        satellite DSMs, larger for coarse products (e.g. 30 m global DEMs).
        Empirically (0.5 m lidar DTM and 0.5 m satellite DSM, ~150 px per
        neighborhood): the neighborhood median does NOT tighten checkpoint
        NMAD — a *systematic* geolocation offset shifts the whole neighborhood
        with the product, and only co-registration removes it. What radius
        sampling buys instead: outlier/artifact robustness, comparison
        stability before/after co-registration, and the per-point
        ``<name>_nmad`` as a local roughness/quality flag (useful for
        filtering assessment points).

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of ``gdf`` with a column named after the raster (``r.name`` or
        the file stem; ``'sampled'`` when anonymous), NaN where the raster has
        nodata or the point falls outside it. Radius mode adds ``<name>_nmad``
        and ``<name>_n`` alongside.
    """
    if radius is not None:
        if not (np.isfinite(radius) and radius > 0):
            raise ValueError(f"radius must be a positive number, got {radius!r}")
        if method != "linear":  # non-default method + radius: ambiguous request
            raise ValueError(
                "radius neighborhood sampling is mutually exclusive with interpolation; "
                f"leave method at its default (got method={method!r})"
            )
    elif method not in METHOD_HALO:
        raise ValueError(
            f"unsupported method {method!r}; supported: {sorted(METHOD_HALO)} (plan B11 — "
            "higher-order methods need a larger tile halo and seam validation)"
        )
    da = _squeeze_band(_open_dataarray(r))
    if height_kind is not None and height_kind not in ("ellipsoidal", "gravity"):
        raise ValueError(f"height_kind must be 'ellipsoidal' or 'gravity', got {height_kind!r}")
    _check_crs(gdf, da, check_crs, declared_crs=declared_crs, height_kind=height_kind,
               allow_vertical_mismatch=allow_vertical_mismatch)
    src_fn = da.encoding.get("source")
    name = da.name or (Path(src_fn).stem if src_fn else "sampled")

    out = gdf.copy()
    xs_pt = out.geometry.x.to_numpy(dtype="float64")
    ys_pt = out.geometry.y.to_numpy(dtype="float64")
    if src_fn:
        res = _sample_windowed(src_fn, xs_pt, ys_pt, method, block, radius=radius)
    else:
        res = _sample_in_memory(da, xs_pt, ys_pt, method, radius=radius)
    if radius is None:
        vals = res
        out[name] = vals
    else:
        vals, nmad, cnt = res
        out[name] = vals
        out[f"{name}_nmad"] = nmad
        out[f"{name}_n"] = cnt
    logger.info("sampled %s at %d points (%s): %d valid", name, len(out),
                f"radius={radius}" if radius is not None else method,
                int(np.isfinite(vals).sum()))
    if diff:
        if col not in out.columns:
            raise ValueError(f"diff=True needs point-height column {col!r} in gdf")
        out[f"{name} minus {col}"] = out[name] - out[col]
    return out
