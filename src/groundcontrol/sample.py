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


def _horizontal_2d(crs) -> pyproj.CRS:
    """Horizontal 2D component of a CRS (compound -> horizontal member; 3D -> 2D)."""
    crs = pyproj.CRS.from_user_input(crs)
    if crs.is_compound:
        crs = pyproj.CRS(crs.sub_crs_list[0])
    return crs.to_2d()


def _check_crs(gdf, da: xr.DataArray, check_crs: bool) -> None:
    """Fail loud on point/raster frame disagreement (never silently mis-sample)."""
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
        raise ValueError(
            "points and raster are in different CRSs — refusing to silently mis-sample. "
            f"points: {pyproj.CRS.from_user_input(gdf.crs).name!r}; "
            f"raster: {pyproj.CRS.from_user_input(raster_crs).name!r}. "
            "Transform the points into the raster's CRS first (see groundcontrol.crs), "
            "or pass check_crs=False (expert use) to override."
        )


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


def _sample_windowed(src_fn: str, xs_pt, ys_pt, method: str, block: int) -> np.ndarray:
    """Block-wise windowed sampling from the raster's source file (memory-safe)."""
    halo = METHOD_HALO[method]
    vals = np.full(len(xs_pt), np.nan, dtype="float64")
    with rasterio.open(src_fn) as ds:
        T = ds.transform
        if T.b or T.d:
            raise NotImplementedError("rotated/sheared rasters are not supported")
        nodata = ds.nodata
        H, W = ds.height, ds.width
        finite = np.isfinite(xs_pt) & np.isfinite(ys_pt)
        rows = np.full(len(xs_pt), -1, dtype="int64")
        cols = np.full(len(xs_pt), -1, dtype="int64")
        # containing-pixel indices; T.e < 0 (north-up) and T.e > 0 (south-up) both work
        cols[finite] = np.floor((xs_pt[finite] - T.c) / T.a).astype("int64")
        rows[finite] = np.floor((ys_pt[finite] - T.f) / T.e).astype("int64")
        inb = finite & (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
        if not inb.any():
            return vals
        rmin, rmax = rows[inb].min(), rows[inb].max()
        cmin, cmax = cols[inb].min(), cols[inb].max()
        for r0 in range(rmin, rmax + 1, block):
            for c0 in range(cmin, cmax + 1, block):
                sel = inb & (rows >= r0) & (rows < r0 + block) & (cols >= c0) & (cols < c0 + block)
                if not sel.any():
                    continue
                sr, sc = rows[sel], cols[sel]
                # tile + halo, clamped to the dataset (window never leaves it)
                rr0, rr1 = max(0, sr.min() - halo), min(H, sr.max() + halo + 1)
                cc0, cc1 = max(0, sc.min() - halo), min(W, sc.max() + halo + 1)
                win = Window(cc0, rr0, cc1 - cc0, rr1 - rr0)
                arr = ds.read(1, window=win).astype("float64")
                arr = _mask_nodata(arr, nodata)
                wt = ds.window_transform(win)
                ny, nx = arr.shape
                xc = wt.c + (np.arange(nx) + 0.5) * wt.a
                yc = wt.f + (np.arange(ny) + 0.5) * wt.e
                arr, xc, yc = _ascending(arr, xc, yc)
                da = xr.DataArray(arr, coords={"y": yc, "x": xc}, dims=("y", "x"))
                idx = np.where(sel)[0]
                xq = xr.DataArray(xs_pt[idx], dims="z")
                yq = xr.DataArray(ys_pt[idx], dims="z")
                s = (da.sel(x=xq, y=yq, method="nearest") if method == "nearest"
                     else da.interp(x=xq, y=yq, method=method))
                vals[idx] = s.values
    return vals


def _sample_in_memory(da: xr.DataArray, xs_pt, ys_pt, method: str) -> np.ndarray:
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
                  block: int = 4096, check_crs: bool = True):
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
        Sampling method (plan B11: other methods are rejected).
    diff : bool
        Also add ``'<name> minus <col>'`` (raster minus point heights).
    block : int
        Tile size in pixels for the windowed path.
    check_crs : bool
        Require point/raster CRS agreement (default). ``False`` skips the
        check with a loud warning — expert use only.

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of ``gdf`` with a column named after the raster (``r.name`` or
        the file stem; ``'sampled'`` when anonymous), NaN where the raster has
        nodata or the point falls outside it.
    """
    if method not in METHOD_HALO:
        raise ValueError(
            f"unsupported method {method!r}; supported: {sorted(METHOD_HALO)} (plan B11 — "
            "higher-order methods need a larger tile halo and seam validation)"
        )
    da = _squeeze_band(_open_dataarray(r))
    _check_crs(gdf, da, check_crs)
    src_fn = da.encoding.get("source")
    name = da.name or (Path(src_fn).stem if src_fn else "sampled")

    out = gdf.copy()
    xs_pt = out.geometry.x.to_numpy(dtype="float64")
    ys_pt = out.geometry.y.to_numpy(dtype="float64")
    if src_fn:
        vals = _sample_windowed(src_fn, xs_pt, ys_pt, method, block)
    else:
        vals = _sample_in_memory(da, xs_pt, ys_pt, method)
    out[name] = vals
    logger.info("sampled %s at %d points (%s): %d valid", name, len(out), method,
                int(np.isfinite(vals).sum()))
    if diff:
        if col not in out.columns:
            raise ValueError(f"diff=True needs point-height column {col!r} in gdf")
        out[f"{name} minus {col}"] = out[name] - out[col]
    return out
