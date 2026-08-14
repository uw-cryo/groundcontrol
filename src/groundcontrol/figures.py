"""Standard per-site control-point figure bundle.

Every site-level control fetch should ship the same default figure set
(requested by David, 2026-07-10, after the Casa Grande cal-range run):

1. ``{site}_control_map.png``       all points by type over shaded relief
2. ``{site}_monument_types.png``    NGS monuments faceted by posSource /
                                    vertSource / vertOrder (datasheet quality
                                    attributes retained in ``raw``)
3. ``{site}_midas_velocity_horiz.png``    MIDAS horizontal quiver (plot.py)
4. ``{site}_midas_velocity_vertical.png`` same, colored by vel_u (RED =
                                    subsidence) — the vertical-motion view

Conventions: NVA plots ABOVE VVA; AOI outline is light/transparent; scalebar
on map panels. Relief underlay (``hs_tif`` grayscale + optional ``dem_tif``
with ``cmap`` at ``dem_alpha``) is optional — figures degrade to plain maps.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: point_type -> (marker, color, size, zorder, label). GNSS/OPUS plots BEHIND
#: the 3DEP checkpoints; NVA above VVA (owner figure review, 2026-07-15).
def _heliport_marker():
    """FAA VFR-chart heliport symbol as a Path marker: 'H' inside a circle
    ring. Compound path: outer circle + reversed inner circle (annulus via
    winding) + a TextPath 'H' scaled into the ring."""
    from matplotlib.path import Path as _P
    from matplotlib.textpath import TextPath

    outer = _P.circle((0, 0), 1.0)
    inner = _P.circle((0, 0), 0.78)
    inner = _P(inner.vertices[::-1], inner.codes)  # reverse winding -> ring
    h = TextPath((0, 0), "H", size=1.0)
    b = h.get_extents()
    verts = ((h.vertices - ((b.x0 + b.x1) / 2.0, (b.y0 + b.y1) / 2.0))
             / max(b.width, b.height) * 1.15)
    return _P.make_compound_path(outer, inner, _P(verts, h.codes))


#: package-level marker key, convention-based where conventions exist
#: (primary sources verified 2026-08-13):
#: - helipad = H-in-circle: exact match to the FAA Aeronautical Chart
#:   Users' Guide heliport symbol (aeronav.faa.gov/user_guide, p. 23);
#: - runway end / displaced threshold = chevrons (matplotlib carets 6/7):
#:   simplification of the FAA CUG runway-construction bars + arrow/
#:   chevron stems (p. 124), which are runway-oriented and don't reduce
#:   to a scatter marker;
#: - NGS monument '+': near the USGS topo benchmark "x" (USGS
#:   Topographic Map Symbols, pubs.usgs.gov/gip/TopographicMapSymbols).
#:   The NGS web-map de facto scheme (circle = vertical, triangle =
#:   horizontal, square = combined; filled = order 1) is a possible
#:   future refinement requiring a control-type split per monument;
#: - GNSS star / 3DEP circle+square: no authority defines symbols for
#:   CORS or lidar checkpoints — house choices, kept distinct from the
#:   triangle/circle/square control conventions above.
#: Values: (marker, color, size, zorder, label).
POINT_STYLE = {
    "monument": ("+", "#111111", 30, 4, "NGS monument"),
    "gnss": ("*", "#0033A0", 90, 5, "GNSS/OPUS"),
    "VVA": ("s", "#E69F00", 45, 6, "3DEP VVA"),
    "NVA": ("o", "#C00000", 55, 7, "3DEP NVA"),
    "runway_end": (6, "#1B7837", 55, 6, "FAA runway end"),
    "displaced_threshold": (7, "#66A61E", 50, 6, "FAA displaced threshold"),
    "helipad": (_heliport_marker(), "#1B7837", 110, 6, "FAA helipad"),
}
#: legend order: the two 3DEP checkpoint classes adjacent, then GNSS, then
#: NGS, then the FAA runway classes.
LEGEND_ORDER = ("NVA", "VVA", "gnss", "monument", "runway_end",
                "displaced_threshold", "helipad")
#: dz map/histogram colormap, CENTRALIZED for easy revert (owner 2026-07-15):
#: RdYlBu puts RED = negative dz (product below control) — the same
#: red-means-down convention as the subsidence/rate maps. Revert to the old
#: look by setting this back to "RdBu_r".
DZ_CMAP = "RdYlBu"
#: standard symmetric color-limit tiers (m); empirical limits snap UP to a
#: tier so figures stay comparable across sites (owner spec 2026-07-04/16)
CLIM_TIERS = (0.10, 0.25, 0.50, 1.0, 2.5, 5.0)
_INK, _MUT = "#222222", "#777777"

_CPT_RAINBOW_CACHE = {}


def cpt_rainbow(reverse: bool = False):
    """The group's standard elevation ramp for color shaded relief.

    Canonical vendored copy of the GMT/cpt-city ``rainbow`` palette
    (``data/rainbow.cpt``, the same file imview bundles) so no repo needs an
    ``imview`` import — this helper replaces the duplicated
    try-imview/turbo fallbacks in downstream figure code. Render at ~0.4
    alpha over a multidirectional gray hillshade
    (:func:`groundcontrol.plot.hillshade`); house rule in env figures.md.
    """
    if reverse not in _CPT_RAINBOW_CACHE:
        from matplotlib.colors import LinearSegmentedColormap

        if reverse:  # exact mirror of the forward LUT
            _CPT_RAINBOW_CACHE[True] = cpt_rainbow(False).reversed()
            return _CPT_RAINBOW_CACHE[True]
        fn = Path(__file__).parent / "data" / "rainbow.cpt"
        z, rgb = [], []
        for line in fn.read_text().splitlines():
            p = line.split()
            if not p or p[0].startswith("#") or p[0] in ("B", "F", "N"):
                continue
            # each line: z1 r g b z2 r g b — keep the leading edge, plus the
            # trailing edge of the final segment
            z.append(float(p[0]))
            rgb.append((int(p[1]) / 255, int(p[2]) / 255, int(p[3]) / 255))
            last = (float(p[4]), (int(p[5]) / 255, int(p[6]) / 255,
                                  int(p[7]) / 255))
        z.append(last[0])
        rgb.append(last[1])
        z0, z1 = z[0], z[-1]
        pos = [(v - z0) / (z1 - z0) for v in z]
        _CPT_RAINBOW_CACHE[False] = LinearSegmentedColormap.from_list(
            "cpt_rainbow", list(zip(pos, rgb)))
    return _CPT_RAINBOW_CACHE[reverse]


def _point_azimuth(r) -> float:
    """Optional true azimuth (deg) for a point row: a ``true_az`` column
    when present, else the raw-JSON field (dispatcher-normalized frames
    keep source extras in ``raw``). NaN when unavailable."""
    import json as _json

    v = r.get("true_az") if hasattr(r, "get") else None
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        raw = r.get("raw") if hasattr(r, "get") else None
        if isinstance(raw, str):
            try:
                v = _json.loads(raw).get("true_az")
            except ValueError:
                v = None
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def point_context_gallery(points, layers, outdir, site_name, *,
                          half_m=60.0, tier_tag=None, interp="antialiased",
                          scale_len=25, id_col="id", class_col=None,
                          class_colors=None, subset_tag="station",
                          ncell=None, max_rows=12, sort=True, dpi=200):
    """Per-point context contact sheet: one row-cell of image panels per point.

    OPT-IN QA/QC figure (not part of the default reels): for each point, a
    horizontal strip of windows from ``layers`` — e.g. TrueOrtho RGB | lidar
    intensity | DSM color shaded relief — so the physical setting of every
    control point (roof mount, mast, pier, bare ground) is reviewable at a
    glance. Grew out of the MDV monument work and the Casa Grande cal-range
    contact sheets (sandbox drivers, 2026-07/08).

    Parameters
    ----------
    points : GeoDataFrame with point geometry, an ``id_col`` column, and a
        CRS. Coordinates are reprojected per layer when a raster's CRS
        differs.
    layers : sequence of ``(tag, path, kind)``; ``kind`` one of ``"rgb"``
        (bands 1-3, per-band 0.5-99.5% stretch), ``"gray"`` (band 1,
        0.5-99.5% stretch), ``"relief"`` (band 1 as :func:`cpt_rainbow` at
        0.4 alpha over a multidirectional hillshade — env figures.md house
        style). Panels render left-to-right in list order. ``path`` may be
        a LIST of paths — a fallback chain: the first source whose window
        holds >1% valid pixels renders (e.g. ``[ortho, web_basemap]`` so an
        ortho nodata hole falls back to fetched imagery); if every source
        is empty the last renders as-is (an honest blank, never invented).
    half_m : half-window in meters (60 -> 120 m context; ~15 with
        ``interp="nearest"`` for a native-pixel tier).
    tier_tag : filename tag; defaults to ``f"{2*half_m:g}m"``.
    class_col / class_colors : optional point-class column + {class: color}
        for the marker circles (default single crimson). The
        ``class_colors`` KEY ORDER is also the sort priority.
    ncell : point-cells per row (default 3 for 3+ layers else 4).
    max_rows : rows per sheet; longer subsets paginate into
        ``..._gallery_<tier>_pN.png`` pages (owner 2026-08-13: single
        very-long sheets do not review well).
    sort : order cells by (class, id) — class from the ``class_colors``
        key order, and the id's facility prefix groups airports/stations.
        Pass False to keep the caller's order.

    Returns the LIST of written paths (one per page;
    ``<outdir>/<site>_<subset_tag>_gallery_<tier>[_pN].png``).

    A layer that cannot be read at a point renders an "unavailable" panel
    rather than failing the sheet (points outside one product's footprint
    are expected at multi-product sites).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from rasterio.warp import transform as _rio_transform
    from rasterio.windows import Window

    from .plot import add_scalebar, hillshade

    def _window(src, x, y):
        # per-axis pixel sizes: transform.a (x) and .e (y) differ on
        # non-square-pixel rasters (Copilot review, PR #17)
        px = abs(src.transform.a)
        py = abs(src.transform.e)
        halfx = max(4, int(round(half_m / px)))
        halfy = max(4, int(round(half_m / py)))
        row, col = src.index(x, y)
        arr = src.read(window=Window(col - halfx, row - halfy, 2 * halfx,
                                     2 * halfy), boundless=True,
                       fill_value=src.nodata if src.nodata is not None else 0
                       ).astype("float64")
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        return arr, [x - halfx * px, x + halfx * px,
                     y - halfy * py, y + halfy * py], (px, py)

    def _valid_frac(arr):
        # fraction of pixels carrying signal: finite AND (any band) nonzero
        # — Byte RGB mosaics fill gaps with 0 and carry no nodata tag
        ok = np.isfinite(arr).all(axis=0) & (arr != 0).any(axis=0)
        return float(ok.mean()) if ok.size else 0.0

    def _panel(ax, dss, kind, x0, y0):
        for i, src in enumerate(dss):
            x, y = x0, y0
            if src.crs is not None and points.crs is not None \
                    and src.crs.to_wkt() != points.crs.to_wkt():
                xs, ys = _rio_transform(points.crs, src.crs, [x], [y])
                x, y = xs[0], ys[0]
            arr, ext, (px, py) = _window(src, x, y)
            if _valid_frac(arr) > 0.01 or i == len(dss) - 1:
                if i:
                    logger.info("fallback source %d used at (%.0f, %.0f)",
                                i, x0, y0)
                break
        if kind == "rgb":
            img = arr[:3]
            lo = np.nanpercentile(img, 0.5, axis=(1, 2))[:, None, None]
            hi = np.nanpercentile(img, 99.5, axis=(1, 2))[:, None, None]
            img = np.clip((img - lo) / np.where(hi > lo, hi - lo, 1), 0, 1)
            ax.imshow(np.moveaxis(np.nan_to_num(img), 0, -1), extent=ext,
                      zorder=0, interpolation=interp)
        elif kind == "gray":
            lo, hi = np.nanpercentile(arr[0], (0.5, 99.5))
            ax.imshow(arr[0], cmap="gray", vmin=lo, vmax=hi, extent=ext,
                      zorder=0, interpolation=interp)
        elif kind == "relief":
            b = arr[0]
            lo, hi = np.nanpercentile(b, (1, 99))
            if hi - lo < 2:  # flat water/lake ice: don't tint pure noise
                mid = 0.5 * (hi + lo)
                lo, hi = mid - 1, mid + 1
            ax.imshow(hillshade(b, dx=px, dy=py, multidirectional=True),
                      cmap="gray", vmin=0, vmax=1, extent=ext, zorder=0,
                      interpolation=interp)
            ax.imshow(b, cmap=cpt_rainbow(), vmin=lo, vmax=hi, alpha=0.4,
                      extent=ext, zorder=1, interpolation=interp)
            ax.text(0.03, 0.03, f"z {lo:.0f}..{hi:.0f} m",
                    transform=ax.transAxes, fontsize=6.5, color="white",
                    bbox=dict(fc="black", alpha=0.45, pad=1.5))
        else:
            raise ValueError(f"unknown layer kind {kind!r}")
        return x, y, ext

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tier = tier_tag or f"{2 * half_m:g}m"
    srcs = []  # built inside the try: a failed open must not leak the others
    try:
        for tag, p, kind in layers:
            chain = p if isinstance(p, (list, tuple)) else [p]
            srcs.append((tag, [rasterio.open(q) for q in chain], kind))
        from matplotlib.markers import MarkerStyle
        from matplotlib.transforms import Affine2D

        n = len(points)
        npanel = len(srcs)
        if ncell is None:
            ncell = 3 if npanel >= 3 else 4
        # intelligent order (owner 2026-08-13): class first (class_colors key
        # order = priority), then id — the facility prefix in the id groups
        # airports/stations naturally within each class
        groups = [points]
        if sort and id_col in points.columns:
            if class_col and class_col in points.columns:
                rank = {c: k for k, c in enumerate(class_colors or {})}
                ckey = points[class_col].map(
                    lambda c: rank.get(c, len(rank))).to_numpy()
                points = points.iloc[np.lexsort(
                    (points[id_col].astype(str).to_numpy(), ckey))]
                # classes NEVER share a page (owner 2026-08-13: surveyed
                # and estimated review as separate sheets)
                groups = [g for _, g in points.groupby(
                    points[class_col].map(lambda c: rank.get(c, len(rank))),
                    sort=True)]
            else:
                points = points.sort_values(id_col)
                groups = [points]
        per_page = max(1, max_rows) * ncell
        pages = [g.iloc[k:k + per_page] for g in groups
                 for k in range(0, len(g), per_page)] or [points]
        out_paths = []
        for pg, pts_pg in enumerate(pages, start=1):
            nrow = int(np.ceil(len(pts_pg) / ncell))
            pw = 2.7
            # ABSOLUTE title/footer margins: fractional top= on a tall sheet
            # reserved inches of whitespace under the title (owner 2026-08-13)
            fig_h = (pw + 0.42) * nrow + 0.85
            fig = plt.figure(figsize=(pw * npanel * ncell + 0.5, fig_h))
            wr = ([1] * npanel + [0.12]) * (ncell - 1) + [1] * npanel
            gs = fig.add_gridspec(nrow, (npanel + 1) * ncell - 1,
                                  width_ratios=wr, hspace=0.16, wspace=0.04)
            for i, (_, r) in enumerate(pts_pg.iterrows()):
                row_i, cell = divmod(i, ncell)
                cls = r[class_col] if class_col else None
                color = (class_colors or {}).get(cls, "#C00000")
                for j, (tag, chain, kind) in enumerate(srcs):
                    ax = fig.add_subplot(gs[row_i, cell * (npanel + 1) + j])
                    try:
                        x, y, ext = _panel(ax, chain, kind,
                                           r.geometry.x, r.geometry.y)
                        # locator = the package marker key's shape for this
                        # point_type, drawn as an outline so the imagery
                        # stays readable (unfilled markers take color=, not
                        # facecolors="none" — matplotlib warns otherwise)
                        ptype = str(r.get("point_type", "")) \
                            if "point_type" in r else ""
                        mk = POINT_STYLE.get(ptype, ("o",))[0]
                        if MarkerStyle(mk).is_filled():
                            mkw = dict(facecolors="none", edgecolors=color)
                        else:
                            mkw = dict(color=color)
                        # runway/threshold chevrons rotate to the published
                        # runway-end true alignment (E46), tip pointing
                        # inward along the runway; grid convergence is
                        # < ~2 deg at site scale — symbology, not survey
                        az = _point_azimuth(r)
                        if np.isfinite(az) and ptype in (
                                "runway_end", "displaced_threshold"):
                            mk = MarkerStyle(
                                mk, transform=Affine2D().rotate_deg(-az))
                        # helipad H-ring locator SURROUNDS the pad paint
                        s = 450 if ptype == "helipad" else 170
                        ax.scatter([x], [y], s=s, marker=mk,
                                   linewidths=2.0, zorder=5, **mkw)
                        ax.set_xlim(ext[0], ext[1])
                        ax.set_ylim(ext[2], ext[3])
                    except Exception as e:
                        ax.text(0.5, 0.5, f"{tag}\nunavailable", ha="center",
                                va="center", transform=ax.transAxes,
                                fontsize=8)
                        logger.warning("%s %s panel failed: %s",
                                       r[id_col], tag, e)
                    ax.set_aspect("equal")
                    ax.set_xticks([]), ax.set_yticks([])
                    if j == 0:
                        label = f"{r[id_col]}" + (f" · {cls}" if cls else "")
                        ax.set_title(label, fontsize=8.5, loc="left")
                    if j == npanel - 1:
                        add_scalebar(ax, length=scale_len,
                                     label=f"{scale_len} m")
            tags = " | ".join(t for t, _, _ in srcs)
            page_cls = ""
            if class_col and class_col in pts_pg.columns \
                    and pts_pg[class_col].nunique() == 1:
                page_cls = f" — {pts_pg[class_col].iloc[0].upper()}"
            page_note = (f"{page_cls} — page {pg}/{len(pages)}"
                         if len(pages) > 1 else page_cls)
            fig.suptitle(
                f"{site_name} {subset_tag} points — {tags} ({2*half_m:.0f} m "
                f"windows{', native pixels' if interp == 'nearest' else ''})"
                f"{page_note}",
                fontsize=12, y=1.0 - 0.12 / fig_h)
            fig.subplots_adjust(left=0.01, right=0.995,
                                top=1.0 - 0.52 / fig_h, bottom=0.18 / fig_h)
            suffix = f"_p{pg}" if len(pages) > 1 else ""
            fp = outdir / f"{site_name}_{subset_tag}_gallery_{tier}{suffix}.png"
            fig.savefig(fp, dpi=dpi)
            plt.close(fig)
            out_paths.append(fp)
    finally:
        for _, chain, _ in srcs:
            for src in chain:
                src.close()
    logger.info("wrote %d page(s), %d points: %s", len(out_paths), n,
                [p.name for p in out_paths])
    return out_paths


def snap_clim(values, k=3.0, tiers=CLIM_TIERS):
    """Empirical symmetric color limit: ``|median| + k*NMAD`` of the finite
    values, snapped UP to the next standard tier — data-driven (owner
    2026-07-16: no hardcoded limits) yet comparable across figures. The
    typical spread renders mid-ramp, never saturated."""
    v = np.asarray(values, dtype="float64")
    v = v[np.isfinite(v)]
    if not v.size:
        return tiers[0]
    med = float(np.median(v))
    nmad = 1.4826 * float(np.median(np.abs(v - med)))
    need = abs(med) + k * max(nmad, 1e-6)
    for t in tiers:
        if need <= t:
            return t
    return tiers[-1]


def _datum_tag(crs):
    """Short datum note for height labels, e.g. 'NAD83(2011) ellipsoid'."""
    import pyproj
    try:
        c = pyproj.CRS.from_user_input(crs)
        if c.is_compound:  # orthometric target: name the vertical member,
            return c.sub_crs_list[1].name  # e.g. "NAVD88 height" — not "ellipsoid"
        return f"{(c.geodetic_crs or c).name} ellipsoid"
    except Exception:  # pragma: no cover - label fallback only
        return "ellipsoid"
_FACETS = ("posSource", "vertSource", "vertOrder")


def _raw_field(series, key):
    def get(r):
        if isinstance(r, str):
            try:
                r = json.loads(r)
            except Exception:
                return None
        if not isinstance(r, dict):
            return None  # missing raw arrives as None OR float NaN (pandas>=3)
        v = r.get(key)
        v = (v or "").strip() if isinstance(v, str) else v
        return v if v else None
    return series.apply(get)


def _relief(ax, dem_tif, hs_tif, cmap, dem_alpha, fig):
    import rasterio
    ext = None
    if hs_tif is not None:
        with rasterio.open(hs_tif) as src:
            hs = src.read(1, masked=True).astype("f4").filled(np.nan)
            bb = src.bounds
        ext = [bb.left, bb.right, bb.bottom, bb.top]
        ax.imshow(hs, cmap="gray", vmin=1, vmax=255, extent=ext,
                  interpolation="antialiased", interpolation_stage="rgba")
    if dem_tif is not None and cmap is not None:
        with rasterio.open(dem_tif) as src:
            z = src.read(1, masked=True).filled(np.nan)
            bb = src.bounds
            dem_crs = src.crs
        ext = [bb.left, bb.right, bb.bottom, bb.top]
        im = ax.imshow(z, cmap=cmap, alpha=dem_alpha, extent=ext,
                       vmin=np.nanpercentile(z, 2),
                       vmax=np.nanpercentile(z, 98), interpolation="antialiased", interpolation_stage="rgba")
        if fig is not None:
            cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
            cb.set_label(f"Elevation (m, {_datum_tag(dem_crs)})",
                         fontsize=9, color=_INK)
            cb.ax.tick_params(labelsize=8, colors=_MUT)
    return ext


def _finish_map(ax, aoi_gdf, clip_to_aoi=True):
    """Ticks off + scalebar. With ``clip_to_aoi`` the axes are limited to the
    AOI bounds and the dashed outline is dropped (redundant when the map IS
    the AOI); pass False to keep the outline on un-clipped maps."""
    from .plot import add_scalebar
    if aoi_gdf is not None:
        if clip_to_aoi:
            b = aoi_gdf.total_bounds
            ax.set_xlim(b[0], b[2])
            ax.set_ylim(b[1], b[3])
        else:
            aoi_gdf.boundary.plot(ax=ax, color=_INK, lw=1.0, ls="--", alpha=0.45)
    ax.set_xticks([])
    ax.set_yticks([])
    # equal aspect is the map contract (env figures.md) and silences the
    # matplotlib-scalebar unequal-aspect warning (#23)
    ax.set_aspect("equal")
    add_scalebar(ax)


def _aspect_panel_w(aoi_gdf, map_h, lo=0.5, hi=1.5):
    """Width (inches) for an equal-aspect map of ``aoi_gdf`` drawn ``map_h``
    inches tall, so the map fills its axes instead of letterboxing (the source
    of the tall-narrow-AOI whitespace). Clamped to ``[lo, hi]``*``map_h`` so
    extreme aspect ratios stay sane; ``None`` or degenerate (zero-width or
    zero-height bounds) aoi -> square panel."""
    if aoi_gdf is None:
        return map_h
    b = aoi_gdf.total_bounds
    dx, dy = float(b[2] - b[0]), float(b[3] - b[1])
    asp = dy / dx if (dx > 0 and dy > 0) else 1.0
    return float(np.clip(map_h / asp, lo * map_h, hi * map_h))


def standard_control_figures(control, aoi, outdir, site_name, *,
                             dem_tif=None, hs_tif=None, cmap=None,
                             dem_alpha=0.4, midas_frame="IGS14",
                             buffer_km=60.0, clip_to_aoi=True,
                             label_points=True, dpi=200):
    """Write the default control figure bundle for a site; returns paths.

    ``label_points`` (owner request 2026-08-13, the NGS-map convention):
    sparse, named classes get text labels — NGL/CORS station ids per point,
    FAA points one label per AIRPORT (grouped by the id prefix; per-runway-
    end labels would be unreadable). Dense classes (3DEP checkpoints, NGS
    monuments) are never labeled.
    """
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = []
    aoi_gdf = gpd.read_file(aoi) if isinstance(aoi, (str, Path)) else aoi

    # figure CRS: the DEM's if given, else the AOI's UTM estimate
    import rasterio
    if dem_tif is not None:
        with rasterio.open(dem_tif) as src:
            fig_crs = src.crs
    else:
        fig_crs = aoi_gdf.estimate_utm_crs()
    ctl = control.to_crs(fig_crs)
    aoi_p = aoi_gdf.to_crs(fig_crs)

    # ---- 1. control map ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(10.5, 10))
    _relief(ax, dem_tif, hs_tif, cmap, dem_alpha, fig)
    from matplotlib.markers import MarkerStyle
    from matplotlib.transforms import Affine2D

    by_type = {}
    for ptype, (mk, col, sz, zo, lab) in POINT_STYLE.items():
        sub = ctl[ctl.point_type == ptype]
        if not len(sub):
            continue
        lw = 1.1 if mk == "+" else 0.5
        ec = "white" if mk != "+" else col
        if ptype in ("runway_end", "displaced_threshold"):
            # chevrons rotate to the published runway-end true alignment
            # (tips point inward along the runway; owner 2026-08-13)
            for _, rr in sub.iterrows():
                az = _point_azimuth(rr)
                m = MarkerStyle(mk, transform=Affine2D().rotate_deg(-az)) \
                    if np.isfinite(az) else mk
                ax.scatter([rr.geometry.x], [rr.geometry.y], marker=m, s=sz,
                           c=col, linewidths=lw, edgecolors=ec, zorder=zo)
        else:
            ax.scatter(sub.geometry.x, sub.geometry.y, marker=mk, s=sz,
                       c=col, linewidths=lw, edgecolors=ec, zorder=zo)
        by_type[ptype] = Line2D([], [], marker=mk, ls="", color=col, ms=9,
                                label=f"{lab} (n={len(sub)})")
    if label_points and "source" in ctl.columns:
        import matplotlib.patheffects as _pe
        halo = [_pe.withStroke(linewidth=2.2, foreground="white")]
        # GNSS/CORS labels place FIRST and unconditionally; FAA airport
        # labels yield to them (owner 2026-08-13): dodge below on a close
        # approach, drop entirely on a collision
        anchors = []
        for _, r in ctl[ctl["source"] == "ngl"].iterrows():
            ax.annotate(str(r["id"]), (r.geometry.x, r.geometry.y),
                        xytext=(5, 4), textcoords="offset points",
                        fontsize=7, fontweight="bold", color="#0033A0",
                        path_effects=halo, zorder=8)
            anchors.append((float(r.geometry.x), float(r.geometry.y)))
        f = ctl[ctl["source"] == "faa"]
        if len(f):
            b = ctl.total_bounds
            dmin = 0.03 * max(b[2] - b[0], b[3] - b[1])
            anch = np.asarray(anchors, dtype="float64") \
                if anchors else np.empty((0, 2))
            for name, sub in f.groupby(f["id"].astype(str).str.split("_").str[0]):
                cx = float(sub.geometry.x.mean())
                cy = float(sub.geometry.y.mean())
                d = np.min(np.hypot(anch[:, 0] - cx, anch[:, 1] - cy)) \
                    if len(anch) else np.inf
                if d < 0.5 * dmin:
                    continue  # too close to a placed label: drop, don't clash
                dy, va = ((7, "bottom") if d >= dmin else (-9, "top"))
                ax.annotate(name, (cx, cy), xytext=(0, dy),
                            textcoords="offset points", ha="center", va=va,
                            fontsize=8, fontweight="bold", color="#1B7837",
                            path_effects=halo, zorder=8)
                # placed FAA labels become anchors too, so FAA labels also
                # dodge each other (dense-cluster clashes, owner 2026-08-13)
                anch = np.vstack([anch, [[cx, cy]]]) if len(anch) \
                    else np.array([[cx, cy]], dtype="float64")
    handles = [by_type[p] for p in LEGEND_ORDER if p in by_type]
    if not clip_to_aoi:
        handles.append(Line2D([], [], ls="--", color=_INK, alpha=0.45,
                              label="AOI"))
    ax.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.92)
    _finish_map(ax, aoi_p, clip_to_aoi)
    ax.set_title(f"{site_name} — control points ({len(ctl)} usable)",
                 fontsize=11, color=_INK)
    fig.tight_layout()
    fp = outdir / f"{site_name}_control_map.png"
    fig.savefig(fp, dpi=dpi)
    plt.close(fig)
    out.append(fp)

    # ---- 2. NGS monument-type facets ---------------------------------------
    mon = ctl[ctl.point_type == "monument"]
    if len(mon):
        fig, axes = plt.subplots(1, len(_FACETS), figsize=(5.6 * len(_FACETS), 6),
                                 sharex=True, sharey=True)
        cyc = ["#0033A0", "#C00000", "#005F20", "#8B008B", "#8B4E00",
               "#111111"]
        for ax, key in zip(np.atleast_1d(axes), _FACETS):
            _relief(ax, dem_tif, hs_tif, None, dem_alpha, None)
            vals = _raw_field(mon["raw"], key).fillna("(none)")
            top = vals.value_counts().index.tolist()[:5]
            vals = vals.where(vals.isin(top), "other")
            for i, v in enumerate(pd.unique(vals)):
                s = mon[vals == v]
                ax.scatter(s.geometry.x, s.geometry.y, s=16, marker="o",
                           c=cyc[i % 6], edgecolors="white", linewidths=0.4,
                           zorder=5, label=f"{v} ({len(s)})")
            _finish_map(ax, aoi_p, clip_to_aoi)
            ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9)
            ax.set_title(f"NGS monuments by {key}", fontsize=10, color=_INK)
        fig.suptitle(f"{site_name} — NGS monument datasheet attributes "
                     f"(n={len(mon)})", fontsize=11.5, color=_INK)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fp = outdir / f"{site_name}_monument_types.png"
        fig.savefig(fp, dpi=dpi)
        plt.close(fig)
        out.append(fp)

    # ---- 3+4. MIDAS motion figures ------------------------------------------
    try:
        from .plot import plot_velocity_vectors
        from .sources.ngl import read_midas
        st = read_midas(midas_frame)
        b = aoi_gdf.to_crs(4326).total_bounds
        st = st[st.lon.between(b[0] - 3, b[2] + 3)
                & st.lat.between(b[1] - 3, b[3] + 3)]
        for cbv, tag in ((False, "horiz"), (True, "vertical")):
            fp = outdir / f"{site_name}_midas_velocity_{tag}.png"
            plot_velocity_vectors(
                st, aoi=aoi_gdf, buffer_km=buffer_km, out_fn=str(fp),
                color_by_vertical=cbv, hs_tif=hs_tif,
                title=f"{site_name} — MIDAS ({midas_frame}) "
                      f"{'vertical-colored ' if cbv else ''}velocity field")
            out.append(fp)
    except Exception as exc:  # network etc. — the map figures still ship
        logger.warning("MIDAS velocity figures skipped: %s", exc)

    logger.info("standard control figures: %s", [p.name for p in out])
    return out


def _nmad(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def _ngs_gate(v, mult):
    """NMAD outlier gate for the NGS histogram panel. Skipped when NMAD == 0
    (>=50% identical, quantized residuals) — a floor would keep only the
    majority value and annotate fake-perfect stats; mirrors
    accuracy.error_report."""
    med0, nm0 = np.median(v), _nmad(v)
    return v[np.abs(v - med0) < mult * nm0] if nm0 > 0 else v


def validation_dz_figures(sampled, aoi, outdir, site_name, *, products=("DSM", "DTM"),
                          hs_tif=None, point_lim=None, vendor_lim=None, wide_lim=None,
                          ngs_nmad_gate=3.0, dpi=200):
    # hs_tif: a single path, or a {product: path} dict for PRODUCT-MATCHED
    # backgrounds (DTM diffs belong on the DTM hillshade — David, 2026-07-15).
    """Product-vs-control vertical-offset validation figures (standard bundle
    item 5; requested by David 2026-07-15 after the SF run).

    One figure per product: (a) map of control points over shaded relief
    colored by ``dh_<product>_before`` (product - control, RdBu_r); (b)
    histograms for the survey-grade segments (vendor NVA/VVA, GNSS/OPUS);
    (c) histogram for NGS monuments after a ``ngs_nmad_gate``-NMAD filter.
    Segment rules: NVA validates DSM and DTM; VVA validates DTM only;
    NGS/OPUS shown for both as datum-sanity context. Median/NMAD/n
    annotated per segment.

    Limits (``point_lim``/``vendor_lim``/``wide_lim``) default to
    EMPIRICAL, tier-snapped values from the plotted dz (:func:`snap_clim`)
    — issue #23: the former lidar-tuned constants (0.25/0.6/4.0 m)
    saturated the map and clipped the histograms into piles at the edges
    on photogrammetric DEMs (NMAD ~1-8 m). Pass explicit values to pin
    comparable limits across runs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = []
    # accept a path or a GeoDataFrame aoi (siblings do the same) and reproject
    # to the sampled frame, so the map clip + aspect use the plotted CRS
    if aoi is not None:
        import geopandas as gpd
        if isinstance(aoi, (str, Path)):
            aoi = gpd.read_file(aoi)
        aoi = aoi.to_crs(sampled.crs)
    seg_defs = {  # label -> (mask fn, style key, in DSM, in DTM)
        "3DEP NVA": (lambda d: (d["source"] == "3dep") & (d["point_type"] == "NVA"),
                     "NVA", True, True),
        "3DEP VVA": (lambda d: (d["source"] == "3dep") & (d["point_type"] == "VVA"),
                     "VVA", False, True),
        "GNSS/OPUS": (lambda d: d["source"] == "opus", "gnss", True, True),
        "NGS monument": (lambda d: d["source"] == "ngs", "monument", True, True),
    }
    for prod in products:
        col = f"dh_{prod}_before"
        if col not in sampled.columns:
            logger.warning("validation_dz: no column %s, skipping %s", col, prod)
            continue
        # aspect-aware: size the map column to the AOI so the equal-aspect map
        # fills it (no tall-narrow letterboxing); +colorbar allowance in-column
        map_h = 6.2
        mcol = _aspect_panel_w(aoi, map_h - 0.7) + 0.9
        hist_w = 4.6
        fig, axes = plt.subplots(
            1, 3, figsize=(mcol + 2 * hist_w, map_h),
            gridspec_kw=dict(width_ratios=[mcol, hist_w, hist_w]))
        hs_prod = hs_tif.get(prod) if isinstance(hs_tif, dict) else hs_tif
        _relief(axes[0], None, hs_prod, None, 0.0, None)
        use = sampled[np.isfinite(sampled[col])]
        pl = point_lim if point_lim is not None else snap_clim(use[col], k=3.0)
        sc = axes[0].scatter(use.geometry.x, use.geometry.y, c=use[col],
                             cmap=DZ_CMAP, vmin=-pl, vmax=pl,
                             s=34, edgecolors="#333333", linewidths=0.5, zorder=5)
        cb = fig.colorbar(sc, ax=axes[0], shrink=0.75, pad=0.02, extend="both")
        cb.set_label(f"dz = {prod} − control (m)", fontsize=9, color=_INK)
        cb.ax.tick_params(labelsize=8, colors=_MUT)
        _finish_map(axes[0], aoi)
        axes[0].set_title(f"{site_name} {prod} − control  (n={len(use)})",
                          fontsize=11, color=_INK)

        for ax, labels, lim_over in (
                (axes[1], [lbl for lbl, s in seg_defs.items()
                           if s[2 if prod == "DSM" else 3] and lbl != "NGS monument"],
                 vendor_lim),
                (axes[2], ["NGS monument"], wide_lim)):
            seg_vals = {}
            for lab in labels:
                maskfn, style, *_ = seg_defs[lab]
                v = use.loc[maskfn(use), col].to_numpy(float)
                v = v[np.isfinite(v)]
                if lab == "NGS monument" and len(v):
                    v = _ngs_gate(v, ngs_nmad_gate)
                if len(v):
                    seg_vals[lab] = v
            # empirical tier-snapped panel limit unless overridden (#23);
            # never tighter than the map so the panels stay comparable
            lim = lim_over if lim_over is not None else (
                max(snap_clim(np.concatenate(list(seg_vals.values())), k=3.0),
                    pl) if seg_vals else pl)
            txt = []
            for lab, v in seg_vals.items():
                color = POINT_STYLE[seg_defs[lab][1]][1]
                ax.hist(np.clip(v, -lim, lim), bins=41, range=(-lim, lim),
                        histtype="stepfilled", alpha=0.45, color=color,
                        edgecolor=color, label=lab)
                txt.append(f"{lab}: med {np.median(v):+.3f}, "
                           f"NMAD {_nmad(v):.3f}, n={len(v)}")
            ax.axvline(0, color=_INK, lw=0.8)
            ax.set_xlim(-lim, lim)
            ax.set_xlabel(f"dz = {prod} − control (m)", fontsize=9, color=_INK)
            if txt:  # legend/stats only when something plotted (#23: an
                # NGS-only site rendered a bare axes + empty legend box)
                ax.legend(fontsize=8, loc="upper right")
                ax.text(0.02, 0.98, "\n".join(txt), transform=ax.transAxes,
                        fontsize=8, va="top", color=_INK,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  alpha=0.85))
            else:
                ax.text(0.5, 0.5, "no matching checkpoints in AOI",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=9, color=_MUT)
            ax.tick_params(labelsize=8, colors=_MUT)
            ax.grid(alpha=0.25, lw=0.5)
        axes[1].set_title("survey-grade segments", fontsize=10, color=_INK)
        axes[2].set_title(f"NGS monuments ({ngs_nmad_gate:.0f}-NMAD filtered)",
                          fontsize=10, color=_INK)
        fp = outdir / f"{site_name}_validation_dz_{prod}.png"
        fig.tight_layout()
        fig.savefig(fp, dpi=dpi)
        plt.close(fig)
        out.append(fp)
        logger.info("wrote %s", fp)
    return out


#: family key -> (title, [(subclass label, row mask fn, point_type style key
#: or hex color, marker[, products])]). Optional 5th element restricts the
#: subclass to those products: VVA canopy checkpoints validate the DTM only —
#: an expected DSM bias is not an error, so VVA is EXCLUDED from the DSM
#: figure (owner 2026-07-15) rather than shown as a huge tail.
DZ_FAMILIES = {
    "3dep": ("3DEP CHECKPOINTS", [
        ("NVA", lambda d: (d["source"] == "3dep") & (d["point_type"] == "NVA"),
         "NVA", "o"),
        ("VVA", lambda d: (d["source"] == "3dep") & (d["point_type"] == "VVA"),
         "VVA", "s", ("DTM",)),
    ]),
    "gnss": ("GNSS/OPUS", [
        ("GNSS/OPUS", lambda d: d["source"] == "opus", "gnss", "o"),
    ]),
    "ngs_best": ("NGS MONUMENTS (best)", [
        ("NGS best", None, "monument", "o"),   # mask injected from ngs_best
    ]),
    # FAA NASR runway control, split by published coordinate provenance
    # (raw['pos_class'] from sources/faa.py): the surveyed class is
    # AC 150/5300-18C survey-grade; OWNER/FAA-EST/ADO positions are
    # meters-to-tens-of-meters (LV A/B 2026-08-13: NMAD 0.019 vs 2.78 m)
    # short panel labels: long ones collide on narrow-aspect AOIs (SF);
    # surveyed = 3RD PARTY SURVEY/NGS/MILITARY/ARPTS CONTRACTOR,
    # estimated = OWNER/FAA-EST IMAGERY/ADO/OE-AAA/blank
    "faa": ("FAA RUNWAY CONTROL (by position source)", [
        ("Surveyed",
         lambda d: (d["source"] == "faa")
         & (_raw_field(d["raw"], "pos_class") == "surveyed"),
         "runway_end", "^"),
        ("Estimated",
         lambda d: (d["source"] == "faa")
         & (_raw_field(d["raw"], "pos_class") == "estimated"),
         "#8C6BB1", "v"),
    ]),
}


def default_ngs_best(sampled):
    """Initial empirical 'best NGS' tier (Casa Grande assessment, 2026-07-15):
    ADJUSTED horizontal AND (published NAD 83(2011) realization OR GPS-grade
    vertical). Large-AOI dz vs DTM: median -0.05 m, NMAD ~0.12, ~1-3% gross
    outliers -- near GNSS quality; every looser tier degrades sharply
    (NAD83(1992) 0.27 NMAD, NAD83(1986) 0.63, SCALED horizontal 0.92/36%
    gross). Expect iteration -- pass a custom mask/callable to
    family_dz_figures(ngs_best=...) as the definition evolves.
    """
    pos = _raw_field(sampled["raw"], "posSource") == "ADJUSTED"
    vert = _raw_field(sampled["raw"], "vertSource").isin(
        ["GPS OBS", "ADJUSTED", "READJUSTED"])
    f2011 = sampled["ref_frame"].astype("string").str.replace(" ", "") == "NAD83(2011)"
    return (sampled["source"] == "ngs") & pos & (f2011 | vert)


def family_dz_figures(sampled, aoi, outdir, site_name, *, products=("DSM", "DTM"),
                      hs_tif=None, families=("3dep", "gnss", "ngs_best"),
                      ngs_best=None, extra_families=None, lims=None,
                      overlays=None, dpi=200):
    """Per-family dz figures: one map PER SUBCLASS (co-located NVA/VVA pairs
    overplot on a shared map — owner review 2026-07-15) + ONE combined
    histogram with per-subclass med/NMAD/n stats (stats lines are colored per
    class and double as the legend), per (family, product).

    A subclass with a products restriction (5th tuple element, e.g. VVA ->
    ``("DTM",)``) is dropped from other products' figures — an EXPECTED bias
    (canopy vs DSM) is not an error to display. Colors ride on
    :data:`DZ_CMAP` (centralized; RED = product below control).

    ``extra_families`` merges site-specific entries over
    :data:`DZ_FAMILIES`; subclass style is a POINT_STYLE key or hex color.
    ``hs_tif``: path or {product: path} product-matched hillshade.
    ``lims``: {family: (map_clim, hist_lim)} in meters.
    ``overlays``: GeoDataFrame (any CRS) drawn as dashed outlines on every
    map — e.g. per-lidar-project footprints so seams are attributable.
    """
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    aoi_gdf = gpd.read_file(aoi) if isinstance(aoi, (str, Path)) else aoi
    if aoi_gdf is not None:
        aoi_gdf = aoi_gdf.to_crs(sampled.crs)
    if overlays is not None:
        overlays = overlays.to_crs(sampled.crs)
    lims = lims or {}   # {family: (map_lim, hist_lim)} override; else empirical
    datum = _datum_tag(sampled.crs)
    catalog = {**DZ_FAMILIES, **(extra_families or {})}
    out = []
    for fam in families:
        title, subclasses = catalog[fam]
        if fam == "ngs_best":
            mask = ngs_best(sampled) if callable(ngs_best) else ngs_best
            if mask is None:
                mask = default_ngs_best(sampled)
            # fillna BEFORE the bool cast: default_ngs_best yields Kleene-NA
            # for rows with missing raw/ref_frame fields — exclude, don't
            # crash. A user Series must share the frame's index: constructing
            # with index= would label-ALIGN it (a positional mask built on a
            # RangeIndex against a .loc-filtered frame silently empties).
            if isinstance(mask, pd.Series):
                if not mask.index.equals(sampled.index):
                    raise ValueError(
                        "ngs_best mask index does not match the sampled frame "
                        "— align it (or pass a plain boolean array)")
                m = mask
            else:
                m = pd.Series(np.asarray(mask), index=sampled.index)
            mask = pd.Series(m.astype("boolean").fillna(False)
                             .to_numpy(dtype=bool), index=sampled.index)
            subclasses = [("NGS best", lambda d, m=mask: m, "monument", "o")]
        for prod in products:
            col = f"dh_{prod}_before"
            if col not in sampled.columns:
                logger.warning("family_dz: no column %s, skipping %s/%s",
                               col, fam, prod)
                continue
            subs = [s for s in subclasses
                    if len(s) < 5 or s[4] is None or prod in s[4]]
            if not subs:
                continue
            # empirical, tier-snapped color/hist limits from THIS figure's
            # own dz values (owner 2026-07-16: no hardcoded limits) — the
            # typical spread renders mid-ramp, not saturated
            if fam in lims:
                map_lim, hist_lim = lims[fam]
            else:
                fig_v = np.concatenate([
                    sampled.loc[pd.Series(s[1](sampled)).fillna(False)
                                .to_numpy(dtype=bool), col]
                    .to_numpy(dtype="float64") for s in subs]) if subs else []
                map_lim = snap_clim(fig_v, k=3.0)
                hist_lim = max(snap_clim(fig_v, k=6.0), map_lim)
            n_sub = len(subs)
            # aspect-aware map columns (fill the axes; kill the map->colorbar
            # gap on tall-narrow AOIs); the shared-colorbar allowance rides on
            # the map columns in BOTH figsize and width_ratios so the inch
            # widths stay literal
            map_h = 6.4
            mcol = _aspect_panel_w(aoi_gdf, map_h - 0.7) + 0.9 / n_sub
            hist_w = 5.0
            fig, axes = plt.subplots(
                1, n_sub + 1,
                figsize=(mcol * n_sub + hist_w, map_h),
                gridspec_kw=dict(width_ratios=[mcol] * n_sub + [hist_w]))
            axh = axes[-1]
            hs_prod = hs_tif.get(prod) if isinstance(hs_tif, dict) else hs_tif
            sc, stats_lines, n_gap = None, [], 0
            for axm, sub in zip(axes[:-1], subs):
                lab, maskfn, style, mk = sub[:4]
                _relief(axm, None, hs_prod, None, 0.0, None)
                if overlays is not None:
                    overlays.boundary.plot(ax=axm, color=_INK, lw=0.9, ls="--",
                                           alpha=0.55, zorder=4)
                # nullable dtypes (string == comparisons) yield NA: NA -> False
                m = pd.Series(maskfn(sampled)).fillna(False).to_numpy(dtype=bool)
                seg = sampled[m]
                v = seg[col].to_numpy(dtype="float64")
                fin = np.isfinite(v)
                n_gap += int((~fin).sum())
                color = POINT_STYLE[style][1] if style in POINT_STYLE else style
                # NEUTRAL point outlines — class colors clash with the dz ramp
                # (owner 2026-07-16); subclass identity = per-map panel title
                sc = axm.scatter(seg.geometry.x[fin], seg.geometry.y[fin],
                                 c=v[fin], cmap=DZ_CMAP, vmin=-map_lim,
                                 vmax=map_lim, s=52, marker=mk,
                                 edgecolors="#404040", linewidths=0.8, zorder=5)
                _finish_map(axm, aoi_gdf)
                axm.set_title(f"{lab} (n={int(fin.sum())})", fontsize=10.5,
                              color=_INK)
                if fin.any():
                    vv = v[fin]
                    axh.hist(np.clip(vv, -hist_lim, hist_lim), bins=41,
                             range=(-hist_lim, hist_lim), histtype="stepfilled",
                             alpha=0.45, color=color, edgecolor=color)
                    # dual-track stats (owner 2026-07-16): robust pair, then
                    # the ASPRS-Ed.2 parametric set after a 3*NMAD gate
                    from .accuracy import error_report
                    er = error_report(vv)
                    stats_lines.append(
                        (f"{lab}: med {er['median']:+.3f}, "
                         f"NMAD {er['nmad']:.3f}, n={er['n']}", color))
                    stats_lines.append(
                        (f"  mean {er['mean']:+.3f}, σ {er['std']:.3f}, "
                         f"RMSE {er['rmse']:.3f}, LE90 {er['le90']:.3f}"
                         + (f" ({er['n_outliers']} out)" if er["n_outliers"]
                            else ""), color))
            if sc is not None:
                cb = fig.colorbar(sc, ax=list(axes[:-1]), shrink=0.75,
                                  pad=0.015, extend="both")
                cb.set_label(f"dz = {prod} \u2212 control "
                             f"(m, {datum})\n[\u00b1{map_lim:g} m tier]",
                             fontsize=9, color=_INK)
                cb.ax.tick_params(labelsize=8, colors=_MUT)
            axh.axvline(0, color=_INK, lw=0.8)
            axh.set_xlim(-hist_lim, hist_lim)
            axh.set_xlabel(f"dz = {prod} \u2212 control (m)", fontsize=9,
                           color=_INK)
            # colored stats lines double as the legend (no overlap issues)
            for i, (line, color) in enumerate(stats_lines):
                axh.text(0.02, 0.98 - 0.05 * i, line, transform=axh.transAxes,
                         fontsize=8.5, va="top", color=color,
                         fontweight="bold" if not line.startswith("  ") else
                         "normal",
                         bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                   ec="none", alpha=0.8))
            if "xform_acc_m" in sampled.columns:
                b = np.nanmedian(sampled["xform_acc_m"].to_numpy(dtype="float64"))
                if np.isfinite(b):
                    axh.text(0.02, 0.02, f"stated 3D transform budget ±{b:g} m",
                             transform=axh.transAxes, fontsize=8, color=_MUT,
                             va="bottom",
                             bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                       ec="none", alpha=0.8))
            axh.tick_params(labelsize=8, colors=_MUT)
            axh.grid(alpha=0.25, lw=0.5)
            gap = f"; {n_gap} unsampled (nodata/gap)" if n_gap else ""
            fig.suptitle(f"{site_name} {prod} \u2212 control \u2014 {title}{gap}",
                         fontsize=11.5, color=_INK)
            fp = outdir / f"{site_name}_dz_{fam}_{prod}.png"
            fig.savefig(fp, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            out.append(fp)
            logger.info("wrote %s", fp)
    return out
