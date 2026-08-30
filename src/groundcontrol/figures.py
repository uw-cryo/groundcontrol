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
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: point_type -> (marker, color, size, zorder, label). GNSS plots BEHIND
#: the 3DEP checkpoints; NVA above VVA (owner figure review, 2026-07-15).
#: package-level marker key, convention-based where conventions exist
#: (primary sources verified 2026-08-13):
#: - helipad = thick X (owner 2026-08-31: ONE symbol everywhere — the
#:   FAA-chart H-in-circle read badly at map scale and could not carry a
#:   dz ramp fill);
#: - runway end / displaced threshold = FILLED triangles up/down (owner
#:   2026-08-31: line-only carets could not carry a dz fill + outline),
#:   rotated to the published runway alignment where drawn on maps —
#:   simplification of the FAA CUG runway-construction bars + arrow/
#:   chevron stems (p. 124), which don't reduce to a scatter marker;
#: - NGS monument 'P' (SOLID plus, owner 2026-08-31: the line-only '+'
#:   could not carry a dz fill + outline): near the USGS topo benchmark
#:   "x" (USGS
#:   Topographic Map Symbols, pubs.usgs.gov/gip/TopographicMapSymbols).
#:   The NGS web-map de facto scheme (circle = vertical, triangle =
#:   horizontal, square = combined; filled = order 1) is a possible
#:   future refinement requiring a control-type split per monument;
#: - GNSS star / 3DEP circle+square: no authority defines symbols for
#:   CORS or lidar checkpoints — house choices, kept distinct from the
#:   triangle/circle/square control conventions above. One star family
#:   split by color along the PER-ROW occupation class (owner taxonomy,
#:   2026-08-22; sources.ngl.occupation_class — each station's own record
#:   earns its class): a dark-to-light blue ramp — deep = continuous
#:   (permanent hardware, photo-ID candidates), mid = semi-continuous (the
#:   declared 0.2-0.6 density buffer band), sky = campaign (episodic;
#:   post-occupation nothing visible but the monument) — plus gray = the
#:   pre-split "gnss" label carried by products written before the split,
#:   kept so they still render. DELIBERATE map-vs-stats difference for
#:   legacy OPUS rows (audit round 4): the map keeps them gray (the reader
#:   sees the parquet predates the split) while the stats table folds them
#:   into campaign (OPUS) so applies stays continuous with main.
#: Values: (marker, color, size, zorder, label).
POINT_STYLE = {
    "monument": ("P", "#111111", 30, 4, "NGS monument"),
    # occupation-class ramp: three lightness steps of one blue family
    # (owner iterations 2026-08-30: three close blues failed, then white
    # failed on white backgrounds) — near-black navy / mid blue / light
    # blue; the light fill takes a dark edge on maps (_edge_for) and a
    # slightly darkened ink for text/histograms (class_ink).
    "gnss_cont": ("*", "#08306B", 90, 5, "GNSS continuous"),
    "gnss_semicont": ("*", "#3E8EC4", 90, 5, "GNSS semi-continuous"),
    "gnss_campaign": ("*", "#A6CEE3", 90, 5, "GNSS campaign"),
    "gnss": ("*", "#888888", 90, 5, "GNSS (pre-split)"),
    "VVA": ("s", "#E69F00", 45, 6, "3DEP VVA"),
    "NVA": ("o", "#C00000", 55, 7, "3DEP NVA"),
    "runway_end": ("^", "#1B7837", 55, 6, "FAA runway end"),
    "displaced_threshold": ("v", "#66A61E", 50, 6, "FAA displaced threshold"),
    "helipad": ("X", "#1B7837", 60, 6, "FAA helipad"),
}
#: legend order: the two 3DEP checkpoint classes adjacent, then the GNSS
#: occupation classes dark-to-light (continuous, semi-continuous, campaign,
#: pre-split legacy), then NGS, then the FAA runway classes.
LEGEND_ORDER = ("NVA", "VVA", "gnss_cont", "gnss_semicont", "gnss_campaign",
                "gnss", "monument", "runway_end", "displaced_threshold",
                "helipad")
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


#: XYZ tile providers for the contact-sheet RGB basemap (promoted from the
#: sandbox station-gallery driver, 2026-08-29). Tiles carry the provider's
#: ToS/attribution terms; the provider is credited in the sheet title.
#: Google reaches higher native zoom at polar latitudes (MDV recon
#: 2026-08-11: real content to z16 vs ESRI z13).
WEB_BASEMAP_PROVIDERS = {
    "esri": ("(c) Esri World Imagery",
             "https://server.arcgisonline.com/ArcGIS/rest/services/"
             "World_Imagery/MapServer/tile/${z}/${y}/${x}"),
    "google": ("(c) Google Satellite",
               "https://mt1.google.com/vt/lyrs=s&amp;x=${x}&amp;y=${y}&amp;z=${z}"),
    # open terrain layer for MAP underlays (house style is shaded relief;
    # AOI-only runs have no DEM to shade — owner 2026-08-30)
    "esri_hillshade": ("(c) Esri World Hillshade",
                       "https://server.arcgisonline.com/ArcGIS/rest/services/"
                       "Elevation/World_Hillshade/MapServer/tile/${z}/${y}/${x}"),
}

_WMS_XML = """<GDAL_WMS>
  <Service name="TMS"><ServerUrl>{url}</ServerUrl></Service>
  <UserAgent>groundcontrol contact sheets (GDAL)</UserAgent>
  <ZeroBlockHttpCodes>204,400,403,404,500,503</ZeroBlockHttpCodes>
  <ZeroBlockOnServerException>true</ZeroBlockOnServerException>
  <DataWindow>
    <UpperLeftX>-20037508.34</UpperLeftX><UpperLeftY>20037508.34</UpperLeftY>
    <LowerRightX>20037508.34</LowerRightX><LowerRightY>-20037508.34</LowerRightY>
    <TileLevel>{z}</TileLevel><TileCountX>1</TileCountX><TileCountY>1</TileCountY>
    <YOrigin>top</YOrigin>
  </DataWindow>
  <Projection>EPSG:3857</Projection>
  <BlockSizeX>256</BlockSizeX><BlockSizeY>256</BlockSizeY>
  <BandsCount>3</BandsCount><DataType>Byte</DataType>
  <MaxConnections>4</MaxConnections><Timeout>15</Timeout><Cache/>
</GDAL_WMS>
"""


def open_web_basemap(dst_crs, bounds, *, provider="esri", tile_level="auto",
                     margin_m=500.0, probe_xy=None):
    """Tiled web imagery as an OPEN dataset warped into ``dst_crs`` over
    ``bounds`` (+margin) — the contact sheets' default RGB panel. Windowed
    reads fetch only the tiles they touch; an in-memory ``WarpedVRT`` (no
    gdalwarp subprocess) serves true ground meters at any latitude, with
    the pixel size matched to ``tile_level``'s native resolution at the
    site latitude. ``tile_level="auto"`` (default) probes z19 down to z15
    and keeps the deepest level where at least HALF the probe locations
    show real content (vs :data:`WEB_BLANK_CHROMA`) — ``probe_xy``: up to a
    dozen ``(x, y)`` points in ``dst_crs`` (the contact sheets pass their
    control points: a 190 km Nepal AOI whose bbox center hits Kathmandu's
    z19 tiles must not pick z19 for the 11 rural stations; 2026-08-30),
    else 3 bbox fractions. Providers' max level varies by region: Esri
    rural Nepal tops out well below its z19 CONUS coverage.
    Returns ``(base_dataset, warped_vrt)`` — the CALLER closes both (vrt
    first) — or ``None`` with a warning when the WMS driver, the network,
    or the CRS math is unavailable (the sheets then simply lack the RGB
    panel; auxiliary imagery is not worth failing an assessment over).
    """
    import math

    import pyproj
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT

    label, url = WEB_BASEMAP_PROVIDERS[provider]

    def _open(z):
        crs = pyproj.CRS.from_user_input(dst_crs)
        minx, miny, maxx, maxy = (float(v) for v in bounds)
        to4326 = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lat = to4326.transform((minx + maxx) / 2, (miny + maxy) / 2)[1]
        res = 156543.03392 * math.cos(math.radians(lat)) / 2 ** z
        m = margin_m
        if crs.is_geographic:
            res, m = res / 111320.0, margin_m / 111320.0
        x0, y0, x1, y1 = minx - m, miny - m, maxx + m, maxy + m
        w = max(1, int(math.ceil((x1 - x0) / res)))
        h = max(1, int(math.ceil((y1 - y0) / res)))
        base = rasterio.open(_WMS_XML.format(url=url, z=z))
        try:
            vrt = WarpedVRT(base, crs=crs.to_wkt(),
                            transform=from_origin(x0, y1, res, res),
                            width=w, height=h)
        except Exception:
            base.close()
            raise
        vrt.gc_web_basemap = True  # placeholder-tile detection is scoped to us
        return base, vrt, res, w, h

    def _has_content(vrt, w, h):
        from rasterio.windows import Window
        if probe_xy:
            rcs = []
            inv = ~vrt.transform
            for x, y in list(probe_xy)[:12]:
                c, r = inv * (float(x), float(y))
                if 0 <= c < w and 0 <= r < h:
                    rcs.append((int(r), int(c)))
        else:
            rcs = [(int(h * f), int(w * f)) for f in (0.5, 0.25, 0.75)]
        if not rcs:
            rcs = [(h // 2, w // 2)]
        n_ok = 0
        for r, c in rcs:
            win = Window(max(0, c - 24), max(0, r - 24),
                         min(48, w), min(48, h))
            arr = vrt.read(window=win).astype("float64")
            if (arr.size and arr.shape[0] >= 3
                    and np.abs(np.diff(arr[:3], axis=0)).mean() >= WEB_BLANK_CHROMA):
                n_ok += 1
        # at least half the probed locations must show content: one urban
        # station must not pin a deep level the rural majority lacks
        return n_ok >= max(1, (len(rcs) + 1) // 2)

    try:
        levels = ((19, 18, 17, 16, 15) if tile_level == "auto"
                  else (int(tile_level),))
        base = vrt = None
        for z in levels:
            if base is not None:
                vrt.close()
                base.close()
            base, vrt, res, w, h = _open(z)
            if len(levels) == 1 or _has_content(vrt, w, h):
                break
        else:
            logger.warning("web basemap %s: no content found down to z%d at "
                           "this AOI; keeping the last level (honest blanks)",
                           provider, levels[-1])
    except Exception as e:
        logger.warning("web basemap (%s) unavailable, RGB panel skipped: %s",
                       provider, e)
        return None
    logger.info("web basemap %s: z%d%s, %.3g units/px, %dx%d over %s",
                provider, z, " (auto)" if tile_level == "auto" else "",
                res, w, h, [round(b) for b in bounds])
    return base, vrt


#: standard output-layout: per-SOURCE subdirectories under --outdir (owner
#: 2026-08-30). Top level keeps the combined artifacts (control map,
#: validation dz, MIDAS maps, parquet/CSV); source-specific figures land in
#: their source's subdir. GNSS-class subsets share "gnss" (opus + ngl +
#: campaign are one physical class of marks).
SOURCE_DIRS = {
    # contact-sheet subsets
    "3dep_nva": "3dep", "3dep_vva": "3dep",
    "opus": "gnss", "cors": "gnss", "gnss_other": "gnss",
    "faa_runway": "faa",
    # family_dz_figures families
    "3dep": "3dep", "gnss": "gnss", "ngs_best": "ngs", "faa": "faa",
}

#: standard contact-sheet zoom tiers: (half-window m, tag, interpolation,
#: scalebar m) — 120 m context + native-pixel 30 m (owner 2026-08-11: "see
#: the 0.5 m pixels, maybe the antenna"; sandbox site_station_gallery TIERS)
SHEET_TIERS = ((60.0, "120m", "antialiased", 25), (15.0, "30m", "nearest", 10))


def _sheet_subsets(sampled):
    """Contact-sheet subsets broken out by source/class — only what came back
    for the AOI/DEM renders (owner 2026-08-29): CORS (continuous GNSS), OPUS
    campaign, other GNSS, FAA runway control, 3DEP NVA, 3DEP VVA. Dense NGS
    monuments never get sheets. Returns {tag: (points, class_col, colors)}."""
    subsets = {}
    pt = sampled.get("point_type")
    src = sampled.get("source")
    if pt is None:
        return subsets
    pt = pt.astype("string")

    def _add(tag, mask, cls=None, colors=None):
        sub = sampled[mask.fillna(False)]
        if len(sub):
            subsets[tag] = (sub, cls, colors)

    gnss_style = {c: POINT_STYLE[c][1] for c in
                  ("gnss_cont", "gnss_semicont", "gnss_campaign", "gnss")}
    _add("cors", pt == "gnss_cont", "point_type", gnss_style)
    _add("opus", (src == "opus") & pt.str.startswith("gnss")
         if src is not None else pt == "__never__", "point_type", gnss_style)
    _add("gnss_other", pt.str.startswith("gnss") & (pt != "gnss_cont")
         & ((src != "opus") if src is not None else True),
         "point_type", gnss_style)
    if src is not None:
        faa = sampled[(src == "faa").fillna(False)]
        if len(faa):
            cls = colors = None
            if "raw" in faa.columns:
                faa = faa.assign(pos_class=_raw_field(faa["raw"], "pos_class"))
                if faa["pos_class"].notna().any():
                    cls = "pos_class"
                    colors = {"surveyed": "crimson", "estimated": "darkorange"}
            subsets["faa_runway"] = (faa, cls, colors)
    _add("3dep_nva", pt == "NVA")
    _add("3dep_vva", pt == "VVA")
    return subsets


def context_sheets(sampled, products, outdir, site_name, *, rgb=None,
                   intensity=None, basemap="esri", tiers=SHEET_TIERS, dpi=150):
    """STANDARD per-point context contact sheets, per subset and zoom tier.

    For every subset :func:`_sheet_subsets` finds (CORS / OPUS / other GNSS /
    FAA runway / 3DEP NVA / 3DEP VVA — whatever came back for the AOI or
    input DEM) and every tier in ``tiers`` (default 120 m antialiased + 30 m
    native-pixel), one :func:`point_context_gallery` page set. The layer
    stack adapts to what exists (owner 2026-08-29: relief alone is not
    enough, and neither intensity nor a DEM can be assumed):

    - RGB imagery — ``rgb`` ortho path(s) and/or the ``basemap`` web
      provider (:data:`WEB_BASEMAP_PROVIDERS` key, default ``"esri"``:
      network tiles, credited in the title; ``None`` for offline) as the
      nodata-fallback chain;
    - ``intensity`` grayscale, when given;
    - one color shaded relief per path-backed product, when any.

    An AOI-only fetch (no products, no intensity) gets RGB-only sheets;
    with no renderable layer at all, no sheets. Control in a geographic CRS
    (the fetch landing) gets its web basemap built in the AOI's estimated
    UTM. Returns the written pages
    (``<site>_<subset>_gallery_<tier>[_pN].png``).
    """
    from contextlib import ExitStack

    relief = [(f"{name} relief", p, "relief") for name, p in (products or {}).items()
              if isinstance(p, (str, Path))]
    subsets = _sheet_subsets(sampled)
    if not subsets:
        return []
    out = []
    with ExitStack() as stack:
        chain, tag = list(rgb) if isinstance(rgb, (list, tuple)) else \
            ([rgb] if rgb else []), "RGB ortho"
        if basemap is not None:
            all_pts = pd.concat([p for p, _, _ in subsets.values()])
            map_crs = sampled.crs
            if map_crs is not None and map_crs.is_geographic:
                map_crs = all_pts.estimate_utm_crs()   # meter windows need a grid
                all_pts = all_pts.to_crs(map_crs)
            step = max(1, len(all_pts) // 12)
            web = open_web_basemap(
                map_crs, all_pts.total_bounds, provider=basemap,
                probe_xy=list(zip(all_pts.geometry.x[::step],
                                  all_pts.geometry.y[::step])))
            if web is not None:
                base, vrt = web
                stack.callback(base.close)
                stack.callback(vrt.close)
                chain.append(vrt)  # pre-opened: gallery reads, we close
                label = WEB_BASEMAP_PROVIDERS[basemap][0]
                tag = f"RGB ortho ({label} fallback)" if rgb else label
        layers = ([(tag, chain if len(chain) > 1 else chain[0], "rgb")]
                  if chain else [])
        if intensity is not None:
            layers.append(("intensity", intensity, "gray"))
        layers += relief
        if not layers:
            logger.info("context sheets skipped: no RGB/intensity/product layer")
            return []
        for stag, (pts, cls, colors) in subsets.items():
            if "id" not in pts.columns:  # synthetic frames; schema always has id
                pts = pts.assign(id=pts.index.astype(str))
            sub_out = Path(outdir) / SOURCE_DIRS.get(stag, stag)
            for half_m, ttag, interp, slen in tiers:
                out += point_context_gallery(
                    pts, layers, sub_out, site_name, half_m=half_m,
                    tier_tag=ttag, interp=interp, scale_len=slen,
                    class_col=cls, class_colors=colors, subset_tag=stag,
                    dpi=dpi)
    return out


def point_context_gallery(points, layers, outdir, site_name, *,
                          half_m=60.0, tier_tag=None, interp="antialiased",
                          scale_len=25, id_col="id", class_col=None,
                          class_colors=None, subset_tag="station",
                          ncell=None, max_rows=12, sort=True, dpi=200):
    """Per-point context contact sheet: one row-cell of image panels per point.

    For each point, a horizontal strip of windows from ``layers`` — e.g.
    TrueOrtho RGB | lidar intensity | DSM color shaded relief — so the
    physical setting of every control point (roof mount, mast, pier, bare
    ground) is reviewable at a glance. STANDARD for the GNSS/FAA subsets
    via :func:`context_sheets` in ``assess_products`` (owner 2026-08-29;
    formerly opt-in); call directly for custom layer stacks or subsets.
    Grew out of the MDV monument work and the Casa Grande cal-range
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
        A chain entry may also be an already-OPEN rasterio dataset (e.g. a
        ``WarpedVRT`` from :func:`open_web_basemap`): it is read in place
        and NOT closed — the caller owns it.
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
        win = Window(col - halfx, row - halfy, 2 * halfx, 2 * halfy)
        try:
            arr = src.read(window=win, boundless=True,
                           fill_value=src.nodata if src.nodata is not None
                           else 0).astype("float64")
        except Exception:
            # WarpedVRT (the web-basemap panel) refuses boundless reads:
            # read the in-bounds overlap and pad the rest with NaN
            arr = np.full((src.count, 2 * halfy, 2 * halfx), np.nan)
            r0 = max(0, row - halfy)
            r1 = min(src.height, row + halfy)
            c0 = max(0, col - halfx)
            c1 = min(src.width, col + halfx)
            if r1 > r0 and c1 > c0:
                sub = src.read(window=Window(c0, r0, c1 - c0, r1 - r0)
                               ).astype("float64")
                arr[:, r0 - (row - halfy):r1 - (row - halfy),
                    c0 - (col - halfx):c1 - (col - halfx)] = sub
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        return arr, [x - halfx * px, x + halfx * px,
                     y - halfy * py, y + halfy * py], (px, py)

    def _valid_frac(arr, src, kind):
        # fraction of pixels carrying signal. The all-bands-zero heuristic
        # applies ONLY to untagged RGB (Byte mosaics fill gaps with 0 and
        # carry no nodata) — zero is a legitimate value in single-band
        # products (intensity, elevations near 0 m; Copilot review, PR #25)
        ok = np.isfinite(arr).all(axis=0)
        if kind == "rgb" and src.nodata is None:
            ok &= (arr != 0).any(axis=0)
            # a web provider's placeholder tile is pure achromatic (gray or
            # black + text): web-basemap sources only — user orthos may be
            # legitimately grayscale (KH-9)
            if (getattr(src, "gc_web_basemap", False) and arr.shape[0] >= 3
                    and ok.size and np.abs(np.diff(arr[:3], axis=0)).mean()
                    < WEB_BLANK_CHROMA):
                return 0.0
        return float(ok.mean()) if ok.size else 0.0

    def _panel(ax, dss, kind, x0, y0):
        for i, src in enumerate(dss):
            x, y = x0, y0
            if src.crs is not None and points.crs is not None \
                    and src.crs.to_wkt() != points.crs.to_wkt():
                xs, ys = _rio_transform(points.crs, src.crs, [x], [y])
                x, y = xs[0], ys[0]
            arr, ext, (px, py) = _window(src, x, y)
            frac = _valid_frac(arr, src, kind)
            if frac > 0.01 or i == len(dss) - 1:
                if i:
                    logger.info("fallback source %d used at (%.0f, %.0f)",
                                i, x0, y0)
                break
        if frac == 0.0:
            # honest blank, labeled (owner 2026-08-29: bare white panels
            # read as a bug) — the point is outside every source's data.
            # RETURN here: stretching an all-NaN window is pure
            # RuntimeWarning noise (owner 2026-09-01 report)
            ax.text(0.5, 0.12, "outside data extent", transform=ax.transAxes,
                    ha="center", fontsize=6.5, color="#888888")
            return x, y, ext
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
    owned = []  # datasets THIS call opened (pre-opened entries stay the caller's)
    try:
        for tag, p, kind in layers:
            chain = p if isinstance(p, (list, tuple)) else [p]
            # open sequentially: a comprehension that raises mid-chain
            # leaks the already-opened members (Copilot review, PR #25)
            opened = []
            for q in chain:
                if hasattr(q, "read"):  # already-open dataset (caller-owned)
                    opened.append(q)
                else:
                    ds = rasterio.open(q)
                    owned.append(ds)
                    opened.append(ds)
            srcs.append((tag, opened, kind))
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
                        try:
                            x, y, ext = _panel(ax, chain, kind,
                                               r.geometry.x, r.geometry.y)
                        except Exception:
                            # ONE retry: web-tile reads fail transiently
                            # (rate limits, dropped connections) and a
                            # second windowed read usually lands
                            ax.clear()
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
                        s = 170
                        ax.scatter([x], [y], s=s, marker=mk,
                                   linewidths=2.0, zorder=5, **mkw)
                        ax.set_xlim(ext[0], ext[1])
                        ax.set_ylim(ext[2], ext[3])
                    except Exception as e:
                        ax.clear()
                        ax.text(0.5, 0.5, f"{tag}\nunavailable", ha="center",
                                va="center", transform=ax.transAxes,
                                fontsize=8)
                        logger.warning(
                            "%s %s panel failed after retry (web-tile reads "
                            "can be transient; panel left blank): %s",
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
                f"{SHEET_SUBSET_TITLES.get(subset_tag, subset_tag)} points "
                f"— {tags} ({2*half_m:.0f} m "
                f"windows{', native pixels' if interp == 'nearest' else ''})"
                f": {site_name}{page_note}",
                fontsize=12, y=1.0 - 0.12 / fig_h)
            fig.subplots_adjust(left=0.01, right=0.995,
                                top=1.0 - 0.52 / fig_h, bottom=0.18 / fig_h)
            suffix = f"_p{pg}" if len(pages) > 1 else ""
            # JPEG q85 (owner 2026-09-01): the sheets are photo-heavy —
            # 15 MB PNGs compress to ~1-2 MB with no review-relevant loss
            fp = outdir / f"{site_name}_{subset_tag}_gallery_{tier}{suffix}.jpg"
            fig.savefig(fp, dpi=dpi, pil_kwargs={"quality": 85})
            plt.close(fig)
            out_paths.append(fp)
    finally:
        for src in owned:
            src.close()
    for fp_ in out_paths:
        logger.info("wrote %s", fp_)
    logger.info("%d contact-sheet page(s), %d points", len(out_paths), n)
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
        if v == "nan":  # pre-null-fix parquets serialized missing values as
            return None  # str(NaN); normalize like expand_attributes so the
        return v if v else None  # two raw readers bucket identically (rd 4)
    return series.apply(get)


#: mean inter-band difference below which a WEB-BASEMAP window is treated
#: as a provider placeholder ("map data not yet available" tiles are pure
#: achromatic gray/black + text — measured chroma 0.0 vs >= 24 for every
#: real window, Nepal/CG 2026-08-30). Applied ONLY to web-basemap sources:
#: a grayscale USER ortho (KH-9!) is legitimate achromatic imagery.
WEB_BLANK_CHROMA = 3.0

#: auto-hillshade decimation: longest DEM side read at most this many px
HILLSHADE_MAX_PX = 2048


def hillshade_from_raster(path, *, max_px: int = HILLSHADE_MAX_PX):
    """``(hillshade01, extent)`` computed from an elevation raster — the
    figure underlay when no pre-rendered hillshade is supplied.

    Band 1 is read decimated to at most ``max_px`` on the longest side
    (overviews are used when the raster has them; a huge raster without
    overviews is read once at full resolution, so pass a pre-rendered
    ``gdaldem hillshade`` for repeated runs on big mosaics), nodata -> NaN
    so holes stay transparent, then :func:`groundcontrol.plot.hillshade`
    (multidirectional, house style). The tuple is accepted wherever the
    figure helpers take an ``hs_tif`` path (:func:`_relief`). Returns
    ``None`` (no underlay) for a rotated or south-up grid.
    """
    import math

    import rasterio
    from rasterio.enums import Resampling

    from .plot import hillshade

    with rasterio.open(path) as src:
        t = src.transform
        if t.b or t.d or t.e >= 0:
            # imshow(extent=) draws an axis-aligned north-up box: a rotated
            # or south-up grid would render misplaced relief under correctly
            # placed points. Plain map instead (pass a pre-rendered --hs).
            logger.warning("hillshade from %s skipped: transform is rotated/"
                           "south-up (%s); pass a pre-rendered hillshade", path, t)
            return None
        f = max(1, math.ceil(max(src.width, src.height) / max_px))
        h, w = math.ceil(src.height / f), math.ceil(src.width / f)
        z = src.read(1, out_shape=(h, w), masked=True,
                     resampling=Resampling.average).astype("float64").filled(np.nan)
        b = src.bounds
    dx = (b.right - b.left) / w
    dy = (b.top - b.bottom) / h
    logger.info("hillshade from %s: %dx%d px (1/%d)", path, w, h, f)
    return hillshade(z, dx=dx, dy=dy, multidirectional=True), \
        [b.left, b.right, b.bottom, b.top]


def _relief(ax, dem_tif, hs_tif, cmap, dem_alpha, fig):
    """Grayscale hillshade underlay from ``hs_tif`` — a pre-rendered Byte
    hillshade path (gdaldem 1..255) or a ``(array01, extent)`` tuple from
    :func:`hillshade_from_raster` — plus an optional colored ``dem_tif``."""
    import rasterio
    ext = None
    if isinstance(hs_tif, tuple):
        hs, ext = hs_tif
        ax.imshow(hs, cmap="gray", vmin=0.0, vmax=1.0, extent=ext,
                  interpolation="antialiased", interpolation_stage="rgba")
    elif hs_tif is not None:
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
        zf = z[np.isfinite(z)]
        if not zf.size:   # empty/all-nodata DEM: no tint, no colorbar
            return
        im = ax.imshow(z, cmap=cmap, alpha=dem_alpha, extent=ext,
                       vmin=np.percentile(zf, 2),
                       vmax=np.percentile(zf, 98),
                       interpolation="antialiased",
                       interpolation_stage="rgba")
        if fig is not None:
            cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
            cb.set_label(f"Elevation (m, {_datum_tag(dem_crs)})",
                         fontsize=9, color=_INK)
            cb.ax.tick_params(labelsize=8, colors=_MUT)
    return ext


def _finish_map(ax, aoi_gdf, clip_to_aoi=True, points=None):
    """Ticks off + scalebar. With ``clip_to_aoi`` the axes are limited to the
    AOI bounds and the dashed outline is dropped (redundant when the map IS
    the AOI); pass False to keep the outline on un-clipped maps. With no AOI
    the axes are limited to ``points`` (a GeoDataFrame) plus a margin, so a
    full-mosaic hillshade underlay cannot zoom the map out to the product
    (review round 1: the auto hillshade made bbox-AOI maps specks)."""
    from .plot import add_scalebar
    if aoi_gdf is not None:
        if clip_to_aoi:
            b = aoi_gdf.total_bounds
            ax.set_xlim(b[0], b[2])
            ax.set_ylim(b[1], b[3])
        else:
            aoi_gdf.boundary.plot(ax=ax, color=_INK, lw=1.0, ls="--", alpha=0.45)
    elif points is not None and len(points):
        b = points.total_bounds
        if np.isfinite(b).all():
            # zero span (one point): a fixed margin in the frame's units
            unit = 1e-3 if (points.crs is not None and points.crs.is_geographic) else 1.0
            m = 0.05 * max(b[2] - b[0], b[3] - b[1]) or unit
            ax.set_xlim(b[0] - m, b[2] + m)
            ax.set_ylim(b[1] - m, b[3] + m)
    ax.set_xticks([])
    ax.set_yticks([])
    # equal aspect is the map contract (env figures.md) and silences the
    # matplotlib-scalebar unequal-aspect warning (#23)
    ax.set_aspect("equal")
    crs = (aoi_gdf.crs if aoi_gdf is not None
           else points.crs if points is not None else None)
    add_scalebar(ax, crs=crs)


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


def _map_panel_size(gdf, *, base=7.0, min_in=2.6, max_in=12.0, max_h=None):
    """Equal-aspect map panel size ``(w, h)`` in inches from ``gdf`` bounds
    (an AOI or points GeoDataFrame), ASPECT-TRUE so the drawn map fills its
    axes with no letterbox — the fixed-canvas root cause of the wide/tall-AOI
    whitespace (owner layout audit 2026-08-30). ``base`` is the panel side
    for a square AOI; the long side of an extreme aspect is capped at
    ``max_in`` first — then, for figures whose side columns would stretch
    with a full-height map (validation/family), ``max_h`` caps the height —
    and finally the short side is floored at ``min_in`` (a >4:1 strip
    letterboxes slightly rather than becoming unreadable).
    ``None``/empty/degenerate bounds -> a ``base`` square."""
    asp = 1.0
    if gdf is not None and len(gdf):
        b = gdf.total_bounds
        dx, dy = float(b[2] - b[0]), float(b[3] - b[1])
        if np.isfinite(dx) and np.isfinite(dy) and dx > 0 and dy > 0:
            asp = dy / dx
    w, h = base / asp ** 0.5, base * asp ** 0.5
    if max(w, h) > max_in:
        f = max_in / max(w, h)
        w, h = w * f, h * f
    if max_h is not None and h > max_h:
        f = max_h / h
        w, h = w * f, h * f
    return max(w, min_in), max(h, min_in)


#: too-pale-for-white-background fills -> the legible ink used for
#: histogram fills/edges and stats text (class_ink); the GNSS-family hue
#: keeps the campaign class visually in the family
_PALE_INK = {"white": "#4477AA", "#A6CEE3": "#6FA3D0"}


#: contact-sheet subset display names — titles say what the subset IS,
#: not the internal tag (owner 2026-08-31 title sweep); unknown tags
#: (sandbox callers) fall back to the tag itself.
SHEET_SUBSET_TITLES = {
    "cors": "GNSS continuous (CORS)",
    "opus": "OPUS shared solutions",
    "gnss_other": "GNSS semi-continuous / campaign",
    "faa_runway": "FAA runway",
    "3dep_nva": "3DEP NVA checkpoint",
    "3dep_vva": "3DEP VVA checkpoint",
}

def _sparse_boost(n: int) -> float:
    """Marker-size multiplier keyed on the MAP-TOTAL point count (owner
    2026-09-01: three monuments vanished on a full-map hillshade; the
    per-class version then mixed marker scales on one map — sparse FAA
    next to dense NGS — which read as inconsistency). ONE factor per
    map, applied to every class uniformly."""
    return 3.0 if n <= 10 else (1.8 if n <= 50 else 1.0)


def class_ink(style_key_or_color):
    """Text/histogram color for a POINT_STYLE key or raw color — the pale
    map fills (light-blue campaign stars) fall back to :data:`_PALE_INK`
    so every white-background consumer agrees (owner 2026-08-30)."""
    col = (POINT_STYLE[style_key_or_color][1]
           if style_key_or_color in POINT_STYLE else style_key_or_color)
    return _PALE_INK.get(col, col)


def _edge_for(mk, col):
    """Map-marker edge: white halo normally; pale fills flip to the dark
    family edge so a white star stays visible on the hillshade."""
    if col in _PALE_INK:
        return "#08306B"
    return "white"


#: _web_map_underlay render cache: the MIDAS figure draws the SAME
#: underlay on two panels (owner 2026-09-01: duplicate esri fetches in
#: the log) — key (crs, rounded bounds, provider, max_px), tiny cap.
_UNDERLAY_CACHE: dict = {}


def _web_map_underlay(ax, crs, bounds, provider="esri_hillshade",
                      max_px=2400):
    """Web-tile underlay for a control map with no DEM (the AOI-only path,
    owner 2026-08-30): fixed tile level sized to the map span (the chroma
    probe cannot judge a grayscale hillshade layer), drawn gray with the
    provider credited on-axes."""
    import math

    label, _ = WEB_BASEMAP_PROVIDERS[provider]
    minx, miny, maxx, maxy = (float(v) for v in bounds)
    key = (str(crs), round(minx, 6), round(miny, 6), round(maxx, 6),
           round(maxy, 6), provider, max_px)
    hit = _UNDERLAY_CACHE.get(key)
    if hit is not None:
        arr, ext = hit
        ax.imshow(arr, cmap="gray", vmin=0, vmax=255, alpha=0.9, extent=ext,
                  zorder=0, interpolation="antialiased",
                  interpolation_stage="rgba")
        ax.text(0.995, 0.005, label, transform=ax.transAxes, ha="right",
                va="bottom", fontsize=6.5, color="#555555")
        return
    span = max(maxx - minx, maxy - miny)
    import pyproj
    if pyproj.CRS.from_user_input(crs).is_geographic:
        span *= 111320.0
    z = int(np.clip(round(math.log2(156543.0 / max(span / max_px, 0.01))), 8, 16))
    got = open_web_basemap(crs, bounds, provider=provider, tile_level=z,
                           margin_m=0.0)
    if got is None:
        return
    base, vrt = got
    try:
        dec = max(1, int(np.ceil(max(vrt.width, vrt.height) / max_px)))
        arr = vrt.read(1, out_shape=(vrt.height // dec, vrt.width // dec)
                       ).astype("f4")
        hb = vrt.bounds
    finally:
        vrt.close()
        base.close()
    ext = [hb.left, hb.right, hb.bottom, hb.top]
    if len(_UNDERLAY_CACHE) > 6:   # tiny working set: figure bundles only
        _UNDERLAY_CACHE.clear()
    _UNDERLAY_CACHE[key] = (arr, ext)
    ax.imshow(arr, cmap="gray", vmin=0, vmax=255, alpha=0.9, extent=ext,
              zorder=0, interpolation="antialiased",
              interpolation_stage="rgba")
    ax.text(0.995, 0.005, label, transform=ax.transAxes, ha="right",
            va="bottom", fontsize=6.5, color="#555555")


def control_map_figure(ctl, aoi_p, outdir, site_name, *, dem_tif=None,
                       hs_tif=None, cmap=None, dem_alpha=0.4,
                       clip_to_aoi=True, label_points=True,
                       label_gnss_ids=False, fname=None, title=None,
                       basemap="esri_hillshade", dpi=200):
    """ONE control map (the combined-map format/symbols): POINT_STYLE
    markers, rotated runway chevrons, sparse-class labels, legend with
    counts, hillshade underlay, scalebar. ``ctl`` must already be in the
    figure CRS and ``aoi_p`` projected to it (or None).

    ``label_gnss_ids=True`` widens the id labels from NGL/CORS to every
    GNSS-class point (owner 2026-08-30: the per-source maps are the
    locators for the contact sheets, and OPUS ids are sparse enough to
    read there — never on the combined map).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.markers import MarkerStyle
    from matplotlib.transforms import Affine2D

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # figsize follows the AOI/point-extent aspect (owner layout audit
    # 2026-08-30: the fixed 10.5x10 canvas left huge blank bands left/right
    # of tall maps and above/below wide ones); bbox_inches="tight" at
    # savefig trims the residual canvas
    mw, mh = _map_panel_size(aoi_p if aoi_p is not None else ctl,
                             base=8.0, min_in=3.5, max_in=12.0)
    fig, ax = plt.subplots(figsize=(mw + 0.4, mh + 0.7))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.94)
    _relief(ax, dem_tif, hs_tif, cmap, dem_alpha, fig)
    if dem_tif is None and hs_tif is None and basemap is not None:
        # no DEM to shade (AOI-only): open web hillshade underlay, credited
        try:
            b = (aoi_p.total_bounds if aoi_p is not None else ctl.total_bounds)
            _web_map_underlay(ax, ctl.crs, b, provider=basemap)
        except Exception as exc:  # underlay is auxiliary — never fatal
            logger.warning("map underlay skipped: %s", exc)

    by_type = {}
    boost = _sparse_boost(len(ctl))          # ONE factor for the whole map
    for ptype, (mk, col, sz, zo, lab) in POINT_STYLE.items():
        sub = ctl[ctl.point_type == ptype]
        if not len(sub):
            continue
        sz = int(sz * boost)
        lw = 0.5 if boost == 1.0 else 1.0
        ec = _edge_for(mk, col)
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
        # legend swatch (every POINT_STYLE marker is filled since
        # 2026-08-31): pale fills keep the dark family edge
        by_type[ptype] = Line2D(
            [], [], marker=mk, ls="", ms=9, markerfacecolor=col,
            markeredgecolor=(ec if col in _PALE_INK else col),
            color=class_ink(col), label=f"{lab} (n={len(sub)})")
    if label_points and "source" in ctl.columns and "id" in ctl.columns:
        import matplotlib.patheffects as _pe
        halo = [_pe.withStroke(linewidth=2.2, foreground="white")]
        # GNSS/CORS labels place FIRST and unconditionally; FAA airport
        # labels yield to them (owner 2026-08-13): dodge below on a close
        # approach, drop entirely on a collision
        gmask = ctl["source"] == "ngl"
        if label_gnss_ids and "point_type" in ctl.columns:
            gmask = gmask | ctl["point_type"].astype("string").str.startswith(
                "gnss").fillna(False)
        anchors = []
        for _, r in ctl[gmask].iterrows():
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
    _finish_map(ax, aoi_p, clip_to_aoi, points=ctl)
    ax.set_title(title or f"Control points (n={len(ctl)}): {site_name}",
                 fontsize=11, color=_INK)
    fp = outdir / (fname or f"{site_name}_control_map.png")
    fig.savefig(fp, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fp


def _ngl_series(sid, frame):
    """(decyear, E, N, U) full-value arrays for one station (cached tenv3)."""
    from .sources import ngl as _ngl
    ts = _ngl.read_tenv3(sid, frame=frame)
    t = ts["decyear"].to_numpy(dtype="float64")
    e = (ts["e0"].to_numpy(dtype="float64") + ts["east"].to_numpy(dtype="float64"))
    n = (ts["n0"].to_numpy(dtype="float64") + ts["north"].to_numpy(dtype="float64"))
    u = ts["height"].to_numpy(dtype="float64")
    return t, e, n, u


def _bin_medians(t, v, bin_yr):
    bins = np.round(t / bin_yr) * bin_yr
    bt = np.unique(bins)
    bv = np.array([float(np.nanmedian(vb))
                   if np.isfinite(vb := v[bins == b]).any() else np.nan
                   for b in bt])
    return bt, bv


def _row_steps(raw, key):
    import json
    try:
        return [float(x) for x in ((json.loads(raw) or {}).get(key) or [])]
    except (TypeError, ValueError):
        return []


def gnss_timeseries(control, outdir, site_name, *, frame="IGS14",
                    bin_yr=0.05, dpi=200):
    """TOP-LEVEL standard NGL component time series (owner 2026-08-30: the
    complementary panel set to the MIDAS velocity maps): three stacked
    panels — dE, dN, dU — every NGL station median-removed and reduced to
    ``bin_yr`` bin medians, colored per panel by that component's MIDAS
    rate on the RdYlBu ramp (RED = negative; for dU that is subsidence),
    earthquake steps dashed. Returns the path or None (no NGL rows).

    Part of the standard bundle: :func:`standard_control_figures` (and so
    :func:`groundcontrol.assess.assess_products`) already emits this into
    the ``ngl/`` subdir — calling it directly as well duplicates the
    figure in a second directory."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "source" not in control.columns:
        return None
    st = control[control["source"] == "ngl"]
    if not len(st):
        return None
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    comps = (("dE", "vel_e", 1), ("dN", "vel_n", 2), ("dU", "vel_u", 3))
    series = {}
    steps_all = set()
    for _, r in st.iterrows():
        sid = str(r["id"])
        try:
            series[sid] = _ngl_series(sid, frame)
        except Exception as exc:
            logger.warning("tenv3 for %s unavailable (%s); skipped", sid, exc)
            continue
        steps_all.update(_row_steps(r.get("raw"), "eq_steps"))
    if not series:
        return None
    import matplotlib.patheffects as pe

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    def _lim(vals):
        v = pd.to_numeric(vals, errors="coerce") * 1000.0
        return max(float(np.nanpercentile(np.abs(v), 98))
                   if np.isfinite(v).any() else 1.0, 0.5)

    # dE and dN share ONE color scale (owner 2026-08-31) so horizontal
    # rates are comparable across the two panels; dU keeps its own.
    lim_h = _lim(pd.concat([pd.to_numeric(st.get("vel_e"), errors="coerce"),
                            pd.to_numeric(st.get("vel_n"), errors="coerce")]))
    for ax, (lab, vcol, ci) in zip(axes, comps):
        lim = lim_h if vcol in ("vel_e", "vel_n") else _lim(st.get(vcol))
        cmap = plt.get_cmap("RdYlBu")
        norm = plt.Normalize(-lim, lim)
        for (_, r) in st.iterrows():
            sid = str(r["id"])
            if sid not in series:
                continue
            t = series[sid][0]
            v = (series[sid][ci] - np.nanmedian(series[sid][ci])) * 1000.0
            bt, bv = _bin_medians(t, v, bin_yr)
            rate = pd.to_numeric(pd.Series([r.get(vcol)]),
                                 errors="coerce").iloc[0]
            col = cmap(norm(rate * 1000.0)) if np.isfinite(rate) else "0.5"
            ax.plot(bt, bv, ".-", ms=2.2, lw=0.7, color=col, alpha=0.85)
            # thin dark halo (owner 2026-08-31): light ramp colors made
            # the station names unreadable on white
            ax.annotate(sid, (bt[-1], bv[-1]), xytext=(4, 0),
                        textcoords="offset points", fontsize=6.5, color=col,
                        fontweight="bold",
                        path_effects=[pe.withStroke(linewidth=1.1,
                                                    foreground="0.25")])
        for s_ in sorted(steps_all):
            ax.axvline(s_, color="0.4", lw=0.9, ls="--", zorder=1)
        ax.axhline(0, color=_INK, lw=0.6, alpha=0.5)
        ax.set_ylabel(f"{lab} (mm, median-removed)", fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=ax, pad=0.01)
        cb.set_label(f"MIDAS {vcol} (mm/yr"
                     + (", shared E/N scale)" if vcol != "vel_u" else ")"),
                     fontsize=8)
        cb.ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("year")
    axes[0].set_title(f"NGL E/N/U component series ({bin_yr:g}-yr bin "
                      f"medians; dashed = earthquake steps), "
                      f"n={len(series)} stations: {site_name}",
                      fontsize=11, color=_INK)
    fp = outdir / f"{site_name}_gnss_timeseries.png"
    fig.savefig(fp, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fp


def step_aware_fit(dy, U, step_dates_dy, min_span=2.0, min_n=300,
                   nmad_mult=3.0, offset_min_m=0.003, offset_sig=3.0,
                   win_yr=0.75):
    """Step-offset rate fit over the FULL series, keeping only REAL steps.

    The owner-approved fit (graduated verbatim from the sandbox Las Vegas
    analysis, 2026-08-31). NGL steps.txt lists *candidate* events (any
    equipment change incl. receiver/firmware swaps, plus nearby
    earthquakes) — most produce NO height offset (owner: UNR1 is a single
    continuous series despite logged events). So:

    1. fit common slope + per-candidate-segment intercepts (robust, full
       series);
    2. test each candidate's offset locally (medians of slope-detrended
       data in windows beside the step, clipped at neighboring
       candidates): keep only ``|offset| > max(offset_sig * SE,
       offset_min_m)``;
    3. refit with the KEPT steps only.

    Candidates logged close together (antenna + radome entries are often
    days apart) are clustered within 0.25 yr and tested as ONE composite
    step — clipping test windows at each other starved the test and
    silently rejected real steps (e.g. NVCA). Short records (span <
    ``min_span`` yr or n < ``min_n``) get no rate (checkpoints).

    Returns dict: class, rate_mm_yr, t0, t1, n_used, used, outliers,
    offsets [(step_dy, offset_m) kept], steps_rejected [step_dy...],
    model {slope_m_yr, intercepts, steps} for drawing fit lines.
    """
    dy = np.asarray(dy, dtype="float64")
    U = np.asarray(U, dtype="float64")
    t0, t1 = float(dy.min()), float(dy.max())
    if (t1 - t0) < min_span or len(dy) < min_n:
        return {"class": "short", "rate_mm_yr": None, "t0": t0, "t1": t1,
                "n_used": len(dy), "used": np.ones(len(dy), bool),
                "outliers": np.zeros(len(dy), bool), "offsets": [],
                "steps_rejected": [], "model": None}
    cand = np.sort(np.asarray(step_dates_dy, dtype="float64"))
    cand = cand[(cand > t0) & (cand < t1)]

    def robust_fit(steps):
        seg = np.searchsorted(steps, dy, side="right")
        nseg = len(steps) + 1

        def _lsq(mask):
            X = np.zeros((int(mask.sum()), 1 + nseg))
            X[:, 0] = dy[mask]
            X[np.arange(int(mask.sum())), 1 + seg[mask]] = 1.0
            coef, *_ = np.linalg.lstsq(X, U[mask], rcond=None)
            return coef

        inl = np.ones(len(dy), bool)
        coef = _lsq(inl)
        resid = U - (coef[0] * dy + coef[1 + seg])
        med = np.median(resid[inl])
        nmad = 1.4826 * np.median(np.abs(resid[inl] - med))
        out = (np.abs(resid - med) > nmad_mult * nmad) if nmad > 0 \
            else np.zeros(len(dy), bool)
        if out.any():
            inl = ~out
            coef = _lsq(inl)
        return coef, seg, inl, out

    # cluster candidates logged close together; test each cluster as ONE
    # composite step at its first date
    reps = []
    for c in cand:
        if reps and (c - reps[-1][-1]) <= 0.25:
            reps[-1].append(float(c))
        else:
            reps.append([float(c)])
    rep_dates = np.array([r[0] for r in reps])

    # pass 1: all cluster representatives -> initial common slope
    coef, seg, inl, out = robust_fit(rep_dates)
    detr = U - coef[0] * dy
    kept, rejected = [], []
    edges = np.concatenate([[t0], rep_dates, [t1]])
    for k, c in enumerate(rep_dates):
        c_end = reps[k][-1]                      # cluster spans first..last entry
        for widen in (win_yr, 10.0):             # local window, then widen fully
            lo = max(c - widen, edges[k])        # clip at neighboring CLUSTERS
            hi = min(c_end + widen, edges[k + 2])
            b = inl & (dy >= lo) & (dy < c)
            a = inl & (dy >= c_end) & (dy < hi)
            if b.sum() >= 20 and a.sum() >= 20:
                break
        if b.sum() < 20 or a.sum() < 20:         # record-edge cluster: an offset
            rejected.append(float(c))            # there cannot corrupt the slope
            continue
        mb = np.median(detr[b])
        ma = np.median(detr[a])
        nb = 1.4826 * np.median(np.abs(detr[b] - mb))
        na = 1.4826 * np.median(np.abs(detr[a] - ma))
        se = np.hypot(nb / np.sqrt(b.sum()), na / np.sqrt(a.sum()))
        if abs(ma - mb) > max(offset_sig * se, offset_min_m):
            kept.append(float(c))
        else:
            rejected.append(float(c))
    # pass 2: kept steps only
    kept_arr = np.asarray(kept, dtype="float64")
    coef, seg, inl, out = robust_fit(kept_arr)
    seg_counts = np.bincount(seg[inl], minlength=len(kept) + 1)
    offsets = [(c, float(coef[2 + k] - coef[1 + k]))
               for k, c in enumerate(kept)
               if seg_counts[k] > 0 and seg_counts[k + 1] > 0]
    return {"class": "rate", "rate_mm_yr": float(coef[0] * 1000.0),
            "t0": t0, "t1": t1, "n_used": int(inl.sum()),
            "used": inl, "outliers": out, "offsets": offsets,
            "steps_rejected": rejected,
            "model": {"slope_m_yr": float(coef[0]),
                      "intercepts": [float(v) for v in coef[1:]],
                      "steps": kept}}


def gnss_station_series(control, outdir, site_name, *, frame="IGS14",
                        dpi=200):
    """Per-station NGL daily vertical small multiples — the sandbox
    ``run_site_gnss.timeseries_figure`` layout adopted verbatim (owner
    2026-08-31: "find those and use, don't create something new"): daily
    U − station median (BLUE = used, GRAY = outliers, rasterized), ORANGE
    :func:`step_aware_fit` segments (common slope + per-segment
    intercepts), RED = kept (significant) steps, dotted GRAY = rejected
    candidates. Candidate steps come from the row's own steps.txt
    evidence (``raw`` eq_steps + equip_steps). Short records get no rate
    (checkpoint only). Returns the path or None (no NGL rows).

    Part of the standard bundle: :func:`standard_control_figures` (and so
    :func:`groundcontrol.assess.assess_products`) already emits this into
    the ``ngl/`` subdir — calling it directly as well duplicates the
    figure in a second directory."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "source" not in control.columns:
        return None
    st = control[control["source"] == "ngl"]
    if not len(st):
        return None
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, r in st.iterrows():
        sid = str(r["id"])
        try:
            t, _, _, u = _ngl_series(sid, frame)
        except Exception as exc:
            logger.warning("tenv3 for %s unavailable (%s); skipped", sid, exc)
            continue
        cand = sorted(set(_row_steps(r.get("raw"), "eq_steps")
                          + _row_steps(r.get("raw"), "equip_steps")))
        rows.append((sid, t, u, step_aware_fit(t, u, np.asarray(cand))))
    if not rows:
        return None
    rows.sort(key=lambda x: x[0])
    n = len(rows)
    ncols = min(5, max(2, n))
    nrow = int(np.ceil(n / ncols))
    fig_h = 1.9 * nrow + 0.75          # +title band (inches, nrow-proof)
    fig, axes = plt.subplots(nrow, ncols, figsize=(3.1 * ncols, fig_h),
                             sharex=True, squeeze=False)
    for k, (sid, dy_s, u, fit) in enumerate(rows):
        ax = axes[k // ncols][k % ncols]
        u_med = float(np.median(u))
        urel = u - u_med
        if fit["class"] == "rate":
            used = fit["used"]  # step OFFSETS are in-model; gray = outliers only
            ax.plot(dy_s[~used], urel[~used], ".", ms=1.5, color="0.55",
                    alpha=0.45, rasterized=True)
            ax.plot(dy_s[used], urel[used], ".", ms=1.0, color="tab:blue",
                    rasterized=True)
            mdl = fit["model"]
            seg_edges = np.concatenate([[dy_s.min()], mdl["steps"],
                                        [dy_s.max()]])
            for j in range(len(seg_edges) - 1):
                xx = np.array([seg_edges[j], seg_edges[j + 1]])
                yy = mdl["slope_m_yr"] * xx + mdl["intercepts"][j] - u_med
                ax.plot(xx, yy, "-", color="tab:orange", lw=1.2, zorder=5)
            for sdy in mdl["steps"]:              # KEPT (significant) steps
                ax.axvline(sdy, color="tab:red", lw=0.7, alpha=0.8)
            for sdy in fit["steps_rejected"]:     # candidates tested, no offset
                ax.axvline(sdy, color="0.6", lw=0.5, ls=":", alpha=0.6)
            nk = len(mdl["steps"])
            ttl = (f"{sid}  {fit['rate_mm_yr']:+.1f} mm/yr "
                   f"({nk} STEP{'S' if nk != 1 else ''})")
        else:
            ax.plot(dy_s, urel, ".", ms=1.0, color="0.55", alpha=0.45,
                    rasterized=True)
            ttl = f"{sid}  SHORT OCCUPATION (CHECKPOINT ONLY)"
        ax.axhline(0, color="0.5", lw=0.4)
        ax.set_title(ttl, fontsize=8)
        ax.tick_params(labelsize=7)
    for k in range(n, nrow * ncols):
        axes[k // ncols][k % ncols].set_visible(False)
    fig.supylabel("U - STATION MEDIAN (m)",
                  fontsize=min(10, 4 + 2 * nrow))  # short figs: don't clip
    fig.supxlabel("DECIMAL YEAR", fontsize=10, y=0.02)
    fig.suptitle(f"NGL daily vertical series, step-aware fits ({frame}), "
                 f"n={n} stations: {site_name}", fontsize=11, color=_INK,
                 y=1.0 - 0.10 / fig_h, va="top")
    fig.text(0.5, 1.0 - 0.38 / fig_h,
             "ORANGE = STEP-AWARE FIT (COMMON SLOPE + SEGMENT INTERCEPTS)"
             " \u00b7 RED = KEPT STEPS, DOTTED GRAY = REJECTED CANDIDATES"
             " \u00b7 BLUE = USED, GRAY = OUTLIERS/UNRATED",
             ha="center", va="top", fontsize=7.2, color="0.35")
    fig.tight_layout(rect=(0, 0.02, 1, 1.0 - 0.62 / fig_h))
    fp = outdir / f"{site_name}_gnss_station_series.png"
    fig.savefig(fp, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fp


def decyear_ts(ts):
    """Decimal year of a timestamp (thin local wrapper; crs.decyear is the
    canonical scalar converter)."""
    from .crs import decyear as _dy
    return _dy(ts)


def standard_control_figures(control, aoi, outdir, site_name, *,
                             dem_tif=None, hs_tif=None, cmap=None,
                             dem_alpha=0.4, midas_frame="IGS14",
                             midas_velocities=True, map_basemap="esri_hillshade",
                             buffer_km=None, clip_to_aoi=True,
                             label_points=True, dpi=200):
    """Write the default control figure bundle for a site; returns paths.

    ``label_points`` (owner request 2026-08-13, the NGS-map convention):
    sparse, named classes get text labels — NGL/CORS station ids per point,
    FAA points one label per AIRPORT (grouped by the id prefix; per-runway-
    end labels would be unreadable). Dense classes (3DEP checkpoints, NGS
    monuments) are never labeled.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = []
    from .aoi import read_aoi
    aoi_gdf = read_aoi(aoi) if isinstance(aoi, (str, Path)) else aoi

    # figure CRS: the DEM's if given, else the AOI's (or control's) UTM estimate
    import rasterio
    if dem_tif is not None:
        with rasterio.open(dem_tif) as src:
            fig_crs = src.crs
    elif aoi_gdf is not None:
        fig_crs = aoi_gdf.estimate_utm_crs()
    else:
        fig_crs = control.estimate_utm_crs()
    ctl = control.to_crs(fig_crs)
    aoi_p = aoi_gdf.to_crs(fig_crs) if aoi_gdf is not None else None

    # ---- 1. control map (combined, top level) ------------------------------
    out.append(control_map_figure(
        ctl, aoi_p, outdir, site_name, dem_tif=dem_tif, hs_tif=hs_tif,
        cmap=cmap, dem_alpha=dem_alpha, clip_to_aoi=clip_to_aoi,
        label_points=label_points, basemap=map_basemap, dpi=dpi))

    # ---- 1b. per-source maps, one per source subdir (owner 2026-08-30:
    # the locator for that source's contact sheets — same format/symbols
    # as the combined map, with GNSS-class ids labeled where sparse)
    dir_masks = {}
    if "source" in ctl.columns:
        dir_masks["3dep"] = (ctl["source"] == "3dep")
        dir_masks["ngs"] = (ctl["source"] == "ngs")
        dir_masks["faa"] = (ctl["source"] == "faa")
    if "point_type" in ctl.columns:
        dir_masks["gnss"] = ctl["point_type"].astype("string").str.startswith(
            "gnss").fillna(False)
    for dname, m in dir_masks.items():
        if not m.any():
            continue
        sub = ctl[m]
        out.append(control_map_figure(
            sub, aoi_p, outdir / dname, site_name, dem_tif=dem_tif,
            hs_tif=hs_tif, cmap=cmap, dem_alpha=dem_alpha,
            clip_to_aoi=clip_to_aoi, label_points=label_points,
            label_gnss_ids=True, fname=f"{site_name}_{dname}_map.png",
            title=f"{dname.upper()} control points (n={len(sub)}): "
                  f"{site_name}",
            basemap=map_basemap, dpi=dpi))

    # ---- 2. NGS monument-type facets (ngs/ subdir: source-specific) --------
    mon = (ctl[ctl.point_type == "monument"] if "raw" in ctl.columns
           else ctl.iloc[:0])  # facets read the raw datasheet fields
    if len(mon):
        (outdir / "ngs").mkdir(parents=True, exist_ok=True)
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
            # legend BEFORE _finish_map: the scalebar auto-locator skips
            # the corner a legend already holds, but only if it exists
            # when the scalebar is placed (owner 2026-09-01: facet legend
            # rendered under the scalebar)
            ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9)
            _finish_map(ax, aoi_p, clip_to_aoi)
            ax.set_title(f"NGS monuments by {key}", fontsize=10, color=_INK)
        fig.suptitle(f"NGS monument datasheet attributes (n={len(mon)}): "
                     f"{site_name}", fontsize=11.5, color=_INK)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fp = outdir / "ngs" / f"{site_name}_monument_types.png"
        fig.savefig(fp, dpi=dpi)
        plt.close(fig)
        out.append(fp)

    # ---- 3+4. MIDAS motion figures (network fetch: gated) -------------------
    if not midas_velocities:
        logger.info("MIDAS velocity figures skipped (midas_velocities=False)")
        return out
    ngl_scope = aoi_gdf
    if ngl_scope is None:
        # aoi=None must not silently drop the standard NGL bundle (rasuwa
        # shakeout 2026-08-31 — same silent-omission class as the
        # default-ON flip): scope the velocity/series figures to the
        # product footprint, else the control extent. Map CLIPPING is
        # untouched — this box exists only for this block.
        import geopandas as gpd
        import rasterio
        from shapely.geometry import box
        if dem_tif is not None:
            with rasterio.open(dem_tif) as src:
                ngl_scope = gpd.GeoDataFrame(geometry=[box(*src.bounds)],
                                             crs=src.crs)
            logger.info("MIDAS/NGL figure scope: aoi=None -> product bounds")
        else:
            ngl_scope = gpd.GeoDataFrame(geometry=[box(*ctl.total_bounds)],
                                         crs=ctl.crs)
            logger.info("MIDAS/NGL figure scope: aoi=None -> control bounds")
    try:
        from .plot import plot_velocity_vectors
        from .sources.ngl import read_midas
        st = read_midas(midas_frame)
        b = ngl_scope.to_crs(4326).total_bounds
        st = st[st.lon.between(b[0] - 3, b[2] + 3)
                & st.lat.between(b[1] - 3, b[3] + 3)]
        if not len(st):
            # the FAA/3DEP no-sites pattern (owner 2026-08-31): a region
            # with no MIDAS stations emits no velocity/series figures
            logger.info("MIDAS velocity figures skipped: no MIDAS "
                        "stations within the buffered AOI")
            return out
        # ONE combined figure (owner 2026-08-30): horizontal quiver |
        # vertical-colored, over the DEM hillshade, EQUAL-SIZE panels (the
        # colorbar gets its own axis instead of shrinking the right map).
        # The vertical panel drops the ref key and the interp annotation —
        # horizontal numbers live on the horizontal panel, whose interp
        # label now carries both H and U with their 1-sigma spreads.
        ngl_dir = outdir / "ngl"   # owner 2026-08-31: MIDAS + NGL series
        ngl_dir.mkdir(parents=True, exist_ok=True)   # live in ngl/
        fig2 = plt.figure(figsize=(18.6, 9))
        gs2 = fig2.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.03],
                                wspace=0.14)
        axh_ = fig2.add_subplot(gs2[0, 0])
        axv_ = fig2.add_subplot(gs2[0, 1], sharey=axh_)  # shared latitude
        cax_ = fig2.add_subplot(gs2[0, 2])
        hs_path = hs_tif if isinstance(hs_tif, (str, Path)) else None
        # map buffer = the interpolation search radius (owner 2026-08-31:
        # one consistent area around the site), web hillshade under the
        # DEM's own hillshade so the buffer zone is never blank
        if buffer_km is None:
            from .velocity import DEFAULT_RADIUS_KM as buffer_km
        vel_bmap = map_basemap
        plot_velocity_vectors(
            st, aoi=ngl_scope, buffer_km=buffer_km, ax=axh_,
            color_by_vertical=False, hs_tif=hs_path, dem_tif=dem_tif,
            basemap=vel_bmap, title="Horizontal motion (mm/yr)")
        plot_velocity_vectors(
            st, aoi=ngl_scope, buffer_km=buffer_km, ax=axv_,
            color_by_vertical=True, hs_tif=hs_path, dem_tif=dem_tif,
            basemap=vel_bmap, cbar_ax=cax_, show_ref=False,
            title="Vertical motion (mm/yr)")
        plt.setp(axv_.get_yticklabels(), visible=False)
        axv_.set_ylabel("")
        fig2.suptitle("GNSS velocities \u2014 MIDAS (Median Interannual "
                      f"Difference Adjusted for Skewness), {midas_frame}: "
                      f"{site_name}", fontsize=13, color=_INK)
        fp = ngl_dir / f"{site_name}_midas_velocity.png"
        fig2.savefig(fp, dpi=dpi, bbox_inches="tight")
        plt.close(fig2)
        out.append(fp)
        # ---- 5+6. NGL series beside their complementary MIDAS maps:
        # E/N/U common series + per-station step-aware small multiples
        for fn in (gnss_timeseries, gnss_station_series):
            try:
                fp_ts = fn(control, ngl_dir, site_name, frame=midas_frame,
                           dpi=dpi)
                if fp_ts is not None:
                    out.append(fp_ts)
            except Exception as exc:  # network etc. — the bundle still ships
                logger.warning("%s skipped: %s", fn.__name__, exc)
    except Exception as exc:  # network etc. — the map figures still ship
        logger.warning("MIDAS velocity figures skipped: %s", exc)

    for p_ in out:
        logger.info("wrote %s", p_)
    return out


def stats_lines(label, values, color):
    """THE dual-track stats lines for figure text blocks (owner 2026-08-30:
    formatting was duplicated across validation/family figures and the
    order is standardized here — n FIRST, then the robust pair, then the
    ASPRS Ed.2 parametric set after error_report's 3*NMAD gate). Returns
    ``[(text, color, bold), ...]``.
    """
    from .accuracy import error_report
    er = error_report(values)
    return [
        (f"{label}: n={er['n']}, med {er['median']:+.3f}, "
         f"NMAD {er['nmad']:.3f}", color, True),
        (f"   mean {er['mean']:+.3f}, σ {er['std']:.3f}, "
         f"RMSE {er['rmse']:.3f}, LE90 {er['le90']:.3f}"
         + (f" ({er['n_outliers']} out)" if er["n_outliers"] else ""),
         color, False),
    ]


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


#: validation-figure style per assess.SEGMENTS label: a POINT_STYLE key or a
#: raw hex. Lives in figures (SEGMENTS' 3-tuple shape is a de facto contract
#: — sandbox MDV_SEGMENTS uses it, so style cannot move into the tuple);
#: the sync test pins these keys to SEGMENTS' so a label add/rename fails in
#: CI, not at figure time. Campaign labels carry DISTINCT colors (audit
#: round 4: three same-color histograms hid the ARP-vs-ground-mark
#: distinction the split exists for).
_SEG_STYLE = {
    "3DEP NVA": "NVA", "3DEP VVA": "VVA",
    "GNSS continuous": "gnss_cont",
    "GNSS semi-continuous": "gnss_semicont",
    "GNSS campaign (OPUS)": "gnss_campaign",
    "GNSS campaign (NGL)": "#005F73",
    "GNSS campaign (other)": "#B07AA1",
    "GNSS (pre-split)": "gnss",
    "NGS monument": "monument",
    "FAA runway surveyed": "runway_end",
    "FAA other": "#8C6BB1",
    "OTHER (unsegmented)": "gnss",  # never rendered (context, non-GNSS
                                    # label) — placeholder for the sync test
}


def validation_dz_figures(sampled, aoi, outdir, site_name, *, products=("DSM", "DTM"),
                          hs_tif=None, point_lim=None, vendor_lim=None, wide_lim=None,
                          ngs_nmad_gate=3.0, dpi=200):
    # hs_tif: a single path, or a {product: path} dict for PRODUCT-MATCHED
    # backgrounds (DTM diffs belong on the DTM hillshade — David, 2026-07-15);
    # each value a pre-rendered hillshade path or a hillshade_from_raster()
    # tuple (assess_products derives one per product when none is given).
    """Product-vs-control vertical-offset validation figures (standard bundle
    item 5; requested by David 2026-07-15 after the SF run).

    One figure per product: (a) map of control points over shaded relief
    colored by ``dh_<product>_before`` (product - control, RdBu_r); (b)
    histograms for the survey-grade segments (vendor NVA/VVA, the GNSS
    occupation classes); (c) histogram for NGS monuments after a
    ``ngs_nmad_gate``-NMAD filter.
    Segment rules: NVA validates DSM and DTM; VVA validates DTM only;
    GNSS and NGS shown for both as datum-sanity context. Median/NMAD/n
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
        if isinstance(aoi, (str, Path)):
            from .aoi import read_aoi
            aoi = read_aoi(aoi)
        aoi = aoi.to_crs(sampled.crs)
    # ONE taxonomy: masks and DSM/DTM applicability come from assess.SEGMENTS
    # (audit round 2: this dict was a hand-copied twin and had already
    # diverged once); figures add only the style (module-level _SEG_STYLE —
    # the sync test pins its keys to SEGMENTS', so a label add/rename fails
    # in CI, not mid-run). Lazy import — assess imports figures lazily too,
    # no cycle.
    from groundcontrol.assess import SEGMENTS as _SEGMENTS, is_dtm_product
    seg_defs = {  # label -> (mask fn, style key or hex, in DSM, in DTM)
        lbl: (fn, _SEG_STYLE[lbl], in_dsm, in_dtm)
        for lbl, (fn, in_dsm, in_dtm) in _SEGMENTS.items()
    }
    for prod in products:
        col = f"dh_{prod}_before"
        if col not in sampled.columns:
            logger.warning("validation_dz: no column %s, skipping %s", col, prod)
            continue
        # Layout (owner iteration 2026-08-30): the equal-aspect map DOMINATES
        # and is sized to its true aspect so no letterbox whitespace; the two
        # histograms stack beside it on ONE SHARED x-axis; the dual-track
        # stats live in their own text panel below the histograms (3+
        # sources never fit inside a histogram box). Layout audit 2026-08-30:
        # the map COLUMN is now aspect-true too (the former width-only clamp
        # left the drawn map floating in an over-wide column with a dead gap
        # to the colorbar), margins are explicit inches so the cell height
        # matches what the panel width assumed, and the figure height adapts
        # down for wide AOIs, floored by the histogram+stats column.
        use = sampled[np.isfinite(sampled[col])]
        mw, mh = _map_panel_size(aoi if aoi is not None else use,
                                 base=6.6, min_in=2.0, max_in=12.0,
                                 max_h=6.6)
        sup = 0.55                            # title strip above the map
        hist_w = 4.2
        # the colorbar gets its OWN column (owner 2026-08-31, tall SF
        # AOI: attached to the map axes it landed against the histograms
        # and its label overprinted their spines); the column is wider
        # than the bar — the slack carries the left-side ticks + label —
        # and the bar itself is repositioned flush right after draw; hist
        # rows squished so the dual-track stats block breathes (2026-08-31)
        cb_w = 0.9
        fig_h = max(mh + 0.3, 5.6) + sup
        fig = plt.figure(figsize=(mw + cb_w + hist_w + 1.1, fig_h))
        gs = fig.add_gridspec(3, 3, width_ratios=[mw, cb_w, hist_w],
                              height_ratios=[0.82, 0.82, 0.98],
                              left=0.01, right=0.985, bottom=0.07,
                              top=1.0 - sup / fig_h,
                              hspace=0.3, wspace=0.14)
        ax_map = fig.add_subplot(gs[:, 0])
        ax_map.set_anchor("NW")  # aspect slack goes right/below — the map
        #                          hugs the title, never floats mid-column
        cax = fig.add_subplot(gs[:, 1])   # full map height, never floating
        ax_s = fig.add_subplot(gs[0, 2])
        ax_n = fig.add_subplot(gs[1, 2], sharex=ax_s)
        ax_t = fig.add_subplot(gs[2, 2])
        ax_t.set_axis_off()
        axes = [ax_map, ax_s, ax_n]
        hs_prod = hs_tif.get(prod) if isinstance(hs_tif, dict) else hs_tif
        _relief(axes[0], None, hs_prod, None, 0.0, None)
        pl = point_lim if point_lim is not None else snap_clim(use[col], k=3.0)
        # marker SHAPE carries class identity (owner 2026-08-30: identical
        # circles hid which points were NVA vs GNSS vs monuments vs FAA);
        # color stays the dz ramp (class colors clash with it, owner
        # 2026-07-16), POINT_STYLE shapes match the control map and sheets
        import matplotlib as _mpl
        from matplotlib.lines import Line2D
        norm = _mpl.colors.Normalize(vmin=-pl, vmax=pl)
        handles = []
        boost = _sparse_boost(len(use))      # ONE factor for the whole map
        if "point_type" in use.columns and use["point_type"].notna().any():
            pts_order = [t for t in LEGEND_ORDER
                         if (use["point_type"] == t).any()]
            pts_order += [t for t in use["point_type"].dropna().unique()
                          if t not in pts_order]
            for pt in pts_order:
                mk, _, msz, _, mlab = POINT_STYLE.get(
                    pt, ("o", "#888888", 34, 5, str(pt)))
                sub = use[use["point_type"] == pt]
                # every POINT_STYLE marker is FILLED (owner 2026-08-31: the
                # dz ramp needs face + thin dark edge; line-only shapes and
                # their under-stroke workaround are retired)
                axes[0].scatter(sub.geometry.x, sub.geometry.y, c=sub[col],
                                cmap=DZ_CMAP, norm=norm, marker=mk,
                                s=int(max(11, int(msz * 0.24)) * boost),
                                edgecolors="#333333",
                                linewidths=0.35 if boost == 1.0 else 0.7,
                                zorder=5)
                handles.append(Line2D([], [], marker=mk, ls="", color="#333333",
                                      ms=6, label=f"{mlab} ({len(sub)})"))
        else:
            axes[0].scatter(use.geometry.x, use.geometry.y, c=use[col],
                            cmap=DZ_CMAP, norm=norm, s=15,
                            edgecolors="#333333", linewidths=0.35, zorder=5)
        if handles:
            axes[0].legend(handles=handles, loc="lower left", fontsize=7,
                           framealpha=0.85, borderpad=0.4, handletextpad=0.4)
        sc = _mpl.cm.ScalarMappable(norm=norm, cmap=DZ_CMAP)
        cb = fig.colorbar(sc, cax=cax, extend="both")
        # ticks + label LEFT of the bar: the right side faces the
        # histograms and the label overprinted their y-ticks
        cb.ax.yaxis.set_ticks_position("left")
        cb.ax.yaxis.set_label_position("left")
        cb.set_label(f"dz = {prod} − control (m)", fontsize=9, color=_INK)
        cb.ax.tick_params(labelsize=8, colors=_MUT)
        _finish_map(axes[0], aoi, points=use)
        # FIGURE-level single-line title (owner 2026-09-01: an axes-level
        # title wrapped oddly across map aspects — the figure is always
        # wide enough, whatever the AOI shape)
        fig.suptitle(f"Vertical difference (m, {prod} minus control), "
                     f"n={len(use)}: {site_name}", x=0.01, y=0.995,
                     ha="left", va="top", fontsize=12, color=_INK)

        is_dtm = is_dtm_product(prod)  # the ONE DSM/DTM classifier (round 4)
        panels = []                       # (ax, seg_vals, seg_raw, own_lim)
        for ax, labels, lim_over in (
                # display rule != applies rule: context-only GNSS segments
                # (applies False/False in the stats) still render as
                # datum-sanity context per this figure's contract — the
                # validation flags alone silently emptied the GNSS
                # histograms (audit round 3). Empty segments drop out below.
                (ax_s, [lbl for lbl, s in seg_defs.items()
                        if ((s[3] if is_dtm else s[2])
                            or lbl.startswith("GNSS"))
                        and lbl != "NGS monument"],
                 vendor_lim),
                (ax_n, ["NGS monument"], wide_lim)):
            seg_vals = {}
            seg_raw = {}   # un-gated values: the dual-track stats source
            for lab in labels:
                maskfn, style, *_ = seg_defs[lab]
                v = use.loc[maskfn(use), col].to_numpy(float)
                v = v[np.isfinite(v)]
                raw_v = v
                if lab == "NGS monument" and len(v):
                    v = _ngs_gate(v, ngs_nmad_gate)   # display gate only
                if len(v):
                    seg_vals[lab] = v
                    seg_raw[lab] = raw_v
            # per-panel empirical tier (#23) from the panel's OWN values;
            # context-only segments never set the scale (audit round 4)
            val_vals = [v for lab, v in seg_vals.items()
                        if (seg_defs[lab][3] if is_dtm else seg_defs[lab][2])
                        or lab == "NGS monument"]
            own = lim_over if lim_over is not None else (
                snap_clim(np.concatenate(val_vals or list(seg_vals.values())),
                          k=3.0) if seg_vals else pl)
            panels.append((ax, seg_vals, seg_raw, own))
        # ONE shared x-axis across both histograms (owner 2026-08-30): the
        # wider panel's tier wins, typically the NGS monuments'
        lim = max(own for _, _, _, own in panels)
        # bin width follows the tighter (survey) spread so its spike still
        # resolves inside the shared, wider limits (owner 2026-08-30)
        sv = panels[0][1]
        pooled = np.concatenate(list(sv.values())) if sv else np.array([])
        bw = max(float(_nmad(pooled)) / 2.0, lim / 150.0) if pooled.size \
            else lim / 40.0
        nbins = int(np.clip(round(2 * lim / bw), 41, 201))
        txt_lines = []
        for ax, seg_vals, seg_raw, _own in panels:
            for lab, v in seg_vals.items():
                color = class_ink(seg_defs[lab][1])  # centralized legible ink
                ax.hist(np.clip(v, -lim, lim), bins=nbins, range=(-lim, lim),
                        histtype="stepfilled", alpha=0.45, color=color,
                        edgecolor=color, label=lab)
                # centralized dual-track lines (stats_lines: n first),
                # rendered OUTSIDE the histograms in their own panel
                # (owner 2026-08-30: 3+ sources never fit in a corner box)
                txt_lines.extend(stats_lines(lab, seg_raw[lab], color))
            ax.axvline(0, color=_INK, lw=0.8)
            ax.set_xlim(-lim, lim)
            if not seg_vals:
                ax.text(0.5, 0.5, "no matching checkpoints in AOI",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=9, color=_MUT)
            ax.tick_params(labelsize=8, colors=_MUT)
            ax.grid(alpha=0.25, lw=0.5)
        plt.setp(ax_s.get_xticklabels(), visible=False)
        ax_n.set_xlabel(f"dz = {prod} − control (m)", fontsize=9, color=_INK)
        if "xform_acc_m" in sampled.columns:
            _xa = sampled["xform_acc_m"].to_numpy(dtype="float64")
            if np.isfinite(_xa).any():
                txt_lines.append(("stated 3D transform budget "
                                  f"±{np.nanmedian(_xa):g} m", _MUT, False))
        if txt_lines:
            step = min(0.10, 0.97 / len(txt_lines))
            for i, (line, color, bold) in enumerate(txt_lines):
                ax_t.text(0.0, 0.98 - step * i, line, transform=ax_t.transAxes,
                          fontsize=7.5, va="top", color=color,
                          fontweight="bold" if bold else "normal")
        ax_s.set_title("survey-grade points", fontsize=10, color=_INK)
        ax_n.set_title(f"NGS monuments ({ngs_nmad_gate:.0f}-NMAD filtered)",
                       fontsize=10, color=_INK)
        fp = outdir / f"{site_name}_validation_dz_{prod}.png"
        # equal-aspect shrinks the MAP's axes box inside its gridspec
        # cell; clamp the colorbar to the map's final drawn height so it
        # never extends past the map (owner 2026-09-01), and to a fixed
        # 0.28-in bar flush right in its column (the column slack holds
        # the left-side ticks + label)
        fig.canvas.draw()
        pm = ax_map.get_position()
        figw = fig.get_size_inches()[0]
        bw = 0.28 / figw
        # anchored to the DRAWN map edge + room for the left-side
        # ticks/label (owner 2026-09-01: flush-right in the column left
        # the slack between map and bar — a floating colorbar)
        cax.set_position([pm.x1 + 0.85 / figw, pm.y0, bw, pm.height])
        # bbox_inches trims the residual outer margin (tight_layout fights
        # the colorbar + spanning-gridspec combination)
        fig.savefig(fp, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        out.append(fp)
        logger.info("wrote %s", fp)
    return out


def _opus_tier(d):
    """Row-wise OPUS stability tier for DZ_FAMILIES masks (lazy import —
    figures must stay importable without the sources subpackage loaded).
    Only OPUS rows are decoded: the family masks are all opus-gated, and
    the un-gated version JSON-parsed the FULL multi-source raw column on
    every mask evaluation — up to 12x per figure call (audit finding)."""
    from groundcontrol.sources.ngs import opus_stability_tier
    out = pd.Series(pd.NA, index=d.index, dtype="string")
    m = (d["source"] == "opus").fillna(False)
    if m.any():
        out[m] = opus_stability_tier(d[m])
    return out


#: family key -> (title, [(subclass label, row mask fn, point_type style key
#: or hex color, marker[, products])]). Optional 5th element restricts the
#: subclass to those products: VVA canopy checkpoints validate the DTM only —
#: an expected DSM bias is not an error, so VVA is EXCLUDED from the DSM
#: figure (owner 2026-07-15) rather than shown as a huge tail.
DZ_FAMILIES = {
    "3dep": ("3DEP checkpoints", [
        ("Non-Vegetated Vertical Accuracy (NVA)",
         lambda d: (d["source"] == "3dep") & (d["point_type"] == "NVA"),
         "NVA", "o"),
        # no products restriction (owner 2026-08-30): VVA renders on the DSM
        # figure too — the canopy bias is informative, and `applies` in the
        # stats CSV still says it does not validate a DSM
        ("Vegetated Vertical Accuracy (VVA)",
         lambda d: (d["source"] == "3dep") & (d["point_type"] == "VVA"),
         "VVA", "s"),
    ]),
    # GNSS by PER-ROW occupation class (owner taxonomy, 2026-08-22): each
    # station's own record earns its class (sources.ngl.occupation_class).
    # Continuous stations have permanent hardware (photo-ID candidates,
    # velocity-rich); campaign occupations leave nothing but the monument.
    # campaign split by HEIGHT BASIS, mirroring assess.SEGMENTS (audit
    # round 3: one pooled Campaign panel blended OPUS ground marks with
    # NGL antenna-reference points, a mixture whose med/NMAD matches
    # neither). Legacy OPUS rows fold into Campaign (OPUS) like the stats
    # table; empty subclasses are skipped at render time.
    "gnss": ("GNSS by occupation class", [
        ("Continuous", lambda d: d["point_type"] == "gnss_cont",
         "gnss_cont", "o"),
        ("Semi-continuous", lambda d: d["point_type"] == "gnss_semicont",
         "gnss_semicont", "o"),
        ("Campaign (OPUS)", lambda d: (d["source"] == "opus")
         & d["point_type"].isin(["gnss_campaign", "gnss"]),
         "gnss_campaign", "o"),
        ("Campaign (NGL ARP)", lambda d: (d["source"] == "ngl")
         & (d["point_type"] == "gnss_campaign"), "#7BA3CF", "^"),
        ("Campaign (other)", lambda d: (d["point_type"] == "gnss_campaign")
         & ~d["source"].isin(["opus", "ngl"]), "#888888", "s"),
        ("Pre-split (non-OPUS)", lambda d: (d["point_type"] == "gnss")
         & (d["source"] != "opus"), "gnss", "o"),
    ]),
    # OPUS campaign marks by NGS monument-stability tier (owner taxonomy,
    # 2026-08-22; sources.ngs.opus_stability_tier — decoded from each
    # record's stabilityCode, archive-wide split ~49% A/B vs 50% C/D).
    # The NGS code book is glossed in-figure (owner note 2026-08-22: bare
    # "A/B"/"C/D" is opaque to readers): monument detail in the full-width
    # figure title, panel labels kept short — long labels collide across
    # adjacent panel titles (caught on the LV render). Unknown gets its
    # own panel: rows without a decodable code must stay visible, never
    # silently fall out. Okabe-Ito blue/vermillion = a quality contrast,
    # deliberately not the occupation-class blue ramp.
    "opus_stability": ("OPUS campaign marks (NGS stability code: "
                       "A/B = bedrock/deep-set, expected to hold; "
                       "C/D = surface/shallow, may move)", [
        # _opus_tier is NA for every non-OPUS row, so the tier comparison
        # alone gates A/B and C/D; only the not-coded mask needs the
        # explicit source gate (isna alone would match every other source)
        ("A/B (expected to hold)",
         lambda d: _opus_tier(d) == "A/B", "#0072B2", "o"),
        ("C/D (may move)",
         lambda d: _opus_tier(d) == "C/D", "#D55E00", "o"),
        ("stability not coded",
         lambda d: (d["source"] == "opus") & _opus_tier(d).isna(),
         "#888888", "o"),
    ]),
    "ngs_best": ("NGS monuments, best vertical classes", [
        ("NGS best", None, "monument", "o"),   # mask injected from ngs_best
    ]),
    # FAA NASR runway control, split by published coordinate provenance
    # (raw['pos_class'] from sources/faa.py): the surveyed class is
    # AC 150/5300-18C survey-grade; OWNER/FAA-EST/ADO positions are
    # meters-to-tens-of-meters (LV A/B 2026-08-13: NMAD 0.019 vs 2.78 m)
    # short panel labels: long ones collide on narrow-aspect AOIs (SF);
    # surveyed = 3RD PARTY SURVEY/NGS/MILITARY/ARPTS CONTRACTOR,
    # estimated = OWNER/FAA-EST IMAGERY/ADO/OE-AAA/blank
    "faa": ("FAA runways by position source", [
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if isinstance(aoi, (str, Path)):
        from .aoi import read_aoi
        aoi_gdf = read_aoi(aoi)
    else:
        aoi_gdf = aoi
    if aoi_gdf is not None:
        aoi_gdf = aoi_gdf.to_crs(sampled.crs)
    if overlays is not None:
        overlays = overlays.to_crs(sampled.crs)
    lims = lims or {}   # {family: (map_lim, hist_lim)} override; else empirical
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
            # evaluate each subclass mask ONCE and reuse it for the empty
            # filter, the limit pool, and the map/hist render below —
            # re-running maskfns re-parses raw JSON per call (_opus_tier;
            # Copilot PR #28). Nullable dtypes (string == comparisons)
            # yield Kleene-NA: NA -> False before the bool cast.
            sub_masks = [pd.Series(s[1](sampled)).fillna(False)
                         .to_numpy(dtype=bool) for s in subs]
            # skip empty subclasses: the class taxonomies carry many
            # mutually exclusive subclasses and an n=0 map panel is layout
            # noise (owner empty-panel note + audit round 3)
            keep = [i for i, m in enumerate(sub_masks) if m.any()]
            subs = [subs[i] for i in keep]
            sub_masks = [sub_masks[i] for i in keep]
            # per-PRODUCT: a subclass whose dz is entirely non-finite draws
            # an empty panel (owner 2026-08-30: NGL rows carry NaN dz by
            # design until the per-frame vertical landing) — skip it and
            # say so, never render a blank map
            keep2 = [i for i, m in enumerate(sub_masks)
                     if np.isfinite(sampled.loc[np.asarray(m, dtype=bool),
                                                col].to_numpy(dtype="float64")).any()]
            if len(keep2) < len(subs):
                dropped = [subs[i][0] for i in range(len(subs))
                           if i not in keep2]
                logger.info("family %s/%s: subclass(es) %s have no finite dz "
                            "(e.g. vertically-unassessable rows) — panels "
                            "omitted", fam, prod, dropped)
            subs = [subs[i] for i in keep2]
            sub_masks = [sub_masks[i] for i in keep2]
            if not subs:
                continue
            # empirical, tier-snapped color/hist limits from THIS figure's
            # own dz values (owner 2026-07-16: no hardcoded limits) — the
            # typical spread renders mid-ramp, not saturated
            if fam in lims:
                map_lim, hist_lim = lims[fam]
            else:
                fig_v = np.concatenate([
                    sampled.loc[m, col].to_numpy(dtype="float64")
                    for m in sub_masks]) if subs else []
                map_lim = snap_clim(fig_v, k=3.0)
                hist_lim = max(snap_clim(fig_v, k=6.0), map_lim)
            n_sub = len(subs)
            hs_prod = hs_tif.get(prod) if isinstance(hs_tif, dict) else hs_tif
            # ONE frame for every panel of this figure (all plotted points):
            # per-panel framing rendered side-by-side maps at different
            # scales, and an all-gap panel fell back to the full product
            # (review round 2)
            fig_pts = sampled[np.logical_or.reduce([np.asarray(m, dtype=bool)
                                                     for m in sub_masks])
                              & np.isfinite(sampled[col].to_numpy(dtype="float64"))]
            # settled layout (owner 2026-08-30, matches validation_dz):
            # maps dominate on the left; histogram top-right; the dual-track
            # stats OUTSIDE in their own bottom-right panel. Layout audit
            # 2026-08-30: panels are aspect-true (the fixed 6.8-in canvas +
            # width-clamped columns left wide-AOI maps floating as strips
            # with a taller colorbar beside them), and WIDE panels STACK
            # vertically — n side-by-side 11-in strips made the figure
            # unreadably wide; a tall/square AOI keeps the settled row.
            mw, mh = _map_panel_size(aoi_gdf if aoi_gdf is not None
                                     else fig_pts,
                                     base=6.2, min_in=2.2, max_in=11.0,
                                     max_h=6.0)
            stacked = n_sub > 1 and mh < 0.6 * mw
            # panel titles wrap to the panel width (text never reworded):
            # long subclass names overlapped across narrow tall-AOI panels
            wrap_w = max(16, int(mw * 9))
            tlines = max(len(textwrap.wrap(f"{sub[0]} (n={len(sampled)})",
                                           wrap_w)) for sub in subs)
            ttl = 0.26 * tlines + 0.08         # per-panel title strip (in)
            hist_w, cb_w = 4.2, 0.9
            if stacked:
                maps_w, maps_h = mw, n_sub * (mh + ttl)
            else:
                maps_w = n_sub * mw + 0.25 * (n_sub - 1)
                maps_h = mh + ttl
            sup = 0.4                          # suptitle strip (in)
            fig_h = max(maps_h, 5.4) + sup
            fig_w = maps_w + cb_w + hist_w + 0.9
            fig = plt.figure(figsize=(fig_w, fig_h))
            # the colorbar column is wider than the bar — the slack carries
            # the left-side ticks + label — and the bar is repositioned
            # flush right + clamped to the maps after draw
            gs = fig.add_gridspec(1, 3, width_ratios=[maps_w, cb_w, hist_w],
                                  left=0.015, right=0.99, bottom=0.07,
                                  top=1.0 - (sup + ttl) / fig_h, wspace=0.06)
            if stacked:
                gsm = gs[0, 0].subgridspec(n_sub, 1,
                                           hspace=(ttl + 0.15) / max(mh, 1.0))
                _axm = [fig.add_subplot(gsm[i, 0]) for i in range(n_sub)]
            else:
                gsm = gs[0, 0].subgridspec(1, n_sub, wspace=0.05)
                _axm = [fig.add_subplot(gsm[0, i]) for i in range(n_sub)]
            for a in _axm:      # aspect slack hugs the title, never floats
                a.set_anchor("N")
            cax = fig.add_subplot(gs[0, 1])
            gsr = gs[0, 2].subgridspec(2, 1, height_ratios=[1.0, 0.62],
                                       hspace=0.28)
            axh = fig.add_subplot(gsr[0, 0])
            axt = fig.add_subplot(gsr[1, 0])
            axt.set_axis_off()
            axes = _axm + [axh]
            sc, fam_lines, n_gap = None, [], 0
            for axm, sub, m in zip(axes[:-1], subs, sub_masks):
                lab, _, style, mk = sub[:4]
                _relief(axm, None, hs_prod, None, 0.0, None)
                if overlays is not None:
                    overlays.boundary.plot(ax=axm, color=_INK, lw=0.9, ls="--",
                                           alpha=0.55, zorder=4)
                seg = sampled[m]
                v = seg[col].to_numpy(dtype="float64")
                fin = np.isfinite(v)
                n_gap += int((~fin).sum())
                color = class_ink(style)   # centralized legible ink
                # NEUTRAL point outlines — class colors clash with the dz ramp
                # (owner 2026-07-16); subclass identity = per-map panel title
                sc = axm.scatter(seg.geometry.x[fin], seg.geometry.y[fin],
                                 c=v[fin], cmap=DZ_CMAP, vmin=-map_lim,
                                 vmax=map_lim, s=34, marker=mk,
                                 edgecolors="#404040", linewidths=0.6, zorder=5)
                _finish_map(axm, aoi_gdf, points=fig_pts)
                axm.set_title(textwrap.fill(f"{lab} (n={int(fin.sum())})",
                                            wrap_w), fontsize=10.5,
                              color=_INK)
                if fin.any():
                    vv = v[fin]
                    bw = max(float(_nmad(vv)) / 2.0, hist_lim / 150.0)
                    nb = int(np.clip(round(2 * hist_lim / bw), 21, 161))
                    axh.hist(np.clip(vv, -hist_lim, hist_lim), bins=nb,
                             range=(-hist_lim, hist_lim), histtype="stepfilled",
                             alpha=0.45, color=color, edgecolor=color)
                    # centralized dual-track lines (stats_lines: n first,
                    # owner 2026-08-30)
                    fam_lines.extend((t, c) for t, c, _b
                                     in stats_lines(lab, vv, color))
            if sc is not None:
                cb = fig.colorbar(sc, cax=cax, extend="both")
                # ticks + label LEFT of the bar (the validation-figure
                # convention): the right side faces the histogram column
                cb.ax.yaxis.set_ticks_position("left")
                cb.ax.yaxis.set_label_position("left")
                # dz is relative — same-frame by construction (the
                # transform landed control in the product CRS), so the
                # datum is provenance metadata, not a plot label (owner
                # 2026-08-31)
                cb.set_label(f"dz = {prod} \u2212 control (m)"
                             f"\n[\u00b1{map_lim:g} m tier]",
                             fontsize=9, color=_INK)
                cb.ax.tick_params(labelsize=8, colors=_MUT)
            else:                              # no drawable subclass
                cax.set_axis_off()
            axh.axvline(0, color=_INK, lw=0.8)
            axh.set_xlim(-hist_lim, hist_lim)
            axh.set_xlabel(f"dz = {prod} \u2212 control (m)", fontsize=9,
                           color=_INK)
            # colored stats lines OUTSIDE the histogram, in their own panel
            # (owner 2026-08-30, matching the validation figure)
            xa = (sampled["xform_acc_m"].to_numpy(dtype="float64")
                  if "xform_acc_m" in sampled.columns else np.array([np.nan]))
            if np.isfinite(xa).any():
                b = np.nanmedian(xa)
                if np.isfinite(b):
                    fam_lines.append((f"stated 3D transform budget ±{b:g} m",
                                      _MUT))
            if fam_lines:
                step = min(0.13, 0.96 / len(fam_lines))
                for i, (line, color) in enumerate(fam_lines):
                    axt.text(0.0, 0.98 - step * i, line,
                             transform=axt.transAxes, fontsize=8, va="top",
                             color=color,
                             fontweight="normal"
                             if line.startswith(("   ", "stated"))
                             else "bold")
            axh.tick_params(labelsize=8, colors=_MUT)
            axh.grid(alpha=0.25, lw=0.5)
            gap = f"; {n_gap} unsampled (nodata/gap)" if n_gap else ""
            # left-anchored, matching the validation figure (owner
            # 2026-09-01: dz titles were a mix of centered and left)
            fig.suptitle(f"Vertical difference (m, {prod} minus control) "
                         f"\u2014 {title}{gap}: {site_name}",
                         x=0.01, y=0.995, ha="left", va="top",
                         fontsize=12, color=_INK)
            fp = outdir / f"{site_name}_dz_{fam}_{prod}.png"
            # equal-aspect shrinks the map boxes inside their cells; clamp
            # the colorbar to the union of the DRAWN maps (owner 2026-09-01
            # validation convention) as a fixed 0.28-in bar flush right
            fig.canvas.draw()
            pos = [a.get_position() for a in _axm]
            y0, y1 = min(p.y0 for p in pos), max(p.y1 for p in pos)
            x1m = max(p.x1 for p in pos)
            bw = 0.28 / fig_w
            # anchored to the drawn maps' right edge + tick/label room
            cax.set_position([x1m + 0.85 / fig_w, y0, bw, y1 - y0])
            fig.savefig(fp, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            out.append(fp)
            logger.info("wrote %s", fp)
    return out
