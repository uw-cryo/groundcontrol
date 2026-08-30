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

#: point_type -> (marker, color, size, zorder, label). GNSS plots BEHIND
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
    "monument": ("+", "#111111", 30, 4, "NGS monument"),
    "gnss_cont": ("*", "#0033A0", 90, 5, "GNSS continuous"),
    "gnss_semicont": ("*", "#3B6FCB", 90, 5, "GNSS semi-continuous"),
    "gnss_campaign": ("*", "#56B4E9", 90, 5, "GNSS campaign"),
    "gnss": ("*", "#888888", 90, 5, "GNSS (pre-split)"),
    "VVA": ("s", "#E69F00", 45, 6, "3DEP VVA"),
    "NVA": ("o", "#C00000", 55, 7, "3DEP NVA"),
    "runway_end": (6, "#1B7837", 55, 6, "FAA runway end"),
    "displaced_threshold": (7, "#66A61E", 50, 6, "FAA displaced threshold"),
    "helipad": (_heliport_marker(), "#1B7837", 110, 6, "FAA helipad"),
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
    logger.info("web basemap %s: z%d%s, %.2f units/px, %dx%d over %s",
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
            # read as a bug) — the point is outside every source's data
            ax.text(0.5, 0.12, "outside data extent", transform=ax.transAxes,
                    ha="center", fontsize=6.5, color="#888888")
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
        for src in owned:
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
        im = ax.imshow(z, cmap=cmap, alpha=dem_alpha, extent=ext,
                       vmin=np.nanpercentile(z, 2),
                       vmax=np.nanpercentile(z, 98), interpolation="antialiased", interpolation_stage="rgba")
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


def control_map_figure(ctl, aoi_p, outdir, site_name, *, dem_tif=None,
                       hs_tif=None, cmap=None, dem_alpha=0.4,
                       clip_to_aoi=True, label_points=True,
                       label_gnss_ids=False, fname=None, title=None, dpi=200):
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
    fig, ax = plt.subplots(figsize=(10.5, 10))
    _relief(ax, dem_tif, hs_tif, cmap, dem_alpha, fig)

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
    ax.set_title(title or f"{site_name} — control points (n={len(ctl)})",
                 fontsize=11, color=_INK)
    fig.tight_layout()
    fp = outdir / (fname or f"{site_name}_control_map.png")
    fig.savefig(fp, dpi=dpi)
    plt.close(fig)
    return fp


def standard_control_figures(control, aoi, outdir, site_name, *,
                             dem_tif=None, hs_tif=None, cmap=None,
                             dem_alpha=0.4, midas_frame="IGS14",
                             midas_velocities=True,
                             buffer_km=60.0, clip_to_aoi=True,
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
        label_points=label_points, dpi=dpi))

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
            title=f"{site_name} — {dname} control points (n={len(sub)})",
            dpi=dpi))

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
            _finish_map(ax, aoi_p, clip_to_aoi)
            ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9)
            ax.set_title(f"NGS monuments by {key}", fontsize=10, color=_INK)
        fig.suptitle(f"{site_name} — NGS monument datasheet attributes "
                     f"(n={len(mon)})", fontsize=11.5, color=_INK)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fp = outdir / "ngs" / f"{site_name}_monument_types.png"
        fig.savefig(fp, dpi=dpi)
        plt.close(fig)
        out.append(fp)

    # ---- 3+4. MIDAS motion figures (network fetch: gated) -------------------
    if not midas_velocities or aoi_gdf is None:
        logger.info("MIDAS velocity figures skipped (midas_velocities=%s, "
                    "aoi=%s)", midas_velocities, aoi_gdf is not None)
        return out
    try:
        from .plot import plot_velocity_vectors
        from .sources.ngl import read_midas
        st = read_midas(midas_frame)
        b = aoi_gdf.to_crs(4326).total_bounds
        st = st[st.lon.between(b[0] - 3, b[2] + 3)
                & st.lat.between(b[1] - 3, b[3] + 3)]
        # ONE combined figure (owner 2026-08-30): horizontal quiver |
        # vertical-colored, over the DEM hillshade (dem_tif warps + shades
        # in plot_velocity_vectors; a pre-rendered PATH hs_tif also works,
        # but the in-memory auto-hillshade tuple cannot be reprojected)
        fig2, (axh_, axv_) = plt.subplots(1, 2, figsize=(18, 9))
        hs_path = hs_tif if isinstance(hs_tif, (str, Path)) else None
        for ax_, cbv in ((axh_, False), (axv_, True)):
            plot_velocity_vectors(
                st, aoi=aoi_gdf, buffer_km=buffer_km, ax=ax_,
                color_by_vertical=cbv, hs_tif=hs_path, dem_tif=dem_tif,
                title=("vertical-colored" if cbv else "horizontal"))
        fig2.suptitle(f"{site_name} — MIDAS ({midas_frame}) velocity field",
                      fontsize=13, color=_INK)
        fp = outdir / f"{site_name}_midas_velocity.png"
        fig2.savefig(fp, dpi=dpi, bbox_inches="tight")
        plt.close(fig2)
        out.append(fp)
    except Exception as exc:  # network etc. — the map figures still ship
        logger.warning("MIDAS velocity figures skipped: %s", exc)

    logger.info("standard control figures: %s", [p.name for p in out])
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
    "GNSS campaign (NGL)": "#7BA3CF",
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
        # sources never fit inside a histogram box).
        map_h = 7.4
        title_cb = 0.55                       # title strip above the map
        asp_w = _aspect_panel_w(aoi, map_h - title_cb, lo=0.4, hi=2.0)
        mcol = asp_w + 1.15                   # + colorbar column
        hist_w = 4.2
        fig = plt.figure(figsize=(mcol + hist_w, map_h))
        gs = fig.add_gridspec(3, 2, width_ratios=[mcol, hist_w],
                              height_ratios=[1.0, 1.0, 0.62],
                              hspace=0.3, wspace=0.08)
        ax_map = fig.add_subplot(gs[:, 0])
        ax_s = fig.add_subplot(gs[0, 1])
        ax_n = fig.add_subplot(gs[1, 1], sharex=ax_s)
        ax_t = fig.add_subplot(gs[2, 1])
        ax_t.set_axis_off()
        axes = [ax_map, ax_s, ax_n]
        hs_prod = hs_tif.get(prod) if isinstance(hs_tif, dict) else hs_tif
        _relief(axes[0], None, hs_prod, None, 0.0, None)
        use = sampled[np.isfinite(sampled[col])]
        pl = point_lim if point_lim is not None else snap_clim(use[col], k=3.0)
        # marker SHAPE carries class identity (owner 2026-08-30: identical
        # circles hid which points were NVA vs GNSS vs monuments vs FAA);
        # color stays the dz ramp (class colors clash with it, owner
        # 2026-07-16), POINT_STYLE shapes match the control map and sheets
        import matplotlib as _mpl
        from matplotlib.lines import Line2D
        norm = _mpl.colors.Normalize(vmin=-pl, vmax=pl)
        handles = []
        if "point_type" in use.columns and use["point_type"].notna().any():
            pts_order = [t for t in LEGEND_ORDER
                         if (use["point_type"] == t).any()]
            pts_order += [t for t in use["point_type"].dropna().unique()
                          if t not in pts_order]
            for pt in pts_order:
                mk, _, msz, _, mlab = POINT_STYLE.get(
                    pt, ("o", "#888888", 34, 5, str(pt)))
                sub = use[use["point_type"] == pt]
                lw = 1.0 if mk in ("+", "x") else 0.35
                axes[0].scatter(sub.geometry.x, sub.geometry.y, c=sub[col],
                                cmap=DZ_CMAP, norm=norm, marker=mk,
                                s=max(15, int(msz * 0.32)),
                                edgecolors="#333333", linewidths=lw, zorder=5)
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
        cb = fig.colorbar(sc, ax=axes[0], shrink=0.75, pad=0.02, extend="both")
        cb.set_label(f"dz = {prod} − control (m)", fontsize=9, color=_INK)
        cb.ax.tick_params(labelsize=8, colors=_MUT)
        _finish_map(axes[0], aoi, points=use)
        axes[0].set_title(f"{site_name} {prod} − control  (n={len(use)})",
                          fontsize=11, color=_INK)

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
                key = seg_defs[lab][1]  # POINT_STYLE key or raw hex
                color = POINT_STYLE[key][1] if key in POINT_STYLE else key
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
            step = min(0.115, 0.96 / len(txt_lines))
            for i, (line, color, bold) in enumerate(txt_lines):
                ax_t.text(0.0, 0.98 - step * i, line, transform=ax_t.transAxes,
                          fontsize=7.5, va="top", color=color,
                          fontweight="bold" if bold else "normal")
        ax_s.set_title("survey-grade points", fontsize=10, color=_INK)
        ax_n.set_title(f"NGS monuments ({ngs_nmad_gate:.0f}-NMAD filtered)",
                       fontsize=10, color=_INK)
        fp = outdir / f"{site_name}_validation_dz_{prod}.png"
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
    "3dep": ("3DEP CHECKPOINTS", [
        ("NVA", lambda d: (d["source"] == "3dep") & (d["point_type"] == "NVA"),
         "NVA", "o"),
        ("VVA", lambda d: (d["source"] == "3dep") & (d["point_type"] == "VVA"),
         "VVA", "s", ("DTM",)),
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
    "gnss": ("GNSS CONTROL (by occupation class)", [
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
    "opus_stability": ("OPUS CAMPAIGN MARKS (NGS stability code: "
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
            # aspect-aware map columns (fill the axes; kill the map->colorbar
            # gap on tall-narrow AOIs); the shared-colorbar allowance rides on
            # the map columns in BOTH figsize and width_ratios so the inch
            # widths stay literal
            # settled layout (owner 2026-08-30, matches validation_dz):
            # maps dominate and span both rows; histogram top-right; the
            # dual-track stats OUTSIDE in their own bottom-right panel
            map_h = 6.8
            mcol = _aspect_panel_w(aoi_gdf, map_h - 0.6, lo=0.4, hi=2.0) \
                + 0.9 / n_sub
            hist_w = 4.2
            fig = plt.figure(figsize=(mcol * n_sub + hist_w, map_h))
            gs = fig.add_gridspec(2, n_sub + 1,
                                  width_ratios=[mcol] * n_sub + [hist_w],
                                  height_ratios=[1.0, 0.55],
                                  hspace=0.22, wspace=0.1)
            _axm = [fig.add_subplot(gs[:, i]) for i in range(n_sub)]
            axh = fig.add_subplot(gs[0, n_sub])
            axt = fig.add_subplot(gs[1, n_sub])
            axt.set_axis_off()
            axes = _axm + [axh]
            hs_prod = hs_tif.get(prod) if isinstance(hs_tif, dict) else hs_tif
            # ONE frame for every panel of this figure (all plotted points):
            # per-panel framing rendered side-by-side maps at different
            # scales, and an all-gap panel fell back to the full product
            # (review round 2)
            fig_pts = sampled[np.logical_or.reduce([np.asarray(m, dtype=bool)
                                                     for m in sub_masks])
                              & np.isfinite(sampled[col].to_numpy(dtype="float64"))]
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
                color = POINT_STYLE[style][1] if style in POINT_STYLE else style
                # NEUTRAL point outlines — class colors clash with the dz ramp
                # (owner 2026-07-16); subclass identity = per-map panel title
                sc = axm.scatter(seg.geometry.x[fin], seg.geometry.y[fin],
                                 c=v[fin], cmap=DZ_CMAP, vmin=-map_lim,
                                 vmax=map_lim, s=52, marker=mk,
                                 edgecolors="#404040", linewidths=0.8, zorder=5)
                _finish_map(axm, aoi_gdf, points=fig_pts)
                axm.set_title(f"{lab} (n={int(fin.sum())})", fontsize=10.5,
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
            fig.suptitle(f"{site_name} {prod} \u2212 control \u2014 {title}{gap}",
                         fontsize=11.5, color=_INK)
            fp = outdir / f"{site_name}_dz_{fam}_{prod}.png"
            fig.savefig(fp, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            out.append(fp)
            logger.info("wrote %s", fp)
    return out
