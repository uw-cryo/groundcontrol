"""Area-of-interest resolution: ONE contract for every AOI form.

An AOI reaches the package in one of four shapes and every entry point
(``fetch_control``, the CLIs, the figure helpers) accepts all of them:

- a ``(minlon, minlat, maxlon, maxlat)`` bbox in EPSG:4326;
- a vector file in any OGR-readable format (GeoJSON preferred; GPKG,
  Shapefile, ... — whatever :func:`geopandas.read_file` opens);
- a gridded elevation raster (DEM / DSM / DTM: GeoTIFF, VRT, COG, ...) —
  the AOI is then the raster's GRID EXTENT (:func:`raster_footprint`;
  ``valid=True`` / ``--valid-footprint`` opts into the valid-data
  footprint, so a mosaic with holes fetches control for the covered
  ground only);
- a GeoDataFrame / GeoSeries / shapely geometry already in memory.

:func:`resolve_aoi` reduces any of these to ``(bounds_4326, polygon_4326)``
— sources fetch by bbox, then the dispatcher clips to the polygon.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#: coarse footprint mask: longest raster side is decimated to this many
#: pixels (a 1 m / 60 km mosaic -> ~60 m footprint cells; membership at the
#: AOI edge is not a survey question)
FOOTPRINT_MAX_PX = 1024


#: raster_footprint: above this many polygonized valid patches the exact
#: union is abandoned for the valid-data bounds box — strip/tile mosaics
#: polygonize into thousands of pieces and unary_union goes effectively
#: quadratic (rasuwa 67-strip corridor mosaic: 5+ min CPU, killed;
#: 2026-08-31). Normal footprints are a handful of pieces.
FOOTPRINT_MAX_PIECES = 2000


def _checked_4326(gdf, src_crs, path, src_geom):
    """Reprojected-footprint sanity gate. ``to_crs(4326)`` moves vertices
    independently, so a projected raster crossing the antimeridian or
    containing a pole comes back as a self-intersecting ring wrapping the
    long way around the globe (probed: 200x200 km EPSG:32760 near Fiji ->
    359.994 deg of longitude that EXCLUDES the raster's own centre and
    CONTAINS the Atlantic; same for EPSG:3031 at lon 180 and a raster over
    the South Pole). Every real control point is then dropped and
    unrelated ones kept — fail loud instead of fetching a global AOI.

    Discriminator (round-2 audit): a LEGITIMATELY global projected raster
    (world EPSG:3857/4087) also spans ~360 deg but keeps a valid ring
    that CONTAINS the independently-reprojected raster centre; a true
    wrap is self-intersecting and excludes it. ``src_geom`` = the
    footprint polygon in the source CRS, for that centre."""
    import numpy as np
    import pyproj

    crs_obj = pyproj.CRS(src_crs)
    if crs_obj.is_geographic:
        return gdf  # identity-ish reprojection: no vertex-wise wrap
    b = gdf.total_bounds
    geom = gdf.geometry.iloc[0]
    if (b[2] - b[0]) <= 180.0 and geom.is_valid:
        return gdf
    if geom.is_valid:
        from shapely.geometry import Point as _Point
        # representative_point, not centroid: a centroid can fall OUTSIDE
        # the polygon (disjoint valid stripes, nodata hole at centre) and
        # would falsely refuse a legitimate global raster (round-3 audit)
        c = src_geom.representative_point()
        lon, lat = pyproj.Transformer.from_crs(
            crs_obj, "EPSG:4326", always_xy=True).transform(c.x, c.y)
        if (np.isfinite(lon) and np.isfinite(lat)
                and -90.0 <= lat <= 90.0
                and geom.contains(_Point(lon, lat))):
            return gdf  # valid ring containing its own centre: global
    raise ValueError(
        f"raster {os.fspath(path)}: its EPSG:4326 footprint spans "
        f"{b[2] - b[0]:.2f} deg of longitude"
        + ("" if geom.is_valid else " and is self-intersecting")
        + " — the raster likely crosses the antimeridian or contains "
        "a pole, which vertex-wise reprojection cannot represent. "
        "Pass an explicit AOI instead of the raster footprint "
        "(--aoi with an EPSG:4326 bbox or vector, split at the "
        "antimeridian if needed)")


def _grid_extent_gdf(src, path):
    """The raster's grid extent as an EPSG:4326 GeoDataFrame — the
    transform-mapped corner quadrilateral (correct for rotated grids),
    densified so straight projected edges curve in lon/lat."""
    import geopandas as gpd
    from shapely.geometry import Polygon

    corners = [(0, 0), (src.width, 0), (src.width, src.height),
               (0, src.height)]
    quad = Polygon([src.transform * c for c in corners])
    bx = quad.bounds
    span = max(bx[2] - bx[0], bx[3] - bx[1])
    if span > 0:
        quad = quad.segmentize(span / 200.0)
    gdf = gpd.GeoDataFrame({"source_raster": [os.fspath(path)]},
                           geometry=[quad], crs=src.crs)
    return _checked_4326(gdf.to_crs(4326), src.crs, path, quad)


def _cap_ring_points(poly, tol0: float, max_points):
    """Simplify a footprint polygon until no ring exceeds ``max_points``
    vertices — the ``gdal_footprint -max_points`` behavior (its default is
    100 per ring): Douglas-Peucker at doubling tolerance, starting from
    the decimated pixel size (sub-pixel staircase detail is noise)."""
    if max_points is None:
        return poly
    from shapely.geometry import MultiPolygon

    def rings(g):
        parts = g.geoms if isinstance(g, MultiPolygon) else [g]
        for part in parts:
            yield part.exterior
            yield from part.interiors

    out, tol = poly, tol0
    for _ in range(24):
        if all(len(r.coords) <= max_points for r in rings(out)):
            break   # under the cap already: no simplification at all
        out = poly.simplify(tol, preserve_topology=True)
        tol *= 2.0
    return out if not out.is_empty else poly


def raster_footprint(path, *, max_px: int = FOOTPRINT_MAX_PX,
                     valid: bool = False, simplify: float | None = None,
                     max_points: int | None = 100):
    """Footprint of a raster as a one-row GeoDataFrame in EPSG:4326.

    Default: the GRID EXTENT (transform-mapped corner quadrilateral; no
    mask read — instant on any raster). ``valid=True`` removes nodata and
    polygonizes the VALID pixels instead, following the conventions of
    GDAL's ``gdal_footprint`` utility: ``simplify`` is a Douglas-Peucker
    tolerance in georeferenced units (like ``-simplify``), and
    ``max_points`` caps the vertices per ring (like ``-max_points``,
    same default 100; ``None`` = unlimited) — a staircase mask edge is
    pixel noise, not signal.

    In valid mode, band 1's mask (nodata / alpha / internal mask, per GDAL — the band the
    assessment samples) is read decimated to at most ``max_px`` on the
    longest side (average resampling: for a NODATA-derived mask read at
    full resolution GDAL keeps a coarse cell valid when any source pixel
    is, so small patches survive; for an alpha band or internal mask the
    average rounds a patch below ~1/500 of a cell away; and a raster whose
    overviews were built with ``nearest`` — gdaladdo's default — can drop
    valid patches smaller than the overview stride before this read sees
    them, since the mask comes from the pyramid: ``max_px`` >= the longest
    side forces the full-resolution read) and vectorised; the union is
    densified before
    reprojection so long straight edges of a projected raster follow the
    true graticule. A raster without a nodata tag has an all-valid mask
    (untagged NaN counts as valid), so its footprint is its bounds — set
    the tag (``gdal_edit -a_nodata``) for holes to be excluded. Raises
    ``ValueError`` for a raster with no CRS (an AOI needs one) or no valid
    pixels. A footprint fragmenting into more than
    :data:`FOOTPRINT_MAX_PIECES` patches (strip/tile mosaics) is
    simplified to the valid-data bounds box with a warning — the exact
    union is unbounded-cost there.
    """
    import geopandas as gpd
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.features import shapes
    from rasterio.transform import Affine
    from shapely.geometry import shape
    from shapely.ops import unary_union

    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(
                f"raster {os.fspath(path)} has no CRS; an AOI needs one "
                "(gdal_edit -a_srs, or pass a vector AOI / bbox instead)")
        if not valid:
            # DEFAULT (owner 2026-08-30): the grid extent, no mask read.
            # The AOI only scopes the fetch and frames the figures —
            # points over nodata NaN out at sampling and are reported as
            # gaps, never propagated — so the (potentially full-raster)
            # valid-pixel polygonization is opt-in (valid=True /
            # --valid-footprint) rather than a default cost.
            logger.info("raster footprint %s: grid extent (default; "
                        "valid=True polygonizes the valid pixels)",
                        os.fspath(path))
            return _grid_extent_gdf(src, path)
        f = max(1, math.ceil(max(src.width, src.height) / max_px))
        h, w = math.ceil(src.height / f), math.ceil(src.width / f)
        # untagged raster: the mask is all-valid by definition — the
        # footprint IS the grid extent, skip the mask read
        from rasterio.enums import MaskFlags
        if all(fl == MaskFlags.all_valid for fl in src.mask_flag_enums[0]):
            logger.info("raster footprint %s: no nodata/mask tagged — "
                        "footprint = grid extent (no mask read)",
                        os.fspath(path))
            return _grid_extent_gdf(src, path)
        logger.info("raster footprint %s: reading valid-data mask "
                    "(%dx%d px at 1/%d)%s", os.fspath(path), src.width,
                    src.height, f,
                    "" if f == 1 or src.overviews(1) else
                    " — NO OVERVIEWS: the full raster is read to build "
                    "the decimated mask; gdaladdo to speed this up")
        # average, not nearest: a valid patch smaller than the decimation
        # factor must not vanish (fetch extent errs on the side of coverage)
        mask = src.read_masks(1, out_shape=(h, w), resampling=Resampling.average)
        from_pyramid = f > 1 and bool(src.overviews(1))
        if from_pyramid:
            logger.warning("raster footprint %s read from the overview pyramid (1/%d): "
                           "valid patches smaller than the overview stride may be "
                           "missing if the overviews were built with nearest",
                           os.fspath(path), f)
        # out_shape scales the grid non-uniformly by ceil rounding; recover
        # the exact per-axis pixel size from the shape actually read
        t = src.transform * Affine.scale(src.width / w, src.height / h)
        valid = mask > 0
        if not valid.any():
            hint = (" (mask read from the overview pyramid: retry with max_px >= the "
                    "longest side)" if from_pyramid else "")
            raise ValueError(
                f"raster {os.fspath(path)} has no valid pixels at 1/{f} decimation "
                f"(mask flags {[m.name for m in src.mask_flag_enums[0]]}){hint}; pass a "
                "vector AOI if the raster is not empty")
        geoms = [shape(g) for g, v in shapes(valid.astype("uint8"), mask=valid,
                                             transform=t)]
        if len(geoms) > FOOTPRINT_MAX_PIECES:
            # heavily fragmented mosaic: the exact union is unbounded-cost
            # for a footprint most callers use as a fetch/figure extent —
            # fall back to the valid-data BOUNDS box, loudly (pass a
            # vector AOI when the exact outline matters)
            from shapely.geometry import box as _box
            rows = np.flatnonzero(valid.any(axis=1))
            cols = np.flatnonzero(valid.any(axis=0))
            c0 = t * (int(cols[0]), int(rows[-1]) + 1)
            c1 = t * (int(cols[-1]) + 1, int(rows[0]))
            poly = _box(min(c0[0], c1[0]), min(c0[1], c1[1]),
                        max(c0[0], c1[0]), max(c0[1], c1[1]))
            logger.warning(
                "raster footprint %s: %d disjoint valid patches (> %d) — "
                "simplified to the valid-data bounds box; pass a vector "
                "AOI for an exact footprint", os.fspath(path), len(geoms),
                FOOTPRINT_MAX_PIECES)
        else:
            poly = unary_union(geoms)
            # simplification, gdal_footprint conventions: an explicit
            # -simplify tolerance (georeferenced units) wins; else rings
            # are capped at max_points vertices (gdal_footprint default
            # 100) — a staircase mask edge is pixel noise, not signal
            if simplify is not None:
                poly = poly.simplify(simplify, preserve_topology=True) or poly
            else:
                poly = _cap_ring_points(poly, max(abs(t.a), abs(t.e)),
                                        max_points)
        # densify so straight projected edges curve correctly in lon/lat;
        # tolerance from the geometry itself (affine coefficients are not
        # pixel sizes under rotation: review round 1, 90 deg -> crash)
        bx = poly.bounds
        span = max(bx[2] - bx[0], bx[3] - bx[1])
        if span > 0:
            poly = poly.segmentize(span / 200.0)
        n_ok = int(valid.sum())
        logger.info("raster footprint %s: %d/%d coarse cells valid (1/%d)",
                    os.fspath(path), n_ok, int(np.prod(valid.shape)), f)
        gdf = gpd.GeoDataFrame({"source_raster": [os.fspath(path)]},
                               geometry=[poly], crs=src.crs)
        src_crs = src.crs
    return _checked_4326(gdf.to_crs(4326), src_crs, path, poly)


#: read straight as vector (no raster probe, which logs a GDAL error line)
_VECTOR_SUFFIXES = (".geojson", ".json", ".gpkg", ".shp", ".kml", ".gml", ".fgb",
                    ".gdb", ".parquet", ".geoparquet")


def _read_vector(path):
    import geopandas as gpd

    p = os.fspath(path)
    gdf = (gpd.read_parquet(p) if p.lower().endswith((".parquet", ".geoparquet"))
           else gpd.read_file(p))
    if gdf.crs is None:
        raise ValueError(f"AOI {p} has no CRS")
    return gdf.to_crs(4326)


def read_aoi(path, *, valid_footprint: bool = False):
    """AOI file -> GeoDataFrame in EPSG:4326: a vector file (by suffix, or
    whatever :func:`geopandas.read_file` opens; GeoParquet via
    ``read_parquet``) or a raster via :func:`raster_footprint`. A path
    neither reader opens raises ``ValueError`` carrying both messages."""
    import rasterio
    from rasterio.errors import RasterioIOError

    p = os.fspath(path)
    if p.lower().rstrip("/").endswith(_VECTOR_SUFFIXES):
        try:
            return _read_vector(p)
        except Exception as e:
            raise ValueError(f"{p}: not a readable vector AOI ({e})") from e
    try:
        with rasterio.open(p):
            pass
    except RasterioIOError as raster_err:
        try:
            return _read_vector(p)
        except Exception as e:
            raise ValueError(
                f"{p}: not a readable vector AOI ({e}) nor a raster "
                f"({raster_err})") from e
    return raster_footprint(p, valid=valid_footprint)


def resolve_aoi(aoi):
    """Any AOI form -> ``(bounds_4326, polygon_4326_or_None)``.

    A 4-sequence is a bbox (polygon ``None``: sources already fetch by
    bbox); a path resolves via :func:`read_aoi`; a GeoDataFrame/GeoSeries
    is reprojected and dissolved; a bare shapely geometry is taken as
    already lon/lat. Multiple features dissolve into one (multi)polygon.
    """
    import geopandas as gpd
    import numpy as np
    from shapely.geometry.base import BaseGeometry

    if isinstance(aoi, BaseGeometry):
        return tuple(float(v) for v in aoi.bounds), aoi
    # ndarray included: gdf.total_bounds is the natural way to hand over a
    # bbox and it arrives as numpy.ndarray, not tuple/list
    if (isinstance(aoi, (tuple, list))
            or (isinstance(aoi, np.ndarray) and aoi.ndim == 1)) and len(aoi) == 4:
        try:  # duck-typed like the original dispatcher (numpy scalars, Decimal, str)
            return tuple(float(v) for v in aoi), None
        except (TypeError, ValueError):
            pass  # e.g. four geometries: not a bbox
    if isinstance(aoi, (str, bytes)) or hasattr(aoi, "__fspath__"):
        aoi = read_aoi(Path(os.fsdecode(aoi)) if isinstance(aoi, bytes) else aoi)
    if isinstance(aoi, gpd.GeoSeries):
        aoi = gpd.GeoDataFrame(geometry=aoi)
    if isinstance(aoi, gpd.GeoDataFrame):
        if aoi.crs is None:
            raise ValueError("AOI GeoDataFrame has no CRS")
        aoi4326 = aoi.to_crs(4326)
        poly = aoi4326.union_all()
        if poly.is_empty:
            raise ValueError("AOI has no geometry")
        return tuple(float(v) for v in aoi4326.total_bounds), poly
    raise TypeError(f"unsupported AOI type: {type(aoi)!r}")


def union_footprints(paths, *, valid: bool = False):
    """One EPSG:4326 GeoDataFrame covering every raster in ``paths`` (the
    default AOI of ``groundcontrol-assess`` when none is given). Default:
    grid extents; ``valid=True`` polygonizes each raster's valid pixels,
    nodata removed (costly on large no-overview rasters)."""
    import geopandas as gpd
    import pandas as pd

    parts = [raster_footprint(p, valid=valid) for p in paths]
    if len(parts) == 1:
        return parts[0]
    for part in parts:
        # defensive (raster_footprint gates wrapped footprints): a single
        # invalid part makes GEOS throw an unattributed TopologyException
        # from union_all — name the raster instead
        g = part.geometry.iloc[0]
        if not g.is_valid:
            raise ValueError(
                f"footprint of {part['source_raster'].iloc[0]} is not a "
                "valid polygon; cannot union the product footprints — "
                "pass an explicit --aoi")
    merged = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=4326)
    return gpd.GeoDataFrame({"source_raster": [";".join(merged["source_raster"])]},
                            geometry=[merged.union_all()], crs=4326)
