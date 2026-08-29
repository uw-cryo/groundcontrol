"""Console-script entry points (plan: CLI wrapper scripts).

Entry points live inside the package (a console script cannot live outside
``src/``). ``groundcontrol-assess`` arrives with Increment 2.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


def fetch_control_main(argv=None) -> int:
    """``groundcontrol-fetch`` — AOI -> control GeoParquet/CSV + provenance."""
    p = argparse.ArgumentParser(
        prog="groundcontrol-fetch",
        description="Fetch ground control points for an AOI and export them "
                    "with transform provenance.",
    )
    p.add_argument("--aoi", required=True,
                   help="AOI: 'minx,miny,maxx,maxy' bbox in EPSG:4326 lon/lat "
                        "(use --aoi=-112,32.6,... for negative longitudes), a "
                        "vector file (GeoJSON preferred; any OGR format), or a "
                        "DEM/DSM/DTM raster whose valid-data footprint becomes "
                        "the AOI")
    p.add_argument("--sources", default="3dep,ngs,opus",
                   help="comma-separated sources (default: 3dep,ngs,opus; also "
                        "ngl, faa)")
    p.add_argument("--out", required=True, help="output path (.parquet or .csv)")
    p.add_argument("--target-crs", default=None,
                   help="target 3D CRS (NOT YET IMPLEMENTED — interim landing is "
                        "EPSG:6318 + NAVD88; passing this raises)")
    p.add_argument("--target-epoch", type=float, default=None,
                   help="target coordinate epoch, decimal year (NOT YET IMPLEMENTED)")
    args = p.parse_args(argv)

    from groundcontrol import io
    from groundcontrol.sources import fetch_control

    aoi = _parse_aoi(args.aoi)
    sources = tuple(s.strip() for s in args.sources.split(",") if s.strip())
    # Preflight before the fetch (see io.check_export_support): a failure
    # here costs nothing; after the fetch it costs the points. The AOI is
    # READ last: a raster footprint (or a remote file) is the one costly
    # input, and an --out typo should not pay for it.
    out = _preflight(io.check_export_support, args.out)
    _check_sources(sources)
    aoi = _load_aoi(aoi)

    gdf, status = fetch_control(aoi, sources=sources,
                                target_crs=args.target_crs, target_epoch=args.target_epoch)
    for name, s in status.items():
        line = f"  {name:6s} {s['n_rows']:6d} rows"
        if s["error"]:
            line += f"  ERROR: {s['error']}"
        print(line, file=sys.stderr)
    if not len(gdf):
        print("no control points fetched from any source", file=sys.stderr)
        return 1
    io.write(gdf, out, status=status,
             command="groundcontrol-fetch " + " ".join(argv or sys.argv[1:]))
    print(f"wrote {out} ({len(gdf)} points) + provenance sidecar", file=sys.stderr)
    return 0


def _parse_kv(pairs, flag):
    out = {}
    for item in pairs or []:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            raise SystemExit(f"error: {flag} expects NAME=PATH, got {item!r}")
        if name in out:
            raise SystemExit(
                f"error: {flag} got NAME {name!r} twice ({out[name]!r} and {path!r}); "
                "a repeat would silently drop the first")
        out[name] = path
    return out


def _preflight(check, *args, **kwargs):
    """Run a pre-fetch check; report a failure as a clean CLI error, not a traceback."""
    try:
        return check(*args, **kwargs)
    except (ValueError, ImportError, OSError) as e:
        raise SystemExit(f"error: {e}") from e


_AOI_FILE_SUFFIXES = (".geojson", ".json", ".gpkg", ".shp", ".parquet")

#: an extension-shaped tail (".tif", ".gpkg"); a bbox typo like "33.0x" or
#: "4.5.6" is not one and must keep the float-parse diagnostic (round 3)
_PATHY_TAIL = re.compile(r"\.[^\W\d_]\w{0,15}$")


# GDAL/OGR driver-prefixed connection strings and subdataset specs that are
# not local paths to probe (left to GDAL, like URLs and /vsi paths).
_GDAL_PREFIX = re.compile(
    r"^(PG|PGB|MySQL|MSSQL|OCI|ODBC|WFS|WMS|WCS|NETCDF|HDF4|HDF5|ZARR|GRIB|"
    r"SENTINEL2|DERIVED_SUBDATASET|GTIFF_DIR|NITF_IM|RASTERLITE|GPKG|SQLite|"
    r"MEM|TILEDB|PDS4|ISIS3|EEDAI)(_[A-Za-z0-9]+)*:",  # HDF4_SDS:, SENTINEL2_L1C:, ...
    re.IGNORECASE)


def _is_remote(path: str) -> bool:
    """URL, GDAL /vsi path, or a known driver-prefixed string (``PG:...``,
    ``NETCDF:"f.nc":var``): left to GDAL rather than probed as a local file.
    A local file that happens to carry such a name still counts as local."""
    if "://" in path or path.startswith("/vsi"):
        return True
    if not _GDAL_PREFIX.match(path):
        return False
    try:
        return not Path(path).exists()
    except OSError:  # ENAMETOOLONG etc.: not a local path either way
        return True


def _parse_aoi(spec):
    """--aoi: 'minx,miny,maxx,maxy' (EPSG:4326 lon/lat) -> tuple; otherwise a
    vector-file path, which must exist (URLs and /vsi paths are left to GDAL)."""
    import math
    if "," in spec and not spec.lower().endswith(_AOI_FILE_SUFFIXES):
        try:
            vals = tuple(float(v) for v in spec.split(","))
        except ValueError as e:
            # a path with a comma in it (site,2024/dem.tif, ~/x,y/a.tif, a
            # remote a,b.tif): anything path-shaped goes to the path branch
            # (which reports "file not found" for a typo, not a bbox error)
            if (_is_remote(spec) or "/" in spec or os.sep in spec
                    or _PATHY_TAIL.search(os.path.basename(spec))
                    or os.path.exists(os.path.expanduser(spec))):
                return _parse_aoi_path(spec)
            raise SystemExit(f"error: --aoi: expected minx,miny,maxx,maxy, got {spec!r} "
                             f"({e})") from e
        if len(vals) != 4:
            raise SystemExit(f"error: --aoi: expected 4 numbers, got {len(vals)}: {spec!r}")
        if not all(math.isfinite(v) for v in vals):
            raise SystemExit(f"error: --aoi: bbox values must be finite: {spec!r}")
        if not (vals[0] < vals[2] and vals[1] < vals[3]):
            raise SystemExit(f"error: --aoi: expected minx<maxx and miny<maxy: {spec!r}")
        return vals
    if not spec:
        raise SystemExit("error: --aoi: empty")
    return _parse_aoi_path(spec)


def _parse_aoi_path(spec):
    if not _is_remote(spec):
        from groundcontrol.io import expand_user_path
        path = _preflight(expand_user_path, spec)
        try:
            found = path.exists()  # a directory datasource (shapefile dir, .gdb) is fine
        except OSError as e:  # ENAMETOOLONG etc.
            raise SystemExit(f"error: --aoi: not a usable path: {spec[:60]}... ({e})") from e
        if not found:
            raise SystemExit(f"error: --aoi: file not found: {spec}")
        return str(path)
    return spec


def _load_aoi(aoi):
    """A parsed --aoi -> what fetch_control/figures consume: the bbox tuple
    as-is, else the file read up front (vector -> its features; DEM/DSM/DTM
    raster -> its valid-data footprint; both EPSG:4326) so an unreadable
    file fails here, before the fetch, naming the flag."""
    if isinstance(aoi, tuple):
        return aoi
    from groundcontrol.aoi import read_aoi
    try:
        return read_aoi(aoi)
    except (ValueError, OSError) as e:  # both readers failed, or no CRS / no data
        raise SystemExit(f"error: --aoi {e}") from e


def _default_site_name(products):
    """--site-name default: the first product's file stem (BYOD run of
    ``--product DEM=site_dsm.tif`` -> ``site_dsm_*`` artifacts)."""
    stem = Path(next(iter(products.values()))).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_.") or "site"


def _embedded_target_crs(products):
    """--target-crs default: the products' own CRS, accepted ONLY when it
    declares the height datum (compound or 3D) and every product agrees;
    a 2D raster CRS says nothing about heights and is refused, naming what
    to pass (fail-loud: never infer a vertical datum)."""
    import pyproj
    import rasterio

    seen = {}
    for name, path in products.items():
        with rasterio.open(path) as src:
            crs = src.crs
        if crs is None:
            raise SystemExit(f"error: --target-crs is required: product {name}={path} "
                             "has no CRS")
        crs = pyproj.CRS.from_user_input(crs)
        from groundcontrol.assess import has_vertical_axis
        if not has_vertical_axis(crs):
            raise SystemExit(
                f"error: --target-crs is required: product {name}={path} declares a CRS "
                f"without a height axis ({crs.name}), which says nothing about its height "
                "datum. Pass the "
                "product's 3D CRS, e.g. '<horizontal EPSG>+5703' for NAVD88 heights on "
                "that horizontal (EPSG:6341+5703 = NAD83(2011) UTM 12N + NAVD88), or a "
                ".wkt file with a vertical member (groundcontrol.geodesy.build_utm_* "
                "constructs ellipsoidal-height UTM frames)")
        seen[name] = crs
    first = next(iter(seen.values()))
    for name, crs in seen.items():
        if not crs.equals(first):
            names = {n: c.name for n, c in seen.items()}
            raise SystemExit("error: --target-crs is required: products declare "
                             f"different CRSs {names}")
    print(f"target CRS from the product's embedded CRS: {first.name}", file=sys.stderr)
    return first.to_wkt()


def _resolve_crs(spec, flag):
    """A --target-crs/--source-crs value: EPSG code/authority string, WKT, or a
    .wkt/.prj file (a file that cannot be read is reported naming the flag)."""
    from pathlib import Path
    p = Path(spec)
    try:
        if p.suffix.lower() in (".wkt", ".prj"):
            from groundcontrol.io import expand_user_path
            return expand_user_path(p).read_text()
        try:
            is_file = p.is_file()
        except OSError:  # ENAMETOOLONG: an inline WKT string is not a path
            is_file = False
        return p.read_text() if is_file else spec
    except (OSError, ValueError) as e:  # missing/binary/directory "file"
        raise ValueError(f"{flag} {spec}: cannot read as a CRS file ({e})") from e


def _validate_crs(spec, flag):
    """A CRS typo (EPSG:99999, '') must fail before the fetch, not after it."""
    import pyproj
    try:
        pyproj.CRS.from_user_input(spec)
    except pyproj.exceptions.CRSError as e:
        raise SystemExit(f"error: {flag}: not a valid CRS ({e})") from e


def _check_rasters(paths, flag):
    """Open each raster (local path, URL or /vsi path alike) and, for a local
    VRT/mosaic, confirm the files it references exist -- so a typo'd or
    renamed DEM fails before the fetch, not after it. Returns the mapping
    with ``~`` expanded in local paths."""
    import os

    import rasterio
    from groundcontrol.io import expand_user_path
    out = {}
    for name, path in paths.items():
        if not _is_remote(str(path)):
            path = str(_preflight(expand_user_path, path))
        try:
            with rasterio.open(path) as ds:
                files = list(ds.files or [])
        except Exception as e:  # rasterio/GDAL raise several unrelated hierarchies
            raise SystemExit(f"error: {flag} {name}={path}: cannot open raster "
                             f"({type(e).__name__}: {e})") from e
        # os.path.exists (not Path.exists): an over-long resolved source path
        # must read as "missing", not raise ENAMETOOLONG past _preflight.
        missing = [f for f in files if not _is_remote(f) and not os.path.exists(f)]
        if missing:
            raise SystemExit(f"error: {flag} {name}={path}: references {len(missing)} "
                             f"missing file(s), first: {missing[0]}")
        out[name] = path
    return out


def _check_sources(sources):
    """A typo'd --sources name is not a source failure: the dispatcher would
    degrade it to a status-line error and write a product silently missing
    that source. And the 3DEP checkpoint source reads a parquet file, so
    with a broken engine it fails after the network round-trip with
    geopandas' bare message -- check both up front, whatever the output
    format."""
    from groundcontrol import io
    from groundcontrol.sources import PROVIDERS
    unknown = [s for s in sources if s not in PROVIDERS]
    if unknown:
        raise SystemExit(f"error: --sources: unknown {unknown}; "
                         f"available: {sorted(PROVIDERS)}")
    if "3dep" in sources:
        _preflight(io.check_parquet_engine, "the 3DEP checkpoint source (--sources 3dep)")


def _check_control_cache(cache):
    """Preflight the control cache (``--control`` or the default under
    ``--outdir``): an existing cache is read by content whatever its name
    (quarantine renames like ``*_STALE_*`` stay usable) but must be a single
    file; a new one is written as GeoParquet and read back on later runs, so
    it must be .parquet. The output-directory checks for a new cache come
    after --outdir is created (see assess_dem_main)."""
    from groundcontrol import io
    if cache.is_dir():  # pyarrow would read it as a partitioned dataset: silent wrong numbers
        raise ValueError(f"control cache {cache}: is a directory (must be a single "
                         "GeoParquet file)")
    if not cache.exists() and cache.suffix.lower() != ".parquet":
        raise ValueError(
            f"control cache {cache}: a new cache must be .parquet "
            "(it is read back as GeoParquet on later runs)")
    io.check_parquet_engine(cache)  # read or write, the cache is parquet


def _read_cache(cache):
    import geopandas as gpd
    try:
        return gpd.read_parquet(cache)
    except MemoryError:
        raise  # a valid but large cache is not "unreadable"
    except Exception as e:  # pyarrow raises ValueError *and* RuntimeError subclasses
        raise ValueError(f"control cache {cache}: not a readable GeoParquet file "
                         f"({type(e).__name__}: {e})") from e


def _make_outdir(outdir):
    import os
    if os.path.lexists(outdir) and not outdir.is_dir():  # file or dangling symlink
        raise ValueError(f"--outdir {outdir}: exists and is not a directory")
    try:
        outdir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"--outdir {outdir}: cannot create ({e})") from e


def assess_dem_main(argv=None) -> int:
    """``groundcontrol-assess`` — DEM(+AOI) -> fetch -> sample -> stats + figures."""
    p = argparse.ArgumentParser(
        prog="groundcontrol-assess",
        description="Assess DEM products against fetched/cached ground control: "
                    "transform control into the product frame, sample, and write "
                    "dz stats + standard validation figures.",
    )
    p.add_argument("--product", action="append", required=True,
                   help="NAME=PATH gridded elevation raster to assess (repeatable; "
                        "any GDAL format: GeoTIFF/COG/VRT/...). A NAME containing "
                        "'DTM' is assessed under the bare-earth rules (VVA "
                        "checkpoints apply); any other name (DSM, DEM, ...) under "
                        "the surface rules")
    p.add_argument("--aoi", default=None,
                   help="AOI: 'minx,miny,maxx,maxy' bbox (EPSG:4326 lon/lat), a vector "
                        "file (GeoJSON preferred; any OGR format), or a raster whose "
                        "valid-data footprint is used. Default: the union of the "
                        "--product footprints")
    p.add_argument("--target-crs", default=None,
                   help="product 3D CRS: EPSG/authority string ('EPSG:6341+5703'), WKT, "
                        "or a .wkt file. Default: the product's own embedded CRS, "
                        "accepted only when it is compound/3D (declares the height "
                        "datum); a 2D raster CRS is refused, never guessed")
    p.add_argument("--target-epoch", type=float, default=2010.0,
                   help="transform-time epoch tt, decimal year (default 2010.0; "
                        "inert for static-frame targets)")
    p.add_argument("--source-crs", default=None,
                   help="override the control landing CRS (default EPSG:6318+5703, "
                        "the fetch_control contract) — expert use, e.g. a cache "
                        "already in another frame")
    p.add_argument("--control", default=None,
                   help="control GeoParquet cache: reused when present, else fetched "
                        "from --sources and written here (default: <outdir>/<site-name>_control.parquet)")
    p.add_argument("--sources", default="3dep,ngs,opus",
                   help="comma-separated fetch sources (default: 3dep,ngs,opus)")
    p.add_argument("--outdir", required=True, help="output directory")
    p.add_argument("--site-name", default=None,
                   help="prefix for output artifacts (default: the first product's "
                        "file stem)")
    p.add_argument("--method", default="linear", choices=("linear", "nearest"),
                   help="point sampling method (default: linear)")
    p.add_argument("--radius", type=float, default=None,
                   help="neighborhood-median sampling radius in raster units "
                        "(mutually exclusive with --method)")
    p.add_argument("--hs", action="append", default=None,
                   help="NAME=PATH pre-rendered hillshade underlay for figures "
                        "(repeatable, product-matched; single unnamed path also "
                        "accepted). Default: a hillshade computed from each product")
    p.add_argument("--no-figures", action="store_true", help="skip figure output")
    p.add_argument("--point-lim", type=float, default=None,
                   help="pin the validation-figure map color limit (m); default "
                        "empirical tier-snapped from the plotted dz (issue #23)")
    p.add_argument("--vendor-lim", type=float, default=None,
                   help="pin the survey-grade histogram limit (m); default empirical")
    p.add_argument("--wide-lim", type=float, default=None,
                   help="pin the NGS-monument histogram limit (m); default empirical")
    args = p.parse_args(argv)
    if args.radius is not None and args.method != p.get_default("method"):
        p.error("--radius and --method are mutually exclusive (radius mode "
                "computes a neighborhood median)")

    from groundcontrol import io

    # Validate everything cheap before the (network) fetch -- the AOI file,
    # NAME=PATH pairs, CRS strings/files, the parquet engine, the control
    # cache, then the rasters (may touch the network for remote ones), then
    # the output directory -- so a typo fails here, not after the points
    # are fetched.
    aoi = _parse_aoi(args.aoi) if args.aoi is not None else None
    products = _parse_kv(args.product, "--product")
    hs = None
    if args.hs:
        if len(args.hs) == 1 and "=" not in args.hs[0]:
            hs = args.hs[0]
        else:
            hs = _parse_kv(args.hs, "--hs")
    target_crs = None
    if args.target_crs is not None:
        target_crs = _preflight(_resolve_crs, args.target_crs, "--target-crs")
        _validate_crs(target_crs, "--target-crs")
    source_crs = None
    if args.source_crs is not None:
        source_crs = _preflight(_resolve_crs, args.source_crs, "--source-crs")
        _validate_crs(source_crs, "--source-crs")
    sources = tuple(s.strip() for s in args.sources.split(",") if s.strip())

    outdir = _preflight(io.expand_user_path, args.outdir)
    site_name = args.site_name or _default_site_name(products)
    cache = (_preflight(io.expand_user_path, args.control) if args.control
             else outdir / f"{site_name}_control.parquet")
    _preflight(_check_control_cache, cache)
    _check_sources(sources)  # also on the cache path: a typo'd name must not become a warning
    products = _check_rasters(products, "--product")
    if target_crs is None:
        target_crs = _embedded_target_crs(products)
    if isinstance(hs, str):
        hs = _check_rasters({"hillshade": hs}, "--hs")["hillshade"]
    elif hs:
        hs = _check_rasters(hs, "--hs")
    control = _preflight(_read_cache, cache) if cache.exists() else None
    # Only now create --outdir: every INPUT check above is side-effect free,
    # so a rejected input leaves nothing behind (an output-side rejection
    # below can leave an empty --outdir; the export checks need it to exist).
    _preflight(_make_outdir, outdir)
    if control is None:
        _preflight(io.check_export_support, cache)
    _preflight(io.check_export_support, outdir / f"{site_name}_assessed.parquet")
    _preflight(io.check_export_support, outdir / f"{site_name}_dz_stats.csv",
               sidecar=False)
    # The AOI is resolved LAST (after every cheap check): a product footprint
    # reads every product's mask, the one costly input step.
    if aoi is None:
        from groundcontrol.aoi import union_footprints
        aoi = _preflight(union_footprints, list(products.values()))
    else:
        aoi = _load_aoi(aoi)
    if isinstance(aoi, tuple):  # bbox: fetch by bounds, but clip/outline the maps to it
        import geopandas as gpd
        from shapely.geometry import box
        aoi_fig = gpd.GeoDataFrame(geometry=[box(*aoi)], crs=4326)
    else:
        aoi_fig = aoi

    if control is not None:
        print(f"control cache: {cache} ({len(control)} points)", file=sys.stderr)
        have = (set(control["source"].dropna().unique())
                if "source" in control.columns else set())
        if set(sources) - have:
            print(f"warning: control cache lacks requested source(s) "
                  f"{sorted(set(sources) - have)} (cache has {sorted(have)}); "
                  f"delete {cache} to re-fetch", file=sys.stderr)
    else:
        from groundcontrol.sources import fetch_control
        control, status = fetch_control(aoi, sources=sources)
        for name, s in status.items():
            line = f"  {name:6s} {s['n_rows']:6d} rows"
            if s["error"]:
                line += f"  ERROR: {s['error']}"
            print(line, file=sys.stderr)
        if not len(control):
            print("no control points fetched from any source", file=sys.stderr)
            return 1
        io.write(control, cache, status=status,
                 command="groundcontrol-assess " + " ".join(argv or sys.argv[1:]))
        print(f"wrote control cache {cache} ({len(control)} points)", file=sys.stderr)

    from groundcontrol.assess import assess_products  # ~0.5 s; after the preflight

    sampled, stats, artifacts = assess_products(
        control, products, target_crs,
        outdir=outdir, site_name=site_name, aoi=aoi_fig,
        hs=hs, target_epoch=args.target_epoch, method=args.method,
        radius=args.radius, source_crs=source_crs, figures=not args.no_figures,
        point_lim=args.point_lim, vendor_lim=args.vendor_lim,
        wide_lim=args.wide_lim,
        command="groundcontrol-assess " + " ".join(argv or sys.argv[1:]))

    t = artifacts["transform"]
    a = t["accuracy_m"]
    # PROJ reports 0/-1 for defining ties — print "unknown", never a fake-exact
    acc_note = f"{a} m" if (a is not None and a > 0) else "unknown"
    print(f"transform: {t['description']} (stated accuracy {acc_note})", file=sys.stderr)
    with_stats = stats[stats.segment != "ALL"] if len(stats) else stats
    print(with_stats.to_string(index=False), file=sys.stderr)
    for k, v in artifacts.items():
        if k != "transform":
            print(f"wrote {v}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(fetch_control_main())
