"""Console-script entry points (plan: CLI wrapper scripts).

Entry points live inside the package (a console script cannot live outside
``src/``). ``groundcontrol-assess`` arrives with Increment 2.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path


def _setup_logging(quiet: bool = False) -> None:
    """Route the package's own INFO logs to stderr (owner 2026-08-30: a
    large run sat silent for minutes while the footprint/fetch worked —
    the pipeline narrates itself at INFO, but nothing configured a
    handler). Third-party loggers (botocore, rasterio) stay untouched."""
    lg = logging.getLogger("groundcontrol")
    if not lg.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(levelname).1s %(name)s: %(message)s"))
        lg.addHandler(h)
    lg.setLevel(logging.WARNING if quiet else logging.INFO)
    lg.propagate = False


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
                        "DEM/DSM/DTM raster whose grid extent becomes the AOI "
                        "(--valid-footprint removes nodata, polygon around valid pixels)")
    p.add_argument("--sources", default="3dep,ngs,opus,ngl,faa",
                   help="comma-separated sources (default: every provider — "
                        "3dep,ngs,opus,ngl,faa; owner 2026-08-30: NGL is "
                        "first-class, not opt-in)")
    p.add_argument("--out", required=True, help="output path (.parquet or .csv)")
    p.add_argument("--target-crs", default=None,
                   help="target 3D CRS (NOT YET IMPLEMENTED — interim landing is "
                        "EPSG:6318 + NAVD88; passing this raises)")
    p.add_argument("--target-epoch", type=float, default=None,
                   help="target coordinate epoch, decimal year (NOT YET IMPLEMENTED)")
    p.add_argument("--landing-crs", default=None,
                   help="override the interim horizontal landing frame (default "
                        "EPSG:6318 + NAVD88 heights, the CONUS contract). Required "
                        "outside the NAD83 area of use: e.g. --landing-crs EPSG:7912 "
                        "for Nepal NGL control. Geographic CRS only")
    p.add_argument("--context-sheets", action="store_true",
                   help="also write the per-point context contact sheets "
                        "(RGB web-basemap windows per fetched point — the "
                        "slow part; off by default)")
    p.add_argument("--no-figures", action="store_true",
                   help="write only the control file + provenance; skip the "
                        "standard figure set (contact sheets, labeled control "
                        "map, MIDAS velocity + NGL series figures)")
    p.add_argument("--basemap", default="esri", choices=("esri", "google", "none"),
                   help="web-imagery provider for the figures (default: esri; "
                        "'none' for offline runs)")
    p.add_argument("--valid-footprint", action="store_true",
                   help="for a raster --aoi: remove nodata and polygonize "
                        "the VALID pixels (as gdal_footprint does; rings "
                        "simplified to <=100 points) instead of the default "
                        "grid extent (costly on large rasters without "
                        "overviews)")
    p.add_argument("--refresh", action="store_true",
                   help="force re-download of every shared-cache file this "
                        "run touches (source catalogs, station series, web "
                        "indexes) — the fresh-run switch; caches otherwise "
                        "refresh on their own staleness windows")
    p.add_argument("--quiet", action="store_true",
                   help="suppress the package's INFO progress logging")
    args = p.parse_args(argv)
    _setup_logging(args.quiet)
    if args.refresh:
        os.environ["GROUNDCONTROL_REFRESH"] = "1"

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
    if args.landing_crs is not None:
        from groundcontrol.sources import validate_landing_crs
        _validate_crs(args.landing_crs, "--landing-crs")
        _preflight(validate_landing_crs, args.landing_crs)
    aoi = _load_aoi(aoi, valid_footprint=args.valid_footprint)

    print(f"querying sources: {', '.join(sources)} ...", file=sys.stderr)
    gdf, status = fetch_control(aoi, sources=sources,
                                target_crs=args.target_crs, target_epoch=args.target_epoch,
                                landing_crs=args.landing_crs)
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
    if not args.no_figures:
        # figures are ON by default, matching groundcontrol-assess (owner
        # 2026-08-31: "you shouldn't have to specify"); a source that found
        # no sites simply contributes no figures
        from groundcontrol.figures import context_sheets, standard_control_figures
        sheets = []
        if args.context_sheets:   # opt-in: the slow figure component
            sheets = context_sheets(gdf, {}, Path(out).parent, Path(out).stem,
                                    basemap=None if args.basemap == "none"
                                    else args.basemap)
        # the labeled all-sources control map that locates each sheet cell,
        # plus the MIDAS velocity + NGL time-series figures (owner
        # 2026-08-30: the AOI-only path gets the full standard set too)
        aoi_fig = aoi
        if isinstance(aoi, tuple):   # bbox: the box IS the AOI for figures
            import geopandas as gpd
            from shapely.geometry import box
            aoi_fig = gpd.GeoDataFrame(geometry=[box(*aoi)], crs=4326)
        figs = standard_control_figures(
            gdf, aoi_fig,
            Path(out).parent, Path(out).stem, midas_velocities=True,
            map_basemap=None if args.basemap == "none" else "esri_hillshade")
        # per-file paths are in the INFO log (the writers log each one);
        # stdout gets the count, not a raw list (owner 2026-08-30)
        print(f"wrote {len(sheets) + len(figs)} figures next to {out}",
              file=sys.stderr)
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


def _load_aoi(aoi, valid_footprint=False):
    """A parsed --aoi -> what fetch_control/figures consume: the bbox tuple
    as-is, else the file read up front (vector -> its features; DEM/DSM/DTM
    raster -> its grid extent, or its valid-pixel footprint with
    ``valid_footprint``; both EPSG:4326) so an unreadable file fails here,
    before the fetch, naming the flag."""
    if isinstance(aoi, tuple):
        return aoi
    from groundcontrol.aoi import read_aoi
    try:
        return read_aoi(aoi, valid_footprint=valid_footprint)
    except (ValueError, OSError) as e:  # both readers failed, or no CRS / no data
        raise SystemExit(f"error: --aoi {e}") from e


def _default_site_name(products):
    """--site-name default: the product file stem; for a DSM/DTM pair the
    COMMON PREFIX of the stems, so no product token names the site (owner
    2026-08-30: a DTM figure titled ``..._0.5m-DSM_mos`` read as the wrong
    product). Falls back to the first stem when the prefix is too short to
    identify anything."""
    import os
    stems = [Path(p).stem for p in products.values()]
    stem = stems[0]
    if len(stems) > 1:
        common = os.path.commonprefix(stems)
        # a prefix ending mid-token is a name FRAGMENT, not a name: DSM_mos/
        # DTM_no_fill_mos share "...-D" (both continue with D) and rstrip
        # cannot remove it — cut back to the last separator (vantor-06 /
        # owner 2026-08-31: "-D" reached ~19 output names and figure titles)
        if any(len(st) > len(common) for st in stems):
            cut = max(common.rfind(c) for c in "_-.")
            if cut > 0:
                common = common[:cut]
        common = common.rstrip("_-. ")
        if len(common) >= 3:
            stem = common
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_.") or "site"


def _classify_input(path):
    """'raster' | 'vector' for a positional INPUT path (fail-loud else)."""
    from groundcontrol.io import expand_user_path
    fp = path
    if not _is_remote(str(path)):
        fp = str(expand_user_path(Path(path)))
        if not Path(fp).exists():
            raise ValueError(f"input {path}: file not found")
    import rasterio
    try:
        with rasterio.open(fp):
            return "raster"
    except Exception:
        pass
    try:
        import pyogrio
        pyogrio.read_info(fp)
        return "vector"
    except Exception as e:
        raise ValueError(
            f"input {path}: not a readable raster or vector ({e})") from e


#: --vdatum product presets (owner 2026-08-30): named products resolve to
#: the researched frame so strip/mosaic users need none of the frame
#: archaeology. Evidence + caveats: docs/vdatum.md.
VDATUM_PRESETS = {
    "3dep": "EPSG:5703",
    "cop30": "EPSG:3855:itrf2014",
    "precision3d": "ellipsoid:g1674",
    "earthdem": "ellipsoid:itrf2014",
    "arcticdem": "ellipsoid:itrf2014",
    "rema": "ellipsoid:itrf2014",
}

#: presets are DATED SNAPSHOTS of each product line's source-delivery
#: datum (owner 2026-08-30: products version, and datums move — NSRS 2022
#: will replace NAVD88; PGC registration policies change per release);
#: the note is printed at resolve time so the assumption is on the record.
_PRESET_NOTES = {
    "3dep": "NAVD88 orthometric as delivered (GEOID18-realized chain; the "
            "NSRS 2022 modernization will change this) [as of 2026-08]",
    "cop30": "Copernicus GLO-30/90: EGM2008 heights, grid rebased to "
             "ITRF2014 [as of 2026-08]",
    "precision3d": "Vantor-stated WGS84 G1674 (= ITRF2008 @ 2005.0) "
                   "[as of 2026-08]",
    "earthdem": "UNREGISTERED strips + mosaics (v1.1): meters-level "
                "vertical bias, coregistration still required [as of 2026-08]",
    "arcticdem": "strips unregistered (~4 m); mosaic v4.1 anchored to "
                 "GLO-30 outside Greenland — docs/vdatum.md [as of 2026-08]",
    "rema": "strips unregistered (~4 m); mosaic v2 IS2-aligned (ITRF2014) "
            "— docs/vdatum.md [as of 2026-08]",
}

_PGC_NAME_RE = r"setsm|arcticdem|rema|earthdem|utm\d{2}[ns]_\d"


def _compound_vertical(crs):
    """The gravity-related HEIGHT vertical member of a compound CRS, else
    None. A compound's vertical member defines the height datum even when
    the horizontal sits on a datum ensemble — it must never be demoted
    away. The gate keeps a WKT 'ellipsoidal height' VerticalCRS — for
    which the realization IS the height datum — out of the datum-defined
    branch (round-2 audit). Discrimination (round-4 recipe, verified over
    every EPSG vertical and compound):
    - a 'Gravity-related height' axis is authoritative (247 of the 299
      EPSG verticals; the rest are depth-type);
    - otherwise the stem 'ellipsoid' anywhere in the CRS or axis names
      marks it ellipsoidal (0 registered entries contain it, so no real
      geoid is lost) — GDAL's WKT1 VERT_CS round-trip destroys axis
      names ('Gravity-related height' -> 'Up'), so the CRS name must
      carry the test too; a name with NEITHER marker is gravity-related
      by the WKT spec's own definition of a vertical CRS;
    - depth-type axes (direction down) are refused: heights would
      sign-flip downstream, which assumes height-up.
    Registration is NOT the test (round-3: a genuine but unregistered
    national geoid — the non-CONUS case this package exists for — must
    stay datum-defining)."""
    import pyproj
    if not crs.is_compound:
        return None

    def _gravity_height(c):
        if any((a.direction or "").lower() == "down" for a in c.axis_info):
            return False
        axes = " ".join(a.name or "" for a in c.axis_info).lower()
        if "gravity" in axes:
            return True
        txt = ((c.name or "") + " " + axes).lower()
        # 'depth' stem too: WKT1 round-trips erase axis DIRECTION as well
        # as names, so a custom depth CRS reads back as ('Up', up) —
        # 50/52 EPSG depth verticals carry 'depth' in the name, 0/246
        # height verticals do (round-5 audit)
        return "ellipsoid" not in txt and "depth" not in txt

    return next((c for c in (pyproj.CRS(s) for s in crs.sub_crs_list)
                 if c.is_vertical and _gravity_height(c)), None)


def _vdatum_target_crs(products, vdatum):
    """--vdatum resolver: each product's embedded 2D horizontal CRS + the
    stated vertical datum -> ONE full 3D target (geodesy.with_vdatum);
    every product must agree on the horizontal. The 3D/compound-embedded
    case is refused — the product already declares its heights."""
    import pyproj
    import rasterio

    from groundcontrol.assess import has_vertical_axis
    from groundcontrol.geodesy import with_vdatum
    key = vdatum.strip().lower()
    if key in VDATUM_PRESETS:
        print(f"--vdatum preset '{key}' -> {VDATUM_PRESETS[key]} "
              f"({_PRESET_NOTES[key]})", file=sys.stderr)
        vdatum = VDATUM_PRESETS[key]
    seen = {}
    for name, path in products.items():
        with rasterio.open(path) as src:
            crs = src.crs
        if crs is None:
            raise ValueError(f"--vdatum: product {name}={path} has no CRS "
                             "to attach the vertical datum to; pass the "
                             "full 3D frame via --target-crs")
        crs = pyproj.CRS.from_user_input(crs)
        if has_vertical_axis(crs):
            from groundcontrol.geodesy import is_wgs84_ensemble
            vert = _compound_vertical(crs)
            if vert is not None:
                # a compound with a gravity-related vertical declares its
                # height datum unambiguously WHATEVER the horizontal
                # ensemble means — demoting it here would silently discard
                # that vertical (COP30 EPSG:32645+3855: -37 m at Rasuwa)
                raise ValueError(
                    f"--vdatum: product {name}={path} already declares its "
                    f"heights ({crs.name}) — the vertical member "
                    f"'{vert.name}' defines the height datum; drop "
                    "--vdatum, or override with --target-crs")
            if not is_wgs84_ensemble(crs):
                raise ValueError(f"--vdatum: product {name}={path} already "
                                 f"declares its heights ({crs.name}); drop "
                                 "--vdatum, or override with --target-crs")
            # 3D on the ENSEMBLE is a declaration in name only (~2 m of
            # ambiguity) — --vdatum is exactly the disambiguation the
            # embedded-CRS refusal asks for (owner catch-22 report,
            # 2026-08-30): proceed from the demoted horizontal
            crs = crs.to_2d()
        if crs.is_projected and any(
                abs(a.unit_conversion_factor - 1.0) > 1e-12
                for a in crs.axis_info):
            # the rebase keeps these axes (H1) but the composed target's
            # HEIGHT axis is metres, and groundcontrol assumes elevations
            # are metres throughout — a raster storing ftUS heights would
            # be wrong by 3.28x. Full ftUS support is deferred; be loud.
            print(f"warning: product {name} uses non-metre horizontal "
                  f"units ({crs.axis_info[0].unit_name}); elevations are "
                  "assumed METRES — a raster storing survey-foot heights "
                  "will be wrong by 3.28x (full ftUS support pending)",
                  file=sys.stderr)
        seen[name] = crs
    first = next(iter(seen.values()))
    for name, crs in seen.items():
        if not crs.equals(first):
            names = {n: c.name for n, c in seen.items()}
            raise ValueError("--vdatum: products declare different "
                             f"horizontal CRSs {names}; pass --target-crs")
    out = with_vdatum(first, vdatum)
    print(f"target CRS: {out.name} (product horizontal + --vdatum "
          f"{vdatum})", file=sys.stderr)
    return out.to_wkt()


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
        from groundcontrol.geodesy import is_wgs84_ensemble
        if not has_vertical_axis(crs):
            code = crs.to_epsg()
            hz = f"EPSG:{code}" if code else "<horizontal EPSG>"
            if is_wgs84_ensemble(crs):
                # EPSG:326xx/327xx: the bare-ellipsoid form is itself
                # refused (ensemble ambiguity) — suggest realizations
                choices = (
                    "  --vdatum ellipsoid:itrf2014   ITRF2014 ellipsoidal "
                    "heights (SETSM EarthDEM/ArcticDEM/REMA,\n"
                    "                                most modern satellite "
                    "photogrammetry)\n"
                    "  --vdatum ellipsoid:itrf2020   ITRF2020 ellipsoidal "
                    "heights\n"
                    "  --vdatum ellipsoid:g2139      WGS 84 (G2139) "
                    "ellipsoidal heights\n"
                    "  --vdatum EPSG:3855            EGM2008 orthometric\n"
                    "note: this horizontal names the WGS 84 ENSEMBLE "
                    "(~2 m ambiguity, not a realization),\n"
                    "so a bare '--vdatum ellipsoid' is refused")
                import re as _re
                if _re.search(_PGC_NAME_RE, Path(path).name, _re.I):
                    choices += (
                        "\nthis filename looks like a PGC SETSM product — "
                        "presets apply the researched frame:\n"
                        "  --vdatum earthdem | arcticdem | rema\n"
                        "(evidence and caveats: docs/vdatum.md)")
            else:
                choices = (
                    f"  --vdatum ellipsoid       heights on the "
                    f"{crs.name.split(' / ')[0]} ellipsoid\n"
                    "  --vdatum EPSG:5703       NAVD88 orthometric "
                    "(CONUS lidar/3DEP)\n"
                    "  --vdatum EPSG:3855       EGM2008 orthometric\n"
                    f"  --target-crs {hz}+5703   the equivalent compound "
                    "form")
            raise SystemExit(
                f"error: --target-crs or --vdatum is required: product {name}={path} "
                f"declares a CRS without a height axis ({crs.name}"
                + (f", {hz}" if code else "") + "), which says nothing "
                "about its height datum — that lives in the product report, "
                "never guessed here. Common choices for this horizontal:\n"
                + choices + "\n"
                "(a .wkt file with a vertical member also works; "
                "groundcontrol.geodesy.with_vdatum/build_utm_* construct these)")
        elif is_wgs84_ensemble(crs) and _compound_vertical(crs) is not None:
            # compound with a gravity-related vertical on an ensemble
            # horizontal (COP30 native EPSG:4326+3855): the heights are
            # datum-defined by the vertical whichever WGS84 member the
            # grid sits on; only OUR horizontal transform legs are
            # ambiguous. Rebase them onto ITRF2014, loudly — mirroring
            # geodesy.with_vdatum's orthometric-on-ensemble branch.
            from pyproj.crs import CompoundCRS

            from groundcontrol.geodesy import (ITRF2014_EPSG,
                                               rebase_projection_2d)
            vert = _compound_vertical(crs)
            h2 = rebase_projection_2d(pyproj.CRS(crs.sub_crs_list[0]).to_2d(),
                                      ITRF2014_EPSG, "ITRF2014")
            crs = pyproj.CRS(CompoundCRS(name=f"{h2.name} + {vert.name}",
                                         components=[h2, vert]))
            print(f"product {name}: horizontal is the WGS 84 ENSEMBLE but "
                  f"heights are '{vert.name}' regardless of the member — "
                  f"using ITRF2014 for the transform legs ({crs.name})",
                  file=sys.stderr)
        elif is_wgs84_ensemble(crs):
            raise SystemExit(
                f"error: product {name}={path} declares 3D heights on the "
                f"WGS 84 ENSEMBLE ({crs.name}) — ~2 m of deliberate "
                "ambiguity, not a realization; transforms to it inherit a "
                "meter-class member-agnostic chain. State the realization "
                "the heights are actually on: --vdatum ellipsoid:itrf2014 "
                "(SETSM EarthDEM/ArcticDEM/REMA), ellipsoid:itrf2020, "
                "ellipsoid:g2139, ... or pass --target-crs")
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
    p.add_argument("inputs", nargs="*", metavar="INPUT",
                   help="positional inputs: raster product(s) and/or ONE vector "
                        "AOI file. A raster whose filename contains 'DTM' is "
                        "assessed under the bare-earth rules, any other under "
                        "the surface rules (name products explicitly with "
                        "--product to override). ONE site's product family "
                        "per run: at most one surface + one bare-earth "
                        "product; independent acquisitions are separate "
                        "runs. With only a vector AOI, runs the AOI-only "
                        "fetch + standard figures (= groundcontrol-fetch)")
    p.add_argument("--product", action="append", default=None,
                   help="NAME=PATH gridded elevation raster to assess (repeatable; "
                        "any GDAL format: GeoTIFF/COG/VRT/...). A NAME containing "
                        "'DTM' is assessed under the bare-earth rules (VVA "
                        "checkpoints apply); any other name (DSM, DEM, ...) under "
                        "the surface rules. Optional when rasters are given "
                        "positionally")
    p.add_argument("--aoi", default=None,
                   help="AOI: 'minx,miny,maxx,maxy' bbox (EPSG:4326 lon/lat), a vector "
                        "file (GeoJSON preferred; any OGR format), or a raster whose "
                        "grid extent is used (--valid-footprint removes nodata, "
                        "polygon around valid pixels). Default: the union "
                        "of the --product extents")
    p.add_argument("--target-crs", default=None,
                   help="product 3D CRS: EPSG/authority string ('EPSG:6341+5703'), WKT, "
                        "or a .wkt file. Default: the product's own embedded CRS, "
                        "accepted only when it is compound/3D (declares the height "
                        "datum); a 2D raster CRS is refused, never guessed — "
                        "pair it with --vdatum instead")
    p.add_argument("--vdatum", default=None,
                   help="vertical datum of the product heights, combined with the "
                        "product's own 2D horizontal CRS into the full 3D target: "
                        "'ellipsoid' (heights on the horizontal datum's ellipsoid), "
                        "'ellipsoid:<realization>' (itrf2020/itrf2014/itrf2008/"
                        "g2139/g1674 — REQUIRED for WGS84-ensemble horizontals "
                        "like EPSG:326xx), a product PRESET (3dep, cop30, "
                        "precision3d, earthdem, arcticdem, rema — the "
                        "researched source-delivery frame, dated; see "
                        "docs/vdatum.md), "
                        "or any vertical CRS ('EPSG:5703' NAVD88, 'EPSG:3855' "
                        "EGM2008, 'NAVD88 height', ...). Mutually exclusive with "
                        "--target-crs; a geoid model name (GEOID18) is not a CRS — "
                        "pass the vertical CRS it realizes")
    p.add_argument("--target-epoch", type=float, default=2010.0,
                   help="transform-time epoch tt, decimal year (default 2010.0; "
                        "inert for static-frame targets)")
    p.add_argument("--source-crs", default=None,
                   help="override the control landing CRS (default EPSG:6318+5703, "
                        "the fetch_control contract; derived automatically with "
                        "the landing for a non-NAD83 target) — expert use, e.g. "
                        "a cache already in another frame")
    p.add_argument("--landing-crs", default=None,
                   help="horizontal landing frame for the control fetch "
                        "(default: the CONUS EPSG:6318 + NAVD88 contract; "
                        "DERIVED from the target datum when that is outside "
                        "the NAD83 family — e.g. an ITRF2014 target lands at "
                        "EPSG:7912, the Nepal pattern). Geographic CRS only")
    p.add_argument("--control", default=None,
                   help="control GeoParquet cache: reused when present, else fetched "
                        "from --sources and written here (default: <outdir>/<site-name>_control.parquet)")
    p.add_argument("--sources", default="3dep,ngs,opus,ngl,faa",
                   help="comma-separated fetch sources (default: every provider)")
    p.add_argument("--outdir", default=None,
                   help="output directory (default: <input stem>_groundcontrol/ "
                        "next to the first input)")
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
    p.add_argument("--rgb", action="append", default=None,
                   help="RGB ortho raster for the contact sheets (repeatable: "
                        "fallback chain; the --basemap web imagery rides behind "
                        "them either way)")
    p.add_argument("--intensity", default=None,
                   help="lidar-intensity raster: grayscale contact-sheet panel")
    p.add_argument("--basemap", default="esri", choices=("esri", "google", "none"),
                   help="web-imagery provider for the contact sheets' RGB panel "
                        "(fetched over the network, credited on the sheet; "
                        "'none' for offline runs; default: esri)")
    p.add_argument("--context-sheets", action="store_true",
                   help="also write the per-point context contact sheets "
                        "(web-imagery windows per GNSS/FAA/3DEP point — "
                        "useful for photo-ID QA, but the slow part of the "
                        "figure stage; off by default)")
    p.add_argument("--no-figures", action="store_true", help="skip figure output")
    p.add_argument("--point-lim", type=float, default=None,
                   help="pin the validation-figure map color limit (m); default "
                        "empirical tier-snapped from the plotted dz (issue #23)")
    p.add_argument("--vendor-lim", type=float, default=None,
                   help="pin the survey-grade histogram limit (m); default empirical")
    p.add_argument("--wide-lim", type=float, default=None,
                   help="pin the NGS-monument histogram limit (m); default empirical")
    p.add_argument("--valid-footprint", action="store_true",
                   help="remove nodata and polygonize the VALID pixels as "
                        "each product's AOI footprint (as gdal_footprint "
                        "does; rings simplified to <=100 points) instead of "
                        "the default grid extent. Costly on large rasters "
                        "without overviews; the default is safe — points "
                        "over nodata NaN out at sampling and are reported "
                        "as gaps")
    p.add_argument("--refresh", action="store_true",
                   help="force re-download of every shared-cache file this "
                        "run touches AND ignore an existing per-site control "
                        "cache (re-fetch + overwrite) — the fresh-run switch")
    p.add_argument("--quiet", action="store_true",
                   help="suppress the package's INFO progress logging")
    args = p.parse_args(argv)
    _setup_logging(args.quiet)
    if args.refresh:
        os.environ["GROUNDCONTROL_REFRESH"] = "1"
    if args.radius is not None and args.method != p.get_default("method"):
        p.error("--radius and --method are mutually exclusive (radius mode "
                "computes a neighborhood median)")
    if args.vdatum is not None and args.target_crs is not None:
        p.error("--vdatum and --target-crs are mutually exclusive: --target-crs "
                "already declares the vertical datum")

    from groundcontrol import io

    # Validate everything cheap before the (network) fetch -- the AOI file,
    # NAME=PATH pairs, CRS strings/files, the parquet engine, the control
    # cache, then the rasters (may touch the network for remote ones), then
    # the output directory -- so a typo fails here, not after the points
    # are fetched.
    products = _parse_kv(args.product, "--product")
    pos_vector = None
    for item in args.inputs:
        kind = _preflight(_classify_input, item)
        if kind == "raster":
            name = "DTM" if "dtm" in Path(item).stem.lower() else "DSM"
            if name in products:   # second same-class raster: refused below
                name = Path(item).stem
            products[name] = item
        else:
            if pos_vector is not None:
                raise SystemExit("error: more than one vector input "
                                 f"({pos_vector!r}, {item!r}); pass extra "
                                 "vectors via --aoi or as --product rasters")
            if args.aoi is not None:
                raise SystemExit(f"error: vector input {item!r} conflicts "
                                 "with --aoi")
            pos_vector = item
    if not products and pos_vector is None:
        p.error("no inputs: pass raster product(s) and/or a vector AOI "
                "(positionally, or via --product/--aoi)")
    if products:
        from groundcontrol.assess import check_product_family
        _preflight(check_product_family, products)
    first_input = (args.inputs[0] if args.inputs
                   else next(iter(products.values())))
    if args.outdir is None:
        args.outdir = str(Path(first_input).parent
                          / (Path(first_input).stem + "_groundcontrol"))
        print(f"outdir (default): {args.outdir}", file=sys.stderr)
    if not products:
        # AOI-only: the standard fetch path IS this run (owner 2026-08-31:
        # `groundcontrol-assess aoi.geojson` should just work)
        for flag, val in (("--target-crs", args.target_crs),
                          ("--vdatum", args.vdatum),
                          ("--source-crs", args.source_crs),
                          ("--hs", args.hs), ("--rgb", args.rgb),
                          ("--intensity", args.intensity)):
            if val:
                raise SystemExit(f"error: {flag} needs a raster product; "
                                 "an AOI-only run has none")
        site = args.site_name or Path(pos_vector).stem
        out = Path(args.outdir)
        _preflight(_make_outdir, out)
        print("no raster product: running the AOI-only fetch "
              "(groundcontrol-fetch) with the standard figure set",
              file=sys.stderr)
        argv2 = ["--aoi", pos_vector, "--sources", args.sources,
                 "--out", str(out / f"{site}_control.parquet"),
                 "--basemap", args.basemap]
        if args.no_figures:
            argv2.append("--no-figures")
        return fetch_control_main(argv2)
    if pos_vector is not None:
        args.aoi = pos_vector
    aoi = _parse_aoi(args.aoi) if args.aoi is not None else None
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
    rgb = None
    if args.rgb:
        checked = _check_rasters(dict(enumerate(args.rgb)), "--rgb")
        rgb = [checked[i] for i in range(len(args.rgb))]
    intensity = None
    if args.intensity is not None:
        intensity = _check_rasters({"intensity": args.intensity},
                                   "--intensity")["intensity"]
    if args.vdatum is not None:
        target_crs = _preflight(_vdatum_target_crs, products, args.vdatum)
    if target_crs is None:
        target_crs = _embedded_target_crs(products)
    # explicit landing override (the fetch CLI's flag): validated here;
    # the AUTO derivation happens after the AOI is resolved — the landing
    # is a property of WHERE the AOI is, never of the target frame (a
    # CONUS AOI keeps the NAD83/NAVD88 contract even for an ITRF target:
    # regression 2026-08-30, orthometric rows masked under a 7912 landing)
    landing = args.landing_crs
    if landing is not None:
        from groundcontrol.sources import validate_landing_crs
        _validate_crs(landing, "--landing-crs")
        _preflight(validate_landing_crs, landing)
    if len(products) == 2 and args.aoi is None:
        # DSM/DTM pair sanity (owner 2026-08-30): a product family shares
        # ground — disjoint bounds mean independent acquisitions, which
        # are separate runs. An explicit --aoi is the deliberate override.
        import rasterio
        from shapely.geometry import box
        (na, pa), (nb, pb) = products.items()
        with rasterio.open(pa) as a, rasterio.open(pb) as b:
            if not box(*a.bounds).intersects(box(*b.bounds)):
                raise SystemExit(
                    f"error: products {na} and {nb} have disjoint extents "
                    "— a DSM/DTM family shares ground. Independent "
                    "acquisitions are separate runs (or pass --aoi to "
                    "override deliberately)")
    if isinstance(hs, str):
        hs = _check_rasters({"hillshade": hs}, "--hs")["hillshade"]
    elif hs:
        hs = _check_rasters(hs, "--hs")
    control = (None if args.refresh
               else _preflight(_read_cache, cache) if cache.exists() else None)
    if args.refresh and cache.exists():
        print(f"--refresh: ignoring control cache {cache} (re-fetching)",
              file=sys.stderr)
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
        if args.valid_footprint:
            print("deriving the AOI from the product valid-pixel footprint "
                  "(--valid-footprint: a large raster without overviews is "
                  "read in full here)", file=sys.stderr)
        aoi = _preflight(union_footprints, list(products.values()),
                         valid=args.valid_footprint)
    else:
        aoi = _load_aoi(aoi, valid_footprint=args.valid_footprint)
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
        queried = None
        try:   # the provenance sidecar records which sources were QUERIED
            import json
            side = json.loads(
                Path(str(cache) + ".provenance.json").read_text())
            st = side.get("status") or {}
            if st:
                queried = set(st)
        except Exception:
            pass
        missing = set(sources) - (queried if queried is not None else have)
        if missing:
            print(f"warning: control cache never queried source(s) "
                  f"{sorted(missing)} (cache rows: {sorted(have)}); "
                  f"delete {cache} or --refresh to re-fetch",
                  file=sys.stderr)
    else:
        if landing is None:
            # AOI outside the NAD83 landing's area of use (Nepal, not CONUS):
            # fall back to the target's own realized geographic base; a CONUS
            # AOI keeps the default contract regardless of target
            b4326 = (aoi if isinstance(aoi, tuple)
                     else tuple(aoi.to_crs(4326).total_bounds))
            import pyproj as _pp
            _aou = _pp.CRS.from_epsg(6318).area_of_use
            lat_ok = not (b4326[3] < _aou.south or b4326[1] > _aou.north)
            if _aou.west <= _aou.east:
                lon_ok = not (b4326[2] < _aou.west or b4326[0] > _aou.east)
            else:   # NAD83's area of use WRAPS the antimeridian (Alaska)
                lon_ok = (b4326[2] >= _aou.west) or (b4326[0] <= _aou.east)
            inside = lat_ok and lon_ok
            if not inside:
                from groundcontrol.geodesy import (NAD83_FAMILY_GEOGRAPHIC,
                                                   is_wgs84_ensemble)
                _tgt = _pp.CRS.from_user_input(target_crs)
                _h = _pp.CRS(_tgt.sub_crs_list[0]) if _tgt.is_compound else _tgt
                _base = _h.geodetic_crs
                _c2 = (_base.to_2d().to_epsg()
                       if _base is not None and not is_wgs84_ensemble(_base)
                       else None)
                if _c2 is None or _c2 in NAD83_FAMILY_GEOGRAPHIC:
                    raise SystemExit(
                        "error: the AOI lies outside the NAD83/NAVD88 interim "
                        "landing's area of use and no landing can be derived "
                        "from the target — pass --landing-crs (e.g. EPSG:7912 "
                        "for ITRF2014 control)")
                _c3 = _base.to_3d().to_epsg()
                landing = f"EPSG:{_c3 or _c2}"
                from groundcontrol.sources import validate_landing_crs
                _preflight(validate_landing_crs, landing)
                print(f"landing (auto): {landing} — the AOI is outside the "
                      "NAD83/NAVD88 interim contract's area of use",
                      file=sys.stderr)
        from groundcontrol.sources import fetch_control
        print(f"querying sources: {', '.join(sources)} ...", file=sys.stderr)
        control, status = fetch_control(aoi, sources=sources,
                                        landing_crs=landing)
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

    if source_crs is None and control.crs is not None:
        # the CONTROL's own declared frame decides the source (owner
        # 2026-08-30: a reused non-CONUS cache met the default NAD83
        # contract and the frame guard refused — reading the cache's CRS
        # is a declaration, not a guess). NAD83-family/ensemble/projected
        # frames keep the CONUS contract default; a realized non-NAD83
        # geographic landing (ITRF2014 -> EPSG:7912) means ellipsoidal
        # heights on that frame, the landing convention that wrote it.
        import pyproj as _pp

        from groundcontrol.geodesy import (NAD83_FAMILY_GEOGRAPHIC,
                                           is_wgs84_ensemble)
        _c = _pp.CRS.from_user_input(control.crs)
        if _c.is_geographic and not is_wgs84_ensemble(_c):
            _c2 = _c.to_2d().to_epsg()
            if _c2 is not None and _c2 not in NAD83_FAMILY_GEOGRAPHIC:
                _c3 = _c.to_3d().to_epsg()
                source_crs = f"EPSG:{_c3 or _c2}"
                print(f"source CRS (from the control's declared frame): "
                      f"{source_crs} — heights ellipsoidal on it",
                      file=sys.stderr)

    from groundcontrol.assess import assess_products  # ~0.5 s; after the preflight

    sampled, stats, artifacts = assess_products(
        control, products, target_crs,
        outdir=outdir, site_name=site_name, aoi=aoi_fig,
        hs=hs, rgb=rgb, intensity=intensity,
        basemap=None if args.basemap == "none" else args.basemap,
        midas_velocities=True,  # explicit at the entry point (default too)
        sheets=args.context_sheets,
        target_epoch=args.target_epoch, method=args.method,
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
    n_figs = 0
    for k, v in artifacts.items():
        if k == "transform":
            continue
        if isinstance(v, (list, tuple)):
            n_figs += len(v)     # figure groups: paths are in the INFO log
        else:
            print(f"wrote {v}", file=sys.stderr)
    if n_figs:
        print(f"wrote {n_figs} figures under {outdir} "
              "(per-file paths in the INFO log)", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(fetch_control_main())
