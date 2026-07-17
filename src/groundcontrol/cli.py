"""Console-script entry points (plan: CLI wrapper scripts).

Entry points live inside the package (a console script cannot live outside
``src/``). ``groundcontrol-assess`` arrives with Increment 2.
"""

from __future__ import annotations

import argparse
import sys


def fetch_control_main(argv=None) -> int:
    """``groundcontrol-fetch`` — AOI -> control GeoParquet/CSV + provenance."""
    p = argparse.ArgumentParser(
        prog="groundcontrol-fetch",
        description="Fetch ground control points for an AOI and export them "
                    "with transform provenance.",
    )
    p.add_argument("--aoi", required=True,
                   help="AOI: vector file (geojson/gpkg/...) or 'minx,miny,maxx,maxy' "
                        "bbox in EPSG:4326 (lon/lat). Use --aoi=-112,32.6,... for "
                        "bboxes with negative longitudes.")
    p.add_argument("--sources", default="3dep,ngs,opus",
                   help="comma-separated sources (default: 3dep,ngs,opus)")
    p.add_argument("--out", required=True, help="output path (.parquet or .csv)")
    p.add_argument("--target-crs", default=None,
                   help="target 3D CRS (NOT YET IMPLEMENTED — interim landing is "
                        "EPSG:6318 + NAVD88; passing this raises)")
    p.add_argument("--target-epoch", type=float, default=None,
                   help="target coordinate epoch, decimal year (NOT YET IMPLEMENTED)")
    args = p.parse_args(argv)

    from groundcontrol import io
    from groundcontrol.sources import fetch_control

    aoi = args.aoi
    if "," in aoi and not any(aoi.lower().endswith(s) for s in
                              (".geojson", ".json", ".gpkg", ".shp", ".parquet")):
        aoi = tuple(float(v) for v in aoi.split(","))
    sources = tuple(s.strip() for s in args.sources.split(",") if s.strip())

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
    io.write(gdf, args.out, status=status,
             command="groundcontrol-fetch " + " ".join(argv or sys.argv[1:]))
    print(f"wrote {args.out} ({len(gdf)} points) + provenance sidecar", file=sys.stderr)
    return 0


def _parse_kv(pairs, flag):
    out = {}
    for item in pairs or []:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            raise SystemExit(f"{flag} expects NAME=PATH, got {item!r}")
        if name in out:
            raise SystemExit(
                f"{flag} got NAME {name!r} twice ({out[name]!r} and {path!r}); "
                "a repeat would silently drop the first")
        out[name] = path
    return out


def _resolve_crs(spec):
    """A --target-crs value: EPSG code/authority string, WKT, or a .wkt file."""
    from pathlib import Path
    p = Path(spec)
    if p.suffix.lower() in (".wkt", ".prj") or (p.exists() and p.is_file()):
        return p.read_text()
    return spec


def assess_dem_main(argv=None) -> int:
    """``groundcontrol-assess`` — DEM(+AOI) -> fetch -> sample -> stats + figures."""
    p = argparse.ArgumentParser(
        prog="groundcontrol-assess",
        description="Assess DEM products against fetched/cached ground control: "
                    "transform control into the product frame, sample, and write "
                    "dz stats + standard validation figures.",
    )
    p.add_argument("--aoi", required=True,
                   help="AOI vector file or 'minx,miny,maxx,maxy' bbox (EPSG:4326)")
    p.add_argument("--product", action="append", required=True,
                   help="NAME=PATH raster to assess (repeatable; e.g. DSM=dsm.vrt)")
    p.add_argument("--target-crs", required=True,
                   help="product 3D CRS: EPSG/authority string, WKT, or path to a .wkt file")
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
    p.add_argument("--site-name", required=True, help="prefix for output artifacts")
    p.add_argument("--method", default="linear", choices=("linear", "nearest"),
                   help="point sampling method (default: linear)")
    p.add_argument("--radius", type=float, default=None,
                   help="neighborhood-median sampling radius in raster units "
                        "(mutually exclusive with --method)")
    p.add_argument("--hs", action="append", default=None,
                   help="NAME=PATH hillshade underlay for figures (repeatable, "
                        "product-matched; single unnamed path also accepted)")
    p.add_argument("--no-figures", action="store_true", help="skip figure output")
    args = p.parse_args(argv)

    from pathlib import Path

    from groundcontrol import io
    from groundcontrol.assess import assess_products

    aoi = args.aoi
    if "," in aoi and not any(aoi.lower().endswith(s) for s in
                              (".geojson", ".json", ".gpkg", ".shp", ".parquet")):
        aoi = tuple(float(v) for v in aoi.split(","))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.control) if args.control else outdir / f"{args.site_name}_control.parquet"
    if cache.exists():
        import geopandas as gpd
        control = gpd.read_parquet(cache)
        print(f"control cache: {cache} ({len(control)} points)", file=sys.stderr)
        req = {s.strip() for s in args.sources.split(",") if s.strip()}
        have = (set(control["source"].dropna().unique())
                if "source" in control.columns else set())
        if req - have:
            print(f"warning: control cache lacks requested source(s) "
                  f"{sorted(req - have)} (cache has {sorted(have)}); "
                  f"delete {cache} to re-fetch", file=sys.stderr)
    else:
        from groundcontrol.sources import fetch_control
        sources = tuple(s.strip() for s in args.sources.split(",") if s.strip())
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

    products = _parse_kv(args.product, "--product")
    hs = None
    if args.hs:
        if len(args.hs) == 1 and "=" not in args.hs[0]:
            hs = args.hs[0]
        else:
            hs = _parse_kv(args.hs, "--hs")

    sampled, stats, artifacts = assess_products(
        control, products, _resolve_crs(args.target_crs),
        outdir=outdir, site_name=args.site_name, aoi=aoi if not isinstance(aoi, tuple) else None,
        hs=hs, target_epoch=args.target_epoch, method=args.method,
        radius=args.radius, source_crs=args.source_crs, figures=not args.no_figures,
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
