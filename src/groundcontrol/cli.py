"""Console-script entry points (plan: CLI wrapper scripts).

Entry points live inside the package (a console script cannot live outside
``src/``). ``groundcontrol-assess`` arrives with Increment 2.
"""

from __future__ import annotations

import argparse
import json
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


def assess_dem_main(argv=None) -> int:
    """``groundcontrol-assess`` — DEM(+AOI) -> fetch -> sample -> stats + figures."""
    sys.exit("groundcontrol-assess: not yet implemented (Increment 2 — see docs/plan.md)")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(fetch_control_main())
