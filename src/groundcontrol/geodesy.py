"""Programmatic CRS construction + datum-transform preflight (general geodesy).

Consolidated from ``lidar_tools/src/lidar_tools/geodesy.py`` (c1b9560) per
``docs/consolidation_geodesy.md`` — the general-purpose subset only. The
lidar/raster-specific pieces (EPT null-tie compound builders, GDAL raster
epoch stamping) deliberately stay in lidar_tools; groundcontrol remains
GDAL-free. Function names and signatures match the lidar_tools originals so
a future re-point is mechanical.

Two capabilities here overlap :mod:`groundcontrol.crs` by design (see the
consolidation map): :func:`preflight_vertical_transform` is the
provenance-returning superset of ``crs.get_transformer`` (grid download,
``prefer_grids``, area-of-use containment), and the UTM builders complement
``crs.transform_points`` for constructing explicit 3D target frames.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pyproj
import pyproj.network
from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.crs import ProjectedCRS
from pyproj.crs.coordinate_operation import UTMConversion
from pyproj.transformer import TransformerGroup

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = [
    "DEFAULT_COORDINATE_EPOCH",
    "WGS84_G2139_EPSG",
    "NAD83_2011_EPSG",
    "WGS84_G1674_EPSG",
    "ITRF2020_EPSG",
    "ITRF2008_EPSG",
    "ITRF2014_EPSG",
    "NAD83_FAMILY_GEOGRAPHIC",
    "GEOID_GRID_HINTS",
    "OUTPUT_DATUM_BUILDERS",
    "geographic_base_epsg",
    "geoid_grid_hint",
    "utm_zone_label",
    "build_utm_realization_3d",
    "build_utm_g2139_3d",
    "build_utm_nad83_2011_3d",
    "build_utm_g1674_3d",
    "build_utm_itrf2020_3d",
    "build_utm_itrf2008_3d",
    "build_utm_itrf2014_3d",
    "build_utm_target",
    "epoch_pinned_pipeline",
    "navd88_offset",
    "write_crs_file",
    "library_versions",
    "preflight_vertical_transform",
]

#: 3DEP lidar sources are NAD83(2011), epoch-reduced to 2010.0, and the
#: NAD83(2011)<->ITRF time-dependent Helmert is evaluated at its 2010.0
#: reference epoch, so 3DEP-derived outputs in a dynamic frame are
#: coordinates at epoch 2010.0 (docs/crs_implementation.md §1).
DEFAULT_COORDINATE_EPOCH = 2010.0

#: WGS 84 (G2139) geographic 2D
WGS84_G2139_EPSG = 9755
#: NAD83(2011) geographic 2D
NAD83_2011_EPSG = 6318
WGS84_G1674_EPSG = 9056
ITRF2020_EPSG = 9990
ITRF2008_EPSG = 8999
ITRF2014_EPSG = 9000

#: Geographic 2D bases of the NAD83 family (North America plate): the only
#: datums for which the 3DEP EPT null-tie treatment and the ITRF Helmert at
#: epoch 2010.0 are valid. PA11/MA11 (Pacific/Mariana plates) are
#: deliberately excluded — a North-America Helmert is the wrong
#: transformation there.
NAD83_FAMILY_GEOGRAPHIC = {
    4269,  # NAD83(1986)
    4152,  # NAD83(HARN)
    4759,  # NAD83(NSRS2007)
    6318,  # NAD83(2011)
    6783,  # NAD83(CORS96)
}

#: Survey geoid declaration -> PROJ grid-name fragment, for selecting the
#: transformation that uses the survey's production geoid model.
GEOID_GRID_HINTS = {
    "GEOID18": "g2018",
    "GEOID12B": "g2012b",
    "GEOID12A": "g2012a",
    "GEOID09": "geoid09",
    "GEOID06": "geoid06",
    "GEOID03": "geoid03",
    "GEOID99": "geoid99",
}


def geographic_base_epsg(crs_input) -> int:
    """Geographic 2D base EPSG of a declared horizontal CRS, validated as
    NAD83-family (e.g. 7131 NAD83(2011)/SP CA-3 ftUS -> 6318; 26910
    NAD83/UTM 10N -> 4269).

    Complements :func:`groundcontrol.crs.ngs_datum_to_epsg` (which maps NGS
    datum *strings*); this takes a CRS object/code and extracts + validates
    the base datum.

    Raises ``ValueError`` if the base cannot be identified or is not
    NAD83-family (including Pacific/Mariana-plate PA11/MA11 realizations).
    """
    s = str(crs_input).strip()
    crs = CRS.from_epsg(int(s)) if s.isdigit() else CRS.from_user_input(crs_input)
    base = crs.geodetic_crs
    code = base.to_epsg() if base is not None else None
    if code is not None and code not in NAD83_FAMILY_GEOGRAPHIC:
        # projected CRSs may report a 3D/other variant: try name matching
        if (base is not None and base.name.startswith("NAD83")
                and "PA11" not in base.name and "MA11" not in base.name):
            for cand in NAD83_FAMILY_GEOGRAPHIC:
                if CRS.from_epsg(cand).name == base.name:
                    return cand
    if code in NAD83_FAMILY_GEOGRAPHIC:
        return code
    raise ValueError(
        f"Declared horizontal CRS '{crs.name}' has base "
        f"'{base.name if base is not None else None}' (EPSG:{code}), which is "
        "not a supported NAD83-family (North America plate) datum. The EPT "
        "null-tie treatment and the epoch-2010.0 Helmert do not apply — "
        "handle this survey explicitly."
    )


def geoid_grid_hint(geoid_name) -> str | None:
    """PROJ grid-name fragment for a survey's declared geoid model
    (e.g. 'GEOID12B' -> 'g2012b'), or None when unknown/absent."""
    if geoid_name is None:
        return None
    key = str(geoid_name).upper().replace(" ", "")
    return GEOID_GRID_HINTS.get(key)


def utm_zone_label(utm_epsg: int) -> str:
    """UTM zone label (e.g. '10N' or '19S') for a WGS84 UTM EPSG code
    (326xx northern, 327xx southern)."""
    prefix, zone = divmod(utm_epsg, 100)
    if prefix == 326:
        return f"{zone}N"
    if prefix == 327:
        return f"{zone}S"
    raise ValueError(
        f"EPSG:{utm_epsg} is not a WGS84 UTM CRS (expected 326xx or 327xx)")


def build_utm_realization_3d(utm_epsg: int, base_epsg: int, base_name: str) -> CRS:
    """Build a 3D UTM CRS on an arbitrary geographic realization.

    UTM conversion (zone/hemisphere from ``utm_epsg``, e.g. from
    ``gdf.estimate_utm_crs().to_epsg()``) on the given geographic 2D base,
    promoted to 3D (ellipsoidal heights). Programmatic construction keeps the
    hemisphere's false northing correct — the WKT-template text substitution
    this replaced silently kept the northern false northing (0) for
    southern-hemisphere zones.
    """
    label = utm_zone_label(utm_epsg)
    zone, hemisphere = label[:-1], label[-1]
    return ProjectedCRS(
        conversion=UTMConversion(zone, hemisphere=hemisphere),
        geodetic_crs=CRS.from_epsg(base_epsg),
        name=f"{base_name} / UTM zone {label}",
    ).to_3d()


def build_utm_g2139_3d(utm_epsg: int) -> CRS:
    """3D UTM CRS on WGS 84 (G2139) (dynamic — stamp an epoch)."""
    return build_utm_realization_3d(utm_epsg, WGS84_G2139_EPSG, "WGS 84 (G2139)")


def build_utm_nad83_2011_3d(utm_epsg: int) -> CRS:
    """3D UTM CRS on static NAD83(2011) — the most native output for 3DEP
    sources: no time-dependent Helmert, no coordinate epoch; ellipsoidal
    heights on GRS 1980."""
    return build_utm_realization_3d(utm_epsg, NAD83_2011_EPSG, "NAD83(2011)")


def build_utm_g1674_3d(utm_epsg: int) -> CRS:
    """3D UTM CRS on WGS 84 (G1674) (~ITRF2008; dynamic — stamp an epoch)."""
    return build_utm_realization_3d(utm_epsg, WGS84_G1674_EPSG, "WGS 84 (G1674)")


def build_utm_itrf2020_3d(utm_epsg: int) -> CRS:
    """3D UTM CRS on ITRF2020 (dynamic — stamp an epoch)."""
    return build_utm_realization_3d(utm_epsg, ITRF2020_EPSG, "ITRF2020")


def build_utm_itrf2008_3d(utm_epsg: int) -> CRS:
    """3D UTM CRS on ITRF2008 (≡ WGS 84 (G1674) to ~cm; dynamic).

    ⚠ Prefer this over the wgs84_g1674 target when coord_epoch is used:
    with GDAL selecting the operation (no -ct), a WGS84-realization target
    gets the NULL NAD83<->WGS84 tie HORIZONTALLY (~1.2-1.3 m CONUS error),
    while the ITRF alias finds the direct time-dependent
    ITRFxxxx<->NAD83(2011) Helmert (verified empirically, LV T2 2026-07-09).
    """
    return build_utm_realization_3d(utm_epsg, ITRF2008_EPSG, "ITRF2008")


def build_utm_itrf2014_3d(utm_epsg: int) -> CRS:
    """3D UTM CRS on ITRF2014 (≡ WGS 84 (G2139) to ~cm; dynamic).

    ⚠ Same GDAL null-tie caveat as :func:`build_utm_itrf2008_3d`: use this
    instead of wgs84_g2139 whenever coord_epoch is passed.
    """
    return build_utm_realization_3d(utm_epsg, ITRF2014_EPSG, "ITRF2014")


#: Selectable output-datum realizations for an auto-built local-UTM target.
#: key -> (3D UTM builder, filename datum label). Arbitrary output CRSs
#: beyond these are still supported by passing an explicit dst_crs WKT file.
OUTPUT_DATUM_BUILDERS = {
    "wgs84_g2139": (build_utm_g2139_3d, "WGS84_G2139"),
    "nad83_2011": (build_utm_nad83_2011_3d, "NAD83_2011"),
    "wgs84_g1674": (build_utm_g1674_3d, "WGS84_G1674"),
    "itrf2020": (build_utm_itrf2020_3d, "ITRF2020"),
    "itrf2008": (build_utm_itrf2008_3d, "ITRF2008"),
    "itrf2014": (build_utm_itrf2014_3d, "ITRF2014"),
}


def build_utm_target(utm_epsg: int, output_datum: str = "wgs84_g2139") -> tuple[CRS, str]:
    """Auto-target 3D UTM CRS and its canonical WKT basename for a UTM zone
    and a selectable output datum realization.

    Returns the 3D UTM CRS and a basename like 'UTM_10N_NAD83_2011_3D.wkt'
    (caller joins it with the run directory).
    """
    try:
        builder, label = OUTPUT_DATUM_BUILDERS[output_datum]
    except KeyError:
        raise ValueError(
            f"Unknown output_datum '{output_datum}'; choose from "
            f"{sorted(OUTPUT_DATUM_BUILDERS)} or pass an explicit dst_crs WKT file.")
    return builder(utm_epsg), f"UTM_{utm_zone_label(utm_epsg)}_{label}_3D.wkt"


def epoch_pinned_pipeline(src_crs, dst_crs, coord_epoch: float,
                          aoi_bounds=None, require_substrings=()) -> str:
    """Resolve ONE explicit PROJ pipeline with the target coordinate epoch
    baked in (``projinfo --t_epoch``), for enforcement via gdalwarp ``-ct``.

    Operation AUTO-selection proved unstable across source datum declarations
    (LV four-frame validation 2026-07-10): GDAL free selection with
    ``-t_coord_epoch`` null-tied the horizontal Helmert for WGS84-realization
    targets, and flipped to null horizontal for ITRF targets when the source
    compound declared its true NAD83(2011) base. The only robust contract is
    an explicit pipeline. projinfo emits the top-ranked operation with
    ``+proj=set +v_4=<epoch>`` bookends, so the time-dependent Helmert is
    evaluated at the requested epoch without 4D input — usable as a static
    ``-ct`` string and recordable as provenance.

    This is groundcontrol's only subprocess call: ``projinfo`` ships with
    PROJ builds (conda ``proj`` package; system PROJ). pip-wheel-only envs
    may lack the CLI — this fails loud with a clear error rather than
    guessing a pipeline.

    Parameters
    ----------
    src_crs, dst_crs
        CRS object, WKT string, or path to a WKT file.
    coord_epoch
        Target coordinate epoch (decimal year).
    aoi_bounds
        Optional (west, south, east, north) degrees — passed as ``--bbox``
        so area-appropriate operations rank first.
    require_substrings
        Substrings that MUST appear in the selected pipeline (e.g.
        ``["+proj=helmert", "vgridshift"]``) — fail loud on a null or
        wrong-geoid route instead of producing silently shifted rasters.
    """
    def _wkt(c):
        if isinstance(c, CRS):
            return c.to_wkt()
        p = Path(str(c))
        if p.exists():
            return p.read_text()
        return str(c)

    exe = Path(sys.executable).parent / "projinfo"
    projinfo = str(exe) if exe.exists() else shutil.which("projinfo")
    if projinfo is None:
        raise RuntimeError(
            "projinfo executable not found (ships with PROJ builds, e.g. the "
            "conda-forge 'proj' package); cannot resolve an explicit pipeline")
    cmd = [projinfo, "-s", _wkt(src_crs), "-t", _wkt(dst_crs),
           "--t_epoch", str(coord_epoch), "--hide-ballpark",
           "--spatial-test", "intersects", "-o", "PROJ", "--single-line"]
    if aoi_bounds is not None:
        w, s, e, n = aoi_bounds
        cmd += ["--bbox", f"{w},{s},{e},{n}"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"projinfo failed: {out.stderr[-500:]}")
    pipelines = [ln.strip() for ln in out.stdout.splitlines()
                 if ln.strip().startswith("+proj=pipeline")]
    if not pipelines:
        raise RuntimeError(
            f"projinfo returned no pipeline for {coord_epoch=}: "
            f"{out.stdout[-500:]}")
    pipe = pipelines[0]
    # projinfo echoes the epoch with its own formatting — compare numerically,
    # not as a formatted substring (a ':g' tag spuriously rejects e.g. 2020.15625)
    epoch_match = re.search(r"\+proj=set \+v_4=([0-9.]+)", pipe)
    missing = []
    if epoch_match is None or abs(float(epoch_match.group(1)) - float(coord_epoch)) > 1e-6:
        missing.append(f"+proj=set +v_4={coord_epoch}")
    missing += [s for s in require_substrings if s not in pipe]
    if missing:
        raise RuntimeError(
            f"selected pipeline lacks required component(s) {missing}: {pipe}")
    return pipe


def navd88_offset(lon: float, lat: float) -> float:
    """Local NAVD88-to-NAD83(2011)-ellipsoidal offset N (meters, ~-18..-35 in
    CONUS): the ellipsoidal height of the zero-orthometric surface. Used as
    the expected signature of an already-ellipsoidal source in a vertical
    datum check. Requires the geoid grid (verify with the preflight first).
    """
    t = Transformer.from_crs("EPSG:6318+5703", "EPSG:6319", always_xy=True)
    return float(t.transform(lon, lat, 0.0)[2])


def write_crs_file(crs: CRS, outfn: str | Path) -> str:
    """Write a CRS definition as pretty WKT2:2019 (provenance sidecar for the
    exact CRS used). Returns the filename, usable as a gdal/PDAL SRS argument.
    """
    outfn = Path(outfn)
    logger.info("writing CRS definition to %s", outfn)
    outfn.write_text(crs.to_wkt(version="WKT2_2019", pretty=True))
    return str(outfn)


def library_versions() -> dict:
    """Versions of the geodesy-relevant libraries, for provenance metadata.

    The ``gdal`` key is present only when the GDAL Python bindings are
    importable — groundcontrol itself does not depend on them.
    """
    versions = {
        "proj": pyproj.proj_version_str,
        "pyproj": pyproj.__version__,
        "proj_data_dir": pyproj.datadir.get_data_dir(),
    }
    try:  # optional: report it when the env has GDAL (lidar_tools parity)
        from osgeo import gdal
        versions["gdal"] = gdal.__version__
    except ImportError:
        pass
    return versions


def preflight_vertical_transform(
    src_crs: CRS | str,
    dst_crs: CRS | str,
    download: bool = True,
    aoi_bounds: tuple = None,
    prefer_grids: str = None,
) -> dict:
    """Verify PROJ can rigorously transform src_crs -> dst_crs before compute.

    The provenance-returning superset of
    :func:`groundcontrol.crs.get_transformer` (same fail-loud
    ``TransformerGroup``/``allow_ballpark=False`` core; see
    ``docs/consolidation_geodesy.md``): adds missing-grid auto-download,
    ``prefer_grids`` selection, an AOI area-of-use containment assert, and
    returns a provenance dict instead of a Transformer. Despite the
    historical name it validates the full 3D path, not only the vertical.

    With required datum-shift grids missing (e.g. GEOID18 us_noaa_g2018u0.tif)
    and PROJ networking off, a warp can silently fall back to a null vertical
    transformation, leaving heights wrong by the geoid undulation (~31 m in
    CONUS) with no error. Run this first: if the best transformation is
    unavailable, it tries to download the missing grids (when networking is
    enabled), then raises rather than continuing toward a silently wrong
    product.

    Parameters
    ----------
    src_crs, dst_crs
        Source/target CRS (anything pyproj accepts).
    download
        Attempt to download missing grids to the PROJ user data directory
        when pyproj networking is enabled, by default True.
    aoi_bounds
        (west, south, east, north) degrees. Scopes transformation selection
        to the AOI and asserts the selected operation's area-of-use contains
        it — without this, the accuracy-ranked best operation can belong to
        another region (e.g. a CONUS geoid grid selected for a Puerto
        Rico/Alaska AOI), then applied out-of-area when enforced via -ct.
    prefer_grids
        Substring (e.g. 'g2012b') selecting the first available
        transformation whose pipeline uses a matching grid — for honoring a
        survey's production geoid model instead of PROJ's default ranking.

    Returns
    -------
    dict
        Provenance record: selected transformation description, PROJ
        pipeline, grids used, and stated accuracy.
    """
    src_crs = CRS.from_user_input(src_crs)
    dst_crs = CRS.from_user_input(dst_crs)
    aoi = (
        AreaOfInterest(
            west_lon_degree=aoi_bounds[0],
            south_lat_degree=aoi_bounds[1],
            east_lon_degree=aoi_bounds[2],
            north_lat_degree=aoi_bounds[3],
        )
        if aoi_bounds is not None
        else None
    )

    def make_group():
        # ballpark operations are the silent-fallback failure mode this
        # preflight exists to prevent: never consider them
        return TransformerGroup(
            src_crs,
            dst_crs,
            always_xy=True,
            area_of_interest=aoi,
            allow_ballpark=False,
        )

    group = make_group()
    if not group.best_available and download and pyproj.network.is_network_enabled():
        logger.info(
            "best available transformation requires datum-shift grids; "
            "downloading to the PROJ user-writable data directory")
        group.download_grids(verbose=True)
        group = make_group()
    if not group.best_available or not group.transformers:
        missing = sorted(
            {
                grid.short_name or grid.full_name
                for op in group.unavailable_operations
                for grid in op.grids
                if not grid.available
            }
        )
        aoi_note = (
            f" within the AOI {aoi_bounds} (no non-ballpark operation covers "
            "its area of use)" if aoi_bounds is not None else "")
        raise RuntimeError(
            f"PROJ cannot rigorously transform '{src_crs.name}' -> "
            f"'{dst_crs.name}'{aoi_note}: missing datum-shift grids {missing}. "
            "If grids are the problem, install them (e.g. 'pyproj sync --file "
            "<grid>' or the conda-forge proj-data package) or set "
            "PROJ_NETWORK=ON to allow on-demand grid download. Refusing to "
            "continue: a silent fallback would leave output heights wrong by "
            "the geoid undulation (~31 m in CONUS).")
    best = group.transformers[0]
    if prefer_grids is not None:
        matching = [
            t for t in group.transformers if prefer_grids in (t.definition or "")
        ]
        if not matching:
            raise RuntimeError(
                f"No available transformation '{src_crs.name}' -> "
                f"'{dst_crs.name}' uses a grid matching '{prefer_grids}' "
                f"(candidates: {[t.description for t in group.transformers[:5]]})")
        best = matching[0]
    # never enforce a pipeline outside its stated validity area
    if aoi_bounds is not None and best.area_of_use is not None:
        a = best.area_of_use
        west, south, east, north = aoi_bounds

        def lon_in(lon):
            # areas of use spanning the antimeridian (e.g. CONUS+Alaska)
            # have west > east
            if a.west <= a.east:
                return a.west <= lon <= a.east
            return lon >= a.west or lon <= a.east

        if not (
            a.south <= south and north <= a.north and lon_in(west) and lon_in(east)
        ):
            raise RuntimeError(
                f"Selected transformation '{best.description}' has area of use "
                f"'{a.name}' ({a.bounds}), which does not contain the AOI "
                f"{aoi_bounds}. Refusing to enforce an out-of-area pipeline.")
    definition = best.definition or ""
    grids = sorted(
        {name for match in re.findall(r"grids=(\S+)", definition)
         for name in match.split(",")})
    logger.info(
        "transform preflight OK: '%s' -> '%s' via %s (accuracy %s m, grids %s)",
        src_crs.name, dst_crs.name, best.description, best.accuracy,
        grids or "none")
    return {
        "source_crs": src_crs.name,
        "target_crs": dst_crs.name,
        "description": best.description,
        "proj_pipeline": definition,
        "grids": grids,
        "accuracy_m": best.accuracy,
        "area_of_use": best.area_of_use.name if best.area_of_use else None,
    }
