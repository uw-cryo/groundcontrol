"""Standardized export: GeoParquet / CSV with transform provenance.

Every export writes a ``<out>.provenance.json`` sidecar, and GeoParquet
additionally embeds the same JSON in the file's key-value metadata
(``groundcontrol:provenance``) so the file is self-auditing even when
separated from the sidecar. Mechanics per docs/crs_implementation.md §7:
``GeoDataFrame.to_parquet`` has no metadata kwarg (geopandas#3182), so the
parquet file is rewritten via pyarrow ``replace_schema_metadata``.

KML export lands behind the ``[kml]`` extra (plan packaging note).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pyproj

from groundcontrol import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

PROVENANCE_SCHEMA = "groundcontrol/provenance-v1"


def _environment() -> dict:
    db = pyproj.database.get_database_metadata
    return {
        "groundcontrol": __version__,
        "pyproj": pyproj.__version__,
        "proj": pyproj.proj_version_str,
        "epsg_db": {"version": db("EPSG.VERSION"), "date": db("EPSG.DATE")},
        "proj_data_version": db("PROJ_DATA.VERSION"),
        "proj_network": pyproj.network.is_network_enabled(),
    }


def build_provenance(gdf, status: dict | None = None, command: str | None = None) -> dict:
    """Assemble the provenance record for an export (sidecar + embedded copy)."""
    transforms = (
        gdf["transform_id"].value_counts(dropna=False).to_dict()
        if "transform_id" in gdf.columns else {}
    )
    return {
        "schema": PROVENANCE_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": command,
        "environment": _environment(),
        "target": {
            "crs_authority": gdf.crs.to_string() if gdf.crs else None,
            "crs_wkt2": gdf.crs.to_wkt(version="WKT2_2019") if gdf.crs else None,
        },
        "n_points": len(gdf),
        "transforms": {str(k): int(v) for k, v in transforms.items()},
        "dispatcher_status": status or {},
    }


def _embed_parquet_metadata(path: Path, provenance: dict) -> None:
    """Rewrite the parquet file with groundcontrol metadata keys alongside 'geo'."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    meta = dict(table.schema.metadata or {})
    meta[b"groundcontrol:provenance"] = json.dumps(provenance).encode()
    meta[b"groundcontrol:schema_version"] = PROVENANCE_SCHEMA.encode()
    if provenance["target"]["crs_authority"]:
        meta[b"groundcontrol:target_crs"] = provenance["target"]["crs_authority"].encode()
    pq.write_table(table.replace_schema_metadata(meta), path)


def _compound_export_crs(gdf):
    """Compound CRS for export when the heights' vertical datum is uniform.

    QGIS-facing honesty (TODO(D1) refinement): the file-level CRS should tell a
    reader what the ``height`` values are. If **every row that has a height**
    carries the same single ``vertical_crs``, promote the export CRS to
    ``horizontal+vertical`` (e.g. EPSG:6318+5703 -> "NAD83(2011) + NAVD88
    height"); otherwise keep the honest 2D horizontal CRS (never claim a
    vertical datum some rows don't have).
    """
    if gdf.crs is None or gdf.crs.is_compound or "vertical_crs" not in gdf.columns:
        return None
    with_height = gdf["height"].notna() if "height" in gdf.columns else gdf.index == gdf.index
    vcodes = gdf.loc[with_height, "vertical_crs"].dropna().unique()
    if len(vcodes) != 1 or gdf.loc[with_height, "vertical_crs"].isna().any():
        return None
    auth = gdf.crs.to_authority()
    if auth is None:
        return None
    try:
        return pyproj.CRS(f"{auth[0]}:{auth[1]}+{vcodes[0].split(':')[-1]}")
    except pyproj.exceptions.CRSError:  # pragma: no cover - defensive
        logger.warning("could not build compound export CRS from %s + %s", auth, vcodes[0])
        return None


def write(gdf, path, status: dict | None = None, command: str | None = None) -> Path:
    """Write control points to ``path`` (.parquet or .csv) + provenance sidecar.

    Returns the output path. GeoParquet keeps full geometry/dtypes and embeds
    the provenance; CSV adds ``x``/``y`` columns (geometry dropped) with a
    ``# provenance:`` header comment pointing at the sidecar. When the heights'
    vertical datum is uniform, the GeoParquet CRS is promoted to the compound
    form (see :func:`_compound_export_crs`).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        compound = _compound_export_crs(gdf)
        if compound is not None:
            gdf = gdf.set_crs(compound, allow_override=True)
            logger.info("export CRS promoted to compound: %s", compound.name)
    provenance = build_provenance(gdf, status=status, command=command)
    if suffix == ".parquet":
        gdf.to_parquet(path)
        _embed_parquet_metadata(path, provenance)
    elif suffix == ".csv":
        df = gdf.copy()
        df["x"] = df.geometry.x
        df["y"] = df.geometry.y
        with open(path, "w") as f:
            f.write(f"# provenance: {path.name}.provenance.json "
                    f"(schema {PROVENANCE_SCHEMA})\n")
            df.drop(columns=[df.geometry.name]).to_csv(f, index=False)
    else:
        raise ValueError(f"unsupported export format {suffix!r} (use .parquet or .csv; "
                         "KML arrives with the [kml] extra)")
    sidecar = path.with_name(path.name + ".provenance.json")
    sidecar.write_text(json.dumps(provenance, indent=1))
    logger.info("wrote %s (+ %s), %d points", path, sidecar.name, len(gdf))
    return path


def read_provenance(path) -> dict | None:
    """Read the embedded provenance from a GeoParquet export (None if absent)."""
    import pyarrow.parquet as pq

    meta = pq.read_schema(Path(path)).metadata or {}
    blob = meta.get(b"groundcontrol:provenance")
    return json.loads(blob) if blob else None
