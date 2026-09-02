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

import errno
import json
import logging
import os
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
        "n_points": int(len(gdf)),
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


_EXPORT_FORMATS = (".parquet", ".csv")


def _unsupported_format_msg(suffix: str) -> str:
    return (f"unsupported export format {suffix!r} (use .parquet or .csv; "
            "KML arrives with the [kml] extra)")


def expand_user_path(path) -> Path:
    """``Path(path).expanduser()`` that fails as ValueError (``~nosuchuser``
    raises RuntimeError, which no CLI error handler expects)."""
    try:
        return Path(path).expanduser()
    except RuntimeError as e:
        raise ValueError(f"{path}: cannot expand '~' ({e})") from e


def check_parquet_engine(what="parquet I/O") -> None:
    """Raise a loud, actionable ImportError if ``pyarrow.parquet`` is not importable.

    ``what`` names the consumer for the message (a path or a description).
    ``pyarrow`` is a declared dependency, but an env can still lack a working
    ``pyarrow.parquet``: a ``--no-deps`` install, or conda-forge's minimal
    ``pyarrow-core`` build, which owns the ``pyarrow`` distribution metadata
    (so pip reports the requirement satisfied) yet ships without the
    ``libparquet`` library. Unlike geopandas' optional-dependency wrapper
    (``raise ... from None``) the original ImportError is chained, so
    "missing" and "broken build" stay distinguishable.
    """
    what = what.name if isinstance(what, Path) else str(what)
    try:
        import pyarrow.parquet  # noqa: F401
    except ImportError as e:
        raise ImportError(
            f"'pyarrow.parquet' failed to import ({e}); it is needed for {what}. "
            "pyarrow is a required dependency of groundcontrol. "
            "In a conda env: conda install -c conda-forge pyarrow (the full package; "
            "pyarrow-core satisfies pip's check but lacks libparquet, so 'pip install "
            "pyarrow' reports it already satisfied). Otherwise: pip install pyarrow. "
            "Verify with: python -c 'import pyarrow.parquet'") from e


def sidecar_path(path) -> Path:
    """The provenance sidecar written next to an export."""
    path = Path(path)
    return path.with_name(path.name + ".provenance.json")


def check_export_support(path, *, sidecar=True) -> Path:
    """Fail loud -- and early -- if ``path`` cannot be written in this environment.

    Checks the format, the parquet engine, and that the product (and, with
    ``sidecar=True``, its provenance sidecar) can be written: each is
    either an existing writable file or creatable in an existing writable
    directory, and neither is a directory. A read-only existing product
    means "do not overwrite" -- pyarrow unlinks the destination before
    failing on it, so this check is what keeps a failed write from
    deleting the previous product; an existing parquet product must also
    be readable (the provenance embed re-reads it). ``~`` is expanded;
    symlinks are written through, as the OS does (a link to a missing file
    is fine when the target's directory exists). Never creates
    directories: a typo'd directory must fail here, not become a silent
    write somewhere else. Returns the expanded path. :func:`write` calls
    this itself; the CLIs call it before any network fetch so the failure
    costs nothing.
    """
    path = expand_user_path(path)
    if path.is_dir():  # before the suffix check: "is a directory" is the better message
        raise IsADirectoryError(f"cannot write {path.name}: {path} is a directory")
    suffix = path.suffix.lower()
    if suffix not in _EXPORT_FORMATS:
        raise ValueError(_unsupported_format_msg(suffix))
    if suffix == ".parquet":
        check_parquet_engine(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(
            f"cannot write {path.name}: output directory {path.parent} does not exist")
    targets = [path, sidecar_path(path)] if sidecar else [path]
    for target in targets:
        if target.is_dir():
            raise IsADirectoryError(f"cannot write {target.name}: {target} is a directory")
        if target.exists():
            need = os.W_OK | (os.R_OK if target == path and suffix == ".parquet" else 0)
            if not os.access(target, need):
                raise PermissionError(
                    f"cannot write {target.name}: {target} exists and is read-only "
                    "(not overwriting)")
            continue
        real = target
        if target.is_symlink():
            try:
                real = Path(os.path.realpath(target, strict=True))
            except OSError as e:
                if e.errno == errno.ELOOP:
                    raise OSError(f"cannot write {target.name}: {target} is a symlink "
                                  "loop") from e
                real = Path(os.path.realpath(target))  # ENOENT: dangling, handled below
        if not real.parent.is_dir():
            raise FileNotFoundError(
                f"cannot write {target.name}: {target} is a symlink into a missing "
                f"directory ({real.parent})")
        if not os.access(real.parent, os.W_OK):
            raise PermissionError(
                f"cannot write {target.name}: output directory {real.parent} is not writable")
    return path


def write(gdf, path, status: dict | None = None, command: str | None = None) -> Path:
    """Write control points to ``path`` (.parquet or .csv) + provenance sidecar.

    Returns the output path. GeoParquet keeps full geometry/dtypes and embeds
    the provenance; CSV adds ``x``/``y`` columns (geometry dropped) with a
    ``# provenance:`` header comment pointing at the sidecar. When the heights'
    vertical datum is uniform, the GeoParquet CRS is promoted to the compound
    form (see :func:`_compound_export_crs`). :func:`check_export_support`
    runs first, so the write itself only fails for reasons the preflight
    cannot see (disk full, a device error, a concurrent change).
    """
    path = check_export_support(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        compound = _compound_export_crs(gdf)
        if compound is not None:
            gdf = gdf.set_crs(compound, allow_override=True)
            logger.info("export CRS promoted to compound: %s", compound.name)
    provenance = build_provenance(gdf, status=status, command=command)
    sidecar = sidecar_path(path)
    if suffix == ".parquet":
        gdf.to_parquet(path)
        _embed_parquet_metadata(path, provenance)
    else:  # .csv (check_export_support already rejected anything else)
        df = gdf.copy()
        df["x"] = df.geometry.x
        df["y"] = df.geometry.y
        with open(path, "w") as f:
            f.write(f"# provenance: {sidecar.name} (schema {PROVENANCE_SCHEMA})\n")
            df.drop(columns=[df.geometry.name]).to_csv(f, index=False)
    sidecar.write_text(json.dumps(provenance, indent=1))
    logger.info("wrote %s (+ %s), %d points", path, sidecar.name, len(gdf))
    return path


def read_provenance(path) -> dict | None:
    """Read the embedded provenance from a GeoParquet export (None if absent)."""
    import pyarrow.parquet as pq

    meta = pq.read_schema(Path(path)).metadata or {}
    blob = meta.get(b"groundcontrol:provenance")
    return json.loads(blob) if blob else None
