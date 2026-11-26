"""USGS 3DEP national survey checkpoints (2004-2025).

Source: ScienceBase item 67075e6bd34e969edc59c3e7 (ver. 2.0, December 2025),
converted by ``scripts/build_checkpoints_3dep.py`` (see ``docs/checkpoints_3dep.md``).
The parquet is published on the UW Kopah object store as ``PARQUET_NAME``,
**pinned by SHA-256** (plan A3), and cached locally once.

145,299 rows. The GeoPackage CRS is compound ``EPSG:6349`` (NAD83(2011) +
NAVD88 height, frame anchor epoch 2010); the parquet stores 2D points in its
horizontal part, ``EPSG:6318``. ``point_type`` values are
``{NVA, VVA, BVA, Unknown}`` (NOT ``Control`` as an earlier plan draft said);
per-point provenance includes ``source_geoid``, source EPSGs/units, and a
``datetime`` measurement date. The harmonized NAVD88-meters height is
``z_meter_vdatum_update`` as USGS published it; ``z_meter_qc`` repairs its
known unit/decimal errors (NaN where unrepairable) and ``qc_height_flag`` names
them. ``qc_horizontal_flag`` marks rows whose source coordinates contradict
their recorded CRS, or whose published lon,lat position comes from a
non-NAD83(2011) source (uncertain by up to ~1 m). See ``docs/checkpoints_3dep.md``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely import force_2d

from groundcontrol.crs import decyear

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#: USGS release version + "-qc"; a new USGS release gets a new name
PARQUET_NAME = "checkpoints-3dep-2.0-qc.parquet"
#: mirrors, tried in order. The UW Kopah object store serves plain HTTPS with
#: range requests (browser tools such as stac-map can read it). A copy attached
#: to a future groundcontrol release can be appended here.
PARQUET_URLS = (
    f"https://s3.kopah.uw.edu/tacolab/groundcontrol/{PARQUET_NAME}",
)
PARQUET_URL = PARQUET_URLS[0]
#: content hash of the pinned file; also keys the local cache, so a re-pin
#: never reuses a stale download (the cache otherwise never expires)
PARQUET_SHA256 = "2008dc659fbfd25e3b8839e5190d6b64b6fc2b2e2072e53205a278b44f75f134"
# Authoritative fallback (document per plan A3): ScienceBase item
# https://www.sciencebase.gov/catalog/item/67075e6bd34e969edc59c3e7


def cache_dir() -> Path:
    d = Path(os.environ.get("GROUNDCONTROL_CACHE_DIR", "~/.cache/groundcontrol")).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_stale(local: Path, max_age_days: float | None = None) -> bool:
    """ONE staleness rule for every shared-cache file: missing, older than
    ``max_age_days`` (``None`` = never expires), or ``GROUNDCONTROL_REFRESH``
    set in the environment (the CLI ``--refresh`` flag: force re-download
    of everything this run touches; ``0``/``false`` disable it)."""
    import time
    if os.environ.get("GROUNDCONTROL_REFRESH", "").strip().lower() not in (
            "", "0", "false"):
        return True
    if not local.exists():
        return True
    if max_age_days is None:
        return False
    return (time.time() - local.stat().st_mtime) > max_age_days * 86400


def cache_write(local: Path, content: str | bytes) -> None:
    """Atomic write for every shared-cache file: tempfile in the same
    directory + ``os.replace``. A ``write_text`` interrupted mid-download
    leaves a 0-byte/truncated file that ``cache_stale`` then trusts for
    days — for ``ngl_steps.txt`` that read as "checked, no earthquake
    steps" and silently defeated the Gorkha step guard."""
    import stat
    import uuid
    tmp = str(local) + f".tmp-{uuid.uuid4().hex[:12]}"
    # O_CREAT with mode 0o666: the KERNEL applies the process umask
    # atomically — never os.umask() peeking, which is process-global and
    # raced under the concurrent NGL per-station pool (round-3 audit:
    # unrelated files created 0666 and the umask left at 0). mkstemp's
    # 0600 carried through os.replace was the round-2 finding.
    fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
    f = None
    try:
        if local.exists():  # a refresh keeps the existing file's mode
            os.fchmod(fd, stat.S_IMODE(local.stat().st_mode))
        # text goes out as UTF-8 explicitly (the fdopen default is the
        # platform locale; Path.write_text, which this replaced, was UTF-8)
        f = (os.fdopen(fd, "wb") if isinstance(content, bytes)
             else os.fdopen(fd, "w", encoding="utf-8"))
        with f:
            f.write(content)
        os.replace(tmp, local)
    except BaseException:
        if f is None:  # fd never handed to a file object: close it here
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def fetch(aoi_bounds_4326, url: str | tuple[str, ...] = PARQUET_URLS,
          sha256: str | None = PARQUET_SHA256) -> gpd.GeoDataFrame:
    """Bbox read of the national checkpoint DB (downloads + caches on first use).

    ``url`` is one URL or several mirrors tried in order. ``sha256`` (pass
    ``None`` for a custom ``url``) is checked on download and names the
    cached file; a mirror that fails or serves other bytes falls through to
    the next.
    """
    import hashlib
    urls = (url,) if isinstance(url, str) else tuple(url)
    name = Path(urls[0]).name
    if sha256:
        name = f"{Path(name).stem}_{sha256[:12]}{Path(name).suffix}"
    local = cache_dir() / name
    if cache_stale(local):
        errors = []
        for u in urls:
            logger.info("downloading %s -> %s", u, local)
            try:
                r = requests.get(u, timeout=120)
                r.raise_for_status()
            except requests.RequestException as e:
                errors.append(f"{u}: {e}")
                continue
            if sha256 and (got := hashlib.sha256(r.content).hexdigest()) != sha256:
                errors.append(f"{u}: SHA-256 {got} does not match the pinned {sha256}")
                continue
            cache_write(local, r.content)
            break
        else:
            raise ValueError("3DEP checkpoint download failed:\n  " + "\n  ".join(errors))
        for e in errors:
            logger.warning("mirror skipped: %s", e)
    return gpd.read_parquet(local, bbox=tuple(aoi_bounds_4326))


#: columns consumed into first-class schema fields (or dropped: bbox); the rest go to ``raw``.
_CONSUMED = {"id", "point_type", "datetime", "z_meter_vdatum_update", "z_meter_qc",
             "accuracy", "geometry", "bbox"}


def _height(raw: gpd.GeoDataFrame) -> pd.Series:
    """``z_meter_qc`` (unit/decimal errors repaired, NaN where unrepairable),
    falling back to ``z_meter_vdatum_update`` for parquet built before it."""
    col = "z_meter_qc" if "z_meter_qc" in raw.columns else "z_meter_vdatum_update"
    return pd.to_numeric(raw[col], errors="coerce")


def parse(raw: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Map the checkpoint columns into the schema shape (native frame)."""
    n = len(raw)
    extras = [c for c in raw.columns if c not in _CONSUMED]
    mdt = pd.to_datetime(raw["datetime"], utc=True, errors="coerce") if n else pd.Series([], dtype="datetime64[ns, UTC]")
    out = gpd.GeoDataFrame(
        {
            "id": raw["id"].astype("string"),
            # TODO(D2): keep source values (NVA/VVA/BVA/Unknown) until the
            # marker_type/assessment_class split is adjudicated.
            "point_type": raw["point_type"].astype("string"),
            "height": _height(raw),
            "height_datum": pd.Series(["NAVD88"] * n, dtype="string"),
            "horizontal_crs": pd.Series(["EPSG:6318"] * n, dtype="string"),
            "vertical_crs": pd.Series(["EPSG:5703"] * n, dtype="string"),
            "ref_frame": pd.Series(["NAD83(2011)"] * n, dtype="string"),
            "frame_epoch": np.full(n, 2010.0),
            # published positions are reduced to the frame epoch (plate-fixed)
            "coord_epoch": np.full(n, 2010.0),
            "measurement_datetime": mdt,
            "measurement_epoch": decyear(mdt) if n else pd.Series([], dtype="float64"),
            # `accuracy` column is authoritatively NULL for this whole release
            # ("USGS did not require an accuracy attribute" — FGDC metadata);
            # spec-based defaults (ASPRS 3x rule / NGS-58) pend D3. QC facts:
            # point_type=BVA = BATHYMETRY checkpoint (exclude for topo control);
            # source_geoid=UNK rows were never VDatum-harmonized (stale
            # z_meter_vdatum_update). qc_height_flag (in raw) marks the rows whose
            # height z_meter_qc repaired or blanked; qc_horizontal_flag (in raw)
            # marks positions uncertain at the ~1 m level. See
            # docs/accuracy_conventions.md and docs/checkpoints_3dep.md. TODO(D3)
            "native_x": raw.geometry.x.to_numpy() if n else np.array([]),
            "native_y": raw.geometry.y.to_numpy() if n else np.array([]),
            "native_h": _height(raw),
            "native_crs": pd.Series(["EPSG:6349"] * n, dtype="string"),
            "raw": pd.Series(
                # missing values (e.g. qc_height_flag on a clean row) serialize as
                # JSON null, never the strings "<NA>"/"nan", so they filter as null
                [json.dumps({k: (None if pd.isna(v := raw.iloc[i][k]) else str(v))
                             for k in extras}) for i in range(n)],
                dtype="string", index=raw.index,
            ),
        },
        # force_2d also strips the compound EPSG:6349 CRS the raw parquet
        # carries; passing raw.geometry directly (empty AOI) conflicts with
        # crs= below and raises in geopandas >= 1.0.
        geometry=force_2d(raw.geometry),
        crs="EPSG:6318",  # 2D horizontal component of EPSG:6349
        index=raw.index,
    )
    return out


#: ``raw`` fields lifted by :func:`expand_attributes` by default
QC_FIELDS = ("qc_height_flag", "qc_horizontal_flag", "source_horizontal_crs_qc",
             "source_axes_swapped_qc", "project_id", "source_geoid")


def expand_attributes(gdf: gpd.GeoDataFrame, fields=QC_FIELDS, prefix: str = "") -> gpd.GeoDataFrame:
    """Lift 3DEP ``raw`` fields (by default the QC flags) into real columns.

    The QC flags are 3DEP-specific, so they ride in ``raw`` rather than the
    shared schema; this returns a copy with them as nullable string columns
    (pd.NA where absent), so unflagged checkpoints filter as null::

        gdf = checkpoints_3dep.expand_attributes(gdf)
        clean = gdf.query("qc_height_flag.isna() and qc_horizontal_flag.isna()",
                          engine="python")

    Rows from other sources get NA too, so in a mixed frame also select
    ``source == "3dep"`` when only checkpoints are wanted. Same reader as
    :func:`groundcontrol.sources.ngs.expand_attributes`; ``prefix`` guards
    against a clash with schema columns (none for the defaults).
    """
    from groundcontrol.sources.ngs import expand_attributes as _expand
    return _expand(gdf, fields=fields, prefix=prefix)
