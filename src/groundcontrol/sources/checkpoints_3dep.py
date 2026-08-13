"""USGS 3DEP national survey checkpoints (2004-2025).

Source: ScienceBase item 67075e6bd34e969edc59c3e7; read from the parquet
mirror **pinned to a commit SHA** (plan A3), cached locally once.

Verified against the real file (2026-07, 145,299 rows): CRS is compound
``EPSG:6349`` (NAD83(2011) + NAVD88 height, frame anchor epoch 2010);
``point_type`` values are ``{NVA, VVA, BVA, Unknown}`` (NOT ``Control`` as an
earlier plan draft said); per-point provenance includes ``source_geoid``,
source EPSGs/units, and a ``datetime`` measurement date; the harmonized
NAVD88-meters height is ``z_meter_vdatum_update``.
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

PARQUET_SHA = "5cf92afad4282fe5901ebd7178343f680a161aac"  # scottyhq/files @ 2026-01-28
PARQUET_URL = (
    "https://raw.githubusercontent.com/scottyhq/files/"
    f"{PARQUET_SHA}/checkpoints_3dep_2004_2025.parquet"
)
# Authoritative fallback (document per plan A3): ScienceBase item
# https://www.sciencebase.gov/catalog/item/67075e6bd34e969edc59c3e7


def cache_dir() -> Path:
    d = Path(os.environ.get("GROUNDCONTROL_CACHE_DIR", "~/.cache/groundcontrol")).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch(aoi_bounds_4326, url: str = PARQUET_URL) -> gpd.GeoDataFrame:
    """Bbox read of the national checkpoint DB (downloads + caches on first use)."""
    local = cache_dir() / Path(url).name
    if not local.exists():
        logger.info("downloading %s -> %s", url, local)
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        local.write_bytes(r.content)
    return gpd.read_parquet(local, bbox=tuple(aoi_bounds_4326))


#: columns consumed into first-class schema fields; the rest go to ``raw``.
_CONSUMED = {"id", "point_type", "datetime", "z_meter_vdatum_update", "accuracy", "geometry"}


def parse(raw: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Map the checkpoint columns into the schema shape (native frame)."""
    n = len(raw)
    extras = [c for c in raw.columns if c not in _CONSUMED]
    mdt = (pd.to_datetime(raw["datetime"], utc=True, errors="coerce") if n
           else pd.Series([], dtype="datetime64[ns, UTC]"))
    out = gpd.GeoDataFrame(
        {
            "id": raw["id"].astype("string"),
            # TODO(D2): keep source values (NVA/VVA/BVA/Unknown) until the
            # marker_type/assessment_class split is adjudicated.
            "point_type": raw["point_type"].astype("string"),
            "height": pd.to_numeric(raw["z_meter_vdatum_update"], errors="coerce"),
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
            # z_meter_vdatum_update). See docs/accuracy_conventions.md. TODO(D3)
            "native_x": raw.geometry.x.to_numpy() if n else np.array([]),
            "native_y": raw.geometry.y.to_numpy() if n else np.array([]),
            "native_h": pd.to_numeric(raw["z_meter_vdatum_update"], errors="coerce"),
            "native_crs": pd.Series(["EPSG:6349"] * n, dtype="string"),
            "raw": pd.Series(
                [json.dumps({k: str(raw.iloc[i][k]) for k in extras}) for i in range(n)],
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
