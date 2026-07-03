"""NGS datasheet (NDE) and OPUS-shared control points via the NGS API.

Ported from the casagrande NGS/OPUS notebook (plan Appendix A1) with the
Appendix B5 fixes: the ``'error'`` body is retried with backoff (never treated
as "too many"), subdivision has a max depth (accepting a possibly-truncated
500 with a warning), child cells overlap by an epsilon with pid-dedup.

Field names verified live against the API (2026-07): NDE carries
``posDatum``/``vertDatum``, network accuracies ``netAccHz``/``netAccU``,
``lastRecovered`` (YYYYMMDD); OPUS carries ``refFrame`` (e.g. NAD_83(2011)),
``epoch`` (e.g. 2010.0000), and the observation session times.
"""

from __future__ import annotations

import json
import logging
import time
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from groundcontrol.crs import decyear

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

NDE_URL = "https://geodesy.noaa.gov/api/nde/bounds"
OPUS_URL = "https://geodesy.noaa.gov/api/opus/bounds"
_API_CAP = 500  # NGS bounds endpoints cap responses at 500 items


def _get_records(params: dict, base_url: str, retries: int = 3) -> list:
    """One bounds request; retries+backoff on the HTTP-200 ``'error'`` body (B5)."""
    for attempt in range(retries):
        r = requests.get(base_url, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            return data
        logger.warning("NGS API returned %r (attempt %d/%d)", data, attempt + 1, retries)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"NGS API kept returning an error body for {params}")


def fetch_bbox(bounds_4326, base_url: str = NDE_URL, max_depth: int = 8,
               eps: float = 1e-6) -> list[dict]:
    """All records in a bbox, bypassing the 500-item cap via quad subdivision."""
    minlon, minlat, maxlon, maxlat = bounds_4326
    seen: set = set()
    out: list[dict] = []

    def _recurse(b_minlat, b_maxlat, b_minlon, b_maxlon, depth):
        params = {"minlat": b_minlat, "maxlat": b_maxlat,
                  "minlon": b_minlon, "maxlon": b_maxlon}
        data = _get_records(params, base_url)
        if len(data) >= _API_CAP and depth < max_depth:
            mid_lat = (b_minlat + b_maxlat) / 2
            mid_lon = (b_minlon + b_maxlon) / 2
            # overlap children by eps; seen-pid dedup absorbs the duplicates (B5)
            _recurse(b_minlat, mid_lat + eps, b_minlon, mid_lon + eps, depth + 1)
            _recurse(b_minlat, mid_lat + eps, mid_lon - eps, b_maxlon, depth + 1)
            _recurse(mid_lat - eps, b_maxlat, b_minlon, mid_lon + eps, depth + 1)
            _recurse(mid_lat - eps, b_maxlat, mid_lon - eps, b_maxlon, depth + 1)
            return
        if len(data) >= _API_CAP:
            warnings.warn(
                f"max subdivision depth {max_depth} hit with a full {_API_CAP}-item "
                "response; results may be truncated in this cell", stacklevel=2)
        for rec in data:
            pid = rec.get("pid")
            if pid and pid not in seen:
                seen.add(pid)
                out.append(rec)

    _recurse(minlat, maxlat, minlon, maxlon, 0)
    return out


def fetch_nde(aoi_bounds_4326) -> list[dict]:
    return fetch_bbox(aoi_bounds_4326, base_url=NDE_URL)


def fetch_opus(aoi_bounds_4326) -> list[dict]:
    return fetch_bbox(aoi_bounds_4326, base_url=OPUS_URL)


def _num(records: pd.DataFrame, col: str) -> pd.Series:
    """Numeric coercion; NGS blanks (' ') and '' become NaN."""
    if col not in records:
        return pd.Series(np.nan, index=records.index)
    return pd.to_numeric(records[col].replace(r"^\s*$", None, regex=True), errors="coerce")


def _frame_fields(datum: pd.Series) -> tuple[pd.Series, pd.Series]:
    """(ref_frame, frame_epoch) from an NGS datum/refFrame string column."""
    ref = datum.astype("string").str.replace("_", " ", regex=False).str.strip()
    is2011 = ref.str.contains("2011", na=False)
    frame_epoch = pd.Series(np.where(is2011, 2010.0, np.nan), index=datum.index)
    return ref, frame_epoch


def parse_nde(records: list[dict]) -> gpd.GeoDataFrame:
    """NDE datasheet records -> schema-shaped native-frame GeoDataFrame."""
    df = pd.DataFrame(records)
    if df.empty:
        from groundcontrol import schema
        return schema.empty(crs="EPSG:6318")
    from groundcontrol.crs import ngs_datum_to_epsg

    lat, lon = _num(df, "lat"), _num(df, "lon")
    ortho, ellip = _num(df, "orthoHt"), _num(df, "ellipHeight")
    ref_frame, frame_epoch = _frame_fields(df.get("posDatum", pd.Series(index=df.index)))
    # B7: per-row native realization CRS (fail-loud on unrecognized strings)
    h_crs = ref_frame.map(ngs_datum_to_epsg).astype("string")
    mdt = pd.to_datetime(df.get("lastRecovered"), format="%Y%m%d", utc=True, errors="coerce")
    consumed = {"pid", "lat", "lon", "orthoHt", "lastRecovered"}
    extras = [c for c in df.columns if c not in consumed]
    out = gpd.GeoDataFrame(
        {
            "id": df["pid"].astype("string"),
            "point_type": pd.Series(["monument"] * len(df), dtype="string"),  # TODO(D2)
            # orthoHt is the well-populated height; ellipHeight is often blank.
            # NAVD88 orthometric heights are invariant under NAD83 horizontal
            # realization changes -> the B7 landing is horizontal-only (B7b).
            "height": ortho,
            "height_datum": df.get("vertDatum", pd.Series(index=df.index)).astype("string"),
            "horizontal_crs": h_crs,
            "vertical_crs": pd.Series(["EPSG:5703"] * len(df), dtype="string"),
            "ref_frame": ref_frame,
            "frame_epoch": frame_epoch,
            "coord_epoch": _num(df, "epoch").fillna(frame_epoch),
            "measurement_datetime": mdt,
            "measurement_epoch": decyear(mdt),
            # NGS network accuracies (netAccHz/netAccU, 95%-confidence) stay in
            # raw until the confidence/units convention is adjudicated. TODO(D3)
            "native_x": lon, "native_y": lat, "native_h": ortho,
            "native_crs": (h_crs + "+5703").astype("string"),
            "raw": pd.Series([json.dumps({k: str(df.iloc[i][k]) for k in extras})
                              for i in range(len(df))], dtype="string", index=df.index),
        },
        # geometry values are per-row mixed-realization until the dispatcher
        # lands them (crs.land_horizontal); no single CRS is claimed here.
        geometry=gpd.points_from_xy(lon, lat),
        crs=None,
        index=df.index,
    )
    return out


def parse_opus(records: list[dict]) -> gpd.GeoDataFrame:
    """OPUS-shared records -> schema-shaped native-frame GeoDataFrame."""
    df = pd.DataFrame(records)
    if df.empty:
        from groundcontrol import schema
        return schema.empty(crs="EPSG:6318")
    from groundcontrol.crs import ngs_datum_to_epsg

    lat, lon = _num(df, "lat"), _num(df, "lon")
    ortho = _num(df, "orthoHt")
    ref_frame, frame_epoch = _frame_fields(df.get("refFrame", pd.Series(index=df.index)))
    h_crs = ref_frame.map(ngs_datum_to_epsg).astype("string")  # B7 (usually all 2011)
    mdt = pd.to_datetime(df.get("obsTimeStart"), utc=True, errors="coerce")
    consumed = {"pid", "lat", "lon", "orthoHt", "obsTimeStart"}
    extras = [c for c in df.columns if c not in consumed]
    return gpd.GeoDataFrame(
        {
            "id": df["pid"].astype("string"),
            "point_type": pd.Series(["gnss"] * len(df), dtype="string"),  # TODO(D2)
            "height": ortho,
            "height_datum": pd.Series(["NAVD88"] * len(df), dtype="string"),
            "horizontal_crs": h_crs,
            "vertical_crs": pd.Series(["EPSG:5703"] * len(df), dtype="string"),
            "ref_frame": ref_frame,
            "frame_epoch": frame_epoch,
            "coord_epoch": _num(df, "epoch").fillna(frame_epoch),
            "measurement_datetime": mdt,
            "measurement_epoch": decyear(mdt),
            "native_x": lon, "native_y": lat, "native_h": ortho,
            "native_crs": (h_crs + "+5703").astype("string"),
            "raw": pd.Series([json.dumps({k: str(df.iloc[i][k]) for k in extras})
                              for i in range(len(df))], dtype="string", index=df.index),
        },
        geometry=gpd.points_from_xy(lon, lat),
        crs=None,
        index=df.index,
    )
