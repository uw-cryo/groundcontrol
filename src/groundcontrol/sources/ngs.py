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


def _parse_recovered(s: pd.Series) -> pd.Series:
    """Parse NGS ``lastRecovered`` dates, which come in mixed precision.

    Real distribution (Casa Grande, 2026-07): 362x ``YYYYMMDD``, 136x ``YYYY``
    (year only), 2x blank. A strict ``%Y%m%d`` parse silently drops 28% of
    measurement dates. Year-only values are assigned **mid-year (July 2)** so
    the derived ``measurement_epoch`` is unbiased (max error ±0.5 yr).
    """
    s = s.astype("string").str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns, UTC]")
    n = s.str.len()
    out[n == 8] = pd.to_datetime(s[n == 8], format="%Y%m%d", utc=True, errors="coerce")
    out[n == 6] = pd.to_datetime(s[n == 6], format="%Y%m", utc=True, errors="coerce")
    out[n == 4] = pd.to_datetime(s[n == 4] + "0702", format="%Y%m%d", utc=True,
                                 errors="coerce")
    return out


def _vert_crs(vert_datum: pd.Series) -> pd.Series:
    """Honest per-row vertical CRS from the NGS ``vertDatum`` string.

    NAVD 88 -> EPSG:5703; NGVD 29 -> EPSG:7968 (m); blank/unknown -> NA (never
    assumed — downstream vertical reconciliation must refuse NA, not guess).
    """
    s = vert_datum.astype("string").str.upper().str.replace(" ", "", regex=False)
    out = pd.Series(pd.NA, index=vert_datum.index, dtype="string")
    out[s.str.contains("NAVD88", na=False)] = "EPSG:5703"
    out[s.str.contains("NGVD29", na=False)] = "EPSG:7968"
    return out


def parse_nde(records: list[dict]) -> gpd.GeoDataFrame:
    """NDE datasheet records -> schema-shaped native-frame GeoDataFrame."""
    df = pd.DataFrame(records)
    if df.empty:
        from groundcontrol import schema
        return schema.empty(crs="EPSG:6318")
    from groundcontrol.crs import ngs_datum_to_epsg

    lat, lon = _num(df, "lat"), _num(df, "lon")
    ortho = _num(df, "orthoHt")
    ref_frame, frame_epoch = _frame_fields(df.get("posDatum", pd.Series(index=df.index)))
    # B7: per-row native realization CRS (fail-loud on unrecognized strings)
    h_crs = ref_frame.map(ngs_datum_to_epsg).astype("string")
    mdt = _parse_recovered(df.get("lastRecovered", pd.Series(index=df.index)))
    consumed = {"pid", "lat", "lon", "orthoHt", "lastRecovered"}
    extras = [c for c in df.columns if c not in consumed]
    out = gpd.GeoDataFrame(
        {
            "id": df["pid"].astype("string"),
            "point_type": pd.Series(["monument"] * len(df), dtype="string"),  # TODO(D2)
            # orthoHt is the well-populated height; ellipHeight is often blank.
            # PUBLISHED orthometric heights are vertical-datum quantities and do
            # not change under horizontal realization transforms -> the B7
            # landing is horizontal-only for this path. Ellipsoidal heights are
            # NOT invariant (NADCON5 carries eht shifts; NCAT HARN->2011 moves
            # eht by -0.072 m at Casa Grande) — an ellipHeight path must
            # transform h during landing (B7b).
            "height": ortho,
            "height_datum": df.get("vertDatum", pd.Series(index=df.index)).astype("string"),
            "horizontal_crs": h_crs,
            "vertical_crs": _vert_crs(df.get("vertDatum", pd.Series(index=df.index))),
            "ref_frame": ref_frame,
            "frame_epoch": frame_epoch,
            "coord_epoch": _num(df, "epoch").fillna(frame_epoch),
            "measurement_datetime": mdt,
            "measurement_epoch": decyear(mdt),
            # acc_h: netAccHz is NGS-stated circular 95% network accuracy in cm
            # (api/nde/meta; dsdata.pdf p.17) — the one directly usable value.
            # acc_v: deliberately NaN — netAccU/netAccEh describe the ELLIPSOID
            # height, and NGS publishes no network accuracy for orthometric
            # heights (our `height`); orthometric accuracy comes from
            # vertSource/order semantics, pending D3 adjudication.
            # See docs/accuracy_conventions.md (WIP). TODO(D3)
            "acc_h": _num(df, "netAccHz") / 100.0,
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
            # OPUS peak-to-peak = RANGE (max−min) of the 3 single-CORS baseline
            # solutions, meters (api/opus/meta) — not σ, not 95%. NGS-endorsed
            # conversion σ ≈ P2P/1.6926 (Schwarz 2006) is deferred to D3;
            # native metric carried as-is. orthoHtP2p is already geoid-padded.
            # See docs/accuracy_conventions.md (WIP). TODO(D3)
            "acc_h": pd.concat([_num(df, "latP2p"), _num(df, "lonP2p")], axis=1).max(axis=1),
            "acc_v": _num(df, "orthoHtP2p"),
            "native_x": lon, "native_y": lat, "native_h": ortho,
            "native_crs": (h_crs + "+5703").astype("string"),
            "raw": pd.Series([json.dumps({k: str(df.iloc[i][k]) for k in extras})
                              for i in range(len(df))], dtype="string", index=df.index),
        },
        geometry=gpd.points_from_xy(lon, lat),
        crs=None,
        index=df.index,
    )


def expand_attributes(gdf, fields=None, prefix="ngs_"):
    """Lift raw datasheet JSON fields into real columns for filtering.

    Every source row keeps its full upstream record in ``raw`` (JSON string);
    systematic monument selection (e.g. isolating a calibration-range
    population by ``stamping``, or quality tiers by ``vertSource``) needs
    those fields as columns. Returns a copy of ``gdf`` with ``<prefix><field>``
    string columns (pd.NA where the field is absent, the row has no ``raw``,
    or the JSON does not parse). Numeric coercion is left to the caller —
    datasheet fields are strings with embedded blanks.

    Parameters
    ----------
    gdf : GeoDataFrame with a ``raw`` column (any source; non-JSON rows -> NA).
    fields : iterable of raw-record keys; default covers the monument
        identification + quality set used by the standard figures.
    prefix : column-name prefix guarding against schema collisions.
    """
    if fields is None:
        fields = ("name", "stamping", "monumentType", "setting", "stability",
                  "condition", "posSource", "posOrder", "vertSource",
                  "vertOrder", "ellipHeight", "geoidModel")
    out = gdf.copy()

    def _parse(r):
        if not isinstance(r, str):
            return {}
        try:
            rec = json.loads(r)
        except (TypeError, ValueError):
            return {}
        return rec if isinstance(rec, dict) else {}

    parsed = out["raw"].apply(_parse) if "raw" in out.columns else pd.Series([{}] * len(out), index=out.index)
    for f in fields:
        # object dtype + per-value str(): a numeric field must format the same
        # ("2", never "2.0") regardless of whether OTHER rows are missing it
        # (apply's dtype inference floats an int column that contains a None)
        vals = pd.Series([rec.get(f) for rec in parsed], index=out.index,
                         dtype=object)
        vals = vals.map(lambda v: (v.strip() if isinstance(v, str) else str(v))
                        if v is not None else None)
        out[prefix + f] = pd.Series(vals, index=out.index, dtype="string").replace("", pd.NA)
    return out
