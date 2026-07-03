"""Nevada Geodetic Lab (NGL) GNSS station positions — Increment 1.5a.

Global daily GNSS position time series from the University of Nevada, Reno
Geodetic Laboratory (plan Appendix A2; endpoints verified live 2026-06-26).
Citation: Blewitt G., Hammond W.C. & Kreemer C. (2018), *Harnessing the GPS
data explosion for interdisciplinary science*, Eos, 99,
https://doi.org/10.1029/2018EO104623. (MIDAS velocities — Blewitt et al.
2016 — arrive with Increment 1.5b.)

Pattern: one cached station-index GET (``DataHoldings.txt``) -> bbox +
temporal filter -> per-station ``tenv3`` fetch (throttled) -> position at the
requested epoch/time-range as the component-wise **median** of the
``_latitude/_longitude/__height`` columns inside a +/-30-day window (path A of
plan A2). With no epoch/time-range the median of the last 30 available
solutions is emitted (a "current position") and ``coord_epoch`` records the
median decimal year actually used.

Verified gotchas baked in:

- ``StaOrigName`` may contain spaces (e.g. ``FRE2 1291``) or be absent —
  index rows are parsed with a **bounded** ``line.split(None, 11)`` (12
  fields max), never naive whitespace tokenization (plan A2).
- Longitudes (index ``Long`` and tenv3 ``_longitude``) are wrapped 0-360;
  normalized via ``((lon + 180) % 360) - 180`` — **not** ``mod 360`` (plan B1).
- tenv3 URLs double the frame directory:
  ``gps_timeseries/<FRAME>/tenv3/<FRAME>/<SSSS>.tenv3`` (the flat path on the
  portal page is stale/404).
- IGS14/IGS20 EPSG codes are a pyproj dead end (zero-parameter ties crash or
  silently no-op) — data are labelled with the aliased ITRF codes
  (IGS14 -> ``EPSG:7912``, IGS20 -> ``EPSG:9989``) and the IGS name is kept in
  ``ref_frame`` as provenance (docs/crs_implementation.md §3).

Heights are **ellipsoidal** in the (dynamic) data frame — ``height_datum`` is
``"ellipsoidal"`` and ``vertical_crs`` carries the aliased 3D frame code; the
interim dispatcher landing is horizontal-only, so ``height`` rides through as
the native-frame ellipsoidal value with honest provenance labels.

**Antenna height (``ant_m`` in ``raw``; owner requirement 2026-07):** GNSS
antennas sit on tripods/pillars/masts ~1-2 m above the ground surface that a
lidar/stereo DSM/DTM actually sees. The median tenv3 ``_ant(m)`` of the used
window is carried per point as ``ant_m`` so the assessment path can remove
the antenna-vs-ground offset before differencing (note: NGL positions are
already antenna-reference-point solutions; ``ant_m`` documents the monument
setup — a 0.0 value means no antenna-height information, not "on the
ground"). Fine-resolution DSMs may also resolve the monument itself.

TODO(1.5b): steps.txt discontinuity handling (clip the median window to the
segment between adjacent equipment/earthquake steps, plan B10) and MIDAS
velocities (``vel_e/n/u`` + path-B extrapolation outside the station span).
"""

from __future__ import annotations

import io
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from groundcontrol.crs import decyear, decyear_inv
from groundcontrol.sources.checkpoints_3dep import cache_dir

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

INDEX_URL = "https://geodesy.unr.edu/NGLStationPages/DataHoldings.txt"
#: doubled frame directory — verified live 2026-06-26 (plan A2).
TENV3_URL = "https://geodesy.unr.edu/gps_timeseries/{frame}/tenv3/{frame}/{sta}.tenv3"

#: IGS frame -> aliased ITRF EPSG code (docs/crs_implementation.md §3).
#: NEVER use the EPSG IGS codes (9018/10178...): pyproj cannot instantiate the
#: zero-parameter ties and cs2cs silently no-ops them.
FRAME_TO_EPSG = {"IGS14": "EPSG:7912", "IGS20": "EPSG:9989"}

INDEX_MAX_AGE_DAYS = 7.0     # station index cache refresh threshold
WINDOW_DAYS = 30.0           # path-A half-width around the requested epoch
LAST_N_SOLUTIONS = 30        # "current position" window when no epoch given
MAX_WORKERS = 4              # per-station fetch throttle (plan A2)

#: tenv3 columns parse() requires after header cleaning (fail loud on drift).
_TENV3_REQUIRED = {"site", "date", "decyear", "ant", "sig_e", "sig_n", "sig_u",
                   "latitude", "longitude", "height"}


def _wrap_lon(lon):
    """Normalize longitude(s) to [-180, 180) — plan B1 (NOT ``mod 360``)."""
    return ((np.asarray(lon, dtype="float64") + 180.0) % 360.0) - 180.0


# ---------------------------------------------------------------------------
# Station index (DataHoldings.txt)
# ---------------------------------------------------------------------------

def parse_dataholdings(text: str) -> pd.DataFrame:
    """Parse the DataHoldings.txt station index.

    Columns: ``Sta Lat(deg) Long(deg) Hgt(m) X Y Z Dtbeg Dtend Dtmod NumSol
    StaOrigName``. ``StaOrigName`` may contain spaces or be absent entirely —
    every row is parsed with a bounded ``split(None, 11)`` (verified gotcha,
    plan A2: naive tokenization breaks ~row 1170). Longitude is 0-360 in the
    file and normalized here (plan B1).
    """
    rows = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.startswith("Sta "):
            continue  # blank / header
        parts = line.split(None, 11)  # bounded: StaOrigName keeps its spaces
        if len(parts) < 11:
            raise ValueError(
                f"DataHoldings.txt line {lineno} has {len(parts)} fields "
                f"(need >= 11): {line!r}"
            )
        rows.append({
            "sta": parts[0],
            "lat": float(parts[1]),
            "lon": float(_wrap_lon(float(parts[2]))),
            "hgt": float(parts[3]),
            "dtbeg": parts[7],
            "dtend": parts[8],
            "dtmod": parts[9],
            "num_sol": int(parts[10]),
            "sta_orig_name": parts[11].strip() if len(parts) == 12 else "",
        })
    df = pd.DataFrame(rows)
    if len(df):
        df["dtbeg"] = pd.to_datetime(df["dtbeg"], format="%Y-%m-%d", utc=True)
        df["dtend"] = pd.to_datetime(df["dtend"], format="%Y-%m-%d", utc=True)
    return df


def _load_index(url: str = INDEX_URL, max_age_days: float = INDEX_MAX_AGE_DAYS) -> pd.DataFrame:
    """Cached station index (~/.cache/groundcontrol; GROUNDCONTROL_CACHE_DIR
    override), refreshed when older than ``max_age_days``."""
    local = cache_dir() / "ngl_DataHoldings.txt"
    stale = (not local.exists()
             or (time.time() - local.stat().st_mtime) > max_age_days * 86400)
    if stale:
        logger.info("downloading %s -> %s", url, local)
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        local.write_text(r.text)
    return parse_dataholdings(local.read_text())


def _normalize_time_range(time_range) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(start, end) as UTC Timestamps; entries may be decimal years, strings,
    or datetimes."""
    if len(time_range) != 2:
        raise ValueError(f"time_range must be (start, end), got {time_range!r}")
    out = []
    for v in time_range:
        if isinstance(v, (int, float, np.floating)):
            v = decyear_inv(float(v))
        out.append(pd.to_datetime(v, utc=True))
    t0, t1 = out
    if t0 > t1:
        raise ValueError(f"time_range start {t0} is after end {t1}")
    return t0, t1


def _select_stations(index: pd.DataFrame, aoi_bounds_4326, epoch=None,
                     time_range=None) -> pd.DataFrame:
    """Bbox filter (post lon-normalization) + [Dtbeg, Dtend] temporal overlap."""
    minx, miny, maxx, maxy = (float(v) for v in aoi_bounds_4326)
    sel = index[(index["lon"] >= minx) & (index["lon"] <= maxx)
                & (index["lat"] >= miny) & (index["lat"] <= maxy)]
    if epoch is not None:
        t = pd.Timestamp(decyear_inv(float(epoch))).tz_localize("UTC")
        pad = pd.Timedelta(days=WINDOW_DAYS)
        sel = sel[(sel["dtbeg"] <= t + pad) & (sel["dtend"] >= t - pad)]
    elif time_range is not None:
        t0, t1 = _normalize_time_range(time_range)
        sel = sel[(sel["dtbeg"] <= t1) & (sel["dtend"] >= t0)]
    return sel.sort_values("sta").reset_index(drop=True)


def _station_meta(row) -> dict:
    """JSON-friendly index metadata carried into fetch()'s raw payload."""
    return {
        "sta": row["sta"],
        "index_lat": row["lat"],
        "index_lon": row["lon"],
        "index_hgt": row["hgt"],
        "dtbeg": str(pd.Timestamp(row["dtbeg"]).date()),
        "dtend": str(pd.Timestamp(row["dtend"]).date()),
        "num_sol": int(row["num_sol"]),
        "sta_orig_name": row["sta_orig_name"],
    }


# ---------------------------------------------------------------------------
# fetch — network half (index + per-station tenv3)
# ---------------------------------------------------------------------------

def fetch(aoi_bounds_4326, frame: str = "IGS14", epoch=None, time_range=None,
          max_stations: int | None = None) -> dict:
    """Fetch raw per-station NGL data for an AOI.

    Parameters
    ----------
    aoi_bounds_4326 : (minlon, minlat, maxlon, maxlat) in EPSG:4326 degrees.
    frame : ``"IGS14"`` or ``"IGS20"`` (NGL frame directory).
    epoch : optional target decimal year (path-A +/-30-day median window).
    time_range : optional (start, end) — decimal years, date strings, or
        datetimes; mutually exclusive with ``epoch``.
    max_stations : optional cap on the number of stations fetched (stations
        are sorted by ID for determinism; used by tests/previews).

    Returns the raw payload consumed by :func:`parse` (which is pure/offline):
    ``{"frame", "epoch", "time_range", "stations": [{"meta", "tenv3"}, ...]}``.
    """
    if frame not in FRAME_TO_EPSG:
        raise ValueError(f"unknown NGL frame {frame!r}; supported: {sorted(FRAME_TO_EPSG)}")
    if epoch is not None and time_range is not None:
        raise ValueError("pass either epoch or time_range, not both")
    index = _load_index()
    sel = _select_stations(index, aoi_bounds_4326, epoch=epoch, time_range=time_range)
    if max_stations is not None:
        sel = sel.iloc[:max_stations]
    logger.info("NGL: %d candidate station(s) in bbox %s (frame %s)",
                len(sel), tuple(aoi_bounds_4326), frame)

    def _get(row):
        url = TENV3_URL.format(frame=frame, sta=row["sta"])
        r = requests.get(url, timeout=120)
        if r.status_code == 404:
            # indexed station without a series in this frame directory —
            # skip with a loud warning (not silent: recorded in the log)
            logger.warning("NGL station %s: no %s tenv3 at %s (404); skipping",
                           row["sta"], frame, url)
            return None
        r.raise_for_status()
        return {"meta": _station_meta(row), "tenv3": r.text}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        results = list(ex.map(_get, (row for _, row in sel.iterrows())))
    stations = [s for s in results if s is not None]
    return {
        "frame": frame,
        "epoch": None if epoch is None else float(epoch),
        "time_range": None if time_range is None else tuple(time_range),
        "stations": stations,
    }


# ---------------------------------------------------------------------------
# parse — pure half (tenv3 -> position-at-epoch -> schema shape)
# ---------------------------------------------------------------------------

def parse_tenv3(text: str) -> pd.DataFrame:
    """Parse a tenv3 file into a DataFrame with cleaned column names.

    Header names like ``_latitude(deg)``/``____up(m)`` are cleaned by
    stripping leading underscores and the unit suffix; ``YYMMMDD`` is parsed
    to a UTC ``date`` column and ``yyyy.yyyy`` renamed ``decyear``.
    Longitude gets the B1 wrap. Rows are sorted by date.
    """
    df = pd.read_csv(io.StringIO(text), sep=r"\s+")
    df.columns = [re.sub(r"\(.*\)$", "", c).lstrip("_") for c in df.columns]
    df = df.rename(columns={"YYMMMDD": "date", "yyyy.yyyy": "decyear"})
    missing = _TENV3_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"tenv3 file missing expected columns {sorted(missing)}; "
                         f"got {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"], format="%y%b%d", utc=True)
    df["longitude"] = _wrap_lon(df["longitude"].to_numpy())
    return df.sort_values("date").reset_index(drop=True)


def _median_lon(lon: np.ndarray) -> float:
    """Median longitude, antimeridian-safe (a +/-180-straddling window would
    otherwise median to ~0)."""
    lon = np.asarray(lon, dtype="float64")
    if lon.max() - lon.min() > 180.0:
        return float(_wrap_lon(np.median(lon % 360.0)))
    return float(np.median(lon))


def _select_window(ts: pd.DataFrame, epoch=None, time_range=None) -> tuple[pd.DataFrame, dict]:
    """Select the solution window for position-at-epoch (path A).

    - ``epoch``: solutions within +/-``WINDOW_DAYS`` of the target date.
    - ``time_range``: solutions inside [start, end].
    - neither: the last ``LAST_N_SOLUTIONS`` available solutions
      ("current position").

    Returns ``(window_rows, window_descriptor)``; empty rows mean the station
    must be dropped (with a logged warning) by the caller.

    TODO(1.5b): clip the window to the steps.txt segment containing the
    target (equipment + earthquake discontinuities, plan B10).
    """
    if epoch is not None:
        t = pd.Timestamp(decyear_inv(float(epoch))).tz_localize("UTC")
        pad = pd.Timedelta(days=WINDOW_DAYS)
        win = ts[(ts["date"] >= t - pad) & (ts["date"] <= t + pad)]
        desc = {"mode": "epoch", "target_epoch": float(epoch),
                "half_width_days": WINDOW_DAYS}
    elif time_range is not None:
        t0, t1 = _normalize_time_range(time_range)
        win = ts[(ts["date"] >= t0) & (ts["date"] <= t1)]
        desc = {"mode": "time_range", "start": str(t0), "end": str(t1)}
    else:
        win = ts.iloc[-LAST_N_SOLUTIONS:]
        desc = {"mode": "last_n", "n": LAST_N_SOLUTIONS}
    if len(win):
        desc["first_used"] = str(win["date"].iloc[0].date())
        desc["last_used"] = str(win["date"].iloc[-1].date())
    return win, desc


def _position_from_window(win: pd.DataFrame) -> dict:
    """Component-wise median position + epoch bookkeeping for a non-empty window."""
    med_dy = float(np.median(win["decyear"].to_numpy()))
    # measurement_datetime = date of the used solution nearest the median epoch
    nearest = win.iloc[int(np.argmin(np.abs(win["decyear"].to_numpy() - med_dy)))]
    return {
        "lat": float(np.median(win["latitude"].to_numpy())),
        "lon": _median_lon(win["longitude"].to_numpy()),
        "height": float(np.median(win["height"].to_numpy())),
        "coord_epoch": med_dy,
        "measurement_datetime": nearest["date"],
        "n_solutions_used": int(len(win)),
        # per-solution formal sigmas: medians go to raw, not acc_* (TODO(D3))
        "sig_e_m": float(np.median(win["sig_e"].to_numpy())),
        "sig_n_m": float(np.median(win["sig_n"].to_numpy())),
        "sig_u_m": float(np.median(win["sig_u"].to_numpy())),
        # antenna height above the monument/ground — owner requirement: the
        # assessment must correct for antenna-vs-ground offset (module docstring)
        "ant_m": float(np.median(win["ant"].to_numpy())),
    }


def parse(raw: dict) -> gpd.GeoDataFrame:
    """Raw fetch() payload -> schema-shaped native-frame GeoDataFrame.

    Pure/offline: consumes the ``{"frame", "epoch", "time_range", "stations"}``
    dict. Stations whose solution window is empty are dropped with a logged
    warning (never silently interpolated — path-B MIDAS extrapolation is
    TODO(1.5b)).
    """
    frame = raw["frame"]
    if frame not in FRAME_TO_EPSG:
        raise ValueError(f"unknown NGL frame {frame!r}; supported: {sorted(FRAME_TO_EPSG)}")
    crs_code = FRAME_TO_EPSG[frame]  # aliased ITRF code (crs_implementation §3)
    records = []
    for station in raw["stations"]:
        meta = station["meta"]
        ts = parse_tenv3(station["tenv3"])
        win, window_desc = _select_window(ts, epoch=raw.get("epoch"),
                                          time_range=raw.get("time_range"))
        if not len(win):
            logger.warning(
                "NGL station %s: no solutions in the requested window %s "
                "(station span %s..%s); dropping", meta["sta"], window_desc,
                meta.get("dtbeg"), meta.get("dtend"))
            continue
        pos = _position_from_window(win)
        records.append({
            "id": meta["sta"],
            "point_type": "gnss",  # TODO(D2)
            "height": pos["height"],            # ELLIPSOIDAL, native frame
            "height_datum": "ellipsoidal",
            "horizontal_crs": crs_code,
            "vertical_crs": crs_code,           # 3D frame code carries the vertical
            "ref_frame": frame,                 # IGS provenance (aliased at ingestion)
            "frame_epoch": np.nan,              # dynamic frame: no reference epoch
            "coord_epoch": pos["coord_epoch"],  # feeds the 4D tt (TODO(D6))
            "measurement_datetime": pos["measurement_datetime"],
            "measurement_epoch": decyear(pos["measurement_datetime"]),
            # acc_h/acc_v deliberately NaN: sig_e/n/u are per-solution formal
            # sigmas, not a calibrated accuracy — medians carried in raw. TODO(D3)
            "acc_h": np.nan,
            "acc_v": np.nan,
            "vel_e": np.nan, "vel_n": np.nan, "vel_u": np.nan,  # TODO(1.5b) MIDAS
            "native_x": pos["lon"],
            "native_y": pos["lat"],
            "native_h": pos["height"],
            "native_crs": crs_code,
            "raw": json.dumps({
                "sta_orig_name": meta.get("sta_orig_name", ""),
                "dtbeg": meta.get("dtbeg"),
                "dtend": meta.get("dtend"),
                "num_sol": meta.get("num_sol"),
                "n_solutions_used": pos["n_solutions_used"],
                "window": window_desc,
                "sig_e_m": pos["sig_e_m"],
                "sig_n_m": pos["sig_n_m"],
                "sig_u_m": pos["sig_u_m"],
                # antenna height (m) above the ground/monument the DSM/DTM
                # sees — assessment must remove it before differencing
                "ant_m": pos["ant_m"],
            }),
        })
    if not records:
        from groundcontrol import schema
        return schema.empty(crs=None)
    df = pd.DataFrame.from_records(records)
    for col in ("id", "point_type", "height_datum", "horizontal_crs",
                "vertical_crs", "ref_frame", "native_crs", "raw"):
        df[col] = df[col].astype("string")
    df["measurement_datetime"] = pd.to_datetime(df["measurement_datetime"], utc=True)
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["native_x"], df["native_y"]),
        # native dynamic-frame coordinates; the dispatcher lands them
        # (crs.land_horizontal passes per-row coord_epoch as tt — TODO(D6))
        crs=None,
    )
