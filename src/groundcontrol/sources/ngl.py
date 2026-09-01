"""Nevada Geodetic Lab (NGL) GNSS station positions — Increment 1.5a.

Global daily GNSS position time series from the University of Nevada, Reno
Geodetic Laboratory (plan Appendix A2; endpoints verified live 2026-06-26).
Citation: Blewitt G., Hammond W.C. & Kreemer C. (2018), *Harnessing the GPS
data explosion for interdisciplinary science*, Eos, 99,
https://doi.org/10.1029/2018EO104623. Station-level MIDAS velocities (Blewitt
et al. 2016) are downloaded/parsed by :func:`read_midas` and, when
``fetch(with_velocities=True)`` (the default), joined by station ID into each
station's own per-point ``vel_e/n/u`` (the tier-1 velocity
:func:`groundcontrol.crs.propagate_epoch` consumes). Spatial interpolation of
this network to arbitrary (non-station) points lives in
:mod:`groundcontrol.velocity`. Path-B MIDAS extrapolation of a *position*
outside a station's daily-series span remains TODO(1.5b).

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

Step discontinuities (plan 1.5b/B10): :func:`read_steps` fetches/parses the
``steps.txt`` database of equipment (type 1) and earthquake (type 2) offsets
so callers can split a station's series into clean segments (abrupt height
jumps from antenna swaps must never leak into positions or rates).

TODO(1.5b): step-aware window clipping inside :func:`_select_window` (clip
the median window to the segment between adjacent steps, plan B10) and path-B
MIDAS *position* extrapolation outside a station's daily-series span (the
per-point velocity join itself is done — see the MIDAS note above).
"""

from __future__ import annotations

import io
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from groundcontrol.crs import decyear, decyear_inv
from groundcontrol.sources.checkpoints_3dep import cache_dir, cache_stale, cache_write

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

INDEX_URL = "https://geodesy.unr.edu/NGLStationPages/DataHoldings.txt"
#: doubled frame directory — verified live 2026-06-26 (plan A2).
TENV3_URL = "https://geodesy.unr.edu/gps_timeseries/{frame}/tenv3/{frame}/{sta}.tenv3"
#: station step (discontinuity) database — plan A2/B10.
STEPS_URL = "https://geodesy.unr.edu/NGLStationPages/steps.txt"
#: MIDAS velocity solutions (Blewitt et al. 2016). Lives under /velocities/ —
#: the gps_timeseries/<FRAME>/midas/ path linked from the portal 404s
#: (verified live 2026-07-04).
MIDAS_URL = "https://geodesy.unr.edu/velocities/midas.{frame}.txt"

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
    stale = cache_stale(local, max_age_days)
    if stale:
        logger.info("downloading %s -> %s", url, local)
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        cache_write(local, r.text)
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

#: MIDAS is published for IGS14 only (``midas.IGS20.txt`` does not exist —
#: read_midas docstring). Inter-realization velocity differences (ITRF2014 vs
#: ITRF2020) are sub-mm/yr, so IGS14 MIDAS velocities are reused for an IGS20
#: source with a logged note; the residual (<~0.7 mm/yr x delta-t) is well
#: below the decimetre budget the epoch propagation targets.
_MIDAS_FRAME = "IGS14"


def _midas_velocity_map(frame: str) -> dict:
    """``{station_id: {vel_e/n/u, sig_vel_e/n/u}}`` from MIDAS (cached), or ``{}``.

    Source-agnostic velocities are keyed by the 4-char station ID for the
    station-ID join in :func:`fetch` -> :func:`parse` (each GNSS mark carries
    its OWN MIDAS velocity). Never raises: a missing/unreachable MIDAS file
    logs a warning and yields an empty map (velocities stay NaN — honest).
    """
    if frame != _MIDAS_FRAME:
        logger.info("NGL: MIDAS published for %s only; reusing %s velocities for "
                    "frame %s (sub-mm/yr inter-realization difference)",
                    _MIDAS_FRAME, _MIDAS_FRAME, frame)
    try:
        m = read_midas(frame=_MIDAS_FRAME)
    except Exception as e:  # network/404/parse — degrade to no velocities
        logger.warning("NGL: MIDAS velocities unavailable (%s: %s); vel_e/n/u stay NaN",
                       type(e).__name__, e)
        return {}
    cols = ["vel_e", "vel_n", "vel_u", "sig_vel_e", "sig_vel_n", "sig_vel_u"]
    return {str(r["sta"]): {c: float(r[c]) for c in cols} for _, r in m.iterrows()}


def _attach_networks(stations) -> None:
    """Attach corroborated network memberships to each station's meta.

    Membership is supplementary curation evidence with an explicit "not
    checked" state (raw ``networks=null``), so registry-list failures
    degrade with a loud warning instead of aborting the position fetch —
    the ``_midas_velocity_map`` pattern (audit finding: the original
    inline block let a geodesy.noaa.gov / network.igs.org outage kill the
    whole per-AOI fetch). ``meta['networks']`` is simply left unset on
    failure, which parse() records as null.
    """
    from groundcontrol import networks  # lazy: avoid import cycles
    tables = {}
    # per-registry degrade (audit round 4): one registry being down must
    # not discard the other's successfully fetched evidence
    for key in networks.NETWORKS:
        try:
            tables[key] = networks.NETWORKS[key]()
        except Exception as e:  # noqa: BLE001 — degrade, never fabricate
            logger.warning(
                "network list %r unavailable (%s: %s); membership for it "
                "stays unchecked", key, type(e).__name__, e)
    if not tables:
        logger.warning("no curated-network lists available; stations will "
                       "record networks=null (= not checked)")
        return
    for s in stations:
        m = s["meta"]
        m["networks"] = networks.membership(
            m["sta"], m["index_lat"], m["index_lon"], tables)
        # which registries the membership list actually consulted — a
        # partial check must not read as "checked everywhere"
        m["networks_checked"] = sorted(tables)


def _attach_steps(stations) -> None:
    """Attach each station's earthquake-step epochs (steps.txt type 2) to
    ``meta['eq_steps']`` as sorted decimal years — the evidence
    ``crs.propagate_epoch``'s step guard consumes (owner 2026-08-30).
    ``[]`` = checked, no earthquake steps (verified honest: steps.txt only
    lists stations WITH steps); absent/None = NOT checked (a steps fetch
    failure degrades with a loud warning, never aborts the position fetch —
    the _attach_networks pattern). ``meta['eq_steps_through']`` carries the
    catalog vintage (the file's latest type-2 date, decimal year) so the
    guard can refuse to call an interval extending past it "checked"
    (audit 2026-08-30: a fresh event would otherwise read as clean).
    Type-1 (equipment) steps are deliberately OUT of scope here: they are
    instrumental height jumps, not ground displacement — handled by the
    position-window/ant_m machinery, not epoch propagation."""
    try:
        steps = read_steps()
        if not len(steps):
            # the real catalog is ~142k rows; zero rows = empty/truncated
            # cache (interrupted download). Treating it as "checked, no
            # steps" silently defeated the Gorkha guard: through=None also
            # disabled the vintage backstop, so a 2014 point propagated
            # across the 2015 steps with zero warnings.
            raise ValueError("steps.txt parsed to zero rows (empty or "
                             "truncated cache?) — refusing to read that "
                             "as 'checked, no steps'")
        # vintage from the FULL catalog; per-station epochs only for the
        # FETCHED stations (profiling 2026-08-30: converting the whole
        # 142k-row catalog through per-row decyear() burned ~90 s of CPU
        # per fetch and GIL-convoyed the other sources)
        eq_all = steps[steps["type"] == 2]
        through = decyear(eq_all["date"].max()) if len(eq_all) else None
        want = {s["meta"]["sta"] for s in stations}
        mine = steps[steps["sta"].isin(want)]
        eq = mine[mine["type"] == 2]
        per_sta = {sta: sorted(decyear(d) for d in grp["date"])
                   for sta, grp in eq.groupby("sta")}
        equip = mine[mine["type"] == 1]
        per_sta_eqp = {sta: sorted(decyear(d) for d in grp["date"])
                       for sta, grp in equip.groupby("sta")}
    except Exception as e:
        logger.warning("NGL steps.txt unavailable (%s: %s): eq_steps not "
                       "attached — propagate_epoch cannot step-check these "
                       "rows", type(e).__name__, e)
        return
    for s in stations:
        s["meta"]["eq_steps"] = per_sta.get(s["meta"]["sta"], [])
        s["meta"]["eq_steps_through"] = through
        # type-1 EQUIPMENT steps (antenna/radome changes): instrumental
        # height jumps — figure annotation evidence, never a propagation
        # guard input (owner 2026-08-30 station-series figure)
        s["meta"]["equip_steps"] = per_sta_eqp.get(s["meta"]["sta"], [])


def fetch(aoi_bounds_4326, frame: str = "IGS14", epoch=None, time_range=None,
          max_stations: int | None = None, with_velocities: bool = True,
          with_networks: bool = True, with_steps: bool = True) -> dict:
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
    with_velocities : attach each station's own MIDAS ENU velocity (one extra
        cached GET) to ``meta['midas']`` for :func:`parse` -> ``vel_e/n/u``.
        Default True; set False to skip the MIDAS fetch entirely.
    with_networks : attach corroborated curated-network memberships
        (``groundcontrol.networks``: ID join + coordinate check against each
        registry list, one cached GET per network) to ``meta['networks']``
        for :func:`parse` -> ``raw["networks"]``. Default True; set False to
        skip — the raw evidence then records null (= not checked). List
        failures degrade to the same null with a loud warning; they never
        abort the position fetch (see :func:`_attach_networks`).
    with_steps : attach each station's earthquake-step epochs from the
        cached ``steps.txt`` (one GET at most) to ``meta['eq_steps']`` for
        :func:`parse` -> ``raw["eq_steps"]`` — consumed by
        ``crs.propagate_epoch``'s step guard. Same degrade contract:
        failure -> null (= not checked) with a loud warning.

    Returns the raw payload consumed by :func:`parse` (which is pure/offline):
    ``{"frame", "epoch", "time_range", "stations": [{"meta", "tenv3"}, ...]}``
    where each ``meta`` carries an optional ``midas`` velocity sub-dict and
    an optional ``networks`` membership list.
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

    logger.info("NGL: fetching %d daily series (tenv3, %d at a time from "
                "geodesy.unr.edu — the slow part; each is cached for later "
                "runs)", len(sel), MAX_WORKERS)

    def _get(row):
        try:
            text = _tenv3_text(row["sta"], frame)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                # indexed station without a series in this frame directory
                # — skip with a loud warning (recorded in the log)
                logger.warning("NGL station %s: no %s tenv3 (404); skipping",
                               row["sta"], frame)
                return None
            raise
        logger.info("NGL: %s series ready", row["sta"])
        return {"meta": _station_meta(row), "tenv3": text}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS + 2) as ex:
        # warm the big shared catalogs CONCURRENTLY with the per-station
        # pulls (owner 2026-08-30: the ~40 MB steps catalog + MIDAS table
        # downloaded serially AFTER the tenv3s and read as a hang): the
        # attach helpers then re-read the warm disk cache in seconds.
        # Failures are swallowed here on purpose — each attach path
        # handles and logs its own degrade mode.
        warm = []
        if with_velocities:
            warm.append(ex.submit(_midas_text, _MIDAS_FRAME))
        if with_steps:
            warm.append(ex.submit(_steps_text))
        results = list(ex.map(_get, (row for _, row in sel.iterrows())))
        for f in warm:
            try:
                f.result()
            except Exception:
                pass
    stations = [s for s in results if s is not None]
    if not stations:
        # nothing to enrich: skip the MIDAS/network/steps catalog fetches
        # entirely (owner 2026-08-30: a 0-candidate Alaska tile hung for
        # minutes downloading the full steps catalog for an empty list)
        return {"frame": frame,
                "epoch": None if epoch is None else float(epoch),
                "time_range": None if time_range is None
                else tuple(time_range),
                "stations": []}
    if with_velocities:
        vmap = _midas_velocity_map(frame)
        for s in stations:
            s["meta"]["midas"] = vmap.get(s["meta"]["sta"])  # None if absent
    if with_networks:
        _attach_networks(stations)
    if with_steps:
        _attach_steps(stations)
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


def read_tenv3(station: str, frame: str = "IGS14",
               max_age_days: float = INDEX_MAX_AGE_DAYS) -> pd.DataFrame:
    """Full daily position time series for one NGL station (cached download).

    The station-level companion to :func:`fetch` (which reduces each series to
    a position-at-epoch): fetches the complete ``tenv3`` file for ``station``
    in ``frame`` from :data:`TENV3_URL` (doubled frame directory — verified
    gotcha, module docstring), caches the text in the groundcontrol cache dir
    (``ngl_<STA>_<frame>.tenv3``; refreshed when older than ``max_age_days``),
    and returns :func:`parse_tenv3`'s cleaned, date-sorted DataFrame.

    Useful columns: ``date``/``decyear`` (plan-B9 pair), ``east``/``north``/
    ``up`` (m, fractional parts relative to the integer ``e0``/``n0``/``u0``
    references), ``sig_e``/``sig_n``/``sig_u`` per-solution formal sigmas,
    ``ant`` antenna height, and full ``latitude``/``longitude``/``height``.

    Raises ``ValueError`` for an unknown frame and ``requests.HTTPError`` for
    a missing station/series (404: indexed station without a series in that
    frame directory).
    """
    if frame not in FRAME_TO_EPSG:
        raise ValueError(f"unknown NGL frame {frame!r}; supported: {sorted(FRAME_TO_EPSG)}")
    return parse_tenv3(_tenv3_text(station, frame, max_age_days))


def _tenv3_text(station: str, frame: str,
                max_age_days: float = INDEX_MAX_AGE_DAYS) -> str:
    """One station's raw tenv3 text, through the per-station disk cache
    (``ngl_<STA>_<frame>.tenv3``, DataHoldings staleness pattern). Shared
    by :func:`read_tenv3` AND :func:`fetch` (owner 2026-08-30: fetch
    bypassed the cache, so every assess run re-downloaded every series —
    ~100 s for 9 stations — and the figure stage then downloaded them all
    AGAIN through read_tenv3). Raises ``requests.HTTPError`` on 404."""
    station = str(station).strip().upper()
    local = cache_dir() / f"ngl_{station}_{frame}.tenv3"
    stale = cache_stale(local, max_age_days)
    if stale:
        url = TENV3_URL.format(frame=frame, sta=station)
        logger.info("downloading %s -> %s", url, local)
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        cache_write(local, r.text)
    return local.read_text()


def parse_steps(text: str) -> pd.DataFrame:
    """Parse the NGL ``steps.txt`` discontinuity database (plan 1.5b/B10).

    Two whitespace-delimited row shapes share the first three columns
    ``site  YYMMMDD  type``; the trailing fields vary by type, so every row
    is split with a **bounded** ``split(None, 3)`` first (the DataHoldings
    lesson: never naive whitespace tokenization):

    - type ``1`` (equipment change, plan A2): one trailing field -> ``event``
      (e.g. ``Antenna_Type_Changed``; may be ``Unknown``).
    - type ``2`` (earthquake): four trailing fields -> ``threshold_km`` (the
      magnitude-dependent inclusion radius), ``distance_km`` (station to
      epicenter), ``magnitude``, ``event_id`` (USGS).

    Returns a DataFrame with columns ``sta, date, type, event, threshold_km,
    distance_km, magnitude, event_id`` sorted by station then date; the
    type-specific columns are NA where they do not apply. ``YYMMMDD`` uses a
    2-digit year (``%y`` pivot: 69-99 -> 19xx), same convention as tenv3.
    """
    rows = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split(None, 3)  # bounded: trailing fields vary by type
        if len(parts) < 4:
            raise ValueError(f"steps.txt line {lineno} has {len(parts)} fields "
                             f"(need >= 4): {line!r}")
        sta, date, typ, rest = parts[0], parts[1], parts[2], parts[3]
        if typ == "1":
            rows.append({"sta": sta, "date": date, "type": 1,
                         "event": rest.strip()})
        elif typ == "2":
            tail = rest.split(None, 3)
            if len(tail) != 4:
                raise ValueError(f"steps.txt line {lineno}: type-2 row needs 4 "
                                 f"trailing fields, got {len(tail)}: {line!r}")
            rows.append({"sta": sta, "date": date, "type": 2,
                         "threshold_km": float(tail[0]),
                         "distance_km": float(tail[1]),
                         "magnitude": float(tail[2]),
                         "event_id": tail[3].strip()})
        else:
            raise ValueError(f"steps.txt line {lineno}: unknown step type "
                             f"{typ!r} (expected 1 or 2): {line!r}")
    df = pd.DataFrame(rows, columns=["sta", "date", "type", "event",
                                     "threshold_km", "distance_km",
                                     "magnitude", "event_id"])
    if len(df):
        df["date"] = pd.to_datetime(df["date"], format="%y%b%d", utc=True)
        for col in ("sta", "event", "event_id"):
            df[col] = df[col].astype("string")
    return df.sort_values(["sta", "date"], kind="stable").reset_index(drop=True)


def _steps_text(max_age_days: float = INDEX_MAX_AGE_DAYS) -> str:
    """Raw steps.txt through the disk cache — download only, NO parsing
    (the I/O-only warmer :func:`fetch` runs concurrently; parsing 40 MB
    of text inside the pool convoys the GIL against other sources)."""
    local = cache_dir() / "ngl_steps.txt"
    stale = cache_stale(local, max_age_days)
    if stale:
        logger.info("downloading %s -> %s (large catalog; first run or "
                    "stale cache — subsequent runs read the local copy)",
                    STEPS_URL, local)
        r = requests.get(STEPS_URL, timeout=120)
        r.raise_for_status()
        cache_write(local, r.text)
    return local.read_text()


def read_steps(station: str | None = None,
               max_age_days: float = INDEX_MAX_AGE_DAYS) -> pd.DataFrame:
    """Station step (discontinuity) table — cached download (plan 1.5b/B10).

    Fetches :data:`STEPS_URL` once and caches the text in the groundcontrol
    cache dir (``ngl_steps.txt``, refreshed when older than ``max_age_days``
    — the DataHoldings pattern), returning :func:`parse_steps`'s DataFrame,
    optionally filtered to one ``station`` (ID normalized to upper case).

    Plan 1.5b / Appendix B10: step dates split a station's daily series into
    clean segments — a position window or velocity fit must not straddle a
    type-1 equipment change (antenna/receiver swaps cause abrupt, purely
    instrumental height jumps) or a type-2 earthquake offset. This delivers
    the data half; the step-aware window clipping in :func:`_select_window`
    remains TODO(1.5b).
    """
    steps = parse_steps(_steps_text(max_age_days))
    if station is not None:
        sta = str(station).strip().upper()
        steps = steps[steps["sta"] == sta].reset_index(drop=True)
    return steps


#: midas.<frame>.txt column layout, from the official readme
#: (https://geodesy.unr.edu/velocities/midas.readme.txt) and verified against
#: the live IGS14 file 2026-07-04 (every row has exactly 27 fields).
#: Velocities/uncertainties are m/yr; ``n_steps`` is the number of steps
#: ASSUMED by MIDAS from the steps.txt database — the only per-station
#: step-count metadata NGL publishes beyond steps.txt itself.
_MIDAS_COLUMNS = [
    "sta", "version", "t0", "t1", "duration_yr",          # 1-5
    "n_epochs", "n_good", "n_pairs",                      # 6-8
    "vel_e", "vel_n", "vel_u",                            # 9-11  (m/yr)
    "sig_vel_e", "sig_vel_n", "sig_vel_u",                # 12-14 (m/yr)
    "off_e", "off_n", "off_u",                            # 15-17 offset @ t0 (m)
    "frac_out_e", "frac_out_n", "frac_out_u",             # 18-20
    "sd_pairs_e", "sd_pairs_n", "sd_pairs_u",             # 21-23
    "n_steps",                                            # 24
    "lat", "lon", "hgt",                                  # 25-27
]


def parse_midas(text: str) -> pd.DataFrame:
    """Parse a MIDAS velocity file (``midas.<frame>.txt``) into a DataFrame.

    Column layout per :data:`_MIDAS_COLUMNS` (readme verified against the
    live file). Headerless whitespace-delimited rows; every row must carry
    exactly 27 fields (fail loud on layout drift — the DataHoldings lesson).
    Longitude comes 0-360-ish continuous (observed < -180 too) and gets the
    plan-B1 wrap.
    """
    df = pd.read_csv(io.StringIO(text), sep=r"\s+", header=None)
    if df.shape[1] != len(_MIDAS_COLUMNS):
        raise ValueError(
            f"MIDAS file has {df.shape[1]} columns; expected "
            f"{len(_MIDAS_COLUMNS)} ({_MIDAS_COLUMNS})"
        )
    df.columns = _MIDAS_COLUMNS
    bad = df.isna().any(axis=1)
    if bad.any():
        raise ValueError(
            f"MIDAS file has {int(bad.sum())} short/unparseable row(s), "
            f"first at line {int(np.flatnonzero(bad)[0]) + 1}"
        )
    for col in ("sta", "version"):
        df[col] = df[col].astype("string")
    for col in ("n_epochs", "n_good", "n_pairs", "n_steps"):
        df[col] = df[col].astype("int64")
    df["lon"] = _wrap_lon(df["lon"].to_numpy())
    return df.sort_values("sta", kind="stable").reset_index(drop=True)


def read_midas(frame: str = "IGS14",
               max_age_days: float = INDEX_MAX_AGE_DAYS) -> pd.DataFrame:
    """MIDAS station velocities — cached download (Blewitt et al. 2016).

    NGL's step-resistant velocity estimator (median of all data-pair slopes;
    steps assumed at steps.txt dates — no step detection). Fetches
    :data:`MIDAS_URL` for ``frame`` once, caches the text in the
    groundcontrol cache dir (``ngl_midas_<frame>.txt``, refreshed when older
    than ``max_age_days`` — the DataHoldings pattern) and returns
    :func:`parse_midas`'s DataFrame (velocities in **m/yr**; see
    :data:`_MIDAS_COLUMNS`).

    ``frame="IGS14"`` is verified live; NGL also publishes plate-fixed
    variants (``NA``, ``PA``, ...) at the same URL pattern — an unknown code
    fails loud with an HTTPError 404 (``midas.IGS20.txt`` does NOT exist as
    of 2026-07-04). Note the IGS14 file is a full-network weekly product:
    stations absent from it (e.g. too-new stations) simply have no MIDAS
    velocity yet.
    """
    frame = str(frame).strip()
    if not frame:
        raise ValueError("frame must be a non-empty MIDAS frame code, e.g. 'IGS14'")
    return parse_midas(_midas_text(frame, max_age_days))


def _midas_text(frame: str,
                max_age_days: float = INDEX_MAX_AGE_DAYS) -> str:
    """Raw MIDAS table text through the disk cache — download only (the
    I/O-only warmer; see :func:`_steps_text`)."""
    local = cache_dir() / f"ngl_midas_{frame}.txt"
    stale = cache_stale(local, max_age_days)
    if stale:
        url = MIDAS_URL.format(frame=frame)
        logger.info("downloading %s -> %s", url, local)
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        cache_write(local, r.text)
    return local.read_text()


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


# ------------------------------------------------------------------------
# Per-row occupation class (owner taxonomy, 2026-08-22)
# ------------------------------------------------------------------------
#: Occupation-class thresholds, applied to each station's OWN DataHoldings
#: record — never inferred from the source archive. Evidence basis
#: (sandbox/gnss_class_study.py, archive-wide run 2026-08-14, 23,693
#: stations): 95% of labeled NGS CORS sit at solution density >= 0.62 (P5)
#: with span >= 2.1 yr, so >= 0.6 marks demonstrated continuous operation;
#: the episodic cluster separates from the continuous mode across a broad
#: low-density trough (~0.05-0.3), and 0.2 is the round-number cut inside
#: it. The 0.2-0.6 band is a DECLARED buffer ("semi-continuous"), not a
#: measured population boundary. Terminology: "campaign" and "continuous
#: station" follow the GAGE/EarthScope glossary
#: (https://www.unavco.org/help/glossary/glossary.html); "semi-continuous"
#: is established community usage for the in-between band (no glossary
#: entry). Owner-confirmed 2026-08-22, explicitly adjustable.
CAMPAIGN_MAX_SPAN_YR = 1.0
CAMPAIGN_MAX_DENSITY = 0.2
CONTINUOUS_MIN_DENSITY = 0.6


def occupation_evidence(dtbeg, dtend, num_sol):
    """(span_yr, density) from a station's DataHoldings fields.

    THE archive-study metrics, factored so :func:`parse`, tests, and any
    re-derivation tool share one implementation: ``span_yr`` = (dtend -
    dtbeg) in years; ``density`` = solutions per day of inclusive span
    (``num_sol / (span_days + 1)`` — the +1 also covers a single-day record
    with denominator 1; the study script pinned single-day density to 1.0
    for its histogram, but pinning FABRICATES evidence, e.g. "1.0" for a
    zero-solution placeholder record, so raw carries the measured ratio —
    audit round 3). Fail-loud: ``dtend`` before ``dtbeg`` is corrupt
    evidence and raises — the old ``span_d > 0`` guard silently fabricated
    density for negative spans too (audit round 1).
    """
    span_d = (pd.Timestamp(dtend) - pd.Timestamp(dtbeg)).days
    if span_d < 0:
        raise ValueError(
            f"dtend {dtend!r} precedes dtbeg {dtbeg!r} (span {span_d} d) — "
            "corrupt DataHoldings evidence")
    return span_d / 365.25, num_sol / (span_d + 1)


def occupation_class(span_yr, density):
    """Occupation class demonstrated by a station's own archive record.

    ``span_yr``/``density`` come from :func:`occupation_evidence` and are
    carried UNROUNDED in ``raw`` so the class stays re-derivable without a
    refetch (rounding flipped the class at the 0.2/0.6 knife edges, audit
    finding). Returns
    ``"gnss_campaign"`` (span < 1 yr OR density < 0.2: the record
    demonstrates episodic occupation), ``"gnss_semicont"`` (0.2-0.6 buffer
    band), or ``"gnss_cont"`` (density >= 0.6). Fail-loud: non-finite
    evidence raises instead of defaulting a class.

    The class is an OCCUPATION-PATTERN claim only, never a quality claim
    (owner, 2026-08-22): a continuous record can come from an excellent
    CORS monument or from a problematic station (rooftop/mast ARP,
    unstable or on-ice monument, nonlinear motion). Judge quality from
    separate evidence — monument/stability metadata, MIDAS behavior and
    steps.txt, CORS membership, dh consistency — never from this label.
    """
    span_yr, density = float(span_yr), float(density)
    if not (np.isfinite(span_yr) and np.isfinite(density)) or span_yr < 0:
        raise ValueError(
            f"occupation_class needs finite non-negative evidence, got "
            f"span_yr={span_yr!r} density={density!r} — DataHoldings record "
            "incomplete or corrupt")
    if span_yr < CAMPAIGN_MAX_SPAN_YR or density < CAMPAIGN_MAX_DENSITY:
        return "gnss_campaign"
    if density < CONTINUOUS_MIN_DENSITY:
        return "gnss_semicont"
    return "gnss_cont"


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
        # station's OWN MIDAS ENU velocity (m/yr), station-ID join attached by
        # fetch(with_velocities=True). None -> vel_e/n/u stay NaN (honest: the
        # station has no MIDAS solution yet, e.g. too-new). This is the tier-1
        # per-point velocity propagate_epoch consumes for the station itself;
        # arbitrary (non-station) points are filled by spatial interpolation of
        # this same network in groundcontrol.velocity (fill_velocities).
        mv = meta.get("midas") or {}
        # Per-row occupation class from the station's own DataHoldings
        # record (never from the source archive). Corrupt/missing evidence
        # drops THAT station with a loud warning — the empty-window
        # convention above (audit round 4: raising here let one malformed
        # index row abort the whole AOI, which the dispatcher's per-source
        # catch then reduced to a near-silent n_rows=0).
        if any(meta.get(k) is None for k in ("dtbeg", "dtend", "num_sol")):
            logger.warning(
                "NGL station %s: DataHoldings evidence (dtbeg/dtend/"
                "num_sol) missing — cannot assign an occupation class; "
                "dropping station", meta.get("sta"))
            continue
        try:
            span_yr, density = occupation_evidence(
                meta["dtbeg"], meta["dtend"], meta["num_sol"])
        except ValueError as e:
            logger.warning("NGL station %s: %s; dropping station",
                           meta.get("sta"), e)
            continue
        records.append({
            "id": meta["sta"],
            "point_type": occupation_class(span_yr, density),  # TODO(D2)
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
            "vel_e": mv.get("vel_e", np.nan),   # MIDAS (Blewitt et al. 2016)
            "vel_n": mv.get("vel_n", np.nan),
            "vel_u": mv.get("vel_u", np.nan),
            "native_x": pos["lon"],
            "native_y": pos["lat"],
            "native_h": pos["height"],
            "native_crs": crs_code,
            "raw": json.dumps({
                "sta_orig_name": meta.get("sta_orig_name", ""),
                "dtbeg": meta.get("dtbeg"),
                "dtend": meta.get("dtend"),
                "num_sol": meta.get("num_sol"),
                # occupation-class evidence (occupation_evidence), stored
                # UNROUNDED: point_type must re-derive exactly from these
                # (rounding flipped knife-edge classes, audit finding)
                "span_yr": span_yr,
                "density": density,
                # corroborated curated-network memberships (networks.py):
                # [] = checked, member of none; null = fetch did not check;
                # networks_checked = which registries were consulted (a
                # partial check must not read as checked-everywhere)
                "networks": meta.get("networks"),
                "networks_checked": meta.get("networks_checked"),
                # earthquake-step epochs (steps.txt type 2, decimal years);
                # [] = checked-none, null = not checked — the step-guard
                # evidence for propagate_epoch (owner 2026-08-30)
                "eq_steps": meta.get("eq_steps"),
                "eq_steps_through": meta.get("eq_steps_through"),
                "equip_steps": meta.get("equip_steps"),
                "n_solutions_used": pos["n_solutions_used"],
                "window": window_desc,
                "sig_e_m": pos["sig_e_m"],
                "sig_n_m": pos["sig_n_m"],
                "sig_u_m": pos["sig_u_m"],
                # antenna height (m) above the ground/monument the DSM/DTM
                # sees — assessment must remove it before differencing
                "ant_m": pos["ant_m"],
                # MIDAS velocity formal sigmas (m/yr), where a solution exists
                "sig_vel_e": mv.get("sig_vel_e"),
                "sig_vel_n": mv.get("sig_vel_n"),
                "sig_vel_u": mv.get("sig_vel_u"),
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
    # native dynamic-frame coordinates; the dispatcher lands them
    # (crs.land_horizontal passes per-row coord_epoch as tt — TODO(D6)).
    # The frame-level CRS is tagged when every row shares one code, so
    # driving parse() directly (the non-CONUS workflow, rasuwa 2026-08-29)
    # yields a usable GeoDataFrame without a manual set_crs; mixed codes
    # keep crs=None (per-row horizontal_crs is the authority either way).
    codes = df["horizontal_crs"].dropna().unique()
    crs_tag = None
    if len(codes) == 1:
        # per-row codes are 3D dynamic frames (EPSG:7912/9989) but the
        # geometry is 2D and the epoch lives per-row in coord_epoch — a 3D
        # stamp is exactly the shape validate_landing_crs refuses, and it
        # turned a downstream gdf.to_crs(6318) from a loud TypeError into
        # a silent ~1.5 m shift with no coordinate epoch applied. Stamp
        # the 2D counterpart the landing gate demotes to instead.
        from groundcontrol.sources import validate_landing_crs
        try:
            crs_tag = validate_landing_crs(codes[0])
        except ValueError as exc:  # future frame alias failing the gates:
            logger.warning("parse: frame-level CRS %s not stampable (%s); "
                           "leaving crs=None (per-row horizontal_crs is "
                           "the authority)", codes[0], exc)
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["native_x"], df["native_y"]),
        crs=crs_tag,
    )
