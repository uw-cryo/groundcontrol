"""FAA NASR runway ends and displaced thresholds (photo-identifiable control).

Source: the FAA's 28-day NASR subscription, legacy fixed-width ``APT.zip``
(record types APT/ATT/RWY/ARS/RMK, logical record 1531 bytes). Public
domain, no authentication; cycle effective dates run on a 28-day cadence
anchored at 2022-12-01 (verified against live cycles 2022-04-21 through
2026-08-06). Byte offsets below were verified against the published record
layout (``Layout_Data/apt_rf.txt``) and a full-cycle parse recount
(23,196 RWY records / 23,766 surveyed end coordinates, cycle 2026-08-06).

Runway ends and displaced thresholds are painted, photo-identifiable
pavement features surveyed under AC 150/5300-18C: +/-1.00 ft horizontal,
+/-0.25 ft orthometric vertical at 95% for the surveyed class. NASR also
publishes per-point coordinate provenance (``RUNWAY END POSITION SOURCE``):
empirically (Las Vegas 1 m lidar A/B, 2026-08-13) the surveyed class
(3RD PARTY SURVEY / NGS / MILITARY / ARPTS CONTRACTOR) closes at
~2 cm NMAD vertical, while OWNER / FAA-EST IMAGERY / ADO points (most
heliports) are meters to tens of meters off, some quantized to whole
arcseconds. ALL points are returned with provenance in ``raw``; filter at
assessment time with :func:`pos_class` (the NGS ``vertSource`` pattern).

Datum: NASR states bare "NAD 83" with no realization tag or epoch.
AC 150/5300-16B requires "the most current adjustment", and rows are
tagged ``EPSG:6318`` (NAD83(2011)) here: there is no NAD83 *ensemble*
EPSG code — ``EPSG:4269`` formally means NAD83(1986), and tagging that
would make downstream landing apply NADCON5 1986->2011 grid shifts that
CORRUPT modern surveyed coordinates (the Las Vegas A/B closes at 2 cm
under the 2011 reading). Pre-modernization survey dates ride along in
``raw['pos_src_date']`` for callers who need to judge realization vintage.
Elevations are feet MSL, NAVD88 per the AC (the most recent NGS hybrid
geoid at survey time); NASR never publishes ellipsoid height (ARINC 424
field 5.225 via the CIFP distribution is the only public channel — a
possible future join, not implemented here).

Facilities under ownership codes MA/MN/MR/CG publish elevations from a
separate survey pipeline referenced to EGM96 MSL rather than NAVD88
(verified against 3DEP across the cycle's facilities, 2026-09-01). A row
at such a facility whose elevation source is in :data:`EGM96_ELEV_SRC`
(the rule is ownership AND source; the same source strings at other
facilities read NAVD88 and are the next cohort to verify) declares EGM96
(``EPSG:5773``); its natives carry the published horizontal with the
NAD83(2011) ellipsoidal height derived through the explicit geoid + frame
tie chain (:func:`_egm96_to_nad83_2011_h`), so the re-target is a pure
NAD83(2011) -> product operation. PROJ's own NAD83(2011)+EGM96 compound
would land 0.7-0.9 m low across CONUS (it skips the NAD83(2011)-ITRF2014
Helmert through a null WGS 84 step). 3RD PARTY SURVEY records read NAVD88;
other sources stay NA with NA natives, so they land nowhere (h_ell NaN)
rather than through a guessed datum. :func:`pos_class` classes the whole
facility ``"mil"``: no published accuracy, own context segment, outside
the surveyed tier.
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from datetime import date, timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from groundcontrol.crs import decyear
from groundcontrol.sources.checkpoints_3dep import cache_dir, cache_stale, cache_write

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#: cycle cadence anchor (an actual published effective date; 28-day steps
#: reproduce every observed cycle URL 2022-2026)
CYCLE_ANCHOR = date(2022, 12, 1)
APT_URL = "https://nfdc.faa.gov/webContent/28DaySub/{cycle}/APT.zip"

FT = 0.3048  # feet -> meters (layout elevations are feet MSL)

#: AC 150/5300-18C survey accuracy for runway ends / displaced thresholds
#: (95%), applied to the surveyed provenance class only; the estimated
#: class has no reported accuracy (acc_* left null, honestly unknown).
ACC_H_SURVEYED = 0.30   # +/-1.00 ft horizontal
ACC_V_SURVEYED = 0.076  # +/-0.25 ft orthometric

#: position-source values treated as survey-grade (LV-verified 2026-08-13:
#: this class sits on the threshold paint at ~2 cm vertical NMAD; everything
#: else — OWNER, FAA-EST IMAGERY, ADO, FAA OE/AAA, blank, ... — is meters
#: to tens of meters class)
SURVEYED_SOURCES = frozenset(
    {"3RD PARTY SURVEY", "NGS", "ARPTS CONTRACTOR", "MILITARY"})


#: APT ownership codes whose facilities publish EGM96 MSL elevations
MIL_OWNERSHIP = {"MA", "MN", "MR", "CG"}
#: NASR elevation sources published on EGM96 MSL at those facilities
#: (verified 2026-09-01; see the parse() note).
EGM96_ELEV_SRC = {"MILITARY", "DOD (NGA)", "AVN", "NGS"}
MIL_EGM96_DATUM = "EGM96 MSL (declared per elevation source; verified 2026-09-01)"
#: native frame of an EGM96-declared row: NAD83(2011) 3D — the published
#: horizontal with the ellipsoidal height derived at parse time
MIL_NATIVE_CRS = "EPSG:6319"


def _egm96_to_nad83_2011_h(lon, lat, height, epoch, aoi_bounds_4326=None):
    """Published EGM96 orthometric height -> NAD83(2011) ellipsoidal height
    at the published lon/lat, through the explicit chain EGM96 -> WGS 84
    (geoid grid) -> ITRF2014 (WGS 84 taken as ITRF2014, cm-level) ->
    NAD83(2011) (the time-dependent tie at the row's coord_epoch). The
    horizontal never moves: the natives then re-target through a pure
    NAD83(2011) -> product operation, exact for every target shape
    (round-6 audits: a compound native chained differently per target —
    +0.74 m on an EGM96-vertical target, up to 1.6 m of horizontal
    displacement elsewhere). Returns ``(h_ell, stated_acc_m)`` with NaN
    where the geoid grid or the frame tie is unavailable (Pacific
    territories, ...): the caller leaves those rows unverified. Never
    raises — one such point must not take down a cycle's parse."""
    from pyproj import Transformer

    from groundcontrol.crs import NoTransformPathError, get_transformer
    lon = np.asarray(lon, dtype="float64")
    lat = np.asarray(lat, dtype="float64")
    height = np.asarray(height, dtype="float64")
    epoch = np.asarray(epoch, dtype="float64")
    out = np.full(lon.shape, np.nan)
    acc = float("nan")
    fin = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(height) & np.isfinite(epoch)
    if not fin.any():
        return out, acc
    b = aoi_bounds_4326 or (float(np.nanmin(lon)) - 1.0, float(np.nanmin(lat)) - 1.0,
                            float(np.nanmax(lon)) + 1.0, float(np.nanmax(lat)) + 1.0)
    try:
        geoid = Transformer.from_crs("EPSG:4326+5773", "EPSG:4979",
                                     always_xy=True, allow_ballpark=False)
        _, _, hw = geoid.transform(lon[fin], lat[fin], height[fin], errcheck=False)
        hw = np.asarray(hw, dtype="float64")
        tie = get_transformer("EPSG:7912", "EPSG:6319", aoi_bounds_4326=b)
        _, _, hn, _ = tie.transform(lon[fin], lat[fin], hw, epoch[fin], errcheck=False)
        hn = np.asarray(hn, dtype="float64")
    except (NoTransformPathError, Exception) as exc:  # noqa: BLE001 — fail soft, per row
        logger.warning("faa: EGM96 -> NAD83(2011) chain unavailable for this AOI "
                       "(%s); %d EGM96-declared row(s) left unverified", exc, int(fin.sum()))
        return out, acc
    ok = np.isfinite(hw) & np.isfinite(hn)
    res = np.full(int(fin.sum()), np.nan)
    res[ok] = hn[ok]
    out[fin] = res
    parts = [a for a in (geoid.accuracy, tie.accuracy) if a is not None and a > 0]
    acc = float(sum(parts)) if parts else float("nan")
    return out, acc


def pos_class(src, ownership=None) -> str:
    """Provenance class for a NASR position source string.

    A row at a facility under :data:`MIL_OWNERSHIP` classes ``"mil"``
    regardless of its position source: those facilities publish
    elevations on EGM96 MSL, not NAVD88 (module docstring). The class
    keeps them out of the surveyed accuracy tier (no published accuracy)
    and visible as their own segment; the datum itself is declared per
    elevation source in :func:`parse`."""
    if str(ownership).strip().upper() in MIL_OWNERSHIP:
        return "mil"
    return "surveyed" if str(src).strip().upper() in SURVEYED_SOURCES \
        else "estimated"


def current_cycle(today: date | None = None) -> str:
    """Effective date (ISO) of the NASR cycle containing ``today``."""
    today = today or date.today()
    n = (today - CYCLE_ANCHOR).days // 28
    return (CYCLE_ANCHOR + timedelta(days=28 * n)).isoformat()


def fetch(aoi_bounds_4326, cycle: str | None = None) -> dict:
    """Download-once cache of the cycle's APT.zip; returns the raw payload.

    The payload keeps only APT/RWY record lines (the other record types
    carry no coordinates) plus the cycle tag and the AOI bounds for the
    offline :func:`parse` half. If the computed current cycle 404s (cycle
    boundary/publication lag), the previous cycle is tried once before
    failing loud.
    """
    cycles = [cycle] if cycle else None
    if cycles is None:
        cur = current_cycle()
        prev = (date.fromisoformat(cur) - timedelta(days=28)).isoformat()
        cycles = [cur, prev]
    local = None
    for cyc in cycles:
        local = cache_dir() / f"faa_APT_{cyc}.zip"
        if not cache_stale(local):
            break
        url = APT_URL.format(cycle=cyc)
        logger.info("downloading %s -> %s", url, local)
        r = requests.get(url, timeout=300)
        if r.status_code == 404 and cyc != cycles[-1]:
            logger.warning("cycle %s not published yet (404); trying %s",
                           cyc, cycles[-1])
            continue
        r.raise_for_status()
        cache_write(local, r.content)
        break
    else:  # pragma: no cover - loop always breaks or raises
        raise RuntimeError(f"no NASR cycle reachable: {cycles}")
    lines = []
    with zipfile.ZipFile(local) as z:
        members = [n for n in z.namelist() if n.upper().endswith(".TXT")]
        if len(members) != 1:
            raise ValueError(f"expected one .txt member in {local}: {members}")
        f = io.TextIOWrapper(z.open(members[0]), encoding="latin-1")
        for rec in f:
            if rec[0:3] in ("APT", "RWY"):
                lines.append(rec)
    cyc_used = local.stem.rsplit("_", 1)[1]
    return {"cycle": cyc_used, "aoi_bounds_4326": tuple(aoi_bounds_4326),
            "lines": lines}


# --- fixed-width slices (0-based; layout doc positions are 1-based) --------
_APT_SITE = slice(3, 14)       # 00004 L11 landing facility site number
_APT_TYPE = slice(14, 27)      # 00015 L13 facility type (AIRPORT/HELIPORT/..)
_APT_LOCID = slice(27, 31)     # 00028 L4  location identifier
_APT_NAME = slice(133, 183)    # 00134 L50 official facility name
_APT_OWNER = slice(183, 185)   # 00184 L2  ownership: PU/PR public/private,
                               #           MA/MN/MR/CG see MIL_OWNERSHIP
_APT_USE = slice(185, 187)     # 00186 L2  facility use: PU/PR
_RWY_SITE = slice(3, 14)       # 00004 L11 site number (joins APT record)
_RWY_ID = slice(16, 23)        # 00017 L7  runway identification '01L/19R'
_END_OFF = 222                 # reciprocal offset, GEOGRAPHIC blocks
_SRC_OFF = 291                 # reciprocal offset, ADDITIONAL-DATA blocks
_END_ID = slice(65, 68)        # 00066 L3  end identifier
_END_AZ = slice(68, 71)        # 00069 L3  runway end true alignment (deg)
_END_LAT = slice(103, 115)     # 00104 L12 physical end latitude (seconds)
_END_LON = slice(130, 142)     # 00131 L12 physical end longitude (seconds)
_END_ELEV = slice(142, 149)    # 00143 L7  end elevation, feet MSL
_DT_LAT = slice(171, 183)      # 00172 L12 displaced threshold lat (seconds)
_DT_LON = slice(198, 210)      # 00199 L12 displaced threshold lon (seconds)
_DT_ELEV = slice(210, 217)     # 00211 L7  displaced threshold elev, ft MSL
_DT_LEN = slice(217, 221)      # 00218 L4  displaced threshold length, ft
_END_POS_SRC = slice(568, 584)   # 00569 L16 end position source
_END_POS_DATE = slice(584, 594)  # 00585 L10 position source date MM/DD/YYYY
_END_ELEV_SRC = slice(594, 610)  # 00595 L16 end elevation source
_DT_POS_SRC = slice(620, 636)    # 00621 L16 displaced threshold pos source
_DT_POS_DATE = slice(636, 646)   # 00637 L10 displaced threshold pos date
_DT_ELEV_SRC = slice(646, 662)   # 00647 L16 displaced threshold elev source


def _shift(sl: slice, off: int) -> slice:
    return slice(sl.start + off, sl.stop + off)


def _sec(s: str) -> float:
    """'130806.0700N' (total arcseconds + hemisphere) -> signed degrees."""
    s = s.strip()
    if not s:
        return np.nan
    val = float(s[:-1]) / 3600.0
    return -val if s[-1] in "SW" else val


def _feet_m(s: str) -> float:
    s = s.strip()
    return float(s) * FT if s else np.nan


def _rows(lines) -> list[dict]:
    apts: dict[str, dict] = {}
    rows: list[dict] = []
    for rec in lines:
        rt = rec[0:3]
        if rt == "APT":
            apts[rec[_APT_SITE].strip()] = {
                "fac_type": rec[_APT_TYPE].strip(),
                "loc_id": rec[_APT_LOCID].strip(),
                "name": rec[_APT_NAME].strip(),
                "ownership": rec[_APT_OWNER].strip(),
                "fac_use": rec[_APT_USE].strip(),
            }
        elif rt == "RWY":
            site = rec[_RWY_SITE].strip()
            ap = apts.get(site)
            if ap is None:  # file is sorted APT-first per facility
                raise ValueError(f"RWY record precedes APT for site {site}")
            rwy = rec[_RWY_ID].strip()
            for off, soff in ((0, 0), (_END_OFF, _SRC_OFF)):
                end_id = rec[_shift(_END_ID, off)].strip()
                if not end_id:
                    continue
                lat = _sec(rec[_shift(_END_LAT, off)])
                lon = _sec(rec[_shift(_END_LON, off)])
                if not (np.isfinite(lat) and np.isfinite(lon)):
                    continue
                base = {**ap, "site_no": site, "rwy": rwy, "end": end_id}
                # helipad "runways" (H1/H2...) are pad points, not runway
                # ends — their own marker class and photo-ID semantics
                ptype = "helipad" if rwy.upper().startswith("H") else "runway_end"
                # E46 true alignment (deg): the heading of the runway
                # direction this end names — i.e. pointing INWARD along the
                # runway from this end. Feeds oriented map/gallery chevrons.
                az = rec[_shift(_END_AZ, off)].strip()
                true_az = float(az) if az.isdigit() else np.nan
                rows.append({
                    **base, "point_type": ptype,
                    "id": f"{ap['loc_id'] or site}_{end_id}",
                    "lat": lat, "lon": lon,
                    "height": _feet_m(rec[_shift(_END_ELEV, off)]),
                    "true_az": true_az,
                    "pos_src": rec[_shift(_END_POS_SRC, soff)].strip(),
                    "pos_src_date": rec[_shift(_END_POS_DATE, soff)].strip(),
                    "elev_src": rec[_shift(_END_ELEV_SRC, soff)].strip(),
                })
                dlat = _sec(rec[_shift(_DT_LAT, off)])
                dlon = _sec(rec[_shift(_DT_LON, off)])
                if np.isfinite(dlat) and np.isfinite(dlon):
                    rows.append({
                        **base, "point_type": "displaced_threshold",
                        "id": f"{ap['loc_id'] or site}_{end_id}_DT",
                        "lat": dlat, "lon": dlon,
                        "height": _feet_m(rec[_shift(_DT_ELEV, off)]),
                        "true_az": true_az,
                        "pos_src": rec[_shift(_DT_POS_SRC, soff)].strip(),
                        "pos_src_date": rec[_shift(_DT_POS_DATE, soff)].strip(),
                        # same vocabulary as the end's elevation source;
                        # without it every threshold of that class fell to NA
                        # (nationwide count, 2026-09-01)
                        "elev_src": rec[_shift(_DT_ELEV_SRC, soff)].strip(),
                        "dt_len_ft": rec[_shift(_DT_LEN, off)].strip(),
                    })
    return rows


#: row keys consumed into first-class schema fields; the rest go to ``raw``
_CONSUMED = {"id", "point_type", "lat", "lon", "height"}


def parse(raw: dict) -> gpd.GeoDataFrame:
    """APT/RWY record lines -> schema-shaped GeoDataFrame (native frame).

    Emits EVERY runway end / displaced threshold with published coordinates
    inside the AOI bounds — including estimated-provenance and seaplane/
    heliport facilities. Quality filtering is an assessment-time decision:
    use :func:`pos_class` on ``raw['pos_src']`` (and ``raw['fac_type']``)
    exactly like the NGS ``vertSource`` classes.
    """
    df = pd.DataFrame(_rows(raw["lines"]))
    if len(df):
        minlon, minlat, maxlon, maxlat = raw["aoi_bounds_4326"]
        keep = ((df["lon"] >= minlon) & (df["lon"] <= maxlon)
                & (df["lat"] >= minlat) & (df["lat"] <= maxlat))
        df = df[keep].reset_index(drop=True)
    n = len(df)
    if not n:  # schema-shaped empty frame with a valid CRS
        df = pd.DataFrame(columns=["id", "point_type", "lat", "lon", "height",
                                   "pos_src", "pos_src_date"])
    own = df["ownership"] if "ownership" in df.columns \
        else pd.Series([None] * n, index=df.index)
    cls = [pos_class(ps, ow) for ps, ow in zip(df.get("pos_src", []), own)] \
        if n else []
    surveyed = np.array([c == "surveyed" for c in cls], dtype=bool)
    mil = np.array([c == "mil" for c in cls], dtype=bool)
    mdt = pd.to_datetime(df["pos_src_date"], format="%m/%d/%Y",
                         errors="coerce", utc=True) \
        if n else pd.Series([], dtype="datetime64[ns, UTC]")
    extras = [c for c in df.columns if c not in _CONSUMED]
    # rows at MIL_OWNERSHIP facilities whose elevation source is in
    # EGM96_ELEV_SRC (ownership AND source; verified against 3DEP across
    # the cycle's facilities, 2026-09-01) DECLARE EGM96 (EPSG:5773); their
    # natives carry the derived NAD83(2011) ellipsoidal height on the
    # published horizontal. 3RD PARTY SURVEY records read NAVD88. Any
    # other source (ADO, blank, ...) stays NA with NA natives: unverified,
    # never guessed, h_ell NaN downstream (round-6 audit: a NAVD88 native
    # had been landing them 0.5-1.9 m from their own stated reading).
    height_datum = pd.Series(["NAVD88"] * n, dtype="string")
    vertical_crs = pd.Series(["EPSG:5703"] * n, dtype="string")
    # COPIES (a to_numpy() view would alias the published columns)
    native_x = np.array(df["lon"], dtype="float64") if n else np.array([])
    native_y = np.array(df["lat"], dtype="float64") if n else np.array([])
    native_h = np.array(pd.to_numeric(df["height"], errors="coerce"),
                        dtype="float64") if n else np.array([])
    native_crs = pd.Series(["EPSG:6349"] * n, dtype="string")
    coord_epoch = np.full(n, 2010.0)
    if n and mil.any():
        esrc = (df["elev_src"].astype("string").str.strip().str.upper()
                if "elev_src" in df.columns
                else pd.Series([pd.NA] * n, dtype="string"))
        egm = mil & esrc.isin(EGM96_ELEV_SRC).fillna(False).to_numpy(dtype=bool)
        third = mil & esrc.eq("3RD PARTY SURVEY").fillna(False).to_numpy(
            dtype=bool)
        other = mil & ~egm & ~third
        height_datum[egm] = MIL_EGM96_DATUM
        vertical_crs[egm] = "EPSG:5773"
        height_datum[other] = "MSL (EGM96 assumed; unverified)"
        vertical_crs[other] = pd.NA
        native_crs[other] = pd.NA      # undeclared datum lands nowhere
        # (third-party surveys keep the NAVD88 defaults)
        if egm.any():
            h_ell, chain_acc = _egm96_to_nad83_2011_h(
                native_x[egm], native_y[egm], native_h[egm], coord_epoch[egm],
                raw.get("aoi_bounds_4326"))
            tied = np.isfinite(h_ell)
            idx = np.flatnonzero(egm)
            native_h[idx[tied]] = h_ell[tied]
            native_crs[idx[tied]] = MIL_NATIVE_CRS
            if tied.any():
                df.loc[df.index[idx[tied]], "native_chain"] = (
                    "EGM96 grid + ITRF2014->NAD83(2011) tie at coord_epoch"
                    f" (stated accuracy {chain_acc:.2f} m)")
                if "native_chain" not in extras:   # rides into raw
                    extras.append("native_chain")
            if (~tied).any():
                # no geoid grid / frame tie here: the datum is still EGM96
                # but the row cannot be landed honestly -> NA, unverified,
                # NA natives (the same fail-loud NA as the other sources)
                vertical_crs[idx[~tied]] = pd.NA
                native_crs[idx[~tied]] = pd.NA
                height_datum[idx[~tied]] = ("MSL (EGM96 assumed; frame tie "
                                            "unavailable here — unverified)")
                logger.warning("faa: %d EGM96-declared row(s) outside "
                               "the geoid / frame-tie coverage left with "
                               "vertical_crs NA", int((~tied).sum()))
    out = gpd.GeoDataFrame(
        {
            "id": df["id"].astype("string"),
            # TODO(D2): new marker-type values (runway_end,
            # displaced_threshold) pending the point_type split adjudication
            "point_type": df["point_type"].astype("string"),
            "height": pd.to_numeric(df["height"], errors="coerce"),
            "height_datum": height_datum,
            "horizontal_crs": pd.Series(["EPSG:6318"] * n, dtype="string"),
            "vertical_crs": vertical_crs,
            "ref_frame": pd.Series(["NAD83(2011)"] * n, dtype="string"),
            "frame_epoch": np.full(n, 2010.0),
            # plate-fixed published positions; reduced-to-frame-epoch reading
            # (same convention as checkpoints_3dep)
            "coord_epoch": coord_epoch,
            "measurement_datetime": mdt,
            "measurement_epoch": decyear(mdt) if n else
            pd.Series([], dtype="float64"),
            # spec accuracy for the surveyed class only; estimated rows have
            # NO reported accuracy — null, never a fabricated bound (TODO(D3))
            "acc_h": np.where(surveyed, ACC_H_SURVEYED, np.nan),
            "acc_v": np.where(surveyed, ACC_V_SURVEYED, np.nan),
            "native_x": native_x,
            "native_y": native_y,
            "native_h": native_h,
            "native_crs": native_crs,
            "raw": pd.Series(
                [json.dumps({**{k: str(df.iloc[i][k]) for k in extras
                                if pd.notna(df.iloc[i][k])},
                             "pos_class": cls[i]})
                 for i in range(n)], dtype="string", index=df.index),
        },
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:6318",
        index=df.index,
    )
    return out
