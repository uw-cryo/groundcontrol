"""Curated reference-network membership evidence.

Registry of curated GNSS reference networks with authoritative
machine-readable station lists. Membership is EARNED per station by ID
join PLUS coordinate corroboration — 4-char station IDs collide across
networks (the reason IGS moved to 9-character names around 2013), so an
ID-only join is never trusted. The result is carried as evidence in each
row's ``raw`` (``"networks": [...]``): ``[]`` means checked and member of
none; ``null``/absent means the fetch did not check.

Membership is a CURATION claim — someone operates the station to a
published network standard — one evidence axis alongside the occupation
class (pattern; :func:`groundcontrol.sources.ngl.occupation_class`) and
monument stability (:func:`groundcontrol.sources.ngs.opus_stability_tier`).
It is NOT a quality guarantee (owner, 2026-08-22): CORS includes rooftop
antennas; judge quality from the full evidence set.

Registry keys are NAMESPACED — "ngs_cors", never bare "cors": "CORS" is a
generic international term (GSI's own literature describes Japan's ~1,300
station GEONET as GNSS CORS, and e.g. InaCORS/CORSnet-NSW embed it in
their names), so an unqualified "cors" flag would be ambiguous the moment
a non-US network joins the registry.

Design rule (owner, 2026-08-22): network catalogs are consulted for
MEMBERSHIP EVIDENCE ONLY — positions, velocities, and time series always
come from the control source (NGL's uniform processing), never from a
network's own coordinate products or real-time (RTK/RTS) streams; those
are positioning services, not evidence about the station.

Candidates to add once a true MEMBER list is verified: EarthScope/GAGE
NOTA (probed 2026-08-22: the public GAGE ``sites/v1`` service serves the
whole global archive — ~4,300 stations including community networks, with
per-session monument descriptions — but NOT NOTA membership, and it has
no network filter; the authenticated EarthScope API is the likely
authority), GSI GEONET, EUREF EPN.
"""

from __future__ import annotations

import functools
import json
import logging
import time

import pandas as pd
import requests

from groundcontrol.sources.checkpoints_3dep import cache_dir

logger = logging.getLogger(__name__)

#: NGS bulk CORS composite: 4-char ID, ITRF2014 ARP position (DMS), status —
#: membership + corroboration coordinates in one fetch (verified 2026-08-22;
#: dates_sites.txt carries dates but no coordinates).
NGS_CORS_URL = ("https://geodesy.noaa.gov/corsdata/coord/coord_14/"
                "itrf2014_geo.comp.txt")
#: IGS station API (verified 2026-08-22: JSON, 814 stations, paginated via
#: a "next" cursor; 9-char names whose first 4 chars are the station ID).
IGS_URL = "https://network.igs.org/api/public/stations/?format=json&length=1000"
LIST_MAX_AGE_DAYS = 30.0
#: ID matches must be corroborated within this distance. Network catalog
#: positions and the NGL index are all ITRF-family (agreement well under a
#: few m); 100 m rejects same-ID different-station collisions without ever
#: rejecting a true match.
MATCH_TOL_M = 100.0


def parse_ngs_cors(text: str) -> pd.DataFrame:
    """Parse the NGS ITRF2014 composite -> DataFrame[id, lat, lon, status].

    Fixed 17-token data rows (SITE EPOCH lat-DMS+hemi lon-DMS+hemi eht
    Vn Ve Vu country state status). Preamble lines (date, title, headers,
    underscore rules) are skipped only until the first row that LOOKS like
    a data row (4-char site + float epoch); from there every data-shaped or
    subsequent line must conform or the parse fails loud (the DataHoldings
    lesson: silent tolerance hides layout drift — an earlier lineno-based
    carve-out silently swallowed a malformed FIRST data row, audit finding).
    """

    def _data_shaped(t):
        if len(t) < 2 or len(t[0]) != 4:
            return False
        try:
            float(t[1])
        except ValueError:
            return False
        return True

    rows, seen_data = [], False
    for lineno, line in enumerate(text.splitlines(), start=1):
        t = line.split()
        if not t:
            continue
        if len(t) != 17 or len(t[0]) != 4:
            if not seen_data and not _data_shaped(t):
                continue  # preamble (title/header/rule lines)
            raise ValueError(
                f"NGS CORS composite line {lineno}: expected 17-token data "
                f"row, got {len(t)}: {line!r}")
        seen_data = True
        # hemisphere tokens are validated, not guessed: a 17-token layout
        # drift that moves these letters would otherwise silently sign-flip
        # every coordinate (audit round 2)
        if t[5] not in ("N", "S") or t[9] not in ("E", "W"):
            raise ValueError(
                f"NGS CORS composite line {lineno}: hemisphere tokens "
                f"{t[5]!r}/{t[9]!r} not N/S / E/W: {line!r}")
        lat = (float(t[2]) + float(t[3]) / 60 + float(t[4]) / 3600)
        lon = (float(t[6]) + float(t[7]) / 60 + float(t[8]) / 3600)
        rows.append({
            "id": t[0].upper(),
            "lat": lat if t[5] == "N" else -lat,
            "lon": lon if t[9] == "E" else -lon,
            "status": t[16],
        })
    if not rows:
        raise ValueError("NGS CORS composite: no data rows parsed")
    return pd.DataFrame(rows)


def parse_igs(records: list[dict]) -> pd.DataFrame:
    """IGS station records -> DataFrame[id, name, lat, lon].

    ``id`` = first 4 chars of the 9-char name (the NGL/CORS convention);
    ``llh`` = [lat, lon, height]. Records without both fail loud.
    """
    rows = []
    for rec in records:
        name, llh = rec.get("name"), rec.get("llh")
        if not name or len(name) < 4 or not llh or len(llh) < 2:
            raise ValueError(f"IGS station record missing name/llh: {rec!r}")
        rows.append({"id": name[:4].upper(), "name": name,
                     "lat": float(llh[0]), "lon": float(llh[1])})
    if not rows:
        raise ValueError("IGS station list: no records parsed")
    return pd.DataFrame(rows)


def _stale(local) -> bool:
    return (not local.exists()
            or (time.time() - local.stat().st_mtime) > LIST_MAX_AGE_DAYS * 86400)


def cors_members(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows the composite carries that are NOT CORS members.

    Verified live 2026-08-22: the file includes 23 stations with status
    ``IGS_not_CORS`` (IGS stations listed for coordinate completeness) —
    claiming ngs_cors for those would be a false membership.
    """
    return df[df["status"] != "IGS_not_CORS"].reset_index(drop=True)


@functools.lru_cache(maxsize=1)
def load_ngs_cors() -> pd.DataFrame:
    """Cached NGS CORS member list (~/.cache/groundcontrol, DataHoldings
    pattern; additionally memoized per process — multi-site drivers call
    fetch repeatedly and the roster changes at most daily. Treat the
    returned frame as read-only.

    Known limitation (verified live 2026-08-22): the composite carries
    ~2,400 stations vs ~2,950 in dates_sites.txt — the remainder (mostly
    long-decommissioned) have no computed ITRF2014 position, so their
    membership cannot be coordinate-corroborated and is NOT claimed.
    """
    local = cache_dir() / "ngs_cors_itrf2014_geo_comp.txt"
    if _stale(local):
        logger.info("downloading %s -> %s", NGS_CORS_URL, local)
        # short timeout: supplementary evidence on a possibly egress-blocked
        # host must not stall a fetch for minutes (audit round 2); failures
        # degrade via ngl._attach_networks
        r = requests.get(NGS_CORS_URL, timeout=30)
        r.raise_for_status()
        # validate BEFORE caching: a 200-OK error/maintenance page written
        # first would poison the cache for LIST_MAX_AGE_DAYS (audit finding)
        df = cors_members(parse_ngs_cors(r.text))
        # atomic write: a crash mid-write must not leave a fresh-mtime
        # partial file poisoning the cache either (audit round 3)
        tmp = local.with_suffix(local.suffix + ".part")
        tmp.write_text(r.text)
        tmp.replace(local)
        return df
    return cors_members(parse_ngs_cors(local.read_text()))


@functools.lru_cache(maxsize=1)
def load_igs() -> pd.DataFrame:
    """Cached IGS station list, following the API's "next" cursor to the
    end. Memoized per process like load_ngs_cors; treat as read-only."""
    local = cache_dir() / "igs_stations.json"
    if _stale(local):
        records, url, pages = [], IGS_URL, 0
        while url:
            pages += 1
            if pages > 50:  # cursor-loop backstop (audit round 2): a
                raise ValueError(  # self-referencing "next" must not spin
                    "IGS pagination exceeded 50 pages — cursor loop?")
            logger.info("downloading %s", url)
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            page = r.json()
            records.extend(page["data"])
            url = page.get("next")
        # recordsFiltered = the current public roster this endpoint serves
        # (recordsTotal counts historical stations it never returns —
        # verified live 2026-08-22: 533 filtered of 814 total). REQUIRED:
        # without it a truncated list would cache as complete and stations
        # would record a false "checked, member of none" (audit round 2).
        expect = page.get("recordsFiltered")
        if expect is None:
            raise ValueError("IGS response lost the recordsFiltered count — "
                             "cannot verify pagination completeness")
        if len(records) != expect:
            raise ValueError(
                f"IGS pagination incomplete: {len(records)} of {expect}")
        # validate BEFORE caching + atomic write (see load_ngs_cors)
        df = parse_igs(records)
        tmp = local.with_suffix(local.suffix + ".part")
        tmp.write_text(json.dumps(records))
        tmp.replace(local)
        return df
    return parse_igs(json.loads(local.read_text()))


#: key -> loader. Namespaced keys only (see module docstring). Extending =
#: one entry here + a loader returning DataFrame[id, lat, lon, ...].
NETWORKS = {
    "ngs_cors": load_ngs_cors,
    "igs": load_igs,
}


def _dist_m(lat1, lon1, lat2, lon2):
    """Haversine distance (m) — one shared implementation: reuses
    velocity._haversine_km (which carries the arcsin-domain clip this
    module's first copy lacked, audit finding). NOTE the helper takes
    lon-first."""
    from groundcontrol.velocity import _haversine_km
    return _haversine_km(lon1, lat1, lon2, lat2) * 1000.0


def membership(sta_id: str, lat: float, lon: float,
               tables: dict[str, pd.DataFrame]) -> list[str]:
    """Corroborated network memberships for one station.

    A network is claimed only when ``sta_id`` appears in its list AND at
    least one same-ID entry lies within :data:`MATCH_TOL_M` of
    (``lat``, ``lon``) — an ID hit at the wrong coordinates is a 4-char
    collision, not a membership.
    """
    out = []
    for key, table in tables.items():
        hits = table[table["id"] == sta_id.upper()]
        if len(hits) and bool((_dist_m(lat, lon, hits["lat"].to_numpy(),
                                       hits["lon"].to_numpy())
                               <= MATCH_TOL_M).any()):
            out.append(key)
    return out


def load_networks(keys=None) -> dict[str, pd.DataFrame]:
    """Load the registry tables (all by default) for :func:`membership`."""
    keys = list(NETWORKS) if keys is None else list(keys)
    unknown = [k for k in keys if k not in NETWORKS]
    if unknown:
        raise ValueError(f"unknown network key(s) {unknown}; "
                         f"registry: {sorted(NETWORKS)}")
    return {k: NETWORKS[k]() for k in keys}


def network_member(gdf, key: str) -> pd.Series:
    """Boolean mask: is each row a corroborated member of ``key``?

    Reads the ``"networks"`` evidence list from ``raw``. Rows where the
    fetch did not check (``networks`` null/absent, no parseable ``raw`` —
    e.g. other sources or pre-networks products) return pd.NA, never
    False: "not checked" and "checked, not a member" stay distinct.
    """
    if key not in NETWORKS:
        raise ValueError(f"unknown network key {key!r}; "
                         f"registry: {sorted(NETWORKS)}")
    if "raw" not in gdf.columns:  # column-subset frames carry no evidence:
        return pd.Series(pd.NA, index=gdf.index,  # all-NA, matching the
                         dtype="boolean")         # expand_attributes guard

    def _one(raw):
        try:
            rec = json.loads(raw)
        except (TypeError, ValueError):
            return pd.NA
        if not isinstance(rec, dict):  # valid-but-non-object JSON (null,
            return pd.NA               # list, number): no evidence, never
        nets = rec.get("networks")     # an AttributeError (audit finding)
        if nets is None:
            return pd.NA
        if not isinstance(nets, list):  # corrupt evidence (e.g. a string —
            return pd.NA                # `in` would substring-match a false
                                        # membership; audit round 2)
        checked = rec.get("networks_checked")
        if isinstance(checked, list) and key not in checked:
            return pd.NA  # partial check: this registry wasn't consulted
                          # for the row (audit round 4); rows without the
                          # key (pre-partial-degrade products) keep the old
                          # fully-checked semantics
        return key in nets

    return gdf["raw"].map(_one).astype("boolean")
