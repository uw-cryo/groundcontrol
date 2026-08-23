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

import json
import logging
import time

import numpy as np
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
    Vn Ve Vu country state status); header/underscore lines are skipped by
    shape, and any non-conforming non-header line fails loud (the
    DataHoldings lesson: silent tolerance hides layout drift).
    """
    rows = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        t = line.split()
        if len(t) != 17 or len(t[0]) != 4:
            if t and t[0] == "SITE":  # column-header row
                continue
            if not t or lineno <= 8 or set(line.strip()) <= {"_"}:
                continue  # preamble/blank/rule lines
            raise ValueError(
                f"NGS CORS composite line {lineno}: expected 17-token data "
                f"row, got {len(t)}: {line!r}")
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


def load_ngs_cors() -> pd.DataFrame:
    """Cached NGS CORS member list (~/.cache/groundcontrol, DataHoldings
    pattern).

    Known limitation (verified live 2026-08-22): the composite carries
    ~2,400 stations vs ~2,950 in dates_sites.txt — the remainder (mostly
    long-decommissioned) have no computed ITRF2014 position, so their
    membership cannot be coordinate-corroborated and is NOT claimed.
    """
    local = cache_dir() / "ngs_cors_itrf2014_geo_comp.txt"
    if _stale(local):
        logger.info("downloading %s -> %s", NGS_CORS_URL, local)
        r = requests.get(NGS_CORS_URL, timeout=120)
        r.raise_for_status()
        local.write_text(r.text)
    return cors_members(parse_ngs_cors(local.read_text()))


def load_igs() -> pd.DataFrame:
    """Cached IGS station list, following the API's "next" cursor to the end."""
    local = cache_dir() / "igs_stations.json"
    if _stale(local):
        records, url = [], IGS_URL
        while url:
            logger.info("downloading %s", url)
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            page = r.json()
            records.extend(page["data"])
            url = page.get("next")
        # recordsFiltered = the current public roster this endpoint serves
        # (recordsTotal counts historical stations it never returns —
        # verified live 2026-08-22: 533 filtered of 814 total)
        expect = page.get("recordsFiltered")
        if expect is not None and len(records) != expect:
            raise ValueError(
                f"IGS pagination incomplete: {len(records)} of {expect}")
        local.write_text(json.dumps(records))
    return parse_igs(json.loads(local.read_text()))


#: key -> loader. Namespaced keys only (see module docstring). Extending =
#: one entry here + a loader returning DataFrame[id, lat, lon, ...].
NETWORKS = {
    "ngs_cors": load_ngs_cors,
    "igs": load_igs,
}


def _dist_m(lat1, lon1, lat2, lon2):
    """Haversine distance (m); fine at corroboration scales."""
    p = np.pi / 180.0
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p)
         * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * 6371000.0 * np.arcsin(np.sqrt(a))


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

    def _one(raw):
        try:
            nets = json.loads(raw).get("networks")
        except (TypeError, ValueError):
            return pd.NA
        return pd.NA if nets is None else key in nets

    return gdf["raw"].map(_one).astype("boolean")
