"""Curated reference-network membership evidence (networks.py)."""

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest

from groundcontrol import networks

DATA = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# list parsers (recorded real samples, 2026-08-22)
# ---------------------------------------------------------------------------

def test_parse_ngs_cors_sample():
    df = networks.parse_ngs_cors((DATA / "ngs_cors_coord_sample.txt").read_text())
    assert len(df) == 13  # 20-line head: 7 preamble + 13 data rows
    r = df.iloc[0]
    # 1LSU: 30 24 26.72881 N / 91 10 48.94290 W — exact DMS conversion
    assert r["id"] == "1LSU"
    assert r["lat"] == pytest.approx(30 + 24 / 60 + 26.72881 / 3600)
    assert r["lon"] == pytest.approx(-(91 + 10 / 60 + 48.94290 / 3600))
    # multi-word-free status tokens, including Non-Operational rows
    assert set(df["status"]) <= {"Operational", "Non-Operational",
                                 "Decommissioned", "Suspended"}
    assert (df["status"] == "Non-Operational").any()


def test_cors_members_drops_igs_not_cors():
    # the live composite carries 23 IGS_not_CORS rows (coordinate
    # completeness); they must never earn ngs_cors membership
    df = networks.parse_ngs_cors(
        (DATA / "ngs_cors_coord_sample.txt").read_text()
        + "\nZIMM  2010.00  46 52 37.00000 N    7 27 55.00000 E   956.000"
          "     0.0     0.0     0.0    SZ      --      IGS_not_CORS\n")
    kept = networks.cors_members(df)
    assert "ZIMM" in set(df["id"]) and "ZIMM" not in set(kept["id"])
    assert len(kept) == len(df) - 1


def test_parse_ngs_cors_fails_loud_on_layout_drift():
    good = (DATA / "ngs_cors_coord_sample.txt").read_text()
    with pytest.raises(ValueError, match="17-token"):
        networks.parse_ngs_cors(good + "\nABCD 2010.00 not a data row\n")
    with pytest.raises(ValueError, match="no data rows"):
        networks.parse_ngs_cors("June 08, 2025\n\nheader only\n")
    # a malformed FIRST data row must fail loud too — the old lineno<=8
    # carve-out silently swallowed it (audit finding). Data-shaped =
    # 4-char site + float epoch, even when truncated.
    preamble = "\n".join(good.splitlines()[:7])
    with pytest.raises(ValueError, match="17-token"):
        networks.parse_ngs_cors(preamble + "\nABCD  2010.00  30 24\n")
    # hemisphere tokens are validated, never guessed: a 17-token drift that
    # moves them would silently sign-flip coordinates (audit round 2)
    bad_hemi = good.splitlines()[7].replace(" N ", " X ", 1)
    with pytest.raises(ValueError, match="hemisphere"):
        networks.parse_ngs_cors(preamble + "\n" + bad_hemi + "\n")


def test_parse_igs_sample():
    recs = json.loads((DATA / "igs_stations_sample.json").read_text())
    df = networks.parse_igs(recs)
    assert len(df) == 5
    assert df.iloc[0]["name"] == "ABMF00GLP"
    assert df.iloc[0]["id"] == "ABMF"  # 4-char ID = first 4 of the 9-char name
    assert df.iloc[0]["lat"] == pytest.approx(16.2623, abs=1e-3)
    with pytest.raises(ValueError, match="name/llh"):
        networks.parse_igs([{"name": "X"}])


# ---------------------------------------------------------------------------
# corroborated membership
# ---------------------------------------------------------------------------

def _tables():
    return {
        "ngs_cors": pd.DataFrame({
            "id": ["AAAA", "BBBB"],
            "lat": [40.0, 41.0], "lon": [-110.0, -111.0],
            "status": ["Operational"] * 2}),
        "igs": pd.DataFrame({
            "id": ["AAAA"], "name": ["AAAA00USA"],
            "lat": [40.0], "lon": [-110.0]}),
    }


def test_membership_requires_id_and_coordinates():
    t = _tables()
    # ID + coordinate agreement (<100 m) -> member of both
    assert networks.membership("AAAA", 40.0, -110.0, t) == ["ngs_cors", "igs"]
    # 4-char ID collision: same ID ~5 km away is NOT a membership
    assert networks.membership("AAAA", 40.045, -110.0, t) == []
    # ID absent from every list -> no membership, coordinates irrelevant
    assert networks.membership("ZZZZ", 40.0, -110.0, t) == []
    # case-normalized join
    assert networks.membership("bbbb", 41.0, -111.0, t) == ["ngs_cors"]


def test_load_networks_unknown_key_raises():
    with pytest.raises(ValueError, match="unknown network"):
        networks.load_networks(["ngs_cors", "geonet"])


# ---------------------------------------------------------------------------
# network_member mask (raw evidence -> boolean)
# ---------------------------------------------------------------------------

def test_network_member_mask_three_states():
    g = gpd.GeoDataFrame(
        {"raw": pd.Series([
            json.dumps({"networks": ["ngs_cors", "igs"]}),  # member
            json.dumps({"networks": []}),        # checked, member of none
            json.dumps({"networks": None}),      # fetch did not check
            json.dumps({"other": 1}),            # key absent = not checked
            "not json",                          # other source / no evidence
            "null",                              # valid JSON, not a dict —
            "[1, 2]",                            # must be NA, never an
            "5",                                 # AttributeError (audit)
            json.dumps({"networks": "ngs_cors"}),  # non-list value: NA, never
        ], dtype="string")},                       # a substring-match True
        geometry=gpd.points_from_xy(range(9), [0.0] * 9), crs="EPSG:4326")
    m = networks.network_member(g, "ngs_cors")
    assert m.dtype == "boolean"
    assert m.iloc[0] == True  # noqa: E712  (pandas BooleanDtype comparison)
    assert m.iloc[1] == False  # noqa: E712  "checked, not a member" is False
    assert m.iloc[2:].isna().all()  # "not checked" stays NA, never False
    with pytest.raises(ValueError, match="unknown network"):
        networks.network_member(g, "cors")  # bare "cors" is not a key
    # a column-subset frame without raw carries no evidence: all-NA,
    # matching the expand_attributes guard (audit round 3)
    noraw = g.drop(columns=["raw"])
    assert networks.network_member(noraw, "ngs_cors").isna().all()


def test_network_member_honors_partial_check():
    # a partial-degrade fetch records WHICH registries it consulted; a
    # membership question about an unconsulted registry is NA, never a
    # confident False (audit round 4)
    g = gpd.GeoDataFrame(
        {"raw": pd.Series([
            json.dumps({"networks": [], "networks_checked": ["ngs_cors"]}),
            json.dumps({"networks": ["ngs_cors"],
                        "networks_checked": ["ngs_cors"]}),
            json.dumps({"networks": []}),  # pre-partial-degrade product:
        ], dtype="string")},              # fully-checked semantics kept
        geometry=gpd.points_from_xy(range(3), [0.0] * 3), crs="EPSG:4326")
    cors = networks.network_member(g, "ngs_cors")
    igs = networks.network_member(g, "igs")
    assert cors.iloc[0] == False and cors.iloc[1] == True  # noqa: E712
    assert igs.iloc[0] is pd.NA or pd.isna(igs.iloc[0])  # unconsulted -> NA
    assert igs.iloc[2] == False  # noqa: E712  legacy raw: checked-none


# ---------------------------------------------------------------------------
# live lists (network marker)
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_live_ngs_cors_list():
    df = networks.load_ngs_cors()
    # 2,395 member rows as of 2026-08 (2,418 in the composite minus 23
    # IGS_not_CORS); ~530 dates_sites stations lack computed positions and
    # are deliberately absent (uncorroboratable — see load_ngs_cors)
    assert len(df) > 2300
    assert (df["status"] != "IGS_not_CORS").all()
    assert df["id"].str.len().eq(4).all()
    assert df["lat"].between(-90, 90).all()


@pytest.mark.network
def test_live_igs_list():
    df = networks.load_igs()
    # current public roster = recordsFiltered (533 as of 2026-08); the
    # API's recordsTotal counts historical stations it never returns
    assert len(df) > 500
    assert df["name"].str.len().eq(9).all()
