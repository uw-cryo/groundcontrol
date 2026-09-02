"""Source parse() tests (offline, recorded fixtures) + the dispatcher contract.

Fixtures recorded live 2026-07 (Casa Grande area): a 12-record NDE sample, a
1-record OPUS sample, and a 23-row clip of the pinned national 3DEP
checkpoints parquet. fetch() halves are @network (Increment 1 integration).
"""

import json
from pathlib import Path

import geopandas as gpd
import pytest

from groundcontrol import schema
from groundcontrol.sources import checkpoints_3dep, fetch_control, ngs

DATA = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# parse() — pure, offline
# ---------------------------------------------------------------------------

def test_parse_3dep_sample():
    raw = gpd.read_parquet(DATA / "checkpoints_3dep_sample.parquet")
    out = schema.normalize(checkpoints_3dep.parse(raw), source="3dep")
    schema.validate(out)
    assert len(out) == len(raw) > 0
    assert set(out["point_type"].unique()) <= {"NVA", "VVA", "BVA", "Unknown"}
    assert out["height"].notna().all() and (out["height"] > 0).all()
    assert (out["coord_epoch"] == 2010.0).all()
    assert (out["frame_epoch"] == 2010.0).all()
    assert out["measurement_datetime"].notna().all()
    assert not out.geometry.has_z.any()
    # per-point provenance survives in raw as JSON
    assert "source_geoid" in json.loads(out["raw"].iloc[0])


def test_parse_3dep_empty_aoi():
    """AOI with no checkpoints: parse must not crash on the raw compound CRS.

    Regression: the empty branch passed ``raw.geometry`` still carrying the
    parquet's compound EPSG:6349, conflicting with ``crs="EPSG:6318"``
    (geopandas >= 1.0 raises).
    """
    raw = gpd.read_parquet(DATA / "checkpoints_3dep_sample.parquet").iloc[0:0]
    out = schema.normalize(checkpoints_3dep.parse(raw), source="3dep")
    assert len(out) == 0
    assert out.crs is not None and out.crs.to_epsg() == 6318


def test_parse_nde_sample():
    records = json.loads((DATA / "ngs_nde_sample.json").read_text())
    out = schema.normalize(ngs.parse_nde(records), source="ngs")
    schema.validate(out, require_crs=False)  # un-landed: no single CRS claimed yet
    assert len(out) == len(records)
    assert out["id"].str.match(r"^[A-Z]{2}\d{4}$").all()  # NGS PIDs
    assert out.crs is None  # per-row mixed realizations until land_horizontal
    # B7: every row maps to a known realization EPSG
    assert set(out["horizontal_crs"].unique()) <= {
        "EPSG:6318", "EPSG:4152", "EPSG:4269", "EPSG:4759", "EPSG:8860", "EPSG:6783"}
    # blanks (' ') coerced to NaN, not zeros
    assert out["height"].isna().sum() < len(out)
    # mixed realizations are the norm — provenance must be carried
    assert out["ref_frame"].notna().all()


def test_vert_crs_mapping_is_honest():
    """vertical_crs maps the actual vertDatum string; unknown -> NA, never assumed."""
    import pandas as pd
    s = pd.Series(["NAVD 88", "NGVD 29", " ", "", None])
    out = ngs._vert_crs(s)
    assert out.iloc[0] == "EPSG:5703"
    assert out.iloc[1] == "EPSG:7968"
    assert out.iloc[2:].isna().all()


def test_parse_nde_empty():
    out = ngs.parse_nde([])
    assert len(out) == 0
    schema.validate(schema.normalize(out, source="ngs") if len(out) else out,
                    require_crs=False)


def test_parse_opus_sample():
    records = json.loads((DATA / "ngs_opus_sample.json").read_text())
    out = schema.normalize(ngs.parse_opus(records), source="opus")
    schema.validate(out, require_crs=False)  # un-landed
    assert (out["point_type"] == "gnss_campaign").all()
    assert (out["horizontal_crs"] == "EPSG:6318").all()  # OPUS is NAD_83(2011)
    assert (out["coord_epoch"] == 2010.0).all()  # refFrame NAD_83(2011), epoch 2010.0000
    assert out["measurement_datetime"].notna().all()  # obsTimeStart


def test_opus_stability_tier():
    # per-row taxonomy (2026-08-22): tier decoded from each record's own
    # stabilityCode. Every branch is exercised on NON-empty selections —
    # the recorded fixture carries a single "C" record, so mask-based
    # .all() assertions over it were vacuous (audit finding)
    import pandas as pd
    base = json.loads((DATA / "ngs_opus_sample.json").read_text())[0]
    recs = []
    for i, code in enumerate(["A", "B", "C", "D", "X", None]):
        r = dict(base, pid=f"AB{i:04d}")
        if code is None:
            r.pop("stabilityCode", None)
        else:
            r["stabilityCode"] = code
        recs.append(r)
    out = ngs.parse_opus(recs)
    tier = ngs.opus_stability_tier(out)
    assert list(tier.iloc[:4]) == ["A/B", "A/B", "C/D", "C/D"]
    assert tier.iloc[4:].isna().all()  # out-of-range + missing -> NA, no guess
    # a missing code serializes as JSON null in raw, never the string "nan"
    assert json.loads(out.iloc[5]["raw"])["stabilityCode"] is None
    # and pre-null-fix parquets (literal "nan" strings) bucket identically:
    # expand_attributes normalizes "nan" to NA at the one reader
    old_style = out.iloc[[0]].copy()
    old_style["raw"] = pd.Series([json.dumps({"stabilityCode": "nan"})],
                                 dtype="string", index=old_style.index)
    assert ngs.opus_stability_tier(old_style).isna().all()
    assert ngs.expand_attributes(old_style, fields=["stabilityCode"],
                                 prefix="x_")["x_stabilityCode"].isna().all()
    # rows with no parseable raw (other sources) -> NA
    foreign = out.copy()
    foreign["raw"] = pd.Series(["not json"] * len(out), dtype="string",
                               index=out.index)
    assert ngs.opus_stability_tier(foreign).isna().all()


# ---------------------------------------------------------------------------
# dispatcher contract (offline, monkeypatched)
# ---------------------------------------------------------------------------

def _fake_provider_ok(bounds):
    return json.loads((DATA / "ngs_nde_sample.json").read_text())


def _fake_provider_boom(bounds):
    raise RuntimeError("boom")


def test_dispatcher_degrades_gracefully(monkeypatch):
    """One source raising must not take down the others (plan dispatcher contract)."""
    import groundcontrol.sources as srcs
    monkeypatch.setitem(srcs.PROVIDERS, "ngs", (_fake_provider_ok, ngs.parse_nde))
    monkeypatch.setitem(srcs.PROVIDERS, "3dep", (_fake_provider_boom, checkpoints_3dep.parse))
    gdf, status = fetch_control((-112, 32, -111, 33), sources=("3dep", "ngs"))
    assert status["3dep"]["error"] and "boom" in status["3dep"]["error"]
    assert status["ngs"]["error"] is None and status["ngs"]["n_rows"] > 0
    assert len(gdf) == status["ngs"]["n_rows"]
    schema.validate(gdf)


def test_dispatcher_total_failure_returns_empty_schema(monkeypatch):
    import groundcontrol.sources as srcs
    monkeypatch.setitem(srcs.PROVIDERS, "ngs", (_fake_provider_boom, ngs.parse_nde))
    gdf, status = fetch_control((-112, 32, -111, 33), sources=("ngs",))
    assert len(gdf) == 0
    assert list(schema.COLUMNS.keys()) == [c for c in gdf.columns if c != "geometry"]


def test_dispatcher_unknown_source_reported():
    gdf, status = fetch_control((-112, 32, -111, 33), sources=("nope",))
    assert "unknown source" in status["nope"]["error"]


def test_target_crs_not_implemented_is_loud():
    with pytest.raises(NotImplementedError, match="target"):
        fetch_control((-112, 32, -111, 33), target_crs="EPSG:9989")


# ---------------------------------------------------------------------------
# live integration (network)
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_fetch_control_casa_grande_live():
    bbox = (-111.89736772738966, 32.6429637773595,
            -111.58046670965014, 32.913396504245796)
    gdf, status = fetch_control(bbox)
    assert all(v["error"] is None for v in status.values())
    assert status["ngs"]["n_rows"] > 400  # regression: notebook-era count ~505
    assert status["3dep"]["n_rows"] > 10
    schema.validate(gdf)


def test_parse_nde_quarantines_unmapped_realization():
    """#21: an exotic realization must cost its ROW, never the source."""
    records = json.loads((DATA / "ngs_nde_sample.json").read_text())
    bad = dict(records[0])
    bad["pid"] = "XX9999"
    bad["posDatum"] = "NAD 83(CORS)"
    out = ngs.parse_nde(records + [bad])
    assert len(out) == len(ngs.parse_nde(records))
    assert "XX9999" not in set(out["id"])
    sk = out.attrs["skipped"]
    assert sk["n"] == 1 and any("CORS" in k for k in sk["reasons"])


def test_dispatcher_reports_quarantined_rows(monkeypatch):
    """#21: quarantine surfaces as n_skipped/skip_reasons in status."""
    import groundcontrol.sources as srcs
    records = json.loads((DATA / "ngs_nde_sample.json").read_text())
    bad = dict(records[0])
    bad["pid"] = "XX9999"
    bad["posDatum"] = "NAD 83(CORS)"
    monkeypatch.setitem(srcs.PROVIDERS, "ngs",
                        (lambda b: records + [bad], ngs.parse_nde))
    gdf, status = fetch_control((-112, 32, -111, 33), sources=("ngs",))
    assert status["ngs"]["error"] is None and status["ngs"]["n_rows"] > 0
    assert status["ngs"]["n_skipped"] == 1
    assert any("CORS" in k for k in status["ngs"]["skip_reasons"])
    assert "XX9999" not in set(gdf["id"])


def test_parse_nde_all_rows_quarantined_returns_schema_empty():
    """#21 edge: EVERY row unmappable must yield a schema-shaped empty
    frame with a valid CRS (not a CRS-less frame that fails landing)."""
    records = json.loads((DATA / "ngs_nde_sample.json").read_text())
    bad = [dict(r, posDatum="NAD 83(CORS)") for r in records]
    out = ngs.parse_nde(bad)
    assert len(out) == 0 and out.crs is not None
    assert out.attrs["skipped"]["n"] == len(bad)


def test_parse_nde_quarantines_missing_realization():
    """#21 follow-up (Copilot, PR #26): a null posDatum maps to pd.NA and
    must quarantine as '(missing)', not raise TypeError."""
    records = json.loads((DATA / "ngs_nde_sample.json").read_text())
    bad = dict(records[0])
    bad["pid"] = "XX9998"
    bad.pop("posDatum", None)
    out = ngs.parse_nde(records + [bad])
    assert "XX9998" not in set(out["id"])
    assert out.attrs["skipped"]["reasons"].get("(missing)") == 1


# ---------------------------------------------- landing_crs (rasuwa 2026-08-29)

def _nepal_rows(crs_code="EPSG:7912", coord_epoch=2022.27, n=3):
    """Schema-shaped dynamic-frame rows near Rasuwa (the NGL Nepal shape)."""
    import geopandas as gpd
    import numpy as np
    lons = np.linspace(85.2, 85.4, n)
    lats = np.linspace(28.1, 28.3, n)
    return gpd.GeoDataFrame({
        "id": [f"NP{i:02d}" for i in range(n)],
        "point_type": ["gnss_cont"] * n,
        "height": np.linspace(1400.0, 3900.0, n),
        "height_datum": ["ellipsoidal"] * n,
        "horizontal_crs": [crs_code] * n,
        "vertical_crs": [crs_code] * n,
        "ref_frame": ["IGS14"] * n,
        "coord_epoch": [coord_epoch] * n,
    }, geometry=gpd.points_from_xy(lons, lats), crs=None)


def _fake_provider(rows):
    def fetch(bounds):
        return {}

    def parse(raw):
        return rows.copy()
    return fetch, parse


def test_cache_write_text_is_utf8(tmp_path, monkeypatch):
    """The atomic writer's text branch is explicitly UTF-8 (PR #29 review:
    fdopen's default is the platform locale; the write_text it replaced
    was UTF-8) — a cached tenv3/steps line with non-ASCII bytes must round
    trip on any locale."""
    import io as _stdio   # os.fdopen's DEFAULT comes from io.text_encoding
    # emulate a Latin-1 locale: an explicit encoding passes through, an
    # omitted one (the pre-fix code) resolves to latin-1 and must fail
    monkeypatch.setattr(_stdio, "text_encoding",
                        lambda encoding, stacklevel=2: encoding or "latin-1")
    target = tmp_path / "steps.txt"
    text = "P123  \u00e9\u2014\u5730 step\n"
    checkpoints_3dep.cache_write(target, text)
    assert target.read_bytes() == text.encode("utf-8")
    assert target.read_text(encoding="utf-8") == text
    checkpoints_3dep.cache_write(target, b"\xff\xfebytes")     # bytes branch untouched
    assert target.read_bytes() == b"\xff\xfebytes"


def test_fetch_control_source_options(monkeypatch):
    """Per-source parse options reach the named source's parse; a name that
    is not a provider raises (a typo is not a silently ignored option); an
    option the source does not accept degrades THAT source into status."""
    from groundcontrol import sources
    seen = {}
    fetch, parse = _fake_provider(_nepal_rows())

    def parse_opt(raw, flavor="plain"):
        seen["flavor"] = flavor
        return parse(raw)
    monkeypatch.setitem(sources.PROVIDERS, "fake", (fetch, parse_opt))
    gdf, status = sources.fetch_control((85.0, 28.0, 86.0, 29.0), sources=("fake",),
                                        landing_crs="EPSG:7912",
                                        source_options={"fake": {"flavor": "wide"}})
    assert seen == {"flavor": "wide"} and status["fake"]["n_rows"] == 3
    gdf, status = sources.fetch_control((85.0, 28.0, 86.0, 29.0), sources=("fake",),
                                        landing_crs="EPSG:7912")
    assert seen == {"flavor": "plain"}
    with pytest.raises(ValueError, match="unknown source"):
        sources.fetch_control((85.0, 28.0, 86.0, 29.0), sources=("fake",),
                              source_options={"fakke": {"flavor": "wide"}})
    with pytest.raises(ValueError, match="not in sources"):   # valid, unrequested
        sources.fetch_control((85.0, 28.0, 86.0, 29.0), sources=("fake",),
                              source_options={"ngs": {"flavor": "wide"}})
    monkeypatch.setitem(sources.PROVIDERS, "fake", (fetch, parse))   # no kwargs
    gdf, status = sources.fetch_control((85.0, 28.0, 86.0, 29.0), sources=("fake",),
                                        landing_crs="EPSG:7912",
                                        source_options={"fake": {"flavor": "wide"}})
    assert status["fake"]["n_rows"] == 0 and "TypeError" in status["fake"]["error"]


def _fixture_faa_provider():
    """The offline FAA fixture as a provider (no network, no shared cache)."""
    from pathlib import Path

    from groundcontrol.sources import faa
    lines = (Path(__file__).parent / "data" / "faa_apt_sample.txt") \
        .read_text(encoding="latin-1").splitlines(keepends=True)

    def fetch(bounds):
        return {"cycle": "2026-08-06", "aoi_bounds_4326": tuple(bounds),
                "lines": lines}
    return fetch, faa.parse


def test_fetch_cli_faa_ownership_flag(monkeypatch, tmp_path, capsys):
    """--faa-ownership rides to fetch_control as the faa parse option and
    changes the rows written; the default sends no option (the source
    default applies); an unrequested faa never appears in source_options;
    a bad code is preflighted (exit error, nothing fetched) — round-7."""
    import geopandas as gpd

    from groundcontrol import sources
    from groundcontrol.cli import fetch_control_main
    calls = []
    real = sources.fetch_control

    def spy(*a, **k):
        calls.append(k.get("source_options"))
        return real(*a, **k)
    monkeypatch.setattr(sources, "fetch_control", spy)
    monkeypatch.setitem(sources.PROVIDERS, "fake", _fake_provider(_nepal_rows()))
    monkeypatch.setitem(sources.PROVIDERS, "faa", _fixture_faa_provider())
    out = tmp_path / "c.parquet"
    nepal = ["--aoi=85.0,28.0,86.0,29.0", "--landing-crs", "EPSG:7912",
             "--no-figures", "--out", str(out)]
    lv = ["--aoi=-115.35,35.9,-115.0,36.4", "--no-figures", "--out", str(out)]
    assert fetch_control_main(nepal + ["--sources", "fake"]) == 0
    assert fetch_control_main(lv + ["--sources", "faa", "--faa-ownership", "all"]) == 0
    assert len(gpd.read_parquet(out)) == 27
    assert fetch_control_main(lv + ["--sources", "faa"]) == 0
    assert len(gpd.read_parquet(out)) == 22
    assert calls == [{}, {"faa": {"ownership": "all"}}, {}]
    with pytest.raises(SystemExit, match="unknown code"):
        fetch_control_main(lv + ["--sources", "faa", "--faa-ownership", "PU,XX"])
    assert len(calls) == 3                       # preflight: nothing fetched
    # the flag is irrelevant without the faa source: no preflight, no option
    assert fetch_control_main(nepal + ["--sources", "fake",
                                       "--faa-ownership", "PU,XX"]) == 0
    assert calls[-1] == {}


def test_assess_cache_ownership_warning(tmp_path, capsys):
    """A reused control cache is not re-filtered: the cache branch warns
    from the DATA when cached faa rows sit outside --faa-ownership, and
    from the sidecar's skip count when the flag would widen (round-7)."""
    import argparse
    import json

    from groundcontrol.cli import _check_cache_ownership
    from groundcontrol.sources import faa
    fetch, _ = _fixture_faa_provider()
    ctl = faa.parse(fetch((-180.0, -90.0, 180.0, 90.0)), ownership="all")
    ctl["source"] = "faa"
    cache = tmp_path / "site_control.parquet"
    Path(str(cache) + ".provenance.json").write_text(json.dumps(
        {"status": {"faa": {"n_rows": 26, "n_skipped": 5,
                            "skip_reasons": {"ownership filter": 5}}}}))
    ns = argparse.Namespace(faa_ownership="PU,PR")
    _check_cache_ownership(ctl, cache, ns, ("faa",))
    err = capsys.readouterr().err
    assert "holds 5 faa row(s) at facilities outside --faa-ownership PU,PR" in err
    ns = argparse.Namespace(faa_ownership="all")
    _check_cache_ownership(ctl[ctl["id"].str[:3] != "LSV"], cache, ns, ("faa",))
    err = capsys.readouterr().err
    assert "fetched under an ownership filter (5 faa row(s) skipped)" in err
    # civil cache, civil flag: silence; faa not requested: silence
    ns = argparse.Namespace(faa_ownership="PU,PR")
    _check_cache_ownership(ctl[ctl["id"].str[:3] != "LSV"], cache, ns, ("faa",))
    _check_cache_ownership(ctl, cache, ns, ("ngs",))
    assert capsys.readouterr().err == ""


def test_fetch_default_landing_degrades_outside_conus(monkeypatch):
    """CHARACTERIZATION (passes on main too — the pre-fix behavior IS the
    fail-loud contract): the Nepal repro (rasuwa session, 753ae54).
    ITRF->NAD83(2011) has a North America area of use, so the default
    landing fail-louds — per source, into status, never a crash."""
    from groundcontrol import sources
    monkeypatch.setitem(sources.PROVIDERS, "fake", _fake_provider(_nepal_rows()))
    gdf, status = sources.fetch_control((85.0, 28.0, 86.0, 29.0), sources=("fake",))
    assert len(gdf) == 0
    assert status["fake"]["n_rows"] == 0
    assert "NoTransformPathError" in status["fake"]["error"]


def test_fetch_landing_crs_serves_nepal_at_2d_counterpart(monkeypatch):
    """landing_crs=EPSG:7912 (3D) lands at EPSG:9000 (round 2: the frame-level
    CRS must be 2D like the 6318 default — no stamped height axis); the leg
    is the null ITRF2014 offset at acc 0, tt honored; heights untouched."""
    from groundcontrol import sources
    monkeypatch.setitem(sources.PROVIDERS, "fake", _fake_provider(_nepal_rows()))
    gdf, status = sources.fetch_control((85.0, 28.0, 86.0, 29.0), sources=("fake",),
                                        landing_crs="EPSG:7912")
    assert status["fake"] == {"n_rows": 3, "error": None}
    assert len(gdf) == 3 and gdf.crs.to_epsg() == 9000
    assert (gdf["transform_id"] == "land:EPSG:7912->EPSG:9000|acc=0.0m|tt=coord_epoch").all()
    assert gdf["height"].notna().all()          # ellipsoidal heights ride through
    # rows already tagged with the 2D code take the identity relabel
    monkeypatch.setitem(sources.PROVIDERS, "fake",
                        _fake_provider(_nepal_rows(crs_code="EPSG:9000")))
    gdf, _ = sources.fetch_control((85.0, 28.0, 86.0, 29.0), sources=("fake",),
                                   landing_crs="EPSG:9000")
    assert (gdf["transform_id"] == "land:identity:EPSG:9000").all()


def test_fetch_landing_crs_rejects_nonhorizontal_and_ensembles():
    """Horizontal-only gate (re-audit HIGH: a compound landing stamped NAVD88
    on ellipsoidal heights) + no datum ensembles."""
    from groundcontrol import sources
    for bad, msg in (("EPSG:32645", "plain geographic"),      # projected
                     ("EPSG:7789", "plain geographic"),        # geocentric
                     ("EPSG:6318+5703", "plain geographic"),   # compound (HIGH)
                     ("EPSG:6349", "plain geographic"),        # compound by code
                     ("EPSG:4326", "ENSEMBLE"),                # WGS84 ensemble
                     ("EPSG:4979", "ENSEMBLE"),
                     # OGC authority: no EPSG entry -> refused by the
                     # resolve gate (still the ensemble, still refused)
                     ("OGC:CRS84", "does not resolve to an EPSG")):
        with pytest.raises(ValueError, match=msg):
            sources.validate_landing_crs(bad)
    with pytest.raises(ValueError, match="plain geographic"):
        sources.fetch_control((85.0, 28.0, 86.0, 29.0), sources=(),
                              landing_crs="EPSG:6318+5703")


def test_fetch_landing_crs_normalized_spellings(monkeypatch):
    """Int / lowercase / 3D spellings all normalize to one 2D code with one
    transform_id (re-audit LOWs); validate_landing_crs is idempotent and
    gates the RESOLVED entry, so no spelling bypasses the ensemble/bound
    refusals (round 2)."""
    from groundcontrol import sources
    monkeypatch.setitem(sources.PROVIDERS, "fake", _fake_provider(_nepal_rows()))
    for spelling in (7912, "epsg:7912", "EPSG:9000"):
        gdf, _ = sources.fetch_control((85.0, 28.0, 86.0, 29.0), sources=("fake",),
                                       landing_crs=spelling)
        assert gdf.crs.to_epsg() == 9000, spelling
    v = sources.validate_landing_crs("EPSG:7912")
    assert v == "EPSG:9000" and sources.validate_landing_crs(v) == v   # idempotent
    assert sources.validate_landing_crs("EPSG:6319") == "EPSG:6318"    # demotes to default
    with pytest.raises(ValueError, match="ENSEMBLE"):                  # spelling bypass
        sources.validate_landing_crs("+proj=longlat +datum=WGS84 +no_defs")
    with pytest.raises(ValueError, match="BoundCRS"):
        sources.validate_landing_crs(
            "+proj=longlat +ellps=GRS80 +towgs84=1000,0,0,0,0,0,0 +no_defs")
    with pytest.raises(ValueError, match="does not resolve to an EPSG"):
        sources.validate_landing_crs("+proj=longlat +ellps=airy +no_defs")


def test_fetch_landing_static_source_into_dynamic_frame_degrades(monkeypatch):
    """A non-dynamic subset cannot enter a dynamic landing frame without the
    D6 target-epoch semantics: the guard raises inside land_horizontal and
    the dispatcher degrades that source into status. CONUS variant is the
    load-bearing one (re-audit MED: in Nepal the 6318->9989 op is missing
    anyway, so the old test failed a guard revert for the wrong reason;
    in CONUS the op EXISTS and a guardless landing silently evaluates the
    time-dependent Helmert at t_epoch — measured 0.20 m at epoch 2022)."""
    import geopandas as gpd
    import numpy as np
    from groundcontrol import sources
    conus = _nepal_rows(crs_code="EPSG:6318")
    conus = gpd.GeoDataFrame(conus, geometry=gpd.points_from_xy(
        np.linspace(-112.2, -112.0, len(conus)),
        np.linspace(32.7, 32.9, len(conus))), crs=None)
    for rows, bbox in ((conus, (-112.5, 32.5, -111.5, 33.0)),
                       (_nepal_rows(crs_code="EPSG:6318"), (85.0, 28.0, 86.0, 29.0))):
        monkeypatch.setitem(sources.PROVIDERS, "fake", _fake_provider(rows))
        gdf, status = sources.fetch_control(bbox, sources=("fake",),
                                            landing_crs="EPSG:9989")
        assert len(gdf) == 0
        assert "not a dynamic frame" in status["fake"]["error"]
        assert "target epoch" in status["fake"]["error"]


def test_fetch_cli_landing_crs_flag(tmp_path, monkeypatch):
    from groundcontrol import sources
    from groundcontrol.cli import fetch_control_main
    monkeypatch.setitem(sources.PROVIDERS, "fake", _fake_provider(_nepal_rows()))
    out = tmp_path / "nepal.parquet"
    rc = fetch_control_main(["--aoi=85.0,28.0,86.0,29.0", "--sources", "fake",
                             "--landing-crs", "EPSG:7912", "--out", str(out)])
    assert rc == 0
    import geopandas as gpd
    # in-memory landing is EPSG:9000 (2D); io's honest compound promotion
    # composes it with the UNIFORM per-row vertical_crs EPSG:7912, and PROJ
    # resolves EPSG:9000+7912 back to the registered 3D EPSG:7912 — correct
    # here because every height IS ITRF2014 ellipsoidal (round-3 L3-4)
    assert gpd.read_parquet(out).crs.to_epsg() == 7912
    with pytest.raises(SystemExit, match="--landing-crs: not a valid CRS"):
        fetch_control_main(["--aoi=85.0,28.0,86.0,29.0", "--sources", "fake",
                            "--landing-crs", "EPSG:99999999", "--out", str(out)])
    # a projected landing is a VALID CRS: the horizontal-only gate must still
    # produce a clean CLI error, not a traceback (re-audit MED), before any
    # fetch; the expensive AOI read also comes after it (see
    # test_fetch_cli_reads_aoi_only_after_output_checks for the read ordering)
    from groundcontrol import aoi as aoi_pkg
    aoi_file = tmp_path / "aoi.geojson"
    aoi_file.write_text("{}")   # exists (passes the cheap probe), never read
    monkeypatch.setattr(aoi_pkg, "read_aoi",
                        lambda *a, **k: pytest.fail("AOI read before the landing gate"))
    with pytest.raises(SystemExit, match="error: .*plain geographic"):
        fetch_control_main(["--aoi", str(aoi_file), "--sources", "fake",
                            "--landing-crs", "EPSG:32645", "--out", str(out)])
