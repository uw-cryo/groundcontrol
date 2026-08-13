"""NGL GNSS source tests (Increment 1.5a) + the D6 dynamic-frame landing rule.

Offline fixtures recorded live 2026-07-02:

- ``ngl_dataholdings_sample.txt`` — 20 real DataHoldings.txt rows incl. a
  spaced ``StaOrigName`` (AINR -> ``FRE2 1291``), a row with NO StaOrigName
  (00NA), CONUS rows with ``Long`` > 180 (the plan-B1 wrap), an
  antimeridian-crossing row (RYR1, Long 180.5261), and the Las Vegas cluster.
- ``ngl_CLV1_IGS14_sample.tenv3`` — header + 60 consecutive real rows of
  station CLV1 (Las Vegas), 2017.859-2018.032.

The land_horizontal tt-sensitivity tests are offline: ITRF2014->NAD83(2011)
is a time-dependent Helmert (EPSG:8970) — no grids required.
"""

import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from groundcontrol import schema
from groundcontrol.crs import decyear, is_dynamic_frame, land_horizontal
from groundcontrol.sources import PROVIDERS, ngl

DATA = Path(__file__).parent / "data"
LV_BBOX = (-115.47, 35.87, -114.87, 36.44)  # Las Vegas


def _index():
    return ngl.parse_dataholdings((DATA / "ngl_dataholdings_sample.txt").read_text())


def _tenv3_text():
    return (DATA / "ngl_CLV1_IGS14_sample.tenv3").read_text()


_MIDAS_VEL_COLS = ["vel_e", "vel_n", "vel_u", "sig_vel_e", "sig_vel_n", "sig_vel_u"]


def _midas_map():
    """station id -> MIDAS velocity sub-dict, from the offline fixture."""
    m = ngl.parse_midas((DATA / "ngl_midas_sample.txt").read_text())
    return {str(r["sta"]): {c: float(r[c]) for c in _MIDAS_VEL_COLS}
            for _, r in m.iterrows()}


def _raw(frame="IGS14", epoch=None, time_range=None, sta="CLV1", with_midas=True):
    """Build a fetch()-shaped payload from the offline fixtures.

    ``with_midas`` mirrors ``fetch(with_velocities=True)``: attach the station's
    own MIDAS velocity (or None if absent from the fixture) to ``meta['midas']``.
    """
    idx = _index()
    row = idx[idx["sta"] == sta].iloc[0]
    meta = ngl._station_meta(row)
    if with_midas:
        meta["midas"] = _midas_map().get(sta)
    return {
        "frame": frame, "epoch": epoch, "time_range": time_range,
        "stations": [{"meta": meta, "tenv3": _tenv3_text()}],
    }


# ---------------------------------------------------------------------------
# DataHoldings index parsing (bounded split + B1 wrap)
# ---------------------------------------------------------------------------

def test_parse_dataholdings_bounded_split_keeps_spaced_names():
    idx = _index()
    assert len(idx) == 20
    # verified gotcha: StaOrigName contains spaces — a naive split breaks here
    assert idx.loc[idx["sta"] == "AINR", "sta_orig_name"].iloc[0] == "FRE2 1291"
    # ...and may be absent entirely (11-field row)
    assert idx.loc[idx["sta"] == "00NA", "sta_orig_name"].iloc[0] == ""
    assert idx["num_sol"].gt(0).all()
    assert idx["dtbeg"].notna().all() and idx["dtend"].notna().all()


def test_lon_wrap_b1_conus_and_antimeridian():
    idx = _index().set_index("sta")
    # CONUS: 244.7418 -> -115.2582 (mod 360 would leave it in the wrong hemisphere)
    assert idx.loc["CLV1", "lon"] == pytest.approx(-115.2582, abs=1e-4)
    assert idx.loc["APEX", "lon"] == pytest.approx(-114.9318, abs=1e-4)
    # antimeridian: 180.5261 -> -179.4739 (side preservation), 179.3013 stays
    assert idx.loc["RYR1", "lon"] == pytest.approx(-179.4739, abs=1e-4)
    assert idx.loc["AC66", "lon"] == pytest.approx(179.3013, abs=1e-4)
    assert idx["lon"].between(-180, 180).all()


def test_wrap_lon_regression_not_mod_360():
    # plan B1: the "mod 360" note is backwards — a CONUS 250.x must go negative
    assert ngl._wrap_lon(250.5) == pytest.approx(-109.5)
    assert ngl._wrap_lon(-109.5) == pytest.approx(-109.5)  # idempotent


def test_bbox_filter_las_vegas():
    sel = ngl._select_stations(_index(), LV_BBOX)
    assert len(sel) > 5
    assert {"APEX", "CLV1", "NVBM"} <= set(sel["sta"])
    assert "00NA" not in set(sel["sta"])  # Australia
    assert "AINR" not in set(sel["sta"])  # Austria


def test_temporal_overlap_filter():
    idx = _index()
    # epoch 2000.0 +/-30 d: APEX (1999-2009) overlaps; CLV1 (2008-) does not
    sel = ngl._select_stations(idx, LV_BBOX, epoch=2000.0)
    assert "APEX" in set(sel["sta"]) and "CLV1" not in set(sel["sta"])
    # time_range fully inside CLV1's span, after APEX decommissioning
    sel = ngl._select_stations(idx, LV_BBOX, time_range=("2020-01-01", "2020-12-31"))
    assert "CLV1" in set(sel["sta"]) and "APEX" not in set(sel["sta"])


# ---------------------------------------------------------------------------
# tenv3 parsing
# ---------------------------------------------------------------------------

def test_parse_tenv3_columns_and_values():
    ts = ngl.parse_tenv3(_tenv3_text())
    assert len(ts) == 60
    assert set(ts.columns) >= ngl._TENV3_REQUIRED
    assert (ts["site"] == "CLV1").all()
    assert ts["date"].is_monotonic_increasing
    # B1 wrap applied to the tenv3 longitude column too
    assert ts["longitude"].between(-180, 180).all()
    assert ts["longitude"].median() == pytest.approx(-115.2582, abs=1e-3)
    assert ts["latitude"].median() == pytest.approx(36.2146, abs=1e-3)
    assert ts["height"].between(700, 707).all()  # CLV1 ellipsoidal height ~703 m


def test_read_tenv3_offline_from_cache(tmp_path, monkeypatch):
    """read_tenv3 serves a fresh cached file with NO network hit."""
    monkeypatch.setenv("GROUNDCONTROL_CACHE_DIR", str(tmp_path))
    (tmp_path / "ngl_CLV1_IGS14.tenv3").write_text(_tenv3_text())

    def _boom(*a, **k):
        raise AssertionError("network hit despite fresh cache")

    monkeypatch.setattr(ngl.requests, "get", _boom)
    ts = ngl.read_tenv3("clv1")  # station ID normalized to upper case
    assert len(ts) == 60
    assert set(ts.columns) >= ngl._TENV3_REQUIRED
    assert ts["date"].is_monotonic_increasing
    # full up = integer reference + fractional part must equal the height column
    up_full = ts["u0"].to_numpy() + ts["up"].to_numpy()
    np.testing.assert_allclose(up_full, ts["height"].to_numpy(), atol=1e-5)


def test_read_tenv3_stale_cache_refreshes(tmp_path, monkeypatch):
    import os
    monkeypatch.setenv("GROUNDCONTROL_CACHE_DIR", str(tmp_path))
    local = tmp_path / "ngl_CLV1_IGS14.tenv3"
    local.write_text("stale")
    eight_days_ago = pd.Timestamp.now().timestamp() - 8 * 86400
    os.utime(local, (eight_days_ago, eight_days_ago))

    class _Resp:
        text = _tenv3_text()

        def raise_for_status(self):
            pass

    monkeypatch.setattr(ngl.requests, "get", lambda *a, **k: _Resp())
    assert len(ngl.read_tenv3("CLV1")) == 60
    assert "CLV1" in local.read_text()  # cache was rewritten


def test_read_tenv3_unknown_frame_raises():
    with pytest.raises(ValueError, match="IGS14"):
        ngl.read_tenv3("CLV1", frame="ITRF97")


# ---------------------------------------------------------------------------
# steps.txt (plan 1.5b/B10) — fixture: 15 real lines recorded live 2026-07-04
# (9 type-1 incl. an "Unknown" event, 6 type-2 incl. pre-2000 dates for the
# %y century pivot; Las Vegas stations + GOL2 for the 1994 Northridge row)
# ---------------------------------------------------------------------------

def _steps_text():
    return (DATA / "ngl_steps_sample.txt").read_text()


def test_parse_steps_both_types_bounded_split():
    st = ngl.parse_steps(_steps_text())
    assert len(st) == 15
    assert list(st.columns) == ["sta", "date", "type", "event", "threshold_km",
                                "distance_km", "magnitude", "event_id"]
    assert st["type"].isin([1, 2]).all()
    eq, ev = st[st["type"] == 2], st[st["type"] == 1]
    assert len(ev) == 9 and len(eq) == 6
    # type-specific columns are NA exactly where they do not apply
    assert ev["event"].notna().all()
    assert ev[["threshold_km", "distance_km", "magnitude"]].isna().all().all()
    assert ev["event_id"].isna().all()
    assert eq["event"].isna().all()
    assert eq[["threshold_km", "distance_km", "magnitude"]].notna().all().all()
    # known type-1 row (antenna change -> abrupt instrumental height jump)
    nvbm = st[(st["sta"] == "NVBM") & (st["type"] == 1)]
    assert "Antenna_and_Radome_Type_Changed" in set(nvbm["event"])
    assert pd.Timestamp("2016-11-15", tz="UTC") in set(nvbm["date"])
    # known type-2 row, all four trailing fields
    apex = st[(st["sta"] == "APEX") & (st["type"] == 2)].iloc[0]
    assert apex["date"] == pd.Timestamp("1999-10-16", tz="UTC")  # %y pivot: 99 -> 1999
    assert apex["threshold_km"] == pytest.approx(575.440)
    assert apex["distance_km"] == pytest.approx(225.783)
    assert apex["magnitude"] == pytest.approx(7.1)
    assert apex["event_id"] == "ci9108652"
    # century pivot both ways: 94 -> 1994, 10 -> 2010
    assert st[st["sta"] == "GOL2"]["date"].iloc[0].year == 1994
    assert st[st["sta"] == "CLV1"]["date"].iloc[0].year == 2010
    # sorted by station then date
    assert st.equals(st.sort_values(["sta", "date"], kind="stable")
                     .reset_index(drop=True))


def test_parse_steps_malformed_rows_raise():
    with pytest.raises(ValueError, match="line 1"):
        ngl.parse_steps("NVBM  16NOV15  1\n")  # type-1 without an event
    with pytest.raises(ValueError, match="type-2"):
        ngl.parse_steps("NVBM  10APR04  2   645.654   409.795\n")
    with pytest.raises(ValueError, match="unknown step type"):
        ngl.parse_steps("NVBM  10APR04  3  what\n")


def test_read_steps_offline_from_cache(tmp_path, monkeypatch):
    """read_steps serves a fresh cached file with NO network hit."""
    monkeypatch.setenv("GROUNDCONTROL_CACHE_DIR", str(tmp_path))
    (tmp_path / "ngl_steps.txt").write_text(_steps_text())

    def _boom(*a, **k):
        raise AssertionError("network hit despite fresh cache")

    monkeypatch.setattr(ngl.requests, "get", _boom)
    st = ngl.read_steps()
    assert len(st) == 15
    # station filter normalizes case; NVBM has 2 equipment + 2 earthquake rows
    nvbm = ngl.read_steps("nvbm")
    assert (nvbm["sta"] == "NVBM").all()
    assert nvbm["type"].value_counts().to_dict() == {1: 2, 2: 2}
    assert nvbm["date"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# MIDAS velocities (Blewitt et al. 2016) — fixture: 9 real midas.IGS14.txt
# rows recorded live 2026-07-04 (Las Vegas CLV1/NVCA/UNR1, Bay Area
# P224/SLAC/TIBB, Casa Grande AZCG/CAS7, and 00NA whose raw longitude is
# -229.156 — the continuous-longitude wrap case)
# ---------------------------------------------------------------------------

def _midas_text():
    return (DATA / "ngl_midas_sample.txt").read_text()


def test_parse_midas_layout_and_values():
    df = ngl.parse_midas(_midas_text())
    assert list(df.columns) == ngl._MIDAS_COLUMNS  # 27 readme columns
    assert len(df) == 9
    r = df.set_index("sta").loc["CLV1"]
    # velocities/uncertainties are m/yr; n_steps = steps ASSUMED from steps.txt
    assert r["vel_u"] == pytest.approx(-0.001612)
    assert r["sig_vel_u"] == pytest.approx(0.000514)
    assert r["n_steps"] == 4
    assert df["n_steps"].dtype == "int64"
    assert (df["duration_yr"] > 0).all()
    assert (df["t1"] > df["t0"]).all()
    # B1 wrap: 00NA comes as lon -229.156 (< -180!) -> +130.844 (Australia)
    assert df.set_index("sta").loc["00NA", "lon"] == pytest.approx(130.844, abs=1e-3)
    assert df["lon"].between(-180, 180).all()
    assert df["sta"].is_monotonic_increasing


def test_parse_midas_layout_drift_raises():
    with pytest.raises(ValueError, match="27"):
        ngl.parse_midas("AAAA MIDAS5 2008.0 2018.0 10.0\n")  # wrong column count
    # a single short row among good ones must fail loud, not NaN-fill
    with pytest.raises(ValueError, match="row"):
        ngl.parse_midas(_midas_text() + "ZZZZ MIDAS5 2008.0 2018.0 10.0 1 2 3\n")


def test_read_midas_offline_from_cache(tmp_path, monkeypatch):
    """read_midas serves a fresh cached file with NO network hit."""
    monkeypatch.setenv("GROUNDCONTROL_CACHE_DIR", str(tmp_path))
    (tmp_path / "ngl_midas_IGS14.txt").write_text(_midas_text())

    def _boom(*a, **k):
        raise AssertionError("network hit despite fresh cache")

    monkeypatch.setattr(ngl.requests, "get", _boom)
    df = ngl.read_midas()  # frame="IGS14" default matches the cache name
    assert len(df) == 9
    assert {"vel_u", "sig_vel_u", "n_steps"} <= set(df.columns)


def test_read_midas_stale_cache_refreshes(tmp_path, monkeypatch):
    import os
    monkeypatch.setenv("GROUNDCONTROL_CACHE_DIR", str(tmp_path))
    local = tmp_path / "ngl_midas_IGS14.txt"
    local.write_text("stale")
    eight_days_ago = pd.Timestamp.now().timestamp() - 8 * 86400
    os.utime(local, (eight_days_ago, eight_days_ago))

    class _Resp:
        text = _midas_text()

        def raise_for_status(self):
            pass

    monkeypatch.setattr(ngl.requests, "get", lambda *a, **k: _Resp())
    assert len(ngl.read_midas("IGS14")) == 9
    assert "CLV1" in local.read_text()  # cache was rewritten


def test_read_midas_empty_frame_raises():
    with pytest.raises(ValueError, match="frame"):
        ngl.read_midas("  ")


def test_coord_epoch_crosscheck_yymmmdd_vs_decimal_year():
    """Plan B9: every tenv3 row carries BOTH representations — they must agree."""
    ts = ngl.parse_tenv3(_tenv3_text())
    dy_from_date = decyear(ts["date"].dt.tz_localize(None))
    assert (dy_from_date - ts["decyear"]).abs().max() <= 0.003  # ~1 day


# ---------------------------------------------------------------------------
# position-at-epoch (path A)
# ---------------------------------------------------------------------------

def _synthetic_tenv3(dates, lats, lons, heights, ant=1.5):
    df = pd.DataFrame({
        "site": "SYNT",
        "date": pd.to_datetime(dates, utc=True),
        "latitude": lats, "longitude": lons, "height": heights,
        "sig_e": 0.001, "sig_n": 0.002, "sig_u": 0.003, "ant": ant,
    })
    df["decyear"] = decyear(df["date"].dt.tz_localize(None)).to_numpy()
    return df.sort_values("date").reset_index(drop=True)


def test_position_at_epoch_median_math_exact():
    # 5 solutions around 2020-06-01; component-wise medians are exact row values
    ts = _synthetic_tenv3(
        ["2020-05-28", "2020-05-30", "2020-06-01", "2020-06-03", "2020-06-05"],
        lats=[36.10, 36.30, 36.20, 36.50, 36.40],
        lons=[-115.3, -115.1, -115.2, -115.5, -115.4],
        heights=[703.0, 705.0, 704.0, 707.0, 706.0],
    )
    epoch = decyear(pd.Timestamp("2020-06-01"))
    win, desc = ngl._select_window(ts, epoch=epoch)
    assert len(win) == 5 and desc["mode"] == "epoch"
    pos = ngl._position_from_window(win)
    assert pos["lat"] == 36.30 and pos["lon"] == -115.30 and pos["height"] == 705.0
    assert pos["n_solutions_used"] == 5
    assert pos["ant_m"] == 1.5
    assert (pos["sig_e_m"], pos["sig_n_m"], pos["sig_u_m"]) == (0.001, 0.002, 0.003)
    # coord_epoch = median decimal year; measurement_datetime = nearest solution
    assert pos["coord_epoch"] == pytest.approx(decyear(pd.Timestamp("2020-06-01")))
    assert pos["measurement_datetime"] == pd.Timestamp("2020-06-01", tz="UTC")


def test_window_epoch_half_width_30_days():
    ts = _synthetic_tenv3(
        ["2020-01-01", "2020-05-25", "2020-06-01", "2020-07-15"],
        lats=[36.0] * 4, lons=[-115.0] * 4, heights=[700.0] * 4,
    )
    win, _ = ngl._select_window(ts, epoch=decyear(pd.Timestamp("2020-06-01")))
    # 2020-01-01 (152 d away) and 2020-07-15 (44 d) fall outside +/-30 d
    assert list(win["date"].dt.strftime("%Y-%m-%d")) == ["2020-05-25", "2020-06-01"]


def test_window_last_n_when_no_epoch():
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    ts = _synthetic_tenv3(dates, lats=[36.0] * 50, lons=[-115.0] * 50,
                          heights=[700.0] * 50)
    win, desc = ngl._select_window(ts)
    assert len(win) == ngl.LAST_N_SOLUTIONS == 30
    assert desc["mode"] == "last_n"
    assert win["date"].iloc[-1] == ts["date"].iloc[-1]  # the most recent solutions


def test_empty_window_drops_station_with_warning(caplog):
    raw = _raw(epoch=1995.0)  # far outside the fixture's 2017.86-2018.03 span
    with caplog.at_level(logging.WARNING, logger="groundcontrol.sources.ngl"):
        out = ngl.parse(raw)
    assert len(out) == 0
    assert any("CLV1" in r.message and "dropping" in r.message for r in caplog.records)
    # empty result is still schema-shaped (dispatcher contract)
    assert set(schema.COLUMNS) <= set(out.columns)


# ---------------------------------------------------------------------------
# parse() -> schema shape
# ---------------------------------------------------------------------------

def test_parse_schema_valid_and_frame_aliased():
    raw = _raw(epoch=2017.95)
    out = schema.normalize(ngl.parse(raw), source="ngl")
    schema.validate(out, require_crs=False)  # un-landed: native dynamic frame
    assert len(out) == 1
    r = out.iloc[0]
    assert r["id"] == "CLV1" and r["point_type"] == "gnss"
    # §3 frame aliasing: IGS14 -> EPSG:7912, NEVER an EPSG IGS code
    assert r["horizontal_crs"] == "EPSG:7912"
    assert r["vertical_crs"] == "EPSG:7912"
    assert r["native_crs"] == "EPSG:7912"
    assert r["ref_frame"] == "IGS14"          # provenance keeps the IGS name
    assert np.isnan(r["frame_epoch"])         # dynamic frame
    assert r["height_datum"] == "ellipsoidal"
    assert 700 < r["height"] < 707
    # coord_epoch = median decimal year of the used window, near the target
    assert abs(r["coord_epoch"] - 2017.95) < 0.09
    assert abs(r["measurement_epoch"] - r["coord_epoch"]) < 0.09
    assert r["measurement_datetime"].year == 2017
    # accuracy stays NaN (TODO D3); sigmas + antenna height live in raw
    assert np.isnan(r["acc_h"]) and np.isnan(r["acc_v"])
    # per-point MIDAS velocity now joined by station ID (was 1.5b)
    assert r["vel_e"] == pytest.approx(-0.015169)  # CLV1 MIDAS (Blewitt 2016), m/yr
    assert r["vel_n"] == pytest.approx(-0.008432)
    assert r["vel_u"] == pytest.approx(-0.001612)
    payload = json.loads(r["raw"])
    assert payload["n_solutions_used"] > 0
    assert 0 < payload["sig_e_m"] < 0.01
    assert "ant_m" in payload  # antenna height — assessment must correct for it
    assert payload["window"]["mode"] == "epoch"
    assert payload["num_sol"] == 5854
    # geometry = emitted median position, 2D
    assert r.geometry.x == pytest.approx(r["native_x"]) == pytest.approx(-115.2582, abs=1e-3)
    assert r.geometry.y == pytest.approx(r["native_y"])
    assert out.crs is None  # native frame; the dispatcher lands it


def test_parse_igs20_aliases_to_itrf2020():
    out = ngl.parse(_raw(frame="IGS20", epoch=2017.95))
    assert (out["horizontal_crs"] == "EPSG:9989").all()
    assert (out["vertical_crs"] == "EPSG:9989").all()
    assert (out["ref_frame"] == "IGS20").all()


def test_parse_joins_midas_velocity_and_sigmas():
    """fetch(with_velocities=True) -> meta['midas'] -> per-point vel_e/n/u + sig in raw."""
    r = ngl.parse(_raw(epoch=2017.95)).iloc[0]
    assert (r["vel_e"], r["vel_n"], r["vel_u"]) == pytest.approx(
        (-0.015169, -0.008432, -0.001612))  # CLV1 MIDAS IGS14, m/yr
    payload = json.loads(r["raw"])
    assert payload["sig_vel_e"] == pytest.approx(0.000197)
    assert payload["sig_vel_u"] == pytest.approx(0.000514)


def test_parse_without_midas_leaves_velocity_nan():
    """No MIDAS attached (with_velocities=False, or station absent) -> honest NaN."""
    r = ngl.parse(_raw(epoch=2017.95, with_midas=False)).iloc[0]
    assert np.isnan(r["vel_e"]) and np.isnan(r["vel_n"]) and np.isnan(r["vel_u"])
    payload = json.loads(r["raw"])
    assert payload["sig_vel_e"] is None  # no MIDAS solution -> None, not fabricated


def test_midas_velocity_map_absent_station_is_none():
    """Station-ID join: a station not in the MIDAS file maps to None (NaN velocity)."""
    vmap = _midas_map()
    assert "CLV1" in vmap and vmap.get("NOPE") is None


def test_unknown_frame_raises():
    with pytest.raises(ValueError, match="IGS14"):
        ngl.parse(_raw(frame="ITRF97"))
    with pytest.raises(ValueError, match="IGS14"):
        ngl.fetch(LV_BBOX, frame="ITRF97")


def test_epoch_and_time_range_mutually_exclusive():
    with pytest.raises(ValueError, match="not both"):
        ngl.fetch(LV_BBOX, epoch=2020.0, time_range=(2019.0, 2021.0))


def test_ngl_registered_in_providers():
    assert PROVIDERS["ngl"] == (ngl.fetch, ngl.parse)


# ---------------------------------------------------------------------------
# station-index cache (7-day refresh, GROUNDCONTROL_CACHE_DIR)
# ---------------------------------------------------------------------------

def test_index_cache_fresh_needs_no_network(tmp_path, monkeypatch):
    monkeypatch.setenv("GROUNDCONTROL_CACHE_DIR", str(tmp_path))
    (tmp_path / "ngl_DataHoldings.txt").write_text(
        (DATA / "ngl_dataholdings_sample.txt").read_text())

    def _boom(*a, **k):
        raise AssertionError("network hit despite fresh cache")

    monkeypatch.setattr(ngl.requests, "get", _boom)
    assert len(ngl._load_index()) == 20


def test_index_cache_stale_refreshes(tmp_path, monkeypatch):
    import os
    monkeypatch.setenv("GROUNDCONTROL_CACHE_DIR", str(tmp_path))
    local = tmp_path / "ngl_DataHoldings.txt"
    local.write_text("stale")
    eight_days_ago = pd.Timestamp.now().timestamp() - 8 * 86400
    os.utime(local, (eight_days_ago, eight_days_ago))

    class _Resp:
        text = (DATA / "ngl_dataholdings_sample.txt").read_text()

        def raise_for_status(self):
            pass

    monkeypatch.setattr(ngl.requests, "get", lambda *a, **k: _Resp())
    assert len(ngl._load_index()) == 20
    assert "CLV1" in local.read_text()  # cache was rewritten


# ---------------------------------------------------------------------------
# land_horizontal: the D6 tt rule (offline — Helmert, no grids)
# ---------------------------------------------------------------------------

def test_is_dynamic_frame():
    assert is_dynamic_frame("EPSG:7912")       # ITRF2014
    assert is_dynamic_frame("EPSG:9989")       # ITRF2020
    assert not is_dynamic_frame("EPSG:6318")   # NAD83(2011) — plate-fixed
    assert not is_dynamic_frame("EPSG:4152")   # NAD83(HARN)


def _dynamic_gdf(epochs):
    n = len(epochs)
    return gpd.GeoDataFrame(
        {
            "horizontal_crs": pd.array(["EPSG:7912"] * n, dtype="string"),
            "coord_epoch": np.asarray(epochs, dtype="float64"),
            "transform_id": pd.array([pd.NA] * n, dtype="string"),
        },
        geometry=[Point(-115.1, 36.1)] * n,
        crs=None,
    )


def test_land_horizontal_tt_epoch_sensitivity():
    """TODO(D6) arbiter: identical ITRF2014 coordinates at coord_epoch 2015 vs
    2025 must land ~1.5-2.5 cm/yr apart in NAD83(2011) — an omitted tt would
    land them identically (silently evaluated at t_epoch=2010)."""
    out = land_horizontal(_dynamic_gdf([2015.0, 2025.0]), target="EPSG:6318")
    assert out.crs is not None and out.crs.to_epsg() == 6318
    dx = (out.geometry.x.iloc[1] - out.geometry.x.iloc[0]) * 111_320 * np.cos(np.radians(36.1))
    dy = (out.geometry.y.iloc[1] - out.geometry.y.iloc[0]) * 110_574
    d = float(np.hypot(dx, dy))
    assert 0.10 < d < 0.30, f"10-yr epoch sensitivity {d:.3f} m outside 1.0-3.0 cm/yr"
    assert out["transform_id"].str.endswith("tt=coord_epoch").all()
    # the landing moved the point off its ITRF position (~0.8-1.5 m NAD83 offset)
    assert abs(out.geometry.x.iloc[0] - (-115.1)) * 111_320 > 0.2


def test_land_horizontal_tt_matches_fixed_t_epoch_at_2010():
    """coord_epoch=2010.0 must reproduce the legacy no-tt result exactly
    (PROJ evaluates an omitted tt at the operation's t_epoch=2010)."""
    from groundcontrol.crs import get_transformer
    landed = land_horizontal(_dynamic_gdf([2010.0]), target="EPSG:6318")
    t = get_transformer("EPSG:7912", "EPSG:6318",
                        aoi_bounds_4326=(-115.2, 36.0, -115.0, 36.2))
    x_no_tt, y_no_tt = t.transform([-115.1], [36.1], errcheck=True)
    assert landed.geometry.x.iloc[0] == pytest.approx(x_no_tt[0], abs=1e-12)
    assert landed.geometry.y.iloc[0] == pytest.approx(y_no_tt[0], abs=1e-12)


def test_land_dynamic_frame_nan_coord_epoch_raises():
    with pytest.raises(ValueError, match="NaN coord_epoch"):
        land_horizontal(_dynamic_gdf([np.nan]), target="EPSG:6318")


def test_land_dynamic_frame_missing_coord_epoch_raises():
    gdf = _dynamic_gdf([2020.0]).drop(columns=["coord_epoch"])
    with pytest.raises(ValueError, match="coord_epoch"):
        land_horizontal(gdf, target="EPSG:6318")


def test_parse_then_land_end_to_end_offline():
    """Provider -> dispatcher-style landing: NGL native EPSG:7912 rows land in
    EPSG:6318 with per-row tt (the Increment 1.5a integration path)."""
    from groundcontrol import crs
    out = schema.normalize(ngl.parse(_raw(epoch=2017.95)), source="ngl")
    landed = crs.land_horizontal(out, target="EPSG:6318")
    schema.validate(landed)
    assert landed.crs.to_epsg() == 6318
    # NAD83(2011) vs ITRF2014@2017.95 in Nevada: decimeter-to-meter offset
    shift_m = float(np.hypot(
        (landed.geometry.x - landed["native_x"]) * 111_320 * np.cos(np.radians(36.2)),
        (landed.geometry.y - landed["native_y"]) * 110_574,
    ).iloc[0])
    assert 0.5 < shift_m < 3.0
    assert landed["transform_id"].iloc[0].endswith("tt=coord_epoch")


# ---------------------------------------------------------------------------
# live integration (network)
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_fetch_las_vegas_live():
    raw = ngl.fetch(LV_BBOX, frame="IGS14", max_stations=1)
    assert raw["frame"] == "IGS14" and len(raw["stations"]) == 1
    out = schema.normalize(ngl.parse(raw), source="ngl")
    schema.validate(out, require_crs=False)
    assert len(out) == 1
    r = out.iloc[0]
    assert LV_BBOX[0] <= r.geometry.x <= LV_BBOX[2]
    assert LV_BBOX[1] <= r.geometry.y <= LV_BBOX[3]
    assert 400 < r["height"] < 1600  # Las Vegas valley ellipsoidal heights
    assert r["horizontal_crs"] == "EPSG:7912"
    assert r["coord_epoch"] > 2000
    assert "ant_m" in json.loads(r["raw"])
    # with_velocities=True (default): the station carries its own MIDAS velocity
    # when it has a MIDAS solution (finite & cm/yr-scale) — else honest NaN.
    v = r["vel_e"]
    assert np.isnan(v) or abs(v) < 0.1  # m/yr
    # and it lands (time-dependent Helmert with per-row tt)
    landed = land_horizontal(out, target="EPSG:6318")
    assert landed.crs.to_epsg() == 6318
