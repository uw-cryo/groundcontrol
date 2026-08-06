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
    (geopandas >= 1.0 raises). Seen live on an Atlanta DEM-footprint AOI.
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
    assert (out["point_type"] == "gnss").all()
    assert (out["horizontal_crs"] == "EPSG:6318").all()  # OPUS is NAD_83(2011)
    assert (out["coord_epoch"] == 2010.0).all()  # refFrame NAD_83(2011), epoch 2010.0000
    assert out["measurement_datetime"].notna().all()  # obsTimeStart


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
