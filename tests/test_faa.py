"""FAA NASR runway source: offline parse/fixture tests + live fetch.

Fixture ``faa_apt_sample.txt`` holds 13 real fixed-width records (4 APT +
9 RWY) captured from the live 2026-08-06 cycle: LAS (Harry Reid) and VGT
(North Las Vegas) with surveyed ends and displaced thresholds, NV53 (a
hospital heliport, FAA-EST IMAGERY provenance), and 5AZ3 (Pegasus Airpark
AZ, estimated-provenance GA field with displaced thresholds). ``fetch()``
is ``@network``; parsing is offline.
"""

import json
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from groundcontrol import schema
from groundcontrol.sources import faa

DATA = Path(__file__).parent / "data"
FIXTURE_CYCLE = "2026-08-06"
WORLD = (-180.0, -90.0, 180.0, 90.0)
LV_BBOX = (-115.35, 35.9, -115.0, 36.4)


def _raw(bounds=WORLD):
    # NOT .splitlines(): latin-1 records contain bytes like \x85 that
    # str.splitlines treats as Unicode line boundaries, corrupting records
    with open(DATA / "faa_apt_sample.txt", encoding="latin-1") as f:
        lines = f.readlines()
    return {"cycle": FIXTURE_CYCLE, "aoi_bounds_4326": bounds, "lines": lines}


def test_parse_counts_and_types():
    out = faa.parse(_raw())
    assert len(out) == 26
    assert (out["point_type"] == "runway_end").sum() == 16
    assert (out["point_type"] == "displaced_threshold").sum() == 9
    # NV53 H1 is a hospital helipad: pad point, not a runway end
    assert list(out.loc[out["point_type"] == "helipad", "id"]) == ["NV53_H1"]
    assert out.crs is not None and out.crs.to_epsg() == 6318
    # every point carries coordinates; heights all present in this sample
    assert out.geometry.notna().all()
    assert out["height"].notna().all()


def test_parse_values_las_01l():
    """Regression anchor: LAS runway 01L threshold (fixture cycle values).

    664.83 m = 2181.2 ft — consistent with the published Harry Reid field
    elevation (~2181 ft), an independent plausibility check on the
    fixed-width slices and the ft->m conversion.
    """
    out = faa.parse(_raw()).set_index("id")
    r = out.loc["LAS_01L"]
    assert r.geometry.y == pytest.approx(36.075327, abs=1e-6)
    assert r.geometry.x == pytest.approx(-115.170364, abs=1e-6)
    assert r["height"] == pytest.approx(664.830, abs=0.001)
    assert r["height_datum"] == "NAVD88"
    assert r["vertical_crs"] == "EPSG:5703"
    assert r["horizontal_crs"] == "EPSG:6318"


def test_provenance_classes_and_accuracy():
    out = faa.parse(_raw())
    cls = out["raw"].map(lambda s: json.loads(s)["pos_class"])
    srcs = out["raw"].map(lambda s: json.loads(s).get("pos_src", ""))
    surveyed = cls == "surveyed"
    # LAS/VGT are 3RD PARTY SURVEY; NV53 heliport and 5AZ3 are estimated
    assert set(out.loc[surveyed, "id"].str[:3]) == {"LAS", "VGT"}
    assert (~surveyed).sum() == 5  # NV53_H1 + four 5AZ3 points
    assert set(srcs[~surveyed]) == {"FAA-EST IMAGERY", "ADO"}
    # spec accuracy attaches to the surveyed class ONLY; estimated rows
    # honestly carry no accuracy (never a fabricated bound)
    assert np.allclose(out.loc[surveyed, "acc_h"], faa.ACC_H_SURVEYED)
    assert np.allclose(out.loc[surveyed, "acc_v"], faa.ACC_V_SURVEYED)
    assert out.loc[~surveyed, "acc_h"].isna().all()
    assert out.loc[~surveyed, "acc_v"].isna().all()


def test_measurement_datetime_from_pos_src_date():
    out = faa.parse(_raw()).set_index("id")
    # LAS surveyed 11/30/2024 (fixture cycle)
    assert out.loc["LAS_01L", "measurement_datetime"] == pd.Timestamp(
        "2024-11-30", tz="UTC")
    assert out.loc["LAS_01L", "measurement_epoch"] == pytest.approx(
        2024.913, abs=0.01)


def test_bbox_filter_and_empty():
    lv = faa.parse(_raw(LV_BBOX))
    assert set(lv["id"].str[:4]) == {"LAS_", "VGT_", "NV53"}
    assert len(lv) == 22  # 26 minus the four 5AZ3 (Arizona) points
    empty = faa.parse(_raw((0.0, 0.0, 1.0, 1.0)))
    assert len(empty) == 0
    assert empty.crs is not None
    assert "id" in empty.columns and "height" in empty.columns


def test_schema_conformance():
    out = faa.parse(_raw())
    norm = schema.normalize(out, source="faa")
    schema.validate(norm)
    assert (norm["source"] == "faa").all()
    # transformability contract: resolvable horizontal CRS, uniform
    # non-null vertical CRS, non-null coord_epoch (plate-fixed reading)
    assert norm["horizontal_crs"].notna().all()
    assert norm["vertical_crs"].nunique() == 1
    assert norm["coord_epoch"].notna().all()


def test_pos_class_vocabulary():
    assert faa.pos_class("3RD PARTY SURVEY") == "surveyed"
    assert faa.pos_class("MILITARY") == "surveyed"
    assert faa.pos_class("NGS") == "surveyed"
    assert faa.pos_class("FAA-EST IMAGERY") == "estimated"
    assert faa.pos_class("OWNER") == "estimated"
    assert faa.pos_class("") == "estimated"
    assert faa.pos_class(None) == "estimated"


def test_current_cycle_cadence():
    assert faa.current_cycle(date(2022, 12, 1)) == "2022-12-01"
    assert faa.current_cycle(date(2022, 12, 28)) == "2022-12-01"
    assert faa.current_cycle(date(2022, 12, 29)) == "2022-12-29"
    # observed live cycles reproduce from the anchor (module docstring)
    assert faa.current_cycle(date(2025, 7, 15)) == "2025-07-10"
    assert faa.current_cycle(date(2026, 8, 13)) == "2026-08-06"


def test_fetch_serves_from_cache(tmp_path, monkeypatch):
    """A cached cycle zip is served with ZERO network calls."""
    monkeypatch.setenv("GROUNDCONTROL_CACHE_DIR", str(tmp_path))
    zip_fn = tmp_path / f"faa_APT_{FIXTURE_CYCLE}.zip"
    with zipfile.ZipFile(zip_fn, "w") as z:
        z.writestr("APT.txt",
                   (DATA / "faa_apt_sample.txt").read_text(encoding="latin-1"))

    def _boom(*a, **k):  # any HTTP touch is a failure
        raise AssertionError("network hit despite warm cache")

    monkeypatch.setattr(faa.requests, "get", _boom)
    raw = faa.fetch(LV_BBOX, cycle=FIXTURE_CYCLE)
    assert raw["cycle"] == FIXTURE_CYCLE
    out = faa.parse(raw)
    assert len(out) == 22


@pytest.mark.network
def test_fetch_live_las_vegas():
    raw = faa.fetch(LV_BBOX)
    out = faa.parse(raw)
    # the LV valley holds LAS + VGT + heliports: comfortably > 20 points
    assert len(out) > 20
    norm = schema.normalize(out, source="faa")
    schema.validate(norm)


def test_true_alignment_in_raw():
    """E46 runway-end true alignment feeds the oriented map/gallery
    chevrons; reciprocal ends differ by 180 and helipads carry none."""
    out = faa.parse(_raw()).set_index("id")
    az = {i: json.loads(out.loc[i, "raw"]).get("true_az")
          for i in ("LAS_01L", "LAS_19R", "LAS_01L_DT", "NV53_H1")}
    assert az["LAS_01L"] == "25.0" and az["LAS_19R"] == "205.0"
    assert az["LAS_01L_DT"] == "25.0"   # DT rides its end's alignment
    assert az["NV53_H1"] is None        # helipads have no runway azimuth
