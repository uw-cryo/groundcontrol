"""io.py + CLI tests (offline; CLI uses monkeypatched providers)."""

import json

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from groundcontrol import io, schema


def _gdf():
    g = gpd.GeoDataFrame(
        {"id": ["A", "B"], "height": [10.0, 20.0],
         "transform_id": ["land:identity:EPSG:6318", "land:EPSG:4152->EPSG:6318|acc=0.15m"],
         "raw": [{"k": 1}, {"k": 2}]},
        geometry=[Point(-111.7, 32.8), Point(-111.6, 32.9)], crs="EPSG:6318")
    return schema.normalize(g, source="user")


def test_write_parquet_embeds_provenance(tmp_path):
    out = tmp_path / "control.parquet"
    status = {"user": {"n_rows": 2, "error": None}}
    io.write(_gdf(), out, status=status, command="test")
    # sidecar
    sidecar = json.loads((tmp_path / "control.parquet.provenance.json").read_text())
    assert sidecar["schema"] == io.PROVENANCE_SCHEMA
    assert sidecar["n_points"] == 2
    assert sidecar["dispatcher_status"] == status
    assert sidecar["environment"]["pyproj"]
    assert "land:EPSG:4152->EPSG:6318|acc=0.15m" in sidecar["transforms"]
    # embedded copy survives the pyarrow rewrite; file still reads as GeoParquet
    embedded = io.read_provenance(out)
    assert embedded == sidecar
    back = gpd.read_parquet(out)
    schema.validate(back)
    assert len(back) == 2 and back.crs.to_epsg() == 6318


def test_write_csv_has_header_comment_and_xy(tmp_path):
    out = tmp_path / "control.csv"
    io.write(_gdf(), out)
    lines = out.read_text().splitlines()
    assert lines[0].startswith("# provenance:")
    header = lines[1].split(",")
    assert "x" in header and "y" in header and "coord_epoch" in header
    assert (tmp_path / "control.csv.provenance.json").exists()


def test_parquet_crs_promoted_to_compound_when_vertical_uniform(tmp_path):
    """QGIS-facing: file CRS says what the heights are (NAD83(2011)+NAVD88)."""
    g = _gdf()
    g["vertical_crs"] = pd.array(["EPSG:5703", "EPSG:5703"], dtype="string")
    out = tmp_path / "c.parquet"
    io.write(g, out)
    back = gpd.read_parquet(out)
    assert back.crs.is_compound
    assert "NAVD88" in back.crs.name


def test_parquet_crs_stays_2d_when_vertical_mixed(tmp_path):
    g = _gdf()
    g["vertical_crs"] = pd.array(["EPSG:5703", "EPSG:7968"], dtype="string")
    out = tmp_path / "c.parquet"
    io.write(g, out)
    back = gpd.read_parquet(out)
    assert not back.crs.is_compound and back.crs.to_epsg() == 6318


def test_parquet_crs_ignores_heightless_rows_for_promotion(tmp_path):
    """Rows without a height can't veto the compound claim (their vertical_crs is vacuous)."""
    import numpy as np
    g = _gdf()
    g["vertical_crs"] = pd.array(["EPSG:5703", pd.NA], dtype="string")
    g.loc[g.index[1], "height"] = np.nan
    out = tmp_path / "c.parquet"
    io.write(g, out)
    assert gpd.read_parquet(out).crs.is_compound


def test_write_rejects_unknown_format(tmp_path):
    with pytest.raises(ValueError, match="unsupported export format"):
        io.write(_gdf(), tmp_path / "control.xlsx")


def test_cli_fetch_end_to_end(tmp_path, monkeypatch):
    """Milestone command, offline: fetch (mocked) -> land -> export -> read back."""
    import groundcontrol.sources as srcs
    from groundcontrol.cli import fetch_control_main
    from groundcontrol.sources import ngs
    from pathlib import Path

    records = json.loads((Path(__file__).parent / "data" / "ngs_nde_sample.json").read_text())
    monkeypatch.setitem(srcs.PROVIDERS, "ngs", (lambda b: records, ngs.parse_nde))
    # avoid network for the landing of pre-2011 rows: keep only 2011 rows? No —
    # land via identity by mapping every row's datum to the target frame.
    import groundcontrol.crs as gc

    class _Ident:
        description, accuracy = "identity (test)", 0.0

        def transform(self, x, y, errcheck=False):
            return x, y

    monkeypatch.setattr(gc, "get_transformer", lambda *a, **k: _Ident())

    out = tmp_path / "control.parquet"
    rc = fetch_control_main(["--aoi=-112,32.6,-111.5,33.0",
                             "--sources", "ngs", "--out", str(out)])
    assert rc == 0
    back = gpd.read_parquet(out)
    schema.validate(back)
    assert len(back) == len(records)
    assert io.read_provenance(out)["dispatcher_status"]["ngs"]["n_rows"] == len(records)


def test_cli_total_failure_exits_nonzero(tmp_path, monkeypatch):
    import groundcontrol.sources as srcs
    from groundcontrol.cli import fetch_control_main

    def boom(b):
        raise RuntimeError("down")

    monkeypatch.setitem(srcs.PROVIDERS, "ngs", (boom, lambda r: r))
    rc = fetch_control_main(["--aoi=-112,32.6,-111.5,33.0",
                             "--sources", "ngs", "--out", str(tmp_path / "c.parquet")])
    assert rc == 1
