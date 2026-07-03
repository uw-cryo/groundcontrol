"""schema.py unit tests — empty()/normalize()/validate() + GeoParquet round-trip.

All offline. The per-source normalize() fixtures land with Increment 1 step 3.
"""

import json

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from groundcontrol import schema


def _sample_gdf(crs="EPSG:4326"):
    return gpd.GeoDataFrame(
        {
            "id": ["PT1", "PT2"],
            "height": [368.9, 412.1],
            "coord_epoch": [2010.0, 2015.5],
            "raw": [{"note": "a"}, {"note": "b"}],
        },
        geometry=[Point(-111.7, 32.8), Point(-111.6, 32.9)],
        crs=crs,
    )


def test_empty_has_full_column_set_and_zero_rows():
    e = schema.empty()
    assert len(e) == 0
    assert list(schema.COLUMNS.keys()) == [c for c in e.columns if c != "geometry"]
    for name, dtype in schema.COLUMNS.items():
        assert str(e[name].dtype) == dtype, name


def test_empty_concat_compatible_with_populated():
    populated = schema.normalize(_sample_gdf(), source="user")
    combined = pd.concat([schema.empty(crs="EPSG:4326"), populated])
    assert len(combined) == 2
    assert isinstance(combined, gpd.GeoDataFrame)


def test_normalize_fills_missing_columns_and_orders():
    out = schema.normalize(_sample_gdf(), source="user")
    schema.validate(out)
    assert out["source"].eq("user").all()
    assert out["vel_e"].isna().all()  # unfilled columns exist as NaN
    assert list(out.columns)[:-1] == list(schema.COLUMNS.keys())


def test_normalize_serializes_raw_dicts_to_json():
    out = schema.normalize(_sample_gdf(), source="user")
    assert json.loads(out["raw"].iloc[0]) == {"note": "a"}  # D5


def test_validate_rejects_missing_column():
    out = schema.normalize(_sample_gdf(), source="user").drop(columns=["coord_epoch"])
    with pytest.raises(schema.SchemaError, match="coord_epoch"):
        schema.validate(out)


def test_rejects_3d_geometry_at_normalize():
    """3D input is rejected at the schema boundary (fail-loud, per D1)."""
    g = _sample_gdf()
    g.geometry = gpd.GeoSeries([Point(-111.7, 32.8, 100.0), Point(-111.6, 32.9, 90.0)], crs=g.crs)
    with pytest.raises(schema.SchemaError, match="2D"):
        schema.normalize(g, source="user")


def test_validate_requires_crs():
    out = schema.normalize(_sample_gdf(crs=None), source="user")
    with pytest.raises(schema.SchemaError, match="CRS"):
        schema.validate(out, require_crs=True)


def test_geoparquet_roundtrip(tmp_path):
    """The freeze-gate round-trip: normalize() -> export -> read-back cleanly."""
    out = schema.normalize(_sample_gdf(), source="user")
    fn = tmp_path / "control.parquet"
    out.to_parquet(fn)
    back = gpd.read_parquet(fn)
    schema.validate(back)
    assert list(back.columns) == list(out.columns)
    assert json.loads(back["raw"].iloc[1]) == {"note": "b"}
    assert back.crs == out.crs
    pd.testing.assert_series_equal(back["coord_epoch"], out["coord_epoch"])
