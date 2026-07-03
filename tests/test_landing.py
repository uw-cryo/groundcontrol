"""B7 per-datum landing tests: realization mapping, subset landing, NCAT golden.

The NCAT golden fixture (tests/data/ncat_golden.json) was recorded live 2026-07
from https://geodesy.noaa.gov/api/ncat/llh — NGS's own NADCON5 implementation,
independent of PROJ. Our landing must reproduce it within survey tolerance.
Landing tests that exercise real NADCON5 grids are @network (grids stream from
cdn.proj.org until the clipped-grid fixtures land — crs_implementation §8.10).
"""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from groundcontrol.crs import land_horizontal, ngs_datum_to_epsg

DATA = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# realization mapping (offline)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("datum,expected", [
    ("NAD 83(2011)", "EPSG:6318"),
    ("NAD83(2011)", "EPSG:6318"),
    ("NAD_83(2011)", "EPSG:6318"),
    ("NAD 83(1986)", "EPSG:4269"),
    ("NAD 83(1992)", "EPSG:4152"),   # state HPGN/HARN adjustment years
    ("NAD 83(1994)", "EPSG:4152"),
    ("NAD 83(HARN)", "EPSG:4152"),
    ("NAD 83(FBN)", "EPSG:8860"),
    ("NAD 83(NSRS2007)", "EPSG:4759"),
    ("NAD 83(CORS96)", "EPSG:6783"),
])
def test_ngs_datum_mapping(datum, expected):
    assert ngs_datum_to_epsg(datum) == expected


@pytest.mark.parametrize("bad", ["NAD 83(2035)", "ITRF2016", "WGS84", "", None, "garbage"])
def test_unknown_realization_raises_listing_supported(bad):
    with pytest.raises(ValueError, match="supported"):
        ngs_datum_to_epsg(bad)


# ---------------------------------------------------------------------------
# land_horizontal mechanics (offline, monkeypatched transformer)
# ---------------------------------------------------------------------------

def _mixed_gdf():
    return gpd.GeoDataFrame(
        {
            "horizontal_crs": pd.array(
                ["EPSG:6318", "EPSG:4152", "EPSG:4152", "EPSG:4269"], dtype="string"),
            "transform_id": pd.array([pd.NA] * 4, dtype="string"),
        },
        geometry=[Point(-111.70, 32.80), Point(-111.71, 32.81),
                  Point(-111.72, 32.82), Point(-111.73, 32.83)],
        crs=None,
    )


def test_land_horizontal_groups_and_uses_subset_bounds(monkeypatch):
    """B7a: one transformer per datum subset, built from the SUBSET's bounds."""
    import groundcontrol.crs as gc

    calls = []

    class _FakeT:
        description, accuracy = "fake", 0.1

        def transform(self, x, y, errcheck=False):
            return np.asarray(x) + 0.001, np.asarray(y)  # visible shift

    def fake_get_transformer(src, tgt, aoi_bounds_4326=None):
        calls.append((src, tgt, aoi_bounds_4326))
        return _FakeT()

    monkeypatch.setattr(gc, "get_transformer", fake_get_transformer)
    out = land_horizontal(_mixed_gdf(), target="EPSG:6318")

    # one call per non-target datum (4152, 4269) — never for the 6318 subset
    assert sorted(c[0] for c in calls) == ["EPSG:4152", "EPSG:4269"]
    harn_call = next(c for c in calls if c[0] == "EPSG:4152")
    minx, miny, maxx, maxy = harn_call[2]
    assert (minx, maxx) == (-111.72, -111.71) and (miny, maxy) == (32.81, 32.82)

    # transformed rows moved; identity rows didn't; provenance untouched
    assert out.geometry.x.iloc[1] == pytest.approx(-111.709, abs=1e-9)
    assert out.geometry.x.iloc[0] == pytest.approx(-111.70)
    assert out.crs is not None and out.crs.to_epsg() == 6318
    assert out["transform_id"].iloc[0] == "land:identity:EPSG:6318"
    assert out["transform_id"].iloc[1].startswith("land:EPSG:4152->EPSG:6318")
    assert (out["horizontal_crs"] == ["EPSG:6318", "EPSG:4152", "EPSG:4152", "EPSG:4269"]).all()


def test_land_horizontal_empty_frame():
    out = land_horizontal(_mixed_gdf().iloc[:0], target="EPSG:6318")
    assert len(out) == 0 and out.crs.to_epsg() == 6318


# ---------------------------------------------------------------------------
# NCAT golden comparison (network: NADCON5 grids stream from cdn.proj.org)
# ---------------------------------------------------------------------------

@pytest.mark.network
@pytest.mark.parametrize("in_datum,epsg", [("NAD83(HARN)", "EPSG:4152"),
                                           ("NAD83(1986)", "EPSG:4269")])
def test_landing_matches_ncat_golden(in_datum, epsg):
    """Our PROJ/NADCON5 landing must reproduce NGS NCAT within ~1 cm (plan §8 item 8)."""
    golden = json.loads((DATA / "ncat_golden.json").read_text())
    g = golden["results"][in_datum]
    pt = golden["point"]
    gdf = gpd.GeoDataFrame(
        {"horizontal_crs": pd.array([epsg], dtype="string"),
         "transform_id": pd.array([pd.NA], dtype="string")},
        geometry=[Point(pt["lon"], pt["lat"])], crs=None,
    )
    out = land_horizontal(gdf, target="EPSG:6318")
    dlat_m = abs(out.geometry.y.iloc[0] - float(g["destLat"])) * 110_574
    dlon_m = abs(out.geometry.x.iloc[0] - float(g["destLon"])) * 111_320 * 0.84
    assert dlat_m < 0.01, f"{in_datum} lat differs from NCAT by {dlat_m:.4f} m"
    assert dlon_m < 0.01, f"{in_datum} lon differs from NCAT by {dlon_m:.4f} m"
