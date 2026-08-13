"""accuracy.py tests — hand-computed fixtures for the plan A5 primitives (B8 contract)."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from groundcontrol import accuracy
from groundcontrol.accuracy import med_nmad, resid_stats, robust_normalize

# ---------------------------------------------------------------------------
# med_nmad — pinned contract: 1-D Series/array in, (float, float) out (B8)
# ---------------------------------------------------------------------------

def test_med_nmad_hand_computed():
    # [1,2,3,4,100]: median 3; |a-3| = [2,1,0,1,97] -> MAD 1 -> NMAD 1.4826
    med, nmad = med_nmad([1.0, 2.0, 3.0, 4.0, 100.0])
    assert med == 3.0
    assert nmad == pytest.approx(1.4826)


def test_med_nmad_returns_float_tuple():
    out = med_nmad(pd.Series([1.0, 2.0, 3.0]))
    assert isinstance(out, tuple) and len(out) == 2
    assert isinstance(out[0], float) and isinstance(out[1], float)


def test_med_nmad_ignores_nonfinite():
    med, nmad = med_nmad([1.0, np.nan, 2.0, np.inf, 3.0, 4.0, 100.0])
    assert med == 3.0 and nmad == pytest.approx(1.4826)


def test_med_nmad_empty_and_all_nan():
    assert all(np.isnan(v) for v in med_nmad([]))
    assert all(np.isnan(v) for v in med_nmad([np.nan, np.nan]))


def test_med_nmad_rejects_2d_input():
    """B8: the Series-vs-DataFrame ambiguity is resolved by rejecting non-1-D."""
    with pytest.raises(ValueError, match="1-D"):
        med_nmad(np.ones((3, 2)))
    with pytest.raises(ValueError, match="1-D"):
        med_nmad(pd.DataFrame({"a": [1.0], "b": [2.0]}))


# ---------------------------------------------------------------------------
# resid_stats — hand-computed, and single-source-of-truth for the constant
# ---------------------------------------------------------------------------

def test_resid_stats_hand_computed():
    # a = [1,2,3,4,100]: mean 22; std = sqrt(1522); rmse = sqrt(2006)
    st = resid_stats([1.0, 2.0, 3.0, 4.0, 100.0])
    assert st["n"] == 5
    assert st["median"] == 3.0
    assert st["mean"] == pytest.approx(22.0)
    assert st["nmad"] == pytest.approx(1.4826)
    assert st["std"] == pytest.approx(np.sqrt(1522.0))
    assert st["rmse"] == pytest.approx(np.sqrt(2006.0))


def test_resid_stats_symmetric_case():
    # [-1, 0, 1]: median 0, mean 0, MAD 1 -> NMAD 1.4826, std = rmse = sqrt(2/3)
    st = resid_stats([-1.0, 0.0, 1.0])
    assert st["median"] == 0.0 and st["mean"] == 0.0
    assert st["nmad"] == pytest.approx(1.4826)
    assert st["std"] == pytest.approx(np.sqrt(2.0 / 3.0))
    assert st["rmse"] == pytest.approx(np.sqrt(2.0 / 3.0))


def test_resid_stats_ignores_nan_and_empty():
    st = resid_stats([np.nan, 1.0, np.nan])
    assert st["n"] == 1 and st["median"] == 1.0
    st0 = resid_stats([])
    assert st0["n"] == 0
    assert all(np.isnan(st0[k]) for k in ("median", "mean", "nmad", "std", "rmse"))


def test_resid_stats_calls_med_nmad(monkeypatch):
    """B8: one source of truth for the 1.4826 constant — resid_stats must delegate."""
    sentinel = (123.0, 456.0)
    monkeypatch.setattr(accuracy, "med_nmad", lambda s: sentinel)
    st = accuracy.resid_stats([1.0, 2.0, 3.0])
    assert st["median"] == 123.0 and st["nmad"] == 456.0


# ---------------------------------------------------------------------------
# robust_normalize — the outlier filter
# ---------------------------------------------------------------------------

def _gdf(values):
    return gpd.GeoDataFrame(
        {"dh": values},
        geometry=gpd.points_from_xy(np.arange(len(values)), np.zeros(len(values))),
        crs="EPSG:4326",
    )


def test_robust_normalize_filters_outlier():
    # inliers ~0; 30 m blunder is way beyond 3*NMAD
    gdf = _gdf([0.1, -0.2, 0.05, -0.05, 0.15, 30.0])
    mask = robust_normalize(gdf, "dh", nmad_mult=3.0)
    assert mask.dtype == bool
    assert list(mask) == [True, True, True, True, True, False]


def test_robust_normalize_nan_rows_excluded():
    gdf = _gdf([0.1, np.nan, 0.05, 25.0])
    mask = robust_normalize(gdf, "dh")
    assert not mask.iloc[1] and not mask.iloc[3]
    assert mask.iloc[0] and mask.iloc[2]


def test_robust_normalize_preserves_index():
    gdf = _gdf([0.0, 0.1, 40.0]).set_index(pd.Index([10, 20, 30]))
    mask = robust_normalize(gdf, "dh")
    assert list(mask.index) == [10, 20, 30]
    assert bool(mask.loc[10]) and not bool(mask.loc[30])


def test_robust_normalize_uses_med_nmad(monkeypatch):
    """One source of truth (B8): the bounds come from med_nmad."""
    monkeypatch.setattr(accuracy, "med_nmad", lambda s: (0.0, 1.0))
    mask = accuracy.robust_normalize(_gdf([2.9, 3.1, -2.9, -3.1]), "dh", nmad_mult=3.0)
    assert list(mask) == [True, False, True, False]
