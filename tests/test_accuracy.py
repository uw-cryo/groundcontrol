"""accuracy.py tests — hand-computed fixtures for the plan A5 primitives (B8 contract)."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

import groundcontrol.accuracy as accuracy
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


# ---------------------------------------------------------------------------
# error_report_3d — three-axis / combined report (ASPRS Ed. 2 Annex C factors)
# ---------------------------------------------------------------------------

def test_accuracy_factors_match_normal_theory():
    """Independent recomputation: the module constants are the Rayleigh and
    normal quantiles they claim to be (ASPRS prints 1.5175 for 1.51743, so 1e-4)."""
    from scipy.stats import norm
    ce90_sigma = np.sqrt(-2 * np.log(0.1))   # Rayleigh 90th pct, circular normal
    ce95_sigma = np.sqrt(-2 * np.log(0.05))
    assert accuracy.CE90_PER_SIGMA == pytest.approx(ce90_sigma, abs=1e-4)
    assert accuracy.CE95_PER_SIGMA == pytest.approx(ce95_sigma, abs=1e-4)
    assert accuracy.CE90_PER_RMSE_R == pytest.approx(ce90_sigma / np.sqrt(2), abs=1e-4)
    assert accuracy.CE95_PER_RMSE_R == pytest.approx(ce95_sigma / np.sqrt(2), abs=1e-4)
    assert accuracy.LE90_PER_RMSE_Z == pytest.approx(norm.ppf(0.95), abs=1e-4)
    assert accuracy.LE95_PER_RMSE_Z == pytest.approx(norm.ppf(0.975), abs=1e-4)


def test_error_report_3d_hand_computed_ratio_below_nssda_limit():
    # zero-mean, symmetric; per-axis RMSE 1, 2, 3 -> horizontal ratio 0.5 < 0.6:
    # NSSDA has no closed form, so formula CE is NaN and the empirical CE stands
    de, dn, du = [1, -1, 1, -1], [2, -2, 2, -2], [3, -3, 3, -3]
    rep = accuracy.error_report_3d(de, dn, du)
    assert set(rep) == {"e", "n", "u", "combined"}
    assert rep["e"]["rmse"] == 1.0 and rep["n"]["rmse"] == 2.0 and rep["u"]["rmse"] == 3.0
    c = rep["combined"]
    assert c["n"] == 4 and c["n_used"] == 4 and c["n_outliers"] == 0
    assert c["bias_2d"] == 0.0 and c["bias_3d"] == 0.0
    assert c["rmse_r"] == pytest.approx(np.sqrt(5))
    assert c["rmse_3d"] == pytest.approx(np.sqrt(14))
    assert c["ce_form"] == "elliptical_unsupported"
    assert np.isnan(c["ce90_formula"]) and np.isnan(c["ce95_formula"])
    assert c["ce90_empirical"] == pytest.approx(np.sqrt(5))  # every radial error is sqrt(5)
    assert c["le90_formula"] == pytest.approx(accuracy.LE90_PER_RMSE_Z * 3)
    assert c["le95_formula"] == pytest.approx(accuracy.LE95_PER_RMSE_Z * 3)
    assert c["le90_empirical"] == 3.0


def test_error_report_3d_nssda_half_sum_unequal_axes():
    # RMSE_e 1.0, RMSE_n 0.8 (ratio 0.8, in range): CE = k * 0.5 * (1.0 + 0.8) = k * 0.9,
    # which differs from the circular k_r * rmse_r = k_r * 1.2806 -> pins the half-sum form
    de, dn, du = [1, -1] * 10, [0.8, -0.8] * 10, [0.0] * 20
    c = accuracy.error_report_3d(de, dn, du)["combined"]
    assert c["ce_form"] == "nssda"
    assert c["ce90_formula"] == pytest.approx(2.1460 * 0.9)
    assert c["ce95_formula"] == pytest.approx(2.4477 * 0.9)
    assert c["ce95_formula"] != pytest.approx(1.7308 * np.hypot(1.0, 0.8))


def test_error_report_3d_nssda_ratio_boundary():
    # ratio exactly 0.6 is in range (>=); 0.59 is not
    n = 20
    ok = accuracy.error_report_3d([1, -1] * 10, [0.6, -0.6] * 10, [0.0] * n)["combined"]
    assert ok["ce_form"] == "nssda" and ok["ce95_formula"] == pytest.approx(2.4477 * 0.8)
    bad = accuracy.error_report_3d([1, -1] * 10, [0.59, -0.59] * 10, [0.0] * n)["combined"]
    assert bad["ce_form"] == "elliptical_unsupported" and np.isnan(bad["ce95_formula"])


def test_error_report_3d_circular_form_recovers_ce90_sigma():
    # equal per-axis RMSE 1 -> half-sum 1 -> CE90 = 2.1460 (= 1.5175 * sqrt(2) * 1)
    v = [1, -1] * 10
    c = accuracy.error_report_3d(v, v, v)["combined"]
    assert c["ce_form"] == "nssda"
    assert c["ce90_formula"] == pytest.approx(2.1460, abs=1e-4)
    assert c["ce95_formula"] == pytest.approx(2.4477, abs=1e-4)
    assert c["ce95_formula"] == pytest.approx(accuracy.CE95_PER_RMSE_R * c["rmse_r"], abs=1e-4)


def test_error_report_3d_nssda_formula_vs_exact_bivariate_normal():
    """The module's formula CE95 on a large zero-mean bivariate-normal sample
    (ratio 0.8, inside the NSSDA range) is within 5% of the sample's exact 95%
    radius; at ratio 0.3 (outside) the module reports NaN, not a biased value.
    The 5% bound holds near 0.8 only (round-1 integration: -5.9% at 0.6)."""
    rng = np.random.default_rng(42)
    N = 400_000
    x, y = rng.normal(0, 1.0, N), rng.normal(0, 0.8, N)
    c = accuracy.error_report_3d(x, y, np.zeros(N))["combined"]
    assert c["ce_form"] == "nssda"
    assert abs(c["ce95_formula"] / c["ce95_empirical"] - 1) < 0.05
    y3 = rng.normal(0, 0.3, N)
    c3 = accuracy.error_report_3d(x, y3, np.zeros(N))["combined"]
    assert c3["ce_form"] == "elliptical_unsupported" and np.isnan(c3["ce95_formula"])
    assert np.isfinite(c3["ce95_empirical"])


def test_error_report_3d_warns_below_asprs_minimum(caplog):
    import logging
    assert accuracy.ASPRS_MIN_CHECKPOINTS == 30  # Ed. 2 (2023) Change #5
    with caplog.at_level(logging.WARNING, logger="groundcontrol.accuracy"):
        accuracy.error_report_3d([0.1, -0.1, 0.2], [0.1, 0.0, -0.1], [0.0, 0.1, 0.2])
    assert "below the ASPRS Ed. 2 minimum of 30" in caplog.text


def test_error_report_3d_warning_counts_post_gate_rows(caplog):
    """The threshold is n_used (after the joint gate), not n: 30 finite rows with
    one gated blunder is a 29-point sample and must warn (Copilot, PR #30)."""
    import logging
    base = np.array([0.1, -0.1] * 15)  # median 0, NMAD 0.148, gate +-0.445: all kept
    de, dn, du = base.copy(), base.copy(), base.copy()
    du[0] = 50.0
    with caplog.at_level(logging.WARNING, logger="groundcontrol.accuracy"):
        c = accuracy.error_report_3d(de, dn, du)["combined"]
    assert c["n"] == 30 and c["n_used"] == 29
    assert "29 joint checkpoints after the outlier gate" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="groundcontrol.accuracy"):
        c = accuracy.error_report_3d(base, base, base)["combined"]
    assert c["n"] == 30 and c["n_used"] == 30 and "below the ASPRS" not in caplog.text


def test_error_report_3d_bias_inflates_rmse_not_removed():
    # constant offset (1, 0, 2): bias_2d 1, bias_3d sqrt(5); rmse == |bias| per axis
    n = 10
    c = accuracy.error_report_3d([1.0] * n, [0.0] * n, [2.0] * n)["combined"]
    assert c["bias_2d"] == 1.0 and c["bias_3d"] == pytest.approx(np.sqrt(5))
    assert c["rmse_r"] == 1.0 and c["rmse_3d"] == pytest.approx(np.sqrt(5))
    assert c["le90_formula"] == pytest.approx(accuracy.LE90_PER_RMSE_Z * 2)


def test_error_report_3d_joint_gate_uses_one_point_set():
    """A U-only blunder drops out of the combined set but stays in the E report."""
    rng = np.random.default_rng(0)
    de, dn = rng.normal(0, 0.1, 50), rng.normal(0, 0.1, 50)
    du = rng.normal(0, 0.1, 50)
    du[7] = 50.0
    rep = accuracy.error_report_3d(de, dn, du)
    assert rep["u"]["n_outliers"] == 1 and rep["e"]["n_outliers"] == 0
    c = rep["combined"]
    assert c["n_used"] == 49 and c["n_outliers"] == 1
    assert c["le90_empirical"] < 1.0
    # combined horizontal stats exclude the blunder row even though E/N were fine there
    keep = np.ones(50, bool)
    keep[7] = False
    assert c["rmse_r"] == pytest.approx(
        np.hypot(np.sqrt((de[keep] ** 2).mean()), np.sqrt((dn[keep] ** 2).mean())))


def test_error_report_3d_gate_uses_each_axis_own_finite_set():
    """M3 (review round 1): the joint gate must equal the per-axis report's gate,
    computed on that axis's own finite values, not on the finite-in-all subset."""
    de = np.array([0, .1, -.1, .2, -.2, .35, .3, .3, .3])
    dn = np.zeros(9)
    du = np.zeros(9)
    du[[0, 1, 2]] = np.nan
    rep = accuracy.error_report_3d(de, dn, du)
    assert rep["e"]["n_outliers"] == 0
    c = rep["combined"]
    assert c["n"] == 6 and c["n_used"] == 6 and c["n_outliers"] == 0


def test_error_report_3d_nonfinite_rows_dropped_and_counted():
    de = [0.1, np.nan, 0.2, -0.1]
    dn = [0.0, 0.1, np.inf, 0.1]
    du = [0.2, 0.1, 0.0, -0.2]
    c = accuracy.error_report_3d(de, dn, du)["combined"]
    assert c["n"] == 2 and c["n_used"] == 2  # rows 0 and 3 only


def test_error_report_3d_empty_returns_nan(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="groundcontrol.accuracy"):
        c = accuracy.error_report_3d([], [], [])["combined"]
    assert "0 joint checkpoints after the outlier gate (n_used), below" in caplog.text
    assert c["n"] == 0 and c["n_used"] == 0 and c["ce_form"] is None
    assert np.isnan(c["rmse_r"]) and np.isnan(c["ce90_formula"]) and np.isnan(c["le90_empirical"])


def test_error_report_3d_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal length"):
        accuracy.error_report_3d([1.0, 2.0], [1.0], [1.0, 2.0])


# ---------------------------------------------------------------------------
# nmad_mult validation and the empty-gate path (geocalval report, 2026-09-24)
# ---------------------------------------------------------------------------

def test_error_report_empty_gate_returns_nan_block(caplog):
    """nmad_mult below 1/NMAD_CONSTANT can reject every residual: used to raise
    IndexError from np.percentile; now the NaN block with n_used=0 + a warning."""
    import logging
    with caplog.at_level(logging.WARNING, logger="groundcontrol.accuracy"):
        rep = accuracy.error_report(np.array([0.0, 10.0]), nmad_mult=0.1)
    assert rep["n"] == 2 and rep["n_used"] == 0 and rep["n_outliers"] == 2
    assert rep["median"] == 5.0 and rep["nmad"] == pytest.approx(5 * 1.4826)
    assert all(np.isnan(rep[k]) for k in ("mean", "std", "rmse", "le90", "le95"))
    assert "rejected all 2 residuals" in caplog.text


def test_error_report_3d_empty_gate_does_not_raise():
    rep = accuracy.error_report_3d([0.0, 10.0], [0.0, 10.0], [0.0, 10.0], nmad_mult=0.1)
    assert rep["e"]["n_used"] == 0 and rep["combined"]["n_used"] == 0
    assert np.isnan(rep["combined"]["rmse_r"])


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_gated_functions_reject_bad_nmad_mult(bad):
    v = [0.0, 1.0, 2.0]
    with pytest.raises(ValueError, match="nmad_mult"):
        accuracy.error_report(v, nmad_mult=bad)
    with pytest.raises(ValueError, match="nmad_mult"):
        accuracy.ce90(v, v, nmad_mult=bad)
    with pytest.raises(ValueError, match="nmad_mult"):
        accuracy.error_report_3d(v, v, v, nmad_mult=bad)
    with pytest.raises(ValueError, match="nmad_mult"):
        accuracy.robust_normalize(_gdf(v), "dh", nmad_mult=bad)


def test_small_but_valid_nmad_mult_still_accepted():
    # 0.5 < 1/1.4826 is legal (not rejected): [0..4] has median 2, NMAD 1.4826,
    # gate +-0.741 keeps only the median -> n_used 1, no exception
    rep = accuracy.error_report([0.0, 1.0, 2.0, 3.0, 4.0], nmad_mult=0.5)
    assert rep["n_used"] == 1 and rep["n_outliers"] == 4 and rep["mean"] == 2.0


def test_nmad_mult_inf_disables_gate_everywhere():
    """np.inf is the documented no-gate spelling (geocalval's ungated "raw" row);
    v0.2.1 refused it, which was a regression from v0.2.0."""
    v = np.array([0.0, 0.1, -0.1, 0.2, 50.0])  # 50 is a 3-NMAD outlier
    gated = accuracy.error_report(v)
    raw = accuracy.error_report(v, nmad_mult=np.inf)
    assert gated["n_outliers"] == 1 and raw["n_outliers"] == 0 and raw["n_used"] == 5
    assert raw["rmse"] == pytest.approx(np.sqrt((v ** 2).mean()))
    assert raw["median"] == gated["median"] and raw["nmad"] == gated["nmad"]
    assert accuracy.ce90(v, v, nmad_mult=np.inf) == pytest.approx(np.percentile(np.hypot(v, v), 90))
    assert accuracy.robust_normalize(_gdf(list(v)), "dh", nmad_mult=np.inf).all()
    c = accuracy.error_report_3d(v, v, v, nmad_mult=np.inf)["combined"]
    assert c["n_used"] == 5 and c["n_outliers"] == 0
    assert c["rmse_r"] == pytest.approx(np.hypot(raw["rmse"], raw["rmse"]))
