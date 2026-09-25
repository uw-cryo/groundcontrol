"""Robust accuracy/residual statistics for DEM assessment.

Port of the accuracy primitives in docs/plan.md Appendix A5 with the B8 fix:
:func:`med_nmad` has a pinned contract — 1-D Series/array in, ``(float, float)``
out — and is the single source of truth for the 1.4826 NMAD constant
(:func:`resid_stats` calls it directly, :func:`robust_normalize` via
:func:`robust_mask`, the single outlier gate). Conventions (NMAD vs std, raw vs filtered reporting)
are documented in docs/accuracy_conventions.md.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#: NMAD scale factor: MAD -> sigma-equivalent for a normal distribution.
NMAD_CONSTANT = 1.4826

#: Normal-theory accuracy factors. Sources: NSSDA (FGDC-STD-007.3-1998,
#: Appendix 3-A) and ASPRS Positional Accuracy Standards Ed. 1 (2014) Annex D
#: for the 95% conversions and the 0.6 axis-ratio rule; NGA circular/linear
#: error (Greenwalt & Schultz 1962) for the 90% factors. ASPRS Ed. 2 (2023)
#: eliminated the 95% confidence level as an accuracy measure (Change #2) and
#: reports RMSE_H, RMSE_V and RMSE_3D = sqrt(RMSE_H^2 + RMSE_V^2) directly;
#: the CE/LE conversions here are the legacy NSSDA/NGA statements a reader
#: may still ask for, never the headline accuracy. Horizontal, for a
#: circular bivariate normal: CE90 = 2.1460*sigma, CE95 = 2.4477*sigma with
#: sigma = RMSE_r/sqrt(2), i.e. CE90 = 1.5175*RMSE_r (NGA-rounded; exact
#: 1.51743) and CE95 = 1.7308*RMSE_r. Vertical: LE90 = 1.6449*RMSE_z,
#: LE95 = 1.9600*RMSE_z. All applied to RMSE (not bias-removed sigma) per
#: NSSDA, so bias inflates the reported CE/LE. The unit tests recompute
#: every constant from the Rayleigh/normal quantiles.
CE90_PER_SIGMA = 2.1460
CE95_PER_SIGMA = 2.4477
CE90_PER_RMSE_R = 1.5175
CE95_PER_RMSE_R = 1.7308
LE90_PER_RMSE_Z = 1.6449
LE95_PER_RMSE_Z = 1.9600
#: NSSDA App. 3-A: for unequal per-axis RMSE the approximation
#: CE = k_sigma * 0.5 * (RMSE_x + RMSE_y) holds only when
#: min(RMSE_x, RMSE_y) / max >= 0.6; below that NSSDA gives no closed form
#: (it defers to elliptical-error methods), so the formula CE is reported as
#: NaN rather than a number biased low (fail-loud; see error_report_3d).
NSSDA_RMSE_RATIO_MIN = 0.6
#: Minimum checkpoint count for a fully compliant product accuracy
#: assessment, ASPRS Ed. 2 (2023) Change #5 (raised from Ed. 1's 20; projects
#: over 1000 km^2 need more, up to 120; very small projects report per
#: Ed. 2 section 7.15).
ASPRS_MIN_CHECKPOINTS = 30


def med_nmad(series, s: float = NMAD_CONSTANT) -> tuple[float, float]:
    """Median and normalized MAD (robust spread) of a 1-D Series/array.

    Contract (plan B8): input must be 1-D (pandas Series, numpy array, or
    array-like); returns ``(median, nmad)`` as floats. Non-finite values are
    ignored; an all-NaN/empty input returns ``(nan, nan)``.
    """
    a = np.asarray(series, dtype="float64")
    if a.ndim != 1:
        raise ValueError(
            f"med_nmad expects a 1-D Series/array (plan B8 contract), got ndim={a.ndim}"
        )
    a = a[np.isfinite(a)]
    if a.size == 0:
        return (float("nan"), float("nan"))
    med = float(np.median(a))
    return med, float(s * np.median(np.abs(a - med)))


def _check_nmad_mult(nmad_mult) -> float:
    """Validate an outlier-gate multiplier: > 0 (``+inf`` allowed), else ValueError.

    The gate keeps |x - median| <= nmad_mult*NMAD. ``nmad_mult=np.inf`` is the
    documented way to switch the gate OFF and get the ungated ("raw") report
    ASPRS Ed. 2 wants beside the gated one (outliers are investigated, not
    silently dropped) — every finite value passes an infinite gate. A
    non-positive or NaN multiplier is a caller bug (NaN comparisons keep
    nothing), so fail loud. Note that any nmad_mult below 1/NMAD_CONSTANT
    (~0.674) can still legitimately reject every value — the MAD only
    guarantees that at least half survive a gate of one MAD — and the gated
    reports return the NaN block with ``n_used=0`` in that case.
    """
    m = float(nmad_mult)
    if np.isnan(m) or m <= 0:
        raise ValueError(f"nmad_mult must be > 0 (np.inf disables the gate), got {nmad_mult!r}")
    return m


def robust_mask(values, nmad_mult: float = 3.0) -> np.ndarray:
    """THE outlier gate: boolean array, True where ``values`` is finite and
    |x - median| <= ``nmad_mult``*NMAD (median/NMAD from the finite values).

    One implementation for every gated statistic in this module (and the
    figure helpers): ``robust_normalize``, ``error_report``, ``ce90`` and
    ``error_report_3d`` all call it, so membership at the boundary is
    identical everywhere. No gate is applied (every finite value is True)
    when NMAD == 0 — >=50% identical values, routine for quantized heights,
    where any gate floor would keep only the majority value and report
    fake-perfect stats — or when ``nmad_mult`` is ``np.inf``, the documented
    ungated ("raw") spelling. Non-finite values are always False.

    Raises ``ValueError`` for non-1-D input (including scalars) and for a
    NaN or non-positive ``nmad_mult``.
    """
    nmad_mult = _check_nmad_mult(nmad_mult)
    a = np.asarray(values, dtype="float64")
    if a.ndim != 1:
        raise ValueError(f"robust_mask expects a 1-D array, got ndim={a.ndim}")
    fin = np.isfinite(a)
    keep = np.zeros(a.size, dtype=bool)
    if not fin.any():
        return keep
    med, nmad = med_nmad(a[fin])
    if nmad > 0 and np.isfinite(nmad_mult):
        keep[fin] = np.abs(a[fin] - med) <= nmad_mult * nmad
    else:
        keep[fin] = True
    return keep


def robust_normalize(gdf, col: str, nmad_mult: float = 3.0):
    """Boolean mask of rows within ± ``nmad_mult``·NMAD of the median of ``col``.

    Returns a boolean Series aligned to ``gdf.index`` (NaN rows are ``False``).
    Thin wrapper over :func:`robust_mask` (inclusive boundary; NMAD == 0 and
    ``np.inf`` mean no gate — before the consolidation this function used a
    strict boundary and returned all-False on NMAD == 0). Use it to filter
    blunders before computing
    standard statistics — report both raw and filtered results
    (docs/accuracy_conventions.md).
    """
    import pandas as pd  # gdf is a (Geo)DataFrame, so pandas is already loaded
    return pd.Series(robust_mask(gdf[col].to_numpy(), nmad_mult), index=gdf.index)


def resid_stats(series) -> dict:
    """Robust + standard residual stats: n, median, mean, nmad, std, rmse.

    Non-finite values are ignored; ``std`` is the population standard
    deviation (ddof=0). Empty/all-NaN input returns ``n=0`` with NaN stats.
    """
    a = np.asarray(series, dtype="float64")
    a = a[np.isfinite(a)]
    if a.size == 0:
        return dict(n=0, median=np.nan, mean=np.nan, nmad=np.nan, std=np.nan, rmse=np.nan)
    med, nmad = med_nmad(a)  # single source of truth for the NMAD constant (B8)
    return dict(
        n=int(a.size),
        median=med,
        mean=float(a.mean()),
        nmad=nmad,
        std=float(a.std()),
        rmse=float(np.sqrt((a**2).mean())),
    )


def error_report(series, nmad_mult: float = 3.0) -> dict:
    """Cal/val-standard error report: robust AND parametric stats, dual-track.

    The robust pair (median/NMAD) is computed on ALL finite residuals; the
    parametric set the accuracy community expects (ASPRS Positional Accuracy
    Standards Ed. 2, 2023: RMSE is the accuracy measure, mean error reported
    separately; USGS Lidar Base Specification 2024; NGA-style LE90/LE95 as
    empirical |error| percentiles) is computed AFTER removing outliers beyond
    ``nmad_mult``*NMAD of the median — report both, plus how many were
    removed, so a reader can reconstruct either convention.

    Returns: n, median, nmad (all finite values); n_used, n_outliers,
    mean, std (1-sigma, ddof=1), rmse, le90, le95 (filtered values).
    ``nmad_mult=np.inf`` disables the gate (``n_used == n``) for the ungated
    "raw" row that is reported beside the gated one.
    """
    nmad_mult = _check_nmad_mult(nmad_mult)
    a = np.asarray(series, dtype="float64")
    a = a[np.isfinite(a)]
    if a.size == 0:
        return dict(n=0, median=np.nan, nmad=np.nan, n_used=0, n_outliers=0,
                    mean=np.nan, std=np.nan, rmse=np.nan, le90=np.nan,
                    le95=np.nan)
    med, nmad = med_nmad(a)
    f = a[robust_mask(a, nmad_mult)]  # NMAD == 0 / inf -> no gate, see robust_mask
    if f.size == 0:
        # only reachable for nmad_mult < 1/NMAD_CONSTANT (see _check_nmad_mult);
        # the robust pair still describes the input, the parametric set does not exist
        logger.warning("error_report: the %g*NMAD gate rejected all %d residuals; "
                       "parametric stats are NaN", nmad_mult, a.size)
        return dict(n=int(a.size), median=med, nmad=nmad, n_used=0,
                    n_outliers=int(a.size), mean=np.nan, std=np.nan, rmse=np.nan,
                    le90=np.nan, le95=np.nan)
    return dict(
        n=int(a.size), median=med, nmad=nmad,
        n_used=int(f.size), n_outliers=int(a.size - f.size),
        mean=float(f.mean()),
        std=float(f.std(ddof=1)) if f.size > 1 else float("nan"),
        rmse=float(np.sqrt((f ** 2).mean())),
        le90=float(np.percentile(np.abs(f), 90)),
        le95=float(np.percentile(np.abs(f), 95)),
    )


def ce90(dx, dy, nmad_mult: float = 3.0) -> float:
    """Empirical CE90 (m): 90th percentile of horizontal radial error, after
    a ``nmad_mult``*NMAD-per-axis outlier gate (NGA-style circular error)."""
    nmad_mult = _check_nmad_mult(nmad_mult)
    dx = np.asarray(dx, dtype="float64")
    dy = np.asarray(dy, dtype="float64")
    fin = np.isfinite(dx) & np.isfinite(dy)
    dx, dy = dx[fin], dy[fin]
    if not dx.size:
        return float("nan")
    keep = robust_mask(dx, nmad_mult) & robust_mask(dy, nmad_mult)
    r = np.hypot(dx[keep], dy[keep])
    return float(np.percentile(r, 90)) if r.size else float("nan")


def error_report_3d(de, dn, du, nmad_mult: float = 3.0) -> dict:
    """Three-axis accuracy report from per-point error vectors (product minus
    reference, meters, in a local E/N/U frame).

    Per axis (``"e"``, ``"n"``, ``"u"``): :func:`error_report` on that axis
    alone (dual-track, its own outlier gate). ``"combined"``: the horizontal,
    vertical and 3-D summary an NSSDA/NGA-style accuracy statement needs,
    computed on ONE point set — rows finite in all three axes that pass all
    three per-axis gates (each gate computed exactly as the per-axis report
    computes it, on that axis's own finite values) — so every combined number
    describes the same points. Because of the joint gate, ``combined``
    horizontal/vertical values can differ from the per-axis ones (e.g.
    ``u.le90`` vs ``combined.le90_empirical``) whenever another axis rejected
    a row; ``n_used`` says how many rows each block used.

    - ``n`` (rows finite in all three axes), ``n_used``, ``n_outliers``;
    - ``bias_2d`` = hypot(mean_e, mean_n); ``bias_3d`` likewise with mean_u;
    - ``rmse_r`` = sqrt(RMSE_e^2 + RMSE_n^2) (NSSDA RMSE_r = ASPRS Ed. 2 RMSE_H);
      ``rmse_3d`` adds RMSE_u^2 (= ASPRS Ed. 2 RMSE_3D, with RMSE_u = RMSE_V);
    - ``ce90_empirical``/``ce95_empirical``: percentiles of the radial error
      hypot(de, dn) (the standalone :func:`ce90` gates on E and N only, so it
      can differ in the presence of U-only outliers);
    - ``ce90_formula``/``ce95_formula``: NSSDA App. 3-A normal-theory
      approximation ``k_sigma * 0.5 * (RMSE_e + RMSE_n)`` (k_sigma = 2.1460 /
      2.4477; equal axes reduce it to ``1.5175 / 1.7308 * RMSE_r``), valid only
      when min(RMSE_e, RMSE_n)/max >= 0.6 (``ce_form="nssda"``); below that
      NSSDA has no closed form and the values are NaN
      (``ce_form="elliptical_unsupported"``) — use the empirical CE;
    - ``le90_empirical``/``le95_empirical``: percentiles of |du|;
      ``le90_formula``/``le95_formula`` = 1.6449 / 1.9600 * RMSE_u.

    ``nmad_mult=np.inf`` disables every gate: per-axis and combined statistics
    then use all rows finite in the respective axes.

    Logs a warning when ``combined.n_used`` — the rows left after the joint
    gate, the sample every combined statistic is computed on — is below
    ``ASPRS_MIN_CHECKPOINTS`` (30, ASPRS Ed. 2), including zero. ``n`` (rows
    finite in all three axes, before the gate) is deliberately not the
    threshold count. Raises
    ``ValueError`` on mismatched lengths or non-1-D input. Empty/all-NaN
    input returns ``n_used=0`` with NaN statistics and ``ce_form=None``.
    """
    nmad_mult = _check_nmad_mult(nmad_mult)
    de = np.asarray(de, dtype="float64")
    dn = np.asarray(dn, dtype="float64")
    du = np.asarray(du, dtype="float64")
    if not (de.shape == dn.shape == du.shape) or de.ndim != 1:
        raise ValueError(
            "error_report_3d expects three 1-D arrays of equal length, got shapes "
            f"{de.shape}, {dn.shape}, {du.shape}"
        )
    axes = {k: error_report(v, nmad_mult=nmad_mult) for k, v in (("e", de), ("n", dn), ("u", du))}

    fin = np.isfinite(de) & np.isfinite(dn) & np.isfinite(du)
    # each axis gated on its own finite values (== the per-axis report's gate)
    keep = robust_mask(de, nmad_mult) & robust_mask(dn, nmad_mult) & robust_mask(du, nmad_mult)
    e, n, u = de[keep], dn[keep], du[keep]
    nan = float("nan")
    combined = dict(
        n=int(fin.sum()), n_used=int(e.size), n_outliers=int(fin.sum() - e.size),
        bias_2d=nan, bias_3d=nan, rmse_r=nan, rmse_3d=nan,
        ce90_empirical=nan, ce95_empirical=nan, ce90_formula=nan, ce95_formula=nan,
        ce_form=None,
        le90_empirical=nan, le95_empirical=nan, le90_formula=nan, le95_formula=nan,
    )
    if e.size < ASPRS_MIN_CHECKPOINTS:
        # post-gate count (n_used), the sample the combined stats describe; placed
        # before the size guard so zero usable rows (empty input, all non-finite,
        # disjoint per-axis gates) is reported loudly, not just as NaNs
        logger.warning(
            "error_report_3d: %d joint checkpoints after the outlier gate (n_used), "
            "below the ASPRS Ed. 2 minimum of %d for a formal accuracy statement",
            e.size, ASPRS_MIN_CHECKPOINTS)
    if e.size:
        rmse_e, rmse_n, rmse_u = (float(np.sqrt((v ** 2).mean())) for v in (e, n, u))
        r = np.hypot(e, n)
        hi, lo = max(rmse_e, rmse_n), min(rmse_e, rmse_n)
        nssda_ok = hi == 0 or lo / hi >= NSSDA_RMSE_RATIO_MIN
        if nssda_ok:
            half_sum = 0.5 * (rmse_e + rmse_n)
            ce90_f, ce95_f = CE90_PER_SIGMA * half_sum, CE95_PER_SIGMA * half_sum
        else:
            ce90_f = ce95_f = nan
        combined.update(
            bias_2d=float(np.hypot(e.mean(), n.mean())),
            bias_3d=float(np.sqrt(e.mean() ** 2 + n.mean() ** 2 + u.mean() ** 2)),
            rmse_r=float(np.hypot(rmse_e, rmse_n)),
            rmse_3d=float(np.sqrt(rmse_e ** 2 + rmse_n ** 2 + rmse_u ** 2)),
            ce90_empirical=float(np.percentile(r, 90)),
            ce95_empirical=float(np.percentile(r, 95)),
            ce90_formula=float(ce90_f), ce95_formula=float(ce95_f),
            ce_form="nssda" if nssda_ok else "elliptical_unsupported",
            le90_empirical=float(np.percentile(np.abs(u), 90)),
            le95_empirical=float(np.percentile(np.abs(u), 95)),
            le90_formula=LE90_PER_RMSE_Z * rmse_u, le95_formula=LE95_PER_RMSE_Z * rmse_u,
        )
    axes["combined"] = combined
    return axes
