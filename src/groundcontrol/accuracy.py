"""Robust accuracy/residual statistics for DEM assessment.

Port of the accuracy primitives in docs/plan.md Appendix A5 with the B8 fix:
:func:`med_nmad` has a pinned contract — 1-D Series/array in, ``(float, float)``
out — and is the single source of truth for the 1.4826 NMAD constant
(:func:`resid_stats` and :func:`robust_normalize` both call it rather than
duplicating the formula). Conventions (NMAD vs std, raw vs filtered reporting)
are documented in docs/accuracy_conventions.md.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#: NMAD scale factor: MAD -> sigma-equivalent for a normal distribution.
NMAD_CONSTANT = 1.4826


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


def robust_normalize(gdf, col: str, nmad_mult: float = 3.0):
    """Boolean mask of rows within ± ``nmad_mult``·NMAD of the median of ``col``.

    Returns a boolean Series aligned to ``gdf.index`` (NaN rows are ``False``).
    Use it to filter blunders before computing standard statistics — report
    both raw and filtered results (docs/accuracy_conventions.md).
    """
    med, nmad = med_nmad(gdf[col])
    return (gdf[col] > med - nmad_mult * nmad) & (gdf[col] < med + nmad_mult * nmad)


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
