"""Quick-look figures for fetched control points and DEM residuals.

The full report figures arrive with Increment 2 (ported from the upstream
``plot`` module); this covers the fetch-side quick look plus the assessment
residual map (:func:`plot_dh_map`) with an optional :func:`hillshade` basemap.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from groundcontrol.accuracy import med_nmad


def plot_control(gdf, title=None, out_fn=None):
    """Two-panel quick look: map colored by source; height histogram by source."""
    fig, (ax_map, ax_hist) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.4, 1]}
    )
    for source, grp in gdf.groupby("source"):
        ax_map.scatter(grp.geometry.x, grp.geometry.y, s=8, alpha=0.7,
                       label=f"{source} (n={len(grp)})")
        h = grp["height"].dropna()
        if len(h):
            ax_hist.hist(h, bins=40, alpha=0.6, label=source)
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.set_aspect("equal")
    ax_map.legend(fontsize=8)
    ax_map.set_title(f"control points ({gdf.crs.to_string() if gdf.crs else 'no CRS'})",
                     fontsize=9)
    ax_hist.set_xlabel("height (m, NAVD88)")
    ax_hist.set_ylabel("count")
    ax_hist.legend(fontsize=8)
    ax_hist.set_title("height distribution", fontsize=9)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if out_fn:
        fig.savefig(out_fn, dpi=150)
    return fig


def hillshade(z, dx: float = 1.0, dy: float = 1.0,
              azdeg: float = 315.0, altdeg: float = 45.0) -> np.ndarray:
    """Gradient hillshade of a north-up elevation array, in [0, 1].

    Plain numpy (Horn/ESRI formulation) — no extra dependencies. Assumes the
    standard raster orientation (row 0 = north edge); ``dx``/``dy`` are pixel
    sizes in the elevation units. NaNs propagate (transparent under most
    colormaps), so mask nodata before calling.
    """
    z = np.asarray(z, dtype="float64")
    g_south, g_east = np.gradient(z, dy, dx)  # d z / d row (southward), d z / d col
    slope = np.arctan(np.hypot(g_east, g_south))
    # ESRI aspect convention (their dz/dy is the southward derivative) — verified
    # numerically against matplotlib.colors.LightSource for all orientations
    aspect = np.arctan2(g_south, -g_east)
    zenith = np.radians(90.0 - altdeg)
    az_math = np.radians((360.0 - azdeg + 90.0) % 360.0)
    hs = (np.cos(zenith) * np.cos(slope)
          + np.sin(zenith) * np.sin(slope) * np.cos(az_math - aspect))
    hs = np.clip(hs, 0.0, 1.0)
    hs[~np.isfinite(z)] = np.nan  # keep nodata holes transparent (gradient fills them)
    return hs


def plot_dh_map(gdf, dh_col: str, hs=None, hs_extent=None, ax=None,
                cmap: str = "RdYlBu", clim=None, s: float = 8, title=None):
    """Map of signed point residuals, optionally over a hillshade basemap.

    ``hs`` is a [0, 1] array (see :func:`hillshade`) drawn in grayscale with
    ``hs_extent`` = (left, right, bottom, top) in the points' CRS. ``clim``
    defaults to symmetric limits about ZERO at 3*NMAD of the finite residuals
    (diverging-map convention: an unbiased DEM reads as neutral color).

    Returns the scatter mappable (``sc.axes`` for the axes;
    ``fig.colorbar(sc)`` for the colorbar).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    if hs is not None:
        ax.imshow(hs, cmap="gray", vmin=0.0, vmax=1.0, extent=hs_extent,
                  interpolation="nearest", zorder=0)
    vals = np.asarray(gdf[dh_col], dtype="float64")
    if clim is None:
        _, nmad = med_nmad(vals)
        lim = 3.0 * nmad if np.isfinite(nmad) and nmad > 0 else \
            (float(np.nanmax(np.abs(vals))) if np.isfinite(vals).any() else 1.0)
        clim = (-lim, lim)
    sc = ax.scatter(gdf.geometry.x, gdf.geometry.y, c=vals, cmap=cmap,
                    vmin=clim[0], vmax=clim[1], s=s,
                    edgecolors="k", linewidths=0.2, zorder=2)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=10)
    return sc
