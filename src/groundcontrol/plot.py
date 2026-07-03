"""Quick-look figures for fetched control points.

The full residual-map/report figures arrive with Increment 2 (ported from the
upstream ``plot`` module); this is the fetch-side quick look.
"""

from __future__ import annotations

import matplotlib.pyplot as plt


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
