"""Standard per-site control-point figure bundle.

Every site-level control fetch should ship the same default figure set
(requested by David, 2026-07-10, after the Casa Grande cal-range run):

1. ``{site}_control_map.png``       all points by type over shaded relief
2. ``{site}_monument_types.png``    NGS monuments faceted by posSource /
                                    vertSource / vertOrder (datasheet quality
                                    attributes retained in ``raw``)
3. ``{site}_midas_velocity_horiz.png``    MIDAS horizontal quiver (plot.py)
4. ``{site}_midas_velocity_vertical.png`` same, colored by vel_u (RED =
                                    subsidence) — the vertical-motion view

Conventions: NVA plots ABOVE VVA; AOI outline is light/transparent; scalebar
on map panels. Relief underlay (``hs_tif`` grayscale + optional ``dem_tif``
with ``cmap`` at ``dem_alpha``) is optional — figures degrade to plain maps.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: point_type -> (marker, color, size, zorder, label). NVA above VVA (zorder).
POINT_STYLE = {
    "monument": ("+", "#111111", 30, 4, "NGS monument"),
    "VVA": ("s", "#E69F00", 45, 5, "3DEP VVA"),
    "gnss": ("*", "#0033A0", 90, 6, "GNSS/OPUS"),
    "NVA": ("o", "#C00000", 55, 7, "3DEP NVA"),
}
_INK, _MUT = "#222222", "#777777"
_FACETS = ("posSource", "vertSource", "vertOrder")


def _raw_field(series, key):
    def get(r):
        if isinstance(r, str):
            try:
                r = json.loads(r)
            except Exception:
                return None
        v = (r or {}).get(key)
        v = (v or "").strip() if isinstance(v, str) else v
        return v if v else None
    return series.apply(get)


def _relief(ax, dem_tif, hs_tif, cmap, dem_alpha, fig):
    import rasterio
    ext = None
    if hs_tif is not None:
        with rasterio.open(hs_tif) as src:
            hs = src.read(1, masked=True).astype("f4").filled(np.nan)
            bb = src.bounds
        ext = [bb.left, bb.right, bb.bottom, bb.top]
        ax.imshow(hs, cmap="gray", vmin=1, vmax=255, extent=ext,
                  interpolation="antialiased", interpolation_stage="rgba")
    if dem_tif is not None and cmap is not None:
        with rasterio.open(dem_tif) as src:
            z = src.read(1, masked=True).filled(np.nan)
            bb = src.bounds
        ext = [bb.left, bb.right, bb.bottom, bb.top]
        im = ax.imshow(z, cmap=cmap, alpha=dem_alpha, extent=ext,
                       vmin=np.nanpercentile(z, 2),
                       vmax=np.nanpercentile(z, 98), interpolation="antialiased", interpolation_stage="rgba")
        if fig is not None:
            cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
            cb.set_label("Elevation (m, ellipsoid)", fontsize=9, color=_INK)
            cb.ax.tick_params(labelsize=8, colors=_MUT)
    return ext


def _finish_map(ax, aoi_gdf):
    from .plot import add_scalebar
    if aoi_gdf is not None:
        aoi_gdf.boundary.plot(ax=ax, color=_INK, lw=1.0, ls="--", alpha=0.45)
    ax.set_xticks([])
    ax.set_yticks([])
    add_scalebar(ax)


def standard_control_figures(control, aoi, outdir, site_name, *,
                             dem_tif=None, hs_tif=None, cmap=None,
                             dem_alpha=0.4, midas_frame="IGS14",
                             buffer_km=60.0, dpi=200):
    """Write the default control figure bundle for a site; returns paths."""
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = []
    aoi_gdf = gpd.read_file(aoi) if isinstance(aoi, (str, Path)) else aoi

    # figure CRS: the DEM's if given, else the AOI's UTM estimate
    import rasterio
    if dem_tif is not None:
        with rasterio.open(dem_tif) as src:
            fig_crs = src.crs
    else:
        fig_crs = aoi_gdf.estimate_utm_crs()
    ctl = control.to_crs(fig_crs)
    aoi_p = aoi_gdf.to_crs(fig_crs)

    # ---- 1. control map ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(10.5, 10))
    _relief(ax, dem_tif, hs_tif, cmap, dem_alpha, fig)
    handles = []
    for ptype, (mk, col, sz, zo, lab) in POINT_STYLE.items():
        sub = ctl[ctl.point_type == ptype]
        if not len(sub):
            continue
        ax.scatter(sub.geometry.x, sub.geometry.y, marker=mk, s=sz, c=col,
                   linewidths=1.1 if mk == "+" else 0.5,
                   edgecolors="white" if mk != "+" else col, zorder=zo)
        handles.append(Line2D([], [], marker=mk, ls="", color=col, ms=9,
                              label=f"{lab} (n={len(sub)})"))
    handles.append(Line2D([], [], ls="--", color=_INK, alpha=0.45,
                          label="AOI"))
    ax.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.92)
    _finish_map(ax, aoi_p)
    ax.set_title(f"{site_name} — control points ({len(ctl)} usable)",
                 fontsize=11, color=_INK)
    fig.tight_layout()
    fp = outdir / f"{site_name}_control_map.png"
    fig.savefig(fp, dpi=dpi)
    plt.close(fig)
    out.append(fp)

    # ---- 2. NGS monument-type facets ---------------------------------------
    mon = ctl[ctl.point_type == "monument"]
    if len(mon):
        fig, axes = plt.subplots(1, len(_FACETS), figsize=(5.6 * len(_FACETS), 6),
                                 sharex=True, sharey=True)
        cyc = ["#0033A0", "#C00000", "#005F20", "#8B008B", "#8B4E00",
               "#111111"]
        for ax, key in zip(np.atleast_1d(axes), _FACETS):
            _relief(ax, dem_tif, hs_tif, None, dem_alpha, None)
            vals = _raw_field(mon["raw"], key).fillna("(none)")
            top = vals.value_counts().index.tolist()[:5]
            vals = vals.where(vals.isin(top), "other")
            for i, v in enumerate(pd.unique(vals)):
                s = mon[vals == v]
                ax.scatter(s.geometry.x, s.geometry.y, s=16, marker="o",
                           c=cyc[i % 6], edgecolors="white", linewidths=0.4,
                           zorder=5, label=f"{v} ({len(s)})")
            _finish_map(ax, aoi_p)
            ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9)
            ax.set_title(f"NGS monuments by {key}", fontsize=10, color=_INK)
        fig.suptitle(f"{site_name} — NGS monument datasheet attributes "
                     f"(n={len(mon)})", fontsize=11.5, color=_INK)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fp = outdir / f"{site_name}_monument_types.png"
        fig.savefig(fp, dpi=dpi)
        plt.close(fig)
        out.append(fp)

    # ---- 3+4. MIDAS motion figures ------------------------------------------
    try:
        from .plot import plot_velocity_vectors
        from .sources.ngl import read_midas
        st = read_midas(midas_frame)
        b = aoi_gdf.to_crs(4326).total_bounds
        st = st[st.lon.between(b[0] - 3, b[2] + 3)
                & st.lat.between(b[1] - 3, b[3] + 3)]
        for cbv, tag in ((False, "horiz"), (True, "vertical")):
            fp = outdir / f"{site_name}_midas_velocity_{tag}.png"
            plot_velocity_vectors(
                st, aoi=aoi_gdf, buffer_km=buffer_km, out_fn=str(fp),
                color_by_vertical=cbv,
                title=f"{site_name} — MIDAS ({midas_frame}) "
                      f"{'vertical-colored ' if cbv else ''}velocity field")
            out.append(fp)
    except Exception as exc:  # network etc. — the map figures still ship
        logger.warning("MIDAS velocity figures skipped: %s", exc)

    logger.info("standard control figures: %s", [p.name for p in out])
    return out


def _nmad(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def validation_dz_figures(sampled, aoi, outdir, site_name, *, products=("DSM", "DTM"),
                          hs_tif=None, point_lim=0.25, vendor_lim=0.6, wide_lim=4.0,
                          ngs_nmad_gate=3.0, dpi=200):
    # hs_tif: a single path, or a {product: path} dict for PRODUCT-MATCHED
    # backgrounds (DTM diffs belong on the DTM hillshade — David, 2026-07-15).
    """Product-vs-control vertical-offset validation figures (standard bundle
    item 5; requested by David 2026-07-15 after the SF run).

    One figure per product: (a) map of control points over shaded relief
    colored by ``dh_<product>_before`` (product - control, RdBu_r, clipped at
    +/- ``point_lim``); (b) histograms for the survey-grade segments (vendor
    NVA/VVA, GNSS/OPUS; +/- ``vendor_lim``); (c) histogram for NGS monuments
    after a ``ngs_nmad_gate``-NMAD filter (+/- ``wide_lim``). Segment rules:
    NVA validates DSM and DTM; VVA validates DTM only; NGS/OPUS shown for
    both as datum-sanity context. Median/NMAD/n annotated per segment.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = []
    seg_defs = {  # label -> (mask fn, style key, in DSM, in DTM)
        "3DEP NVA": (lambda d: (d["source"] == "3dep") & (d["point_type"] == "NVA"),
                     "NVA", True, True),
        "3DEP VVA": (lambda d: (d["source"] == "3dep") & (d["point_type"] == "VVA"),
                     "VVA", False, True),
        "GNSS/OPUS": (lambda d: d["source"] == "opus", "gnss", True, True),
        "NGS monument": (lambda d: d["source"] == "ngs", "monument", True, True),
    }
    for prod in products:
        col = f"dh_{prod}_before"
        if col not in sampled.columns:
            logger.warning("validation_dz: no column %s, skipping %s", col, prod)
            continue
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.2),
                                 gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
        hs_prod = hs_tif.get(prod) if isinstance(hs_tif, dict) else hs_tif
        ext = _relief(axes[0], None, hs_prod, None, 0.0, None)
        use = sampled[np.isfinite(sampled[col])]
        sc = axes[0].scatter(use.geometry.x, use.geometry.y, c=use[col],
                             cmap="RdBu_r", vmin=-point_lim, vmax=point_lim,
                             s=34, edgecolors="#333333", linewidths=0.5, zorder=5)
        cb = fig.colorbar(sc, ax=axes[0], shrink=0.75, pad=0.02, extend="both")
        cb.set_label(f"dz = {prod} − control (m)", fontsize=9, color=_INK)
        cb.ax.tick_params(labelsize=8, colors=_MUT)
        _finish_map(axes[0], aoi)
        axes[0].set_title(f"{site_name} {prod} − control  (n={len(use)})",
                          fontsize=11, color=_INK)

        for ax, labels, lim in (
                (axes[1], [l for l, s in seg_defs.items()
                           if s[2 if prod == "DSM" else 3] and l != "NGS monument"],
                 vendor_lim),
                (axes[2], ["NGS monument"], wide_lim)):
            txt = []
            for lab in labels:
                maskfn, style, *_ = seg_defs[lab]
                v = use.loc[maskfn(use), col].to_numpy(float)
                v = v[np.isfinite(v)]
                if lab == "NGS monument" and len(v):
                    med0, nm0 = np.median(v), _nmad(v)
                    v = v[np.abs(v - med0) < ngs_nmad_gate * max(nm0, 1e-6)]
                if not len(v):
                    continue
                color = POINT_STYLE[style][1]
                ax.hist(np.clip(v, -lim, lim), bins=41, range=(-lim, lim),
                        histtype="stepfilled", alpha=0.45, color=color,
                        edgecolor=color, label=lab)
                txt.append(f"{lab}: med {np.median(v):+.3f}, "
                           f"NMAD {_nmad(v):.3f}, n={len(v)}")
            ax.axvline(0, color=_INK, lw=0.8)
            ax.set_xlim(-lim, lim)
            ax.set_xlabel(f"dz = {prod} − control (m)", fontsize=9, color=_INK)
            ax.legend(fontsize=8, loc="upper right")
            ax.text(0.02, 0.98, "\n".join(txt), transform=ax.transAxes,
                    fontsize=8, va="top", color=_INK,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
            ax.tick_params(labelsize=8, colors=_MUT)
            ax.grid(alpha=0.25, lw=0.5)
        axes[1].set_title("survey-grade segments", fontsize=10, color=_INK)
        axes[2].set_title(f"NGS monuments ({ngs_nmad_gate:.0f}-NMAD filtered)",
                          fontsize=10, color=_INK)
        fp = outdir / f"{site_name}_validation_dz_{prod}.png"
        fig.tight_layout()
        fig.savefig(fp, dpi=dpi)
        plt.close(fig)
        out.append(fp)
        logger.info("wrote %s", fp)
    return out
