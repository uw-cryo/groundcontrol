"""DEM accuracy assessment pipeline (plan Increment 2).

``transform_control`` lands schema-shaped control into the assessed product's
3D frame with ONE direct transformer (heights never via ``to_crs``;
docs/crs_implementation.md §5), ``sample_products`` samples any number of
rasters on that landing, ``summarize_dz`` reduces the per-point offsets to the
standard segment stats, and :func:`assess_products` orchestrates the three plus
the standard validation figures. ``groundcontrol-assess`` (cli.py) wraps this
module; sandbox site drivers should shrink to configuration.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from groundcontrol.accuracy import med_nmad
from groundcontrol.crs import get_transformer

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#: Landing CRS produced by ``sources.fetch_control`` (schema contract):
#: NAD83(2011) geographic + NAVD88 orthometric heights.
CONTROL_LANDING_CRS = "EPSG:6318+5703"

#: label -> (row mask fn, validates DSM, validates DTM) — the segment taxonomy
#: shared with figures.validation_dz_figures (NVA validates both products, VVA
#: is vegetated/DTM-only, NGS/OPUS are datum-sanity context for both).
SEGMENTS = {
    "3DEP NVA": (lambda d: (d["source"] == "3dep") & (d["point_type"] == "NVA"), True, True),
    "3DEP VVA": (lambda d: (d["source"] == "3dep") & (d["point_type"] == "VVA"), False, True),
    "GNSS/OPUS": (lambda d: d["source"] == "opus", True, True),
    "NGS monument": (lambda d: d["source"] == "ngs", True, True),
}


def transform_control(control, target_crs, *, target_epoch=2010.0,
                      source_crs=None, aoi_bounds_4326=None):
    """Land control points into the assessed product's 3D frame.

    One direct 3D transform on coordinate arrays (never ``to_crs`` for
    heights). The default source is the schema landing
    (:data:`CONTROL_LANDING_CRS`); pass ``source_crs`` to override when the
    input is already elsewhere. ``target_epoch`` is the transform time
    ``tt`` fed to time-dependent legs (provisional D6 rule: plate-fixed
    source -> dynamic target uses the product's epoch; for a static-frame
    target like NAD83(2011) the value is inert).

    Returns ``(gdf, info)``: a copy of ``control`` re-geometried in
    ``target_crs`` with a new ``h_ell`` column (target-frame ellipsoidal
    height) — the orthometric ``height`` column rides along unchanged — and a
    dict recording the operation (``description``, ``accuracy_m``,
    ``pipeline``, ``dh_stats``) for reports/provenance.
    """
    import geopandas as gpd

    if aoi_bounds_4326 is None:
        aoi_bounds_4326 = tuple(control.to_crs("EPSG:6318").total_bounds)
    t = get_transformer(source_crs or CONTROL_LANDING_CRS, target_crs,
                        aoi_bounds_4326=aoi_bounds_4326)
    H = control["height"].to_numpy(dtype="float64")
    E, N, h_ell, _ = t.transform(
        control.geometry.x.to_numpy(dtype="float64"),
        control.geometry.y.to_numpy(dtype="float64"),
        H, np.full(len(control), float(target_epoch)), errcheck=True)
    out = control.copy()
    out["h_ell"] = h_ell
    # per-point stated accuracy of the APPLIED operation (PROJ metadata, m).
    # Constant per call today; becomes genuinely per-point once B7 routes
    # each realization through its own chain. NaN = PROJ reports unknown
    # (e.g. defining Helmert ties) — never silently zero. This is the
    # transformation-budget term for partitioning observed dz biases.
    acc = t.accuracy if (t.accuracy is not None and t.accuracy > 0) else float("nan")
    out["xform_acc_m"] = np.full(len(out), acc)
    out = out.set_geometry(gpd.points_from_xy(E, N), crs=target_crs)
    dh = h_ell - H
    finite = np.isfinite(dh)
    info = {
        "source_crs": str(source_crs or CONTROL_LANDING_CRS),
        "target_epoch": float(target_epoch),
        "description": t.description,
        "accuracy_m": t.accuracy,
        "pipeline": t.definition,
        "n_points": int(len(out)),
        # h_ell - H == applied geoid undulation + frame tie; a gross-error tripwire
        "dh_stats": {k: float(v) for k, v in
                     zip(("min", "median", "max"),
                         (np.min(dh[finite]), np.median(dh[finite]), np.max(dh[finite])))}
        if finite.any() else None,
    }
    logger.info("transform_control: %s (accuracy %s m), h_ell-H median %s",
                info["description"], info["accuracy_m"],
                None if info["dh_stats"] is None else f"{info['dh_stats']['median']:.2f}")
    return out, info


def sample_products(gdf, products, *, method="linear", radius=None, block=4096,
                    check_crs=True):
    """Sample each product raster at the control points; standardized columns.

    ``products`` maps a short product name (e.g. ``"DSM"``) to a raster path
    (or DataArray) **in the same CRS as** ``gdf`` (asserted per raster). Adds,
    per product: ``h_<name>`` (sampled height) and ``dh_<name>_before``
    (product minus control ``h_ell``; the ``_before`` suffix is the
    co-registration convention shared with figures.validation_dz_figures).
    Radius mode also carries ``h_<name>_nmad`` / ``h_<name>_n``. NaN where the
    raster has nodata or the point is outside — points in a merge-mosaic gap
    (e.g. a missing DTM tile) stay NaN and are reported, never dropped.
    """
    from groundcontrol.sample import sample_raster

    out = gdf
    for name, r in products.items():
        before = set(out.columns)
        out = sample_raster(out, r, col="h_ell", method=method, diff=True,
                            block=block, check_crs=check_crs, radius=radius)
        new = [c for c in out.columns if c not in before]
        raster_col = next(c for c in new if not c.endswith(("_nmad", "_n"))
                          and " minus " not in c)
        rename = {raster_col: f"h_{name}"}
        for c in new:
            if c.endswith("_nmad"):
                rename[c] = f"h_{name}_nmad"
            elif c.endswith("_n"):
                rename[c] = f"h_{name}_n"
            elif " minus " in c:
                rename[c] = f"dh_{name}_before"
        out = out.rename(columns=rename)
        n_fin = int(np.isfinite(out[f"dh_{name}_before"].to_numpy(dtype="float64")).sum())
        logger.info("sampled %s: %d/%d points finite", name, n_fin, len(out))
    return out


def summarize_dz(sampled, products=None, segments=SEGMENTS):
    """Tidy per-product, per-segment stats table for ``dh_<prod>_before``.

    ``products`` defaults to every ``dh_*_before`` column present. Returns a
    DataFrame with one row per (product, segment) plus an ``ALL`` segment:
    ``n`` (points in segment), ``n_valid`` (finite dh — the honest sampled
    count; the difference is nodata/outside-extent points such as merge-gap
    tiles), ``median_m``/``nmad_m`` (robust, via accuracy.med_nmad),
    ``mean_m``/``std_m``, ``applies`` (False where the segment does not
    validate that product class, e.g. VVA vs a DSM — rows are still reported
    as context, never silently dropped).
    """
    if products is None:
        products = [c[len("dh_"):-len("_before")] for c in sampled.columns
                    if c.startswith("dh_") and c.endswith("_before")]
    rows = []
    for prod in products:
        col = f"dh_{prod}_before"
        v_all = sampled[col].to_numpy(dtype="float64")
        is_dtm = "DTM" in prod.upper()
        for label, (maskfn, in_dsm, in_dtm) in list(segments.items()) + [
                ("ALL", (lambda d: pd.Series(True, index=d.index), True, True))]:
            m = maskfn(sampled).to_numpy(dtype=bool)
            v = v_all[m]
            fin = v[np.isfinite(v)]
            med, nmad = med_nmad(fin)
            rows.append({
                "product": prod, "segment": label,
                "n": int(m.sum()), "n_valid": int(fin.size),
                "median_m": med, "nmad_m": nmad,
                "mean_m": float(np.mean(fin)) if fin.size else float("nan"),
                "std_m": float(np.std(fin)) if fin.size else float("nan"),
                "applies": bool(in_dtm if is_dtm else in_dsm),
            })
    return pd.DataFrame(rows)


def assess_products(control, products, target_crs, *, outdir, site_name,
                    aoi=None, hs=None, target_epoch=2010.0, method="linear",
                    radius=None, source_crs=None, figures=True, write=True,
                    command=None):
    """Fetch-free assessment: transform -> sample -> stats (+ figures, files).

    Parameters mirror the component functions; ``aoi`` (path or GeoDataFrame,
    any CRS) and ``hs`` (hillshade path or ``{product: path}`` dict) feed the
    validation figures. With ``write=True`` the sampled points land in
    ``<outdir>/<site_name>_assessed.parquet`` (io.write provenance sidecar)
    and the stats table in ``<site_name>_dz_stats.csv``.

    Returns ``(sampled, stats, artifacts)`` where ``artifacts`` is a dict of
    written paths plus the ``transform`` info block.
    """
    import geopandas as gpd

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    artifacts = {}

    landed, tinfo = transform_control(control, target_crs,
                                      target_epoch=target_epoch,
                                      source_crs=source_crs)
    artifacts["transform"] = tinfo
    sampled = sample_products(landed, products, method=method, radius=radius)
    stats = summarize_dz(sampled, products=list(products))

    if write:
        from groundcontrol import io
        p = outdir / f"{site_name}_assessed.parquet"
        io.write(sampled, p, status={"assess": {"n_rows": len(sampled),
                                                "error": None,
                                                "transform": tinfo}},
                 command=command or f"groundcontrol.assess.assess_products({site_name})")
        artifacts["assessed_parquet"] = p
        sp = outdir / f"{site_name}_dz_stats.csv"
        stats.to_csv(sp, index=False, float_format="%.4f")
        artifacts["dz_stats_csv"] = sp

    if figures:
        from groundcontrol.figures import validation_dz_figures
        aoi_gdf = None
        if aoi is not None:
            aoi_gdf = gpd.read_file(aoi) if isinstance(aoi, (str, Path)) else aoi
            aoi_gdf = aoi_gdf.to_crs(sampled.crs)
        artifacts["figures"] = validation_dz_figures(
            sampled, aoi_gdf, outdir, site_name,
            products=list(products), hs_tif=hs)
    return sampled, stats, artifacts
