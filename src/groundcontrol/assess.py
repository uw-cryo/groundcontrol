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

from groundcontrol.crs import get_transformer

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#: Landing CRS produced by ``sources.fetch_control`` (schema contract):
#: NAD83(2011) geographic + NAVD88 orthometric heights.
CONTROL_LANDING_CRS = "EPSG:6318+5703"

#: label -> (row mask fn, validates DSM, validates DTM) — THE segment
#: taxonomy: figures.validation_dz_figures DERIVES its seg_defs from this
#: dict (audit round 2: the hand-copied twin had already diverged once).
#: GNSS segments follow the PER-ROW occupation class (owner taxonomy,
#: 2026-08-22; sources.ngl.occupation_class): each station's own record earns
#: its class, so routing is by point_type, qualified by source only where
#: height bases differ. Class = occupation pattern, NOT quality (see
#: ngl.occupation_class).
#: Applicability: only OPUS campaign rows validate products — their heights
#: are the ground mark. Every NGL-fed segment is CONTEXT-ONLY
#: (False, False): NGL heights are the antenna reference point and the
#: assess path applies no ant_m/monument-height correction yet (owner call,
#: MDV 2026-08: NGL dh = frame check only; audit round 2 caught True flags
#: biasing class stats by the full antenna height). Pre-split and
#: other-source GNSS rows are context-only too — height basis unknown or
#: mixed — but stay visible in the table, never silently dropped.
SEGMENTS = {
    "3DEP NVA": (lambda d: (d["source"] == "3dep") & (d["point_type"] == "NVA"), True, True),
    "3DEP VVA": (lambda d: (d["source"] == "3dep") & (d["point_type"] == "VVA"), False, True),
    "GNSS continuous": (lambda d: d["point_type"] == "gnss_cont", False, False),
    "GNSS semi-continuous": (lambda d: d["point_type"] == "gnss_semicont",
                             False, False),
    # legacy pre-split OPUS rows ARE campaign ground marks (OPUS is episodic
    # by definition; same NAVD88 mark heights) — they keep validating here,
    # exactly as they did before the split (audit round 3: routing them to
    # the context-only pre-split row silently flipped applies on every
    # existing parquet — a strictly-additive violation)
    "GNSS campaign (OPUS)": (lambda d: (d["source"] == "opus")
                             & d["point_type"].isin(["gnss_campaign", "gnss"]),
                             True, True),
    "GNSS campaign (NGL)": (lambda d: (d["source"] == "ngl")
                            & (d["point_type"] == "gnss_campaign"), False, False),
    # exhaustiveness backstop: a future source emitting gnss_campaign must
    # surface here (n=0 today), never silently fall out of the table
    "GNSS campaign (other)": (lambda d: (d["point_type"] == "gnss_campaign")
                              & ~d["source"].isin(["opus", "ngl"]), False, False),
    # non-OPUS legacy rows only (OPUS legacy folds into campaign above);
    # height basis unknown or ARP -> context
    "GNSS (pre-split)": (lambda d: (d["point_type"] == "gnss")
                         & (d["source"] != "opus"), False, False),
    "NGS monument": (lambda d: d["source"] == "ngs", True, True),
}


def _unsegmented(d, _segs=tuple(SEGMENTS.values())):
    """Complement of every mask above — rows no segment claims."""
    m = np.zeros(len(d), dtype=bool)
    for fn, _, _ in _segs:
        m |= pd.Series(fn(d)).fillna(False).to_numpy(dtype=bool)
    return pd.Series(~m, index=d.index)


#: audit round 4: a row matching NO segment (e.g. NA point_type, which the
#: schema permits) must surface in the table, never silently vanish —
#: main's source-gated masks caught every row; the class-gated ones don't.
SEGMENTS["OTHER (unsegmented)"] = (_unsegmented, False, False)


def is_dtm_product(name: str) -> bool:
    """Shared DSM/DTM classifier for the SEGMENTS applies flags.

    Audit round 4: figures keyed on ``prod == "DSM"`` while the stats table
    used this rule — the same flags read through two different classifiers
    disagree for names like "dsm" or "DSM_2020". Both consumers now call
    this one function.
    """
    return "DTM" in name.upper()


def has_vertical_axis(crs) -> bool:
    """True when ``crs`` is a full 3D frame: a horizontal pair PLUS a
    gravity-related or ellipsoidal height axis (compound with a vertical
    member, or a 3D geographic/projected CRS). A 2D CRS, a geocentric XYZ
    CRS, a horizontal+temporal compound, or a vertical CRS ALONE do not:
    heights (or the horizontal) would ride through a transform to them
    untransformed. The one rule behind both the ``transform_control`` guard
    and the CLI's embedded-CRS acceptance (review rounds 1-2, 2026-08-27:
    round 2 caught a bare EPSG:5703 target passing an up-axis-only rule).
    """
    import pyproj

    crs = pyproj.CRS.from_user_input(crs)
    dirs = [a.direction.lower() for a in crs.axis_info]
    vertical = [d for d in dirs if d in ("up", "down")]
    horizontal = [d for d in dirs if d not in ("up", "down", "future", "past")
                  and not d.startswith("geocentric")]
    return bool(vertical) and len(horizontal) >= 2


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
    ``target_crs`` with a new ``h_ell`` column holding the target-frame
    height — ellipsoidal for a 3D target; orthometric when ``target_crs`` is
    compound (column name kept stable for the pipeline) — the source
    ``height`` column rides along unchanged — and a dict recording the
    operation (``description``, ``accuracy_m``, ``pipeline``, ``dh_stats``)
    for reports/provenance.
    """
    import geopandas as gpd
    import pyproj

    src = source_crs or CONTROL_LANDING_CRS
    if control.crs is not None:
        assumed = pyproj.CRS(src)
        declared = control.crs
        # a 2D geometry tag says nothing about heights (schema: heights live
        # in the `height` column) -> match the horizontal member; a 3D or
        # compound tag carries height semantics and must match exactly — an
        # EPSG:6319 (ellipsoidal-height) frame run under the default NAVD88
        # landing would get the geoid undulation applied to already-
        # ellipsoidal heights, and the dh_stats tripwire would then read like
        # a plausible geoid signal instead of an error.
        horiz = (assumed.sub_crs_list[0] if assumed.is_compound
                 else assumed.to_2d())

        def _match(x, y):
            # equals() misses equivalent re-encodings (ESRI WKT1 from a .prj);
            # a resolved matching EPSG code is the same CRS
            return x.equals(y) or (x.to_epsg() is not None
                                   and x.to_epsg() == y.to_epsg())

        if not (_match(declared, assumed)
                or (len(declared.axis_info) == 2 and _match(declared, horiz))):
            raise ValueError(
                f"control frame declares CRS {declared.name!r} but the "
                f"transform source is {src!r} ({assumed.name}); refusing to "
                "reinterpret coordinates — pass source_crs= matching the data "
                "(or retag the frame). The vertical datum is never guessed.")
    # issue #22 guard: a 2D (non-compound) target builds a horizontal-only
    # pipeline and heights ride through UNTRANSFORMED — the geoid undulation
    # lands in every dz as bias (Atlanta swept -31 m with a normal-looking
    # NMAD). The ONLY height-inert 2D case is source == target (identity;
    # the end-to-end tests use it deliberately) — anything else must carry
    # 3-axis or compound height semantics — with an actual height AXIS: a
    # geocentric XYZ or horizontal+temporal compound is 3-axis/compound and
    # still carries no height datum (review round 1).
    tgt = pyproj.CRS(target_crs)
    if not has_vertical_axis(tgt):
        src_obj = pyproj.CRS(src)
        same = src_obj.equals(tgt) or (src_obj.to_epsg() is not None
                                       and src_obj.to_epsg() == tgt.to_epsg())
        if not same:
            raise ValueError(
                f"target_crs {tgt.name!r} has no horizontal+height axis pair (2D, "
                f"geocentric, or height-less compound; axes: "
                f"{[a.direction for a in tgt.axis_info]}): heights would "
                "pass through UNTRANSFORMED while the horizontal moves — the "
                "geoid undulation would land in every dz as bias. Pass the "
                "product's 3D CRS (e.g. pyproj.CRS('EPSG:32616').to_3d()) or "
                "a compound 'horizontal+vertical' CRS ('EPSG:32616+5703'); a "
                "height-less target is only valid when source_crs equals it exactly "
                "(same-frame identity).")
    if aoi_bounds_4326 is None:
        aoi_bounds_4326 = tuple(control.to_crs("EPSG:4326").total_bounds)
    t = get_transformer(src, target_crs, aoi_bounds_4326=aoi_bounds_4326)
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
        clash = [c for c in (f"h_{name}", f"dh_{name}_before",
                             f"h_{name}_nmad", f"h_{name}_n") if c in out.columns]
        if clash:
            raise ValueError(
                f"columns {clash} already present — product {name!r} appears to "
                "have been sampled into this frame already; re-sampling would "
                "create duplicate labels and silently mixed statistics. Drop "
                "those columns or use a different product name.")
        before = set(out.columns)
        out = sample_raster(out, r, col="h_ell", method=method, diff=True,
                            block=block, check_crs=check_crs, radius=radius)
        new = [c for c in out.columns if c not in before]
        try:
            raster_col = next(c for c in new if not c.endswith(("_nmad", "_n"))
                              and " minus " not in c)
        except StopIteration:
            raise ValueError(
                f"could not identify the sampled column for product {name!r} "
                f"(new columns: {sorted(new)}); a raster stem that collides "
                "with an existing column (e.g. 'h_ell') or ends in _n/_nmad "
                "breaks the rename — rename the raster file") from None
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
    DataFrame with one row per (product, segment) plus an ``ALL`` segment.
    Dual-track reporting (owner 2026-07-16; ASPRS Ed.2/LBS-2024 vocabulary):
    robust ``median_m``/``nmad_m`` over all finite residuals, then the
    parametric set the cal/val community expects AFTER a 3*NMAD outlier gate
    — ``mean_m`` (bias), ``std_m`` (1-sigma, ddof=1), ``rmse_m``,
    ``le90_m``/``le95_m`` (empirical |error| percentiles) — plus ``n``,
    ``n_valid`` (finite dh; the gap-honesty count), ``n_out`` (removed by the
    gate) and ``xform_acc_m`` (stated transform budget for the chain).
    ``applies`` is False where the segment does not validate that product
    class (e.g. VVA vs a DSM) — rows are reported as context, never dropped.
    """
    from groundcontrol.accuracy import error_report

    if products is None:
        products = [c[len("dh_"):-len("_before")] for c in sampled.columns
                    if c.startswith("dh_") and c.endswith("_before")]
    budget = float("nan")
    if "xform_acc_m" in sampled.columns:
        xa = sampled["xform_acc_m"].to_numpy(dtype="float64")
        if np.isfinite(xa).any():  # all-NaN (PROJ sentinel budget) stays NaN, quietly
            budget = float(np.nanmedian(xa))
    rows = []
    for prod in products:
        col = f"dh_{prod}_before"
        v_all = sampled[col].to_numpy(dtype="float64")
        is_dtm = is_dtm_product(prod)
        for label, (maskfn, in_dsm, in_dtm) in list(segments.items()) + [
                ("ALL", (lambda d: pd.Series(True, index=d.index), True, True))]:
            # fillna(False): source columns are pandas nullable strings — one
            # NA point_type upstream must exclude the row, not crash the table
            m = pd.Series(maskfn(sampled)).fillna(False).to_numpy(dtype=bool)
            r = error_report(v_all[m])
            rows.append({
                "product": prod, "segment": label,
                "n": int(m.sum()), "n_valid": r["n"], "n_out": r["n_outliers"],
                "median_m": r["median"], "nmad_m": r["nmad"],
                "mean_m": r["mean"], "std_m": r["std"], "rmse_m": r["rmse"],
                "le90_m": r["le90"], "le95_m": r["le95"],
                "xform_acc_m": budget,
                "applies": bool(in_dtm if is_dtm else in_dsm),
            })
    return pd.DataFrame(rows)


def assess_products(control, products, target_crs, *, outdir, site_name,
                    aoi=None, hs=None, rgb=None, intensity=None,
                    basemap="esri", target_epoch=2010.0, method="linear",
                    radius=None, source_crs=None, figures=True, write=True,
                    point_lim=None, vendor_lim=None, wide_lim=None,
                    command=None):
    """Fetch-free assessment: transform -> sample -> stats (+ figures, files).

    Parameters mirror the component functions; ``aoi`` (bbox-less: a
    vector/raster path or GeoDataFrame, any CRS — clips and outlines the
    figure maps; ``None`` leaves them unclipped) and ``hs`` (pre-rendered
    hillshade path or ``{product: path}`` dict) feed the validation figures.
    With ``hs=None`` a hillshade is computed from each product raster
    (:func:`figures.hillshade_from_raster`), so a bring-your-own-DEM run
    needs nothing but the DEM. The figure bundle also includes per-point
    context contact sheets for the GNSS and FAA subsets
    (:func:`figures.context_sheets`): RGB imagery (``rgb`` ortho path(s)
    and/or the ``basemap`` web provider — ``"esri"`` by default, fetched
    over the network and credited on the sheet; ``None`` for offline) |
    ``intensity`` raster when given | one shaded-relief panel per product.
    With ``write=True`` the sampled points land in
    ``<outdir>/<site_name>_assessed.parquet`` (io.write provenance sidecar)
    and the stats table in ``<site_name>_dz_stats.csv``.

    Returns ``(sampled, stats, artifacts)`` where ``artifacts`` is a dict of
    written paths plus the ``transform`` info block.

    The two data exports are preflighted (:func:`io.check_export_support`)
    before any transform/sample work. Figures are deliberately not: they
    land in the same, already-verified directory, ``savefig`` does not
    truncate a read-only file, and their names are derived deep in
    ``figures.py`` -- enumerating them here would duplicate that logic.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    if write:
        # Fail before transform/sample if the assessed export cannot be
        # written (missing parquet engine, unwritable dir): same guard the
        # CLIs apply before the fetch, for library callers.
        from groundcontrol import io
        io.check_export_support(outdir / f"{site_name}_assessed.parquet")
        io.check_export_support(outdir / f"{site_name}_dz_stats.csv", sidecar=False)

    landed, tinfo = transform_control(control, target_crs,
                                      target_epoch=target_epoch,
                                      source_crs=source_crs)
    artifacts["transform"] = tinfo
    sampled = sample_products(landed, products, method=method, radius=radius)
    stats = summarize_dz(sampled, products=list(products))

    if write:
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
        from groundcontrol.figures import (context_sheets, hillshade_from_raster,
                                           validation_dz_figures)
        aoi_gdf = None
        if aoi is not None:
            if isinstance(aoi, (str, Path)):
                from groundcontrol.aoi import read_aoi
                aoi_gdf = read_aoi(aoi)  # vector features or raster footprint
            else:
                aoi_gdf = aoi
            aoi_gdf = aoi_gdf.to_crs(sampled.crs)
        if hs is None:
            # product-matched underlays from the products themselves; an
            # in-memory DataArray product has no path -> plain map
            hs = {name: h for name, p in products.items() if isinstance(p, (str, Path))
                  and (h := hillshade_from_raster(p)) is not None}
        artifacts["figures"] = validation_dz_figures(
            sampled, aoi_gdf, outdir, site_name,
            products=list(products), hs_tif=hs,
            point_lim=point_lim, vendor_lim=vendor_lim, wide_lim=wide_lim)
        # standard contact sheets for the sparse photo-identifiable subsets
        # (GNSS occupation classes, FAA runway control; owner 2026-08-29)
        sheets = context_sheets(sampled, products, outdir, site_name,
                                rgb=rgb, intensity=intensity, basemap=basemap)
        if sheets:
            artifacts["context_sheets"] = sheets
    return sampled, stats, artifacts
