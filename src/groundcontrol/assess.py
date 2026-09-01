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
    # FAA NASR runway control: survey-grade = the PAINTED runway features
    # (runway ends + displaced thresholds) in the surveyed provenance class
    # (AC 150/5300-18C; LV A/B 2026-08-13 ~2 cm NMAD; CG 2026-08-30
    # measured med -0.047 / NMAD 0.061). HELIPADS are excluded even when
    # the position source reads surveyed/MILITARY: CG measured a
    # consistent +0.33 m bias on 8 MILITARY-source helipads — a different
    # accuracy class (owner 2026-08-30: some are hand-held GNSS), so they
    # ride with the estimated class as context-only. Before 2026-08-30
    # every FAA row fell to OTHER (unsegmented).
    "FAA runway surveyed": (
        lambda d: (d["source"] == "faa")
        & (_faa_pos_class(d) == "surveyed")
        & d["point_type"].isin(["runway_end", "displaced_threshold"]),
        True, True),
    # service-branch-owned facilities (sources/faa.py MIL_OWNERSHIP):
    # elevations ride the DoD pipeline (EGM96 MSL standard) and the
    # per-point datum is unverifiable (Nellis 2026-08-31: runway ends off
    # by exactly the EGM96-NAVD88 separation, helipad NAVD88 to 3 mm) —
    # own context segment, never in the surveyed tier
    "FAA MIL field": (
        lambda d: (d["source"] == "faa")
        # isin: caches written before the 2026-09-01 rename carry the
        # old "military" value in raw
        & _faa_pos_class(d).isin(("mil", "military")), False, False),
    "FAA other": (
        lambda d: (d["source"] == "faa")
        & ~_faa_pos_class(d).isin(("mil", "military"))
        & ~((_faa_pos_class(d) == "surveyed")
            & d["point_type"].isin(["runway_end", "displaced_threshold"])),
        False, False),
}


def _faa_pos_class(d):
    """Row-wise NASR position-source class from ``raw`` (figures._raw_field
    is the ONE raw reader — round-4 lesson: two readers diverge)."""
    from groundcontrol.figures import _raw_field
    if "raw" not in d.columns:
        return pd.Series(pd.NA, index=d.index, dtype="object")
    return _raw_field(d["raw"], "pos_class")


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

    if len(control) == 0:
        # NaN total_bounds otherwise reach PROJ's AreaOfInterest as an
        # opaque "Invalid latitude"-class error far from the cause
        raise ValueError(
            "transform_control: the control frame is empty — nothing to "
            "transform (check the AOI, source selection, and any "
            "upstream filters)")
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
    # DEPTH targets refuse everywhere, not just the CLI's ensemble branch
    # (round-5 audit: a NAD83 + EPSG:5715 target landed +1500 m control
    # at h_ell=-1500, silently — every downstream comparison assumes
    # height-up). The name stem backstops WKT1 round-trips that erase
    # axis direction: 50/52 EPSG depth verticals say 'depth' in the
    # name, 0/246 height verticals do.
    _depth = any((a.direction or "").lower() == "down"
                 for a in tgt.axis_info)
    if not _depth and tgt.is_compound:
        _depth = any("depth" in (pyproj.CRS(s).name or "").lower()
                     for s in tgt.sub_crs_list if pyproj.CRS(s).is_vertical)
    if _depth:
        raise ValueError(
            f"target_crs {tgt.name!r} carries a DEPTH-type vertical "
            "(positive down): heights would land sign-flipped while every "
            "downstream dz comparison assumes height-up. Pass a "
            "height-type vertical (e.g. EPSG:5703 NAVD88, EPSG:3855 "
            "EGM2008).")
    if aoi_bounds_4326 is None:
        aoi_bounds_4326 = tuple(control.to_crs("EPSG:4326").total_bounds)
    t = get_transformer(src, target_crs, aoi_bounds_4326=aoi_bounds_4326)
    H = control["height"].to_numpy(dtype="float64")
    # per-row VERTICAL compatibility (2026-08-30, found when ngl joined the
    # default sources): the declared source chain is valid only for rows
    # whose vertical_crs matches the declared source's vertical member.
    # NGL interim rows carry NATIVE-frame ellipsoidal heights (schema
    # contract) — running them through the NAVD88 chain would silently add
    # the ~30 m geoid to already-ellipsoidal heights, the exact error
    # class this library refuses. Incompatible rows keep the horizontal
    # leg (positions stay valid for maps/sheets) and get h_ell = NaN:
    # visibly unassessable, never silently wrong. Full per-frame vertical
    # landing is the D6/§5 work.
    n_vert_excluded = 0
    n_vert_native = 0
    vert_note = None
    incompat = None
    if "vertical_crs" in control.columns:
        src_obj2 = pyproj.CRS(src)
        src_vert = None    # gravity-related vertical member (compound src)
        src_frame = None   # 3D frame with ellipsoidal heights
        if src_obj2.is_compound:
            src_vert = pyproj.CRS(src_obj2.sub_crs_list[1])
        elif not src_obj2.is_geocentric and len(src_obj2.axis_info) == 3:
            # a 3D non-compound source declares ELLIPSOIDAL heights on its
            # frame. Before this branch the guard was silently inert here
            # — the exact ellipsoidal-source case a mixed cache hits — and
            # EPSG:5703 rows ran the ellipsoidal chain with the ~30 m
            # geoid applied to already-orthometric heights, counted valid.
            src_frame = src_obj2
        if src_vert is not None or src_frame is not None:
            vc = control["vertical_crs"].astype("string")
            src_code = (src_vert.to_epsg() if src_vert is not None
                        else src_obj2.to_epsg())
            # NA = unknown height datum (sources/ngs.py contract: refuse,
            # never guess) — previously let through by vc.notna() & (...)
            incompat_arr = vc.isna().to_numpy(dtype=bool)
            for val in vc.dropna().unique():
                m = (vc == val).fillna(False).to_numpy(dtype=bool)
                if src_code is not None and str(val) == f"EPSG:{src_code}":
                    continue
                try:
                    v_obj = pyproj.CRS.from_user_input(str(val))
                except Exception:
                    incompat_arr |= m
                    continue
                if src_vert is not None:
                    # orthometric chain: compatible = a vertical CRS on
                    # the SAME datum, by datum identity, not code-string
                    # equality: EPSG:6360 (NAVD88 in ftUS) is the same
                    # datum as EPSG:5703 and was falsely excluded. NO
                    # unit scaling: the schema says 'height' is always
                    # metres and vertical_crs is provenance of the
                    # ORIGINAL values (round-2 audit: scaling here would
                    # silently divide schema-compliant metre heights by
                    # 3.28).
                    if (not v_obj.is_vertical or v_obj.datum is None
                            or src_vert.datum is None
                            or v_obj.datum != src_vert.datum):
                        incompat_arr |= m
                        continue
                    logger.info(
                        "transform_control: %d row(s) carry %s — the "
                        "source vertical datum under a different code; "
                        "heights are metres per schema, used as-is",
                        int(m.sum()), val)
                else:
                    # ellipsoidal chain: compatible = the source frame
                    # itself (per-row codes are aliased 3D frame codes,
                    # ngl schema); a gravity-related vertical or a
                    # DIFFERENT frame goes to the native re-target below
                    if (v_obj.is_vertical or v_obj.datum is None
                            or src_frame.datum is None
                            or v_obj.datum != src_frame.datum):
                        incompat_arr |= m
            incompat = pd.Series(incompat_arr, index=control.index)
            if incompat.any():
                H = H.copy()
                H[incompat.to_numpy(dtype=bool)] = np.nan
    E, N, h_ell, _ = t.transform(
        control.geometry.x.to_numpy(dtype="float64"),
        control.geometry.y.to_numpy(dtype="float64"),
        np.nan_to_num(H, nan=0.0),  # NaN in -> PROJ errcheck aborts; the
        np.full(len(control), float(target_epoch)), errcheck=True)
    out = control.copy()
    # horizontal leg is height-independent for these chains, and the
    # heights of masked rows are discarded below
    h_ell = np.where(np.isfinite(H), h_ell, np.nan)
    acc_row = np.full(len(control), np.nan)
    t_acc = t.accuracy if (t.accuracy is not None and t.accuracy > 0) else float("nan")
    acc_row[np.isfinite(H)] = t_acc
    # NATIVE RE-TARGET for vertically-mismatched rows (owner 2026-08-30:
    # "I need to see the dz values" — for Rasuwa NGL may be ALL the control
    # there is). The schema keeps native_x/y/h + native_crs for lossless
    # re-targeting (§5): each mismatched subset is transformed from its
    # NATIVE 3D coordinates through its own frame's chain — geodetically
    # correct heights (the visible residual is then real: e.g. the NGL
    # antenna-reference offset), never the wrong-chain geoid error. tt =
    # per-row coord_epoch for dynamic native frames (the land_horizontal
    # D6 provisional rule). Rows whose natives/epochs are unusable stay
    # masked NaN — the honest fallback.
    if incompat is not None and incompat.any():
        from groundcontrol.crs import is_dynamic_frame
        native_ok = incompat.copy()
        for c in ("native_x", "native_y", "native_h"):
            native_ok &= (pd.to_numeric(control.get(c), errors="coerce").notna()
                          if c in control.columns else False)
        if "native_crs" in control.columns:
            native_ok &= control["native_crs"].notna()
        else:
            native_ok &= False
        if native_ok.any():
            # positional masks, not label groupby: duplicate index labels
            # (two caches concatenated) made get_indexer raise
            # InvalidIndexError and took down the whole transform
            natmask = native_ok.to_numpy(dtype=bool)
            ncrs_all = control["native_crs"].astype("string")
            for ncrs in ncrs_all[natmask].unique():
                mrow = natmask & (ncrs_all == ncrs).fillna(False).to_numpy(
                    dtype=bool)
                pos = np.flatnonzero(mrow)
                sub = control.iloc[pos]
                nx = pd.to_numeric(sub["native_x"], errors="coerce").to_numpy("float64")
                ny = pd.to_numeric(sub["native_y"], errors="coerce").to_numpy("float64")
                nh = pd.to_numeric(sub["native_h"], errors="coerce").to_numpy("float64")
                if is_dynamic_frame(str(ncrs)):
                    tt2 = pd.to_numeric(sub.get("coord_epoch"),
                                        errors="coerce").to_numpy("float64")
                    usable = np.isfinite(tt2)
                else:
                    tt2 = np.full(len(sub), float(target_epoch))
                    usable = np.ones(len(sub), dtype=bool)
                if not usable.all():
                    logger.warning(
                        "transform_control: %d native %s row(s) lack a finite "
                        "coord_epoch for the dynamic-frame chain; left masked",
                        int((~usable).sum()), ncrs)
                if not usable.any():
                    continue
                try:
                    # the outer AOI bounds are the same physical area and
                    # already 4326; native x/y may be PROJECTED (UTM
                    # eastings fed as "degrees" broke candidate selection)
                    t2 = get_transformer(
                        str(ncrs), target_crs,
                        aoi_bounds_4326=aoi_bounds_4326)
                    E2, N2, h2, _ = t2.transform(nx[usable], ny[usable],
                                                 nh[usable], tt2[usable],
                                                 errcheck=True)
                except Exception as exc:
                    logger.warning("transform_control: native chain %s -> "
                                   "target unavailable (%s); rows stay masked",
                                   ncrs, exc)
                    continue
                p_use = pos[usable]
                E[p_use], N[p_use], h_ell[p_use] = E2, N2, h2
                a2 = t2.accuracy if (t2.accuracy is not None
                                     and t2.accuracy > 0) else float("nan")
                acc_row[p_use] = a2
                n_vert_native += int(usable.sum())
                logger.info("transform_control: %d row(s) re-targeted from "
                            "native %s (chain: %s)", int(usable.sum()), ncrs,
                            t2.description)
        n_vert_excluded = int(incompat.sum()) - n_vert_native
        if incompat.any():
            # NA rows are now refused (never guessed), so the note's value
            # list must survive NA: sorted() raises on pd.NA comparison
            vc_bad = control.loc[incompat, "vertical_crs"].astype("string")
            bad = sorted(vc_bad.dropna().unique())
            if vc_bad.isna().any():
                bad.append("<NA>")
            vert_note = (
                f"{int(incompat.sum())} row(s) carry vertical_crs {bad} != "
                f"the declared source vertical: {n_vert_native} re-targeted "
                f"from native 3D coordinates through their own chain, "
                f"{n_vert_excluded} left with h_ell=NaN (no usable natives).")
            logger.warning("transform_control: %s", vert_note)
    out["h_ell"] = h_ell
    # per-point stated accuracy of the APPLIED operation (PROJ metadata, m).
    # Constant per call today; becomes genuinely per-point once B7 routes
    # each realization through its own chain. NaN = PROJ reports unknown
    # (e.g. defining Helmert ties) — never silently zero. This is the
    # transformation-budget term for partitioning observed dz biases.
    # per-row: the declared chain's stated accuracy, or the native chain's
    # for re-targeted rows; NaN where masked (never silently zero)
    out["xform_acc_m"] = acc_row
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
        "n_vertical_excluded": n_vert_excluded,
        "n_vertical_native": n_vert_native,
        "vertical_note": vert_note,
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
                    check_crs=True, declared_crs=None):
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
        # pre-log: a large VRT mosaic can take a while to open/read, and
        # this stage sat silent after the transform line (owner 2026-08-30,
        # Las Vegas 0.5 m VRT)
        logger.info("sampling %s at %d points ...", name, len(out))
        out = sample_raster(out, r, col="h_ell", method=method, diff=True,
                            block=block, check_crs=check_crs, radius=radius,
                            declared_crs=declared_crs)
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


def check_product_family(products) -> None:
    """One run assesses ONE site's product family: at most one surface
    (DSM-classified) and one bare-earth (DTM-classified) product (owner
    ruling 2026-08-30, after a multi-strip run produced context sheets
    where one of five DEMs covered each point). Multiple same-class
    products are independent acquisitions — separate runs, one per file.
    Raises ``ValueError`` naming the offenders."""
    surface = [n for n in products if not is_dtm_product(n)]
    bare = [n for n in products if is_dtm_product(n)]
    for cls, names in (("surface (DSM)", surface), ("bare-earth (DTM)", bare)):
        if len(names) > 1:
            raise ValueError(
                f"{len(names)} {cls} products in one run ({names}): one "
                "run assesses ONE site's product family — at most one "
                "surface and one bare-earth product. Independent "
                "acquisitions (e.g. multiple EarthDEM strips) are "
                "separate runs: invoke once per file")


def assess_products(control, products, target_crs, *, outdir, site_name,
                    aoi=None, hs=None, rgb=None, intensity=None,
                    basemap="esri", midas_velocities=True, sheets=False,
                    target_epoch=2010.0, method="linear",
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
    The bundle also carries the MIDAS velocity maps and NGL series
    figures (``ngl/`` subdir) by default — ``midas_velocities=False``
    opts OUT (e.g. for a deliberately offline run). Per-point contact
    sheets are OPT-IN (``sheets=True`` / ``--context-sheets``): the
    slow figure component, and most runs don't need them.
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
    check_product_family(products)
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

    import time as _time
    _t0 = _time.monotonic()
    landed, tinfo = transform_control(control, target_crs,
                                      target_epoch=target_epoch,
                                      source_crs=source_crs)
    artifacts["transform"] = tinfo
    # target_crs IS the declaration of the products' true frame: sampling
    # accepts the header-vs-declaration datum reinterpretation (same grid).
    # Record any such override in the transform provenance — previously it
    # existed only as one log line, invisible to the sidecar/stats readers
    _reinterp = {}
    for _name, _p in products.items():
        try:
            import pyproj as _pp
            import rasterio as _rio
            if hasattr(_p, "rio"):
                _rcrs = _p.rio.crs
            else:
                with _rio.open(_p) as _src:
                    _rcrs = _src.crs
            if _rcrs is None:
                continue
            from groundcontrol.sample import _grid_signature, _horizontal_2d
            _r = _pp.CRS.from_user_input(_rcrs)
            _t = _pp.CRS.from_user_input(target_crs)
            if (not _horizontal_2d(_r).equals(_horizontal_2d(_t))
                    and _grid_signature(_r) == _grid_signature(_t)):
                _reinterp[_name] = {"header": _r.name, "declared": _t.name}
        except Exception:  # provenance annotation only — never block
            continue
    if _reinterp:
        tinfo["datum_reinterpretation"] = _reinterp
    sampled = sample_products(landed, products, method=method, radius=radius,
                              declared_crs=target_crs)
    logger.info("transform + sampling: %.1f s", _time.monotonic() - _t0)
    _t0 = _time.monotonic()
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
        if sheets:
            # per-point contact sheets: the SLOW figure component (web-tile
            # windows per point) — opt-in since 2026-08-30 (owner: useful,
            # but not what most users want to wait for by default)
            sheet_paths = context_sheets(sampled, products, outdir, site_name,
                                         rgb=rgb, intensity=intensity,
                                         basemap=basemap)
            if sheet_paths:
                artifacts["context_sheets"] = sheet_paths
        # largest / smallest vertical-residual review sheets: capped
        # per-source selection, so few pages and fast — default ON (owner
        # 2026-08-31: the biased-vs-tight separator is surface context, not
        # datasheet attributes; review it in imagery)
        from groundcontrol.figures import dz_residual_sheets
        residual_paths = dz_residual_sheets(sampled, products, outdir,
                                            site_name, rgb=rgb,
                                            intensity=intensity,
                                            basemap=basemap)
        if residual_paths:
            artifacts["dz_residual_sheets"] = residual_paths
        # the LABELED all-sources control map (+ monument facets, and the
        # MIDAS velocity maps + NGL series — default ON like every other
        # standard figure, owner 2026-08-31: "you shouldn't have to
        # specify"; a site with no MIDAS stations/NGL rows just emits
        # nothing, the FAA/3DEP no-sites pattern, and the network fetch
        # degrades to a logged skip offline): the companion the contact
        # sheets need — dz colors cannot carry class identity, and
        # station/airport labels locate each sheet cell on the map (owner
        # 2026-08-13 spec; wired into the standard bundle 2026-08-30)
        from groundcontrol.figures import (SOURCE_DIRS, family_dz_figures,
                                           standard_control_figures)
        first = next((p for p in products.values()
                      if isinstance(p, (str, Path))), None)
        artifacts["control_figures"] = standard_control_figures(
            sampled, aoi_gdf, outdir, site_name, dem_tif=first,
            hs_tif=(hs.get(next(k for k, p in products.items() if p == first))
                    if isinstance(hs, dict) and first is not None else hs),
            midas_velocities=midas_velocities)
        # per-SOURCE dh map + histogram (family_dz_figures) in each source's
        # subdir (owner 2026-08-30 layout): only families that came back
        fams = []
        src_col = sampled.get("source")
        pt_col = sampled.get("point_type")
        if src_col is not None:
            if (src_col == "3dep").any():
                fams.append("3dep")
            if pt_col is not None and pt_col.astype("string").str.startswith(
                    "gnss").fillna(False).any():
                fams.append("gnss")
            if ((src_col == "ngs").any() and "raw" in sampled.columns
                    and "ref_frame" in sampled.columns):
                fams.append("ngs_best")   # tier mask reads raw + ref_frame
            if (src_col == "faa").any() and "raw" in sampled.columns:
                fams.append("faa")        # provenance split reads raw
        artifacts["family_figures"] = []
        for fam in fams:
            artifacts["family_figures"] += family_dz_figures(
                sampled, aoi_gdf, outdir / SOURCE_DIRS[fam], site_name,
                products=list(products), hs_tif=hs, families=(fam,))
        logger.info("figures: %.1f s", _time.monotonic() - _t0)
    return sampled, stats, artifacts
