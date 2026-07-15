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
        # 'antialiased' low-pass filters when the raster is drawn smaller
        # than its pixel count — 'nearest' moires badly on urban-grid
        # hillshades (David, 2026-07-15).
        ax.imshow(hs, cmap="gray", vmin=0.0, vmax=1.0, extent=hs_extent,
                  interpolation="antialiased", interpolation_stage="rgba",
                  zorder=0)
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


def _resolve_aoi_polygon(aoi):
    """AOI (gdf/GeoSeries/shapely polygon/bbox) -> a single shapely polygon in
    EPSG:4326 (lon/lat degrees), or ``None`` for ``aoi=None``.

    A GeoDataFrame/GeoSeries is reprojected to 4326 (and dissolved to one
    geometry); a bare shapely geometry is assumed to already be lon/lat; a
    ``(minlon, minlat, maxlon, maxlat)`` sequence becomes a box.
    """
    if aoi is None:
        return None
    import geopandas as gpd
    from shapely.geometry import box
    from shapely.geometry.base import BaseGeometry

    if isinstance(aoi, (gpd.GeoDataFrame, gpd.GeoSeries)):
        g = aoi.to_crs(4326) if (aoi.crs is not None and aoi.crs.to_epsg() != 4326) else aoi
        geom = g.geometry
        return geom.union_all() if hasattr(geom, "union_all") else geom.unary_union
    if isinstance(aoi, BaseGeometry):
        return aoi
    seq = np.asarray(aoi, dtype="float64").ravel()
    if seq.size == 4:
        return box(*(float(v) for v in seq))
    raise TypeError("aoi must be a GeoDataFrame/GeoSeries, a shapely geometry, "
                    "a (minlon, minlat, maxlon, maxlat) bbox, or None")


def _pick_label_column(df):
    for col in ("sta", "id", "station", "site"):
        if col in df.columns:
            return col
    return None


def plot_velocity_vectors(stations, aoi=None, buffer_km: float = 50.0, ax=None,
                          out_fn=None, ref_mm_yr: float = 20.0,
                          color_by_vertical: bool = False, title=None, *,
                          lon_col: str = "lon", lat_col: str = "lat",
                          vel_cols=("vel_e", "vel_n", "vel_u"), id_col=None,
                          n_labels: int = 5, vel_to_mm: float = 1000.0,
                          overlay_interp: bool = True, ref_frac: float = 0.12):
    """Horizontal velocity-vector (quiver) map for a GNSS station network.

    The horizontal companion to the sandbox NGL vertical-*rate* maps
    (``run_site_gnss.rate_figure``): each station's horizontal velocity
    (``vel_e``/``vel_n``) is drawn as a **true-azimuth** arrow at its (lon, lat)
    — the standard way a geodetic velocity field is shown. Built to consume a
    MIDAS-style station table (:func:`groundcontrol.sources.ngl.read_midas`, or
    any DataFrame with ``lon_col``/``lat_col`` + ``vel_cols``) — the same feed
    :mod:`groundcontrol.velocity` interpolates.

    Arrows use ``angles='uv'`` so East reads screen-horizontal and North
    screen-vertical (azimuth preserved regardless of the geographic aspect); the
    axes aspect is set to ``1/cos(lat)`` so station *positions* stay geographically
    correct. Velocities are assumed **m/yr** (the module/MIDAS convention) and
    displayed in **mm/yr** (``vel_to_mm=1000``); ``ref_mm_yr`` sizes the
    ``quiverkey`` reference arrow.

    ``aoi`` (any of): a GeoDataFrame/GeoSeries/shapely polygon (a gdf/GeoSeries is
    reprojected from its own CRS; a bare shapely geometry is assumed lon/lat) or a
    ``(minlon, minlat, maxlon, maxlat)`` bbox. Stations are drawn within the AOI
    bounds expanded by ``buffer_km``; those **inside** the AOI polygon are styled
    distinctly (solid crimson) from those only **within the buffer** (gray). With
    ``aoi=None`` every station is drawn and no inside/buffer split is made.

    When an AOI is given and ``overlay_interp`` is set, the network velocity
    interpolated at the AOI centroid
    (:func:`groundcontrol.velocity.interpolate_velocity`) is drawn as a distinct
    heavy green arrow annotated with magnitude / azimuth-from-north and its
    ``quality`` + horizontal-spread flag.

    ``color_by_vertical=True`` colors the arrows by ``vel_u`` (mm/yr) on a
    zero-centered ``RdYlBu`` scale (RED = subsidence — the same convention as
    :func:`plot_dh_map` and the rate maps) and adds a colorbar: the combined
    horizontal + vertical view. The ``n_labels`` stations nearest the AOI
    centroid are labeled.

    Returns the :class:`matplotlib.figure.Figure`; saves to ``out_fn`` if given.
    """
    import matplotlib.patheffects as pe

    lon = np.asarray(stations[lon_col], dtype="float64")
    lat = np.asarray(stations[lat_col], dtype="float64")
    ve = np.asarray(stations[vel_cols[0]], dtype="float64")
    vn = np.asarray(stations[vel_cols[1]], dtype="float64")
    vu = (np.asarray(stations[vel_cols[2]], dtype="float64")
          if len(vel_cols) > 2 and vel_cols[2] in stations else np.full(len(lon), np.nan))
    id_col = id_col or _pick_label_column(stations)
    ids = (stations[id_col].astype("string").to_numpy()
           if id_col is not None else np.array([""] * len(lon)))

    finite = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(ve) & np.isfinite(vn)

    poly = _resolve_aoi_polygon(aoi)
    if poly is not None:
        minx, miny, maxx, maxy = poly.bounds
        mean_lat = 0.5 * (miny + maxy)
        dlat = buffer_km / 111.0
        dlon = buffer_km / (111.0 * max(np.cos(np.radians(mean_lat)), 0.1))
        bx0, by0, bx1, by1 = minx - dlon, miny - dlat, maxx + dlon, maxy + dlat
        clon, clat = float(poly.centroid.x), float(poly.centroid.y)
        sel = finite & (lon >= bx0) & (lon <= bx1) & (lat >= by0) & (lat <= by1)
    else:
        sel = finite.copy()
        if sel.any():
            bx0, bx1 = float(lon[sel].min()), float(lon[sel].max())
            by0, by1 = float(lat[sel].min()), float(lat[sel].max())
            pad_x = 0.05 * (bx1 - bx0 or 1.0)
            pad_y = 0.05 * (by1 - by0 or 1.0)
            bx0, bx1, by0, by1 = bx0 - pad_x, bx1 + pad_x, by0 - pad_y, by1 + pad_y
        else:
            bx0, by0, bx1, by1 = -180.0, -90.0, 180.0, 90.0
        clon, clat = float(np.mean(lon[sel])) if sel.any() else 0.0, \
            float(np.mean(lat[sel])) if sel.any() else 0.0
    mean_lat = 0.5 * (by0 + by1)

    # inside-AOI vs within-buffer classification (in the drawn selection)
    if poly is not None and sel.any():
        import geopandas as gpd
        pts = gpd.GeoSeries(gpd.points_from_xy(lon[sel], lat[sel]), crs=4326)
        inside_sel = pts.within(poly).to_numpy()
    else:
        inside_sel = np.ones(int(sel.sum()), dtype=bool)  # aoi=None: all "primary"
    inside = np.zeros(len(lon), dtype=bool)
    inside[np.flatnonzero(sel)[inside_sel]] = True
    buffered = sel & ~inside

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 9))
    fig = ax.figure
    ax.set_aspect(1.0 / max(np.cos(np.radians(mean_lat)), 0.1))

    if poly is not None:
        import geopandas as gpd
        gpd.GeoSeries([poly], crs=4326).boundary.plot(
            ax=ax, color="0.25", lw=1.4, zorder=1)

    scale = ref_mm_yr / ref_frac  # scale_units='width': ref arrow = ref_frac of width
    qkw = dict(angles="uv", scale_units="width", scale=scale, width=0.0038,
               headwidth=3.4, headlength=4.2, headaxislength=3.6, minshaft=1.4,
               pivot="tail", zorder=3)
    halo = [pe.withStroke(linewidth=2.0, foreground="white")]

    q_ref = None
    cmap = plt.get_cmap("RdYlBu")
    if color_by_vertical:
        vu_mm = vu * vel_to_mm
        vals = vu_mm[sel]
        lim = float(np.nanpercentile(np.abs(vals), 98)) if np.isfinite(vals).any() else 1.0
        lim = max(lim, 0.5)
        norm = plt.Normalize(-lim, lim)
        q_ref = ax.quiver(lon[sel], lat[sel], ve[sel] * vel_to_mm, vn[sel] * vel_to_mm,
                          np.where(np.isfinite(vals), vals, 0.0), cmap=cmap, norm=norm,
                          **qkw)
        # rings mark stations inside the AOI (color already spoken for by vel_u)
        if inside.any():
            ax.scatter(lon[inside], lat[inside], s=46, facecolors="none",
                       edgecolors="k", linewidths=1.1, zorder=4)
        fig.colorbar(q_ref, ax=ax, shrink=0.72, pad=0.02,
                     label="vertical velocity vel_u (mm/yr)  [RED = SUBSIDENCE]")
    else:
        if buffered.any():
            qb = ax.quiver(lon[buffered], lat[buffered], ve[buffered] * vel_to_mm,
                           vn[buffered] * vel_to_mm, color="0.55", alpha=0.75, **qkw)
            ax.scatter(lon[buffered], lat[buffered], s=9, c="0.55", zorder=2)
            q_ref = qb
        if inside.any():
            qi = ax.quiver(lon[inside], lat[inside], ve[inside] * vel_to_mm,
                           vn[inside] * vel_to_mm, color="crimson", alpha=0.95, **qkw)
            ax.scatter(lon[inside], lat[inside], s=16, c="crimson",
                       edgecolors="k", linewidths=0.4, zorder=4)
            q_ref = qi

    if q_ref is not None:
        ax.quiverkey(q_ref, 0.87, 0.07, ref_mm_yr, f"{ref_mm_yr:g} mm/yr",
                     labelpos="N", coordinates="axes", color="k",
                     fontproperties={"size": 8})

    # optional interpolated AOI-centroid velocity (a distinct heavy arrow)
    if poly is not None and overlay_interp:
        from groundcontrol.velocity import interpolate_velocity
        res = interpolate_velocity(clon, clat, stations, lon_col=lon_col,
                                   lat_col=lat_col, vel_cols=vel_cols).iloc[0]
        vei, vni = res["vel_e"], res["vel_n"]
        if np.isfinite(vei) and np.isfinite(vni):
            ui, vi = vei * vel_to_mm, vni * vel_to_mm
            mag = float(np.hypot(ui, vi))
            az = float(np.degrees(np.arctan2(ui, vi)) % 360.0)  # from north, cw
            ax.quiver(clon, clat, ui, vi, color="tab:green", edgecolor="k",
                      linewidth=0.6, **{**qkw, "width": 0.007, "zorder": 6})
            ax.scatter([clon], [clat], marker="*", s=210, c="tab:green",
                       edgecolors="k", linewidths=0.6, zorder=7)
            sh = res.get("vel_spread_h", np.nan)
            sh_mm = sh * vel_to_mm if np.isfinite(sh) else np.nan
            ann = (f"AOI interp: {mag:.1f} mm/yr @ {az:.0f}°N\n"
                   f"{res['quality']} (spread_h "
                   f"{'nan' if not np.isfinite(sh_mm) else f'{sh_mm:.1f}'} mm/yr, "
                   f"n={int(res['n_stations_used'])})")
            ax.annotate(ann, (clon, clat), xytext=(9, -14),
                        textcoords="offset points", fontsize=7.6, color="darkgreen",
                        fontweight="bold", path_effects=halo, zorder=8)

    # label the nearest few stations to the AOI centroid
    if n_labels and sel.any():
        idx = np.flatnonzero(sel)
        cx = (lon[idx] - clon) * np.cos(np.radians(mean_lat))
        cy = lat[idx] - clat
        order = idx[np.argsort(cx * cx + cy * cy)[:int(n_labels)]]
        for i in order:
            ax.annotate(str(ids[i]), (lon[i], lat[i]), xytext=(4, 4),
                        textcoords="offset points", fontsize=7,
                        path_effects=halo, zorder=5)

    ax.set_xlim(bx0, bx1)
    ax.set_ylim(by0, by1)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # inside/buffer legend (skipped when color encodes vel_u)
    if not color_by_vertical and poly is not None:
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], color="crimson", lw=2, label="inside AOI"),
                   Line2D([0], [0], color="0.55", lw=2, label=f"within {buffer_km:g} km")]
        if overlay_interp:
            handles.append(Line2D([0], [0], color="tab:green", lw=2.5,
                                  label="AOI interp (centroid)"))
        ax.legend(handles=handles, fontsize=7.5, loc="upper left", framealpha=0.85)

    n_in = int(inside.sum())
    n_buf = int(buffered.sum())
    if title is None:
        title = "MIDAS horizontal velocities"
    ax.set_title(f"{title}\n{n_in} inside AOI + {n_buf} within {buffer_km:g} km "
                 f"buffer  |  ref {ref_mm_yr:g} mm/yr", fontsize=10)

    fig.tight_layout()
    if out_fn:
        fig.savefig(out_fn, dpi=150, bbox_inches="tight")
    return fig


def nice_scale_length(span: float) -> float:
    """A round 1/2/5 x 10^k length covering roughly 1/5 of ``span``."""
    target = span / 5.0
    k = np.floor(np.log10(target))
    base = target / 10.0**k
    nice = 1.0 if base < 1.5 else (2.0 if base < 3.5 else (5.0 if base < 7.5 else 10.0))
    return float(nice * 10.0**k)


def add_scalebar(ax, length: float | None = None, label: str | None = None,
                 loc: str = "lower right", color: str = "k"):
    """Add an anchored scalebar in data units (meters for projected CRSs).

    Intended companion to :func:`plot_dh_map` when axis tick labels are
    dropped for map-style panels. ``length=None`` picks a round 1/2/5x10^k
    value spanning ~1/5 of the current x-range; the default label renders
    km above 1000 m. Returns the added artist.
    """
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    x0, x1 = ax.get_xlim()
    span = abs(x1 - x0)
    if length is None:
        length = nice_scale_length(span)
    if label is None:
        label = f"{length / 1000.0:g} km" if length >= 1000 else f"{length:g} m"
    bar = AnchoredSizeBar(ax.transData, length, label, loc,
                          pad=0.4, sep=4, borderpad=0.6, frameon=True,
                          size_vertical=span / 300.0, color=color)
    bar.patch.set_alpha(0.7)
    ax.add_artist(bar)
    return bar
