"""Light tests for the plot helpers (hillshade + plot_dh_map). All offline."""

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd
import numpy as np
import pytest

import pandas as pd

from groundcontrol.plot import hillshade, plot_dh_map, plot_velocity_vectors


# ---------------------------------------------------------------------------
# hillshade
# ---------------------------------------------------------------------------

def test_hillshade_flat_is_constant_cos_zenith():
    hs = hillshade(np.full((10, 12), 100.0), altdeg=45.0)
    assert hs.shape == (10, 12)
    np.testing.assert_allclose(hs, np.cos(np.radians(45.0)), rtol=1e-12)


def test_hillshade_range_and_nan_propagation():
    rng = np.random.default_rng(3)
    z = rng.normal(0, 5, (20, 20))
    z[5, 5] = np.nan
    hs = hillshade(z, dx=2.0, dy=2.0)
    finite = hs[np.isfinite(hs)]
    assert finite.min() >= 0.0 and finite.max() <= 1.0
    assert np.isnan(hs[5, 5])  # NaN input propagates (transparent basemap hole)


def test_hillshade_orientation_nw_light():
    """With azdeg=315 (NW light), a NW-facing slope is brighter than SE-facing."""
    j, i = np.meshgrid(np.arange(30), np.arange(30))  # row 0 = north
    z_nw_facing = (j + i).astype(float)   # z increases to SE -> downhill faces NW
    z_se_facing = -z_nw_facing
    assert hillshade(z_nw_facing).mean() > hillshade(z_se_facing).mean()


# ---------------------------------------------------------------------------
# plot_dh_map
# ---------------------------------------------------------------------------

def _gdf(dh):
    n = len(dh)
    return gpd.GeoDataFrame(
        {"dh": dh}, geometry=gpd.points_from_xy(np.arange(n), np.arange(n)),
        crs="EPSG:32611")


def test_plot_dh_map_default_clim_symmetric_3nmad():
    gdf = _gdf([0.1, -0.1, 0.2, -0.2, 0.05])
    sc = plot_dh_map(gdf, "dh")
    lo, hi = sc.get_clim()
    assert lo == pytest.approx(-hi)  # centered on zero
    # 3 * NMAD of the values: median 0.05, |a-med| = [.05,.15,.15,.25,0] -> MAD .15
    assert hi == pytest.approx(3 * 1.4826 * 0.15)
    assert sc.get_cmap().name == "RdYlBu"


def test_plot_dh_map_hillshade_basemap_and_explicit_clim():
    import matplotlib.pyplot as plt

    gdf = _gdf([0.5, -0.5, np.nan])
    hs = hillshade(np.ones((5, 5)))
    fig, ax = plt.subplots()
    sc = plot_dh_map(gdf, "dh", hs=hs, hs_extent=(0, 4, 0, 4), ax=ax,
                     clim=(-1, 1), title="t")
    assert sc.axes is ax
    assert len(ax.images) == 1  # the hillshade layer
    assert sc.get_clim() == (-1, 1)
    assert ax.get_title() == "t"
    plt.close(fig)


# ---------------------------------------------------------------------------
# plot_velocity_vectors (horizontal quiver map) — light smoke tests (headless)
# ---------------------------------------------------------------------------

def _stations():
    """A small synthetic MIDAS-style network (m/yr) around a Las-Vegas-ish
    centroid: coherent SW motion, a few mm/yr of vertical scatter."""
    rng = np.random.default_rng(0)
    lon0, lat0 = -115.15, 36.10
    lon = lon0 + rng.uniform(-0.4, 0.4, 20)
    lat = lat0 + rng.uniform(-0.4, 0.4, 20)
    return pd.DataFrame({
        "sta": [f"S{i:02d}" for i in range(20)],
        "lon": lon, "lat": lat,
        "vel_e": np.full(20, -0.013) + rng.normal(0, 0.0005, 20),  # ~ -13 mm/yr E
        "vel_n": np.full(20, -0.009) + rng.normal(0, 0.0005, 20),  # ~ -9 mm/yr N
        "vel_u": rng.normal(0, 0.002, 20),                          # +/- few mm/yr
    })


def test_plot_velocity_vectors_no_aoi_returns_fig(tmp_path):
    import matplotlib.pyplot as plt

    out = tmp_path / "vv_noaoi.png"
    fig = plot_velocity_vectors(_stations(), out_fn=out)
    assert fig is not None
    assert out.exists()
    ax = fig.axes[0]
    assert any(isinstance(c, matplotlib.quiver.Quiver) for c in ax.collections)
    plt.close(fig)


def test_plot_velocity_vectors_bbox_aoi_and_overlay():
    import matplotlib.pyplot as plt

    # bbox AOI (minlon, minlat, maxlon, maxlat) with stations both in and out
    fig = plot_velocity_vectors(_stations(), aoi=(-115.3, 35.95, -115.0, 36.25),
                                buffer_km=40, title="synthetic")
    ax = fig.axes[0]
    assert "inside AOI" in ax.get_title()
    # quivers for inside + buffer + (possibly) the interp overlay
    assert sum(isinstance(c, matplotlib.quiver.Quiver) for c in ax.collections) >= 1
    plt.close(fig)


def test_plot_velocity_vectors_color_by_vertical_adds_colorbar():
    import matplotlib.pyplot as plt

    fig = plot_velocity_vectors(_stations(), aoi=(-115.3, 35.95, -115.0, 36.25),
                                color_by_vertical=True)
    # a colorbar axes is added alongside the map axes
    assert len(fig.axes) >= 2
    plt.close(fig)


# ---------------------------------------------------------------------------
# scalebar
# ---------------------------------------------------------------------------

def test_nice_scale_length():
    from groundcontrol.plot import nice_scale_length

    assert nice_scale_length(10_000.0) == 2000.0   # 1/5 = 2000 -> 2 km
    assert nice_scale_length(52_000.0) == 10_000.0  # 1/5 = 10400 -> 10 km
    assert nice_scale_length(400.0) == 100.0        # 1/5 = 80 -> 100 m
    assert nice_scale_length(30.0) == 5.0           # 1/5 = 6 -> 5 m


def test_add_scalebar_artist_added():
    import matplotlib.pyplot as plt

    from groundcontrol.plot import add_scalebar

    fig, ax = plt.subplots()
    ax.set_xlim(0, 10_000)
    ax.set_ylim(0, 10_000)
    bar = add_scalebar(ax)
    assert bar in ax.artists
    fig.canvas.draw()  # renders without error
    plt.close(fig)
