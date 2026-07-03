"""Light tests for the plot helpers (hillshade + plot_dh_map). All offline."""

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd
import numpy as np
import pytest

from groundcontrol.plot import hillshade, plot_dh_map


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
