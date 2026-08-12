"""Centralized figure infrastructure: cpt_rainbow, multidirectional
hillshade, matplotlib-scalebar add_scalebar (env figures.md house rules)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from groundcontrol.figures import cpt_rainbow
from groundcontrol.plot import MULTIDIR_AZIMUTHS, add_scalebar, hillshade


class TestCptRainbow:
    def test_endpoints_match_bundled_cpt(self):
        # data/rainbow.cpt: first entry 144,0,111 @ z=1; final trailing
        # entry 105,0,0 @ z=254
        c = cpt_rainbow()
        assert c.name == "cpt_rainbow"
        lo = np.array(c(0.0))[:3] * 255
        hi = np.array(c(1.0))[:3] * 255
        np.testing.assert_allclose(lo, (144, 0, 111), atol=1)
        np.testing.assert_allclose(hi, (105, 0, 0), atol=1)

    def test_reverse_mirrors(self):
        # sample at LUT cell centers: complementary cells quantize exactly,
        # so .reversed() mirrors with no aliasing
        c, r = cpt_rainbow(), cpt_rainbow(reverse=True)
        x = (np.arange(256) + 0.5) / 256
        np.testing.assert_allclose(c(x), r(1 - x), atol=1e-12)
        assert r.name == "cpt_rainbow_r"

    def test_cached(self):
        assert cpt_rainbow() is cpt_rainbow()

    def test_matches_imview_when_available(self):
        # fidelity vs the imview original this vendors (same cpt file);
        # skipped where imview is not installed (e.g. CI)
        gmt = pytest.importorskip("imview.lib.gmtColormap")
        ours, theirs = cpt_rainbow(), gmt.get_rainbow()
        x = np.linspace(0, 1, 256)
        np.testing.assert_allclose(ours(x), theirs(x), atol=1 / 255)


class TestMultidirectionalHillshade:
    def _ridge(self):
        # east-west ridge: strong single-azimuth asymmetry
        y = np.linspace(-1, 1, 64)
        return 50 * np.exp(-(y[:, None] ** 2) / 0.1) * np.ones((64, 64))

    def test_range_and_shape(self):
        hs = hillshade(self._ridge(), multidirectional=True)
        assert hs.shape == (64, 64)
        assert np.nanmin(hs) >= 0 and np.nanmax(hs) <= 1

    def test_is_mean_of_lamps(self):
        z = self._ridge()
        expect = np.nanmean([hillshade(z, azdeg=a)
                             for a in MULTIDIR_AZIMUTHS], axis=0)
        np.testing.assert_allclose(hillshade(z, multidirectional=True), expect)

    def test_nan_propagates(self):
        z = self._ridge()
        z[10, 10] = np.nan
        hs = hillshade(z, multidirectional=True)
        assert np.isnan(hs[10, 10])


class TestAddScalebar:
    def test_fixed_length_and_label(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 500)
        bar = add_scalebar(ax, length=50, label="50 m")
        assert bar in ax.artists
        fig.canvas.draw()  # smoke: renders without error
        plt.close(fig)

    def test_auto_length_promotes_km(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 12000)
        bar = add_scalebar(ax)
        assert bar.fixed_units == "km"  # never "2000.0 m"
        assert bar.fixed_value in (1, 2, 2.5, 5)
        plt.close(fig)
