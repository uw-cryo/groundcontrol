"""crs.py unit tests — decyear per the hardened B9 spec + fail-loud transformer selection.

All offline. The geoid/Helmert fixture tests (docs/crs_implementation.md §8)
land with the committed clipped-grid fixtures.
"""

import numpy as np
import pandas as pd
import pytest

from groundcontrol.crs import NoTransformPathError, decyear, decyear_inv, get_transformer


# ---------------------------------------------------------------------------
# decyear known values (exact, per plan B9)
# ---------------------------------------------------------------------------

def test_decyear_year_start_exact():
    assert decyear("2010-01-01T00:00:00") == 2010.0


def test_decyear_midyear_nonleap_exact():
    # 2010: 365 d; Jan 1 00:00 -> Jul 2 12:00 = 182.5 d = exactly half
    assert decyear("2010-07-02T12:00:00") == 2010.5


def test_decyear_midyear_leap_exact():
    # 2020: 366 d; Jan 1 00:00 -> Jul 2 00:00 = 183 d = exactly half
    assert decyear("2020-07-02T00:00:00") == 2020.5


def test_decyear_polymorphic_scalar_types():
    v = decyear("2010-07-02T12:00:00")
    assert decyear(pd.Timestamp("2010-07-02T12:00:00")) == v
    import datetime
    assert decyear(datetime.datetime(2010, 7, 2, 12)) == v


def test_decyear_series_preserves_index():
    s = pd.Series(pd.to_datetime(["2010-01-01", "2020-07-02"]), index=[10, 20])
    out = decyear(s)
    assert isinstance(out, pd.Series)
    assert list(out.index) == [10, 20]
    assert out.loc[10] == 2010.0
    assert out.loc[20] == 2020.5


def test_decyear_tz_aware_converts_to_utc():
    # 2010-07-02T05:00 in UTC-7 == 2010-07-02T12:00 UTC == exactly 2010.5
    aware = pd.Timestamp("2010-07-02T05:00:00-07:00")
    assert decyear(aware) == 2010.5


def test_decyear_nat_to_nan():
    assert np.isnan(decyear(pd.NaT))
    out = decyear(pd.Series([pd.Timestamp("2010-01-01"), pd.NaT]))
    assert out.iloc[0] == 2010.0 and np.isnan(out.iloc[1])


def test_decyear_inv_roundtrip_subsecond():
    stamps = pd.to_datetime(
        ["2010-01-01T00:00:00", "2010-07-02T12:34:56", "2020-12-31T23:59:59", "1996-02-29T06:00:00"]
    )
    rt = decyear_inv(decyear(pd.Series(stamps)))
    dt = (pd.Series(rt.to_numpy()) - pd.Series(stamps)).abs()
    assert (dt < pd.Timedelta(seconds=1)).all()


def test_decyear_inv_scalar_and_nan():
    assert abs(decyear_inv(2010.5) - pd.Timestamp("2010-07-02T12:00:00")) < pd.Timedelta(seconds=1)
    assert pd.isna(decyear_inv(float("nan")))


# ---------------------------------------------------------------------------
# get_transformer — fail-loud selection
# ---------------------------------------------------------------------------

def test_get_transformer_selects_real_operation():
    # grid-free path, always available: geographic -> UTM
    t = get_transformer("EPSG:4326", "EPSG:32612")
    # a TransformerGroup member exposes a real pipeline (not the placeholder)
    assert "unavailable" not in (t.definition or "unavailable")
    x, y = t.transform(-111.7, 32.8)
    assert 380_000 < x < 440_000 and 3_600_000 < y < 3_660_000


def test_get_transformer_aoi_takes_degrees():
    t = get_transformer("EPSG:4326", "EPSG:32612", aoi_bounds_4326=(-112.0, 32.6, -111.5, 33.0))
    assert t is not None


def test_get_transformer_raises_no_path_not_indexerror(monkeypatch):
    """Empty TransformerGroup must raise NoTransformPathError with diagnostics (B6)."""
    import groundcontrol.crs as gc_crs

    class _FakeTG:
        transformers = []
        unavailable_operations = []
        best_available = False

        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr(gc_crs, "TransformerGroup", _FakeTG)
    with pytest.raises(NoTransformPathError) as ei:
        get_transformer("EPSG:4326", "EPSG:32612")
    assert "allow_ballpark=False" in str(ei.value)
    assert "unavailable_operations" in ei.value.diagnostics
