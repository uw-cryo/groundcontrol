"""crs.compare_frames / assert_same_frame / vertical_kind, and the ensemble refusal
in get_transformer / transform_points. All offline (no grids: every transform here
is a same-datum or null-op pair)."""

import geopandas as gpd
import numpy as np
import pyproj
import pytest

from groundcontrol.crs import (
    assert_same_frame,
    compare_frames,
    get_transformer,
    grid_signature,
    horizontal_2d,
    transform_points,
    vertical_kind,
)

NAD83_UTM11 = "EPSG:6340"
NAD83_UTM11_3D = pyproj.CRS("EPSG:6340").to_3d()
NAD83_UTM11_NAVD88 = "EPSG:6340+5703"
WGS84_UTM11 = "EPSG:32611"                      # ensemble
ITRF2020_UTM11 = pyproj.CRS.from_user_input(
    'PROJCRS["ITRF2020 / UTM zone 11N",BASEGEOGCRS["ITRF2020",DYNAMIC[FRAMEEPOCH[2015]],'
    'DATUM["International Terrestrial Reference Frame 2020",ELLIPSOID["GRS 1980",6378137,'
    '298.257222101]],ID["EPSG",9990]],CONVERSION["UTM zone 11N",METHOD["Transverse Mercator"],'
    'PARAMETER["Latitude of natural origin",0],PARAMETER["Longitude of natural origin",-117],'
    'PARAMETER["Scale factor at natural origin",0.9996],PARAMETER["False easting",500000],'
    'PARAMETER["False northing",0]],CS[Cartesian,2],AXIS["(E)",east],AXIS["(N)",north],'
    'LENGTHUNIT["metre",1]]')


# ---------------------------------------------------------------------------
# vertical_kind
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("crs,kind", [
    (NAD83_UTM11, "undeclared"),
    ("EPSG:6318", "undeclared"),
    (NAD83_UTM11_3D, "ellipsoidal"),
    ("EPSG:6319", "ellipsoidal"),
    ("EPSG:4979", "ellipsoidal"),
    (NAD83_UTM11_NAVD88, "gravity"),
    ("EPSG:6318+5703", "gravity"),
    ("EPSG:32645+3855", "gravity"),              # COP30-style EGM2008
])
def test_vertical_kind(crs, kind):
    assert vertical_kind(crs) == kind


def test_three_axes_without_a_height_axis_are_undeclared():
    # Copilot on PR #35: axis COUNT is not a height declaration
    assert vertical_kind("EPSG:4978") == "undeclared"                 # geocentric XYZ
    time_wkt = ('COMPOUNDCRS["UTM12N + time",' + pyproj.CRS("EPSG:32612").to_wkt() +
                ',TIMECRS["Time",TDATUM["epoch",CALENDAR["proleptic Gregorian"],'
                'TIMEORIGIN[2000-01-01]],CS[TemporalDateTime,1],AXIS["time (T)",future]]]')
    assert vertical_kind(time_wkt) == "undeclared"                    # horizontal + time
    assert compare_frames(time_wkt, "EPSG:32612+5703").vertical_conflict is False


def test_two_different_depth_datums_conflict():
    # Copilot on PR #35: depth kinds were falling through as "same frame"
    a, b = "EPSG:6340+5612", "EPSG:6340+5715"                       # Baltic 1977 depth vs MSL depth
    assert vertical_kind(a) == vertical_kind(b) == "depth"
    assert compare_frames(a, b).vertical_conflict is True
    assert compare_frames(a, a).vertical_conflict is False


def test_vertical_kind_survives_wkt1_round_trip():
    # GDAL's WKT1 drops the 'Gravity-related height' axis name; the CRS name still says NAVD88
    wkt1 = pyproj.CRS(NAD83_UTM11_NAVD88).to_wkt(version="WKT1_GDAL")
    assert vertical_kind(wkt1) == "gravity"


# ---------------------------------------------------------------------------
# compare_frames
# ---------------------------------------------------------------------------
def test_same_crs_is_same_frame():
    c = compare_frames(NAD83_UTM11, NAD83_UTM11)
    assert c.horizontal_equal and c.same_grid and not c.vertical_conflict
    assert not c.ensemble_vs_realization and c.problems == [] and c.describe() == "same frame"


def test_datum_label_difference_on_one_grid_is_a_horizontal_problem():
    # the 2026-10-27 audit case: identical pixels labeled NAD83(2011) and ITRF2020 passed a
    # lattice gate and became a 1.36 m "co-registration offset"
    c = compare_frames(NAD83_UTM11, ITRF2020_UTM11)
    assert not c.horizontal_equal and c.same_grid
    assert c.problems == ["horizontal"] and not c.ensemble_vs_realization
    assert "same grid" in c.describe()


def test_ensemble_vs_realization_is_flagged_and_ensemble_vs_ensemble_is_not():
    c = compare_frames(WGS84_UTM11, NAD83_UTM11)
    assert c.ensemble_vs_realization and c.same_grid and "null operation" in c.describe()
    assert compare_frames(WGS84_UTM11, "EPSG:4326").ensemble_vs_realization is False
    assert compare_frames("EPSG:7912", "EPSG:9989").ensemble_vs_realization is False
    assert compare_frames("EPSG:4326", "EPSG:7912").ensemble_vs_realization is True


def test_vertical_conflict_only_when_both_sides_declare():
    c = compare_frames(NAD83_UTM11_NAVD88, NAD83_UTM11_3D)
    assert c.horizontal_equal and c.vertical_conflict and c.problems == ["vertical"]
    assert c.vertical_a == "gravity" and c.vertical_b == "ellipsoidal"
    for other in (NAD83_UTM11, NAD83_UTM11_3D, NAD83_UTM11_NAVD88):
        assert compare_frames(NAD83_UTM11, other).vertical_conflict is False   # 2D: undeclared


def test_two_different_gravity_verticals_conflict():
    c = compare_frames("EPSG:6318+5703", "EPSG:6318+7968")   # NAVD88 vs NGVD29
    assert c.vertical_conflict and c.vertical_a == c.vertical_b == "gravity"
    assert compare_frames("EPSG:6318+5703", "EPSG:6318+5703").vertical_conflict is False


def test_epoch_compared_only_when_both_given():
    assert compare_frames(NAD83_UTM11, NAD83_UTM11).epoch_equal is None
    assert compare_frames(NAD83_UTM11, NAD83_UTM11, epoch_a=2010.0).epoch_equal is None
    c = compare_frames(NAD83_UTM11, NAD83_UTM11, epoch_a=2010.0, epoch_b=2024.5)
    assert c.epoch_equal is False and c.problems == ["epoch"]
    assert compare_frames(NAD83_UTM11, NAD83_UTM11, epoch_a=2010.0, epoch_b=2010.0).problems == []


def test_same_frame_from_different_encodings_is_equal():
    # a .prj (WKT1_ESRI) of a geographic CRS is not pyproj-.equals its EPSG form
    esri = pyproj.CRS("EPSG:6318").to_wkt(version="WKT1_ESRI")
    assert compare_frames("EPSG:6318", esri).horizontal_equal
    assert compare_frames("EPSG:6319", esri).horizontal_equal           # 3D vs .prj 2D
    assert compare_frames("EPSG:6318+5703", esri).horizontal_equal      # compound member
    assert not compare_frames("EPSG:4269", esri).horizontal_equal       # NAD83(1986) is not 2011
    # a PROJ string is lossy and unregistered: not identified, so not equal (fail loud)
    with pytest.warns(UserWarning):
        assert not compare_frames("EPSG:6318", pyproj.CRS("EPSG:6318").to_proj4()).horizontal_equal


def test_nan_epoch_is_unknown_not_different():
    c = compare_frames(NAD83_UTM11, NAD83_UTM11, epoch_a=float("nan"), epoch_b=2010.0)
    assert c.epoch_equal is None and c.epoch_a is None and c.problems == []


def test_depth_name_stem_is_depth_even_when_the_axis_says_up():
    wkt = ('COMPD_CS["x",' + pyproj.CRS("EPSG:6340").to_wkt(version="WKT1_GDAL") +
           ',VERT_CS["MSL depth",VERT_DATUM["Mean Sea Level",2005],UNIT["metre",1],AXIS["Up",UP]]]')
    assert vertical_kind(wkt) == "depth"


def test_ensemble_target_remedy_names_the_target_side():
    with pytest.raises(ValueError, match="give the target as the realization"):
        get_transformer("EPSG:6318+5703", "EPSG:32611+5703")
    with pytest.raises(ValueError, match="declare the realization the source"):
        get_transformer("EPSG:32611", "EPSG:6340")


def test_public_helpers_match_the_sampler_private_names():
    from groundcontrol.sample import _grid_signature, _horizontal_2d
    assert _horizontal_2d is horizontal_2d and _grid_signature is grid_signature
    assert horizontal_2d(NAD83_UTM11_NAVD88).equals(pyproj.CRS(NAD83_UTM11))
    assert grid_signature(WGS84_UTM11) == grid_signature(NAD83_UTM11)
    assert grid_signature("EPSG:32610") != grid_signature(NAD83_UTM11)


# ---------------------------------------------------------------------------
# assert_same_frame
# ---------------------------------------------------------------------------
def test_assert_same_frame_raises_names_the_problem_and_allows_by_name(caplog):
    with pytest.raises(ValueError, match="vertical datums conflict"):
        assert_same_frame(NAD83_UTM11_NAVD88, NAD83_UTM11_3D, what="reference and product")
    with pytest.raises(ValueError, match=r"allow=\['vertical'\]"):
        assert_same_frame(NAD83_UTM11_NAVD88, NAD83_UTM11_3D)
    with caplog.at_level("WARNING", logger="groundcontrol.crs"):
        c = assert_same_frame(NAD83_UTM11_NAVD88, NAD83_UTM11_3D, allow=("vertical",))
    assert c.vertical_conflict and "allow=['vertical']" in caplog.text
    # allowing one problem does not allow another
    with pytest.raises(ValueError, match="horizontal frames differ"):
        assert_same_frame(NAD83_UTM11_NAVD88, ITRF2020_UTM11, allow=("vertical",))
    with pytest.raises(ValueError, match="coordinate epochs differ"):
        assert_same_frame(NAD83_UTM11, NAD83_UTM11, epoch_a=2010.0, epoch_b=2020.0)


# ---------------------------------------------------------------------------
# get_transformer / transform_points: the ensemble refusal
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("src,dst", [
    ("EPSG:32611", "EPSG:6340"),      # ensemble UTM -> NAD83(2011) UTM (same grid, 2 m null op)
    ("EPSG:4326", "EPSG:6318"),
    ("EPSG:7912", "EPSG:4326"),       # ITRF2014 -> ensemble
    ("EPSG:4979", "EPSG:9989"),
])
def test_get_transformer_refuses_ensemble_vs_realization(src, dst):
    with pytest.raises(ValueError, match="WGS 84 ensemble"):
        get_transformer(src, dst)
    with pytest.warns(UserWarning, match="allow_ensemble=True"):
        t = get_transformer(src, dst, allow_ensemble=True)
    assert t.accuracy is not None


@pytest.mark.parametrize("src,dst", [
    ("EPSG:4326", "EPSG:32611"),      # ensemble -> ensemble
    ("EPSG:6318", "EPSG:6340"),       # same realization
    ("EPSG:7912", "EPSG:9989"),       # realization -> realization (no grid needed)
    ("EPSG:9755", "EPSG:6319"),       # WGS 84 (G2139) -> NAD83(2011): a real member, not the umbrella
])
def test_get_transformer_passes_non_ensemble_pairs(src, dst):
    assert get_transformer(src, dst).accuracy is not None


def test_transform_points_refuses_ensemble_source_and_names_the_remedy():
    g = gpd.GeoDataFrame({"height": [100.0]}, geometry=gpd.points_from_xy([-115.1], [36.1]),
                         crs="EPSG:4979")
    with pytest.raises(ValueError, match="source_crs=EPSG:9989"):
        transform_points(g, "EPSG:6319", tt=2010.0)
    # the remedy: the points are really ITRF2020 (ICESat-2); declare it
    out = transform_points(g, "EPSG:6319", tt=2010.0, source_crs="EPSG:9989")
    assert np.isfinite(out["height"].iloc[0])
    # the escape: accept the null operation on the record
    with pytest.warns(UserWarning, match="allow_ensemble=True"):
        out2 = transform_points(g, "EPSG:6319", tt=2010.0, allow_ensemble=True)
    assert np.isfinite(out2["height"].iloc[0])
