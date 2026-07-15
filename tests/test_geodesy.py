"""groundcontrol.geodesy — the general-purpose subset consolidated from
lidar_tools/geodesy.py (docs/consolidation_geodesy.md)."""

import shutil
import sys
from pathlib import Path

import pytest
from pyproj import CRS

from groundcontrol.crs import is_dynamic_frame
from groundcontrol.geodesy import (
    OUTPUT_DATUM_BUILDERS,
    build_utm_g2139_3d,
    build_utm_itrf2020_3d,
    build_utm_nad83_2011_3d,
    build_utm_target,
    epoch_pinned_pipeline,
    geographic_base_epsg,
    geoid_grid_hint,
    library_versions,
    navd88_offset,
    preflight_vertical_transform,
    utm_zone_label,
    write_crs_file,
)

_HAS_PROJINFO = (Path(sys.executable).parent / "projinfo").exists() or \
    shutil.which("projinfo") is not None


# ---------------------------------------------------------------------------
# zone labels + UTM builders
# ---------------------------------------------------------------------------

def test_utm_zone_label():
    assert utm_zone_label(32610) == "10N"
    assert utm_zone_label(32719) == "19S"
    with pytest.raises(ValueError, match="not a WGS84 UTM"):
        utm_zone_label(4326)


def test_utm_builders_are_3d_on_the_right_datum():
    crs = build_utm_nad83_2011_3d(32610)
    assert len(crs.axis_info) == 3
    assert "NAD83(2011)" in crs.name and "10N" in crs.name
    assert not is_dynamic_frame(crs)  # static realization
    crs = build_utm_itrf2020_3d(32611)
    assert is_dynamic_frame(crs)  # dynamic — needs an epoch stamp
    assert "ITRF2020" in crs.name


def test_southern_hemisphere_false_northing():
    """The regression the programmatic builder fixed: a text-substituted WKT
    template silently kept false northing 0 for southern zones."""
    crs = build_utm_g2139_3d(32719)
    params = {p.name: p.value for p in crs.coordinate_operation.params}
    assert params["False northing"] == 10000000.0


def test_build_utm_target_names_and_errors():
    crs, fn = build_utm_target(32610, output_datum="nad83_2011")
    assert fn == "UTM_10N_NAD83_2011_3D.wkt"
    assert "NAD83(2011)" in crs.name
    with pytest.raises(ValueError, match="Unknown output_datum"):
        build_utm_target(32610, output_datum="nad27")
    assert set(OUTPUT_DATUM_BUILDERS) == {
        "wgs84_g2139", "nad83_2011", "wgs84_g1674",
        "itrf2020", "itrf2008", "itrf2014"}


# ---------------------------------------------------------------------------
# base-datum extraction + geoid hints
# ---------------------------------------------------------------------------

def test_geographic_base_epsg():
    assert geographic_base_epsg(26910) == 4269   # NAD83 / UTM 10N
    assert geographic_base_epsg(6339) == 6318    # NAD83(2011) / UTM 10N
    assert geographic_base_epsg("6318") == 6318  # numeric string passthrough
    with pytest.raises(ValueError, match="not a supported NAD83-family"):
        geographic_base_epsg(32610)  # WGS84 UTM


def test_geoid_grid_hint():
    assert geoid_grid_hint("GEOID12B") == "g2012b"
    assert geoid_grid_hint("geoid 18") == "g2018"  # case/space normalized
    assert geoid_grid_hint(None) is None
    assert geoid_grid_hint("EGM2008") is None


# ---------------------------------------------------------------------------
# preflight (offline-safe cases; grid-download path needs network)
# ---------------------------------------------------------------------------

def test_preflight_projection_only_returns_provenance():
    rec = preflight_vertical_transform("EPSG:4326", "EPSG:32610", download=False)
    assert rec["source_crs"] == "WGS 84"
    assert rec["grids"] == []
    assert "proj_pipeline" in rec and "accuracy_m" in rec


def test_preflight_area_of_use_containment_rejects_out_of_area():
    # NAD83->WGS84 Helmert ops are regional (North America); an AOI reaching
    # Europe intersects the area of use but is not contained by it
    with pytest.raises(RuntimeError, match="does not contain the AOI"):
        preflight_vertical_transform(
            "EPSG:4269", "EPSG:4326", download=False,
            aoi_bounds=(-130.0, 20.0, 10.0, 60.0))
    # positive control: a contained AOI selects a region-appropriate op
    rec = preflight_vertical_transform(
        "EPSG:4269", "EPSG:4326", download=False,
        aoi_bounds=(-115.5, 35.9, -114.9, 36.4))
    assert rec["area_of_use"] is not None


def test_preflight_prefer_grids_fails_loud_when_absent():
    with pytest.raises(RuntimeError, match="matching 'g2018'"):
        preflight_vertical_transform(
            "EPSG:4326", "EPSG:32610", download=False, prefer_grids="g2018")


@pytest.mark.network
def test_preflight_geoid_route_lists_grids():
    rec = preflight_vertical_transform(
        "EPSG:6318+5703", "EPSG:6319",
        aoi_bounds=(-115.5, 35.9, -114.9, 36.4))
    assert any("g2018" in g for g in rec["grids"])


@pytest.mark.network
def test_navd88_offset_conus_magnitude():
    n = navd88_offset(-115.1, 36.1)
    assert -40.0 < n < -15.0  # geoid undulation, CONUS


# ---------------------------------------------------------------------------
# epoch-pinned pipeline (needs the projinfo CLI from a PROJ build)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_PROJINFO, reason="projinfo CLI not on PATH")
def test_epoch_pinned_pipeline_bakes_epoch_and_helmert():
    # --t_epoch applies to the (dynamic) TARGET frame: static NAD83(2011) 3D
    # source -> dynamic ITRF2014 3D target at epoch 2020.0
    pipe = epoch_pinned_pipeline(
        "EPSG:6319", "EPSG:7912", 2020.0,
        require_substrings=["+proj=helmert"])
    assert pipe.startswith("+proj=pipeline")
    assert "+proj=set +v_4=2020" in pipe


@pytest.mark.skipif(not _HAS_PROJINFO, reason="projinfo CLI not on PATH")
def test_epoch_pinned_pipeline_missing_required_substring_raises():
    with pytest.raises(RuntimeError, match="lacks required component"):
        epoch_pinned_pipeline(
            "EPSG:6319", "EPSG:7912", 2020.0,
            require_substrings=["vgridshift"])  # no geoid in this route


# ---------------------------------------------------------------------------
# provenance utilities
# ---------------------------------------------------------------------------

def test_write_crs_file_roundtrip(tmp_path):
    out = write_crs_file(CRS.from_epsg(6318), tmp_path / "crs.wkt")
    text = Path(out).read_text()
    assert text.startswith("GEOGCRS")
    assert "NAD83(2011)" in text
    assert CRS.from_wkt(text).to_epsg() == 6318


def test_library_versions_gdal_free():
    v = library_versions()
    assert {"proj", "pyproj", "proj_data_dir"} <= set(v)
    # gdal key is optional — only when the env happens to have the bindings
