"""crs.dataset_frame: a raster's CRS and GDAL coordinate epoch, read together.

rasterio exposes no coordinate epoch (verified through 1.5.1), so the epoch
comes through the GDAL Python bindings or the gdalinfo executable. CI (pip
wheels) has neither; those tests skip there and the 'unreadable' path is
exercised everywhere by disabling both readers."""

import shutil
import sys

import numpy as np
import pyproj
import pytest
import rasterio
from rasterio.transform import from_origin

from groundcontrol import crs as gc_crs
from groundcontrol.crs import DatasetFrame, EPOCH_SOURCES, compare_frames, dataset_frame

# a reader counts only when it can read epochs: the production gates (GDAL >= 3.4)
try:
    from osgeo import gdal
    HAVE_OSGEO = tuple(int(v) for v in gdal.__version__.split(".")[:2]) >= gc_crs._EPOCH_MIN_GDAL
except ImportError:
    HAVE_OSGEO = False
_gdalinfo = shutil.which("gdalinfo")
HAVE_GDALINFO = (_gdalinfo is not None
                 and (gc_crs._gdalinfo_version(_gdalinfo) or (0, 0)) >= gc_crs._EPOCH_MIN_GDAL)
ANY_READER = HAVE_OSGEO or HAVE_GDALINFO

NAD83_UTM11_3D = pyproj.CRS("EPSG:6340").to_3d()


def _write_tif(path, crs_input=None, epoch=None):
    kw = {"crs": pyproj.CRS.from_user_input(crs_input).to_wkt()} if crs_input else {}
    with rasterio.open(path, "w", driver="GTiff", width=3, height=3, count=1,
                       dtype="float32", transform=from_origin(500000, 4000000, 1, 1),
                       **kw) as dst:
        dst.write(np.zeros((1, 3, 3), np.float32))
    if epoch is not None:  # stamp the way a writer does: through GDAL
        _stamp_epoch(str(path), epoch)
    return str(path)


def _gdal_exceptions():
    """gdal.ExceptionMgr where it exists (GDAL >= 3.7); a no-op context before."""
    import contextlib
    from osgeo import gdal
    return gdal.ExceptionMgr(useExceptions=True) if hasattr(gdal, "ExceptionMgr") \
        else contextlib.nullcontext()


def _stamp_epoch(path, epoch):
    from osgeo import gdal
    with _gdal_exceptions():
        ds = gdal.Open(path, gdal.GA_Update)
        srs = ds.GetSpatialRef()
        srs.SetCoordinateEpoch(epoch)
        ds.SetSpatialRef(srs)
        ds = None


@pytest.fixture
def no_readers(monkeypatch):
    """Neither the GDAL bindings nor gdalinfo: the epoch cannot be checked."""
    monkeypatch.setitem(sys.modules, "osgeo", None)         # import raises ImportError
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: None)


def test_record_shape_and_sources():
    assert EPOCH_SOURCES == ("gdal", "gdalinfo", "undeclared", "unreadable")
    f = DatasetFrame(None, None, "unreadable", "x.tif")
    assert not f.epoch_checked
    assert DatasetFrame(None, None, "undeclared", "x.tif").epoch_checked
    with pytest.raises(AttributeError):  # frozen: a record, not a scratch dict
        f.epoch = 2010.0


def test_unreadable_when_no_reader_is_available(tmp_path, no_readers):
    p = _write_tif(tmp_path / "a.tif", NAD83_UTM11_3D)
    f = dataset_frame(p)
    assert f.epoch is None and f.epoch_source == "unreadable" and not f.epoch_checked
    # the CRS still comes through rasterio, in full (3D kept)
    assert compare_frames(f.crs, NAD83_UTM11_3D).horizontal_equal
    assert len(f.crs.axis_info) == 3


@pytest.mark.skipif(not ANY_READER, reason="needs the GDAL bindings or gdalinfo")
def test_unstamped_file_is_undeclared_not_unreadable(tmp_path):
    f = dataset_frame(_write_tif(tmp_path / "a.tif", "EPSG:6340"))
    assert f.epoch is None and f.epoch_source == "undeclared" and f.epoch_checked


@pytest.mark.skipif(not HAVE_OSGEO, reason="needs the GDAL bindings to stamp and read")
@pytest.mark.parametrize("epoch", [2010.0, 2023.5])
def test_stamped_epoch_reads_back_through_gdal(tmp_path, epoch):
    f = dataset_frame(_write_tif(tmp_path / "a.tif", NAD83_UTM11_3D, epoch=epoch))
    assert f.epoch == pytest.approx(epoch) and f.epoch_source == "gdal"
    assert compare_frames(f.crs, NAD83_UTM11_3D, epoch_a=f.epoch, epoch_b=epoch).epoch_equal


@pytest.mark.skipif(not (HAVE_OSGEO and HAVE_GDALINFO),
                    reason="needs the bindings to stamp and gdalinfo to read")
def test_stamped_epoch_reads_back_through_gdalinfo(tmp_path, monkeypatch):
    p = _write_tif(tmp_path / "a.tif", "EPSG:6340", epoch=2023.5)
    monkeypatch.setitem(sys.modules, "osgeo", None)  # bindings gone: fall through to the CLI
    f = dataset_frame(p)
    assert f.epoch == pytest.approx(2023.5) and f.epoch_source == "gdalinfo"
    # and the same file unstamped through the CLI is undeclared, not unreadable
    g = dataset_frame(_write_tif(tmp_path / "b.tif", "EPSG:6340"))
    assert g.epoch is None and g.epoch_source == "undeclared"


def test_two_products_differ_only_by_epoch(tmp_path):
    """The consumer pattern: the guard sees the stamped epochs, not None."""
    if not HAVE_OSGEO:
        pytest.skip("needs the GDAL bindings to stamp")
    a = dataset_frame(_write_tif(tmp_path / "a.tif", NAD83_UTM11_3D, epoch=2010.0))
    b = dataset_frame(_write_tif(tmp_path / "b.tif", NAD83_UTM11_3D, epoch=2023.5))
    cmp = compare_frames(a.crs, b.crs, epoch_a=a.epoch, epoch_b=b.epoch)
    assert cmp.problems == ["epoch"]


def test_file_without_crs(tmp_path):
    f = dataset_frame(_write_tif(tmp_path / "nocrs.tif"))
    assert f.crs is None and f.epoch is None
    assert f.epoch_source in ("undeclared", "unreadable")


def test_missing_file_raises(tmp_path):
    with pytest.raises(rasterio.errors.RasterioIOError):
        dataset_frame(tmp_path / "nope.tif")


def test_private_readers_return_none_when_absent(tmp_path, no_readers):
    p = _write_tif(tmp_path / "a.tif", "EPSG:6340")
    assert gc_crs._epoch_via_gdal(p) is None
    assert gc_crs._epoch_via_gdalinfo(p) is None


# --- the gdalinfo route with a canned executable: runs in CI, pins the version gate ---

class _Run:
    """subprocess.run stand-in: a gdalinfo of a given version and -json output."""
    def __init__(self, version_line, json_out, rc=0):
        self.version_line, self.json_out, self.rc = version_line, json_out, rc
        self.calls = []

    def __call__(self, args, **kw):
        self.calls.append(args)
        import subprocess
        if args[1] == "--version":
            return subprocess.CompletedProcess(args, 0, stdout=self.version_line, stderr="")
        return subprocess.CompletedProcess(args, self.rc, stdout=self.json_out, stderr="boom")


@pytest.fixture
def canned_gdalinfo(monkeypatch):
    import subprocess
    monkeypatch.setitem(sys.modules, "osgeo", None)
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: "/fake/bin/gdalinfo")

    def make(version_line, json_out, rc=0):
        run = _Run(version_line, json_out, rc)
        monkeypatch.setattr(subprocess, "run", run)
        return run
    return make


MODERN = 'GDAL 3.13.3 "Iowa City", released 2026/08/13\n'


def test_gdalinfo_route_reads_the_top_level_key(tmp_path, canned_gdalinfo):
    p = _write_tif(tmp_path / "a.tif", "EPSG:6340")
    run = canned_gdalinfo(MODERN, '{"coordinateSystem": {"wkt": "..."}, "coordinateEpoch": 2023.5}')
    f = dataset_frame(p)
    assert f.epoch == 2023.5 and f.epoch_source == "gdalinfo"
    assert run.calls[0][1] == "--version" and run.calls[1][1:] == ["-json", p]


def test_gdalinfo_route_without_key_is_undeclared_on_a_modern_gdal(tmp_path, canned_gdalinfo):
    canned_gdalinfo(MODERN, '{"coordinateSystem": {"wkt": "..."}}')
    f = dataset_frame(_write_tif(tmp_path / "a.tif", "EPSG:6340"))
    assert f.epoch is None and f.epoch_source == "undeclared" and f.epoch_checked


@pytest.mark.parametrize("version_line", [
    'GDAL 3.2.2, released 2021/03/05\n',   # pre-RFC 81: never emits the key
    'GDAL 3.0.4, released 2020/01/28\n',
    'something else entirely\n',           # unparseable: not trusted as a reader
])
def test_gdalinfo_route_refuses_to_read_with_an_old_or_unknown_gdal(tmp_path, canned_gdalinfo,
                                                                     version_line):
    """The MEDIUM from review: an old gdalinfo would turn a stamped file into
    'undeclared' with epoch_checked True. It must come back 'unreadable'."""
    run = canned_gdalinfo(version_line, '{"coordinateSystem": {"wkt": "..."}}')
    f = dataset_frame(_write_tif(tmp_path / "a.tif", "EPSG:6340"))
    assert f.epoch is None and f.epoch_source == "unreadable" and not f.epoch_checked
    assert [c[1] for c in run.calls] == ["--version"]   # ran once; -json never run


def test_gdalinfo_failure_raises_not_falls_through(tmp_path, canned_gdalinfo):
    canned_gdalinfo(MODERN, "", rc=1)
    with pytest.raises(OSError, match="gdalinfo failed"):
        dataset_frame(_write_tif(tmp_path / "a.tif", "EPSG:6340"))


@pytest.mark.parametrize("value, expected", [
    (0.0, None), (None, None), (-1.0, None), (2010.0, 2010.0), (1e-9, 1e-9),
])
def test_declared_epoch_gate(value, expected):
    assert gc_crs._declared_epoch(value) == expected


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_non_finite_declared_epoch_raises(value):
    with pytest.raises(ValueError, match="non-finite coordinate epoch"):
        gc_crs._declared_epoch(value)


def test_gdal_dataset_name_strings_pass_through(tmp_path, canned_gdalinfo):
    """A GDAL dataset identifier (not a plain path) must reach the readers
    verbatim (round-2 review: an abspath on every path broke GTIFF_DIR:)."""
    p = _write_tif(tmp_path / "a.tif", "EPSG:6340")
    run = canned_gdalinfo(MODERN, '{"coordinateEpoch": 2010.0}')
    name = f"GTIFF_DIR:1:{p}"
    f = dataset_frame(name)
    assert f.epoch == 2010.0 and f.path == name and run.calls[1][2] == name


def test_relative_dash_leading_name_is_resolved(tmp_path, monkeypatch, canned_gdalinfo):
    monkeypatch.chdir(tmp_path)
    _write_tif(tmp_path / "-dash.tif", "EPSG:6340")
    run = canned_gdalinfo(MODERN, '{"coordinateEpoch": 2010.0}')
    f = dataset_frame("-dash.tif")
    assert f.epoch == 2010.0 and run.calls[1][2] == str(tmp_path / "-dash.tif")
    assert f.path == str(tmp_path / "-dash.tif")


# --- fold-ins from the consumer check-ins (2026-10-28) ---

ITRF2014_GEOG3D = pyproj.CRS("EPSG:7912")


def test_crs_source_dataset_for_a_file_with_geokeys(tmp_path):
    f = dataset_frame(_write_tif(tmp_path / "a.tif", "EPSG:6340"))
    assert f.crs_source == "dataset"
    assert dataset_frame(_write_tif(tmp_path / "nocrs.tif")).crs_source is None


def test_crs_source_pam_when_the_crs_lives_only_in_the_sidecar(tmp_path):
    """A 3-D CRS handed to a GeoTIFF writer lands in .aux.xml with no GeoKeys
    (lidar_tools' final COGs); the file loses its CRS without the sidecar."""
    p = _write_tif(tmp_path / "pamonly.tif")          # no CRS in the file itself
    wkt = NAD83_UTM11_3D.to_wkt()
    (tmp_path / "pamonly.tif.aux.xml").write_text(
        f'<PAMDataset><SRS dataAxisToSRSAxisMapping="1,2">{wkt}</SRS></PAMDataset>')
    f = dataset_frame(p)
    assert f.crs_source == "pam" and len(f.crs.axis_info) == 3
    assert compare_frames(f.crs, NAD83_UTM11_3D).horizontal_equal


@pytest.mark.skipif(not HAVE_OSGEO, reason="needs the GDAL bindings")
def test_crs_source_pam_reproduces_the_translate_3d_case(tmp_path):
    from osgeo import gdal
    src = _write_tif(tmp_path / "src.tif", "EPSG:32611")
    dst = str(tmp_path / "dst.tif")
    with _gdal_exceptions():
        gdal.Translate(dst, src, outputSRS=NAD83_UTM11_3D.to_wkt())
    assert (tmp_path / "dst.tif.aux.xml").exists()
    assert dataset_frame(dst).crs_source == "pam"


def test_dynamic_and_datum_properties():
    static = DatasetFrame(pyproj.CRS("EPSG:6340"), None, "undeclared", "s.tif")
    dyn = DatasetFrame(ITRF2014_GEOG3D, None, "undeclared", "d.tif")
    ens = DatasetFrame(pyproj.CRS("EPSG:32610"), None, "undeclared", "e.tif")
    none = DatasetFrame(None, None, "undeclared", "n.tif")
    assert (static.dynamic, dyn.dynamic, ens.dynamic, none.dynamic) == (False, True, False, None)
    assert static.horizontal_datum == "NAD83 (National Spatial Reference System 2011)"
    assert dyn.horizontal_datum == "International Terrestrial Reference Frame 2014"
    assert (static.ensemble, ens.ensemble, none.ensemble) == (False, True, None)
    assert none.horizontal_datum is None


def test_compare_frames_accepts_dataset_frames():
    a = DatasetFrame(ITRF2014_GEOG3D, 2010.0, "gdal", "a.tif")
    b = DatasetFrame(ITRF2014_GEOG3D, 2023.5, "gdal", "b.tif")
    cmp = compare_frames(a, b)
    assert cmp.problems == ["epoch"] and (cmp.epoch_a, cmp.epoch_b) == (2010.0, 2023.5)
    assert (cmp.dynamic_a, cmp.dynamic_b) == (True, True)
    # a caller's epoch overrides the record's; one side may be a plain CRS
    cmp = compare_frames(a, "EPSG:7912", epoch_a=2023.5, epoch_b=2023.5)
    assert cmp.problems == [] and cmp.epoch_equal is True
    # the undeclared-epoch-in-a-dynamic-frame case stays visible, not a problem
    u = DatasetFrame(ITRF2014_GEOG3D, None, "undeclared", "u.tif")
    cmp = compare_frames(u, b)
    assert cmp.problems == [] and cmp.epoch_equal is None and cmp.dynamic_a is True
    s = DatasetFrame(pyproj.CRS("EPSG:6340"), None, "undeclared", "s.tif")
    assert compare_frames(s, s).dynamic_a is False
    with pytest.raises(ValueError, match="declares no CRS"):
        compare_frames(DatasetFrame(None, None, "undeclared", "n.tif"), b)


def test_assert_same_frame_accepts_dataset_frames():
    from groundcontrol.crs import assert_same_frame
    a = DatasetFrame(ITRF2014_GEOG3D, 2010.0, "gdal", "a.tif")
    b = DatasetFrame(ITRF2014_GEOG3D, 2023.5, "gdal", "b.tif")
    with pytest.raises(ValueError, match="epochs differ"):
        assert_same_frame(a, b)
    assert assert_same_frame(a, b, allow=("epoch",)).problems == ["epoch"]


def test_crs_source_pam_override_when_sidecar_and_keys_disagree(tmp_path, caplog):
    """gdal_edit -ro -a_srs (the COG-safe CRS fix) leaves the old GeoKeys and
    puts the new CRS in .aux.xml; GDAL returns the sidecar's CRS. That is not
    'dataset': a copy without the sidecar carries the OLD CRS silently."""
    p = _write_tif(tmp_path / "ov.tif", "EPSG:32611")      # WGS 84 / UTM 11N in the keys
    (tmp_path / "ov.tif.aux.xml").write_text(
        f'<PAMDataset><SRS dataAxisToSRSAxisMapping="1,2">'
        f'{pyproj.CRS("EPSG:6340").to_wkt()}</SRS></PAMDataset>')
    with caplog.at_level("WARNING", logger="groundcontrol.crs"):
        f = dataset_frame(p)
    assert f.crs_source == "pam-override"
    assert compare_frames(f.crs, "EPSG:6340").horizontal_equal   # the sidecar wins
    assert "overrides the dataset" in caplog.text
    # a stats-only sidecar beside a keyed file is still 'dataset'
    q = _write_tif(tmp_path / "st.tif", "EPSG:6340")
    (tmp_path / "st.tif.aux.xml").write_text(
        '<PAMDataset><PAMRasterBand band="1"><Metadata><MDI key="STATISTICS_MEAN">0</MDI>'
        '</Metadata></PAMRasterBand></PAMDataset>')
    assert dataset_frame(q).crs_source == "dataset"


# --- PR #39 bot findings ---

def test_vertical_only_pam_override_is_not_dataset(tmp_path):
    """Embedded NAD83(2011)+NAVD88, sidecar NAD83(2011)+NGVD29: same horizontal,
    different vertical datum; dropping the sidecar changes the heights' datum."""
    p = _write_tif(tmp_path / "v.tif", "EPSG:6340+5703")
    (tmp_path / "v.tif.aux.xml").write_text(
        f'<PAMDataset><SRS dataAxisToSRSAxisMapping="1,2">'
        f'{pyproj.CRS("EPSG:6340+7968").to_wkt()}</SRS></PAMDataset>')
    f = dataset_frame(p)
    assert f.crs_source == "pam-override"
    assert "NGVD29" in f.crs.name or "7968" in f.crs.to_wkt()


def test_gdal_bindings_older_than_3_4_are_not_a_reader(tmp_path, monkeypatch):
    import types
    fake = types.ModuleType("osgeo")
    fake.gdal = types.SimpleNamespace(__version__="3.2.2")
    monkeypatch.setitem(sys.modules, "osgeo", fake)
    monkeypatch.setitem(sys.modules, "osgeo.gdal", fake.gdal)
    assert gc_crs._epoch_via_gdal(_write_tif(tmp_path / "a.tif", "EPSG:6340")) is None


def test_gdalinfo_route_receives_the_callers_pam_setting(tmp_path, canned_gdalinfo):
    """A caller that hides sidecars (rasterio.Env(GDAL_PAM_ENABLED='NO')) must
    have gdalinfo hide them too, or the epoch could come from a sidecar the
    CRS read ignored."""
    p = _write_tif(tmp_path / "a.tif", "EPSG:6340")
    run = canned_gdalinfo(MODERN, '{"coordinateEpoch": 2010.0}')
    with rasterio.Env(GDAL_PAM_ENABLED="NO"):
        dataset_frame(p)
    json_call = run.calls[1]
    assert "--config" in json_call and "GDAL_PAM_ENABLED" in json_call
    assert json_call[json_call.index("GDAL_PAM_ENABLED") + 1] in ("NO", "False", "false")
    # without a caller setting, no --config is added
    run = canned_gdalinfo(MODERN, '{"coordinateEpoch": 2010.0}')
    dataset_frame(p)
    assert "--config" not in run.calls[1]


@pytest.mark.skipif(not HAVE_OSGEO, reason="needs the GDAL bindings")
def test_bindings_route_honors_the_callers_pam_setting(tmp_path):
    """A stamped epoch that lives only in the sidecar must not be reported as
    checked-and-present when the caller hides sidecars: the CRS read and the
    epoch read must see the same file."""
    p = _write_tif(tmp_path / "a.tif", "EPSG:6340")
    _stamp_epoch(p, 2023.5)                        # GeoTIFF key: in the file itself
    assert dataset_frame(p).epoch == 2023.5
    q = _write_tif(tmp_path / "b.tif", "EPSG:6340")
    (tmp_path / "b.tif.aux.xml").write_text(      # epoch only in the sidecar
        f'<PAMDataset><SRS dataAxisToSRSAxisMapping="1,2" coordinateEpoch="2015.0">'
        f'{pyproj.CRS("EPSG:6340").to_wkt()}</SRS></PAMDataset>')
    assert dataset_frame(q).epoch == 2015.0        # sidecars honored by default
    with rasterio.Env(GDAL_PAM_ENABLED="NO"):
        f = dataset_frame(q)
    assert f.epoch is None and f.epoch_source == "undeclared"   # hidden for both reads
