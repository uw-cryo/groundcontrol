"""CLI preflight: cheap input/output validation runs before any (network) fetch,
and a failed export can never destroy a read-only previous product.

Motivated by a user report: NGS + OPUS fetched 345 points over the network,
then ``io.write`` failed on a missing ``pyarrow.parquet`` and the result was
lost. Every pyarrow import in the package is lazy, so nothing failed earlier.
One test per guarantee, so a failure names what broke.
"""

import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from groundcontrol import io

BBOX = "--aoi=-112,32.6,-111.5,33.0"
CRS = "EPSG:32611"
NX, NY, X0, Y0 = 12, 8, 50.0, 40.0


def _plane_tif(tmp_path, name="plane.tif"):
    """A small planar DEM (same construction as tests/test_assess.py)."""
    j, i = np.meshgrid(np.arange(NX), np.arange(NY))
    arr = (2.0 * (X0 + j + 0.5) + 3.0 * (Y0 + NY - i - 0.5)).astype("float64")
    t = Affine(1.0, 0.0, X0, 0.0, -1.0, Y0 + NY)
    path = tmp_path / name
    with rasterio.open(path, "w", driver="GTiff", height=NY, width=NX, count=1,
                       dtype="float64", crs=CRS, transform=t) as dst:
        dst.write(arr, 1)
    return str(path)


def _points(n=4):
    """Landed control on the plane (columns as the assess tests use them)."""
    xs = np.linspace(X0 + 2.5, X0 + 8.5, n)
    ys = np.linspace(Y0 + 2.5, Y0 + 5.5, n)
    return gpd.GeoDataFrame(
        {"source": (["3dep", "3dep", "opus", "ngs"] * n)[:n],
         "point_type": (["NVA", "VVA", "gnss_campaign", "monument"] * n)[:n],
         "height": 2.0 * xs + 3.0 * ys - 0.05},
        geometry=gpd.points_from_xy(xs, ys), crs=CRS)


def _vrt_over(tmp_path, tif, name="mos.vrt"):
    """A single-source VRT referencing ``tif`` (relative path)."""
    vrt = tmp_path / name
    src = os.path.relpath(tif, tmp_path)
    vrt.write_text(
        f'<VRTDataset rasterXSize="{NX}" rasterYSize="{NY}"><SRS>{CRS}</SRS>'
        f'<GeoTransform>{X0},1,0,{Y0 + NY},0,-1</GeoTransform>'
        '<VRTRasterBand dataType="Float64" band="1"><SimpleSource>'
        f'<SourceFilename relativeToVRT="1">{src}</SourceFilename><SourceBand>1</SourceBand>'
        f'<SrcRect xOff="0" yOff="0" xSize="{NX}" ySize="{NY}"/>'
        f'<DstRect xOff="0" yOff="0" xSize="{NX}" ySize="{NY}"/>'
        '</SimpleSource></VRTRasterBand></VRTDataset>')
    return str(vrt)


def _forbid_fetch(monkeypatch):
    """Any fetch attempt fails the test: preflight must run first."""
    import groundcontrol.sources as srcs

    def boom(*a, **k):
        raise AssertionError("fetch_control called before preflight")

    monkeypatch.setattr(srcs, "fetch_control", boom)


def _hide_pyarrow_parquet(monkeypatch):
    # None in sys.modules makes ``import pyarrow.parquet`` raise ImportError
    # even when the real module was imported earlier in the session.
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)


def _chmod_guarded(path, mode):
    """chmod, skipping the test if mode bits are not enforced (root)."""
    path.chmod(mode)
    if os.access(path, os.W_OK):
        path.chmod(0o700 if path.is_dir() else 0o600)
        pytest.skip("mode bits not enforced for this user")


# ------------------------------------------------------- io.check_export_support

def test_check_export_support_accepts_supported_formats(tmp_path, monkeypatch):
    assert io.check_export_support(tmp_path / "c.parquet") == tmp_path / "c.parquet"
    assert io.check_export_support(tmp_path / "c.CSV") == tmp_path / "c.CSV"
    monkeypatch.chdir(tmp_path)
    assert io.check_export_support("bare_name.csv") == Path("bare_name.csv")  # cwd


def test_check_export_support_rejects_other_suffix(tmp_path):
    with pytest.raises(ValueError, match=r"unsupported export format '.gpkg'"):
        io.check_export_support(tmp_path / "c.gpkg")


def test_check_export_support_rejects_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        io.check_export_support(tmp_path / "no" / "such" / "c.parquet")


def test_check_export_support_rejects_unwritable_dir(tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    try:
        _chmod_guarded(ro, 0o500)
        with pytest.raises(PermissionError, match="not writable"):
            io.check_export_support(ro / "c.csv")
    finally:
        ro.chmod(0o700)


def test_check_export_support_rejects_directory_target(tmp_path):
    (tmp_path / "c.parquet").mkdir()
    with pytest.raises(IsADirectoryError, match="is a directory"):
        io.check_export_support(tmp_path / "c.parquet")


def test_check_export_support_rejects_readonly_existing_file(tmp_path):
    """pyarrow unlinks a read-only destination before failing on it: this
    check is what keeps a failed write from deleting the previous product."""
    f = tmp_path / "c.parquet"
    f.write_bytes(b"PRECIOUS")
    try:
        _chmod_guarded(f, 0o444)
        with pytest.raises(PermissionError, match="read-only"):
            io.check_export_support(f)
        # write-only is not enough for parquet: the provenance embed re-reads it
        f.chmod(0o200)
        with pytest.raises(PermissionError, match="read-only"):
            io.check_export_support(f)
        f.chmod(0o200)
        io.check_export_support(tmp_path / "c.csv")  # unrelated path still fine
    finally:
        f.chmod(0o600)


def test_check_export_support_rejects_readonly_sidecar(tmp_path):
    """A replaced product with a stale sidecar is a provenance lie."""
    f = tmp_path / "c.csv"
    side = io.sidecar_path(f)
    side.write_text("{}")
    try:
        _chmod_guarded(side, 0o444)
        with pytest.raises(PermissionError, match=r"provenance\.json.*read-only"):
            io.check_export_support(f)
        assert io.check_export_support(f, sidecar=False) == f  # e.g. the stats CSV
    finally:
        side.chmod(0o600)


def test_check_export_support_bad_tilde_user_is_value_error():
    with pytest.raises(ValueError, match="cannot expand '~'"):
        io.check_export_support("~nosuchuser999/c.parquet")


def test_check_export_support_symlinks(tmp_path):
    """Symlinks are written through, as main did: a link to a not-yet-existing
    file is fine when its directory exists; a link into a missing directory
    is not."""
    into_missing = tmp_path / "link.parquet"
    os.symlink(tmp_path / "nodir" / "t.parquet", into_missing)
    with pytest.raises(FileNotFoundError, match="missing directory"):
        io.check_export_support(into_missing)
    sub = tmp_path / "real"
    sub.mkdir()
    to_new = tmp_path / "link2.parquet"
    os.symlink(sub / "new.parquet", to_new)
    assert io.check_export_support(to_new) == to_new
    to_existing = tmp_path / "link3.parquet"
    (sub / "t.parquet").write_bytes(b"")
    os.symlink(sub / "t.parquet", to_existing)
    assert io.check_export_support(to_existing) == to_existing
    loop = tmp_path / "loop.parquet"
    os.symlink(loop, loop)
    with pytest.raises(OSError, match="symlink loop"):
        io.check_export_support(loop)
    # entry point outside the cycle: entry -> mid -> mid
    mid = tmp_path / "mid.parquet"
    os.symlink(mid, mid)
    entry = tmp_path / "entry.parquet"
    os.symlink(mid, entry)
    with pytest.raises(OSError, match="symlink loop"):
        io.check_export_support(entry)


def test_check_export_support_writable_file_in_unwritable_dir_is_fine(tmp_path):
    """A shared 0555 directory holding a group-writable product: main wrote
    it in place; the preflight must not demand a writable directory then."""
    d = tmp_path / "shared"
    d.mkdir()
    f = d / "c.csv"
    f.write_text("")
    io.sidecar_path(f).write_text("{}")
    try:
        _chmod_guarded(d, 0o555)
        assert io.check_export_support(f) == f
        with pytest.raises(PermissionError, match="not writable"):
            io.check_export_support(d / "new.csv")
    finally:
        d.chmod(0o700)


def test_check_export_support_missing_pyarrow_parquet_is_loud_and_chained(tmp_path, monkeypatch):
    _hide_pyarrow_parquet(monkeypatch)
    with pytest.raises(ImportError, match=r"pyarrow\.parquet.*conda install") as ei:
        io.check_export_support(tmp_path / "c.parquet")
    # geopandas' wrapper raises ``from None``; ours keeps the cause so a
    # broken build is distinguishable from an absent package.
    assert ei.value.__cause__ is not None
    with pytest.raises(ImportError, match="needed for c.parquet"):
        io.check_parquet_engine(tmp_path / "c.parquet")
    with pytest.raises(ImportError, match="needed for the 3DEP"):
        io.check_parquet_engine("the 3DEP checkpoint source")
    io.check_export_support(tmp_path / "c.csv")  # CSV needs no parquet engine


# ------------------------------------------------------------------ io.write

def test_write_readonly_product_is_refused_before_it_can_be_unlinked(tmp_path):
    """pyarrow unlinks a read-only destination before failing: on main the
    product was gone; the preflight refuses first and it survives."""
    path = tmp_path / "control.parquet"
    path.write_bytes(b"PRECIOUS")
    try:
        _chmod_guarded(path, 0o444)
        with pytest.raises(PermissionError, match="read-only"):
            io.write(_points(), path)
        assert path.read_bytes() == b"PRECIOUS"
        assert not io.sidecar_path(path).exists()
    finally:
        path.chmod(0o600)


def test_write_replaces_existing_product_and_keeps_its_mode(tmp_path):
    path = tmp_path / "control.parquet"
    path.write_bytes(b"OLD")
    path.chmod(0o660)  # group-shared product on a shared volume
    old_umask = os.umask(0o022)  # a fresh file would be 0o644: non-vacuous
    try:
        assert io.write(_points(), path) == path
    finally:
        os.umask(old_umask)
    assert len(gpd.read_parquet(path)) == 4
    # pyarrow truncates in place (same inode), so the mode survives -- a
    # platform/pyarrow property main relied on too; a change would show here.
    assert oct(path.stat().st_mode & 0o777) == oct(0o660)
    assert io.read_provenance(path)["n_points"] == 4
    assert io.sidecar_path(path).exists()


def test_write_through_symlink_keeps_link_and_sidecar_beside_it(tmp_path):
    sub = tmp_path / "real"
    sub.mkdir()
    target = sub / "t.csv"
    target.write_text("")
    link = tmp_path / "control.csv"
    os.symlink(target, link)
    assert io.write(_points(), link) == link
    assert link.is_symlink() and target.stat().st_size > 0
    assert io.sidecar_path(link).exists()


# ----------------------------------------------------------- --target-crs

def test_resolve_crs_inline_wkt_and_epsg_pass_through(tmp_path):
    """A WKT2 compound string exceeds the filesystem's name limit; probing it
    as a path used to raise OSError (ENAMETOOLONG) instead of returning it."""
    import pyproj
    from groundcontrol.cli import _resolve_crs

    wkt = pyproj.CRS("EPSG:32611+5703").to_wkt()
    assert len(wkt) > 1000
    assert _resolve_crs(wkt, "--target-crs") == wkt
    assert _resolve_crs("EPSG:32611+5703", "--target-crs") == "EPSG:32611+5703"
    f = tmp_path / "frame.wkt"
    f.write_text(wkt)
    assert _resolve_crs(str(f), "--target-crs") == wkt
    with pytest.raises(ValueError, match=r"--source-crs .*missing\.wkt.*cannot read"):
        _resolve_crs(str(tmp_path / "missing.wkt"), "--source-crs")


# -------------------------------------------------------------- fetch CLI

def _fetch(argv):
    from groundcontrol.cli import fetch_control_main
    return fetch_control_main(argv)


def test_fetch_rejects_bad_out_suffix(tmp_path, monkeypatch):
    _forbid_fetch(monkeypatch)
    with pytest.raises(SystemExit, match="unsupported export format"):
        _fetch([BBOX, "--sources", "ngs", "--out", str(tmp_path / "c.gpkg")])


def test_fetch_rejects_missing_out_dir(tmp_path, monkeypatch):
    _forbid_fetch(monkeypatch)
    with pytest.raises(SystemExit, match="does not exist"):
        _fetch([BBOX, "--sources", "ngs", "--out", str(tmp_path / "typo" / "c.parquet")])


def test_fetch_rejects_readonly_existing_out(tmp_path, monkeypatch):
    _forbid_fetch(monkeypatch)
    f = tmp_path / "c.parquet"
    f.write_bytes(b"PRECIOUS")
    try:
        _chmod_guarded(f, 0o444)
        with pytest.raises(SystemExit, match="read-only"):
            _fetch([BBOX, "--sources", "ngs", "--out", str(f)])
        assert f.read_bytes() == b"PRECIOUS"
    finally:
        f.chmod(0o600)


def test_fetch_rejects_missing_pyarrow_before_fetch(tmp_path, monkeypatch):
    _forbid_fetch(monkeypatch)
    _hide_pyarrow_parquet(monkeypatch)
    with pytest.raises(SystemExit, match=r"pyarrow\.parquet"):
        _fetch([BBOX, "--sources", "ngs", "--out", str(tmp_path / "c.parquet")])


def test_fetch_rejects_missing_pyarrow_for_3dep_even_with_csv_out(tmp_path, monkeypatch):
    _forbid_fetch(monkeypatch)
    _hide_pyarrow_parquet(monkeypatch)
    with pytest.raises(SystemExit, match=r"pyarrow\.parquet.*3DEP"):
        _fetch([BBOX, "--sources", "3dep,ngs", "--out", str(tmp_path / "c.csv")])
    # without 3dep a CSV export needs no parquet engine at preflight
    with pytest.raises(AssertionError, match="fetch_control called"):
        _fetch([BBOX, "--sources", "ngs", "--out", str(tmp_path / "c.csv")])


def test_fetch_rejects_unknown_source_name(tmp_path, monkeypatch):
    """A typo is not a source failure: the dispatcher would degrade it and
    write a product silently missing that source."""
    _forbid_fetch(monkeypatch)
    with pytest.raises(SystemExit, match=r"--sources: unknown \['opsu'\]"):
        _fetch([BBOX, "--sources", "ngs,opsu", "--out", str(tmp_path / "c.parquet")])


def test_fetch_bad_tilde_user_is_clean_error(tmp_path, monkeypatch):
    _forbid_fetch(monkeypatch)
    with pytest.raises(SystemExit, match="cannot expand"):
        _fetch([BBOX, "--sources", "ngs", "--out", "~nosuchuser999/c.parquet"])
    with pytest.raises(SystemExit, match="cannot expand"):
        _fetch(["--aoi", "~nosuchuser999/a.geojson", "--sources", "ngs",
                "--out", str(tmp_path / "c.parquet")])


@pytest.mark.parametrize("bad", ["--aoi=a,b,c,d", "--aoi=-112,32.6,-111.5",
                                 "--aoi=-112,32.6,nan,33.0", "--aoi=-111.5,32.6,-112,33.0",
                                 "--aoi="])
def test_fetch_rejects_bad_bbox(tmp_path, monkeypatch, bad):
    _forbid_fetch(monkeypatch)
    with pytest.raises(SystemExit, match="--aoi"):
        _fetch([bad, "--sources", "ngs", "--out", str(tmp_path / "c.parquet")])


def test_fetch_accepts_directory_aoi_datasource(tmp_path, monkeypatch):
    """A shapefile directory is a valid OGR datasource (main accepted it)."""
    _forbid_fetch(monkeypatch)
    shpdir = tmp_path / "shpdir"
    shpdir.mkdir()
    _points().to_crs("EPSG:4326")[["geometry"]].to_file(shpdir / "aoi.shp")
    with pytest.raises(AssertionError, match="fetch_control called"):  # preflight passed
        _fetch(["--aoi", str(shpdir), "--sources", "ngs", "--out", str(tmp_path / "c.parquet")])


def test_fetch_passes_driver_connection_string_aoi_to_gdal(tmp_path, monkeypatch):
    """PG:dbname=... is an OGR datasource, not a local file to probe: it is
    handed to GDAL (whose own failure -- no server here -- is reported
    naming the flag, before any fetch)."""
    _forbid_fetch(monkeypatch)
    with pytest.raises(SystemExit, match=r"--aoi PG:dbname=sites: not a readable vector AOI"):
        _fetch(["--aoi", "PG:dbname=sites", "--sources", "ngs", "--out", str(tmp_path / "c.parquet")])


@pytest.mark.parametrize("typo", ["data:aoi.geojson", r"C:\data\aoi.geojson", "z" * 300])
def test_fetch_rejects_colon_typos_that_are_not_gdal_drivers(tmp_path, monkeypatch, typo):
    _forbid_fetch(monkeypatch)
    with pytest.raises(SystemExit, match="file not found|not a usable path"):
        _fetch(["--aoi", typo, "--sources", "ngs", "--out", str(tmp_path / "c.parquet")])


def test_is_remote_classification():
    from groundcontrol.cli import _is_remote
    for remote in ("https://x/a.geojson", "/vsizip/a.zip/a.shp", "PG:dbname=sites",
                   'NETCDF:"x.nc":var', "HDF4_SDS:UNKNOWN:/p/f.hdf:0",
                   "SENTINEL2_L1C:/p/x.zip", "PG:" + "z" * 300):
        assert _is_remote(remote), remote
    for local in ("site.gpkg", "data:aoi.geojson", r"C:\data\aoi.geojson", "aoi.geojson"):
        assert not _is_remote(local), local


def test_fetch_rejects_missing_aoi_file(tmp_path, monkeypatch):
    _forbid_fetch(monkeypatch)
    with pytest.raises(SystemExit, match="file not found"):
        _fetch(["--aoi", str(tmp_path / "site_aoi.geojson"), "--sources", "ngs",
                "--out", str(tmp_path / "c.parquet")])


# ------------------------------------------------------------- assess CLI

def _assess(argv):
    from groundcontrol.cli import assess_dem_main
    return assess_dem_main(argv)


def _base(tmp_path, dsm, target_crs="EPSG:32611"):
    return [BBOX, "--product", f"DSM={dsm}", "--target-crs", target_crs,
            "--outdir", str(tmp_path / "out"), "--site-name", "t", "--no-figures"]


def test_assess_rejects_malformed_product(tmp_path, monkeypatch):
    _forbid_fetch(monkeypatch)
    with pytest.raises(SystemExit, match="NAME=PATH"):
        _assess([BBOX, "--product", "nonsense", "--target-crs", "EPSG:32611",
                 "--outdir", str(tmp_path / "out"), "--site-name", "t"])


def test_assess_rejects_missing_product_raster_and_creates_no_outdir(tmp_path, monkeypatch):
    _forbid_fetch(monkeypatch)
    with pytest.raises(SystemExit, match="cannot open raster"):
        _assess(_base(tmp_path, tmp_path / "typo.tif"))
    assert not (tmp_path / "out").exists()


def test_assess_rejects_missing_hillshade(tmp_path, monkeypatch):
    _forbid_fetch(monkeypatch)
    dsm = _plane_tif(tmp_path)
    with pytest.raises(SystemExit, match=r"--hs.*cannot open raster"):
        _assess(_base(tmp_path, dsm) + ["--hs", str(tmp_path / "nohs.tif")])
    with pytest.raises(SystemExit, match=r"--hs.*cannot open raster"):
        _assess(_base(tmp_path, dsm) + ["--hs", f"DSM={tmp_path / 'nohs.tif'}"])


def test_assess_rejects_missing_aoi_file(tmp_path, monkeypatch):
    _forbid_fetch(monkeypatch)
    dsm = _plane_tif(tmp_path)
    argv = _base(tmp_path, dsm)
    argv[0:1] = ["--aoi", str(tmp_path / "site_aoi.geojson")]
    with pytest.raises(SystemExit, match="file not found"):
        _assess(argv)


def test_assess_rejects_missing_wkt_file_naming_the_flag(tmp_path, monkeypatch):
    _forbid_fetch(monkeypatch)
    dsm = _plane_tif(tmp_path)
    with pytest.raises(SystemExit, match=r"--target-crs .*missing\.wkt.*cannot read"):
        _assess(_base(tmp_path, dsm, target_crs=str(tmp_path / "missing.wkt")))
    with pytest.raises(SystemExit, match=r"--source-crs .*cannot read"):
        _assess(_base(tmp_path, dsm) + ["--source-crs", dsm])  # a GeoTIFF is not WKT


def test_assess_rejects_unknown_source_name(tmp_path, monkeypatch):
    _forbid_fetch(monkeypatch)
    dsm = _plane_tif(tmp_path)
    with pytest.raises(SystemExit, match=r"--sources: unknown \['opsu'\]"):
        _assess(_base(tmp_path, dsm) + ["--sources", "3dep,opsu"])


def test_assess_rejects_target_crs_typo(tmp_path, monkeypatch):
    _forbid_fetch(monkeypatch)
    dsm = _plane_tif(tmp_path)
    with pytest.raises(SystemExit, match="--target-crs: not a valid CRS"):
        _assess(_base(tmp_path, dsm, target_crs="EPSG:99999"))


def test_assess_rejects_empty_source_crs(tmp_path, monkeypatch):
    """'' used to fall through to the default landing CRS silently."""
    _forbid_fetch(monkeypatch)
    dsm = _plane_tif(tmp_path)
    with pytest.raises(SystemExit, match="--source-crs: not a valid CRS"):
        _assess(_base(tmp_path, dsm) + ["--source-crs", ""])


def test_assess_rejects_new_csv_cache(tmp_path, monkeypatch):
    _forbid_fetch(monkeypatch)
    dsm = _plane_tif(tmp_path)
    with pytest.raises(SystemExit, match=r"must be \.parquet"):
        _assess(_base(tmp_path, dsm) + ["--control", str(tmp_path / "new.csv")])


def test_assess_rejects_directory_cache(tmp_path, monkeypatch):
    """pyarrow reads a directory as a partitioned dataset: two copies of the
    cache in a dir named like one would silently double every count."""
    _forbid_fetch(monkeypatch)
    dsm = _plane_tif(tmp_path, "a-DSM_mos.tif")
    d = tmp_path / "t_control.parquet"
    d.mkdir()
    _points().to_parquet(d / "a.parquet")
    _points().to_parquet(d / "b.parquet")
    with pytest.raises(SystemExit, match="control cache .*is a directory"):
        _assess(_base(tmp_path, dsm, target_crs=CRS) + ["--source-crs", CRS,
                                                        "--control", str(d)])


def test_assess_rejects_unknown_source_even_with_cache(tmp_path, monkeypatch):
    _forbid_fetch(monkeypatch)
    dsm = _plane_tif(tmp_path, "a-DSM_mos.tif")
    cache = tmp_path / "ctl.parquet"
    _points().to_parquet(cache)
    with pytest.raises(SystemExit, match=r"--sources: unknown \['opsu'\]"):
        _assess(_base(tmp_path, dsm, target_crs=CRS) + ["--source-crs", CRS,
                                                        "--control", str(cache),
                                                        "--sources", "ngs,opsu"])


def test_assess_rejects_cache_symlink_into_missing_dir(tmp_path, monkeypatch):
    _forbid_fetch(monkeypatch)
    dsm = _plane_tif(tmp_path)
    link = tmp_path / "ctl.parquet"
    os.symlink(tmp_path / "gone" / "ctl.parquet", link)
    with pytest.raises(SystemExit, match="symlink into a missing directory"):
        _assess(_base(tmp_path, dsm) + ["--control", str(link)])


def test_assess_rejects_corrupt_existing_cache(tmp_path, monkeypatch):
    _forbid_fetch(monkeypatch)
    dsm = _plane_tif(tmp_path)
    bad = tmp_path / "ctl.parquet"
    bad.write_text("not parquet")
    with pytest.raises(SystemExit, match="not a readable GeoParquet file"):
        _assess(_base(tmp_path, dsm) + ["--control", str(bad)])
    assert not (tmp_path / "out").exists()


def test_assess_rejects_outdir_that_is_a_file(tmp_path, monkeypatch):
    _forbid_fetch(monkeypatch)
    dsm = _plane_tif(tmp_path)
    (tmp_path / "out").write_text("")
    with pytest.raises(SystemExit, match="--outdir .*not a directory"):
        _assess(_base(tmp_path, dsm))


def test_assess_expands_tilde_in_outdir_and_control(tmp_path, monkeypatch):
    """'~/out' must land in $HOME, not create a literal './~' directory, and
    a '~/...' cache must be found again on the next run."""
    _forbid_fetch(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    dsm = _plane_tif(tmp_path, "a-DSM_mos.tif")
    _points().to_parquet(tmp_path / "ctl.parquet")
    argv = _base(tmp_path, dsm, target_crs=CRS) + ["--source-crs", CRS,
                                                   "--control", "~/ctl.parquet"]
    argv[argv.index("--outdir") + 1] = "~/out"
    assert _assess(argv) == 0
    assert (tmp_path / "out" / "t_dz_stats.csv").exists()
    assert not (tmp_path / "~").exists()


def test_assess_readonly_outdir_with_external_control_is_clean_error(tmp_path, monkeypatch):
    _forbid_fetch(monkeypatch)
    dsm = _plane_tif(tmp_path, "a-DSM_mos.tif")
    cache = tmp_path / "ctl.parquet"
    _points().to_parquet(cache)
    out = tmp_path / "out"
    out.mkdir()
    try:
        _chmod_guarded(out, 0o500)
        with pytest.raises(SystemExit, match="t_assessed.parquet.*not writable"):
            _assess(_base(tmp_path, dsm, target_crs=CRS)
                    + ["--source-crs", CRS, "--control", str(cache)])
    finally:
        out.chmod(0o700)


def test_assess_expands_tilde_in_product(tmp_path, monkeypatch):
    _forbid_fetch(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    _plane_tif(tmp_path, "a-DSM_mos.tif")
    _points().to_parquet(tmp_path / "ctl.parquet")
    argv = _base(tmp_path, "~/a-DSM_mos.tif", target_crs=CRS) + [
        "--source-crs", CRS, "--control", str(tmp_path / "ctl.parquet")]
    assert _assess(argv) == 0


def test_assess_rejects_dangling_symlink_outdir(tmp_path, monkeypatch):
    _forbid_fetch(monkeypatch)
    dsm = _plane_tif(tmp_path)
    os.symlink(tmp_path / "gone", tmp_path / "out")
    with pytest.raises(SystemExit, match="--outdir .*not a directory"):
        _assess(_base(tmp_path, dsm))


def test_assess_rejects_vrt_with_missing_source(tmp_path, monkeypatch):
    """GDAL opens a VRT lazily; a renamed source only fails at read time."""
    _forbid_fetch(monkeypatch)
    tif = _plane_tif(tmp_path, "a-DSM_mos.tif")
    vrt = _vrt_over(tmp_path, tif)
    with rasterio.open(vrt) as ds:
        assert ds.count == 1
    os.rename(tif, tmp_path / "a-DSM_mos_STALE.tif")
    with pytest.raises(SystemExit, match=r"references 1 missing file.*a-DSM_mos\.tif"):
        _assess(_base(tmp_path, vrt))


def test_assess_vrt_with_overlong_source_path_is_clean_error(tmp_path, monkeypatch):
    """A relativeToVRT chain can resolve past PATH_MAX: report 'missing',
    never raise ENAMETOOLONG through the preflight."""
    _forbid_fetch(monkeypatch)
    tif = _plane_tif(tmp_path, "a-DSM_mos.tif")
    vrt = Path(_vrt_over(tmp_path, tif))
    deep = "/".join(["x" * 200] * 6) + "/a-DSM_mos.tif"  # ~1200 chars, each component legal
    vrt.write_text(vrt.read_text().replace('relativeToVRT="1">a-DSM_mos.tif',
                                           f'relativeToVRT="0">{deep}'))
    with pytest.raises(SystemExit, match="missing file"):
        _assess(_base(tmp_path, str(vrt)))


def test_assess_rejects_missing_pyarrow_before_fetch(tmp_path, monkeypatch):
    _forbid_fetch(monkeypatch)
    dsm = _plane_tif(tmp_path)
    _hide_pyarrow_parquet(monkeypatch)
    with pytest.raises(SystemExit, match=r"pyarrow\.parquet"):
        _assess(_base(tmp_path, dsm))


def test_assess_existing_cache_accepted_by_content_not_name(tmp_path, monkeypatch):
    """An existing GeoParquet cache is read whatever its name (quarantine
    renames like *_STALE_v1 stay usable); no fetch happens."""
    _forbid_fetch(monkeypatch)
    dsm = _plane_tif(tmp_path, "a-DSM_mos.tif")
    cache = tmp_path / "t_control.parquet_STALE_v1"
    _points().to_parquet(cache)
    rc = _assess(_base(tmp_path, dsm, target_crs=CRS)
                 + ["--source-crs", CRS, "--control", str(cache)])
    assert rc == 0
    assert (tmp_path / "out" / "t_dz_stats.csv").exists()


def test_assess_source_crs_accepts_wkt_file_like_target_crs(tmp_path, monkeypatch):
    import pyproj

    _forbid_fetch(monkeypatch)
    dsm = _plane_tif(tmp_path, "a-DSM_mos.tif")
    cache = tmp_path / "ctl.parquet"
    _points().to_parquet(cache)
    wkt = tmp_path / "frame.wkt"
    wkt.write_text(pyproj.CRS(CRS).to_wkt())
    rc = _assess(_base(tmp_path, dsm, target_crs=str(wkt))
                 + ["--source-crs", str(wkt), "--control", str(cache)])
    assert rc == 0


# ------------------------------------- positional inputs, --vdatum (2026-08-31)

def _plane_tif_nad83(tmp_path, name="plane_n83.tif"):
    """The _plane_tif grid stamped EPSG:6339 (NAD83(2011) UTM 10N — a
    REALIZED datum; the default fixture's EPSG:32611 is the WGS84
    ensemble, which --vdatum deliberately refuses)."""
    import shutil
    src = _plane_tif(tmp_path)
    path = str(tmp_path / name)
    shutil.copy(src, path)
    with rasterio.open(path, "r+") as dst:
        dst.crs = rasterio.crs.CRS.from_epsg(6339)
    return path


def test_vdatum_target_crs_helper(tmp_path):
    """--vdatum resolver: product 2D horizontal + vertical spec -> 3D target."""
    import pyproj

    from groundcontrol.cli import _vdatum_target_crs
    dsm = _plane_tif_nad83(tmp_path)
    c = pyproj.CRS(_vdatum_target_crs({"DSM": dsm}, "ellipsoid"))
    assert len(c.axis_info) == 3
    assert [a.direction for a in c.axis_info] == ["east", "north", "up"]
    c2 = pyproj.CRS(_vdatum_target_crs({"DSM": dsm}, "EPSG:5703"))
    assert c2.equals(pyproj.CRS("EPSG:6339+5703"))
    with pytest.raises(ValueError, match="not a vertical CRS"):
        _vdatum_target_crs({"DSM": dsm}, "EPSG:4326")
    with pytest.raises(ValueError, match="does not\nresolve as a CRS|does not "):
        _vdatum_target_crs({"DSM": dsm}, "GEOID18")


def test_assess_vdatum_excludes_target_crs(tmp_path, monkeypatch, capsys):
    _forbid_fetch(monkeypatch)
    dsm = _plane_tif(tmp_path)
    with pytest.raises(SystemExit):        # argparse p.error -> exit 2 + stderr
        _assess(_base(tmp_path, dsm) + ["--vdatum", "ellipsoid"])
    assert "mutually exclusive" in capsys.readouterr().err


def test_assess_2d_refusal_suggests_vdatum_choices(tmp_path, monkeypatch):
    """The 2D-CRS refusal names concrete completions with the product's own
    EPSG (owner 2026-08-31: give people the common choices)."""
    _forbid_fetch(monkeypatch)
    dsm = _plane_tif_nad83(tmp_path)
    with pytest.raises(SystemExit) as exc:
        _assess([BBOX, "--product", f"DSM={dsm}",
                 "--outdir", str(tmp_path / "out"), "--no-figures"])
    msg = str(exc.value)
    assert "--vdatum ellipsoid" in msg
    assert "--vdatum EPSG:5703" in msg
    assert "EPSG:6339+5703" in msg


def test_assess_positional_raster_names_and_default_outdir(tmp_path, monkeypatch, capsys):
    """`groundcontrol-assess dem.tif`: stem-classified product name and the
    <stem>_groundcontrol default outdir, then the normal 2D-CRS refusal."""
    _forbid_fetch(monkeypatch)
    dsm = _plane_tif(tmp_path)                      # stem 'plane' -> DSM
    with pytest.raises(SystemExit, match="product DSM="):
        _assess([dsm])
    assert (f"outdir (default): {tmp_path / 'plane_groundcontrol'}"
            in capsys.readouterr().err)
    dtm = _plane_tif(tmp_path, name="site_dtm.tif")  # 'dtm' in stem -> DTM
    with pytest.raises(SystemExit, match="product DTM="):
        _assess([dtm])


def test_assess_positional_vector_dispatches_to_fetch(tmp_path, monkeypatch):
    """`groundcontrol-assess aoi.geojson` (no raster) runs the AOI-only
    fetch path — proven by the forbidden fetch_control being reached."""
    _forbid_fetch(monkeypatch)
    aoi = tmp_path / "site_aoi.geojson"
    _points().to_crs("EPSG:4326")[["geometry"]].to_file(aoi, driver="GeoJSON")
    with pytest.raises(AssertionError, match="fetch_control called"):
        _assess([str(aoi)])
    # the default outdir was derived next to the AOI
    assert (tmp_path / "site_aoi_groundcontrol").is_dir()


def test_assess_positional_rejections(tmp_path, monkeypatch, capsys):
    _forbid_fetch(monkeypatch)
    aoi = tmp_path / "a.geojson"
    _points().to_crs("EPSG:4326")[["geometry"]].to_file(aoi, driver="GeoJSON")
    aoi2 = tmp_path / "b.geojson"
    _points().to_crs("EPSG:4326")[["geometry"]].to_file(aoi2, driver="GeoJSON")
    with pytest.raises(SystemExit, match="more than one vector"):
        _assess([str(aoi), str(aoi2)])
    with pytest.raises(SystemExit, match="conflicts\n.*--aoi|conflicts "):
        _assess([str(aoi), "--aoi", str(aoi2)])
    with pytest.raises(SystemExit):        # argparse p.error -> exit 2 + stderr
        _assess(["--outdir", str(tmp_path / "out")])
    with pytest.raises(SystemExit, match="file not found"):
        _assess([str(tmp_path / "missing.tif")])
    assert "no inputs" in capsys.readouterr().err
    junk = tmp_path / "notes.txt"
    junk.write_text("not geodata")
    with pytest.raises(SystemExit, match="not a readable raster or vector"):
        _assess([str(junk)])
    with pytest.raises(SystemExit, match="--vdatum needs a raster"):
        _assess([str(aoi), "--vdatum", "ellipsoid"])


def _plane_tif_wgs84(tmp_path, name="ens.tif"):
    """The _plane_tif grid stamped EPSG:32610 (WGS 84 ensemble UTM)."""
    import shutil
    src = _plane_tif(tmp_path)
    path = str(tmp_path / name)
    shutil.copy(src, path)
    import rasterio
    with rasterio.open(path, "r+") as dst:
        dst.crs = rasterio.crs.CRS.from_epsg(32610)
    return path


def test_vdatum_refuses_wgs84_ensemble_and_offers_realizations(tmp_path, monkeypatch):
    """EPSG:326xx products (EarthDEM et al.): bare ellipsoid is refused —
    the ensemble is ~2 m ambiguity and PROJ's best chain to it is
    meter-class — and ellipsoid:<realization> is the sanctioned path."""
    import pyproj

    from groundcontrol.cli import _vdatum_target_crs
    from groundcontrol.geodesy import build_utm_itrf2014_3d
    ens = _plane_tif_wgs84(tmp_path)
    with pytest.raises(ValueError, match="ENSEMBLE.*realization"):
        _vdatum_target_crs({"DSM": ens}, "ellipsoid")
    with pytest.raises(ValueError, match="ENSEMBLE"):
        _vdatum_target_crs({"DSM": ens}, "EPSG:5703")
    wkt = _vdatum_target_crs({"DSM": ens}, "ellipsoid:itrf2014")
    assert pyproj.CRS(wkt).equals(build_utm_itrf2014_3d(32610))


def test_assess_2d_ensemble_refusal_suggests_realizations(tmp_path, monkeypatch):
    _forbid_fetch(monkeypatch)
    ens = _plane_tif_wgs84(tmp_path)
    with pytest.raises(SystemExit) as exc:
        _assess([ens, "--outdir", str(tmp_path / "out")])
    msg = str(exc.value)
    assert "ellipsoid:itrf2014" in msg
    assert "ENSEMBLE" in msg
    assert "--vdatum ellipsoid\n" not in msg      # the refused bare form


def test_vdatum_presets_and_polar_rebase(tmp_path):
    """Product presets resolve to the researched frame, and the
    ellipsoid:<realization> rebase handles non-UTM (polar stereo) grids."""
    import pyproj

    from groundcontrol.cli import VDATUM_PRESETS, _vdatum_target_crs
    from groundcontrol.geodesy import build_utm_itrf2014_3d, with_vdatum
    ens = _plane_tif_wgs84(tmp_path)
    wkt = _vdatum_target_crs({"DSM": ens}, "earthdem")
    assert pyproj.CRS(wkt).equals(build_utm_itrf2014_3d(32610))
    assert VDATUM_PRESETS["precision3d"] == "ellipsoid:g1674"
    assert set(VDATUM_PRESETS) == {"3dep", "cop30", "precision3d",
                                   "earthdem", "arcticdem", "rema"}
    c = with_vdatum("EPSG:3413", "ellipsoid:itrf2014")   # ArcticDEM grid
    assert c.name.startswith("ITRF2014 /")
    assert len(c.axis_info) == 3
    # cop30: orthometric vertical on a rebased ensemble grid
    from pyproj.crs import CompoundCRS
    cop = with_vdatum("EPSG:4326", VDATUM_PRESETS["cop30"])
    want = CompoundCRS(name="x", components=[pyproj.CRS.from_epsg(9000),
                                             pyproj.CRS.from_epsg(3855)])
    assert cop.equals(want)


def test_assess_ensemble_refusal_names_pgc_presets(tmp_path, monkeypatch):
    """A SETSM-looking filename adds the preset hint to the refusal."""
    _forbid_fetch(monkeypatch)
    ens = _plane_tif_wgs84(tmp_path, name="SETSM_s2s041_WV03_fake_2m.tif")
    with pytest.raises(SystemExit) as exc:
        _assess([ens, "--outdir", str(tmp_path / "out")])
    assert "--vdatum earthdem" in str(exc.value)
    assert "docs/vdatum.md" in str(exc.value)


def test_one_product_family_per_run(tmp_path, monkeypatch):
    """Owner ruling 2026-09-01: at most one surface + one bare-earth
    product per run; independent acquisitions are separate runs."""
    from groundcontrol.assess import check_product_family
    check_product_family({"DSM": "a.tif", "DTM": "b.tif"})   # the pair: fine
    check_product_family({"DSM": "a.tif"})
    with pytest.raises(ValueError, match="2 surface.*separate runs"):
        check_product_family({"DSM": "a.tif", "strip2": "b.tif"})
    with pytest.raises(ValueError, match="bare-earth"):
        check_product_family({"DTM_2020": "a.tif", "DTM_2021": "b.tif"})
    # CLI: two positional surface rasters refuse BEFORE any fetch
    _forbid_fetch(monkeypatch)
    a = _plane_tif(tmp_path, name="strip_a.tif")
    b = _plane_tif(tmp_path, name="strip_b.tif")
    with pytest.raises(SystemExit, match="separate runs"):
        _assess([a, b, "--outdir", str(tmp_path / "out")])


def test_disjoint_pair_bounds_refused(tmp_path, monkeypatch):
    """A DSM/DTM pair with disjoint extents is not a family."""
    import shutil

    from affine import Affine
    _forbid_fetch(monkeypatch)
    dsm = _plane_tif_nad83(tmp_path, name="site_dsm.tif")
    far = tmp_path / "far_dtm.tif"
    shutil.copy(dsm, far)
    with rasterio.open(far, "r+") as dst:
        dst.transform = Affine(1.0, 0.0, 900000, 0.0, -1.0, 100000)
    with pytest.raises(SystemExit, match="disjoint extents"):
        _assess([str(dsm), str(far), "--vdatum", "ellipsoid",
                 "--outdir", str(tmp_path / "out")])


def test_vdatum_rebase_samples_under_declared_frame(tmp_path, monkeypatch):
    """The --vdatum reinterpretation contract end-to-end (owner bug report
    2026-09-01): points landed in the DECLARED frame (ITRF2014 on the
    raster's own grid) must sample a raster whose header still says the
    ensemble — same grid, datum reinterpretation. A genuinely different
    grid still refuses."""
    import geopandas as gpd
    import numpy as np

    from groundcontrol.geodesy import with_vdatum
    _forbid_fetch(monkeypatch)
    dsm = _plane_tif_wgs84(tmp_path)                 # header: EPSG:32610
    tgt = with_vdatum("EPSG:32610", "ellipsoid:itrf2014")
    xs = np.linspace(X0 + 2.5, X0 + 8.5, 4)
    ys = np.linspace(Y0 + 2.5, Y0 + 5.5, 4)
    pts = gpd.GeoDataFrame({"source": ["ngs"] * 4,
                            "point_type": ["monument"] * 4,
                            "height": 2.0 * xs + 3.0 * ys - 0.05},
                           geometry=gpd.points_from_xy(xs, ys), crs=tgt)
    cache = tmp_path / "ctl.parquet"
    pts.to_parquet(cache)
    wkt = tmp_path / "tgt.wkt"
    wkt.write_text(tgt.to_wkt())
    rc = _assess([dsm, "--vdatum", "ellipsoid:itrf2014",
                  "--source-crs", str(wkt), "--control", str(cache),
                  "--outdir", str(tmp_path / "out"), "--site-name", "r",
                  "--no-figures"])
    assert rc == 0
    out = gpd.read_parquet(tmp_path / "out" / "r_assessed.parquet")
    assert np.isfinite(out["dh_DSM_before"]).all()
    # a different GRID under the declaration still refuses
    from groundcontrol.sample import _grid_signature
    assert _grid_signature("EPSG:32610") == _grid_signature(tgt)
    assert _grid_signature("EPSG:32611") != _grid_signature(tgt)


def test_vdatum_disambiguates_3d_ensemble_product(tmp_path):
    """A product declaring 3D heights on the WGS84 ENSEMBLE is ambiguity
    in name only: the embedded-CRS refusal tells the user to pass
    --vdatum ellipsoid:<realization>, so the vdatum path must accept it
    (owner catch-22 report 2026-09-01). A REALIZED 3D declaration is
    still respected/refused."""
    import pyproj

    from groundcontrol.cli import _vdatum_target_crs
    from groundcontrol.geodesy import build_utm_itrf2014_3d
    dsm = _plane_tif_wgs84(tmp_path)
    with rasterio.open(dsm, "r+") as dst:
        dst.crs = rasterio.crs.CRS.from_wkt(
            pyproj.CRS.from_epsg(32610).to_3d().to_wkt())
    with rasterio.open(dsm) as src:
        from groundcontrol.assess import has_vertical_axis
        if not has_vertical_axis(pyproj.CRS.from_user_input(src.crs)):
            pytest.skip("GDAL build flattens 3D GeoTIFF CRS")
    wkt = _vdatum_target_crs({"DSM": dsm}, "ellipsoid:itrf2014")
    assert pyproj.CRS(wkt).equals(build_utm_itrf2014_3d(32610))
    # realized 3D declaration: still refused
    n83 = _plane_tif_nad83(tmp_path, name="n83_3d.tif")
    with rasterio.open(n83, "r+") as dst:
        dst.crs = rasterio.crs.CRS.from_wkt(
            pyproj.CRS.from_epsg(6339).to_3d().to_wkt())
    with rasterio.open(n83) as src:
        if has_vertical_axis(pyproj.CRS.from_user_input(src.crs)):
            with pytest.raises(ValueError, match="already declares"):
                _vdatum_target_crs({"DSM": n83}, "ellipsoid")
