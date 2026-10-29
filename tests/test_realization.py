"""crs.nad83_realization: one classifier for every NAD83 realization encoding."""

import pyproj
import pytest

from groundcontrol.crs import NAD83_REALIZATIONS, NAD83Realization, nad83_realization

ESRI_HARN_WKT1 = ('GEOGCS["GCS_North_American_1983_HARN",DATUM["D_North_American_1983_HARN",'
                  'SPHEROID["GRS_1980",6378137,298.257222101]],PRIMEM["Greenwich",0],'
                  'UNIT["Degree",0.0174532925199433]]')
LAS_BARE_NAD83_WKT1 = ('PROJCS["NAD83 / UTM zone 18N",GEOGCS["NAD83",DATUM["North_American_Datum_1983",'
                       'SPHEROID["GRS 1980",6378137,298.257222101],TOWGS84[0,0,0,0,0,0,0]],'
                       'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
                       'PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],'
                       'PARAMETER["central_meridian",-75],PARAMETER["scale_factor",0.9996],'
                       'PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1]]')


@pytest.mark.parametrize("crs_like, expected", [
    (4269, "NAD83(1986)"), (26918, "NAD83(1986)"), (5070, "NAD83(1986)"), ("26918", "NAD83(1986)"),
    ("EPSG:26915", "NAD83(1986)"),
    (4152, "NAD83(HARN)"), (3744, "NAD83(HARN)"), (2927, "NAD83(HARN)"), (2893, "NAD83(HARN)"),
    (8545, "NAD83(HARN Corrected)"),
    (4759, "NAD83(NSRS2007)"), (3717, "NAD83(NSRS2007)"),
    (6318, "NAD83(2011)"), (6340, "NAD83(2011)"), (6319, "NAD83(2011)"), ("EPSG:6340+5703", "NAD83(2011)"),
    (pyproj.CRS("EPSG:6340").to_3d(), "NAD83(2011)"),
    (6783, "NAD83(CORS96)"), (8860, "NAD83(FBN)"), (6322, "NAD83(PA11)"), (6324, "NAD83(MA11)"),
    (4617, "NAD83(CSRS)"), (8255, "NAD83(CSRS)v7"),
    (102247, "NAD83(CORS96)"), ("102247", "NAD83(CORS96)"),          # the Esri code WESM carries
    (ESRI_HARN_WKT1, "NAD83(HARN)"), (LAS_BARE_NAD83_WKT1, "NAD83(1986)"),
    ("+proj=utm +zone=11 +datum=NAD83 +units=m +no_defs", "NAD83(1986)"),
])
def test_every_known_encoding_classifies(crs_like, expected):
    r = nad83_realization(crs_like)
    assert isinstance(r, NAD83Realization) and r.realization == expected
    assert r.realization in NAD83_REALIZATIONS
    assert r.bare is (expected == "NAD83(1986)")


def test_epsg_inputs_match_by_datum_code_and_wkt1_by_name_or_code():
    assert nad83_realization(6340).matched_by == "datum_code"
    assert nad83_realization(6340).datum_code == 1116
    assert nad83_realization(3744).datum_code == 6152
    assert nad83_realization(ESRI_HARN_WKT1).realization == "NAD83(HARN)"   # PROJ aliases the Esri name


@pytest.mark.parametrize("crs_like", [4326, 32610, 9755, 7912, 9989, 4267, 4258, 3857, "EPSG:32611+5773"])
def test_non_nad83_is_none(crs_like):
    assert nad83_realization(crs_like) is None


def test_unknown_nad83_variant_raises_by_name():
    wkt = ('GEOGCS["NAD83(Imaginary2030)",DATUM["NAD83_Imaginary_2030",'
           'SPHEROID["GRS 1980",6378137,298.257222101]],PRIMEM["Greenwich",0],'
           'UNIT["degree",0.0174532925199433]]')
    with pytest.raises(ValueError, match="no known realization"):
        nad83_realization(wkt)


def test_vocabulary_is_fixed():
    """Consumers (a published census table) freeze on these tokens."""
    assert NAD83_REALIZATIONS == (
        "NAD83(1986)", "NAD83(HARN)", "NAD83(HARN Corrected)", "NAD83(CORS96)", "NAD83(FBN)",
        "NAD83(NSRS2007)", "NAD83(2011)", "NAD83(PA11)", "NAD83(MA11)", "NAD83(MARP00)",
        "NAD83(PACP00)", "NAD83(CSRS)", "NAD83(CSRS96)", "NAD83(CSRS)v2", "NAD83(CSRS)v3",
        "NAD83(CSRS)v4", "NAD83(CSRS)v5", "NAD83(CSRS)v6", "NAD83(CSRS)v7", "NAD83(CSRS)v8")


#: one geographic CRS per table datum code
GEOG_BY_DATUM = {6269: 4269, 6152: 4152, 1212: 8545, 1133: 6783, 1211: 8860, 6759: 4759, 1116: 6318,
                 1117: 6322, 1118: 6324, 1221: 9072, 1249: 9075, 6140: 4617, 1192: 8232, 1193: 8237,
                 1194: 8240, 1195: 8246, 1196: 8249, 1197: 8252, 1198: 8255, 1365: 10414}


@pytest.mark.parametrize("datum_code, geog", sorted(GEOG_BY_DATUM.items()))
def test_every_table_code_round_trips_by_code_and_by_name(datum_code, geog):
    """The EPSG input hits the code path; its WKT2 re-parse (PROJ drops the
    datum id) must hit the name path with the SAME token (review L5)."""
    from groundcontrol.crs import _NAD83_DATUM_CODES
    try:
        pyproj.CRS.from_epsg(geog)
    except pyproj.exceptions.CRSError:
        pytest.skip(f"EPSG:{geog} not in this PROJ database ({pyproj.proj_version_str})")
    by_code = nad83_realization(geog)
    by_name = nad83_realization(pyproj.CRS.from_epsg(geog).to_wkt())
    assert by_code.matched_by == "datum_code" and by_code.datum_code == datum_code
    assert by_name.matched_by == "datum_name" and by_name.datum_code is None
    assert by_code.realization == by_name.realization == _NAD83_DATUM_CODES[datum_code]


MN_COUNTY_WKT1 = ('GEOGCS["GCS_NAD_1983_HARN_Adj_MN_Anoka",DATUM["D_NAD_1983_HARN_Adj_MN_Anoka",'
                  'SPHEROID["S_GRS_1980_Adj_MN_Anoka",6378418.941,298.257222100883]],'
                  'PRIMEM["Greenwich",0],UNIT["Degree",0.0174532925199433]]')


@pytest.mark.parametrize("crs_like, msg", [
    (MN_COUNTY_WKT1, "county-adjusted"),                  # Esri-only name, raised ellipsoid (review H1)
    ("ESRI:103732", "county-adjusted"),                   # the code the live WESM carries
    ("+proj=utm +zone=11 +ellps=GRS80 +units=m +no_defs", "names an ellipsoid"),      # review M1
    ("+proj=longlat +ellps=GRS80 +towgs84=0,0,0 +no_defs", "names an ellipsoid"),
    (10910, "not in the realization table"),             # California SRN 2025.0 (NAD83 2011), EPSG 1414
])
def test_nad83_family_that_cannot_be_placed_raises(crs_like, msg):
    if isinstance(crs_like, int):
        try:                         # a code newer than the runner's PROJ database
            pyproj.CRS.from_epsg(crs_like)
        except pyproj.exceptions.CRSError:
            pytest.skip(f"EPSG:{crs_like} not in this PROJ database ({pyproj.proj_version_str})")
    with pytest.raises(ValueError, match=msg):
        nad83_realization(crs_like)


def test_integral_float_and_numpy_int_inputs():
    import numpy as np
    assert nad83_realization(26918.0).realization == "NAD83(1986)"
    assert nad83_realization(np.int64(6340)).realization == "NAD83(2011)"
    with pytest.raises(pyproj.exceptions.CRSError):
        nad83_realization(26918.5)


def test_bad_input_raises():
    with pytest.raises(pyproj.exceptions.CRSError):
        nad83_realization("not a crs")
    with pytest.raises(pyproj.exceptions.CRSError):
        nad83_realization(999999999)


@pytest.mark.parametrize("datum_name", ["NAD83(1986)", "NAD83 (1986)", "NAD83(Original)",
                                        "North American Datum 1983 (1986)", "D_NAD_1983"])
def test_authority_less_bare_datum_spellings_classify_as_bare(datum_name):
    """PR #40 bot: an id-less WKT naming the datum by PROJ's own alias
    'NAD83(1986)' must classify as the bare realization, not raise."""
    wkt = (f'GEOGCS["x",DATUM["{datum_name}",SPHEROID["GRS 1980",6378137,298.257222101]],'
           'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]')
    r = nad83_realization(wkt)
    assert r.realization == "NAD83(1986)" and r.bare
