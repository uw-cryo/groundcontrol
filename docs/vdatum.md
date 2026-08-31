# Vertical datum field guide

`groundcontrol-assess --vdatum` joins a stated vertical datum to a product's
own 2D horizontal CRS to build the unambiguous 3D target frame. A 2D CRS is
never trusted for heights (it says nothing about them), and for ELLIPSOIDAL
heights the **WGS 84 ensemble** (`EPSG:326xx`/`327xx`, `EPSG:4326`) is
refused outright: it is ~2 m of deliberate ambiguity spanning realizations
from Transit to G2296, PROJ's best transform chains to it are meter-class
and member-agnostic — and for ellipsoidal heights the realization IS the
height datum, so state what the heights are actually on. An ORTHOMETRIC
vertical on an ensemble grid (`--vdatum EPSG:3855` on a COP30 derivative)
is different: the vertical defines the heights whichever member the grid
sits on, so the horizontal is rebased to ITRF2014 automatically, with a
logged note. This page records what that is for
common products, with the evidence.

## Presets

One preset per product **line**; strip-vs-mosaic and version nuances are
registration questions, recorded in the notes and the sections below (the
frame token is the same either way today — the printed note tells you what
the preset does and does not claim).

| `--vdatum` | resolves to | rationale |
|---|---|---|
| `3dep` | `EPSG:5703` | USGS 3DEP lidar as delivered: NAVD88 orthometric (CONUS; currently GEOID18-realized in the transform chain) |
| `cop30` | `EPSG:3855:itrf2014` | Copernicus GLO-30/90: EGM2008 heights on an ensemble-labeled grid, rebased to ITRF2014 for the horizontal legs |
| `precision3d` | `ellipsoid:g1674` | Vantor states WGS84 G1674 explicitly for Precision3D products; G1674 is aligned to ITRF2008 @ epoch 2005.0 |
| `earthdem` | `ellipsoid:itrf2014` | see below — **unregistered** at every level; expect meters-level vertical bias, coregistration still required |
| `arcticdem` | `ellipsoid:itrf2014` | strips unregistered (~4 m absolute, PGC's figure); mosaic v4.1 anchored to GrIMP v2 (→ IS2/ITRF2014) over Greenland, **Copernicus GLO-30 elsewhere** |
| `rema` | `ellipsoid:itrf2014` | strips unregistered; mosaic v2 aligned to ICESat-2 ATL06 (ITRF2014, ~2019–2021) |

`ellipsoid:<realization>` rebuilds the product's own map projection (UTM or
the polar stereographic grids) on the realized geographic base;
`<vertical>:<realization>` does the same for an orthometric vertical
(`EPSG:3855:itrf2014` = EGM2008 heights on an ITRF2014-rebased grid) —
the two general mechanisms behind every preset.

## Presets are dated snapshots

Every preset records the **source-delivery default as of 2026-08**, and the
resolve-time note says so. These will move:

- **NSRS 2022 modernization**: NAVD88 and the NAD83 realizations are being
  replaced (NATRF2022 + NAPGD2022/GEOID2022). Once 3DEP products start
  shipping in the modernized frames, `3dep` stops being a single answer and
  the product's own vintage decides.
- **PGC registration policy changes per release** (ICESat-2 strip
  registration is promised; ICESat-2 itself moved to ITRF2020 at Release
  007), so `earthdem`/`arcticdem`/`rema` are version-dependent statements.
- **Converted products lie**: many tools convert delivered orthometric
  heights to ellipsoidal (or vice versa) without updating metadata. A
  preset describes the *source delivery*; if your copy has been through a
  conversion pipeline, state what it actually is — and when in doubt, the
  control-based dz itself is the diagnostic (a ~geoid-magnitude offset
  means the vertical datum assumption is wrong).

When product metadata eventually carries version/datum fields worth
trusting, version-aware presets (or reading the product's own declaration)
are the intended upgrade path.

## Why "WGS84 ellipsoid height" is not an answer

**Maxar/DigitalGlobe/Vantor imagery** metadata has said only "WGS 84" since
the 2007 QuickBird product guide; the wording is unchanged in the governing
ISD spec (v1.1.2, 2014) and the XML carries no frame tag at all. The
effective realization is set by the GPS orbit products behind the ephemeris
processing (JPL GIPSY, licensed 2004) and has silently tracked the IGS
frame of the processing era: ITRF2000 → ITRF2005 (2006) → ITRF2008 (2011) →
ITRF2014 (2017) → ITRF2020 (Nov 2022), at cm level between neighbors, with
coordinate epoch = image acquisition date. RPCs — and every DEM built from
them — inherit this. The one place a realization is stated unambiguously is
**Precision3D: WGS84 G1674** (current Vantor product help).

**PGC SETSM products** (EarthDEM / ArcticDEM / REMA) declare
`verticalCoordSys = "WGS84 Ellipsoidal Height"` and an ensemble horizontal;
"ITRF" appears nowhere in PGC's documentation or code. What pins the frame
is *registration*, which differs by product and version:

- ArcticDEM v1–v3 / REMA v1 mosaics: registered to ICESat-1 (GLAS Release-34
  is ITRF2008; epochs 2003–2009).
- REMA v2 mosaic: aligned to ICESat-2 + TanDEM-X PolarDEM → effectively
  ITRF2014 at the IS2 epochs (~2019–2021).
- ArcticDEM v4.1 mosaic: GrIMP v2 (→ IS2/ITRF2014) over Greenland;
  **Copernicus GLO-30** elsewhere.
- Strip DEMs (s2s040/s2s041, current): **no registration**; ICESat-2
  translation vectors are promised but unshipped. s2s030-era strips carried
  optional ICESat-1 offsets in metadata, never applied to the rasters.
- **EarthDEM (v1, v1.1): no ground control or altimetry registration at
  any level** (PGC's words) — heights sit purely in the inherited imagery
  frame, with per-pixel epoch ≈ the median contributing-strip date (the
  tile `mindate`/`maxdate` rasters bound it).

## What the preset does and does not fix

The preset removes the *frame* ambiguity (≤ ~1 dm once a realization is
named) so the transform budget is honest — it does **not** remove the
product's absolute geolocation error. Unregistered strips are ~4 m absolute;
an EarthDEM mosaic averages that down but keeps a spatially correlated
meters-level vertical bias. The control-based dz this package produces is
the measurement of exactly that bias; plan on coregistration afterward. For
multi-year composites, plate motion between contributing strips (~cm/yr ×
year span) is baked into the blend and no single epoch choice can remove
it; when you build your own mosaics from strips, account for per-strip
epochs before or during coregistration and declare one target frame + epoch
for the composite.

ICESat-2 products moved from ITRF2014 to **ITRF2020 at Release 007** (2025):
any future PGC registration release must be checked for the ATL release used
before assuming ITRF2014.
