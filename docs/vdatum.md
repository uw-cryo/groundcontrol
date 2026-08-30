# Vertical datum field guide

`groundcontrol-assess --vdatum` joins a stated vertical datum to a product's
own 2D horizontal CRS to build the unambiguous 3D target frame. A 2D CRS is
never trusted for heights (it says nothing about them), and the **WGS 84
ensemble** (`EPSG:326xx`/`327xx`, `EPSG:4326`) is refused outright: it is
~2 m of deliberate ambiguity spanning realizations from Transit to G2296,
and PROJ's best transform chains to it are meter-class and member-agnostic.
State what the heights are actually on. This page records what that is for
common products, with the evidence.

## Presets

| `--vdatum` | resolves to | rationale |
|---|---|---|
| `3dep` | `EPSG:5703` | USGS 3DEP lidar is NAVD88 orthometric (CONUS) |
| `precision3d` | `ellipsoid:g1674` | Vantor states WGS84 G1674 explicitly for Precision3D products; G1674 is aligned to ITRF2008 @ epoch 2005.0 |
| `earthdem` | `ellipsoid:itrf2014` | see below — **unregistered**; expect meters-level vertical bias, coregistration still required |
| `arcticdem-strip`, `rema-strip` | `ellipsoid:itrf2014` | unregistered SETSM strips, ~4 m absolute (PGC's figure) |
| `rema-mosaic` | `ellipsoid:itrf2014` | REMA v2 is aligned to ICESat-2 ATL06 (ITRF2014, ~2019–2021) |
| `arcticdem-mosaic` | `ellipsoid:itrf2014` | v4.1: Greenland tiles aligned to GrIMP v2 (itself ICESat-2/ITRF2014); tiles **outside Greenland are anchored to Copernicus GLO-30**, which carries its own ~meter-level absolute calibration |

`ellipsoid:<realization>` rebuilds the product's own map projection (UTM or
the polar stereographic grids) on the realized geographic base — the general
mechanism behind every preset.

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
