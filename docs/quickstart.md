# Quickstart — using groundcontrol from another project

Status: **v0.1.2**, public and citable
([10.5281/zenodo.21846300](https://doi.org/10.5281/zenodo.21846300)). Pre-alpha: the
schema is **not frozen** (open decisions D1–D6 in `plan.md`), so **pin the tag** and
expect column renames before v1.

Install from a git tag for now. A **conda-forge package (`groundcontrol`) is the planned
distribution channel**, and that is what downstream conda-forge packages should depend
on once it lands. PyPI is blocked for the moment: an unrelated, abandoned package holds
the `groundcontrol` name there (reclamation in progress), and PyPI also rejects
near-identical names like `ground-control` as too similar — and since PyPI rejects
direct-URL (`git+https://…`) dependencies, a package published on PyPI cannot declare
this one as a requirement until the name resolves.

## Install

```bash
pip install git+https://github.com/uw-cryo/groundcontrol.git@v0.1.2
```

> ⚠️ **Do not run `pip install groundcontrol`.** An unrelated package of that name
> (satellite orbit propagation, last released 2022) occupies it on PyPI, and it installs
> without error while silently not being this library. Use the git tag above.

Into an existing env that already satisfies every entry in `[project] dependencies` of
`pyproject.toml`, add `--no-deps` so pip leaves the solved env alone, then verify — the
second line imports every runtime dependency, which is exactly the check `--no-deps` skips:

```bash
pip install --no-deps git+https://github.com/uw-cryo/groundcontrol.git@v0.1.2
python -c "import groundcontrol.assess, groundcontrol.sample, groundcontrol.sources, groundcontrol.figures, pyarrow.parquet, scipy.interpolate, matplotlib_scalebar.scalebar"
```

`pyarrow` is the known trap: conda-forge's minimal `pyarrow-core` owns the `pyarrow`
metadata (pip reports the requirement satisfied) but lacks the `libparquet` shared library
that `pyarrow._parquet` links against, so `import pyarrow.parquet` fails and with it every
parquet path (the 3DEP source, control export and cache, `read_provenance`). Install the
full `pyarrow` package.

For a **pixi** project, a git dependency is first-class — no index required:

```toml
[pypi-dependencies]
groundcontrol = { git = "https://github.com/uw-cryo/groundcontrol.git", tag = "v0.1.2" }
```

For co-development against a local checkout, `pip install --no-deps -e /path/to/groundcontrol`
still works — but pin the tag in anything reproducible.

## 0. Inputs at a glance

- **AOI** — one contract everywhere (`groundcontrol.aoi.resolve_aoi`): a
  `(minlon, minlat, maxlon, maxlat)` bbox in EPSG:4326; a vector file (GeoJSON
  preferred; anything OGR reads, plus GeoParquet; any CRS); a **gridded elevation
  raster** (DEM/DSM/DTM, any GDAL format) whose valid-data footprint becomes the AOI
  (band 1's nodata mask at ≤1024 px, so edge membership is approximate; untagged NaN
  counts as valid); or an in-memory GeoDataFrame / GeoSeries (any CRS) or shapely
  geometry (taken as lon/lat).
- **Product** — the raster(s) to assess, keyed by a short name; a name containing
  `DTM` gets the bare-earth rules (VVA checkpoints apply), anything else the surface
  rules.
- **Target CRS** — the product's *3D* frame: horizontal + height datum. Never inferred
  from a 2D raster tag; a compound/3D embedded CRS is accepted as declared. Wrong
  vertical datum = geoid-sized bias (~−30 m at Casa Grande), which the stats will show
  you and the library will not paper over.

CLI equivalents (`groundcontrol-fetch`, `groundcontrol-assess`) accept the same forms;
the README's *Usage* section lists the outputs.

## 1. Fetch control points for an AOI

```python
from groundcontrol.sources import fetch_control

# AOI: bbox tuple, vector-file path, DEM/DSM/DTM raster path (footprint), or GeoDataFrame
gdf, status = fetch_control("site_aoi.geojson", sources=("3dep", "ngs", "opus", "ngl"))
gdf, status = fetch_control("dsm.tif", sources=("3dep", "ngs", "opus", "faa"))
# -> normalized schema (docs/plan.md), EPSG:6318 horizontal (interim landing),
#    NAVD88 orthometric heights for 3dep/ngs/opus/faa; ELLIPSOIDAL for ngl.
#    status: {source: {n_rows, error}} — per-source failures degrade gracefully.
```

Filtering that matters (see `docs/accuracy_conventions.md` for why):

```python
usable = gdf[gdf["height"].notna() & (gdf["vertical_crs"] == "EPSG:5703")]
best   = usable[usable["point_type"].isin(["NVA", "VVA"])]          # 3DEP checkpoints
# NGS monuments: quality lives in the raw JSON (vertSource) — GPS OBS ≈ checkpoint-grade;
# leveling-era marks are vintage-limited in deforming areas. BVA = bathymetry (exclude).
```

## 2. Transform into the DEM's frame (heights NEVER via .to_crs)

The DEM's 3D CRS + epoch are **your declared inputs** (embedded WKTs often lie — check the
delivery metadata). One 3D transformer on arrays, per `docs/crs_implementation.md`:

```python
from groundcontrol.crs import transform_points

pts = transform_points(gdf, dem_crs_3d, tt=dem_epoch)   # tt per the (provisional) D6 rule
# source vertical datum is inferred from the uniform vertical_crs column, or pass
# source_crs="EPSG:6318+5703" explicitly; anything ambiguous raises (never guessed).
# Returns a copy: geometry in the DEM frame, `height` transformed (HAE), fail-loud.
```

## 2b. Epoch propagation (stage 2) — move points to the DEM's epoch

Stage 1 above changes *frame*; it does not move a point from its `coord_epoch` to
another epoch within a dynamic frame. That intra-frame move (`x += vel·Δt`, see
`docs/crs_implementation.md` §1) is stage 2:

```python
from groundcontrol.crs import propagate_epoch, ITRF2020PMM

# velocity ladder: per-point MIDAS (vel_e/n/u) -> plate model -> no-op + bound
prop = propagate_epoch(gdf, target_epoch=2020.0,
                       plate_model=ITRF2020PMM("NOAM"))   # single-plate AOI
# AOI straddles a plate boundary (San Andreas)? ITRF2020PMM(None) assigns each
# point via the bundled PB2002 boundaries. Rows without any usable velocity stay
# put; their velocity·Δt bound lands in the durable `epoch_residual_m` column
# (feeds the accuracy budget) plus the attrs['epoch_propagation'] report.
```

Composition order with stage 1 (the two orders commute to mm): run stage 2 in the
(geographic) dynamic source frame first, then land:

```python
prop = propagate_epoch(gdf, target_epoch=2020.0, plate_model=ITRF2020PMM("NOAM"))
landed = land_horizontal(prop, target="EPSG:6318")   # tt = coord_epoch = 2020.0
```

## 3. Sample the DEM + accuracy stats

```python
from groundcontrol.sample import sample_raster
from groundcontrol.accuracy import resid_stats, robust_normalize

out = sample_raster(pts, "dem.tif", col="height", diff=True)   # bilinear;
# points must be in the raster CRS (asserted). radius=3.5 -> neighborhood median +
# <name>_nmad (roughness flag) + _n; see the docstring for radius-choice guidance.
dh = out["<raster> minus height"]
stats = resid_stats(dh[robust_normalize(out, dh.name)])   # n/median/mean/nmad/std/rmse
```

## 4. Export with provenance

```python
from groundcontrol import io
io.write(gdf, "control.parquet", status=status)   # + .provenance.json sidecar,
# provenance embedded in the GeoParquet metadata; CRS promoted to compound
# (e.g. NAD83(2011)+NAVD88) when the heights' vertical datum is uniform.
```

CLI equivalent: `groundcontrol-fetch --aoi site_aoi.geojson --out control.parquet`
(`--aoi` also takes a bbox or a DEM).

## 5. Assess a DEM end to end (bring your own)

`assess_products` is steps 2–4 in one call, writing the assessed GeoParquet, the
per-segment dual-track stats and the validation figures (README *What you get back*):

```python
from groundcontrol.assess import assess_products

target_crs = open("dsm_frame.wkt").read()      # the product's 3D CRS (see §0)
sampled, stats, artifacts = assess_products(
    gdf, {"DSM": "dsm.tif", "DTM": "dtm.tif"}, target_crs,
    outdir="out", site_name="mysite")
# aoi=None -> figures unclipped; pass a path/GeoDataFrame (any CRS; a DEM path
# works too) to clip + outline. hs=None -> a multidirectional hillshade is computed
# from each product (figures.hillshade_from_raster); pass {"DSM": "dsm_hs.tif"}
# for pre-rendered ones on very large mosaics.
```

CLI: `groundcontrol-assess --product DSM=dsm.tif --product DTM=dtm.tif
--target-crs dsm_frame.wkt --outdir out/` (AOI = product footprints, site name = file
stem, hillshade computed).

## 6. Per-point context contact sheets

One strip of image windows per control point — RGB ortho or web basemap, lidar
intensity, color shaded relief — so the physical setting of every point (threshold
paint, roof mount, bare ground) is reviewable at a glance. `assess_products` writes
these automatically (`figures.context_sheets`), broken out by what came back — CORS /
OPUS / other GNSS / FAA runway / 3DEP NVA / 3DEP VVA — at two tiers (120 m context +
30 m native-pixel). Panels adapt to the available layers: RGB imagery — an `rgb=`
ortho and/or the `basemap=` web provider (Esri default, network, credited on the
sheet; `None` offline) — then `intensity=` grayscale when given, then one
shaded-relief panel per product. An AOI-only fetch gets RGB-only sheets
(`groundcontrol-fetch --context-sheets`). Call `point_context_gallery` directly for
custom layer stacks or other subsets. Layers are
`(tag, path, kind)` with `kind` in `"rgb"` / `"gray"` / `"relief"`; a DEM alone gives a
relief-only sheet, and an `"rgb"` path may be a list (fallback chain, e.g. ortho then a
web basemap). Sheets paginate at `max_rows`.

```python
from groundcontrol.figures import point_context_gallery

layers = [("ortho 0.5 m", "ortho.tif", "rgb"),
          ("intensity 1 m", "intensity.tif", "gray"),
          ("DSM relief", "dsm.tif", "relief")]
pages = point_context_gallery(sampled[sampled["source"] == "faa"], layers, "out",
                              "mysite", half_m=60, subset_tag="faa_runway",
                              class_col="point_type")
# -> out/mysite_faa_runway_gallery_120m[_pN].png
```

## Caveats (current interim state)

- Landing frame is fixed at EPSG:6318 horizontal; `target_crs=`/`target_epoch=` raise
  until the full user-chosen landing ships.
- Accuracy columns: only certain-semantics values populated (`acc_h` = NGS 95% network
  accuracy; OPUS raw peak-to-peak) — conventions under review (D3).
- NGL heights are antenna-reference ellipsoidal heights; `raw["ant_m"]` carries the
  antenna offset (subtract before comparing to a DSM/DTM — and the monument itself may
  be raised above ground).
- The product's vertical datum is YOUR declaration (`target_crs`): 3DEP-derived rasters
  may carry NAVD88 or ellipsoidal heights under the same 2D projected CRS tag. If every
  segment of `dz_stats` shows the same ~±30 m offset, the declared height datum is
  wrong, not the DEM.
- Geoid/NADCON5 transforms fetch PROJ grids over the network on first use
  (`PROJ_NETWORK=ON`); they cache locally afterwards.
