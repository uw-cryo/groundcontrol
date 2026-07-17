# Quickstart — using groundcontrol from another project

Status: pre-release (`main` branch). The schema is **not frozen** (open decisions
D1–D6 in `plan.md`) — pin a commit and expect column renames until v0.1.

## Install (into an existing env that already has geopandas>=1.0 / pyproj>=3.5)

```bash
pip install --no-deps -e ~/src/groundcontrol     # or: pip install git+https://github.com/uw-cryo/groundcontrol.git@main
```

## 1. Fetch control points for an AOI

```python
from groundcontrol.sources import fetch_control

# AOI: (minlon, minlat, maxlon, maxlat) EPSG:4326, a GeoDataFrame, or a vector-file path
gdf, status = fetch_control("site_aoi.geojson", sources=("3dep", "ngs", "opus", "ngl"))
# -> normalized schema (docs/plan.md), EPSG:6318 horizontal (interim landing),
#    NAVD88 orthometric heights for 3dep/ngs/opus; ELLIPSOIDAL for ngl.
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

CLI equivalent: `groundcontrol-fetch --aoi site_aoi.geojson --out control.parquet`.

## Caveats (current interim state)

- Landing frame is fixed at EPSG:6318 horizontal; `target_crs=`/`target_epoch=` raise
  until the full user-chosen landing ships.
- Accuracy columns: only certain-semantics values populated (`acc_h` = NGS 95% network
  accuracy; OPUS raw peak-to-peak) — conventions under review (D3).
- NGL heights are antenna-reference ellipsoidal heights; `raw["ant_m"]` carries the
  antenna offset (subtract before comparing to a DSM/DTM — and the monument itself may
  be raised above ground).
- Geoid/NADCON5 transforms fetch PROJ grids over the network on first use
  (`PROJ_NETWORK=ON`); they cache locally afterwards.
