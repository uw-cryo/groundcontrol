# Quickstart — using groundcontrol from another project

Status: pre-release (`increment-1` branch). The schema is **not frozen** (open decisions
D1–D6 in `plan.md`) — pin a commit and expect column renames until v0.1.

## Install (into an existing env that already has geopandas>=1.0 / pyproj>=3.5)

```bash
pip install --no-deps -e ~/src/groundcontrol     # or: pip install git+https://github.com/uw-cryo/groundcontrol.git@increment-1
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

## 3. Sample the DEM + accuracy stats

```python
from groundcontrol.sample import sample_raster
from groundcontrol.accuracy import resid_stats, robust_normalize

pts = gdf.copy(); pts["h_ell"] = h_ell   # points must be in the raster CRS (asserted)
out = sample_raster(pts_in_dem_crs, "dem.tif", col="h_ell", diff=True)   # bilinear
# radius=3.5 -> neighborhood median + <name>_nmad (roughness flag) + _n; see docstring
dh = out["<raster> minus h_ell"]
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
