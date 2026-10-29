# Capabilities

What each part of the package does, in more detail than the README. The module names are
the files under `src/groundcontrol/`.

## Control sources

All sources go through the `fetch_control` dispatcher. Each degrades into a per-source
status report when it is unavailable, empty, or cannot land in the requested frame.

| Source | Key | Notes |
|--------|-----|-------|
| USGS 3DEP checkpoints | `3dep` | National GeoParquet with bbox pushdown. |
| NGS Data Explorer (NDE) | `ngs` | Monumented control, landed per NAD83 realization. |
| OPUS shared solutions | `opus` | Campaign GNSS occupations (`gnss_campaign`): episodic, nothing left on site. The NGS monument-stability tier (A/B versus C/D) is decoded per record. |
| Nevada Geodetic Lab GNSS | `ngl` | Daily `.tenv3` series, `steps.txt` (earthquake-step evidence for epoch propagation), MIDAS velocities. The per-station occupation class (`gnss_cont`, `gnss_semicont`, `gnss_campaign`) is earned by the station's own record; it is an occupation-pattern claim, not a quality tier. Curated-network membership (`networks.py`: NGS CORS, IGS) rides in `raw["networks"]`. |
| FAA NASR runway control | `faa` | Photo-identifiable runway ends, displaced thresholds and helipads from the public-domain 28-day NASR subscription. Per-point position-source provenance (surveyed versus estimated) with AC 150/5300-18C accuracies on the surveyed class. Civil facilities by default (NASR ownership `PU`/`PR`); `--faa-ownership all` widens it. |

## One normalized schema (`schema.py`)

A single canonical control-point GeoDataFrame contract: source and id, height with datum
provenance, `ref_frame`, `frame_epoch`, `coord_epoch` and `measurement_epoch`, per-station
velocities, a durable `epoch_residual_m` honesty column, native coordinates for lossless
re-targeting, and a `transform_id` provenance join key.

## CRS, datum and epoch engine (`crs.py`)

**Transforms.** `get_transformer` is cached, fail-loud and AOI-aware. `transform_points`
is the packaged 3D/4D control-to-DEM-frame transform. `land_horizontal` lands mixed NAD83
realizations per datum through NADCON5, validated against NGS NCAT to under 1 cm;
`landing_crs=` overrides the interim frame for non-CONUS AOIs, so Nepal lands at ITRF.
`get_transformer` refuses a WGS 84 ensemble against a realization until the realization is
declared, because PROJ resolves that pair with a null operation of 2 m stated accuracy.

**Epoch propagation.** `propagate_epoch` moves control in a dynamic frame to the target
epoch with a velocity ladder: per-point MIDAS ENU velocities, then a plate-motion model
(bundled ITRF2020 PMM poles with PB2002 per-point plate assignment), then a no-op with the
velocity-times-dt bound surfaced. Static-frame guards mean plate motion is never
fabricated inside NAD83(2011), and ETRF realizations count as plate-fixed. An
earthquake-step guard means secular velocity never silently carries a point across a
coseismic step: NGL `steps.txt` evidence rides in `raw["eq_steps"]`, a Gorkha-class step
is 0.1 to 2 m that no velocity model contains, and skipped rows surface as `step_blocked`
with an honest unbounded residual. A point that would move but has no velocity from any
tier raises; `on_no_velocity='warn'` keeps it at its epoch with the bound recorded.

**Frame comparison.** `compare_frames` and `assert_same_frame` are the one definition of
"same frame" for every consumer: horizontal datum, datum-blind grid identity, declared
vertical kind (`vertical_kind`: ellipsoidal, gravity, undeclared), WGS 84 ensemble against
realization, and coordinate epoch. A NAVD88 raster against ellipsoidal points, or a
datum-label difference on one grid, is refused by name instead of becoming a
co-registration "offset".

**Dataset identity.** `dataset_frame(path)` reads a raster's CRS and its GDAL coordinate
epoch together, which rasterio does not expose. It says how the epoch was read, keeps
"undeclared" (a reader looked, the file carries no epoch) apart from "unreadable" (no
reader was available), and says whether the CRS sits in the dataset, only in its PAM
sidecar, or in a sidecar overriding a different CRS in the dataset. `is_dynamic_frame`
decides by datum identity when a file format has dropped the WKT2 DYNAMIC property.

**NAD83 realizations.** `nad83_realization` classifies any CRS, EPSG code, WKT or WESM
`horiz_crs` value into one fixed vocabulary of realization tokens, and raises on a
NAD83-family input it cannot place. `realization_shift` predicts the horizontal shift
between two realizations at points through grid-only PROJ routes, with the route recorded,
and raises rather than substituting a null or Helmert route.

**Projected grid to true ENU.** `grid_jacobian`, `grid_convergence` and the vector,
gradient and azimuth conversions give the exact per-point map between grid axes and true
east and north for any projection PROJ can differentiate, with the grids it cannot
describe refused.

## Assessment pipeline (`assess.py`, `groundcontrol-assess`)

`transform_control` applies one direct 3D transform with a declared-CRS guard and a
per-point `xform_acc_m` stated transform budget. `sample_products` samples each product,
`summarize_dz` computes the statistics, and the standard validation figures follow, with
GeoParquet and provenance outputs.

## Accuracy (`accuracy.py`)

Dual-track reporting: robust median and NMAD over all finite residuals, plus the parametric
set the cal/val community expects (mean, standard deviation, RMSE, LE90, LE95, CE90) after
an outlier gate, in the vocabulary of ASPRS Positional Accuracy Standards Ed. 2 and the
USGS Lidar Base Specification. See [accuracy conventions](accuracy_conventions.md).

## DEM sampling (`sample.py`)

Windowed and in-memory paths, bilinear, nearest and radius-neighborhood statistics, and a
`diff` mode. Mosaic gaps are reported, never dropped. `sample_raster` refuses a declared
vertical conflict between the points' heights and the raster's, such as a NAVD88 raster
against ellipsoidal points; an undeclared 2D header is trusted and logged.

## Geodesy utilities (`geodesy.py`)

Programmatic UTM and 3D CRS construction, epoch-pinned PROJ pipelines, and a
vertical-transform preflight where missing geoid grids raise and are never silently zero.

## Figures (`figures.py`, `plot.py`)

The standard per-site control bundle, per-family dz maps with dual-track histograms over
a hillshade (pre-rendered, or computed from the assessed product), MIDAS velocity maps,
and opt-in per-point context contact sheets with RGB ortho, lidar intensity and color
shaded-relief windows around every control point. See the [gallery](gallery.md).

## I/O and provenance (`io.py`)

GeoParquet and CSV export with an embedded, replayable transform-provenance sidecar, and
`read_provenance` to read it back.

## Not yet implemented

- **Fetch-side `--target-crs` and `--target-epoch` landing.** Fetch lands on EPSG:6318
  plus NAVD88; target-frame landing happens in the assess step (`transform_control`).
  Passing a target to `fetch_control` raises.
- **`user_points` source.** Offline CSV and GPKG ingest for vendor checkpoint tables and
  RTK or PPK field campaigns. Designed, not built.
- **Per-point accumulated transform budgets.** `xform_acc_m` covers the assess leg;
  accumulating the per-realization landing legs is next.
- **`epoch_acc_m`.** Velocity-uncertainty propagation through the epoch-propagation tiers.
- **ICESat-2 as a global dense-control source.** Planned; see the
  [sources survey](control_sources_survey.md). For anyone joining it by hand meanwhile:
  v007 products are ITRF2020 at reference epoch 2015.0, not ITRF2014 (releases up to 006
  are ITRF2014). Verify per granule release.
- **The `h_ell` to `h_target` column rename**, which will ship with an alias.
