# groundcontrol

Fetch ground control points for an arbitrary AOI and assess DEM accuracy — with rigorous
3D CRS / datum / epoch handling as the core competency.

> Given an arbitrary AOI, fetch available control points (with correct datum/epoch handling);
> and given a DEM, sample the control, run the accuracy assessment, and produce the
> analysis + visualization for vertical/horizontal accuracy.

## Status

**Pre-alpha, active development (`0.1.0.dev0`).** The control-point fetch path and the
core sampling/accuracy/CRS machinery are implemented and covered by **201 passing tests**;
the user-selectable target-frame landing and the end-to-end `assess` CLI are not wired up
yet (see [Not yet implemented](#not-yet-implemented)).

> **Repository visibility:** private during initial development and testing within the
> `uw-cryo` org. Intended to become public once the API stabilizes — please keep it internal
> for now.

## What works today

**Control sources** — via the `fetch_control` dispatcher, each degrading gracefully into a
per-source status report:

| Source | Key | Notes |
|--------|-----|-------|
| USGS 3DEP checkpoints | `3dep` | GeoParquet with bbox pushdown |
| NGS Data Explorer (NDE) | `ngs` | monumented control |
| OPUS shared solutions | `opus` | GNSS-derived |
| Nevada Geodetic Lab GNSS | `ngl` | daily `.tenv3` series, `steps.txt`, MIDAS velocities |

- **One normalized schema** (`schema.py`) — a single canonical control-point GeoDataFrame
  contract (source/id, height + datum provenance, `ref_frame`, `frame_epoch` / `coord_epoch` /
  `measurement_epoch`, per-station velocities `vel_e/n/u`, native coordinates for lossless
  re-targeting, and a `transform_id` join key into the export's provenance record).
- **CRS / datum / epoch engine** (`crs.py`) — decimal-year helpers; a cached, fail-loud
  `get_transformer`; `transform_points` (packaged 3D/4D control→DEM-frame transform);
  `land_horizontal` (per-datum landing of mixed NAD83 realizations); dynamic-frame detection;
  NGS-datum→EPSG mapping.
- **Stage-2 epoch propagation** (`propagate_epoch`) — the intra-frame velocity·Δt move PROJ
  can't do on its own: a velocity ladder of per-point ENU (MIDAS) → plate-motion model
  (`EulerPoleModel` / `PlateMotionModel` protocol) → no-op, with the un-propagated velocity·Δt
  bound surfaced. Every displacement is cross-checked against an independent pyproj ENU→ECEF
  oracle to < 0.1 mm.
- **Velocity interpolation** (`velocity.py`) — distance-weighted interpolation / fill of MIDAS
  station velocities onto arbitrary points.
- **DEM sampling** (`sample.py`) — `sample_raster` with windowed and in-memory paths,
  bilinear / nearest / radius-neighborhood statistics, and a `diff` mode.
- **Accuracy** (`accuracy.py`) — robust `med_nmad`, `robust_normalize`, `resid_stats`.
- **I/O + provenance** (`io.py`) — GeoParquet / CSV export with an embedded, replayable
  transform-provenance sidecar and compound-CRS export; `read_provenance`.
- **Plotting** (`plot.py`) — control maps, hillshade, RdYlBu residual `dh` maps, MIDAS
  velocity-vector maps, and scalebar helpers.
- **CLI** — `groundcontrol-fetch` (AOI → control GeoParquet/CSV + provenance) is functional.

## Not yet implemented

- **User-chosen `--target-crs` / `--target-epoch` landing.** Fetch currently lands on an
  interim frame — **EPSG:6318 (NAD83(2011)) + NAVD88**; passing a target CRS/epoch raises.
- **`groundcontrol-assess` CLI** — the end-to-end DEM → fetch → sample → stats → figures
  wrapper is still a stub.
- **ICESat-2 as a global dense-control source** — planned (see the sources survey), not built.
- Assorted documented debts (accuracy semantics `D3`, `height_datum` `D4`, and a `coord_20`
  CORS end-to-end truth-test fixture).

## Install (development)

```bash
pip install -e ".[dev]"
pytest -m "not network"   # offline suite; drop the marker to include network-dependent tests
ruff check .              # lint (line-length 100)
```

## Usage

Fetch control for an AOI (bbox or vector file), export GeoParquet + provenance sidecar:

```bash
# bbox is minx,miny,maxx,maxy in EPSG:4326 (lon/lat); use --aoi=... for negative longitudes
groundcontrol-fetch --aoi=-115.3,36.0,-114.9,36.3 --sources 3dep,ngs,opus --out control.parquet
```

From Python (see [`docs/quickstart.md`](docs/quickstart.md) for the full pattern):

```python
from groundcontrol.sources import fetch_control
from groundcontrol import io, sample, accuracy

gdf, status = fetch_control("aoi.geojson", sources=("3dep", "ngs", "opus"))
io.write(gdf, "control.parquet", status=status)
```

## Package layout

```
src/groundcontrol/
  schema.py        canonical control-point GeoDataFrame contract
  crs.py           CRS/datum/epoch transforms, landing, epoch propagation
  velocity.py      MIDAS velocity interpolation / fill
  sample.py        raster sampling (windowed / in-memory / radius)
  accuracy.py      robust residual statistics
  io.py            GeoParquet/CSV export + transform provenance
  plot.py          control / residual / velocity figures
  cli.py           console entry points (groundcontrol-fetch, -assess)
  sources/         3dep, ngs/opus, ngl providers + fetch_control dispatcher
```

## Documentation

- [`docs/plan.md`](docs/plan.md) — full design and migration plan
- [`docs/crs_implementation.md`](docs/crs_implementation.md) — verified CRS/epoch directives,
  transform-provenance spec, and validation-fixture design
- [`docs/quickstart.md`](docs/quickstart.md) — consuming groundcontrol from another project
- [`docs/accuracy_conventions.md`](docs/accuracy_conventions.md) — accuracy semantics (WIP)
- [`docs/control_sources_survey.md`](docs/control_sources_survey.md) — candidate sources
  beyond the implemented set

## Roadmap / next steps

- **Set up GitHub Actions CI** — run the offline test suite (`pytest -m "not network"`) and
  `ruff check` on push / PR. To be added in the next phase.
- Wire user-selectable target CRS/epoch landing through `fetch_control` and the CLI.
- Implement the `groundcontrol-assess` end-to-end CLI.
- Add the `coord_20` CORS end-to-end validation fixture.

## Origin

Core functionality extracted and generalized from prior UW-cryo project code
(`casagrande` NGS/OPUS fetching + a private DEM-accuracy toolkit), 2026.
Datum/epoch recipes follow
[uw-cryo/3D_CRS_Transformation_Resources](https://github.com/uw-cryo/3D_CRS_Transformation_Resources).
