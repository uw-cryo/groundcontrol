# groundcontrol

[![CI](https://github.com/uw-cryo/groundcontrol/actions/workflows/ci.yml/badge.svg)](https://github.com/uw-cryo/groundcontrol/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21846300.svg)](https://doi.org/10.5281/zenodo.21846300)

Fetch ground control points for an arbitrary AOI and assess DEM accuracy — with rigorous
3D CRS / datum / epoch handling as the core competency.

> Given an arbitrary AOI, fetch available control points (with correct datum/epoch handling);
> and given a DEM, sample the control, run the accuracy assessment, and produce the
> analysis + visualization for vertical/horizontal accuracy.

![control map](docs/img/casagrande_large_control_map.png)

## Status

**v0.1.2 — pre-alpha, quiet release.** The fetch → transform → sample → statistics →
figures pipeline works end to end (CLI + Python API) and is covered by **427 offline
tests** run in CI on Python 3.10/3.12, with the geodesy core additionally adversarially
audited (independent review agents; math cross-checked against external oracles). The API
may still move between minor versions — pin the tag if you build on it, and expect sharp
edges to be documented rather than hidden: the library's design principle is **fail-loud**
(no silent datum guesses, no ballpark transforms, no fabricated epoch motion).

## What works today

**Control sources** — via the `fetch_control` dispatcher, each degrading gracefully into a
per-source status report:

| Source | Key | Notes |
|--------|-----|-------|
| USGS 3DEP checkpoints | `3dep` | national GeoParquet with bbox pushdown |
| NGS Data Explorer (NDE) | `ngs` | monumented control, per-realization datum landing |
| OPUS shared solutions | `opus` | campaign GNSS occupations (`gnss_campaign`): episodic, nothing left on site; NGS monument-stability tier (A/B vs C/D) decoded per record |
| Nevada Geodetic Lab GNSS | `ngl` | daily `.tenv3` series, `steps.txt`, MIDAS velocities; per-station occupation class earned by the station's own record (`gnss_cont` / `gnss_semicont` / `gnss_campaign`) — an occupation-pattern claim, not a quality tier |
| FAA NASR runway control | `faa` | photo-identifiable runway ends, displaced thresholds, helipads from the public-domain 28-day NASR subscription; per-point position-source provenance (surveyed vs estimated) with AC 150/5300-18C accuracies on the surveyed class |

- **One normalized schema** (`schema.py`) — a single canonical control-point GeoDataFrame
  contract: source/id, height + datum provenance, `ref_frame`, `frame_epoch` / `coord_epoch` /
  `measurement_epoch`, per-station velocities, a durable `epoch_residual_m` honesty column,
  native coordinates for lossless re-targeting, and a `transform_id` provenance join key.
- **CRS / datum / epoch engine** (`crs.py`) — cached, fail-loud, AOI-aware `get_transformer`;
  `transform_points` (packaged 3D/4D control→DEM-frame transform); `land_horizontal`
  (per-datum landing of mixed NAD83 realizations via NADCON5, validated against NGS NCAT
  to < 1 cm); **stage-2 epoch propagation** (`propagate_epoch`) with a velocity ladder of
  per-point MIDAS ENU → plate-motion model (bundled **ITRF2020 PMM** poles + PB2002
  per-point plate assignment) → no-op with the velocity·Δt bound surfaced; static-frame
  guards so plate motion is never fabricated inside NAD83(2011).
- **Assessment pipeline** (`assess.py` + `groundcontrol-assess`) — `transform_control`
  (one direct 3D transform, declared-CRS guard, per-point `xform_acc_m` stated transform
  budget) → `sample_products` → `summarize_dz` → standard validation figures, with
  GeoParquet + provenance outputs.
- **Accuracy** (`accuracy.py`) — dual-track reporting: robust median/NMAD over all finite
  residuals plus the parametric set the cal/val community expects (mean, σ, RMSE,
  LE90/LE95, CE90) after an outlier gate, per ASPRS Positional Accuracy Standards Ed. 2 /
  USGS Lidar Base Specification vocabulary (see `docs/accuracy_conventions.md`).
- **DEM sampling** (`sample.py`) — windowed and in-memory paths, bilinear / nearest /
  radius-neighborhood statistics, `diff` mode; mosaic gaps reported, never dropped.
- **Geodesy utilities** (`geodesy.py`) — programmatic UTM/3D CRS construction,
  epoch-pinned PROJ pipelines, vertical-transform preflight (missing geoid grids raise,
  never silently zero).
- **Figures** (`figures.py`, `plot.py`) — standard per-site control bundle, per-family
  dz maps + dual-track histograms over a hillshade (pre-rendered, or computed from the
  assessed product), MIDAS velocity maps, and opt-in per-point context contact sheets
  (RGB ortho / lidar intensity / color shaded relief windows around every control point)
  ([gallery](docs/gallery.md)).
- **I/O + provenance** (`io.py`) — GeoParquet / CSV export with an embedded, replayable
  transform-provenance sidecar; `read_provenance`.

## Usage

Two entry points, one input contract.

**AOI** (`--aoi`, `fetch_control(aoi)`) — any of:

| Form | Example | Notes |
|------|---------|-------|
| bbox string | `--aoi=-115.3,36.0,-114.9,36.3` | `minx,miny,maxx,maxy` in EPSG:4326 lon/lat; use the `=` form for negative longitudes |
| vector file | `--aoi site.geojson` | GeoJSON preferred; any OGR-readable format (GPKG, Shapefile, KML, FlatGeobuf, ...) and GeoParquet; any CRS; multiple features dissolve into one AOI |
| elevation raster | `--aoi dsm.tif` | DEM / DSM / DTM in any GDAL format (GeoTIFF, COG, VRT, ...): the AOI is the raster's **valid-data footprint** (band 1's nodata/alpha mask, read at ≤1024 px so edge membership is approximate; untagged NaN counts as valid — set the nodata tag), reprojected from the raster CRS |
| in memory (Python) | `fetch_control(gdf, ...)` | GeoDataFrame / GeoSeries / shapely geometry |

**Products** (`--product NAME=PATH`, repeatable) — gridded elevation rasters to assess, in
any GDAL format. A `NAME` containing `DTM` is assessed under the bare-earth rules (VVA
checkpoints validate it); any other name (`DSM`, `DEM`, ...) under the surface rules.
The assessment needs the product's **3D CRS** — horizontal *and* height datum
(`--target-crs`): an EPSG compound like `EPSG:6341+5703` (NAD83(2011) / UTM 12N +
NAVD88), inline WKT, or a `.wkt`/`.prj` file. A product whose embedded CRS already
declares its vertical datum (compound or 3D) needs nothing; a 2D raster CRS is refused,
never guessed — a wrong or assumed vertical datum shows up as a geoid-sized bias (about
−30 m at Casa Grande), which is the error class this library exists to prevent.

### Fetch control for an AOI

```bash
# runs as-is: Las Vegas bbox, live 3DEP/NGS/OPUS fetch
groundcontrol-fetch --aoi=-115.3,36.0,-114.9,36.3 --sources 3dep,ngs,opus --out control.parquet
# the same for a polygon, or for wherever a DEM has data (+ FAA runway control)
groundcontrol-fetch --aoi site.geojson --out control.parquet
groundcontrol-fetch --aoi dsm.tif --sources 3dep,ngs,opus,faa --out control.parquet
```

### Assess your own DEM

Bring-your-own-DEM is the main use case: the product is the only required input
besides its frame. The AOI defaults to the product's footprint, the site name to the
file stem, and the figure hillshade is computed from the product itself.

```bash
groundcontrol-assess --product DSM=dsm.tif --target-crs dsm_frame.wkt --outdir out/
```

`dsm.tif` and `dsm_frame.wkt` are placeholders for your product and its 3D CRS (for
ellipsoidal heights on a UTM grid, `groundcontrol.geodesy.build_utm_nad83_2011_3d(32612)`
+ `write_crs_file` produce the WKT). Optional: several `--product` entries (a DSM/DTM
pair), `--aoi` to restrict or outline the area, `--sources` (default `3dep,ngs,opus`;
add `ngl`, `faa`), `--control` to reuse a fetched cache, `--hs NAME=PATH` for a
pre-rendered hillshade on very large mosaics, `--site-name`, `--target-epoch`, and
sampling `--method`/`--radius`.

### What you get back

Everything lands in `--outdir`, prefixed by the site name:

| File | Contents |
|------|----------|
| `<site>_control.parquet` + `.provenance.json` | fetched control in the normalized schema (`schema.py`), EPSG:6318 + NAVD88 landing, per-source status; reused on the next run |
| `<site>_assessed.parquet` + `.provenance.json` | control landed in the product frame (`h_ell`, per-point `xform_acc_m` transform budget) with `h_<NAME>` and `dh_<NAME>_before` (product − control) per product; unsampled points (nodata / mosaic gaps) stay as NaN, never dropped |
| `<site>_dz_stats.csv` | one row per product × control segment (3DEP NVA/VVA, GNSS occupation classes, NGS monuments, ...): `n`, `n_valid`, `n_out`, robust `median_m`/`nmad_m`, parametric `mean_m`/`std_m`/`rmse_m`/`le90_m`/`le95_m` after a 3·NMAD gate, `xform_acc_m`, and `applies` (whether that segment validates that product class) |
| `<site>_validation_dz_<NAME>.png` | per product: dz map over the hillshade + dual-track histograms for the survey-grade segments and the NGS monuments ([example](docs/gallery.md#2b-the-clis-own-output-bring-your-own-dem)) |
| `<site>_<subset>_gallery_<tier>[_pN].png` | per-point context contact sheets, broken out by what came back — `cors`, `opus`, `gnss_other`, `faa_runway`, `3dep_nva`, `3dep_vva` — at the two standard tiers (120 m context, 30 m native-pixel). Panels adapt to the available layers: RGB imagery (your `--rgb` ortho and/or `--basemap` web tiles, Esri by default, credited on the sheet; `--basemap none` for offline) \| `--intensity` grayscale when given \| shaded relief per product ([example](docs/gallery.md#7-per-point-context-contact-sheets)). `groundcontrol-fetch --context-sheets` writes the same sheets for an AOI-only fetch (RGB panels only — no DEM required) |

The CLI also prints the per-source row counts, the selected transform with its stated
accuracy, and the stats table to stderr.

From Python (see [`docs/quickstart.md`](docs/quickstart.md) for the full pattern,
including the per-point context contact sheets):

```python
from groundcontrol.sources import fetch_control
from groundcontrol.assess import assess_products
from groundcontrol import io

control, status = fetch_control("dsm.tif", sources=("3dep", "ngs", "opus", "faa"))
io.write(control, "control.parquet", status=status)
sampled, stats, artifacts = assess_products(
    control, {"DSM": "dsm.tif"}, target_crs=open("dsm_frame.wkt").read(),
    outdir="out", site_name="mysite")
```

What the standard outputs look like on a real site: **[docs/gallery.md](docs/gallery.md)**.

![3DEP checkpoint dz](docs/img/casagrande_large_dz_3dep_DTM.png)

![FAA runway control context](docs/img/casagrande_faa_runway_gallery_120m.png)

## Not yet implemented

- **Fetch-side `--target-crs`/`--target-epoch` landing** — fetch lands on
  EPSG:6318 + NAVD88; target-frame landing happens in the assess step
  (`transform_control`). Passing a target to `fetch_control` raises.
- **`user_points` source** — offline CSV/GPKG ingest (vendor checkpoint tables, RTK/PPK
  field campaigns); designed, not built.
- **Per-point accumulated transform budgets** — `xform_acc_m` currently covers the assess
  leg; accumulating the per-realization landing legs is next.
- **`epoch_acc_m`** — velocity-uncertainty propagation through the stage-2 tiers.
- **ICESat-2 as a global dense-control source** — planned (see the sources survey).
  Frame note for anyone joining it by hand meanwhile: v007 products are **ITRF2020**
  (reference epoch 2015.0), not ITRF2014 (releases ≤006 are ITRF2014) — verify per
  granule release.

## Install

Install from a git tag, and **pin it**: the schema is not frozen, so column names may still
change between minor versions. A **conda-forge package is the planned distribution
channel**; PyPI is blocked while an unrelated, abandoned package holds the
`groundcontrol` name (reclamation in progress — PyPI also rejects near-identical names
like `ground-control` as too similar).

> ⚠️ `pip install groundcontrol` silently installs that **unrelated** PyPI package, not
> this library. Use the git URL below.

```bash
pip install git+https://github.com/uw-cryo/groundcontrol.git@v0.1.2
```

Into an env that already satisfies every entry in `[project] dependencies` of
`pyproject.toml`, add `--no-deps` so pip leaves the solved environment alone — then
verify, because `--no-deps` skips exactly that check. `pyarrow` is the known trap:
conda-forge's minimal `pyarrow-core` owns the `pyarrow` metadata (pip reports the
requirement satisfied) but lacks the `libparquet` shared library that `pyarrow._parquet`
links against, so `import pyarrow.parquet` fails and with it every parquet path here (the
3DEP source, control export and cache, `read_provenance`); install the full `pyarrow`
package. This line imports every runtime dependency:

```bash
python -c "import groundcontrol.assess, groundcontrol.sample, groundcontrol.sources, groundcontrol.figures, pyarrow.parquet, scipy.interpolate, matplotlib_scalebar.scalebar"
```

See [`docs/quickstart.md`](docs/quickstart.md) for the downstream-consumer recipe.

For development:

```bash
pip install -e ".[dev]"
pytest -m "not network"   # offline suite; drop the marker to include live-API tests
ruff check .              # lint (line-length 100)
```

## Package layout

```
src/groundcontrol/
  schema.py        canonical control-point GeoDataFrame contract
  crs.py           CRS/datum/epoch transforms, landing, stage-2 epoch propagation, PMM
  geodesy.py       CRS construction, epoch-pinned pipelines, vertical preflight
  velocity.py      MIDAS velocity interpolation / fill
  aoi.py           AOI contract: bbox / vector file / raster footprint / GeoDataFrame
  assess.py        transform -> sample -> stats assessment pipeline
  sample.py        raster sampling (windowed / in-memory / radius)
  accuracy.py      dual-track residual statistics (robust + ASPRS/LBS parametric)
  io.py            GeoParquet/CSV export + transform provenance
  figures.py       standard per-site control + validation figure bundles, contact sheets
  plot.py          map/velocity/hillshade plotting primitives
  cli.py           console entry points (groundcontrol-fetch, groundcontrol-assess)
  sources/         3dep, ngs/opus, ngl, faa providers + fetch_control dispatcher
  data/            bundled ITRF2020 PMM poles + PB2002 plate boundaries (ODC-By 1.0)
```

## Documentation

- [`docs/gallery.md`](docs/gallery.md) — standard outputs on a real site
- [`docs/quickstart.md`](docs/quickstart.md) — consuming groundcontrol from another project
- [`docs/accuracy_conventions.md`](docs/accuracy_conventions.md) — accuracy semantics and
  reporting conventions
- [`docs/crs_implementation.md`](docs/crs_implementation.md) — verified CRS/epoch directives,
  transform-provenance spec, and validation-fixture design
- [`docs/plan.md`](docs/plan.md) — full design and migration plan
- [`docs/control_sources_survey.md`](docs/control_sources_survey.md) — candidate sources
  beyond the implemented set

## Citation

Archived on Zenodo — the concept DOI below always resolves to the latest release, and each
release also gets its own version DOI. Machine-readable metadata lives in
[`CITATION.cff`](CITATION.cff) (GitHub's *Cite this repository* button reads it).

> Shean, D. (2026). *groundcontrol* (v0.1.2). Zenodo. https://doi.org/10.5281/zenodo.21846300

## Origin

Core functionality extracted and generalized from prior UW-cryo project code, 2026.
Datum/epoch recipes follow
[uw-cryo/3D_CRS_Transformation_Resources](https://github.com/uw-cryo/3D_CRS_Transformation_Resources).
Bundled plate-boundary data: Bird (2003) PB2002 via
[fraxen/tectonicplates](https://github.com/fraxen/tectonicplates) (ODC-By 1.0); plate poles:
Altamimi et al. (2023) ITRF2020 PMM.
