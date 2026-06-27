# Migrate control-point preparation into `uw-cryo/groundcontrol`

## Context

Ground-control preparation logic is currently scattered across two project repos and is hard
to reuse for a new AOI:

- **`~/src/casagrande`** — NGS/OPUS point fetching, AOI-aware CRS/datum transforms, epoch
  handling, and KML/CSV export, all living inside the `noaa_ngs_gps_api_CasaGrande.ipynb`
  notebook (plus a few generic helpers in scripts).
- **A private DEM-accuracy repo** — USGS 3DEP checkpoint loading and the accuracy-assessment primitives
  (`sample_raster`, `resid_stats`, `med_nmad`, robust outlier filtering, residual figures),
  partly library-ized in a `utils` / `plot` module and partly stuck in
  per-site notebooks (Las Vegas, San Francisco).

Several existing uw-cryo repos are **reused/referenced, not duplicated** (the user explicitly
does not want to re-implement what these already cover):

- **`~/src/3D_CRS_Transformation_Resources`** (canonical:
  https://github.com/uw-cryo/3D_CRS_Transformation_Resources) — a Jupyter Book / Pixi reference
  site (NOT pip-installable). Authoritative on datum/epoch/geoid handling: documented PROJ-pipeline
  recipes (e.g. NAD83(2011) NAVD88 → ITRF2014 via Helmert `t_epoch=2010` + GEOID18 `vgridshift`),
  custom 3D-UTM `.wkt` files under `3dep/`, hosted global-DEM reproject VRTs under `globaldems/`
  (e.g. `COP30_hh_7912.vrt`), geoid COG URLs at cdn.proj.org, theory in `essentials/`.
- **`~/src/fetch_dem`** (small pip pkg `fetch_dem`, `opentopo_utils.get_dem()` + CLI) — fetches
  global DEMs (COP30/COP90/NASADEM/SRTM/ALOS…) via the **OpenTopography API** and, importantly,
  carries explicit horizontal+vertical datum bookkeeping (COP30 → `EPSG:4326`+`EPSG:3855`
  EGM2008 orthometric; NASADEM → EGM96; `_E` ellipsoidal variants via local geoid conversion).
  Lightweight (requests + GDAL); needs an `OT_API_KEY` (demo key exists).
- **`~/src/lidar_tools`** (pip/conda `lidar_tools`, CLI `lidar-tools rasterize`) — fetches USGS
  3DEP lidar via EPT and rasterizes to DSM/DTM COGs for an AOI; also has `get_copernicus_dem`,
  `get_esa_worldcover`, `confirm_3dep_vertical` (COP30-vs-DEM bare-ground datum check),
  `tap_bounds`, UTM-G2139 WKT helpers. Heavy (PDAL/conda env, AWS-bound) and under active cleanup.
- **`~/src/coincident`** (pip/conda `coincident`) — Planetary-Computer STAC fetch of COP30/NASADEM
  with `load_dem_7912()` (reproject-on-read to ellipsoid via the `globaldems/` VRT) and
  `sample_dem_at_points()`. Keyless but pulls the heavier STAC stack; a v2 alternative backend.

The newly created (empty) repo **`uw-cryo/groundcontrol`** (default branch `main`) will become
a standalone, reusable library + thin wrapper scripts whose objective is:

> Given an arbitrary AOI, fetch available control points (with correct datum/epoch handling);
> and given a DEM, automatically sample the control, run the accuracy assessment, and produce
> the analysis + visualization for vertical/horizontal accuracy.

This is an **accuracy-assessment + control-fetch** library, **not** an ASP GCP-injection tool.

> **Self-contained note (for remote refinement / Ultraplan):** the source repos under `~/src`
> referenced below are not accessible to a cloud agent. The verbatim functions, exact API
> endpoints, file schemas, and CRS recipes the migration reuses are embedded in **Appendix A** at
> the end of this plan, so it can be refined and implemented without local `~/src` access.

> **Reconciliation note (2026-06-26):** this file is the merge of the original local plan and the
> Ultraplan adversarial-review pass. Ultraplan preserved Appendix A verbatim and added: the
> Increment-1.5 split, the single-canonical-CRS schema decision, inline `acc_metrics` formulas,
> the pre-differencing vertical-datum reconciliation step, packaging/PyPI fixes, the dispatcher
> contract, the test-fixture prerequisites, and **Appendix B (correctness bugs in the Appendix A
> snippets)**. The minor internal inconsistencies Ultraplan left (NGL labelled both "v1 source" and
> Increment 1.5; `GnssProvider` named in the Deferred section though decision 4/the NGL note defer
> it; the schema-table `geometry` row vs the CRITICAL single-CRS decision) have been **harmonized**
> in this merge: NGL is marked Increment 1.5, the deferred section uses the minimal `Provider`
> interface, and the schema table reflects the single canonical `EPSG:7912` frame.

### Scope tiers (start small; the rest is the target architecture this is designed to grow into)

- **Increment 1 — MVP (BUILD THIS FIRST):** package scaffold + normalized schema + *minimal*
  CRS/epoch handling + three control-point fetchers — **3DEP checkpoints, NGS/OPUS, user-supplied
  points (CSV/GPKG/parquet)** — + a `fetch_control(aoi)` dispatcher + export (GeoParquet/CSV/KML)
  + the `fetch_control` CLI. Deliverable: "given an AOI, return/export available control points."
  No DEM, no accuracy yet.
  **Source choice rationale (adversarial-review change):** the MVP trades the *hardest* source for
  the *easiest*. 3DEP is one `gpd.read_file(bbox=...)` (trivial); NGS is a documented REST API
  (moderate); **`user_points` is pure offline I/O** — no network/key — so it gives the MVP a fully
  offline acceptance test and exercises `schema.normalize()` end-to-end before the schema freezes.
  **NGL moves to Increment 1.5** (see below): it is the only from-scratch time-series client, its
  working URLs are documented as fragile (doubled-frame path; the obvious one 404s), and it is
  already being split (path A now / MIDAS later) — putting it in the MVP would let the hardest
  component gate the deliverable (and the headline Iceland acceptance test rode entirely on it).
- **Increment 1.5 — NGL GNSS (global):** **1.5a** = DataHoldings index → bbox/date filter →
  single-station tenv3 fetch → nearest-day position (path A). **1.5b** = `steps.txt` window
  splitting + step-aware window median + MIDAS linear-velocity fallback (path B). This is what
  proves NGL global coverage; the Iceland non-CONUS test belongs here, not in the MVP gate.
- **Increment 2 — DEM accuracy assessment:** raster sampling + accuracy stats + figures + report
  + `assess_dem` CLI (reusing the upstream primitives). Adds "given a DEM, assess it."
- **Increment 3 — optional COP30 datum diagnosis:** global reference-DEM fetch (OpenTopography
  via `fetch_dem`) + DEM-vs-COP30 vertical-datum check, off by default behind one flag.
- **Deferred (stubs + notes only):** user-CSV loader polish, lidar_tools dense-3DEP reference DEM,
  point-cloud inputs, secondary GNSS networks (GAGE/EPN/IGS), ASP `pc_align`, upstream
  refactor-to-import.

Each increment is independently shippable; later items never block the MVP. The full
module/structure below is the *target* — Increment 1 only needs schema, crs (minimal), sources
(ngs/checkpoints_3dep/user_points + a minimal `base`), io, and the fetch CLI. (`ngl` lands in 1.5.)

### Decisions (confirmed with user)

1. **groundcontrol is the canonical home** for the shared accuracy primitives. The general
   functions in the private accuracy repo's `utils`/`plot` modules move here as the single source of truth.
   The private accuracy repo keeps working unchanged for now and will be refactored to `import groundcontrol`
   incrementally **later** (explicitly out of scope here — do not break the upstream repo).
   **Confirmed by the private accuracy repo's roadmap:** `sample_raster`
   + accuracy stats are tagged *interim* in the private repo's `utils` module and explicitly belong here; the private repo's
   notebooks will `import groundcontrol` once it exists. **Coordination — PENDING (not yet
   implemented as of 2026-06-27):** the private accuracy repo is about to be refactored to **isolate the block-wise
   sampling and a shared block-wise raster read/write tile-iterator** — logic today duplicated
   across the private repo's `sample_raster` / `_sample_parallel` / `_sample_tile_worker` / `classify_points`
   / `mask_raster` / `classify_mask_raster` (both the read-tile→sample path and the
   read→process→write-tile path). groundcontrol hosts the canonical isolated form; that one
   primitive also backs `refdem.py` bare-ground masking and the `rioxarray`/`rasterio`
   replacements for the GDAL-CLI steps (per the packaging fixes). **The Appendix A5 `sample_raster`
   is a pre-refactor snapshot — re-sync `sample.py` from the isolated upstream primitive once the
   refactor lands, and apply the Appendix B sampler fixes (B3/B4/B11) in that single shared
   primitive so both repos benefit** (single source of truth).
2. **groundcontrol stays generic.** The ASP-pipeline GCP/match glue
   (`assign_gcp_sigmas.py`, `normalize_gcps_per_tile.py`, `rotate_matches_gcps.py`,
   `geojson_to_camgen.py`, `plot_gcp_offsets.py`, `eval_gcp_geometry.py`, `dem2gcp`
   orchestration) **stays in `kh9pc_stereo`**. Only their *generic* sub-logic (point sampling,
   NMAD) is re-expressed as clean library functions here.
3. **v1 assesses DEMs (raster sampling).** Point-cloud (PDAL/COPC/laspy) support is a
   documented follow-on with a stub module + optional extra.
4. **Control sources (revised per adversarial review):** the **MVP (Increment 1)** ships three
   fetchers — USGS 3DEP national checkpoints, NGS/OPUS API, and **user-supplied CSV/GPKG/parquet**
   (which also covers contractor survey points like CA Gaps). NGL GNSS moves to **Increment 1.5**
   (it is the hardest source and was gating the MVP). `user_points` is promoted *into* the MVP
   because it is the cheapest source and the only fully-offline-testable one.
5. **`3D_CRS_Transformation_Resources` is referenced, not duplicated.** groundcontrol's CRS
   module is a thin pyproj wrapper that *implements the documented recipes* and reuses that
   repo's small WKT assets + geoid COG URLs, with docstrings linking to the upstream pages
   (cite, don't re-document the theory).
6. **`lidar_tools` integration is deferred (v2).** Do NOT wire it in for v1 — revisit after its
   planned cleanup/overhaul (its v0.2.0 API is in flux). The dense-3DEP-lidar-reference-DEM
   accuracy path (DEM-vs-DEM) is therefore a v2 follow-on; v1 assesses against sparse control
   points only.
7. **GNSS station positions via Nevada Geodetic Lab (NGL) are a v1 source (specifically
   Increment 1.5, per decision 4), and are global.** NGL is the primary global provider (>19k
   stations, daily PPP time series back to ~1996, free public file server, no key).
   Fetch-position-at-an-arbitrary-epoch / time-range is a first-class feature for GNSS sources
   (sample the daily series step-aware, or fall back to the MIDAS linear velocity model). Secondary
   networks (EarthScope/GAGE, EUREF/EPN, IGS) are deferred to v2; v1 ships only the minimal
   `Provider` contract — the shared `GnssProvider` abstraction + cross-source dedupe are extracted
   when the 2nd network lands (per the NGL provider-interface note).
8. **Global reference DEM (COP30) is a v1 optional capability for vertical-datum diagnosis.**
   COP30 is coarse (30 m) but vertically excellent and cheap/global — ideal for a quick
   DEM-vs-DEM check that exposes vertical-datum mistakes (ellipsoid-vs-geoid shows up as the geoid
   undulation; a constant bias shows up as a uniform offset), especially over low-relief sites.
   **Primary backend = OpenTopography via `fetch_dem`** (user-specified; lightest deps; its
   explicit vertical-datum dicts are exactly what datum diagnosis needs). The Planetary-Computer
   backend (coincident/lidar_tools) is a v2 alternative. This COP30 datum-check overlaps with the
   `lidar_tools` datum checks being built in parallel — groundcontrol is the lightweight home;
   lidar_tools can import it later (consistent with decision 1). Careful CRS/datum accounting is
   the whole point, so it routes through `crs.py` (EGM2008/EGM96 ↔ ellipsoid).

## Target repository structure

`src/`-layout, proper namespaced package (cleaner than the private repo's flat `py-modules`, so the private repo
can later do `from groundcontrol.accuracy import resid_stats`):

```
groundcontrol/
├── pyproject.toml                 # setuptools, name="groundcontrol", python>=3.10
├── README.md                      # overview, install, quickstart
├── LICENSE                        # match uw-cryo convention (BSD/MIT)
├── src/groundcontrol/
│   ├── __init__.py
│   ├── schema.py                  # the normalized control-point schema (central abstraction)
│   ├── crs.py                     # AOI-aware datum+epoch transforms (thin pyproj wrapper of
│   │                              #   3D_CRS_Transformation_Resources recipes; reuses its WKTs)
│   ├── sources/
│   │   ├── __init__.py            # fetch_control(aoi, sources, epoch=, frame=) dispatcher
│   │   ├── base.py                # Provider protocol; GnssProvider base (epoch/frame aware)
│   │   ├── ngs.py                 # NGS/OPUS API fetch + parse
│   │   ├── ngl.py                 # Nevada Geodetic Lab GNSS time series (global, Increment 1.5)
│   │   ├── checkpoints_3dep.py    # USGS 3DEP national checkpoint DB (bbox read)
│   │   ├── user_points.py         # generic CSV/GPKG/parquet ingestion
│   │   └── (v2) gage.py, epn.py, igs.py   # secondary GNSS networks, same interface
│   ├── sample.py                  # point-to-raster sampling (DEM)
│   ├── accuracy.py                # robust stats + NVA/VVA/RMSE@95% metrics
│   ├── plot.py                    # residual maps, histograms, offset vectors, report figures
│   ├── report.py                  # assemble stats table + figures into an accuracy report
│   ├── io.py                      # standardized export: GeoParquet / CSV / KML
│   ├── refdem.py                  # global ref-DEM fetch (COP30 via OpenTopography) +
│   │                              #   vertical-datum diagnosis (DEM-vs-COP30 over bare ground)
│   └── pointcloud.py              # STUB for v2 (PDAL); raises NotImplementedError w/ note
├── scripts/
│   ├── fetch_control.py           # CLI: AOI -> control GeoParquet/CSV/KML
│   └── assess_dem.py              # CLI: DEM(+AOI) -> fetch -> sample -> stats + figures
├── notebooks/
│   ├── example_fetch_control.ipynb        # generalized from casagrande NGS notebook
│   └── example_dem_accuracy_3dep.ipynb    # generalized from the private repo's checkpoint section
├── docs/
│   ├── crs_datum_epochs.md        # adapted from the private repo's crs_datum_epochs.md
│   ├── control_sources.md         # each source: URL, schema, gotchas, rate limits, citations
│   │                              #   (incl. NGL frame-first URLs + MIDAS; GAGE/EPN/IGS for v2)
│   └── accuracy_methods.md        # NVA/VVA/RMSE@95%/NMAD definitions + 3DEP report refs
└── tests/
    ├── test_crs.py, test_schema.py, test_accuracy.py, test_sample.py
    └── test_sources_*.py          # network tests marked/skippable
```

## Central abstraction: normalized control schema (`schema.py`)

Every source returns a GeoDataFrame with the **same columns** so downstream sampling/accuracy
code is source-agnostic:

| column | meaning |
|---|---|
| `source` | `ngs` / `opus` / `ngl` / `3dep_checkpoint` / `user` |
| `id` | station PID / 4-char GNSS ID / checkpoint id / user id |
| `geometry` | point in the **single canonical frame `EPSG:7912`** (see the CRITICAL schema decision below); never per-row mixed-CRS |
| `height` | scalar height value (canonical = ellipsoidal in `EPSG:7912`; original kept in `raw`) |
| `height_datum` | provenance: original datum, e.g. `NAVD88` (orthometric) / ellipsoidal |
| `horizontal_crs`, `vertical_crs` | **provenance of the original values only** (e.g. `EPSG:6318+5703`); geometry itself is always `EPSG:7912` |
| `ref_frame` | realization for GNSS sources: `IGS20` / `IGS14` / plate-fixed (e.g. `NA`) |
| `epoch` | decimal year of the coordinate (via `decyear`) — carried with every point |
| `point_type` | `gnss` / `monument` / `NVA` / `VVA` / `control` |
| `acc_h`, `acc_v` | reported accuracy; for GNSS, per-axis `sig_e/sig_n/sig_u` in `raw` |
| `observed` | observation date / span |
| `raw` | dict of source-specific extra fields |

`schema.py` provides `normalize(df, source) -> GeoDataFrame`, a validator, and an
`empty() -> GeoDataFrame` constructor (full column set + dtypes, zero rows — used by the
dispatcher for failed/empty sources so concat never breaks).

**CRITICAL schema decision (adversarial review — resolve before the schema freezes):** a single
GeoDataFrame has exactly **one** `.crs`. Storing per-row `horizontal_crs`/`vertical_crs` strings
*alongside* a mixed-CRS `geometry` column is internally inconsistent — `to_crs`, `total_bounds`,
spatial ops, and the GeoParquet writer all assume one CRS and would silently corrupt rows not in
it. Therefore `normalize()` must **actually normalize, not annotate**:
- Every source is transformed at fetch time into **one canonical frame — ITRF2014 / `EPSG:7912`
  ellipsoidal** (the plan's own pivot frame) — and `geometry` is always in that single CRS.
- `horizontal_crs` / `vertical_crs` / `height_datum` / `epoch` / `ref_frame` are retained as
  **provenance of the original values only**, never as live per-row coordinate state.
- The original native coordinates, if needed, live in plain numeric `raw` fields — not in
  `geometry`. Downstream sampling/differencing therefore never has to reconcile mixed CRS.
This makes "fetch from N sources, assess against 1 DEM" genuinely uniform instead of deferring
(and mislabeling) the unification.

## Module-by-module migration map (source → destination)

**Fetching / CRS / epoch — from `casagrande/notebooks/noaa_ngs_gps_api_CasaGrande.ipynb`:**
- `ngs_query_bbox` (cell 10, recursive bbox subdivision past the 500-item limit) → `sources/ngs.py`
- `fetch_full_details_by_pid` (cell 111), `get_stations_by_epoch` (cell 142) → `sources/ngs.py`
- `parse_ngs_df` (cell 12, NGS+OPUS column typing, builds `EPSG:6318` geom) → `sources/ngs.py`
- `get_transformer` / `apply_transform` (cell 11, `pyproj.TransformerGroup` with
  `AreaOfInterest` + `allow_ballpark=False`) → `crs.py`
- `decyear` (cell 13) → `crs.py` (epoch helper; feeds time-dependent datum transforms)
- `crs.py` follows the documented recipes in `3D_CRS_Transformation_Resources` rather than
  inventing transforms: NAD83(2011) NAVD88 (EPSG:6318+5703) ↔ ITRF2014 (EPSG:7912) via
  Helmert `t_epoch=2010` + GEOID18 `vgridshift`; geoid grids from cdn.proj.org
  (`us_noaa_g2018u0.tif` etc.); custom 3D-UTM `.wkt` (WGS84 G2139) reused from that repo's
  `3dep/*.wkt` where no EPSG code exists. Prefer EPSG codes; copy a WKT asset only when
  necessary, with an attribution + upstream-link comment.
- `export_kml` (cell 78) → `io.py`
- Drop the Casa-Grande-hardcoded glue (Abrams/CNET PID lists, ground-truth web scraping).

**NGL GNSS time series — new code in `sources/ngl.py` (no existing source to port):**
NGL is a plain Apache file server; v1 implements the station-index → per-station-fetch pattern.
- **Cache-once global assets:** `NGLStationPages/DataHoldings.txt` (cols
  `Sta Lat Long Hgt X Y Z Dtbeg Dtend Dtmod NumSol StaOrigName`, lon 0–360),
  `NGLStationPages/steps.txt` (type 1 = equipment, type 2 = earthquake offsets), and the MIDAS
  velocity files `gps_timeseries/<FRAME>/midas/midas.IGS.txt` (+ per-plate). Cache by date/ETag.
- **Spatial + temporal filter:** keep stations inside the AOI bbox whose `[Dtbeg,Dtend]` overlaps
  the requested epoch/time-range.
- **Per-station fetch (throttle ~≤4 concurrent) — VERIFIED LIVE 2026-06-26 (doubled-frame path;
  the flat `gps_timeseries/tenv3/IGS14/…` on the portal page is STALE/404):**
  - tenv3: `https://geodesy.unr.edu/gps_timeseries/<FRAME>/tenv3/<FRAME>/<SSSS>.tenv3`
    (header row present; columns include `_latitude(deg) _longitude(deg) __height(m)` **directly**
    — a position at a given date is just a row lookup, no e0/east-offset reconstruction needed;
    note tenv3 longitude is wrapped, normalize to −180..180 — see Appendix B1, NOT `mod 360`).
  - txyz2 (geocentric XYZ): `https://geodesy.unr.edu/gps_timeseries/<FRAME>/txyz/<FRAME>/<SSSS>.txyz2`.
  - `<FRAME>` ∈ `IGS14`, `IGS20` (≈ITRF2020); plate-fixed under `.../<FRAME>/tenv3/plates/`.
  - Live test: `DataHoldings.txt` = 23,605 stations; `…/IGS14/tenv3/IGS14/00NA.tenv3` = 200, ~650 KB.
- **Position at an arbitrary epoch / time-range** (the requested feature; lives in `ngl.py`):
  (A) sample the daily series — nearest day or step-aware window median (use the `_latitude/
  _longitude/__height` columns) — when the epoch is inside the station span; (B) MIDAS linear
  model `pos(t)=intercept+velocity·(t−first_epoch)` fallback for epochs outside the span. Always
  emit `epoch` + `ref_frame`. (MVP can ship path A only; MIDAS fallback can follow.)
- **Naming note:** "NGS GNSS records" in the request = these NGL (Nevada Geodetic Lab) time
  series. NGS's own monument/CORS *published positions* come via the NGS/OPUS API path above.
- **No new deps:** `requests` + `pandas` only (stays pip-only). Free/public; cite Blewitt,
  Hammond & Kreemer (2018) for NGL and Blewitt et al. (2016) for MIDAS in docs/README.
- **Provider interface (revised per adversarial review):** `sources/base.py` defines only a
  minimal `Provider` callable contract (`fetch(aoi, **kw) -> schema-valid GeoDataFrame`) in v1.
  **Do NOT build a `GnssProvider` base or cross-source dedupe-by-4-char-ID yet** — with one GNSS
  source it is dead code and interface-by-speculation; extract it when the *second* network
  (GAGE/EPN/IGS, all v2) actually lands and the real shared surface is visible.

**3DEP checkpoints — from the private per-site notebooks (the Las Vegas notebook cells ~305–349,
the San Francisco notebook cells ~289–372):**
- Generalize the inline `gpd.read_file(url, bbox=...)` + `point_type` filtering + reprojection
  into `sources/checkpoints_3dep.py` with parameterized column names and a default source URL
  (national `Checkpoints_3DEP_2004_2025` gpkg/parquet; document ScienceBase + the parquet mirror).

**Global reference DEM (COP30) — `refdem.py` (v1 optional), reusing `fetch_dem`:**
- Fetch COP30 (and optionally COP90/NASADEM/SRTM/ALOS) for the AOI via OpenTopography. **Reuse
  `fetch_dem`** (`opentopo_utils.get_dem()` + its horizontal/vertical-datum dicts) rather than
  re-implementing — depend on it (pip install from GitHub) or vendor the ~70-line `opentopo_utils`
  with attribution if packaging is awkward. `OT_API_KEY` via env (documented; demo key fallback).
- `diagnose_vertical_datum(dem, cop30)`: transform both to a common frame via `crs.py`
  (EGM2008/`EPSG:3855` ↔ ellipsoid; `3D_CRS_Transformation_Resources/globaldems/COP30_hh_7912.vrt`
  is the canonical reproject path), difference over bare/stable ground (worldcover bare class),
  and report median/NMAD dh + a likely-cause hint (≈geoid-undulation signal ⇒ ellipsoid/geoid
  mix-up; uniform offset ⇒ constant datum bias). Mirrors `lidar_tools.confirm_3dep_vertical`;
  groundcontrol owns the lightweight version, lidar_tools imports it later.
- Optional Planetary-Computer backend (`coincident.load_dem_7912` / `lidar_tools.get_copernicus_dem`)
  behind a `[stac]` extra is a v2 alternative — not needed for v1.

**Accuracy primitives — from the private accuracy repo's `utils` module (these become canonical here):**
- `sample_raster` (block-wise, memory-safe, CRS-aware) → `sample.py` — adopt the **isolated**
  block-wise sampler + shared read/write tile-iterator from the private repo's pending refactor (incl. the
  optional parallel `_sample_parallel`/`_sample_tile_worker` variant); see Decision 1 ⏳
- `med_nmad`, `robust_normalize`, `resid_stats` (n/median/mean/NMAD/std/RMSE) → `accuracy.py`
- `parse_pc_align_log`, `pc_align_errors_gdf`, `pc_align_error_diff` → `accuracy.py`
  (optional co-registration helpers; co-registration itself is a stretch, kept thin —
  the **core** accuracy path is DEM−control differencing, no ASP required)
- Add `acc_metrics()` on the differenced residuals. **Define the formulas in the plan, not a
  deferred doc (adversarial review — they are distinct statistics and two implementers would
  diverge):** `NVA = 1.9600 * RMSE_z` over non-vegetated/open-terrain checkpoints (normal-error
  assumption); `VVA = percentile(|dz|, 95)` over vegetated checkpoints (non-parametric);
  `RMSE_z = sqrt(mean(dz**2))`. State the outlier filter applied first (`robust_normalize`,
  `nmad_mult=3`) and the minimum-n below which metrics are NaN. Cite ASPRS 2014 / the 3DEP report
  edition by number in `docs/accuracy_methods.md`.

**Vertical-datum reconciliation before differencing (adversarial review — the core path currently
has the exact bug the COP30 diagnosis is built to catch):** `sample_raster`'s `diff` subtracts raw
control `height` from the raw DEM value with **no geoid/ellipsoid reconciliation**. Make the
pre-sample transform an explicit, tested step in `assess_dem.py`: (1) the DEM's CRS/epoch *and
vertical datum* are required inputs (a GeoTIFF often lacks a vertical CRS — it must be supplied);
(2) control points are transformed (horizontal **and vertical**, via `crs.py`) into the DEM's
frame before sampling; (3) `sample_raster` asserts points and raster share a CRS and raises rather
than silently mis-sampling; (4) the `diff` column is only computed once both sides are in one
vertical frame.

**Figures — from the private accuracy repo's `plot` module:**
- `plot_residual_before_after`, `plot_residual_page`, `plot_pc_align_errors`,
  `plot_offset_vectors` → `plot.py` (drop the P3D/3DEP triangle-closure-specific figures;
  keep the generic residual map / histogram / offset-vector primitives).

**Generic sub-logic re-expressed (NOT moved) from kh9 scripts:**
- The `sample()` + `nmad()` helpers inside `casagrande/eval_gcp_geometry.py` are duplicated by
  the upstream versions above — implement once in `sample.py`/`accuracy.py`; the ASP pointmap
  script itself stays in kh9pc_stereo.

**Docs/reference to carry over:**
- the private repo's `crs_datum_epochs.md` → `docs/crs_datum_epochs.md` (datum/epoch handling reference);
  keep it short and **link out** to `3D_CRS_Transformation_Resources` pages for the full theory.
- the private repo's 3DEP lidar-report notes accuracy numbers → `docs/accuracy_methods.md` (regression refs)
- Reference (do not vendor) `~/src/3D_CRS_Transformation_Resources` (uw-cryo) for the
  canonical datum-transform WKTs/recipes used in CRS/epoch handling.

## CLI wrapper scripts

- `scripts/fetch_control.py --aoi aoi.geojson --sources ngs,ngl,3dep --out control.parquet [--kml]`
  `[--epoch 2015.5 | --time-range 2014-01-01:2016-01-01] [--frame IGS20]`
  → calls `groundcontrol.sources.fetch_control(aoi, sources, epoch=, frame=)`, normalizes, writes
  GeoParquet/CSV/KML. `--epoch`/`--frame` apply to GNSS sources (NGL); ignored by fixed-coord sources.
- `scripts/assess_dem.py --dem dem.tif [--aoi aoi.geojson] [--sources ...] --outdir out/`
  `[--reference-dem cop30]`
  → derives AOI from DEM bounds if not given, fetches control, transforms to the DEM CRS/epoch,
  samples with `sample.py`, computes `accuracy.acc_metrics`, writes a stats table +
  `plot.py` figures + an HTML/markdown `report.py` summary. With `--reference-dem cop30` it also
  fetches COP30 (`refdem.py`) and runs the DEM-vs-COP30 vertical-datum diagnosis (off by default).
- Expose both as `console_scripts` entry points (`groundcontrol-fetch`, `groundcontrol-assess`).

## Packaging & dependencies

`pyproject.toml` (setuptools, src-layout). Core deps (all pip-installable, no PDAL/conda
requirement in v1): `numpy`, `pandas`, `geopandas`, `pyproj`, `shapely`, `rasterio`,
`rioxarray`, `requests`, `matplotlib`, `pyarrow` (GeoParquet). Python `>=3.10`.

**Packaging fixes (adversarial review):**
- **`fetch_dem` must NOT be a hard dependency** — PyPI rejects direct-URL (`git+https://…`) deps,
  so depending on the GitHub-only `fetch_dem` would make `groundcontrol` itself un-publishable.
  **Vendor the ~120-line `opentopo_utils` (with attribution) as the default** COP30 path (the plan
  already offered this as a fallback — promote it to the plan). Keep a live `fetch_dem` import only
  as a dev convenience. Also **replace the `_E`/`gdalwarp`/`gdal_edit.py` CLI subprocesses with
  `rioxarray`/`rasterio`** (already core) so core requires no GDAL-CLI runtime.
- **KML moves behind a `[kml]` extra** (the only thing pulling `fiona`, whose GDAL binding can
  conflict with `rasterio`'s). KML export reprojects to `EPSG:4326` and enables the driver
  explicitly (`fiona.supported_drivers['KML']='rw'`), with a `simplekml` fallback.
- **`crs.py` fetches PROJ geoid grids from cdn.proj.org at runtime** — document this network
  requirement and provide an offline-grid fallback / clear error, since `crs.py` round-trip is an
  Increment-1 unit test that will fail in a sandboxed CI without grid access.

The heavy STAC stack (`pystac-client`/`planetary-computer`/`odc-stac`) is intentionally avoided in
v1 and lives behind a `[stac]` extra. Other optional extras: `[kml]` (fiona/simplekml), `[lidar]`
(lidar_tools, after its overhaul), `[pdal]` (point cloud), `[asp]` (pc_align coreg). Keeping the
v1 core pip-only and PyPI-installable is deliberate.

**Dispatcher contract (adversarial review — was asserted but never specified):**
`fetch_control(aoi, sources, …)` wraps each provider in try/except; each provider returns a
schema-valid frame (possibly zero rows, but always the full normalized columns via
`schema.empty()`). The dispatcher concats only non-empty frames and returns
`(combined_gdf, status: dict[source -> {n_rows, error}])`. On total failure it returns an empty
schema frame and the CLI exits non-zero. A unit test mocks one source raising and asserts the
others still return. This makes "degrade gracefully per-source" (e.g. NGS/3DEP empty over Iceland)
a defined behavior, not a hope.

## Deferred / future integrations (explicitly NOT in v1)

- **`lidar_tools` dense 3DEP reference DEM (v2).** Add `--reference-dem 3dep` that calls
  `lidar-tools rasterize` for the AOI to build a dense 3DEP DTM, then does DEM-vs-DEM
  differencing in addition to sparse control. (COP30 already gives the *cheap global* reference in
  v1; this is the *dense, high-accuracy* US-only counterpart.) Reuse `get_esa_worldcover` for the
  bare-ground mask shared with the COP30 diagnosis. **Revisit after the lidar_tools cleanup** (its
  API is in flux); decide CLI-subprocess vs library-import then. Leave a clear note/issue.
- **Planetary-Computer COP30 backend + extra global DEMs (v2).** `[stac]` extra wrapping
  `coincident.load_dem_7912` / `lidar_tools.get_copernicus_dem` as a keyless alternative to the
  OpenTopography path, plus richer multi-DEM comparison (NASADEM/SRTM/ALOS).
- **Shared datum-check module.** If the COP30 vertical-datum logic converges with the parallel
  `lidar_tools` datum work, factor it into the lightweight `refdem.py` here and have lidar_tools
  import it (avoid two copies of `confirm_*_vertical`).
- **Point-cloud accuracy (v2).** `pointcloud.py` stub + `[pdal]` extra; sample a user point
  cloud against control or difference against the 3DEP reference.
- **ASP co-registration (optional).** Thin wrappers around `pc_align` reusing the
  `parse_pc_align_log` helpers, behind `[asp]`. Core accuracy path stays differencing-only.
- **Secondary GNSS networks (v2).** `gage.py` (EarthScope/GAGE REST web service, Americas),
  `epn.py` (EUREF EPN SINEX/SSC, Europe, ETRFxx frames), `igs.py` (IGS repro3/ITRF2020 SINEX,
  sparse global datum anchor) — each implementing the minimal `Provider` interface; the
  `GnssProvider` abstraction + cross-source dedupe-by-4-char-ID are extracted once this 2nd network
  lands. NGL covers the global case in v1; these add regional cross-checks / extra frames
  (e.g. ETRS89) when needed.
- **upstream refactor-to-import.** Once groundcontrol is published, refactor the private repos to
  `import groundcontrol` for the shared utils (separate task; do not touch the upstream repos here).

## Migration mechanics

- **Fresh copy + refactor**, not git-history preservation. Code spans two repos and is being
  restructured into a package, so `git filter-repo` buys little. Add origin attribution in the
  README and module docstrings ("extracted from casagrande + the private accuracy repo, <date>").
- **Do not delete or modify** the originals in `casagrande` + the private accuracy repo as part of this migration;
  leave them as historical reference. (the upstream repos' later refactor-to-import is a separate task.)
- Work on a feature branch in the new repo and open a PR (no direct pushes to `main`); **ask
  before any commit/push** per project convention.

## Phased execution

**Increment 1 — MVP: fetch control points for an AOI (build + ship this first)**
1. **Scaffold:** repo skeleton, `pyproject.toml` (core deps only), `__init__`, README, LICENSE.
2. **schema + minimal crs:** normalized schema (`schema.py`) + `crs.py` with `get_transformer`
   (AOI-aware pyproj) and `decyear`; enough to land every source in a common geographic CRS and
   carry epoch/frame. Unit tests.
3. **three source fetchers + dispatcher:** `sources/base.py` (minimal `Provider` callable
   contract only — **defer `GnssProvider`** to when the 2nd GNSS network exists, per adversarial
   review; building it against one implementer is interface-by-speculation), `sources/checkpoints_3dep.py`
   (national bbox read — easiest), `sources/ngs.py` (NGS/OPUS, port from the casagrande notebook),
   `sources/user_points.py` (offline CSV/GPKG/parquet → `schema.normalize`), and the
   `fetch_control(aoi, sources, …)` dispatcher with the failure/empty contract above. **Split each
   network source into `fetch()` (network) + `parse()` (pure)** so the parse half is always
   CI-covered offline.
4. **export + CLI:** `io.py` (GeoParquet/CSV; KML behind `[kml]`) + `scripts/fetch_control.py`.
   → Milestone: `fetch_control --aoi casa_grande.geojson --sources 3dep,ngs,user --out control.parquet`.
   (NGL — `sources/ngl.py` — lands in Increment 1.5: DataHoldings index → bbox/date filter →
   per-station tenv3 → position-at-epoch path A, then steps/MIDAS path B.)

**Increment 2 — DEM accuracy assessment**
5. **sample + accuracy:** `sample.py` (`sample_raster` from the private accuracy repo — adopt the isolated block-wise
   sampler/write primitive once that refactor lands; until then port the A5 snapshot and re-sync,
   see Decision 1 ⏳), `accuracy.py` (robust stats + `acc_metrics`).
6. **plot + report + assess_dem CLI:** `plot.py` figures, `report.py`, end-to-end `assess_dem.py`.

**Increment 3 — optional COP30 datum diagnosis**
7. **refdem:** `refdem.py` COP30 fetch via `fetch_dem` + `diagnose_vertical_datum`; wire
   `--reference-dem cop30` into `assess_dem.py`.

**Cross-cutting (as each increment lands)**
8. **notebooks + docs:** generalized example notebook(s); `docs/` pages (control_sources incl.
   verified NGL endpoints, crs_datum_epochs, accuracy_methods).
9. **deferred stubs:** `pointcloud.py` NotImplementedError + `[pdal]`/`[stac]`/`[lidar]`/`[asp]`
   extras; user-CSV loader; secondary-GNSS provider stubs — notes only.

## Verification

**Test fixtures & prerequisites (adversarial review — none of these existed; without them not one
acceptance test was runnable):** the source repos (`casagrande` + the private accuracy repo) and their DEMs are NOT
in this environment, so the deliverables must include committed fixtures:
- `tests/data/casa_grande.geojson` — an inline AOI with literal bbox coordinates (define them in
  the plan/repo; do not reference the inaccessible casagrande repo).
- A small committed **synthetic GeoTIFF** with known pixel values for `test_sample.py` (covering
  north-up **and** south-up transforms, and a NaN-nodata float raster — see Appendix B bugs).
- Hand-computed numeric fixtures for `test_accuracy.py` (NVA/VVA/RMSE@95% with expected values).
- Recorded small response fixtures (a saved NGS JSON, a truncated `DataHoldings.txt`, a clipped
  3DEP parquet) to test each `parse()` offline, separate from `fetch()`.
- The SF/Casa-Grande **DEM end-to-end tests are reclassified as manual / user-supplied** (the
  rasters aren't here); the regression target is inlined: SF NVA mean ≈ **−0.008 m**, @95% ≈
  **0.032 m** (from the private repo's 3DEP lidar-report notes, reproduced here since the source is
  inaccessible).
- **Network endpoints** assumed by the suite (list as preconditions; route through the proxy):
  geodesy.noaa.gov, geodesy.unr.edu, portal.opentopography.org (`OT_API_KEY` env — demo key is
  rate-limited; recommend a real key), raw.githubusercontent.com, sciencebase.gov. All network
  tests are `@pytest.mark.network` + skip-if-no-connectivity / skip-if-no-key.

- **Increment 1 (MVP) acceptance — the primary near-term goal:**
  - **Offline:** `fetch_control --aoi casa_grande.geojson --sources user --in fixture.csv
    --out control.parquet` runs with no network and writes a normalized, schema-valid table — the
    MVP's regression-safe acceptance gate.
  - **Network:** `fetch_control --aoi casa_grande.geojson --sources 3dep,ngs,user --out control.parquet`
    writes a normalized table with rows from each available source, valid single-CRS geometry, and
    correct provenance columns — plus CSV (and KML via `[kml]`) exports.
  - Dispatcher degradation: mock one source raising; assert the others still return and `status`
    records the failure (defined contract, not a hope).
- **Increment 1.5 (NGL) acceptance:** re-run on a **non-CONUS AOI (e.g. Iceland)** to prove NGL
  global coverage (NGS/3DEP empty there — degrade gracefully per-source).
- **Unit tests (offline):**
  - `crs.py`: NAD83(2011)↔WGS84 round-trip within transform accuracy; **plus a *vertical* test the
    original plan lacked (adversarial review): `EPSG:6318+5703 → 7912 → 6318+5703` closure, and a
    fixed-point fixture asserting ellipsoidal height ≈ orthometric + GEOID18 N at a known CONUS
    point** — this catches a reversed `vgridshift` (a ~25 m blunder a horizontal-only test misses).
    `decyear` polymorphic over scalar/Series (the test must match the chosen signature).
  - `schema.py`: heterogeneous source frames normalize to **one canonical CRS** + identical
    columns/dtypes; `schema.empty()` is concat-compatible with populated frames.
  - `accuracy.py` (Increment 2): `med_nmad`/`resid_stats`/RMSE@95% vs hand-computed fixtures.
  - `sample.py` (Increment 2): sampling a synthetic GeoTIFF returns known values; CRS-transform path.
- **Integration (network, marked):**
  - `ngs.py` on the Casa Grande AOI returns NGS stations matching the count/IDs from the original
    `noaa_ngs_gps_api_CasaGrande.ipynb` (regression vs. the source notebook).
  - `checkpoints_3dep.py` on a Las Vegas / SF bbox returns the expected NVA checkpoints.
  - `ngl.py`: parse `DataHoldings.txt`, bbox-filter, fetch a station `tenv3` (doubled-frame URL,
    verified live), confirm position-at-epoch path A (row/window lookup); later, path B (MIDAS)
    agrees within velocity·Δt near the span edge and a `steps.txt` discontinuity is respected.
- **End-to-end:**
  - `assess_dem.py` on an existing internal DEM (e.g. SF) reproduces the documented NVA
    (mean ≈ −0.008 m, @95% ≈ 0.032 m from the private repo's 3DEP lidar-report notes) within tolerance,
    and emits the residual map + histogram + stats table.
  - `assess_dem.py` on a Casa Grande DEM fetches NGS control and produces a coherent report.
  - **COP30 datum diagnosis:** `--reference-dem cop30` on a DEM with a *deliberately wrong*
    vertical datum (feed an ellipsoidal DEM as if orthometric) flags a dh ≈ local geoid
    undulation, not ~0 — proving the diagnosis catches ellipsoid/geoid mix-ups; on a correctly
    referenced DEM the median dh is near zero over bare ground.
- Confirm the private accuracy repo still runs unchanged (no edits made to it in this migration).

---

## Appendix A — Embedded source extracts (plan is self-contained; no `~/src` access required)

Verbatim functions / exact specs the migration reuses, lifted from the local repos a cloud agent
cannot see. Each block notes its origin and destination module. Refactor into the package layout
above (clean up the noted bugs — **see Appendix B; do not copy these as-is**), but preserve behavior.

### A1. NGS/OPUS fetch — `casagrande/notebooks/noaa_ngs_gps_api_CasaGrande.ipynb` → `sources/ngs.py` + `crs.py`

Endpoints (plain `requests.get`):
- NGS bbox: `https://geodesy.noaa.gov/api/nde/bounds` — params `minlat,maxlat,minlon,maxlon`; **500-item cap** (recurse/subdivide).
- OPUS bbox: `https://geodesy.noaa.gov/api/opus/bounds` — same params.
- PID detail: `https://geodesy.noaa.gov/api/nde/pid` — param `pid=comma,list`; batch ≤100.

Datum/units: NGS → NAD83(2011) `EPSG:6318` (2D), `EPSG:6319` (3D ellipsoid), `EPSG:6318+5703`
(3D NAVD88 orthometric). Heights: `orthoHt` (NAVD88, via `geoidHt`/GEOID18), `ellipHeight`.
`epoch` field is a decimal-year string (e.g. `"2010.0"`). Non-2011 datums seen:
NAD83(1992)/(1986) → transform to NAD83(2011).

```python
def ngs_query_bbox(minlat, maxlat, minlon, maxlon, base_url="https://geodesy.noaa.gov/api/nde/bounds"):
    """Fetch all NGS records in a bbox, bypassing the 500-item cap via recursive quad subdivision."""
    seen_pids = set(); all_stations = []
    def fetch_recursive(b_minlat, b_maxlat, b_minlon, b_maxlon):
        params = {"minlat": b_minlat, "maxlat": b_maxlat, "minlon": b_minlon, "maxlon": b_maxlon}
        try:
            response = requests.get(base_url, params=params); response.raise_for_status()
            data = response.json()
            if len(data) == 500 or data == 'error':           # too many → subdivide into 4 quads
                mid_lat = b_minlat + (b_maxlat - b_minlat) / 2
                mid_lon = b_minlon + (b_maxlon - b_minlon) / 2
                fetch_recursive(b_minlat, mid_lat, b_minlon, mid_lon)
                fetch_recursive(b_minlat, mid_lat, mid_lon, b_maxlon)
                fetch_recursive(mid_lat, b_maxlat, b_minlon, mid_lon)
                fetch_recursive(mid_lat, b_maxlat, mid_lon, b_maxlon)
            else:
                for station in data:
                    pid = station.get('pid')
                    if pid and pid not in seen_pids:
                        seen_pids.add(pid); all_stations.append(station)
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
    fetch_recursive(minlat, maxlat, minlon, maxlon)
    if not all_stations: return gpd.GeoDataFrame()
    return pd.DataFrame(all_stations)

def get_transformer(gdf, source_crs, target_crs):
    """AOI-aware, accuracy-checked datum transform selection (no ballpark)."""
    aoi = pyproj.aoi.AreaOfInterest(*gdf.total_bounds)
    tg = pyproj.transformer.TransformerGroup(source_crs, target_crs, area_of_interest=aoi,
                                             allow_ballpark=False, always_xy=True)
    return tg.transformers[0]

def decyear(df, col="date"):
    """datetime Series -> decimal year (e.g. 2010-07-02 -> ~2010.5)."""
    year = df[col].dt.year
    days = (pd.to_datetime(year+1, format='%Y') - pd.to_datetime(year, format='%Y')).dt.days
    return year + (df[col] - pd.to_datetime(year, format='%Y')) / (days * pd.to_timedelta(1, unit="D"))

def fetch_full_details_by_pid(pids: list):
    """Full per-station metadata by PID list (batched ≤100)."""
    base_url = "https://geodesy.noaa.gov/api/nde/pid"; all_station_data = []
    for i in range(0, len(pids), 100):
        params = {"pid": ",".join(pids[i:i+100])}
        try:
            r = requests.get(base_url, params=params); r.raise_for_status()
            all_station_data.extend(r.json())
        except requests.exceptions.RequestException as e:
            print(f"batch {i} failed: {e}"); continue
    return pd.DataFrame(all_station_data).replace(' ', pd.NA)
```

`parse_ngs_df(df)` (also in the notebook) does: parse `lastRecovered`/`observed` to datetime;
coerce the numeric NGS cols (`lat lon ellipHeight epoch orthoHt geoidHt utm* spc* …`) and OPUS
cols (`ts ellHt orthoHtP2p …`); build geometry from `lon/lat`; then per-datum transform any
non-NAD83(2011) rows to NAD83(2011) via `get_transformer`. **Bug to fix on migration:** the
notebook builds the transformer from the *whole* gdf and applies it to *all* geometry inside the
per-datum loop — re-implement to transform only the matching-datum subset (see Appendix B7).
`get_stations_by_epoch` and `export_kml` (KML via `gpd...to_file(driver='KML')`, z=`orthoHt`, crs
`EPSG:6318+5703`) round it out; reuse as-is.

### A2. NGL GNSS — `sources/ngl.py` (endpoints verified live 2026-06-26)

- Station index (one GET, cache): `https://geodesy.unr.edu/NGLStationPages/DataHoldings.txt`
  header `Sta  Lat(deg)  Long(deg)  Hgt(m)  X(m)  Y(m)  Z(m)  Dtbeg  Dtend  Dtmod  NumSol  StaOrigName`;
  23,605 rows; `Long` is 0–360; `Dtbeg/Dtend` = `YYYY-MM-DD` (use for temporal overlap filter).
- Steps/offsets: `https://geodesy.unr.edu/NGLStationPages/steps.txt` (col3 type: `1`=equipment, `2`=earthquake).
- tenv3: `https://geodesy.unr.edu/gps_timeseries/<FRAME>/tenv3/<FRAME>/<SSSS>.tenv3` — **doubled frame**.
  header: `site YYMMMDD yyyy.yyyy __MJD week d reflon _e0(m) __east(m) ____n0(m) _north(m) u0(m)
  ____up(m) _ant(m) sig_e(m) sig_n(m) sig_u(m) __corr_en __corr_eu __corr_nu _latitude(deg)
  _longitude(deg) __height(m)`. → a position at a date is the row's `_latitude/_longitude/__height`
  (longitude wrapped → −180..180 via `((lon+180)%360)-180`, **NOT `mod 360`** — see Appendix B1);
  `sig_e/n/u` are per-axis sigmas; `yyyy.yyyy` is decimal year.
- txyz2 (geocentric): `https://geodesy.unr.edu/gps_timeseries/<FRAME>/txyz/<FRAME>/<SSSS>.txyz2` (X Y Z + sigmas).
- `<FRAME>` ∈ `IGS14`, `IGS20`; plate-fixed under `.../<FRAME>/tenv3/plates/`; MIDAS under `.../<FRAME>/midas/`.
- Live check: `…/IGS14/tenv3/IGS14/00NA.tenv3` → HTTP 200, ~650 KB.

### A3. 3DEP checkpoints — the private per-site notebooks → `sources/checkpoints_3dep.py`

- National DB: USGS ScienceBase item `67075e6bd34e969edc59c3e7`; parquet mirror
  `https://raw.githubusercontent.com/scottyhq/files/refs/heads/main/checkpoints_3dep_2004_2025.parquet`
  (also a `.gpkg`). **Pin to a commit SHA, not `refs/heads/main` (adversarial review — this is a
  personal repo on a mutable branch and is effectively the only working source); document the
  ScienceBase direct-download URL as authoritative fallback and add a config hook to override the
  URL + a cached download with checksum.** Read pattern — **branch on extension (adversarial
  review):** `gpd.read_file(url, bbox=...)` works for `.gpkg`; for **GeoParquet use
  `gpd.read_parquet(url, bbox=...)`** (bbox pushdown needs GeoPandas ≥1.0 + pyarrow — pin/guard the
  version; else read-then-clip). Remote reads may pull the whole file — cache locally once.
```python
# .gpkg path:    chk = gpd.read_file(url, bbox=tuple(aoi_gdf.total_bounds))   # EPSG:4326
# .parquet path: chk = gpd.read_parquet(url, bbox=tuple(aoi_gdf.total_bounds))  # GeoPandas>=1.0
nva = chk[chk['point_type'] == 'NVA']        # point_type ∈ {NVA, VVA, Control}
# height column is chk_z (sometimes h/z); reproject to working CRS via .to_crs(...)
```

### A4. CRS / epoch recipes — `3D_CRS_Transformation_Resources` → `crs.py`

- NAD83(2011) NAVD88 (`EPSG:6318+5703`) → ITRF2014 (`EPSG:7912`): Helmert with `t_epoch=2010` +
  GEOID18 `vgridshift`. Prefer `pyproj.Transformer.from_crs(CRS(6318)+CRS(5703), 7912, area_of_interest=…)`.
- Geoid COGs (cdn.proj.org): GEOID18 `us_noaa_g2018u0.tif`; GEOID12B `us_noaa_g2012bu0.tif`.
- Global-DEM vertical datums: COP30 = EGM2008 (`EPSG:3855`); NASADEM/SRTM = EGM96 (`EPSG:5773`).
- COP30 → ellipsoid reproject-on-read VRT:
  `https://raw.githubusercontent.com/uw-cryo/3D_CRS_Transformation_Resources/refs/heads/main/globaldems/COP30_hh_7912.vrt`
- Custom 3D-UTM WKTs (WGS84 G2139): `3dep/UTM_{4N,10N,11N}_WGS84_G2139_3D.wkt` (copy with attribution only if no EPSG fits).

### A5. Accuracy primitives — the private accuracy repo's `utils` module → `accuracy.py` / `sample.py`

```python
def med_nmad(df, s=1.4826):
    """Median and normalized MAD (robust spread)."""
    med = df.median(); nmad = (df - med).abs().median() * s
    return med, nmad

def robust_normalize(gdf, col, nmad_mult=3.0):
    """Bool index of rows within ± nmad_mult*NMAD of the median for `col`."""
    dh_med, dh_nmad = med_nmad(gdf[col])
    return (gdf[col] > dh_med - nmad_mult*dh_nmad) & (gdf[col] < dh_med + nmad_mult*dh_nmad)

def resid_stats(s):
    """Robust+standard residual stats: n, median, mean, nmad, std, rmse."""
    a = np.asarray(s, float); a = a[np.isfinite(a)]
    if a.size == 0:
        return dict(n=0, median=np.nan, mean=np.nan, nmad=np.nan, std=np.nan, rmse=np.nan)
    med = np.median(a)
    return dict(n=int(a.size), median=float(med), mean=float(a.mean()),
                nmad=float(1.4826*np.median(np.abs(a-med))),
                std=float(a.std()), rmse=float(np.sqrt((a**2).mean())))
```

`sample_raster(gdf, r, col=, method='linear', diff=False, block=4096, workers=1)` — samples a
rioxarray DataArray `r` (with a source file) at the points in `gdf` (in r's CRS), **block-wise**
via windowed `rasterio` reads (tile of `block` px + 2-px halo), bilinear (`method='linear'`) or
nearest; adds column `r.name` (and `'<r.name> minus <col>'` if `diff`). Memory-safe for huge
rasters. The single-worker windowed path is the core (verbatim below); the optional `workers>1`
ThreadPool variant (`_sample_parallel`/`_sample_tile_worker`) is a speedup that can be added later.
**⏳ Pre-refactor snapshot:** the private accuracy repo is isolating this block-wise sampling + a shared windowed
read/write tile-iterator (also used by `mask_raster`/`classify_mask_raster`) into one reusable
primitive that becomes canonical here. Replace this snippet with the isolated version — and apply
the B3/B4/B11 fixes there once — when the refactor lands (see Decision 1).

```python
def sample_raster(gdf, r, col='h_mean', method='linear', diff=False, block=4096, workers=1):
    import rasterio; from rasterio.windows import Window
    src_fn = r.encoding.get('source') if hasattr(r, 'encoding') else None
    if src_fn:
        xs_pt = gdf.geometry.x.values; ys_pt = gdf.geometry.y.values
        vals = np.full(len(gdf), np.nan, dtype='float64')
        with rasterio.open(src_fn) as ds:
            T = ds.transform; nodata = ds.nodata; H, W = ds.height, ds.width
            cols = np.floor((xs_pt - T.c) / T.a).astype(int)
            rows = np.floor((ys_pt - T.f) / T.e).astype(int)
            inb = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
            if inb.any():
                rmin, rmax = rows[inb].min(), rows[inb].max()
                cmin, cmax = cols[inb].min(), cols[inb].max(); halo = 2
                for r0 in range(rmin, rmax + 1, block):
                    for c0 in range(cmin, cmax + 1, block):
                        sel = inb & (rows>=r0)&(rows<r0+block)&(cols>=c0)&(cols<c0+block)
                        if not sel.any(): continue
                        sr, sc = rows[sel], cols[sel]
                        rr0, rr1 = max(0, sr.min()-halo), min(H, sr.max()+halo+1)
                        cc0, cc1 = max(0, sc.min()-halo), min(W, sc.max()+halo+1)
                        win = Window(cc0, rr0, cc1-cc0, rr1-rr0)
                        arr = ds.read(1, window=win, boundless=True,
                                      fill_value=nodata if nodata is not None else np.nan).astype('float32')
                        if nodata is not None: arr[arr == nodata] = np.nan
                        wt = ds.window_transform(win); ny, nx = arr.shape
                        xc = wt.c + (np.arange(nx)+0.5)*wt.a
                        yc = wt.f + (np.arange(ny)+0.5)*wt.e
                        da = xr.DataArray(arr, coords={'y': yc, 'x': xc}, dims=('y','x'))
                        idx = np.where(sel)[0]
                        xq = xr.DataArray(xs_pt[idx], dims='z'); yq = xr.DataArray(ys_pt[idx], dims='z')
                        s = (da.sel(x=xq, y=yq, method='nearest') if method=='nearest'
                             else da.interp(x=xq, y=yq, method=method))
                        vals[idx] = s.values
        gdf[r.name] = vals
    else:                                  # in-memory fallback (computed array, no source file)
        x = xr.DataArray(gdf.geometry.x.values, dims="z"); y = xr.DataArray(gdf.geometry.y.values, dims="z")
        gdf[r.name] = (r.sel(x=x, y=y, method="nearest") if method=='nearest'
                       else r.interp(x=x, y=y, method=method)).values
    if diff:
        gdf[f'{r.name} minus {col}'] = gdf[r.name] - gdf[col]
```

### A6. COP30 fetch — `fetch_dem/opentopo_utils.py` → `refdem.py`

Endpoint: `https://portal.opentopography.org/API/globaldem?demtype={}&west={}&south={}&east={}&north={}&outputFormat=GTiff&API_Key={}`
— `demtype` ∈ `COP30 COP90 NASADEM SRTMGL1 SRTMGL3 AW3D30 SRTM15Plus EU_DTM` (+ `_E` ellipsoidal);
`bounds=(minx,miny,maxx,maxy)` EPSG:4326; key from env `OT_API_KEY` (demo `demoapikeyot2022`).
The crown jewels are the datum dicts (used to stamp the correct compound CRS + drive `_E` geoid→ellipsoid `gdalwarp`):

```python
def get_ot_apikey(variable_name='OT_API_KEY'):
    return os.environ.get(variable_name)

# demtype -> vertical datum (orthometric geoid EPSG, or ellipsoidal for _E)
vertical_geoid_proj_dict = {
    'SRTMGL1_E':'EPSG:4979','AW3D30_E':'EPSG:4979',
    'SRTMGL1':'EPSG:5773','AW3D30':'EPSG:5773','SRTMGL3':'EPSG:5773',
    'SRTM15Plus':'EPSG:5773','NASADEM':'EPSG:5773',
    'COP30':'EPSG:3855','COP90':'EPSG:3855','EU_DTM':'EPSG:3855',   # EGM2008
}
# demtype -> horizontal CRS
horizontal_crs_dict = {
    'SRTMGL1':'EPSG:4326','AW3D30':'EPSG:4326','SRTMGL1_E':'EPSG:4326','AW3D30_E':'EPSG:4326',
    'SRTMGL3':'EPSG:4326','SRTM15Plus':'EPSG:4326','NASADEM':'EPSG:4326',
    'COP30':'EPSG:4326','COP90':'EPSG:4326','EU_DTM':'EPSG:3035',
}
```

`get_dem(demtype, bounds, apikey, out_fn=None, proj='EPSG:4326', local_utm=False, output_res=30)`
GETs the GeoTIFF, sets nodata `-9999`, and (for `_E` or a non-default `proj`) `gdalwarp`s from
`{horizontal}+{vertical}` to the requested output CRS, then `gdal_edit.py -a_srs` stamps the
compound CRS. **Per the packaging fixes above, vendor + re-express with `rioxarray`/`rasterio`
rather than depending on `fetch_dem` or shelling out to GDAL CLIs.** The datum dicts above are the
part that makes the COP30 vertical-datum diagnosis correct. Keyless Planetary-Computer alternative
(`lidar_tools.get_copernicus_dem` / `coincident.load_dem_7912`) is the v2 `[stac]` path.
**Pin the COP30→ellipsoid VRT (`COP30_hh_7912.vrt`) to a tag/commit or commit a copy** rather than
referencing `refs/heads/main` (it references remote COGs — a hidden network+CRS dependency). Also
**verify the `SRTM15Plus` vertical-datum dict entry** (some releases reference EGM2008, not EGM96).

---

## Appendix B — Bugs to fix when porting Appendix A (from the adversarial correctness review)

The Appendix A snippets are verbatim and carry real bugs — **do not copy them as-is.** Fix on port:

**Must-fix (silent wrong results):**
- **B1 — NGL longitude wrap is backwards.** The note "normalize via `mod 360`" is wrong: AOIs and
  the schema use −180..180. `mod 360` turns a CONUS −110° into 250° (wrong hemisphere → bbox filter
  returns zero stations). Use `((lon+180)%360)-180` and apply it to both `DataHoldings.txt` `Long`
  (0–360) and tenv3 `_longitude`. Test a CONUS station (expect negative lon) and an antimeridian case.
- **B2 — GEOID18 direction + per-point epoch (`crs.py`).** Orthometric→ellipsoid **adds** N
  (`h = H + N`); a reversed pipeline gives a ~−25 m blunder. The horizontal-only round-trip in the
  original verification would NOT catch it — add the vertical sign + closure tests (see Verification).
  Also: each point's own `epoch` must drive the time-dependent Helmert, not a single hardcoded 2010
  (that is the NAD83 frame's defining epoch, not every point's observation epoch); otherwise
  horizontal error = velocity·Δt (cm–dm/yr in CONUS).
- **B3 — `sample_raster` never masks NaN-nodata.** `arr==nodata` is always `False` when
  `ds.nodata` is NaN (float rasters) → nodata sentinels returned as real heights. Use
  `if nodata is not None and not np.isnan(nodata): arr[arr==nodata]=np.nan`, then always also mask
  non-finite. Add a NaN-nodata fixture test.
- **B4 — `sample_raster` y-coordinate monotonicity.** North-up rasters give a *descending* `yc`;
  `xarray.interp`/`sel` are a footgun on descending coords. Assert/normalize monotonicity (flip if
  needed); test both north-up and south-up transforms, nearest and linear.

**Must-fix (crashes / data loss):**
- **B5 — NGS recursion (`ngs_query_bbox`).** `data == 'error'` (HTTP 200 body) is treated like
  "too many" → infinite recursion to float underflow on a persistent server error. Treat `'error'`
  distinctly (retry/backoff, don't subdivide). Add a max-depth / min-cell-size guard that accepts a
  possibly-truncated 500 with a warning. Verify NGS bbox inclusivity (half-open bounds → silent
  boundary-point loss); overlap child cells by ε and rely on `seen_pids` dedup.
- **B6 — `get_transformer` returns `tg.transformers[0]` blindly.** With `allow_ballpark=False`,
  `transformers` can be **empty** → `IndexError`. Guard: raise a clear datum error if empty; warn
  on `tg.unavailable_operations` (missing grids → tell the user to enable `PROJ_NETWORK`/download).

**Should-fix (under-specified / correctness):**
- **B7 — the "bug to fix" callout is incomplete.** Transforming only the matching-datum subset is
  necessary but: (a) build the AOI from the *subset's* bounds, not the whole gdf; (b) **heights are
  not transformed** by a 2D transformer — either build 3D points (`EPSG:6319`/compound) so one
  transformer handles x/y/z, or document that NGS-realization height differences are below tolerance
  with a test bounding the error.
- **B8 — `med_nmad` Series-vs-DataFrame ambiguity.** Pin the contract to 1-D Series/array →
  `(float, float)`; have `resid_stats` and `robust_normalize` both call it (currently `resid_stats`
  duplicates the `1.4826*median(|a-med|)` constant — one source of truth, per decision 1).
- **B9 — `decyear` signature.** The verification `decyear("2010-07-02")` passes a string but the
  impl wants a DataFrame+col. Make it scalar/Series-polymorphic (`def decyear(dt)`) and match the
  test. (Also: 2010-07-02 ≈ 2010.496, not exactly 2010.5 — loosen tolerance or use noon DOY 183.)
- **B10 — NGL step-aware window (path A/B).** Pin the window half-width (e.g. ±30 d), clip the
  window to the segment between adjacent `steps.txt` entries containing the epoch (split on type-1
  equipment AND type-2 earthquake), a gap policy (NaN if nearest valid day > N), and bound MIDAS
  extrapolation (only outside span, away from recent steps/post-seismic transients).
- **B11 — `sample_raster` halo vs method.** `halo=2` is the minimum for bilinear and underflows for
  cubic/quintic → NaN/edge artifacts at tile seams. Either set `halo` per method
  (`{nearest:0, linear:2, cubic:3, quintic:4}`) or restrict supported methods to nearest/linear.
  Test tiled-vs-single-window identity at seams (bit-identical for linear).

**Low (hygiene):** `ngs_query_bbox` returns `pd.DataFrame` non-empty but `gpd.GeoDataFrame()` empty
(unify); `.replace(' ', pd.NA)` misses `''`/multi-space (use `r'^\s*$'` regex); verify NGS/OPUS
`lon` sign on a known PID (positive-west would mirror CONUS to Asia).
