# CRS/epoch implementation directives, provenance, and validation

Companion to `plan.md` — the plan defers CRS implementation detail, transform-provenance design,
and the CRS test-fixture spec here. Synthesized from a three-agent verified review (PROJ/GDAL
correctness; provenance/logging; validation fixtures), 2026-07-02. Claims marked **verified** were
reproduced locally against **pyproj 3.7.2 / PROJ 9.7.1 / EPSG v12.029 / GDAL 3.12.1 /
geopandas 1.1.2**; everything else is doc-cited. There are many incorrect CRS resources online —
the authorities used here are proj.org, pyproj docs, gdal.org, PROJ source, and NGS/IERS.

## 1. The two-stage epoch model — the precise scope of PROJ's epoch support

**PROJ is epoch-aware, but only for what EPSG gives it** (re-verified 2026-07-03 on two
independent PROJ installs):

- **Frame transforms: fully epoch-aware.** Cross-frame with epochs
  (`EPSG:7912@2015.0 → EPSG:7911@2005.0`) yields
  `set v_4=2015 → helmert(… rate terms … t_epoch=2010) → set v_4=2005`: the time-dependent
  Helmert *is* evaluated at the source coordinate epoch — then the output epoch is **relabeled**,
  not propagated.
- **Epoch propagation: only with a point-motion model.** The machinery exists — EPSG
  PointMotionOperation support (PROJ ≥ 9.4,
  https://github.com/OSGeo/PROJ/releases/tag/9.4.0; verified live for NAD83(CSRS)v7 via the
  Canada velocity grid) and the generic `+proj=deformation` velocity-grid operator (hand-pipeable
  where a velocity grid exists). What is missing is **data, not machinery**: EPSG v12.029
  registers no ITRF point-motion model, so a same-frame ITRF epoch change
  (`EPSG:9989@2020.0 → @2005.0`) returns `+proj=noop` — "Null geographic offset … **0 m**" — a
  *claimed-exact* no-op for what is really ~15 yr of plate motion. This is the dangerous case:
  it looks like success.

Therefore `crs.py` implements two explicit stages **today**, while the schema deliberately
records enough metadata (`frame_epoch`/`coord_epoch`/`measurement_*`, velocities, native
coordinates) that stage 2 can be delegated to PROJ as EPSG/PROJ point-motion coverage evolves:

1. **Frame transform (PROJ):** 4D transform evaluated at each point's *coordinate epoch* —
   `transformer.transform(x, y, h, tt)` with `tt` = decimal year
   (https://proj.org/en/stable/usage/transformation.html). Time-dependent parameters propagate as
   `P(t) = P(EPOCH) + Ṗ·(t − EPOCH)`
   (https://proj.org/en/stable/operations/transformations/helmert.html). Static steps ignore `tt`,
   so always passing it is safe.
2. **Epoch propagation (groundcontrol):** intra-frame move from the point's coordinate epoch to
   `target_epoch`: `x += vel_enu · (target_epoch − epoch)` using per-point velocities (NGL/MIDAS),
   a plate-motion model, or — if neither exists — leave the point at its own epoch and surface the
   `velocity·Δt` bound in the report. An ITRF-at-fixed-epoch target **requires** stage 2 for every
   ITRF-native source (NGL, ICESat-2); plate-fixed targets (NAD83(2011)) make stage 2 ≈ 0 by
   construction.

Key semantics (fixes plan Appendix B2's conflation):

- `t_epoch=2010` is a **fixed parameter of the EPSG operation** (EPSG:8970 "ITRF2014 to
  NAD83(2011) (1)", accuracy 0 m; EPSG:10334 for ITRF2020) — the parameter reference epoch. It
  never varies per point. Only `tt` does.
- **Verified:** when `tt` is omitted, PROJ silently evaluates at `t_epoch`
  (`helmert.cpp`: `t_obs = (t == HUGE_VAL) ? t_epoch : t`) — results bit-identical to
  `tt=2010.0`. Omitting `tt` is therefore accidentally correct for NGS 2010.00 coordinates and
  silently wrong for everything else. Always pass `tt`.
- **Three distinct epochs, all preserved in the schema with descriptive names (owner decision
  2026-07-03):** `coord_epoch` — the coordinate epoch (when the coordinate values are valid; the
  **only** epoch transforms consume — feeds the 4D `tt`, anchors velocity·Δt); `frame_epoch` —
  the realization's reference epoch (NAD83(2011) = 2010.00; positions are *reduced to* it; NaN
  for dynamic ITRF/IGS frames; **QC role:** `coord_epoch ≠ frame_epoch` on a plate-fixed frame
  flags an unreduced position); `measurement_datetime` (datetime64, UTC — human-friendly) +
  derived `measurement_epoch` (decimal year) — temporal filtering and uncertainty. None of these
  is the EPSG operation's fixed `t_epoch` parameter. OPUS output populates them differently
  across its two lines for the same mark/session (NAD83(2011): 2010.00/2010.00/2025-03-12;
  ITRF2014: NaN/2025.19/2025-03-12) — collapse any pair and one row is unrepresentable.
  Preserve-more principle: NaN where unknown rather than dropped — future transformations (PROJ
  point-motion growth, NATRF2022) will consume what today's cannot.
- **pyproj cannot represent "CRS @ epoch"** (verified: `CRS.from_user_input("EPSG:9989@2020.0")`
  and `COORDINATEMETADATA[...]` WKT both raise `CRSError`; open issue
  https://github.com/pyproj4/pyproj/issues/1558). PROJ itself can (projinfo/cs2cs
  `--s_epoch/--t_epoch`, ≥ 9.4). `target_epoch` is therefore **library-level metadata**
  (file-level keys), never a pyproj CRS property — do not burn time trying to encode it in WKT.

## 2. Transformer construction — the fail-loud pattern

**Verified failure mode:** with `us_noaa_g2018u0.tif` absent and `PROJ_NETWORK=OFF`, the default
`Transformer.from_crs("EPSG:6318+5703", "EPSG:7912")` **silently succeeds with the geoid step
dropped** — NAVD88 heights ride through as pseudo-ellipsoidal (~30 m wrong at Casa Grande); the
only signal is a `UserWarning`. The mandatory pattern:

```python
t = Transformer.from_crs(src, dst, always_xy=True,
                         allow_ballpark=False,   # hard ProjError at creation if only ballpark
                         only_best=True)          # pyproj ≥3.5 / PROJ ≥9.2
out = t.transform(x, y, h, tt, errcheck=True)     # only_best failures return inf WITHOUT errcheck
```

- **Verified:** `allow_ballpark=False` raises at creation on PROJ 9.7 (no silent ballpark);
  `only_best=True` alone returns **silent `inf`** unless `errcheck=True` raises `ProjError`.
  Docs: https://pyproj4.github.io/pyproj/stable/api/transformer.html,
  https://proj.org/en/stable/development/reference/functions.html#c.proj_create_crs_to_crs.
- **Preflight per (source-datum × target) pair** with
  `TransformerGroup(src, dst, area_of_interest=aoi, allow_ballpark=False)`: if
  `not tg.best_available`, surface `tg.unavailable_operations` (names + missing `Grid.url`s) and
  remediate via `tg.download_grids()` / `pyproj sync` / `PROJ_NETWORK=ON`. Empty
  `tg.transformers` → raise `NoTransformPathError` (plan B6), never `IndexError`.
- Ballpark detection post-hoc: there is no `Transformer.is_ballpark`; use
  `CoordinateOperation.has_ballpark_transformation` over `.operations` plus `"Ballpark" in
  .description`; `accuracy == -1` means *unknown*, store as null.
- `pyproj.aoi.AreaOfInterest(west, south, east, north)` takes **degrees**
  (https://pyproj4.github.io/pyproj/stable/api/aoi.html) — build it from
  `gdf.to_crs(4326).total_bounds` (or the AOI GeoJSON), never a projected `total_bounds`. Per plan
  B7: from the **datum-subset's** bounds.
- **PROJ networking is disabled by default** — opt in via `PROJ_NETWORK=ON` /
  `pyproj.network.set_network_enabled(True)`; grids cache chunked in the user dir (`cache.db`,
  300 MB default) — https://proj.org/en/stable/usage/network.html.
- Sanity check (verified): with grids present, the canonical Casa Grande chain
  `EPSG:6318+5703 → EPSG:7912` selects a 7-candidate group, stated accuracy 0.015 m, pipeline
  contains `vgridshift grids=us_noaa_g2018u0.tif` + `helmert … t_epoch=2010`, and confirms
  **h = H + N** (N ≈ −30.7 m; 400 m orthometric → ≈ 369 m ellipsoidal).

## 3. Frame aliases and realization notes

- **IGS14/IGS20 EPSG codes are a dead end in pyproj (verified):**
  `Transformer.from_crs("EPSG:9018", "EPSG:7911")` raises `ProjError`; `TransformerGroup` raises
  `IndexError`; `cs2cs` **silently applies a ballpark no-op**. Root cause: EPSG:9032 / EPSG:10179
  ("ITRF2014 to IGS14 (1)" / "ITRF2020 to IGS20 (1)") use "Time-specific Position Vector
  transform (geocen)" with all-zero parameters, which PROJ cannot instantiate. Since the tie is a
  zero Helmert, `ngl.py` **labels IGS14 data as `EPSG:7912` and IGS20 as `EPSG:9989`** at
  ingestion, keeps `ref_frame='IGS14'/'IGS20'` as provenance, and unit-tests the alias.
- **HARN / older NAD83 realizations** reach NAD83(2011) via PROJ-authored **grid-based NADCON5**
  concatenated operations (verified in proj.db, cdn.proj.org downloads) — they fall under the §2
  hard-error grid policy, not the Helmert path.
- **NATRF2022** (not "NATRF2020"; reference epoch 2020.00) — NGS FAQ
  https://geodesy.noaa.gov/datums/newdatums/FAQNewDatums.shtml; rollout 2024–2026. **Verified:**
  EPSG v12.029 contains zero NATRF/NAPGD records — until EPSG lands, an NATRF target is only
  expressible via NGS-supplied WKT; the user-defined-target design needs no code change.
- Modern ITRF↔ITRF chains (e.g. ITRF2014↔ITRF2008) resolve via published time-dependent Helmerts
  (verified, 0.01 m stated accuracy) — with the §1 caveat: they change frame, not epoch.

## 4. Targets without EPSG codes — programmatic 3D UTM construction

**Verified:** EPSG v12.029 has no CONUS "UTM / ITRF2008" zones (only Mexico, EPSG:6366-6372). The
first-application target "UTM / ITRF2008 epoch 2005.0 / HAE" is constructed programmatically:

```python
from pyproj import CRS, Transformer
from pyproj.crs import ProjectedCRS
from pyproj.crs.coordinate_operation import UTMConversion
target = ProjectedCRS(conversion=UTMConversion(12), geodetic_crs=CRS("EPSG:8999")).to_3d()
Transformer.from_crs(CRS("EPSG:6318+5703"), target, always_xy=True,
                     allow_ballpark=False, only_best=True).transform(lon, lat, H, 2005.0)
```

(verified end-to-end incl. GEOID18 + per-point epoch; epoch sensitivity ≈ 1.4 cm/yr horizontal,
consistent with NA plate motion). Directives: `target_crs` accepts an EPSG string, a WKT2 file, or
a `pyproj.CRS`; `crs.py` ships a `utm_3d(frame_crs, zone)` helper built this way instead of static
WKT files; the constructed WKT2 is exported into output metadata for provenance.

## 5. Heights: never `to_crs` — one explicit 3D/4D transformer

- geopandas `to_crs` transforms z only for geometries that *have* z (verified in 1.1.2 source);
  with the schema's 2D points + `height` column it **never touches the height**, and it has no
  time argument regardless. `.to_crs` is display/AOI-only.
- **Verified trap:** passing heights as `zz` through a transformer built from **2D** CRSs still
  runs them through cart+Helmert as pseudo-ellipsoidal heights with **no geoid step** —
  plausible-looking, wrong.
- Canonical path: one `Transformer.from_crs(src_3d_or_compound, target_3d, …)` applied to
  `(x, y, h, tt)` **arrays**; rebuild 2D geometry from transformed x/y, write transformed h into
  `height`. Source construction: `CRS("EPSG:6318+5703")` or `pyproj.crs.CompoundCRS(...)` — the
  `CRS(6318)+CRS(5703)` addition syntax is invalid (**verified `TypeError`**). Promote 2D targets
  with `CRS.to_3d()`. This is also the resolution of plan B7(b).

## 6. DEM side — coordinate epoch on rasters

- GDAL ≥ 3.4 supports per-dataset coordinate epoch: `SetCoordinateEpoch()/GetCoordinateEpoch()`;
  GeoTIFF `CoordinateEpochGeoKey` code **5120** (DOUBLE); CLI `-a_coord_epoch`,
  `-s/-t_coord_epoch` (https://gdal.org/en/stable/user/coordinate_epoch.html, RFC 81). Verified:
  round-trips through GeoTIFF; `gdalinfo` prints "Coordinate epoch: 2005.0".
- **rasterio has no coordinate-epoch API** (verified, 1.4.4) — read via
  `osgeo.osr.SpatialReference.GetCoordinateEpoch()`; most commercial DSM GeoTIFFs won't carry the
  key, so `assess_dem` takes `--dem-epoch` (and `--dem-vcrs`) as first-class inputs and **requires
  them for dynamic-frame DEMs** when undeclared. Stamp the epoch via GDAL when groundcontrol
  writes rasters. GDAL's epoch-aware warping is frame-relabeling too (§1) — the propagation caveat
  applies to rasters equally.

## 7. Transform provenance & logging

Silent wrong transforms are this library's primary failure mode; every coordinate operation must
be auditable after the fact.

**7.1 TransformRecord** — one per (source × native-datum subset) batch (the plan's B7 per-datum
loop granularity). Fields and the verified pyproj API for each:

| field | API / notes |
|---|---|
| `pipeline` | `Transformer.definition`. **Verified caveat:** on a multi-candidate `from_crs` transformer, `.definition`/`.description` return the literal `"unavailable until proj_trans is called"` and `.accuracy` is `-1.0` — *even after transforming*. Read them from a **`TransformerGroup` member** (the plan's `get_transformer` already returns one) or `Transformer.get_last_used_operation()` (pyproj ≥3.4 / PROJ ≥9.1) after a call; capture both and assert they match. |
| `description`, `steps[]` | `.description`; `.operations` → `CoordinateOperation.name/.method_name/.accuracy/.has_ballpark_transformation/.grids` (https://pyproj4.github.io/pyproj/stable/api/crs/coordinate_operation.html). |
| `accuracy_m` | `.accuracy`; `-1` = unknown → store null, never −1. |
| `source/target_crs_wkt2` | `.to_wkt(version="WKT2_2019")` + authority string (authority strings are not stable across EPSG revisions; WKT2 is the audit form). |
| `aoi`, `epochs` | The `AreaOfInterest` we passed (pyproj doesn't echo it); `epoch_mode: per-point\|fixed\|n/a-static` + min/max of `tt`. |
| `grids[]` | Per step: `Grid.short_name/url/available`; local path resolved by us (search `get_user_data_dir()` + `get_data_dir()` members) + sha256 + bytes; `materialization: file\|network-cache\|missing`. **No whole file exists for network-cached grids** (chunked `cache.db`) — pin those via `PROJ_DATA.VERSION` instead of a digest. |
| `ballpark`, `selection` | `allow_ballpark=False` at construction **and** post-check; `only_best=True`; candidates = `len(tg.transformers)`, `tg.best_available`, `unavailable_operations` count + missing-grid names (records what was *rejected* — half of auditability). |
| `environment` | `pyproj.__version__`, `proj_version_str`, `pyproj.database.get_database_metadata()` for `EPSG.VERSION/EPSG.DATE/PROJ.VERSION/PROJ_DATA.VERSION` (verified: v12.029 / 2025-10-03 / 9.7.1 / 1.24), `network.is_network_enabled()`, datadir paths. |

**Non-PROJ operations are first-class records** (`op_kind`): (a) velocity propagation
(`pos(target_epoch) = pos(epoch) + vel_enu·Δt`, model = MIDAS/plate), (b) NGL position-at-epoch
(path A window parameters, steps.txt segmentation, or path B MIDAS). Without these the audit chain
has gaps exactly where the epoch logic is most bespoke.

**7.2 Outputs.** `io.py` writes `<out>.provenance.json` (schema `groundcontrol/provenance-v1`:
environment, target, aoi, `transforms[]`, dispatcher status) next to every export, and embeds the
same JSON in GeoParquet file metadata under `groundcontrol:provenance` (+ `groundcontrol:target_crs`,
`groundcontrol:target_epoch`) alongside the reserved `geo` key. Mechanics (verified):
`GeoDataFrame.to_parquet` has no metadata kwarg (open FR geopandas#3182) — rewrite via
`pyarrow.parquet.read_table(...).replace_schema_metadata({**existing, ...})` + `write_table`;
cheap at control-point scale; `geo`/`pandas` keys and `read_parquet` round-trip confirmed. CSV/KML
get the sidecar only (+ a `# provenance:` header comment in CSV).

**7.3 Schema impact: one column.** `transform_id` (categorical string, e.g.
`"ngs:EPSG6318+5703:t01"`; chains get composite ids `"t02+t01"`, ordered) joins each row to its
`transforms[]` entry. The **audit invariant**: `native_x/y/h + native_crs + epoch + transform_id`
+ the recorded pipeline + pinned grids reproduce every coordinate exactly via
`Transformer.from_pipeline(pipeline).transform(...)`. `crs.replay(gdf, provenance)` asserts this
in tests — the round-trip *is* the audit test.

**7.4 `crs.explain()` — inspect before running.** `explain(source_crs, target_crs, aoi=None,
epoch=None)` builds the `TransformerGroup` (never transforms) and prints projinfo-style: all
candidates ranked with description + accuracy + instantiability; the selected `[0]` with its full
pipeline; every grid with availability/url/local path; `unavailable_operations` with the download
remedy; ballpark flags; and what `tt` will do (static vs 4D). Exposed as `--explain` on both CLIs
(prints and exits 0). The `ExplainResult` dataclass is the same type that becomes
`TransformRecord.selection` — explain-before, record-after.

**7.5 Fail-loud wiring.** Logging never replaces the raise; each raise carries the same diagnostic
dict the log emits. `NoTransformPathError` (B6) includes source/target authority+WKT2, AOI,
unavailable ops with missing `Grid.url`s, and the remediation hint. All `transform()` calls use
`errcheck=True`. After the first batch, assert
`get_last_used_operation().definition == chosen.definition` — a mismatch means PROJ late-binding
switched operations mid-run (per-point attribution inside one array call is otherwise
unobservable; pinning one op per datum-subset + this assertion is the mitigation).

**7.6 Logging levels** (stdlib only; package logger + `NullHandler` per the library convention):
INFO = one line per transform batch (op description, accuracy, n, grids) + dispatcher status;
DEBUG = full pipeline/WKT2/candidates/`pyproj.show_versions()`; WARNING = unavailable ops,
network-cached (unchecksummable) grids, unknown accuracy, dynamic-frame no-op residual; ERROR =
the shared diagnostic payload before every raise.

**7.7 Reproducibility pinning.** pyproj wheels bundle PROJ + proj.db but **ship no grids**
(https://pyproj4.github.io/pyproj/stable/transformation_grids.html); proj-data releases move
results at mm–cm level. CI: pin pyproj; materialize the needed grids via `pyproj sync` (or pinned
conda `proj-data`); commit `grids.lock` (name → sha256); run tests with `PROJ_NETWORK=OFF` so a
missing grid fails loud (§2) instead of silently fetching a newer one.

**What pyproj cannot expose (do not promise):** the on-disk path PROJ actually opened (we resolve
it ourselves; divergence only observable via `PROJ_DEBUG` stderr); per-point operation attribution
within one array call (see 7.5); `.definition/.accuracy` on multi-candidate `from_crs` objects
(placeholders — see 7.1); checksums for network-cached grids (see 7.7).

## 8. Validation fixtures & tests

All external resources verified live 2026-07-02:

| resource | URL |
|---|---|
| CORS MYCS2 per-station coords — ITRF2014 **and** NAD83(2011), both @2010.0, XYZ+llh+velocities; **frozen** product (fixture-stable) | `https://geodesy.noaa.gov/corsdata/coord/coord_14/<ssss>_14.coord.txt` (e.g. `drv6_14`); https://geodesy.noaa.gov/CORS/news/mycs2/mycs2.shtml |
| CORS MYCS3 — ITRF2020@**2020.0** + NAD83(2011)@**2010.0** w/ velocities in both ("Transformed from ITRF2020 (epoch 2020.0) position" — NGS's own adopted transform) | `https://geodesy.noaa.gov/corsdata/coord/coord_20/<ssss>_20.coord.txt` (e.g. `drv6_20`, `ac60_20` Alaska) |
| HTDP v3.6.0 (independent implementation) | https://geodesy.noaa.gov/TOOLS/Htdp/Htdp.shtml · https://github.com/noaa-ngs/HTDP |
| IERS/ITRF Helmert parameters | https://itrf.ign.fr/docs/solutions/itrf2020/Transfo-ITRF2020_TRFs.txt |
| NGS geoid API (GEOID18, per-point error) | `https://geodesy.noaa.gov/api/geoid/ght?lat=39.0&lon=-105.0` → N = −15.164 ± 0.036 m |
| NGS NCAT (NADCON5 realization transforms) | https://geodesy.noaa.gov/api/ncat/ |
| GeographicLib exact EGM synthesis test set (500k pts, 1 µm) | https://sourceforge.net/projects/geographiclib/files/testdata/GeoidHeights.dat.gz |

**Design insight:** all modern ITRF↔ITRF Helmert sets have **zero rotations** — those tests cannot
catch rotation-sign bugs. Rotation coverage comes from NAD83(2011) (R ≈ 25.9/9.4/11.6 mas ≈ 0.8 m;
Ṙ up to ~0.74 mas/yr ≈ 2.3 cm/yr) and one pre-1994 IERS row (ITRF93, nonzero R and Ṙ).

Prioritized tests (tolerance basis in parentheses):

1. **CORS `coord_14` pair** — ITRF2014@2010.0 → NAD83(2011)@2010.0 (isolates the Helmert; no
   propagation) for ~8 stations spanning CONUS + Alaska; ≤ 3 mm horizontal / ≤ 5 mm per component
   (published rounding only; wrong translation sign ≈ 1–4 m, wrong rotation sign ≈ 0.8 m, wrong
   rate epoch ≈ 23 cm/decade — all unmissable).
2. **CORS `coord_20` pair** — propagate ITRF2020 2020.0→2010.0 with the published velocity, then
   Helmert to NAD83(2011); also assert the commuted order agrees. Alaska (AC60) makes the 10-yr
   velocity term ~20 cm. ≤ 5 mm horizontal / ≤ 10 mm vertical. **This is the end-to-end truth
   test for the two-stage model (§1).**
3. **HTDP precomputed fixtures** — frame transforms at fixed epoch + epoch propagation with a
   *user-supplied* velocity (removes HTDP's internal velocity model from the comparison; exact);
   commit inputs/outputs/version, never a runtime dep. ≤ 2–3 mm.
4. **IERS Helmert-by-hand** — committed ~20-line numpy generator applies the quoted 14-parameter
   rows (ITRF2020→2014/2008 and →ITRF93 for rotation signs) at two epochs; library must match to
   ≤ 0.1 mm; epoch-sensitivity assertion (out(2025)−out(2015) = 1.5–2.5 mm for →2014) fails any
   frozen-parameter implementation; forward∘inverse ≤ 0.01 mm.
5. **Geoid fixtures** — saved NGS API responses (~6 CONUS points) + ~12 GeographicLib EGM
   synthesis rows (Iceland, S-hemisphere, antimeridian): h − H = N with the correct sign (a
   reversed vgridshift errs ~30 m at the Colorado point); GEOID18 ≤ 5 mm vs API; EGM2008 2.5′
   grid ≤ 1 cm and EGM96 15′ ≤ 3 cm vs exact synthesis.
6. **Synthetic velocity simulation** — three fabricated NGL stations (known linear velocity;
   zero-velocity control; Iceland-magnitude v_U = +0.025 m/yr) with sparse synthetic tenv3;
   propagation displacement exact to ≤ 0.1 mm, ≤ 1 mm end-to-end through frame transforms;
   includes the closed-form ENU→ECEF velocity rotation check (catches ENU-applied-as-ECEF).
7. **Closure/consistency matrix** — ~10 points (CONUS, Alaska, Iceland, Chile, NZ, antimeridian
   pair) × all target frames incl. both first-application deliveries: A→B→A ≤ 1 mm (≤ 5 mm geoid
   legs), triangle A→B→C→A ≤ 2 mm; **plate-fixed→plate-fixed near-no-op asserted on the pipeline**
   (`transformer.operations` shows conversion-only — no Helmert/vgridshift), not just numerically;
   antimeridian side-preservation (B1); no ballpark anywhere.
8. **Mixed-realization NGS batch** — recorded bounds JSON containing NAD83(2011)+HARN+1986 rows;
   spy asserts three *distinct* per-subset transformers (B7a — the original whole-batch bug shifts
   HARN rows by decimeters); our HARN/1986→2011 shifts match committed NCAT/NADCON5 golden values
   ≤ 1 cm; `decyear` hardened per plan B9 — known-value fixtures (2010.0 / 2010.5 non-leap /
   2020.5 leap, exact), tz/NaT handling, `decyear_inv` round-trip < 1 s, and the NGL tenv3
   dual-representation cross-check (every row carries both `YYMMMDD` and decimal `yyyy.yyyy`;
   assert agreement ≤ 0.003 yr across a fixture file).
9. **Fail-loud negative tests (11)** — missing GEOID18/NADCON5 grid raises with the grid name +
   remedy (never ballpark, `PROJ_NETWORK=OFF`); unknown realization raises listing supported;
   dynamic target without `target_epoch` raises; per-point epoch NaN under a dynamic target raises
   (policy flag to coerce); velocities-without-epoch raises; 2D-CRS-where-3D-required raises
   (the §5 trap); plate-fixed + Δepoch + no velocities → succeeds with the residual *reported*;
   `sample_raster` CRS mismatch raises; NGL `mod 360` regression (250° → −110°, bbox must match);
   NGS `'error'` body → retry not subdivision (B5).
10. **Committed fixtures ≈ 1–1.5 MB total**, all offline-capable: `tests/data/cors/` (~100 KB),
    `htdp/` (~30 KB), `helmert/` (~5 KB + generator in `tests/tools/`), `geoid/` (~5 KB),
    `ngl/` synthetic tenv3 + truncated real DataHoldings (~150 KB), `ngs/` recorded bounds + NCAT
    golden (~100–200 KB), `wkt/` (2 files), and — the piece that makes it all offline —
    **clipped 1°×1° windows of the real PROJ grids** (`us_noaa_g2018u0.tif`, EGM2008/EGM96,
    NADCON5 CONUS pair; same filenames PROJ resolves) under a test `PROJ_DATA` (~300–500 KB), with
    `PROJ_NETWORK=OFF` in conftest. `@network` tests: drift detectors (re-fetch one `coord_20` and
    diff — catches MYCS4/NATRF2022 transitions), live NGS geoid/NCAT probes, and the plan's live
    source-fetch integration tests.

Implementation priority: 4 → 5 → 1 → 6 → 9 → 7 → 2 → 3 → 8 → network (pure algebra and sign tests
first; they need no network even to build).
