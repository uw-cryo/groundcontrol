# USGS 3DEP survey checkpoints: data release and QC

The `3dep` source reads the USGS 3DEP survey checkpoint database as a GeoParquet file built and
quality-checked in this repo. This page is the reference for the file: where it comes from, its
columns and flags, and how to rebuild and re-screen it. **The findings and their evidence are in
[`checkpoints_quality_assessment.ipynb`](checkpoints_quality_assessment.ipynb)**, written for the
data provider:

- Part 1: height unit, decimal and datum errors, checked against each project's OPR DEM
- Part 2: re-running VDatum on the source fields (project 11309, the 88 suspect heights)
- Part 3: source coordinates that do not fit their recorded coordinate system
- Part 4: how the published lon,lat positions were made (re-labelled, not transformed)
- Part 5: how a lon,lat position error becomes a height error on a slope

## Source and published file

- **Source:** Miller, B.Y., and Cannici, C., 2025, *Consolidated Standardized Survey-Grade
  Checkpoints 3DEP 2004 to 2025* (ver. 2.0, December 2025): U.S. Geological Survey data release,
  <https://doi.org/10.5066/P13NOW9E> ([ScienceBase item](https://www.sciencebase.gov/catalog/item/67075e6bd34e969edc59c3e7)),
  `Checkpoints_3DEP_2004_2025.gpkg` from `Checkpoints_3DEP_2004_2025.zip` (MD5
  `5bd22479b1429087227d8bb4af0ebce0`).
- **Published** on the UW Kopah object store as
  <https://s3.kopah.uw.edu/tacolab/groundcontrol/checkpoints-3dep-2.0-qc.parquet> (USGS release
  version + `-qc`; plain HTTPS with range requests, so browser tools such as stac-map can read it).
  `src/groundcontrol/sources/checkpoints_3dep.py` downloads it, checks it against
  `PARQUET_SHA256`, and caches it under a name that includes the hash.
- **Rebuild:** `python scripts/build_checkpoints_3dep.py` writes
  `data/checkpoints-3dep-2.0-qc.parquet` (not committed) in a few minutes, fetching the NGS geoid
  grids it needs from the PROJ CDN. The build is byte-reproducible: the same source and library
  versions give the same SHA-256.
- **Provenance:** the parquet's `groundcontrol` key-value metadata records the source checksum,
  library versions, changes and QC rules.

**A new USGS release needs the assessment redone.** The build is pinned to the ver. 2.0 MD5 and
stops if ScienceBase publishes a different zip, and every QC rule asserts its row count. Re-run
the notebook and the screen (below), revise `QC_RULES` / `H_RULES` in the build script, and
publish under a new name (`checkpoints-3dep-<version>-qc.parquet`).

## How the file differs from the GeoPackage

| | GeoPackage | parquet |
|---|---|---|
| attribute values | | identical, all 145,299 rows (checked cell by cell) |
| `unique_identifier`, `collection_date` | | renamed `id`, `datetime` (STAC-style names) |
| `OBJECTID` | row counter 1..N | dropped |
| geometry | Point Z, compound EPSG:6349 | 2D Point, EPSG:6318 (the horizontal part); XY identical |
| added columns | | `z_meter_qc`, `qc_height_flag`, `qc_horizontal_flag` |
| layout | | Hilbert-sorted rows, 15 row groups of 10k, `bbox` covering column (GeoParquet 1.1), zstd |

**Why 2D:** the height already lives in columns, and geometry Z would be a third copy that has to
agree with either the USGS value or the repaired one. One compound CRS is also wrong for some
rows: Puerto Rico heights are PRVD02, and Hawaii has no NAVD88.

**Why the layout:** a bbox read touches one of 15 row groups instead of the whole file.

## QC columns

Every USGS value, including `z_meter_vdatum_update`, the geometry and the `source_*` fields, is
unchanged. `z_meter_qc` is the height the `3dep` source uses: `z_meter_vdatum_update` except where
the table says otherwise. Flags are null on unflagged rows, and each column holds at most one flag
per row.

| column | flag | rows | meaning | `z_meter_qc` |
|---|---|---:|---|---|
| `qc_height_flag` | `m_labelled_usft` | 554 | metres labelled U.S. survey feet (CA_Sonoma_2013; the `*_58946_Sacramento*` rows of 58946) | repaired |
| `qc_height_flag` | `m_labelled_ft` | 57 | metres labelled international feet (OR_HarneySilver_2020_A20 VVA points) | repaired |
| `qc_height_flag` | `ft_labelled_m` | 39 | U.S. survey feet labelled metres (Fork_Salt_River_NRCS; `source_geoid=UNK`, so not VDatum-harmonized) | repaired |
| `qc_height_flag` | `decimal_shift` | 34 | decimal point misplaced by 10× or 100× (ME_SouthernArea_2012) | repaired, per-point factor |
| `qc_height_flag` | `ellipsoid_height_labelled_navd88` | 100 | ellipsoid heights labelled NAVD88 (HI_Oahu_2012) | NaN |
| `qc_height_flag` | `vdatum_scaled_not_rerun` | 1,130 | project 11309: scaled by 0.3048 instead of re-running VDatum | VDatum re-run |
| `qc_height_flag` | `vdatum_mismatch` | 88 | more than 2 cm from re-running VDatum on its own source fields; which is right cannot be told | unchanged |
| `qc_horizontal_flag` | `source_crs_mismatch` | 4,748 | source coordinates land more than 1 km from the published lon,lat position in their recorded `source_horizontal_epsg`; do not use the source fields as recorded | unchanged |
| `qc_horizontal_flag` | `source_datum_not_nad83_2011` | 29,591 | source datum other than NAD83(2011): the published lon,lat position does not match a PROJ transform of the source coordinates; uncertain by 0.2–1.2 m, which matters on slopes | unchanged |

A unit repair removes only the unit error, `z + (src × true unit − src × labelled unit)`, so the
VDatum shift USGS applied stays in. Where the true unit is a foot, the U.S. survey foot is assumed
(the two feet differ by 2 ppm).

**Filtering.** In the parquet the flags are ordinary columns. In groundcontrol's normalized output
they ride in `raw` (they are 3DEP-specific, so not schema columns) and
`checkpoints_3dep.expand_attributes` lifts them out:

```python
from groundcontrol.sources import checkpoints_3dep
gdf = checkpoints_3dep.expand_attributes(gdf)   # qc_height_flag, qc_horizontal_flag, project_id, source_geoid
clean = gdf.query("qc_height_flag.isna() and qc_horizontal_flag.isna()", engine="python")
```

Also known and left as published: `source_geoid=UNK` rows were never VDatum-harmonized (see
[accuracy conventions](accuracy_conventions.md)), and some rows share an XY position with another
row.

## The height screen (`scripts/screen_checkpoints_3dep.py`)

The screen found the unit, decimal and datum errors behind the first five height flags. It is the
one part of the method not in the notebook.

**Reference:** every checkpoint was surveyed to assess one lidar project (`project_id`). The screen
maps `project_id` to work units with WESM (downloaded fresh) and samples each work unit's OPR source
DEM through its VRT at the checkpoint. Checkpoints with no OPR DEM, no VRT or no-data at the point
are not screened (it reports how many).

**Test: the DEM / checkpoint ratio.** A unit error *multiplies* the height, while terrain change,
vegetation and survey error *shift* it, so DEM / checkpoint stays near 1 for a good point and lands
on a fixed value for each kind of error:

| DEM / checkpoint | meaning |
|---|---|
| ≈ 1 | consistent |
| ≈ 3.2808 (1 / 0.3048) | checkpoint values are metres but were converted as if feet |
| ≈ 0.3048 | checkpoint values are feet but were treated as metres |
| ≈ 10 or 100 | decimal point misplaced |
| a constant **difference**, not a ratio | a datum offset (e.g. ellipsoid heights labelled NAVD88) |

Per `project_id`, using every point type and only points with |z| > 5 m (near sea level the ratio
is unstable), a project is flagged when its median ratio is more than 10% from 1 (with at least 5
such points), when at least 5 points sit within 3% of one signature, or when its median difference
exceeds 5 m with a small spread. For each flagged project the screen prints the ratios by
`point_type` and `id` prefix (which tracks the source file); that breakdown sets each rule's edges
in `QC_RULES`.

**The DEM's own unit is checked independently.** A DEM's unit label can carry the same error as its
checkpoints, and then the two agree (Fork_Salt_River's DEM is in feet but WESM labels it metres).
So each work unit's DEM unit (from its compound CRS, band unit or WESM `vert_crs`) is checked
against the Copernicus GLO-30 DEM, an independent satellite DEM in metres that only decides feet
vs metres, never judges a checkpoint:

1. A point whose DEM value is at least 100 votes: GLO-30 / DEM ≈ 1 means metres, ≈ 0.3048 feet.
   80% of votes must agree.
2. In low terrain with no such point, the median GLO-30 / DEM over at least 10 points decides when
   it is at least twice as near (log scale) to one unit as to the other.
3. Otherwise a work unit takes the unit of its project's other work units with the same unit
   label, if at least two are decided and all agree; else it stays unverified.

Each rule exists because a simpler one failed on real data: a work-unit median left small-sample
work units unverified; pooling a project's median mis-set mixed-unit work units; single votes at
low elevation were fooled by tree canopy. WESM unit labels that disagree with GLO-30 concern the
DEMs, not the checkpoints; `--units-report` lists them.

**Running it:**

```bash
python scripts/screen_checkpoints_3dep.py --report projects.csv --units-report units.csv
python scripts/screen_checkpoints_3dep.py --height z_meter_qc --check   # after a rebuild: expect no flags
```

A full run reads ~1,800 work-unit VRTs and ~1,100 GLO-30 tiles, roughly 30 minutes with 16
threads. Samples are cached in `data/checkpoints_3dep_opr_samples.parquet`, each finished task in
`data/checkpoints_3dep_opr_samples_parts/`, so an interrupted run resumes; a rerun on the cache
takes seconds.
