# lidar_tools/geodesy.py → groundcontrol consolidation map

Phase E of the stage-2 epoch plan. groundcontrol is the canonical CRS /
transformation library for the uw-cryo ecosystem (owner decision,
2026-07-04); `lidar_tools/src/lidar_tools/geodesy.py` grew a parallel engine
(741 lines as of lidar_tools `c1b9560`, 2026-07). This document is the
symbol-by-symbol map so the merge is mechanical and the two designs don't
drift. The general-purpose subset is ported as `groundcontrol.geodesy` on
this branch (see git history); lidar_tools itself is **not** modified — see
"Packaging constraint" below for when it can re-point.

## Symbol map

Line numbers refer to lidar_tools `c1b9560`.

| lidar_tools geodesy.py | groundcontrol today | disposition |
|---|---|---|
| `geographic_base_epsg` (L68) + `NAD83_FAMILY_GEOGRAPHIC` | `crs.ngs_datum_to_epsg` (string input only) | **ported** — CRS-object variant, general base-datum extraction/validation |
| `geoid_grid_hint` (L111) + `GEOID_GRID_HINTS` | — | **ported** — geoid-name → PROJ grid fragment table |
| `utm_zone_label` (L122) | `figures.py` calls `estimate_utm_crs()` only | **ported** |
| `build_utm_realization_3d` (L212) + wrappers `build_utm_g2139_3d` / `build_utm_nad83_2011_3d` / `build_utm_g1674_3d` / `build_utm_itrf2020_3d` / `build_utm_itrf2008_3d` / `build_utm_itrf2014_3d` (L146–273) | — | **ported** — 3D UTM builders on explicit realizations, incl. the ITRF-alias null-tie workaround knowledge |
| `build_utm_target` (L289) + `OUTPUT_DATUM_BUILDERS` | — | **ported** — zone + output-datum → (3D CRS, canonical WKT filename) |
| `epoch_pinned_pipeline` (L339) | — | **ported** — its own docstring flagged it for migration; PROJ pipeline resolution with `--t_epoch` baked in (first `projinfo` subprocess use in groundcontrol) |
| `preflight_vertical_transform` (L534) | `crs.get_transformer` (same fail-loud TransformerGroup + AOI pattern) | **ported** as `geodesy.preflight_transform`; the delta over `get_transformer`: grid auto-download, `prefer_grids=`, area-of-use containment assert, provenance dict return. `get_transformer` itself unchanged (additive rule); §7.4 `explain()` should eventually absorb both |
| `navd88_offset` (L488) | expressible via `crs.get_transformer`, no helper | **ported** — local geoid-undulation helper |
| `write_crs_file` (L499) | — | **ported** — WKT2:2019 provenance sidecar |
| `library_versions` (L524) | `io._environment` (partial) | **ported** — de-GDAL'd: reports PROJ/pyproj (+ GDAL only if importable) |
| `set_coordinate_epoch` (L687) | `crs.is_dynamic_frame` is the dynamic-ness predicate | **stays in lidar_tools** — in-place raster stamping via `osgeo.gdal`/`osr` (COG layout); groundcontrol is deliberately GDAL-free. A rasterio-based raster epoch-stamp is future §6 work |
| `build_3857_navd88_compound` (L319), `build_ept_3857_navd88_compound` (L415), `build_ept_3857_nad83_2011` (L449) | `io._compound_export_crs` is unrelated | **stays in lidar_tools** — encodes 3DEP EPT-on-AWS null-datum-tie relabeling semantics (point-cloud source domain knowledge, not general geodesy) |
| Constants: `DEFAULT_COORDINATE_EPOCH`, `WGS84_G2139_EPSG`, `NAD83_2011_EPSG`, ITRF alias EPSGs (L36–209) | literal EPSG strings scattered | **ported** with the functions that use them |

## Dependency gaps (why the port is not a copy-paste)

- **`osgeo` (GDAL Python bindings)**: geodesy.py imports `gdal`/`osr` at module
  scope and calls `gdal.UseExceptions()` at import time. groundcontrol imports
  no `osgeo` anywhere (rasterio/rioxarray only) and keeps it that way — the
  GDAL-dependent `set_coordinate_epoch` stays behind; `library_versions` makes
  the GDAL key optional.
- **`projinfo` subprocess**: `epoch_pinned_pipeline` shells out to `projinfo`
  (ships with PROJ; present transitively via the pyproj wheel). This is
  groundcontrol's first subprocess use — isolated in `geodesy.py`, fail-loud
  when the binary is missing.
- Everything else geodesy.py uses (`TransformerGroup`, `AreaOfInterest`,
  `CompoundCRS`/`ProjectedCRS`/`UTMConversion`, `pyproj.network`) is already
  satisfied by groundcontrol's `pyproj>=3.5` floor.

## lidar_tools re-point map (for the future opt-in)

`cli.py` has no geodesy import (reaches it via `driver`). Three modules would
re-point:

| module | symbols used |
|---|---|
| `driver.py` (L92, L95) | `build_utm_target`, `write_crs_file` |
| `dsm_functions.py` (~L1076, function-local) | `navd88_offset` |
| `pdal_pipeline.py` (heaviest; L363–L922) | `build_utm_target`, `write_crs_file`, `NAD83_2011_EPSG`, `geographic_base_epsg`, `geoid_grid_hint`, `preflight_vertical_transform`, `library_versions`, `epoch_pinned_pipeline`, `DEFAULT_COORDINATE_EPOCH`, plus the EPT-compound builders and `set_coordinate_epoch` that stay local |

A partial re-point leaves `pdal_pipeline.py` importing from both locations —
acceptable as a transition state.

## Packaging constraint (why lidar_tools is untouched today)

lidar_tools is conda/pixi-first and public; groundcontrol is setuptools/PyPI
and currently a **private** repo. Making lidar_tools depend on groundcontrol
would break public installs until groundcontrol is published (PyPI or
conda-forge) or the repo goes public. lidar_tools does not import
groundcontrol today, so it keeps working untouched until it opts in; when it
does, the duplicated general-purpose symbols in its geodesy.py should become
thin re-exports of `groundcontrol.geodesy` (or be deleted with the imports
re-pointed per the table above).
