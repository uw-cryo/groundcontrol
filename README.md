# groundcontrol

Fetch ground control points for an arbitrary AOI and assess DEM accuracy — with rigorous
3D CRS / datum / epoch handling as the core competency.

> Given an arbitrary AOI, fetch available control points (with correct datum/epoch handling);
> and given a DEM, automatically sample the control, run the accuracy assessment, and produce
> the analysis + visualization for vertical/horizontal accuracy.

**Status: pre-alpha — Increment 1 (control-point fetch MVP) under construction.**
See [`docs/plan.md`](docs/plan.md) for the full design and
[`docs/crs_implementation.md`](docs/crs_implementation.md) for the verified CRS/epoch
implementation directives, transform-provenance spec, and validation-fixture design.

## Planned capabilities

- **Control sources:** USGS 3DEP checkpoints, NGS/OPUS, user-supplied points (MVP);
  Nevada Geodetic Lab GNSS time series (global) next; ICESat-2 as the global
  dense-control path later.
- **One normalized schema** across all sources, in a **user-chosen target 3D CRS + epoch**,
  with full transform provenance (every coordinate operation auditable and replayable).
- **DEM accuracy assessment:** robust stats (median/NMAD), ASPRS NVA/VVA metrics,
  residual figures and reports.

## Install (development)

```bash
pip install -e ".[dev]"
pytest -m "not network"
```

## Origin

Core functionality extracted and generalized from prior UW-cryo project code
(`casagrande` NGS/OPUS fetching + a private DEM-accuracy toolkit), 2026.
Datum/epoch recipes follow
[uw-cryo/3D_CRS_Transformation_Resources](https://github.com/uw-cryo/3D_CRS_Transformation_Resources).
