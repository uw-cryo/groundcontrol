# data/

Built data files land here and are not committed (see `.gitignore`); they are published as
versioned data releases.

- `checkpoints-3dep-2.0-qc.parquet`: the USGS 3DEP survey checkpoints (ver. 2.0) with height and
  position QC, written by `scripts/build_checkpoints_3dep.py` and published on Kopah under the same name.
  See [docs/checkpoints_3dep.md](../docs/checkpoints_3dep.md).
- `checkpoints_3dep_opr_samples.parquet` (and `*_parts/`): DEM sample cache of
  `scripts/screen_checkpoints_3dep.py`.
- `checkpoints_3dep_position_test.csv`, `checkpoints_3dep_position_test_workunits.csv` (committed):
  results of `scripts/position_test_checkpoints_3dep.py`, read by
  [docs/checkpoints_position_investigation.ipynb](../docs/checkpoints_position_investigation.ipynb).
