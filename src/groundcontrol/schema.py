"""Normalized control-point schema — the central abstraction.

Every source returns a GeoDataFrame with the same columns so downstream
sampling/accuracy code is source-agnostic. Semantics are defined in
docs/plan.md (schema section) and docs/crs_implementation.md.

Transform placement (see plan "Transform placement" clarification): providers
return native-frame, *schema-shaped* frames; the dispatcher performs the CRS
landing into the user-chosen target frame; ``normalize()`` here handles
columns/dtypes/validation only.

Open decisions are marked TODO(D#) per the plan's Open Decisions block —
do NOT silently change these semantics; adjudicate in the plan first.

Extracted and generalized from prior UW-cryo project code, 2026.
"""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column specification
# ---------------------------------------------------------------------------
# TODO(D1): geometry is 2D POINT; the vertical lives in the `height` column.
#           CRS-label dimensionality (2D vs 3D code) to be settled before freeze.
# TODO(D2): `point_type` currently conflates marker type (gnss/monument/checkpoint)
#           and assessment class (NVA/VVA); split pending adjudication.
# TODO(D3): `acc_h`/`acc_v` confidence convention (1-sigma m) and promotion of
#           per-axis `sig_e/n/u` to first-class columns pending adjudication.
# TODO(D4): `height_datum` vs `vertical_crs` dedup pending adjudication.
# D5 (decided at scaffold, per plan): `raw` is a JSON-encoded *string* column —
#           Python dicts do not survive the GeoParquet/CSV round-trip.
# TODO(D6): the tt rule (which epoch feeds the 4D transform) is PROVISIONAL —
#           see docs/crs_implementation.md §1; numerical arbiter = CRS fixtures.
#
# The three time columns are deliberately distinct (preserve-more principle):
#   frame_epoch          realization reference epoch (NaN for dynamic frames)
#   coord_epoch          when the coordinate values are valid (feeds the 4D tt)
#   measurement_datetime / measurement_epoch   when the measurement was made

_STR = "string"
_F64 = "float64"

#: Non-geometry schema columns, in canonical order: name -> pandas dtype.
COLUMNS: dict[str, str] = {
    "source": _STR,             # ngs / opus / ngl / 3dep_checkpoint / user
    "id": _STR,                 # PID / 4-char GNSS ID / checkpoint id / user id
    "height": _F64,             # scalar height in the target frame (HAE unless target orthometric)
    "height_datum": _STR,       # provenance of the original height datum  TODO(D4)
    "horizontal_crs": _STR,     # provenance of original values only
    "vertical_crs": _STR,       # provenance of original values only
    "ref_frame": _STR,          # realization: IGS20 / IGS14 / NAD83(2011) / ...
    "frame_epoch": _F64,        # realization reference epoch (decimal yr; NaN dynamic)
    "coord_epoch": _F64,        # coordinate epoch (decimal yr) — feeds the 4D tt
    "measurement_datetime": "datetime64[ns, UTC]",  # acquisition datetime (human-friendly)
    "measurement_epoch": _F64,  # decyear(measurement_datetime)
    "point_type": _STR,         # gnss / monument / NVA / VVA / control  TODO(D2)
    "acc_h": _F64,              # reported accuracy (m)  TODO(D3)
    "acc_v": _F64,              # reported accuracy (m)  TODO(D3)
    "vel_e": _F64,              # nullable velocities (m/yr; MIDAS for GNSS)
    "vel_n": _F64,
    "vel_u": _F64,
    "epoch_residual_m": _F64,   # NaN = not assessed; 0.0 = epoch-reconciled; >0 = bound (m)

    "native_x": _F64,           # original coordinates — lossless re-targeting
    "native_y": _F64,
    "native_h": _F64,
    "native_crs": _STR,         # frame of the native coordinates
    "raw": _STR,                # JSON string of source-specific extras (D5)
    "transform_id": _STR,       # join key into the export's provenance record
}

#: Columns a provider MUST populate (everything else may be NaN/pd.NA).
REQUIRED_NON_NULL = ("source", "id")


class SchemaError(ValueError):
    """A GeoDataFrame does not conform to the groundcontrol control-point schema."""


def empty(crs=None) -> gpd.GeoDataFrame:
    """Return a zero-row GeoDataFrame with the full column set and dtypes.

    Used by the dispatcher for failed/empty sources so ``pd.concat`` never
    breaks (see the dispatcher contract in docs/plan.md).
    """
    data = {name: pd.Series(dtype=dtype) for name, dtype in COLUMNS.items()}
    return gpd.GeoDataFrame(data, geometry=gpd.GeoSeries([], dtype="geometry"), crs=crs)


def normalize(df: pd.DataFrame, source: str) -> gpd.GeoDataFrame:
    """Coerce a provider frame to the schema shape: full column set + dtypes.

    Shape/validation only — CRS landing into the target frame is the
    dispatcher's job (plan: "Transform placement"). ``df`` must already carry
    point geometry (a GeoDataFrame) in the provider's native frame.
    """
    if not isinstance(df, gpd.GeoDataFrame):
        raise SchemaError(f"normalize() needs a GeoDataFrame with geometry (source={source!r})")
    out = df.copy()
    out["source"] = source
    if "raw" in out.columns and len(out) and not isinstance(out["raw"].iloc[0], (str, type(pd.NA))):
        # D5: dicts are serialized once, at the schema boundary.
        out["raw"] = out["raw"].map(lambda v: json.dumps(v) if isinstance(v, dict) else v)
    for name, dtype in COLUMNS.items():
        if name not in out.columns:
            fill = np.nan if dtype == _F64 else (pd.NaT if dtype.startswith("datetime") else pd.NA)
            out[name] = pd.Series([fill] * len(out), dtype=dtype, index=out.index)
        else:
            try:
                out[name] = out[name].astype(dtype)
            except (TypeError, ValueError) as e:
                raise SchemaError(f"column {name!r} not coercible to {dtype}: {e}") from e
    # canonical column order: schema columns then geometry
    out = out[[*COLUMNS.keys(), out.geometry.name]]
    validate(out, require_crs=False)
    return out


def validate(gdf: gpd.GeoDataFrame, require_crs: bool = True) -> None:
    """Raise :class:`SchemaError` unless ``gdf`` conforms to the schema."""
    missing = [c for c in COLUMNS if c not in gdf.columns]
    if missing:
        raise SchemaError(f"missing schema columns: {missing}")
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise SchemaError("not a GeoDataFrame")
    if require_crs and gdf.crs is None:
        raise SchemaError("GeoDataFrame has no CRS; schema requires a single target frame")
    if len(gdf):
        if gdf.geometry.has_z.any():
            raise SchemaError("geometry must be 2D points; vertical lives in `height` (D1)")
        geom_types = set(gdf.geometry.geom_type.unique()) - {None}
        if geom_types - {"Point"}:
            raise SchemaError(f"geometry must be Point, got {sorted(geom_types)}")
        for col in REQUIRED_NON_NULL:
            if gdf[col].isna().any():
                raise SchemaError(f"column {col!r} must be non-null for every row")
