# Control-point accuracy conventions — WORK IN PROGRESS (D3)

**Status: NOT ADJUDICATED.** This captures the 2026-07 research findings (two web-research
agents, all claims cited) plus the owner's review notes. The proposed mappings below are
**provisional** — reliable per-point accuracies are core to this library (and must support
user-side filtering by accuracy), so nothing here is frozen until reviewed against more than
the three initial test sites. Native source metrics are always preserved in `raw`;
`acc_h`/`acc_v` carry only values whose semantics are certain (see "Currently implemented").

## Owner review notes (to resolve in the next research round)
- **NGS acc_v from netAccU (owner, 2026-07):** geoid-model error is limited (hybrid GEOID18 is
  fitted to NAVD88 benchmarks, relative error ~1–2 cm), so the 1σ ellipsoid-height `netAccU`
  RSS'd with an expected geoid/NAVD88 error term may be a reasonable orthometric acc_v —
  **conditional on `vertSource`**: valid when the orthoHt derives from the ellipsoid height
  (`GPS OBS`), irrelevant for leveled/VERTCON3 heights that never touched the ellipsoid.
  Candidate: `acc_v(GPS OBS) = sqrt((2·netAccU/100)² + σ_geoid²)` at 95%.
- **GPS OBS heights:** horizontal accuracy should be ~2× better than vertical for GNSS —
  but depends on single-point vs differential fix; the source class alone may not determine it.
- **Casa Grande calibration range:** many of the NGS ADJUSTED marks there should have
  excellent *horizontal* network accuracy documented in the old calibration-range/baseline
  reports — consult those before assigning generic order-based defaults to such sites.
- The posOrder 1/2/3 → absolute-meters defaults force *relative* specs into absolute values;
  review carefully (alternative: NaN + carry order as a QC label only).
- Convention choice (95% vs 1σ meters) not yet decided.

## Verified field semantics (authoritative — safe to rely on)

### NGS datasheet API (`/api/nde`) — from api/nde/meta + dsdata.pdf
| field | meaning | units/confidence |
|---|---|---|
| `netAccHz` | horizontal network accuracy, **circular 95%** ("2-sigma in 2-dimensions") | **cm** |
| `netAccN`/`netAccE` | per-axis network accuracy | cm, **1σ** |
| `netAccU` | **ellipsoid-height** network accuracy | cm, **1σ** |
| `netAccEh` | ellipsoid-height network accuracy | cm, **95%** (= 2×netAccU) |
| `posOrder` | FGCS-1984 horizontal order — **frozen pre-2007 legacy label**, *relative* accuracy (1st = 1:100,000 …); NGS ceased using orders with NAD83(NSRS2007) | — |
| `posSource=SCALED` | map-scaled position: **"±6 seconds" / "±180 meters Scaled"** (printed on datasheets); no accurate-SCALED exceptions exist; empirically these are bench marks (vertical control) | — |
| `vertOrder`/`vertClass` | FGCS-1984 leveling order/class — "remains the only accuracy measure" for orthometric heights (dsdata p.13); spec is *relative* (0.5–2.0 mm·√km) | — |
| `vertSource=VERTCON3` | NAVD88 = NGVD29 + VERTCON 3.0 shift; NGS prints "(±2cm)"; CONUS σ = 2.4 cm (TR NOS NGS 68 Table 8-1); per-point 1σ error grids exist via NCAT | — |
| `vertSource=GPS OBS` | 2 cm / 5 cm ellipsoid-height standards + geoid model (NOS NGS 59) when cm-published; dm-published = coarser procedures | — |

**Critical:** `netAcc*` describe the **ellipsoid height**, never the orthometric height — NGS
publishes **no** network accuracy for orthometric heights. Orthometric accuracy must come from
`vertSource`/order semantics.

### OPUS shared (`/api/opus`) — from api/opus/meta + OPUS About + Schwarz 2006
- `latP2p`/`lonP2p`/`ellHtP2p` = **peak-to-peak**: the **range (max−min) of the 3
  single-CORS-baseline solutions**, meters. Not σ, not FWHM.
- `orthoHtP2p` = ellipsoid-height range **already padded with a geoid error estimate**.
- NGS-endorsed conversion: **σ ≈ P2P / 1.6926** (Schwarz 2006; OPUS Projects UG §4.3:
  "approximately 1.7 times the expected standard deviation"); 95% ≈ 1.16 × P2P. A σ estimated
  from 3 samples is itself noisy (var = 0.2755 σ²).
- Good-solution rules of thumb (OPUS About): overall RMS < 3 cm, P2P < 5 cm.

### USGS 3DEP checkpoints (ScienceBase 67075e6bd34e969edc59c3e7)
- `accuracy` column: **all 145,299 values are Null by design** — "USGS did not require an
  accuracy attribute during the data collection time frame" (FGDC metadata, verbatim). Future
  LBS-Online-era points will populate it (units/confidence still unspecified upstream).
  Treat 0.0 as null artifact, never "perfect".
- `point_type=BVA` = **Bathymetry** Vertical Accuracy (submerged/water-edge; 718 nationally)
  — exclude from topographic control by default. `Unknown` (12,144) = valid survey point,
  land cover unverifiable.
- `source_geoid=UNK` (1,254 points): **not harmonized** by the VDatum update —
  `z_meter_vdatum_update` is stale for these.
- Spec basis for defaults: ASPRS 2014 §7.9 (checkpoints ≥ **3×** better than tested product;
  Ed. 2 2023 relaxed to 2×) + LBS Table 4 (QL1/QL2 RMSEz ≤ 0.10 m) ⇒ post-2014 checkpoints
  ≤ 3.3 cm RMSEz; NGS-58 (2/5 cm @95%) is the survey standard ASPRS Annex C.5 invokes.
- `z_meter_vdatum_update`: VDatum-harmonized to GEOID18 (CONUS/PR) / GEOID12B (HI/AK);
  harmonization noise ~1–2 cm (inference; example shift 1.7 cm).

## Proposed mapping (PROVISIONAL — do not implement without owner sign-off)
Convention candidate: meters at 95% (FGDC-STD-007 reporting standard). Priority: measured
network accuracy → measured P2P → NGS source-type statement → order/class convention.
[N] = NGS/USGS-stated; [C] = convention requiring review.

| source / condition | acc_h (95%) | acc_v (95%) | basis |
|---|---|---|---|
| NGS `netAccHz` present | netAccHz/100 | — | [N] |
| NGS `vertSource=VERTCON3` | — | 0.05 (or per-point NCAT grid) | [N] σ=2.4 cm |
| NGS ADJUSTED leveling | — | 0.02 | [C] NGS-anchored |
| NGS `GPS OBS` | (owner: ~acc_v/2? fix-type dependent) | 0.05 cm-published / 0.2 dm | [N]+[C] |
| NGS `posSource=SCALED` | NaN (±180 m — not horizontal control) | — | [N] |
| NGS `posSource=HD_HELD1/2` | 3 / 10 | — | [N] value, [C] confidence |
| NGS ADJUSTED posOrder 1/2/3 | 0.5/1.0/2.0 (⚠ relative→absolute) | — | [C] ⚠ review |
| OPUS | 2.45·σ̄, σ=P2P/1.6926 | 1.16×orthoHtP2p | [N] factor, [C] combination |
| 3DEP ≥2014 | 0.05 | 0.065 (ASPRS 3× ÷ QL2) | spec |
| 3DEP <2014 | 0.05 | 0.121 | spec |
| `vertSource=SCALED` / BVA / UNK-geoid | — | NaN + QC flag | [N] semantics |

Caveat for reports: all NGS accuracies are *as-adjusted* — they do not track mark motion
(Casa Grande is a documented subsidence area; leveling-era heights on unreleveled marks can
err far beyond their class).

## Currently implemented (until D3 is adjudicated)
- `acc_h` (NGS): `netAccHz`/100 — the one directly NGS-stated 95% circular value.
- `acc_v` (NGS): **NaN** — `netAccU`/`netAccEh` describe the ellipsoid height, not our
  orthometric `height`; populating it was wrong and has been removed.
- `acc_h`/`acc_v` (OPUS): **raw P2P values** (max of lat/lon P2P; orthoHtP2p) — the native
  metric, conversion deferred; semantics documented here.
- `acc_*` (3DEP): NaN (`accuracy` column is authoritatively null).
- All native fields preserved in `raw` for later re-derivation.

## Sources
dsdata.pdf (geodesy.noaa.gov/DATASHEET/dsdata.pdf) · api/nde/meta · api/opus/meta ·
OPUS About (geodesy.noaa.gov/OPUS/about.jsp) · OPUS Projects UG §4.3 · Schwarz 2006
(geodesy.noaa.gov/CORS/Articles/SchwarzJSE.pdf) · FGCS 1984 Standards & Specifications ·
FGDC-STD-007.1/.2 · NOAA TR NOS NGS 68 (VERTCON 3.0) · NOAA TM NOS NGS 58 / NGS 59 ·
NGS HARN pages · ASPRS Positional Accuracy Standards Ed.1 (2014) & Ed.2 (2023) ·
USGS Lidar Base Specification 2022 rev. A + LBS Online · ScienceBase 67075e6bd34e969edc59c3e7
(FGDC metadata + data dictionary) · live datasheets CZ1515, CZ0993, DU1581.
