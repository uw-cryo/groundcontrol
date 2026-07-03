# Control-point sources survey — candidates beyond the implemented set

Web-research survey (2026-07, live-verified endpoints; citations inline). Implemented
sources (3DEP checkpoints, NGS/OPUS, NGL) and already-planned ones (ICESat-2, GAGE, EPN,
IGS) are excluded. Feeds the v2 source roadmap in `plan.md`.

## Ranked shortlist

1. **LINZ NZ Geodetic Marks (+ Vertical Marks)** — 133k + 108k marks, CC-BY 4.0 WFS
   (free key), per-point integer order + ellipsoidal heights; NZGD2000 is semi-dynamic so
   it exercises the deformation-model/epoch machinery that generalizes to every national
   register. Cleanest first international provider. https://data.linz.govt.nz/layer/50787-nz-geodetic-marks/
2. **IGN France Géoplateforme geodesy WFS** — ~460k open points (387k leveling benchmarks
   live-counted; RBF/RDF/RGP layers), keyless WFS returning GeoJSON, Etalab 2.0;
   RGF93 v2b (ETRF2000@2019.0). Biggest single open yield found.
   https://data.geopf.fr/annexes/ressources/wfs/geodesie.xml
3. **UK pair: EA LIDAR Ground Truth Surveys + OS Net/passive** — the EA dataset is the only
   true 3DEP-checkpoint analog found anywhere (GNSS checkpoints ±3 cm RMSE, survey date +
   surface type, OGC API-Features, OGL v3); OS adds ~1,000 ETRS89/ODN points as two static
   open TXT files. https://environment.data.gov.uk/dataset/16b4d492-0c0d-410b-9732-65eebcc3d9f9
4. **NOAA CORS MYCS3 coordinates** — per-station coord files + bulk position/velocity CSVs
   (HTTPS + S3 `noaa-cors-pds`); ITRF2020@2020.0 AND NAD83(2011)@2010.0 with sigmas +
   velocities; public domain; LOW effort. Use the *monument* entry (ARP/L1/monument listed
   separately) and filter rooftop stations. https://geodesy.noaa.gov/CORS/news/mycs3/mycs3.shtml
5. **ITRF2020 SSC files** — 1,329 GNSS DOMES sites, positions@2015.0 + velocities +
   per-station mm sigmas + discontinuity (SOLN) + post-seismic (PSD) files; the
   authoritative sparse global anchor and the reference implementation for
   discontinuity/PSD handling. https://itrf.ign.fr/ftp/pub/itrf/itrf2020/
6. **GEDI L2A (strictly filtered)** — 25 m footprints to ±51.6° lat; sub-m vertical only on
   flat open ground after quality/degrade/sensitivity filtering (~10 m geolocation ×
   tan(slope)); the only densifier in checkpoint-poor low latitudes; LOW-MED via SlideRule
   gedil2ap. https://www.earthdata.nasa.gov/data/catalog/lpcloud-gedi02-a-002

**Architecture note:** one generic OGC-WFS/ArcGIS-FeatureServer adapter with per-source
field-mapping config would unlock ~12 open national registers at near-zero marginal code:
NRCan passive (~110k) + Ontario/Alberta/BC, NSW SCIMS (250k, per-mark positional
uncertainty) + other AU states, PDOK RDinfo (NL, CC0), Denmark, Finland NLS, Spain IGN,
swisstopo, Sweden (CC0), Norway, Germany-NRW, and US state services (e.g. NC OneMap).

## Verdict table (surveyed, one line each)

| Source | Verdict |
|---|---|
| NOAA CORS MYCS3 | **Add** — mm-cm dual-frame w/ velocities+sigmas; filter ARP/rooftop |
| NOAA CO-OPS tidal bench marks | Skip — API elevations null, m-level coords, tidal plumbing |
| State DOT/RTN monuments | Skip as class — no aggregator; bluebooked subset already in NGS |
| FAA/NGS PACS-SACS | Skip — inside the NGS datasheet DB (redundant) |
| USACE U-SMART | Skip — DoD-gated in practice, no API |
| BLM PLSS corners | Skip — horizontal-only, m-level, no vertical |
| Geoscience Australia | No national DB; **NSW SCIMS add** (anon REST, per-mark PU/LU); QLD/VIC/SA/TAS open |
| NRCan CSRS passive | **Add (next tier)** — ~110k pts, anonymous JSON bbox endpoint, sigmas+epochs |
| LINZ NZ | **Add first** (see shortlist) |
| GSI Japan | Mostly skip — Survey Act redistribution limits; GEONET F5 = only programmatic entry |
| OS Net / passive UK | **Add** — two open TXT files |
| EA LIDAR Ground Truth | **Add** — only true 3DEP-checkpoint analog found |
| IGN France | **Add** (see shortlist) |
| SAPOS/Germany | Skip SAPOS; NRW HFP CSV worth a cheap adapter; GREF redundant w/ EPN |
| ES/CH/SE/NO/FI/DK/NL | Batch via the generic OGC adapter — all open |
| SONEL | Marginal — vertical-velocity table only; coastal VLM context |
| ITRF2020 SSC | **Add** (see shortlist) |
| IGS cumulative SINEX | Later — freshness layer over ITRF2020; auth + SINEX plumbing |
| JPL sideshow | Optional — easy global table but IGS14, largely redundant w/ NGL |
| SIRGAS2022 | **Add (next tier)** — 587 stations, only dense-ish Latin-America anchors |
| OSM survey_point | Skip — no accuracy metadata, ele errors up to 50 m, ODbL share-alike |
| GEDI L2A | **Add (filtered)** (see shortlist) |
| ICESat/GLAS GLAH14 | Niche — dm-class flat terrain; TOPEX-ellipsoid (−0.707 m) + saturation traps |
| AHN/PNOA/LiDAR-HD/ELVIS checkpoints | None published — open checkpoint releases are rare (UK EA is the exception) |
| DORIS/VLBI/SLR | Skip — tiny subsets of ITRF2020 SSC |

## Coverage gaps (ICESat-2 priority confirmation)

- **Africa**: AFREF never operational; no open registers → ICESat-2/GEDI effectively the only control.
- **Latin America**: only SIRGAS's sparse CORS anchors → altimetry-dependent.
- **Most of Asia**: Japan gated, China/India/SE Asia closed/absent; APREF sparse CORS only.
- **High latitudes >51.6° outside NA/EU/NZ**: no registers AND no GEDI → ICESat-2 is the sole
  source — confirming its top v2 priority.
- **Everywhere**: registers publish order/class, not per-point sigmas → a per-network
  order→σ mapping table is needed (ties into D3); GNSS-type sources give antenna/monument
  heights, not ground → monumentation filtering belongs in the schema.
