# CLI reference

Two entry points, one input contract: `groundcontrol-fetch` fetches control for an AOI,
`groundcontrol-assess` assesses one or more DEMs against it. Both share the AOI forms,
the cache flags and the figure flags below.

## AOI

`--aoi` on the CLI, `fetch_control(aoi)` in Python. Any of:

| Form | Example | Notes |
|------|---------|-------|
| bbox string | `--aoi=-115.3,36.0,-114.9,36.3` | `minx,miny,maxx,maxy` in EPSG:4326 lon/lat. Use the `=` form for negative longitudes. |
| vector file | `--aoi site.geojson` | GeoJSON preferred; any OGR-readable format (GPKG, Shapefile, KML, FlatGeobuf, ...) and GeoParquet; any CRS. Multiple features dissolve into one AOI. |
| elevation raster | `--aoi dsm.tif` | Any GDAL format (GeoTIFF, COG, VRT, ...). The AOI is the raster's grid extent, reprojected from the raster CRS, with no mask read. |
| in memory (Python) | `fetch_control(gdf, ...)` | GeoDataFrame, GeoSeries or shapely geometry. |

`--valid-footprint` makes a raster AOI use the valid-data footprint instead of the grid
extent. The footprint is band 1's nodata or alpha mask, read at 1024 px or less, so edge
membership is approximate. Untagged NaN counts as valid, so set the nodata tag.

## Products

`--product NAME=PATH`, repeatable: gridded elevation rasters to assess, in any GDAL
format. A `NAME` containing `DTM` is assessed under the bare-earth rules, where the VVA
checkpoints validate it. Any other name (`DSM`, `DEM`, ...) is assessed under the surface
rules.

On `groundcontrol-assess`, positional inputs classify themselves: a raster is a product,
a vector is the AOI, and with only a vector the command runs the AOI-only fetch.

### The product's frame

The assessment needs the product's 3D CRS, horizontal and height datum. Three ways to
state it:

- The embedded CRS. A product whose own CRS is compound or 3D needs nothing more.
- `--vdatum`, paired with the raster's own 2D horizontal CRS. Values: `ellipsoid`,
  `ellipsoid:<realization>` (`itrf2014`, `itrf2020`, `g2139`, ...), any vertical CRS
  (`EPSG:5703` NAVD88, `EPSG:3855` EGM2008), or a product preset (`3dep`, `cop30`,
  `precision3d`, `earthdem`, `arcticdem`, `rema`). A preset applies the researched
  source-delivery frame for that product line; it is a dated snapshot, since products
  version and datums move. See the [vertical datum field guide](vdatum.md).
- `--target-crs`: an EPSG compound such as `EPSG:6341+5703` (NAD83(2011) / UTM 12N +
  NAVD88), inline WKT, or a `.wkt`/`.prj` file. For custom frames,
  `groundcontrol.geodesy.with_vdatum`, `build_utm_nad83_2011_3d` and `write_crs_file`
  produce the WKT.

A realization is required for WGS 84 ensemble horizontals such as `EPSG:326xx`: the bare
ensemble is about 2 m of deliberate ambiguity and `ellipsoid` alone is refused on it. A 2D
product with neither `--vdatum` nor `--target-crs` is refused with the common choices
listed. The height datum lives in the product report and is never guessed; a wrong or
assumed vertical datum shows up as a geoid-sized bias, about 30 m at Casa Grande, which
is the error class this library exists to prevent.

## groundcontrol-fetch

```bash
# Las Vegas bbox, live fetch from every provider (the default set)
groundcontrol-fetch --aoi=-115.3,36.0,-114.9,36.3 --out control.parquet
# a polygon, restricted to two sources
groundcontrol-fetch --aoi site.geojson --sources ngs,opus --out control.parquet
# wherever a DEM has data
groundcontrol-fetch --aoi dsm.tif --out control.parquet
# outside the NAD83 area of use the default landing refuses; pick a geographic
# realization as the landing frame, never an ensemble
groundcontrol-fetch --aoi nepal_aoi.geojson --sources ngl --landing-crs EPSG:7912 --out control.parquet
```

Output files land next to `--out`, prefixed by its stem.

### Landing frame

By default control lands horizontally in EPSG:6318 (NAD83(2011)) with heights untouched
and their per-row `vertical_crs` provenance kept. `--landing-crs` overrides the landing
frame. A 3D input such as `EPSG:7912` lands at its 2D counterpart, `EPSG:9000`; the
written file's CRS is promoted back to the 3D code when every row's vertical datum is
that frame's ellipsoid.

`--target-crs` and `--target-epoch` are not implemented for fetch, and passing either one
raises. Landing in the product frame happens in `groundcontrol-assess`.

A source that cannot land in the requested frame degrades into the per-source status
report rather than poisoning the rest. Plate-fixed NGS and 3DEP control into a dynamic
ITRF frame is one such case, pending declared-epoch landing.

The assess step then needs the matching declaration:
`groundcontrol-assess --control nepal_control.parquet --source-crs EPSG:7912 ...`. The
default source is the CONUS `EPSG:6318+5703` contract, and a mismatched control frame is
refused, never reinterpreted.

### Figures and basemaps

By default `groundcontrol-fetch` writes the standard control figure set next to `--out`:
the labeled control map, and the MIDAS velocity maps and NGL station series where GNSS
control is present. It fetches Esri World Imagery tiles to underlay them.

- `--no-figures`: write only the control file and provenance.
- `--basemap none`: keep the figures but skip every tile fetch.
- `--context-sheets`: write the full per-point contact sheets for every point. This is
  the slow figure component.

### Shared flags

- `--sources`: the providers to query. Default: every provider, `3dep,ngs,opus,ngl,faa`.
- `--faa-ownership`: NASR ownership codes the `faa` source keeps. Default `PU,PR`
  (civil facilities); `all` widens it.
- `--refresh`: force re-download of every shared-cache file the run touches. Caches
  otherwise refresh on their own staleness windows.
- `--quiet`: suppress INFO progress logging.
- `--valid-footprint`: see AOI above.

## groundcontrol-assess

```bash
groundcontrol-assess dem.tif                       # the embedded CRS declares the heights
groundcontrol-assess dem.tif --vdatum EPSG:5703    # a 2D CRS: state the vertical datum
groundcontrol-assess --product DSM=dsm.tif --product DTM=dtm.tif --target-crs dsm_frame.wkt --aoi site.geojson
```

The output directory defaults to `<input stem>_groundcontrol/` next to the input, and
the site name to the file stem. The figure hillshade is computed from the product itself.

Options:

- `--aoi`: restrict or outline the area. Defaults to the product footprint.
- `--control`: reuse a fetched control cache instead of fetching.
- `--source-crs`: the control's frame when it is not the CONUS default.
- `--hs NAME=PATH`: a pre-rendered hillshade for very large mosaics.
- `--site-name`, `--outdir`.
- `--target-epoch`: the epoch the control is propagated to.
- `--landing-crs`: the horizontal landing frame for the control fetch. The default is the
  CONUS EPSG:6318 + NAVD88 contract. When the AOI is outside the NAD83 area of use, the
  landing frame is derived from the target datum, so an ITRF2014 target lands at EPSG:7912.
  Geographic CRS only.
- `--method`, `--radius`: point sampling with `linear` (the default) or `nearest`, or a
  neighborhood median within a radius in raster units. The two flags are mutually exclusive.
- `--point-lim`, `--vendor-lim`, `--wide-lim`: pin the empirical dz color and axis tiers
  when figures must be comparable across runs.
- `--rgb`, `--intensity`: an orthoimage and a lidar intensity raster for the contact
  sheets.
- `--no-figures`: skip all figure output.
- `--context-sheets`, `--basemap`, `--sources`, `--faa-ownership`, `--refresh`, `--quiet`:
  as for fetch.

Unless `--no-figures` is set, `groundcontrol-assess` writes the capped residual review
sheets, the 6 largest and 6 smallest dz per control subset and product. `--basemap none`
renders them without tiles.

## Outputs

| File | Contents |
|------|----------|
| `<site>_control.parquet` + `.provenance.json` | fetched control in the normalized schema (`schema.py`), landed horizontally in EPSG:6318 by default, with per-row `vertical_crs` provenance and per-source status. Reused on the next run. |
| `<site>_assessed.parquet` + `.provenance.json` | control landed in the product frame, with `h_ell` and the per-point `xform_acc_m` transform budget, and `h_<NAME>` and `dz_<NAME>` (product minus control, in the product's vertical reference) per product. Unsampled points (nodata, mosaic gaps) stay NaN, never dropped. |
| `<site>_dz_stats.csv` | one row per product and control segment (3DEP NVA/VVA, GNSS occupation classes, NGS monuments, ...): `n`, `n_valid`, `n_out`, robust `median_m` and `nmad_m`, parametric `mean_m`, `std_m`, `rmse_m`, `le90_m`, `le95_m` after a 3 NMAD gate, `xform_acc_m`, and `applies` (whether that segment validates that product class). |
| `<site>_validation_dz_<NAME>.png` | per product: the dz map over the hillshade plus dual-track histograms for the survey-grade segments and the NGS monuments ([example](gallery.md#2b-the-clis-own-output-bring-your-own-dem)). |
| `<site>_<subset>_gallery_<tier>[_pN].jpg` | per-point context contact sheets by subset (`cors`, `opus`, `gnss_other`, `faa_runway`, `3dep_nva`, `3dep_vva`) at two tiers, 120 m context and 30 m native-pixel. Panels adapt to the available layers: RGB imagery (your `--rgb` ortho and/or `--basemap` tiles, credited on the sheet), `--intensity` grayscale when given, and shaded relief per product ([example](gallery.md#7-per-point-context-contact-sheets)). Opt-in with `--context-sheets`; an AOI-only fetch gets RGB panels only. |
| `<site>_<subset>_dz_<NAME>_largest.jpg`, `_smallest.jpg` | per product and control subset (the six above plus `ngs_monument`, `ngs_best`): the 6 largest and 6 smallest absolute dz points as imagery, intensity and relief windows at the two tiers, labeled with dz. The review product that separates a biased control point (mast, roof, canopy, moved pad) from a product error. Written unless `--no-figures` is set. |

The CLI also prints the per-source row counts, the selected transform with its stated
accuracy, and the stats table to stderr.
