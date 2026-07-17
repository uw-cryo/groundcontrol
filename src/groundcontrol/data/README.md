# Bundled data

## `pb2002_plates_decimated.geojson`

Tectonic plate polygons from the PB2002 model:

> Bird, P. (2003), An updated digital model of plate boundaries, *Geochem.
> Geophys. Geosyst.*, 4(3), 1027, doi:10.1029/2001GC000252.

Obtained via the GeoJSON conversion in
[fraxen/tectonicplates](https://github.com/fraxen/tectonicplates)
(Hugo Ahlenius, Nordpil, and Peter Bird), distributed under the
**Open Data Commons Attribution License (ODC-By) v1.0**
<https://opendatacommons.org/licenses/by/1.0/>.

Decimated for bundling (geometry only, `Code` property retained):
`simplify(0.05°, preserve_topology=True)` + coordinates snapped to a 0.001°
grid, 328 KB → ~113 KB. Used by `groundcontrol.crs.ITRF2020PMM(plate=None)`
for per-point plate assignment; the ~0.05° boundary blur is far inside the
physical width of real plate-boundary deformation zones.

## `ITRF2020-PMM.dat`

Verbatim ITRF2020 plate motion model product file (rotation poles in deg/Myr
plus the origin rate bias), from
<https://itrf.ign.fr/docs/solutions/itrf2020/ITRF2020-PMM.dat>:

> Altamimi, Z., Métivier, L., Rebischung, P., Collilieux, X., Chanard, K., &
> Barnéoud, J. (2023), ITRF2020 plate motion model, *Geophys. Res. Lett.*,
> 50, e2023GL106373, doi:10.1029/2023GL106373.

Kept for provenance; the pole table in `groundcontrol.crs` is transcribed
from this file and unit-tested against the paper's independent mas/yr form.
