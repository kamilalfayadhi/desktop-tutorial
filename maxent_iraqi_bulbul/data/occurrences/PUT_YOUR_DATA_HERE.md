Save your Iraqi bulbul occurrence file here as `occurrences.csv` (or change `occurrences.file` in `config.yaml`).

Minimum content: one row per record, with a longitude and a latitude column in decimal degrees (WGS84), e.g.

```
species,longitude,latitude,year
Pycnonotus leucotis,44.36,33.31,2019
Pycnonotus leucotis,47.78,30.51,2021
```

Column names are detected automatically (longitude/latitude, lon/lat, x/y, decimalLongitude/decimalLatitude...).
Optional columns used for cleaning if present: `year`, `coordinateUncertaintyInMeters`, `basisOfRecord`.
CSV, TSV and Excel (.xlsx) files are accepted.

If no file is present, the pipeline downloads records from GBIF automatically.
