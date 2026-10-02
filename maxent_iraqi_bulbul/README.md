# Automatic MaxEnt model – Iraqi bulbul (*Pycnonotus leucotis*)

This pipeline builds a species distribution model with MaxEnt, starting from **occurrence records only**.
It downloads the climate data, picks the variables, tunes and fits the model, and writes the maps and an HTML report.

```
occurrences ─► cleaning + spatial thinning ─► WorldClim 2.1 bioclim (auto-download, cropped)
            ─► background points ─► collinearity filter (VIF + |r|)
            ─► MaxEnt tuning (feature classes × regularization, spatial block CV)
            ─► final model ─► AUC / TSS / Boyce / omission ─► importance + response curves
            ─► suitability + binary maps (GeoTIFF + PNG) ─► optional future (CMIP6) + MESS
            ─► outputs/report.html
```

## Run it

```bash
cd maxent_iraqi_bulbul
pip install -r requirements.txt
# 1) put your records in data/occurrences/occurrences.csv (see the note in that folder), then
python run_pipeline.py
# or point at any file:
python run_pipeline.py --occurrences path/to/my_records.xlsx
```

If no occurrence file is present, records of *Pycnonotus leucotis* inside the study area are downloaded from GBIF.

**On GitHub (no install needed):** the workflow `.github/workflows/maxent.yml` runs the whole pipeline on every push
that touches this folder (or manually from the *Actions* tab → *Run workflow*). The results are committed back to
`maxent_iraqi_bulbul/outputs/` and also attached to the run as a downloadable artifact.

## Settings (`config.yaml`)

| Setting | Default | Meaning |
|---|---|---|
| `species.name` | *Pycnonotus leucotis* | Name used for GBIF and in titles |
| `occurrences.thin_km` | 10 | Minimum distance between kept records (reduces sampling bias) |
| `study_area.bbox` | Iraq + ~1° margin | Area mapped and reported (maps, suitable area, future projections) |
| `study_area.training_bbox` | 34–64°E, 20–40°N | Area the model is trained on: the western range (Turkey/Levant to Iran, the Gulf and western Pakistan), which includes climates hotter than Iraq; `[34, 20, 78, 40]` adds the Indian subcontinent |
| `occurrences.gbif_supplement` | true | Add GBIF records from the training area to your file (duplicates removed) |
| `environment.resolution` | 2.5m (~4.5 km) | WorldClim resolution: 10m, 5m, 2.5m, 30s |
| `environment.keep_variables` | bio_1, bio_12 | Never removed by the collinearity filter |
| `environment.extra_layers` | rivers + tree/cropland/built | Non-climate predictors: distance to major rivers (Natural Earth, capped at `river_distance_cap_km` = 50 km) and land-cover fractions (ESA WorldCover 2021), held constant in future projections |
| `background.buffer_km` | 300 | Background sampled within this distance of records (accessible area) |
| `model.feature_classes` / `regularization_multipliers` | 5 × 6 grid | Candidate models compared by cross-validation |
| `model.selection_metric` | auc_then_or10 | Among settings within 0.005 of the best CV AUC, pick the lowest omission rate, then the strongest regularization |
| `future.enabled` | true | Project to every combination of `gcms` × `ssps` × `periods` (default: MPI-ESM1-2-HR and UKESM1-0-LL, SSP2-4.5 and SSP5-8.5, 2041–2060 and 2061–2080) |
| `future.ensemble` | true | Also average the climate models for each SSP × period (mean suitability, lowest MESS, and a model-agreement map) |

## Outputs (`outputs/`)

* `report.html` – the full report in one file, with figures, tables and interpretation notes.
* `rasters/` – `suitability_current.tif` (cloglog 0–1), binary maps at the max-TSS and 10th-percentile thresholds, plus future/MESS if enabled.
* `figures/` – maps, ROC curve, tuning plot, variable importance, response curves and the correlation matrix.
* `tables/` – cleaned and thinned occurrences, background, tuning results, importance, response curves and `summary.json`.
* `model/maxent_model.pkl` – the fitted model (load it with `elapid.load_object`).
* `pipeline.log` – the full run log.

## Self-test

```bash
python tests/run_self_test.py
```

This builds synthetic climate layers and a virtual species with a known niche, runs the whole pipeline, and
checks that the predicted map matches the true one (Spearman ρ > 0.7; the current version gets 0.997).

## Method notes

* MaxEnt is fitted with [`elapid`](https://github.com/earth-chris/elapid), a Python implementation of Maxent
  (same features: linear, quadratic, hinge, product, threshold; cloglog output).
* Cross-validation uses ENMeval-style "block" partitioning: 4 spatial blocks with equal numbers of records.
  Among settings within 0.005 of the best mean test AUC, the one with the lowest 10th-percentile omission
  rate is chosen (ties go to stronger regularization), then refit on all data.
* Clean your own records carefully. Captive or escaped birds, wrong coordinates and strong observer bias around
  cities all affect the result.
* Cite: Phillips et al. 2006/2017 (MaxEnt), Fick & Hijmans 2017 (WorldClim 2), and GBIF (create a download DOI for publication).
