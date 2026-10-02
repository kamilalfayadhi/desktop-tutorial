#!/usr/bin/env python3
"""Fully automatic MaxEnt species distribution modelling pipeline.

    python run_pipeline.py                      # uses config.yaml
    python run_pipeline.py --occurrences my.csv # use your own occurrence file
    python run_pipeline.py --config other.yaml

Steps: occurrences -> cleaning & thinning -> climate layers -> background -> variable selection
-> model tuning (spatial CV) -> final model -> evaluation -> maps -> (future) -> HTML report.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import elapid as ela
import numpy as np
import pandas as pd
import rasterio
import yaml

from sdm import environment as env
from sdm import modeling, occurrences, plots, report

HERE = Path(__file__).resolve().parent
log = logging.getLogger("maxent")


def cell_area_km2(template: dict) -> np.ndarray:
    """Area of each grid cell (km²) for a lon/lat grid."""
    t = template["transform"]
    lat = t.f + (np.arange(template["height"]) + 0.5) * t.e
    km_per_deg = 111.32
    return (abs(t.a) * km_per_deg) * (abs(t.e) * km_per_deg) * np.cos(np.radians(lat))[:, None] * np.ones(template["width"])


def resolve(path, base=HERE):
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else base / p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--occurrences", help="occurrence file (overrides config)")
    ap.add_argument("--output", help="output directory (overrides config)")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg_dir = Path(args.config).resolve().parent
    if cfg["environment"].get("local_dir"):
        cfg["environment"]["local_dir"] = str(resolve(cfg["environment"]["local_dir"], cfg_dir))
    if args.occurrences:
        cfg["occurrences"]["file"] = str(Path(args.occurrences).resolve())
    out = resolve(args.output or cfg["output_dir"], cfg_dir)
    for sub in ("figures", "rasters", "tables", "model"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    cache = HERE / "data" / "cache"
    seed = cfg["model"]["random_seed"]
    bbox = cfg["study_area"]["bbox"]  # area mapped and reported
    train_bbox = cfg["study_area"].get("training_bbox") or bbox  # area the model is trained on

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(out / "pipeline.log", mode="w")])

    # 1. Occurrences ----------------------------------------------------------------------
    log.info("STEP 1/8  Occurrence data")
    occ_cfg = cfg["occurrences"]
    occ_file = resolve(occ_cfg["file"], cfg_dir)
    have_file = bool(occ_file and occ_file.exists())
    if not have_file and not occ_cfg.get("download_from_gbif_if_missing", True):
        raise FileNotFoundError(f"Occurrence file not found: {occ_file}")
    sources = []
    if have_file:
        user = occurrences.read_occurrence_file(occ_file).assign(source="user file")
        sources.append(user)
    if not have_file or occ_cfg.get("gbif_supplement", False):
        log.info("Downloading GBIF records for the training area %s", train_bbox)
        gbif = occurrences.cached_gbif(cache, occ_cfg.get("gbif_cache_days", 30), cfg["species"]["name"],
                                       train_bbox, occ_cfg.get("gbif_tile_deg", 1.0),
                                       occ_cfg.get("gbif_max_per_tile", 600))
        gbif.to_csv(out / "tables" / "gbif_raw_download.csv", index=False)
        if have_file and "gbifID" in user.columns:  # records already in the user's file
            gbif = gbif[~gbif["gbifID"].astype(str).isin(user["gbifID"].astype(str))]
        sources.append(gbif.assign(source="GBIF"))
    raw = pd.concat(sources, ignore_index=True)
    occ_source = " + ".join(
        f"{'user file ' + occ_file.name if src == 'user file' else 'GBIF'} ({n:,})"
        for src, n in raw["source"].value_counts(sort=False).items())
    cleaned, steps = occurrences.clean(raw, train_bbox, occ_cfg.get("min_year"),
                                       occ_cfg.get("max_coord_uncertainty_m"))

    # 2. Environment ----------------------------------------------------------------------
    log.info("STEP 2/8  Environmental layers")
    layers, template = env.load_layers(cfg, cache, train_bbox)
    all_vars = list(layers)
    valid_all = ~np.isnan(np.stack([layers[v] for v in all_vars])).any(axis=0)

    # keep presences that fall on valid climate cells, one per cell, then thin by distance
    pres_env_all = env.extract(layers, template, cleaned["lon"], cleaned["lat"])
    on_land = ~pres_env_all.isna().any(axis=1).to_numpy()
    cleaned = cleaned[on_land].reset_index(drop=True)
    steps["on_valid_climate_cells"] = len(cleaned)
    rows, cols = map(np.asarray, rasterio.transform.rowcol(template["transform"], cleaned["lon"], cleaned["lat"]))
    cleaned = cleaned.loc[~pd.Series(list(zip(rows, cols))).duplicated().to_numpy()].reset_index(drop=True)
    steps["one_per_raster_cell"] = len(cleaned)
    pres = occurrences.thin(cleaned, occ_cfg.get("thin_km", 0), seed)
    steps[f"thinned_{occ_cfg.get('thin_km', 0)}km"] = len(pres)
    for k, v in steps.items():
        log.info("    %-32s %6d", k, v)
    if len(pres) < 10:
        raise RuntimeError(f"Only {len(pres)} usable presence records - at least 10 are needed (ideally 30+).")
    cleaned.to_csv(out / "tables" / "occurrences_cleaned.csv", index=False)
    pres[["lon", "lat"]].to_csv(out / "tables" / "occurrences_model.csv", index=False)

    # 3. Background -----------------------------------------------------------------------
    log.info("STEP 3/8  Background points")
    bg = modeling.sample_background(template, valid_all, pres["lon"].to_numpy(), pres["lat"].to_numpy(),
                                    cfg["background"]["n_points"], cfg["background"].get("buffer_km"), seed)
    log.info("    %d background points", len(bg))

    # 4. Variable selection ---------------------------------------------------------------
    log.info("STEP 4/8  Variable selection (collinearity)")
    bg_env_all = env.extract(layers, template, bg["lon"], bg["lat"])
    variables, removed = env.select_variables(bg_env_all, cfg["environment"]["vif_threshold"],
                                              cfg["environment"]["correlation_threshold"],
                                              cfg["environment"].get("keep_variables") or [])
    log.info("    retained: %s", ", ".join(variables))
    plots.correlation_heatmap(out / "figures" / "correlation.png", bg_env_all.corr(), variables)
    x_pres = env.extract(layers, template, pres["lon"], pres["lat"])[variables]
    x_bg = bg_env_all[variables]
    pd.concat([bg, x_bg], axis=1).to_csv(out / "tables" / "background.csv", index=False)
    pd.concat([pres[["lon", "lat"]], x_pres], axis=1).to_csv(out / "tables" / "presence_environment.csv", index=False)

    # 5. Tuning ---------------------------------------------------------------------------
    log.info("STEP 5/8  Model tuning")
    if cfg["model"].get("cv_method", "block") == "block":
        pf, bf = modeling.block_folds(pres, bg)
    else:
        pf, bf = modeling.random_folds(len(pres), len(bg), 5, seed)
    tuning = modeling.tune(x_pres, x_bg, pf, bf, cfg["model"]["feature_classes"],
                           cfg["model"]["regularization_multipliers"], seed)
    tuning.to_csv(out / "tables" / "tuning_results.csv", index=False)
    best = modeling.choose_best(tuning, cfg["model"].get("selection_metric", "test_auc"),
                                cfg["model"].get("auc_tolerance", 0.005))
    log.info("    selected: features=%s rm=%s (test AUC %.3f)", best["features"], best["rm"], best["test_auc"])
    plots.tuning_plot(out / "figures" / "tuning.png", tuning, best)

    # 6. Final model & evaluation ---------------------------------------------------------
    log.info("STEP 6/8  Final model and evaluation")
    model = modeling.fit(modeling.make_model(best["features"], float(best["rm"]), seed), x_pres, x_bg)
    ela.save_object(model, out / "model" / "maxent_model.pkl")
    p_pres, p_bg = modeling.predict(model, x_pres), modeling.predict(model, x_bg)
    metrics, thresholds, (fpr, tpr) = modeling.evaluate(p_pres, p_bg)
    metrics.update({"cv_test_auc": float(best["test_auc"]), "cv_test_auc_sd": float(best["test_auc_sd"]),
                    "cv_auc_diff": float(best["auc_diff"]), "cv_omission_rate_10pct": float(best["or10"])})
    for k, v in metrics.items():
        log.info("    %-28s %.3f", k, v)
    importance = modeling.permutation_importance(model, x_pres, x_bg, seed=seed)
    importance.to_csv(out / "tables" / "variable_importance.csv")
    curves = modeling.response_curves(model, x_bg)
    pd.concat({k: v for k, v in curves.items()}, names=["variable"]).to_csv(out / "tables" / "response_curves.csv")
    plots.roc_plot(out / "figures" / "roc.png", fpr, tpr, metrics["training_auc"])
    plots.importance_plot(out / "figures" / "importance.png", importance)
    plots.response_plot(out / "figures" / "response_curves.png", curves, x_pres)

    # 7. Prediction maps ------------------------------------------------------------------
    log.info("STEP 7/8  Prediction maps")
    boundaries = None
    if cfg["study_area"].get("boundaries_file"):
        import geopandas as gpd
        boundaries = gpd.read_file(resolve(cfg["study_area"]["boundaries_file"], cfg_dir)).to_crs("EPSG:4326")
    thr_key = "max_tss (max sensitivity + specificity)"
    thr = thresholds[thr_key]
    name = cfg["species"]["name"]
    current_full = modeling.predict_grid(model, layers, variables)
    if train_bbox != bbox:
        # range-wide map from the training extent, then everything below is for the study area only
        env.write_raster(out / "rasters" / "suitability_training_range.tif", current_full, template)
        plots.suitability_map(out / "figures" / "suitability_range.png", current_full, template, pres,
                              f"{name} – habitat suitability, training range (current)", boundaries)
    train_template = template
    win, template = env.subgrid(train_template, bbox)
    current = current_full[win]
    area = cell_area_km2(template)
    in_bbox = pres["lon"].between(bbox[0], bbox[2]) & pres["lat"].between(bbox[1], bbox[3])
    pres_map = pres[in_bbox]
    binary = np.where(np.isnan(current), np.nan, (current >= thr).astype("float32"))
    suitable_km2 = float(np.nansum(area * (binary == 1)))
    env.write_raster(out / "rasters" / "suitability_current.tif", current, template)
    env.write_raster(out / "rasters" / "binary_current_maxTSS.tif", binary, template)
    p10 = thresholds["10th_percentile_training_presence"]
    env.write_raster(out / "rasters" / "binary_current_p10.tif",
                     np.where(np.isnan(current), np.nan, (current >= p10).astype("float32")), template)
    plots.occurrence_map(out / "figures" / "occurrences.png", cleaned, pres, train_template,
                         layers[all_vars[0]], bg)
    plots.suitability_map(out / "figures" / "suitability_current.png", current, template, pres_map,
                          f"{name} – habitat suitability (current)", boundaries)
    plots.binary_map(out / "figures" / "binary_current.png", binary, template, pres_map,
                     f"{name} – suitable habitat (max-TSS threshold)", thr, boundaries)
    log.info("    suitable area: %.0f km²", suitable_km2)

    future = []

    def save_projection(label, tag, fut, mess, info, agreement=None, n_models=None):
        """Write rasters/figures for one projection and record its summary row."""
        fut_bin = np.where(np.isnan(fut), np.nan, (fut >= thr).astype("float32"))
        env.write_raster(out / "rasters" / f"suitability_{tag}.tif", fut, template)
        env.write_raster(out / "rasters" / f"binary_maxTSS_{tag}.tif", fut_bin, template)
        env.write_raster(out / "rasters" / f"mess_{tag}.tif", mess, template)
        plots.suitability_map(out / "figures" / f"suitability_{tag}.png", fut, template, None,
                              f"{name} – suitability {label}", boundaries)
        plots.change_map(out / "figures" / f"change_{tag}.png", fut - current, template,
                         f"Change in suitability, {label}")
        plots.change_map(out / "figures" / f"mess_{tag}.png", np.clip(mess, -100, 100), template,
                         f"MESS {label} (negative = novel climate)", label="MESS similarity")
        figures = ["suitability", "change", "mess"]
        if agreement is not None:
            env.write_raster(out / "rasters" / f"agreement_{tag}.tif", agreement, template)
            plots.agreement_map(out / "figures" / f"agreement_{tag}.png", agreement, n_models, template, pres_map,
                                f"Model agreement on suitable habitat, SSP{info['ssp']} {info['period']}")
            figures.append("agreement")
        fut_km2 = float(np.nansum(area * (fut_bin == 1)))
        valid = ~np.isnan(mess)
        future.append({
            "label": label, "tag": tag, "figures": figures, **info, "suitable_area_km2": fut_km2,
            "area_change_pct": 100 * (fut_km2 - suitable_km2) / suitable_km2 if suitable_km2 else float("nan"),
            "novel_climate_pct": float(100 * np.mean(mess[valid] < 0)) if valid.any() else float("nan"),
        })
        log.info("    suitable area: %.0f km² (%+.1f%%), novel climate on %.1f%% of cells",
                 fut_km2, future[-1]["area_change_pct"], future[-1]["novel_climate_pct"])

    if cfg.get("future", {}).get("enabled"):
        by_ssp_period: dict[tuple, list] = {}
        for sc in env.future_scenarios(cfg["future"]):
            label = f"{sc['gcm']} SSP{sc['ssp']} {sc['period']}"
            log.info("    future projection: %s", label)
            fut_layers = env.load_future_layers(cfg, cache, template, variables, sc)
            fut = modeling.predict_grid(model, fut_layers, variables)
            mess = modeling.mess(x_bg, fut_layers, variables)
            save_projection(label, f"{sc['gcm']}_ssp{sc['ssp']}_{sc['period']}", fut, mess, sc)
            by_ssp_period.setdefault((sc["ssp"], sc["period"]), []).append((fut, mess))

        # Ensemble mean across climate models: mean suitability, most pessimistic (minimum) MESS,
        # and the number of models that predict suitable habitat in each cell.
        if cfg["future"].get("ensemble", True):
            for (ssp, period), runs in by_ssp_period.items():
                if len(runs) < 2:
                    continue
                label = f"Ensemble mean SSP{ssp} {period}"
                log.info("    future projection: %s", label)
                futs = np.stack([r[0] for r in runs])
                messes = np.stack([r[1] for r in runs])
                any_nan = np.isnan(futs).any(axis=0)
                ens = np.where(any_nan, np.nan, futs.mean(axis=0))
                ens_mess = np.where(np.isnan(messes).any(axis=0), np.nan, messes.min(axis=0))
                agreement = np.where(any_nan, np.nan, (futs >= thr).sum(axis=0).astype("float32"))
                save_projection(label, f"ensemble_ssp{ssp}_{period}", ens, ens_mess,
                                {"gcm": "ensemble mean", "ssp": ssp, "period": period},
                                agreement=agreement, n_models=len(runs))

        pd.DataFrame(future).drop(columns=["tag", "figures"]).to_csv(out / "tables" / "future_scenarios.csv", index=False)

    # 8. Report ---------------------------------------------------------------------------
    log.info("STEP 8/8  Report")
    summary = {
        "species": name, "occurrence_source": occ_source, "cleaning_steps": steps,
        "n_presences": len(pres), "n_background": len(bg), "variables": variables, "removed_variables": removed,
        "selected_model": {"features": best["features"], "regularization_multiplier": float(best["rm"])},
        "metrics": metrics, "thresholds": thresholds, "suitable_area_km2": suitable_km2, "future": future,
        "variable_importance_pct": importance["importance_pct"].round(2).to_dict(),
    }
    (out / "tables" / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    path = report.build(out, {
        "cfg": cfg, "best": best, "metrics": metrics, "thresholds": thresholds, "n_presences": len(pres),
        "n_background": len(bg), "cleaning_steps": steps, "variables": variables, "removed": removed,
        "tuning": tuning, "importance": importance, "suitable_area_km2": suitable_km2, "future": future,
        "occ_source": occ_source, "train_bbox": train_bbox,
    })
    log.info("Done. Report: %s", path)
    return summary


if __name__ == "__main__":
    main()
