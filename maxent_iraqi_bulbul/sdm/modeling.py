"""Background sampling, MaxEnt tuning with spatial cross-validation, final fit and evaluation."""
from __future__ import annotations

import itertools
import logging
import warnings

import elapid as ela
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import BallTree

log = logging.getLogger(__name__)
EARTH_RADIUS_KM = 6371.0


def cell_centers(template: dict, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.nonzero(mask)
    t = template["transform"]
    lon = t.c + (cols + 0.5) * t.a
    lat = t.f + (rows + 0.5) * t.e
    return rows, cols, lon, lat


def sample_background(template, valid_mask, pres_lon, pres_lat, n_points, buffer_km, seed):
    """Random background cells among valid cells within `buffer_km` of any presence."""
    rows, cols, lon, lat = cell_centers(template, valid_mask)
    if buffer_km:
        tree = BallTree(np.radians(np.c_[pres_lat, pres_lon]), metric="haversine")
        dist, _ = tree.query(np.radians(np.c_[lat, lon]), k=1)
        inside = dist[:, 0] * EARTH_RADIUS_KM <= buffer_km
        lon, lat = lon[inside], lat[inside]
    rng = np.random.default_rng(seed)
    n = min(n_points, len(lon))
    if n < n_points:
        log.warning("Only %d candidate background cells available (requested %d)", n, n_points)
    idx = rng.choice(len(lon), size=n, replace=False)
    return pd.DataFrame({"lon": lon[idx], "lat": lat[idx]})


def block_folds(pres: pd.DataFrame, bg: pd.DataFrame):
    """ENMeval-style 'block' partitioning: 4 spatial blocks with equal presence counts.

    Presences are split at the median longitude, each half at its median latitude; the
    background is assigned to blocks using the same boundaries.
    """
    lon_cut = pres["lon"].median()

    def assign(df, lat_w, lat_e):
        west = df["lon"] <= lon_cut
        return np.where(west, np.where(df["lat"] <= lat_w, 0, 1), np.where(df["lat"] <= lat_e, 2, 3))

    lat_w = pres.loc[pres["lon"] <= lon_cut, "lat"].median()
    lat_e = pres.loc[pres["lon"] > lon_cut, "lat"].median()
    return assign(pres, lat_w, lat_e), assign(bg, lat_w, lat_e)


def random_folds(n_pres, n_bg, k, seed):
    y = np.r_[np.ones(n_pres), np.zeros(n_bg)]
    folds = np.zeros(len(y), dtype=int)
    for i, (_, test) in enumerate(StratifiedKFold(k, shuffle=True, random_state=seed).split(y, y)):
        folds[test] = i
    return folds[:n_pres], folds[n_pres:]


def make_model(features: str, rm: float, seed: int) -> ela.MaxentModel:
    return ela.MaxentModel(feature_types=features.split(","), beta_multiplier=rm,
                           transform="cloglog", random_state=seed, n_cpus=1)


def fit(model, x_pres, x_bg):
    x = pd.concat([x_pres, x_bg], ignore_index=True)
    y = np.r_[np.ones(len(x_pres)), np.zeros(len(x_bg))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x, y)
    return model


def predict(model, x: pd.DataFrame) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(model.predict(x)).ravel()


def auc(p_pres, p_bg):
    y = np.r_[np.ones(len(p_pres)), np.zeros(len(p_bg))]
    return roc_auc_score(y, np.r_[p_pres, p_bg])


def tune(x_pres, x_bg, pres_folds, bg_folds, feature_classes, rms, seed) -> pd.DataFrame:
    """Evaluate every feature-class x regularization-multiplier combination with cross-validation."""
    results = []
    for features, rm in itertools.product(feature_classes, rms):
        fold_stats = []
        for k in np.unique(pres_folds):
            tr_p, te_p = pres_folds != k, pres_folds == k
            tr_b, te_b = bg_folds != k, bg_folds == k
            if te_p.sum() == 0 or te_b.sum() == 0:
                continue
            m = fit(make_model(features, rm, seed), x_pres[tr_p], x_bg[tr_b])
            p_trp, p_trb = predict(m, x_pres[tr_p]), predict(m, x_bg[tr_b])
            p_tep, p_teb = predict(m, x_pres[te_p]), predict(m, x_bg[te_b])
            train_auc, test_auc = auc(p_trp, p_trb), auc(p_tep, p_teb)
            or10 = np.mean(p_tep < np.percentile(p_trp, 10))
            fold_stats.append((train_auc, test_auc, train_auc - test_auc, or10))
        s = np.array(fold_stats)
        results.append({
            "features": features, "rm": rm,
            "train_auc": s[:, 0].mean(), "test_auc": s[:, 1].mean(), "test_auc_sd": s[:, 1].std(),
            "auc_diff": s[:, 2].mean(), "or10": s[:, 3].mean(),
        })
        log.info("  %-32s rm=%-4s test AUC=%.3f  AUC diff=%.3f  OR10=%.3f",
                 features, rm, results[-1]["test_auc"], results[-1]["auc_diff"], results[-1]["or10"])
    return pd.DataFrame(results)


def choose_best(tuning: pd.DataFrame, metric: str) -> pd.Series:
    if metric == "auc_diff":
        ranked = tuning.sort_values(["auc_diff", "test_auc"], ascending=[True, False])
    else:
        ranked = tuning.sort_values(["test_auc", "auc_diff"], ascending=[False, True])
    return ranked.iloc[0]


def boyce_index(p_pres, p_bg, n_bins=101, window=0.1):
    """Continuous Boyce index (Hirzel et al. 2006): Spearman r of predicted/expected ratio vs suitability."""
    from scipy.stats import spearmanr
    lo, hi = min(p_bg.min(), p_pres.min()), max(p_bg.max(), p_pres.max())
    width = (hi - lo) * window
    centers = np.linspace(lo + width / 2, hi - width / 2, n_bins)
    ratios = []
    for c in centers:
        pf = np.mean((p_pres >= c - width / 2) & (p_pres <= c + width / 2))
        ef = np.mean((p_bg >= c - width / 2) & (p_bg <= c + width / 2))
        ratios.append(pf / ef if ef > 0 else np.nan)
    ratios = np.array(ratios)
    ok = ~np.isnan(ratios)
    if ok.sum() < 3:
        return np.nan
    return float(spearmanr(centers[ok], ratios[ok]).statistic)


def evaluate(p_pres, p_bg):
    """Discrimination metrics and thresholds for the final model."""
    y = np.r_[np.ones(len(p_pres)), np.zeros(len(p_bg))]
    scores = np.r_[p_pres, p_bg]
    fpr, tpr, thr = roc_curve(y, scores)
    tss = tpr - fpr
    best = np.argmax(tss)
    thresholds = {
        "max_tss (max sensitivity + specificity)": float(min(thr[best], 1.0)),
        "10th_percentile_training_presence": float(np.percentile(p_pres, 10)),
        "minimum_training_presence": float(p_pres.min()),
    }
    metrics = {
        "training_auc": float(roc_auc_score(y, scores)),
        "max_tss": float(tss[best]),
        "continuous_boyce_index": boyce_index(p_pres, p_bg),
    }
    return metrics, thresholds, (fpr, tpr)


def permutation_importance(model, x_pres, x_bg, n_repeats=10, seed=42) -> pd.DataFrame:
    """Drop in training AUC when each variable is randomly permuted (normalised to percent)."""
    rng = np.random.default_rng(seed)
    x = pd.concat([x_pres, x_bg], ignore_index=True)
    y = np.r_[np.ones(len(x_pres)), np.zeros(len(x_bg))]
    base = roc_auc_score(y, predict(model, x))
    drops = {}
    for col in x.columns:
        d = []
        for _ in range(n_repeats):
            xp = x.copy()
            xp[col] = rng.permutation(xp[col].to_numpy())
            d.append(base - roc_auc_score(y, predict(model, xp)))
        drops[col] = (np.mean(d), np.std(d))
    out = pd.DataFrame(drops, index=["auc_drop", "auc_drop_sd"]).T
    total = out["auc_drop"].clip(lower=0).sum()
    out["importance_pct"] = 100 * out["auc_drop"].clip(lower=0) / total if total > 0 else 0.0
    return out.sort_values("importance_pct", ascending=False)


def response_curves(model, x_bg: pd.DataFrame, n=100) -> dict[str, pd.DataFrame]:
    """Marginal response: vary one variable across its range, hold the others at their median."""
    med = x_bg.median()
    curves = {}
    for col in x_bg.columns:
        grid = np.linspace(x_bg[col].quantile(0.005), x_bg[col].quantile(0.995), n)
        x = pd.DataFrame([med.to_numpy()] * n, columns=x_bg.columns)
        x[col] = grid
        curves[col] = pd.DataFrame({"value": grid, "suitability": predict(model, x)})
    return curves


def predict_grid(model, layers: dict[str, np.ndarray], variables: list[str]) -> np.ndarray:
    stack = np.stack([layers[v] for v in variables], axis=-1)
    valid = ~np.isnan(stack).any(axis=-1)
    out = np.full(valid.shape, np.nan, dtype="float32")
    x = pd.DataFrame(stack[valid], columns=variables)
    preds = []
    for start in range(0, len(x), 200_000):
        preds.append(predict(model, x.iloc[start:start + 200_000]))
    out[valid] = np.concatenate(preds) if preds else []
    return out


def mess(reference: pd.DataFrame, layers: dict[str, np.ndarray], variables: list[str]) -> np.ndarray:
    """Multivariate Environmental Similarity Surface (Elith et al. 2010). Negative = extrapolation."""
    shape = layers[variables[0]].shape
    sims = []
    for v in variables:
        ref = np.sort(reference[v].to_numpy())
        lo, hi = ref[0], ref[-1]
        rng_ = hi - lo if hi > lo else 1.0
        p = layers[v]
        f = np.searchsorted(ref, p, side="right") / len(ref) * 100
        s = np.where(f == 0, (p - lo) / rng_ * 100,
            np.where(f <= 50, 2 * f, np.where(f < 100, 2 * (100 - f), (hi - p) / rng_ * 100)))
        sims.append(np.where(np.isnan(p), np.nan, s))
    return np.min(np.stack(sims), axis=0).reshape(shape)
