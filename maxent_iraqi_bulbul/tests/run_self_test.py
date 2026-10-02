"""End-to-end self-test on synthetic data: the model must recover the known virtual-species niche.

    python tests/run_self_test.py
"""
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import run_pipeline  # noqa: E402
from make_synthetic_data import main as make_data  # noqa: E402

make_data(HERE / "synthetic")
summary = run_pipeline.main(["--config", str(HERE / "config_synthetic.yaml")])

with rasterio.open(HERE / "synthetic" / "true_suitability.tif") as t, \
        rasterio.open(HERE / "synthetic" / "outputs" / "rasters" / "suitability_current.tif") as p:
    truth, pred = t.read(1), p.read(1, masked=True).filled(np.nan)
ok = ~np.isnan(pred)
rho = spearmanr(truth[ok], pred[ok]).statistic
print(f"\nSpearman correlation, predicted vs TRUE suitability: {rho:.3f}")
print(f"Spatial-CV test AUC: {summary['metrics']['cv_test_auc']:.3f}")
assert rho > 0.7, "model failed to recover the known niche"
assert summary["metrics"]["cv_test_auc"] > 0.7
print("SELF-TEST PASSED")
