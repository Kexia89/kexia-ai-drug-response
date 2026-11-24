from __future__ import annotations

import math

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) < 2:
        return {k: float("nan") for k in ["mae", "rmse", "r2", "pearson", "spearman"]}
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson": float(pearsonr(y_true, y_pred).statistic),
        "spearman": float(spearmanr(y_true, y_pred).statistic),
    }


def bootstrap_metric_intervals(
    y_true: np.ndarray, y_pred: np.ndarray, seed: int = 42, replicates: int = 500
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    samples: dict[str, list[float]] = {k: [] for k in regression_metrics(y_true, y_pred)}
    for _ in range(replicates):
        idx = rng.integers(0, len(y_true), len(y_true))
        values = regression_metrics(y_true[idx], y_pred[idx])
        for key, value in values.items():
            if np.isfinite(value):
                samples[key].append(value)
    return {
        key: {
            "low_95": float(np.quantile(values, 0.025)) if values else float("nan"),
            "high_95": float(np.quantile(values, 0.975)) if values else float("nan"),
        }
        for key, values in samples.items()
    }

