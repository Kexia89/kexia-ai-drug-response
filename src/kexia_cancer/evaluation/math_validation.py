from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, mean_squared_error

from kexia_cancer.data.raw_response import normalize_gdsc2_raw_wells
from kexia_cancer.simulation.calibration import viability_hill


def _hill_zero_bottom(x: np.ndarray, ec50: float, hill: float) -> np.ndarray:
    return 1 / (1 + np.power(np.maximum(x, 0) / ec50, hill))


def _fit_predict(name: str, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, dict]:
    max_x = float(np.max(train_x))
    positive = train_x[train_x > 0]
    midpoint = float(np.median(positive))
    if name == "hill_3_parameter":
        params, covariance = curve_fit(
            viability_hill, train_x, train_y, p0=[max(0, min(.8, float(np.min(train_y)))), midpoint, 1],
            bounds=([0, 1e-12, .05], [1.2, max(1e6, max_x * 1e4), 8]), maxfev=20_000,
        )
        return viability_hill(test_x, *params), {
            "parameters": params.tolist(),
            "covariance_condition": float(np.linalg.cond(covariance)),
        }
    if name == "hill_zero_bottom":
        params, covariance = curve_fit(
            _hill_zero_bottom, train_x, train_y, p0=[midpoint, 1],
            bounds=([1e-12, .05], [max(1e6, max_x * 1e4), 8]), maxfev=20_000,
        )
        return _hill_zero_bottom(test_x, *params), {
            "parameters": params.tolist(),
            "covariance_condition": float(np.linalg.cond(covariance)),
        }
    if name == "log_linear":
        coeff = np.polyfit(np.log10(np.maximum(train_x, 1e-12)), train_y, 1)
        return np.polyval(coeff, np.log10(np.maximum(test_x, 1e-12))), {"parameters": coeff.tolist()}
    raise ValueError(name)


def validate_endpoint_models(
    raw_path: Path, processed_dir: Path, report_dir: Path, max_pairs: int = 250, seed: int = 42
) -> dict:
    curves, qc = normalize_gdsc2_raw_wells(raw_path)
    curves.to_csv(processed_dir / "breast_gdsc2_normalized_dose_response.csv.gz", index=False, compression="gzip")
    qc.to_csv(report_dir / "raw_normalization_qc.csv", index=False)
    counts = curves.groupby(["SANGER_MODEL_ID", "DRUG_ID"])["CONC"].nunique()
    eligible = counts[counts >= 6].reset_index()[["SANGER_MODEL_ID", "DRUG_ID"]]
    eligible = eligible.sample(n=min(max_pairs, len(eligible)), random_state=seed)
    rows = []
    models = ["hill_3_parameter", "hill_zero_bottom", "log_linear"]
    for pair in eligible.itertuples(index=False):
        group = curves[(curves["SANGER_MODEL_ID"] == pair.SANGER_MODEL_ID) & (curves["DRUG_ID"] == pair.DRUG_ID)]
        # Aggregate any DURATION strata without hiding them in the exported curve table.
        points = group.groupby("CONC", as_index=False).agg(viability_mean=("viability_mean", "mean"), replicates=("replicates", "sum")).sort_values("CONC")
        x, y = points["CONC"].to_numpy(float), points["viability_mean"].to_numpy(float)
        # Deterministic interleaved concentration holdout; extrema remain in training.
        test_mask = np.zeros(len(x), dtype=bool)
        test_mask[2:-1:3] = True
        if test_mask.sum() == 0 or (~test_mask).sum() < 4:
            continue
        for model_name in models:
            try:
                pred, metadata = _fit_predict(model_name, x[~test_mask], y[~test_mask], x[test_mask])
            except (RuntimeError, ValueError, FloatingPointError):
                continue
            for concentration, observed, predicted in zip(x[test_mask], y[test_mask], pred):
                rows.append(
                    {
                        "SANGER_MODEL_ID": pair.SANGER_MODEL_ID,
                        "DRUG_ID": int(pair.DRUG_ID),
                        "model": model_name,
                        "held_out_concentration_uM": concentration,
                        "observed_viability": observed,
                        "predicted_viability": predicted,
                        "absolute_error": abs(observed - predicted),
                        "squared_error": (observed - predicted) ** 2,
                        "fit_metadata": json.dumps(metadata),
                    }
                )
    predictions = pd.DataFrame(rows)
    predictions.to_csv(report_dir / "endpoint_model_holdout_predictions.csv.gz", index=False, compression="gzip")
    metrics = {}
    rng = np.random.default_rng(seed)
    for name, group in predictions.groupby("model"):
        y, p = group["observed_viability"].to_numpy(), group["predicted_viability"].to_numpy()
        bootstrap_mae, bootstrap_rmse = [], []
        for _ in range(500):
            idx = rng.integers(0, len(y), len(y))
            bootstrap_mae.append(mean_absolute_error(y[idx], p[idx]))
            bootstrap_rmse.append(np.sqrt(mean_squared_error(y[idx], p[idx])))
        metrics[name] = {
            "held_out_points": len(group),
            "pairs": int(group[["SANGER_MODEL_ID", "DRUG_ID"]].drop_duplicates().shape[0]),
            "mae": float(mean_absolute_error(y, p)),
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "mae_95_ci": [float(np.quantile(bootstrap_mae, .025)), float(np.quantile(bootstrap_mae, .975))],
            "rmse_95_ci": [float(np.quantile(bootstrap_rmse, .025)), float(np.quantile(bootstrap_rmse, .975))],
        }
    report = {
        "experiment_1": {
            "question": "Can endpoint concentration-response models predict held-out concentrations?",
            "design": "Interleaved concentration holdout within each real cell-compound curve; fit uses remaining concentrations.",
            "metrics": metrics,
        },
        "experiment_2": {
            "question": "Which mathematical endpoint model predicts best?",
            "models": models,
            "same_holdout_for_all_models": True,
        },
        "experiment_3": {
            "status": "Implemented by simulation.sensitivity_ensemble; dynamic parameters remain user-defined without temporal data."
        },
        "experiment_4": {
            "status": "Not performed",
            "reason": "The extracted public breast raw data contains one endpoint duration only; independent temporal observations are unavailable.",
        },
        "raw_qc": qc.iloc[0].to_dict(),
        "interpretation": "These tests validate endpoint viability-curve interpolation, not a tumour-population ODE or a death-rate EC50.",
    }
    (report_dir / "mathematical_validation.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report

