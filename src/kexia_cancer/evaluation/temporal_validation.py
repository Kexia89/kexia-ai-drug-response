from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, mean_squared_error


def logistic(time: np.ndarray, carrying_capacity: float, growth_rate: float, initial: float) -> np.ndarray:
    time = np.asarray(time, float)
    ratio = (carrying_capacity - initial) / max(initial, 1e-12)
    return carrying_capacity / (1 + ratio * np.exp(-growth_rate * time))


def validate_untreated_logistic(counts_path: Path, report_dir: Path) -> dict:
    counts = pd.read_csv(counts_path)
    untreated = counts[
        counts["doxorubicin_nM"].eq(0)
        & ~counts["all_conditions_zero_at_time"].astype(bool)
        & counts["cell_count"].gt(0)
    ].copy()
    rows, predictions = [], []
    for (cell, source), group in untreated.groupby(["cell_line", "source_file"]):
        points = group.groupby("elapsed_hours", as_index=False).agg(
            observed=("cell_count", "mean"), replicates=("cell_count", "size"), sd=("cell_count", "std")
        ).sort_values("elapsed_hours")
        if len(points) < 12:
            continue
        split_at = max(6, int(.7 * len(points)))
        train, test = points.iloc[:split_at], points.iloc[split_at:]
        if len(test) < 3:
            continue
        x0 = float(train["observed"].iloc[0])
        try:
            params, covariance = curve_fit(
                lambda t, k, r: logistic(t, k, r, x0),
                train["elapsed_hours"], train["observed"],
                p0=[max(train["observed"].max() * 1.2, x0 + 1), .02],
                bounds=([x0, 0], [max(points["observed"].max() * 100, x0 * 2), 1]), maxfev=30_000,
            )
        except (RuntimeError, ValueError, FloatingPointError):
            continue
        predicted = logistic(test["elapsed_hours"].to_numpy(), params[0], params[1], x0)
        rmse = float(np.sqrt(mean_squared_error(test["observed"], predicted)))
        rows.append({
            "cell_line": cell, "source_file": source, "train_time_max_h": float(train["elapsed_hours"].max()),
            "test_time_min_h": float(test["elapsed_hours"].min()), "test_time_max_h": float(test["elapsed_hours"].max()),
            "train_points": len(train), "test_points": len(test), "carrying_capacity": params[0],
            "growth_rate_per_hour": params[1], "initial_count": x0,
            "covariance_condition": float(np.linalg.cond(covariance)),
            "test_mae_cells": float(mean_absolute_error(test["observed"], predicted)),
            "test_rmse_cells": rmse, "test_nrmse_by_observed_range": rmse / max(float(test["observed"].max() - test["observed"].min()), 1),
        })
        for (_, point), value in zip(test.iterrows(), predicted):
            predictions.append({"cell_line": cell, "source_file": source, "elapsed_hours": point["elapsed_hours"],
                                "observed_cell_count": point["observed"], "predicted_cell_count": value})
    results = pd.DataFrame(rows)
    pd.DataFrame(predictions).to_csv(report_dir / "phase2_temporal_logistic_predictions.csv", index=False)
    results.to_csv(report_dir / "phase2_temporal_logistic_validation.csv", index=False)
    report = {
        "status": "Partial validation of untreated logistic growth only",
        "experiments": len(results), "cell_lines": sorted(results["cell_line"].unique()) if len(results) else [],
        "median_test_nrmse": float(results["test_nrmse_by_observed_range"].median()) if len(results) else None,
        "design": "Chronological 70/30 split; late times are never used for fitting.",
        "excluded": "Whole-plate all-zero acquisition gaps and nonpositive counts are excluded from fitting but retained in the processed data.",
        "drug_dynamics_status": (
            "Not validated. The source uses pulsed doxorubicin, serial dosing and delayed death/regrowth; "
            "Kexia's current constant-concentration Hill kill-rate ODE does not encode those interventions."
        ),
    }
    (report_dir / "phase2_temporal_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

