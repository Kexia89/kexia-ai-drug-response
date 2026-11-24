from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

from kexia_cancer.data.raw_response import SINGLE_AGENT_TAG
from kexia_cancer.simulation.calibration import viability_hill


def audit_plate_normalization(raw_path: Path, report_dir: Path) -> dict:
    columns = ["BARCODE", "SCAN_ID", "SANGER_MODEL_ID", "TAG", "DRUG_ID", "CONC", "INTENSITY"]
    raw = pd.read_csv(raw_path, usecols=columns, low_memory=False)
    raw["INTENSITY"] = pd.to_numeric(raw["INTENSITY"], errors="coerce")
    keys = ["BARCODE", "SCAN_ID"]
    control_rows = []
    for tag, prefix in [("B", "blank"), ("NC-1", "vehicle")]:
        stats = raw[raw["TAG"].eq(tag)].groupby(keys)["INTENSITY"].agg(["count", "median", "mean", "std"])
        stats.columns = [f"{prefix}_{column}" for column in stats.columns]
        control_rows.append(stats)
    controls = pd.concat(control_rows, axis=1).reset_index()
    controls["dynamic_range"] = controls["vehicle_median"] - controls["blank_median"]
    controls["vehicle_cv"] = controls["vehicle_std"] / controls["vehicle_mean"].replace(0, np.nan)
    controls["blank_cv"] = controls["blank_std"] / controls["blank_mean"].replace(0, np.nan)
    treatments = raw[
        raw["TAG"].astype(str).map(lambda value: bool(SINGLE_AGENT_TAG.match(value)))
        & raw["DRUG_ID"].notna() & raw["CONC"].notna()
    ].merge(controls[keys + ["blank_median", "vehicle_median", "dynamic_range"]], on=keys, how="left")
    treatments = treatments[treatments["dynamic_range"] > 0].copy()
    treatments["viability"] = (treatments["INTENSITY"] - treatments["blank_median"]) / treatments["dynamic_range"]
    per_plate = treatments.groupby(keys).agg(
        treatment_wells=("viability", "size"),
        fraction_above_one=("viability", lambda value: float((value > 1).mean())),
        fraction_below_zero=("viability", lambda value: float((value < 0).mean())),
        viability_median=("viability", "median"),
    ).reset_index().merge(controls, on=keys, how="left")
    per_plate.to_csv(report_dir / "phase2_plate_normalization_qc.csv", index=False)
    by_concentration = treatments.assign(log10_concentration=np.log10(treatments["CONC"].clip(lower=1e-12))).groupby(
        "log10_concentration", as_index=False
    ).agg(wells=("viability", "size"), fraction_above_one=("viability", lambda value: float((value > 1).mean())),
          median_viability=("viability", "median"))
    by_concentration.to_csv(report_dir / "phase2_normalization_by_concentration.csv", index=False)
    correlations = {}
    for column in ["dynamic_range", "vehicle_cv", "blank_cv"]:
        valid = per_plate[[column, "fraction_above_one"]].dropna()
        correlations[column] = float(spearmanr(valid[column], valid["fraction_above_one"]).statistic)
    summary = {
        "plates": len(per_plate), "wells": len(treatments),
        "fraction_above_one": float((treatments["viability"] > 1).mean()),
        "fraction_below_zero": float((treatments["viability"] < 0).mean()),
        "plates_fraction_above_one_median": float(per_plate["fraction_above_one"].median()),
        "plates_fraction_above_one_p95": float(per_plate["fraction_above_one"].quantile(.95)),
        "dynamic_range_median": float(per_plate["dynamic_range"].median()),
        "vehicle_cv_median": float(per_plate["vehicle_cv"].median()),
        "spearman_fraction_above_one": correlations,
        "interpretation": (
            "Values above 1 are retained. They indicate signal above the within-plate vehicle-control median, "
            "which can reflect control variability, measurement noise or biological variation; they are not automatically invalid."
        ),
    }
    return summary


def _fit_hill(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    midpoint = float(np.median(x[x > 0]))
    return curve_fit(
        viability_hill, x, y,
        p0=[max(0, min(.8, float(np.min(y)))), midpoint, 1.0],
        bounds=([0, 1e-12, .05], [1.2, max(1e6, float(np.max(x)) * 1e4), 8]),
        maxfev=30_000,
    )


def audit_hill_models(curves_path: Path, report_dir: Path, max_pairs: int = 120, seed: int = 42) -> dict:
    curves = pd.read_csv(curves_path, low_memory=False)
    grouped = list(curves.groupby(["SANGER_MODEL_ID", "DRUG_ID"], sort=True))
    eligible = [(key, group.groupby("CONC", as_index=False).agg(
        viability_mean=("viability_mean", "mean"), viability_sd=("viability_sd", "mean"),
        replicates=("replicates", "sum")
    ).sort_values("CONC")) for key, group in grouped if group["CONC"].nunique() >= 6]
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(eligible), min(max_pairs, len(eligible)), replace=False)
    fit_rows, extrapolation_rows, stability_rows = [], [], []
    for position in selected:
        (cell, drug), points = eligible[int(position)]
        x, y = points["CONC"].to_numpy(float), points["viability_mean"].to_numpy(float)
        try:
            params, covariance = _fit_hill(x, y)
            fitted = viability_hill(x, *params)
        except (RuntimeError, ValueError, FloatingPointError):
            continue
        fit_rows.append({
            "SANGER_MODEL_ID": cell, "DRUG_ID": drug, "bottom": params[0], "ec50_uM": params[1],
            "hill": params[2], "covariance_condition": float(np.linalg.cond(covariance)),
            "ec50_inside_tested_range": bool(x.min() <= params[1] <= x.max()),
            "min_concentration_uM": x.min(), "max_concentration_uM": x.max(),
            "mae_in_sample": float(np.mean(np.abs(y - fitted))),
            "residual_mean": float(np.mean(y - fitted)),
        })
        try:
            extrapolated_params, _ = _fit_hill(x[:-1], y[:-1])
            predicted = float(viability_hill(np.array([x[-1]]), *extrapolated_params)[0])
            extrapolation_rows.append({
                "SANGER_MODEL_ID": cell, "DRUG_ID": drug, "held_out_concentration_uM": x[-1],
                "observed_viability": y[-1], "predicted_viability": predicted,
                "absolute_error": abs(y[-1] - predicted),
            })
        except (RuntimeError, ValueError, FloatingPointError):
            pass
        leave_one_out = []
        for omitted in range(len(x)):
            if len(x) - 1 < 5:
                continue
            try:
                loo_params, _ = _fit_hill(np.delete(x, omitted), np.delete(y, omitted))
                leave_one_out.append(loo_params)
            except (RuntimeError, ValueError, FloatingPointError):
                continue
        if leave_one_out:
            loo = np.asarray(leave_one_out)
            stability_rows.append({
                "SANGER_MODEL_ID": cell, "DRUG_ID": drug, "successful_leave_one_out_fits": len(loo),
                "bottom_sd": float(np.std(loo[:, 0])), "log10_ec50_sd": float(np.std(np.log10(loo[:, 1]))),
                "hill_sd": float(np.std(loo[:, 2])),
            })
    fits = pd.DataFrame(fit_rows)
    extrapolation = pd.DataFrame(extrapolation_rows)
    stability = pd.DataFrame(stability_rows)
    fits.to_csv(report_dir / "phase2_hill_identifiability.csv", index=False)
    extrapolation.to_csv(report_dir / "phase2_hill_extrapolation.csv", index=False)
    stability.to_csv(report_dir / "phase2_hill_parameter_stability.csv", index=False)
    return {
        "curves_audited": len(fits),
        "fraction_ec50_inside_tested_range": float(fits["ec50_inside_tested_range"].mean()) if len(fits) else None,
        "covariance_condition_median": float(fits["covariance_condition"].replace([np.inf], np.nan).median()) if len(fits) else None,
        "extrapolation_pairs": len(extrapolation),
        "extrapolation_mae": float(extrapolation["absolute_error"].mean()) if len(extrapolation) else None,
        "leave_one_out_log10_ec50_sd_median": float(stability["log10_ec50_sd"].median()) if len(stability) else None,
        "dynamic_validation_status": "Not validated: these are endpoint concentration-response data, not independent temporal trajectories.",
    }


def run_math_audit(raw_path: Path, curves_path: Path, report_dir: Path) -> dict:
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "normalization": audit_plate_normalization(raw_path, report_dir),
        "hill_endpoint_models": audit_hill_models(curves_path, report_dir),
    }
    (report_dir / "phase2_mathematical_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

