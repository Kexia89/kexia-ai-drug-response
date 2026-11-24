from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from kexia_cancer.evaluation.metrics import bootstrap_metric_intervals, regression_metrics
from kexia_cancer.models.splits import make_splits, validate_split

IDENTIFIERS = {"SANGER_MODEL_ID", "CELL_LINE_NAME", "DRUG_ID", "DRUG_NAME", "LN_IC50", "AUC", "IC50_uM"}
CATEGORICAL = ["PUTATIVE_TARGET", "PATHWAY_NAME", "SCREENING_SITE"]


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_candidates = [
        "MIN_CONC", "MAX_CONC", "ploidy_snp6", "ploidy_wes", "ploidy_wgs",
        "mutational_burden", "doubling_time_hours", "growth_rate_per_hour",
    ] + [c for c in df if c.startswith("MUT_") or c.startswith("CN_")]
    numeric = [c for c in numeric_candidates if c in df and df[c].notna().any()]
    categorical = [c for c in CATEGORICAL if c in df]
    return numeric, categorical


def make_pipeline(df: pd.DataFrame, kind: str, seed: int = 42) -> Pipeline:
    numeric, categorical = feature_columns(df)
    preprocess = ColumnTransformer(
        [
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), numeric),
            ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    if kind == "ridge":
        estimator = Ridge(alpha=10.0)
    elif kind == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, max_features=0.7, n_jobs=-1, random_state=seed
        )
    elif kind == "gradient_boosting":
        estimator = HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=300, l2_regularization=1.0, early_stopping=True, random_state=seed
        )
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


@dataclass
class ModelArtifact:
    pipeline: Pipeline
    target: str
    interval_half_width: float
    feature_columns: list[str]
    metadata: dict

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        prediction = self.pipeline.predict(rows)
        return pd.DataFrame(
            {
                "prediction": prediction,
                "lower_95": prediction - self.interval_half_width,
                "upper_95": prediction + self.interval_half_width,
                "uncertainty_method": "split-conformal absolute residual interval (marginal 95%)",
            },
            index=rows.index,
        )


def train_and_evaluate(
    df: pd.DataFrame, model_dir: Path, report_dir: Path, target: str = "LN_IC50", seed: int = 42
) -> dict:
    df = df[df[target].notna()].reset_index(drop=True)
    splits = make_splits(df, seed=seed)
    selection_split = splits["B_unseen_cells"]
    train_indices = selection_split.train.copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(train_indices)
    calibration_size = max(100, int(0.2 * len(train_indices)))
    calibration_idx = train_indices[:calibration_size]
    fit_idx = train_indices[calibration_size:]

    comparison = {}
    fitted = {}
    for kind in ["ridge", "random_forest", "gradient_boosting"]:
        pipeline = make_pipeline(df, kind, seed)
        pipeline.fit(df.iloc[fit_idx], df.iloc[fit_idx][target])
        pred = pipeline.predict(df.iloc[calibration_idx])
        comparison[kind] = regression_metrics(df.iloc[calibration_idx][target].to_numpy(), pred)
        fitted[kind] = pipeline
    winner = min(comparison, key=lambda k: comparison[k]["mae"])
    base_pipeline = fitted[winner]
    calibration_pred = base_pipeline.predict(df.iloc[calibration_idx])
    residuals = np.abs(df.iloc[calibration_idx][target].to_numpy() - calibration_pred)
    conformal_q = float(np.quantile(residuals, 0.95, method="higher"))

    results = {}
    predictions = []
    for name, split in splits.items():
        # Each protocol refits all preprocessing on only that protocol's training rows.
        pipeline = make_pipeline(df, winner, seed)
        protocol_train = split.train
        # Reserve calibration rows inside protocol train for a protocol-specific interval.
        local = protocol_train.copy()
        rng.shuffle(local)
        n_cal = max(50, int(0.15 * len(local)))
        local_cal, local_fit = local[:n_cal], local[n_cal:]
        pipeline.fit(df.iloc[local_fit], df.iloc[local_fit][target])
        local_q = float(
            np.quantile(
                np.abs(df.iloc[local_cal][target].to_numpy() - pipeline.predict(df.iloc[local_cal])),
                0.95,
                method="higher",
            )
        )
        y_test = df.iloc[split.test][target].to_numpy()
        y_pred = pipeline.predict(df.iloc[split.test])
        metrics = regression_metrics(y_test, y_pred)
        results[name] = {
            "split": validate_split(df, split),
            "metrics": metrics,
            "bootstrap_95_ci": bootstrap_metric_intervals(y_test, y_pred, seed=seed, replicates=300),
            "prediction_interval_half_width": local_q,
            "prediction_interval_empirical_coverage": float(np.mean(np.abs(y_test - y_pred) <= local_q)),
        }
        pred_frame = df.iloc[split.test][["SANGER_MODEL_ID", "CELL_LINE_NAME", "DRUG_ID", "DRUG_NAME", target]].copy()
        pred_frame["prediction"] = y_pred
        pred_frame["lower_95"] = y_pred - local_q
        pred_frame["upper_95"] = y_pred + local_q
        pred_frame["protocol"] = name
        predictions.append(pred_frame)

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact = ModelArtifact(
        pipeline=base_pipeline,
        target=target,
        interval_half_width=conformal_q,
        feature_columns=sum((list(x) for x in feature_columns(df)), []),
        metadata={
            "model_kind": winner,
            "seed": seed,
            "model_selection": "Lowest MAE on a calibration subset of cell-held-out training data",
            "target_definition": "Natural logarithm of fitted IC50 in micromolar (GDSC2)",
            "intended_use": "Preclinical computational hypothesis generation only",
        },
    )
    joblib.dump(artifact, model_dir / "gdsc2_breast_ln_ic50.joblib")
    pd.concat(predictions, ignore_index=True).to_csv(report_dir / "model_predictions.csv.gz", index=False, compression="gzip")
    report = {
        "selected_model": winner,
        "candidate_calibration_metrics": comparison,
        "target": target,
        "features": {"numeric": feature_columns(df)[0], "categorical": feature_columns(df)[1]},
        "protocols": results,
        "limitations": [
            "No RNA-seq expression matrix in the first run.",
            "Chemical structure descriptors are not used: the official annotation supplies InChIKey, not a reconstructable structure string.",
            "Intervals are marginal split-conformal intervals and do not guarantee conditional coverage for a specific cell or compound.",
            "GDSC2 measures endpoint viability at 72 h; predictions are not death rates or clinical outcomes.",
        ],
    }
    (report_dir / "model_evaluation.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report

