from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from kexia_cancer.models.baselines import CATEGORICAL, feature_columns
from kexia_cancer.models.splits import make_splits


TARGET = "LN_IC50"
DESCRIPTOR_PREFIX = "desc_"


@dataclass(frozen=True)
class EvaluationSplit:
    name: str
    train: np.ndarray
    test: np.ndarray
    calibration_group: str


@dataclass
class Phase2CombinedArtifact:
    pipeline: Pipeline
    expanded_molecular: pd.DataFrame
    expression_scores: pd.DataFrame
    metadata: dict

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        enriched = rows.merge(self.expanded_molecular, on="SANGER_MODEL_ID", how="left", validate="many_to_one")
        enriched = enriched.merge(self.expression_scores, on="SANGER_MODEL_ID", how="left", validate="many_to_one")
        prediction = self.pipeline.predict(enriched)
        return pd.DataFrame({"prediction": prediction, "model_version": self.metadata["model_version"]}, index=rows.index)


def chemical_identity(frame: pd.DataFrame) -> pd.Series:
    """Conservative verified chemical identity; never merge on names when an identifier exists."""
    inchi = frame.get("INCHI_KEY", pd.Series(index=frame.index, dtype=object)).fillna("").astype(str).str.strip()
    chembl = frame.get("CHEMBL_ID", pd.Series(index=frame.index, dtype=object)).fillna("").astype(str).str.strip()
    drug = frame["DRUG_ID"].astype(str)
    valid_inchi = inchi.str.match(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
    valid_chembl = chembl.str.match(r"^CHEMBL\d+$")
    result = pd.Series("DRUG_ID:" + drug, index=frame.index)
    result.loc[valid_chembl] = "CHEMBL:" + chembl.loc[valid_chembl]
    result.loc[valid_inchi] = "INCHIKEY:" + inchi.loc[valid_inchi]
    return result


def make_independent_splits(frame: pd.DataFrame, seed: int = 42, fraction: float = 0.2) -> dict[str, EvaluationSplit]:
    original = make_splits(frame, seed=seed, fraction=fraction)
    identities = frame["CHEMICAL_IDENTITY"].to_numpy()
    unique_identities = np.array(sorted(pd.unique(identities)))
    rng = np.random.default_rng(seed + 1000)
    held_identities = set(rng.choice(unique_identities, max(1, int(fraction * len(unique_identities))), replace=False))
    compound_test = frame["CHEMICAL_IDENTITY"].isin(held_identities).to_numpy()
    cell_test = np.zeros(len(frame), dtype=bool)
    cell_test[original["B_unseen_cells"].test] = True
    indices = np.arange(len(frame))
    return {
        "A_known_entities_new_pairs": EvaluationSplit(
            "A_known_entities_new_pairs", original["A_known_entities_new_pairs"].train,
            original["A_known_entities_new_pairs"].test, "SANGER_MODEL_ID"
        ),
        "B_unseen_cells": EvaluationSplit("B_unseen_cells", indices[~cell_test], indices[cell_test], "SANGER_MODEL_ID"),
        "C_unseen_chemical_identities": EvaluationSplit(
            "C_unseen_chemical_identities", indices[~compound_test], indices[compound_test], "CHEMICAL_IDENTITY"
        ),
        "D_unseen_cells_and_chemical_identities": EvaluationSplit(
            "D_unseen_cells_and_chemical_identities", indices[(~cell_test) & (~compound_test)],
            indices[cell_test & compound_test], "SANGER_MODEL_ID"
        ),
    }


def regression_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    return {
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
        "r2": float(r2_score(y, prediction)),
        "pearson": float(pearsonr(y, prediction).statistic) if len(y) > 1 else float("nan"),
        "spearman": float(spearmanr(y, prediction).statistic) if len(y) > 1 else float("nan"),
    }


def group_bootstrap_ci(
    y: np.ndarray, prediction: np.ndarray, groups: np.ndarray, seed: int = 42, replicates: int = 100
) -> dict[str, dict[str, float]]:
    y, prediction, groups = np.asarray(y), np.asarray(prediction), np.asarray(groups)
    unique = pd.unique(groups)
    positions = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in regression_metrics(y, prediction)}
    for _ in range(replicates):
        selected = rng.choice(unique, len(unique), replace=True)
        idx = np.concatenate([positions[group] for group in selected])
        values = regression_metrics(y[idx], prediction[idx])
        for key, value in values.items():
            if np.isfinite(value):
                samples[key].append(value)
    return {
        key: {"low_95": float(np.quantile(values, 0.025)), "high_95": float(np.quantile(values, 0.975))}
        for key, values in samples.items() if values
    }


def _expression_scores(
    expression: pd.DataFrame, train_cells: set[str], all_cells: pd.Series, components: int = 10
) -> tuple[pd.DataFrame, dict]:
    exp = expression.set_index("SANGER_MODEL_ID")
    columns = list(exp.columns)
    train_ids = sorted(train_cells & set(exp.index))
    train = exp.loc[train_ids].replace([np.inf, -np.inf], np.nan)
    medians = train.median(axis=0)
    train = train.fillna(medians)
    variances = train.var(axis=0)
    selected = variances.nlargest(min(2000, int((variances > 0).sum()))).index
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train[selected])
    n_components = min(components, len(train_ids) - 1, len(selected))
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(train_scaled)
    available = exp.index.intersection(pd.unique(all_cells))
    transformed = pca.transform(scaler.transform(exp.loc[available, selected].fillna(medians[selected])))
    score_table = pd.DataFrame(transformed, index=available, columns=[f"EXPR_PC_{i+1}" for i in range(n_components)])
    score_frame = score_table.reindex(all_cells.to_numpy()).reset_index(drop=True)
    score_frame.index = all_cells.index
    return score_frame, {
        "training_cells_with_expression": len(train_ids), "selected_by_train_variance": len(selected),
        "components": n_components, "explained_variance_sum": float(pca.explained_variance_ratio_.sum()),
    }


def _pipeline(frame: pd.DataFrame, numeric: list[str], categorical: list[str], select_numeric: bool, seed: int) -> Pipeline:
    numeric_steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if select_numeric and len(numeric) > 128:
        numeric_steps.append(("select", SelectKBest(f_regression, k=128)))
    numeric_steps.append(("scale", StandardScaler()))
    preprocess = ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric),
            ("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical),
        ], remainder="drop", verbose_feature_names_out=True,
    )
    model = HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=120, l2_regularization=1.0,
        early_stopping=True, random_state=seed,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def _variant_frame(
    base: pd.DataFrame, variant: str, expanded: pd.DataFrame, expression: pd.DataFrame,
    chemical: pd.DataFrame, train_cells: set[str]
) -> tuple[pd.DataFrame, list[str], list[str], dict]:
    frame = base.copy()
    metadata: dict = {}
    numeric, categorical = feature_columns(frame)
    if variant.startswith("ablate_"):
        removed = variant.removeprefix("ablate_").split("+")
        numeric = [c for c in numeric if c not in removed]
        categorical = [c for c in categorical if c not in removed]
    elif variant in {"expanded_molecular", "combined"}:
        frame = frame.merge(expanded, on="SANGER_MODEL_ID", how="left", validate="many_to_one")
        numeric += [c for c in expanded.columns if c != "SANGER_MODEL_ID"]
    if variant in {"expression_pca", "combined"}:
        scores, expression_meta = _expression_scores(expression, train_cells, frame["SANGER_MODEL_ID"])
        for column in scores:
            frame[column] = scores[column]
        numeric += list(scores.columns)
        metadata["expression_transform"] = expression_meta
    if variant == "chemical_descriptors":
        descriptors = chemical.drop(columns=[c for c in chemical.columns if c.startswith("source_") or c in {
            "DRUG_NAME", "chembl_canonical_smiles", "canonical_smiles_rdkit", "murcko_scaffold", "inchi_key_verified"
        }], errors="ignore")
        frame = frame.merge(descriptors, on="DRUG_ID", how="inner", validate="many_to_one")
        numeric += [c for c in descriptors if c.startswith(DESCRIPTOR_PREFIX)]
    return frame, list(dict.fromkeys(numeric)), list(dict.fromkeys(categorical)), metadata


def evaluate_variant(
    base: pd.DataFrame, split: EvaluationSplit, variant: str, expanded: pd.DataFrame,
    expression: pd.DataFrame, chemical: pd.DataFrame, seed: int = 42
) -> tuple[dict, pd.DataFrame, Pipeline]:
    train_cells = set(base.iloc[split.train]["SANGER_MODEL_ID"])
    frame, numeric, categorical, metadata = _variant_frame(base, variant, expanded, expression, chemical, train_cells)
    # Inner joins (chemical structures) change row positions; reconstruct membership by stable index.
    train_mask = frame["ROW_ID"].isin(base.iloc[split.train]["ROW_ID"])
    test_mask = frame["ROW_ID"].isin(base.iloc[split.test]["ROW_ID"])
    train, test = frame.loc[train_mask], frame.loc[test_mask]
    pipe = _pipeline(frame, numeric, categorical, select_numeric=variant in {"expanded_molecular", "combined"}, seed=seed)
    pipe.fit(train, train[TARGET])
    prediction = pipe.predict(test)
    metrics = regression_metrics(test[TARGET].to_numpy(), prediction)
    group = test[split.calibration_group].to_numpy()
    interval = group_bootstrap_ci(test[TARGET], prediction, group, seed=seed) if not variant.startswith("ablate_") else {}
    result = {
        "variant": variant, "protocol": split.name, **metrics,
        "train_rows": len(train), "test_rows": len(test),
        "train_cells": train["SANGER_MODEL_ID"].nunique(), "test_cells": test["SANGER_MODEL_ID"].nunique(),
        "train_chemical_identities": train["CHEMICAL_IDENTITY"].nunique(),
        "test_chemical_identities": test["CHEMICAL_IDENTITY"].nunique(),
        "group_bootstrap_95_ci": json.dumps(interval),
        **metadata,
    }
    predictions = test[["ROW_ID", "SANGER_MODEL_ID", "CELL_LINE_NAME", "DRUG_ID", "DRUG_NAME", "CHEMICAL_IDENTITY", TARGET]].copy()
    predictions["prediction"] = prediction
    predictions["residual"] = predictions[TARGET] - prediction
    predictions["absolute_error"] = predictions["residual"].abs()
    predictions["variant"] = variant
    predictions["protocol"] = split.name
    return result, predictions, pipe


def group_conformal(
    base: pd.DataFrame, split: EvaluationSplit, expanded: pd.DataFrame, expression: pd.DataFrame,
    chemical: pd.DataFrame, seed: int = 42, alpha: float = 0.05
) -> dict:
    train = base.iloc[split.train]
    groups = np.array(sorted(train[split.calibration_group].unique()), dtype=object)
    rng = np.random.default_rng(seed + 99)
    rng.shuffle(groups)
    calibration_groups = set(groups[:max(1, int(0.2 * len(groups)))])
    cal_ids = set(train.loc[train[split.calibration_group].isin(calibration_groups), "ROW_ID"])
    fit_ids = set(train.loc[~train[split.calibration_group].isin(calibration_groups), "ROW_ID"])
    frame, numeric, categorical, _ = _variant_frame(
        base, "current", expanded, expression, chemical,
        set(base.loc[base["ROW_ID"].isin(fit_ids), "SANGER_MODEL_ID"]),
    )
    pipe = _pipeline(frame, numeric, categorical, False, seed)
    fit = frame[frame["ROW_ID"].isin(fit_ids)]
    cal = frame[frame["ROW_ID"].isin(cal_ids)]
    test = frame.iloc[split.test]
    pipe.fit(fit, fit[TARGET])
    calibration_residual = np.abs(cal[TARGET].to_numpy() - pipe.predict(cal))
    rank = min(len(calibration_residual) - 1, int(np.ceil((len(calibration_residual) + 1) * (1 - alpha))) - 1)
    q = float(np.sort(calibration_residual)[max(rank, 0)])
    prediction = pipe.predict(test)
    errors = np.abs(test[TARGET].to_numpy() - prediction)
    return {
        "protocol": split.name, "calibration_group": split.calibration_group,
        "fit_groups": int(fit[split.calibration_group].nunique()),
        "calibration_groups": int(cal[split.calibration_group].nunique()),
        "test_groups": int(test[split.calibration_group].nunique()),
        "calibration_rows": len(cal), "test_rows": len(test), "alpha": alpha,
        "half_width": q, "empirical_coverage": float(np.mean(errors <= q)),
        "mean_interval_width": 2 * q,
    }


def permutation_table(pipe: Pipeline, test: pd.DataFrame, columns: list[str], seed: int = 42) -> pd.DataFrame:
    result = permutation_importance(
        pipe, test[columns], test[TARGET], scoring="neg_mean_absolute_error",
        n_repeats=8, random_state=seed, n_jobs=-1,
    )
    return pd.DataFrame({
        "feature": columns, "importance_mae_mean": result.importances_mean,
        "importance_mae_sd": result.importances_std,
    }).sort_values("importance_mae_mean", ascending=False)


def audit_dataset(frame: pd.DataFrame) -> dict:
    exact = int(frame.duplicated().sum())
    pair_duplicates = int(frame.duplicated(["SANGER_MODEL_ID", "DRUG_ID"], keep=False).sum())
    curve_duplicates = int(frame.duplicated("NLME_CURVE_ID", keep=False).sum())
    name_ids = frame.groupby("DRUG_NAME")["DRUG_ID"].nunique()
    identity_ids = frame.groupby("CHEMICAL_IDENTITY")["DRUG_ID"].nunique()
    original = make_splits(frame, seed=42)
    c_train = set(frame.iloc[original["C_unseen_compounds"].train]["CHEMICAL_IDENTITY"])
    c_test = set(frame.iloc[original["C_unseen_compounds"].test]["CHEMICAL_IDENTITY"])
    return {
        "rows": len(frame), "exact_duplicate_rows": exact,
        "duplicate_cell_drug_rows": pair_duplicates, "duplicate_curve_ids": curve_duplicates,
        "drug_names_with_multiple_ids": name_ids[name_ids > 1].sort_index().to_dict(),
        "chemical_identities_with_multiple_drug_ids": identity_ids[identity_ids > 1].sort_index().to_dict(),
        "original_compound_split_identity_overlap": sorted(c_train & c_test),
        "target_definition": "LN_IC50 = natural logarithm of fitted IC50 in micromolar; IC50_uM = exp(LN_IC50).",
    }

