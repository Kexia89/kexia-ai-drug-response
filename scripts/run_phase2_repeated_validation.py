#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kexia_cancer.evaluation.phase2 import (
    EvaluationSplit, chemical_identity, evaluate_variant, make_independent_splits, regression_metrics,
)
from kexia_cancer.paths import PROCESSED_DIR, REPORT_DIR


def _held_groups(frame: pd.DataFrame, candidates: np.ndarray, column: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    groups = np.array(sorted(frame.iloc[candidates][column].unique()), dtype=object)
    rng = np.random.default_rng(seed)
    held = set(rng.choice(groups, max(1, int(.2 * len(groups))), replace=False))
    test = candidates[frame.iloc[candidates][column].isin(held).to_numpy()]
    train = candidates[~frame.iloc[candidates][column].isin(held).to_numpy()]
    return train, test


def _reference_prediction(train: pd.DataFrame, test: pd.DataFrame, protocol: str) -> np.ndarray:
    global_mean = float(train["LN_IC50"].mean())
    if protocol == "repeated_unseen_cells":
        means = train.groupby("CHEMICAL_IDENTITY")["LN_IC50"].mean()
        return test["CHEMICAL_IDENTITY"].map(means).fillna(global_mean).to_numpy()
    means = train.groupby("SANGER_MODEL_ID")["LN_IC50"].mean()
    return test["SANGER_MODEL_ID"].map(means).fillna(global_mean).to_numpy()


def main() -> None:
    base = pd.read_csv(PROCESSED_DIR / "breast_gdsc2_modeling.csv.gz", low_memory=False)
    base = base[base["LN_IC50"].notna()].reset_index(drop=True)
    base.insert(0, "ROW_ID", range(len(base)))
    base["CHEMICAL_IDENTITY"] = chemical_identity(base)
    expanded = pd.read_parquet(PROCESSED_DIR / "breast_expanded_molecular.parquet")
    expression = pd.read_parquet(PROCESSED_DIR / "breast_expression_log2_tpm.parquet")
    chemical = pd.read_csv(PROCESSED_DIR / "compound_rdkit_descriptors.csv")
    rows = []
    designs = [
        ("repeated_unseen_cells", "SANGER_MODEL_ID", "B_unseen_cells", ["current", "combined"]),
        ("repeated_unseen_chemical_identities", "CHEMICAL_IDENTITY", "C_unseen_chemical_identities", ["current", "chemical_descriptors"]),
    ]
    for protocol, group_column, independent_name, variants in designs:
        analysis = base
        if protocol == "repeated_unseen_chemical_identities":
            analysis = base[base["DRUG_ID"].isin(chemical["DRUG_ID"])].reset_index(drop=True)
        independent = make_independent_splits(analysis)
        reserved_test = set(independent[independent_name].test)
        development = np.array([i for i in range(len(analysis)) if i not in reserved_test])
        for repeat, seed in enumerate([101, 202, 303], start=1):
            train_idx, test_idx = _held_groups(analysis, development, group_column, seed)
            split = EvaluationSplit(protocol, train_idx, test_idx, group_column)
            train, test = analysis.iloc[train_idx], analysis.iloc[test_idx]
            reference = _reference_prediction(train, test, protocol)
            rows.append({"protocol": protocol, "repeat": repeat, "seed": seed, "variant": "group_mean_reference",
                         **regression_metrics(test["LN_IC50"], reference), "train_groups": train[group_column].nunique(),
                         "test_groups": test[group_column].nunique()})
            for variant in variants:
                print(f"{protocol} repeat={repeat} variant={variant}", flush=True)
                result, _, _ = evaluate_variant(analysis, split, variant, expanded, expression, chemical, seed=seed)
                rows.append({"protocol": protocol, "repeat": repeat, "seed": seed, "variant": variant,
                             **{key: result[key] for key in ["mae", "rmse", "r2", "pearson", "spearman"]},
                             "train_groups": train[group_column].nunique(), "test_groups": test[group_column].nunique()})
    detail = pd.DataFrame(rows)
    detail.to_csv(REPORT_DIR / "phase2_repeated_group_validation.csv", index=False)
    summary = detail.groupby(["protocol", "variant"], as_index=False).agg(
        repeats=("repeat", "size"), mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
        rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"), r2_mean=("r2", "mean"), r2_sd=("r2", "std"),
    )
    summary.to_csv(REPORT_DIR / "phase2_repeated_group_validation_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
