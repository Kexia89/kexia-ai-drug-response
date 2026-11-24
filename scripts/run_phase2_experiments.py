#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kexia_cancer.evaluation.phase2 import (
    audit_dataset, chemical_identity, evaluate_variant, group_conformal,
    make_independent_splits, permutation_table,
)
from kexia_cancer.models.baselines import feature_columns
from kexia_cancer.paths import PROCESSED_DIR, REPORT_DIR


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(PROCESSED_DIR / "breast_gdsc2_modeling.csv.gz", low_memory=False)
    base = base[base["LN_IC50"].notna()].reset_index(drop=True)
    base.insert(0, "ROW_ID", range(len(base)))
    base["CHEMICAL_IDENTITY"] = chemical_identity(base)
    expanded = pd.read_parquet(PROCESSED_DIR / "breast_expanded_molecular.parquet")
    expression = pd.read_parquet(PROCESSED_DIR / "breast_expression_log2_tpm.parquet")
    chemical = pd.read_csv(PROCESSED_DIR / "compound_rdkit_descriptors.csv")
    splits = make_independent_splits(base)

    audit = audit_dataset(base)
    (REPORT_DIR / "phase2_dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    ablations = [
        "current", "ablate_MIN_CONC", "ablate_MAX_CONC", "ablate_SCREENING_SITE",
        "ablate_PUTATIVE_TARGET", "ablate_PATHWAY_NAME", "ablate_MIN_CONC+MAX_CONC",
        "ablate_SCREENING_SITE+PUTATIVE_TARGET+PATHWAY_NAME",
    ]
    comparisons = ["current", "expanded_molecular", "expression_pca", "combined"]
    results, predictions = [], []
    fitted = {}
    def checkpoint() -> None:
        pd.DataFrame(results).to_csv(REPORT_DIR / "phase2_model_comparison.partial.csv", index=False)
        if predictions:
            pd.concat(predictions, ignore_index=True).to_csv(
                REPORT_DIR / "phase2_predictions.partial.csv.gz", index=False, compression="gzip"
            )
    for protocol, split in splits.items():
        protocol_ablations = ablations if protocol in {"B_unseen_cells", "C_unseen_chemical_identities"} else ["current"]
        protocol_comparisons = comparisons[1:] if protocol in {"B_unseen_cells", "C_unseen_chemical_identities"} else []
        for variant in protocol_ablations:
            print(f"Evaluating {protocol} / {variant}", flush=True)
            result, pred, pipe = evaluate_variant(base, split, variant, expanded, expression, chemical)
            results.append(result); predictions.append(pred)
            checkpoint()
            if protocol == "B_unseen_cells" and variant == "current":
                fitted[(protocol, variant)] = (pipe, split)
        for variant in protocol_comparisons:
            print(f"Evaluating {protocol} / {variant}", flush=True)
            result, pred, _ = evaluate_variant(base, split, variant, expanded, expression, chemical)
            results.append(result); predictions.append(pred)
            checkpoint()

    # Chemical comparisons are restricted to the same verified-structure subset.
    for protocol in ["C_unseen_chemical_identities"]:
        split = splits[protocol]
        for variant in ["chemical_descriptors"]:
            print(f"Evaluating {protocol} / {variant}", flush=True)
            result, pred, _ = evaluate_variant(base, split, variant, expanded, expression, chemical)
            results.append(result); predictions.append(pred)
            checkpoint()

    results_frame = pd.DataFrame(results)
    results_frame.to_csv(REPORT_DIR / "phase2_model_comparison.csv", index=False)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    prediction_frame.to_csv(REPORT_DIR / "phase2_predictions.csv.gz", index=False, compression="gzip")

    intervals = [group_conformal(base, split, expanded, expression, chemical) for split in splits.values()]
    pd.DataFrame(intervals).to_csv(REPORT_DIR / "phase2_interval_calibration.csv", index=False)

    b_predictions = prediction_frame[
        (prediction_frame["protocol"] == "B_unseen_cells") & (prediction_frame["variant"] == "current")
    ]
    b_predictions.groupby(["SANGER_MODEL_ID", "CELL_LINE_NAME"], as_index=False).agg(
        observations=("absolute_error", "size"), mae=("absolute_error", "mean"),
        mean_residual=("residual", "mean"), rmse=("residual", lambda x: float((x.pow(2).mean()) ** .5)),
    ).to_csv(REPORT_DIR / "phase2_errors_by_cell.csv", index=False)
    prediction_frame[prediction_frame["variant"] == "current"].groupby(
        ["DRUG_ID", "DRUG_NAME"], as_index=False
    ).agg(observations=("absolute_error", "size"), mae=("absolute_error", "mean"),
          mean_residual=("residual", "mean")).to_csv(REPORT_DIR / "phase2_errors_by_compound.csv", index=False)

    pipe, b_split = fitted[("B_unseen_cells", "current")]
    numeric, categorical = feature_columns(base)
    test = base.iloc[b_split.test]
    importance = permutation_table(pipe, test, numeric + categorical)
    importance.to_csv(REPORT_DIR / "phase2_feature_importance.csv", index=False)

    summary = {
        "audit": audit,
        "independent_evaluation": "Fixed tests were not used for feature selection or hyperparameter tuning.",
        "expression": "Variance selection, scaling and PCA are fit on unique training cells only for every split.",
        "expanded_molecular": "Imputation and univariate feature selection are inside the training-only sklearn pipeline.",
        "chemical": "Only exact verified ChEMBL/InChIKey structures; chemical evaluation uses identical rows for fair comparison.",
        "subtype_error_analysis": "Not reported unless a verified subtype annotation is present; no subtype is inferred.",
        "outputs": [p.name for p in REPORT_DIR.glob("phase2_*")],
    }
    (REPORT_DIR / "phase2_experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(results_frame[["protocol", "variant", "mae", "rmse", "r2"]].to_string(index=False))


if __name__ == "__main__":
    main()
