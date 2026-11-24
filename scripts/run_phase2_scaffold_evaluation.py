#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kexia_cancer.evaluation.phase2 import EvaluationSplit, chemical_identity, evaluate_variant
from kexia_cancer.paths import PROCESSED_DIR, REPORT_DIR


def main() -> None:
    chemical = pd.read_csv(PROCESSED_DIR / "compound_rdkit_descriptors.csv")
    base = pd.read_csv(PROCESSED_DIR / "breast_gdsc2_modeling.csv.gz", low_memory=False)
    base = base[base["DRUG_ID"].isin(chemical["DRUG_ID"]) & base["LN_IC50"].notna()].reset_index(drop=True)
    base.insert(0, "ROW_ID", range(len(base)))
    base["CHEMICAL_IDENTITY"] = chemical_identity(base)
    scaffold = chemical.set_index("DRUG_ID")["murcko_scaffold"].fillna("")
    base["SCAFFOLD_GROUP"] = base["DRUG_ID"].map(scaffold)
    acyclic = base["SCAFFOLD_GROUP"].eq("")
    base.loc[acyclic, "SCAFFOLD_GROUP"] = "ACYCLIC::" + base.loc[acyclic, "CHEMICAL_IDENTITY"]
    groups = np.array(sorted(base["SCAFFOLD_GROUP"].unique()), dtype=object)
    rng = np.random.default_rng(42)
    held = set(rng.choice(groups, max(1, int(.2 * len(groups))), replace=False))
    test_mask = base["SCAFFOLD_GROUP"].isin(held).to_numpy()
    split = EvaluationSplit("E_unseen_chemical_scaffolds", np.flatnonzero(~test_mask), np.flatnonzero(test_mask), "SCAFFOLD_GROUP")
    expanded = pd.read_parquet(PROCESSED_DIR / "breast_expanded_molecular.parquet")
    expression = pd.read_parquet(PROCESSED_DIR / "breast_expression_log2_tpm.parquet")
    rows = []
    for variant in ["current", "chemical_descriptors"]:
        result, prediction, _ = evaluate_variant(base, split, variant, expanded, expression, chemical)
        rows.append(result)
        prediction.to_csv(REPORT_DIR / f"phase2_scaffold_predictions_{variant}.csv.gz", index=False, compression="gzip")
    pd.DataFrame(rows).to_csv(REPORT_DIR / "phase2_scaffold_evaluation.csv", index=False)
    metadata = {
        "structures": int(chemical["DRUG_ID"].nunique()), "scaffold_groups": len(groups),
        "held_out_scaffold_groups": len(held), "train_test_scaffold_overlap": 0,
        "acyclic_handling": "Empty Murcko scaffolds are grouped by verified chemical identity, not combined into one artificial scaffold.",
    }
    (REPORT_DIR / "phase2_scaffold_evaluation.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows)[["variant", "mae", "rmse", "r2", "test_rows"]].to_string(index=False))


if __name__ == "__main__":
    main()
