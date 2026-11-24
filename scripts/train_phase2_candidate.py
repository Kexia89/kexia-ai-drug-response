#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kexia_cancer.evaluation.phase2 import (
    Phase2CombinedArtifact, _pipeline, _variant_frame, chemical_identity,
)
from kexia_cancer.paths import MODEL_DIR, PROCESSED_DIR


if __name__ == "__main__":
    base = pd.read_csv(PROCESSED_DIR / "breast_gdsc2_modeling.csv.gz", low_memory=False)
    base = base[base["LN_IC50"].notna()].reset_index(drop=True)
    base.insert(0, "ROW_ID", range(len(base)))
    base["CHEMICAL_IDENTITY"] = chemical_identity(base)
    expanded = pd.read_parquet(PROCESSED_DIR / "breast_expanded_molecular.parquet")
    expression = pd.read_parquet(PROCESSED_DIR / "breast_expression_log2_tpm.parquet")
    chemical = pd.read_csv(PROCESSED_DIR / "compound_rdkit_descriptors.csv")
    frame, numeric, categorical, transform_metadata = _variant_frame(
        base, "combined", expanded, expression, chemical, set(base["SANGER_MODEL_ID"])
    )
    pipeline = _pipeline(frame, numeric, categorical, select_numeric=True, seed=42)
    pipeline.fit(frame, frame["LN_IC50"])
    expression_columns = [column for column in frame if column.startswith("EXPR_PC_")]
    scores = frame[["SANGER_MODEL_ID"] + expression_columns].drop_duplicates("SANGER_MODEL_ID")
    metadata = {
        "model_version": "phase2_combined_candidate_2026-11-17",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": "LN_IC50 = ln(IC50 in micromolar)", "training_rows": len(frame),
        "training_cells": frame["SANGER_MODEL_ID"].nunique(),
        "expression_transform": transform_metadata["expression_transform"],
        "intended_use": "Preclinical computational hypothesis generation for represented feature schema only.",
        "evaluation_note": "Performance claims must use held-out Phase 2 reports, not this all-data refit.",
    }
    artifact = Phase2CombinedArtifact(pipeline, expanded, scores, metadata)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / "gdsc2_breast_phase2_combined_candidate.joblib"
    joblib.dump(artifact, path)
    (MODEL_DIR / "gdsc2_breast_phase2_combined_candidate.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"artifact": str(path), **metadata}, indent=2))
