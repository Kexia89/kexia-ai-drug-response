#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kexia_cancer.features.molecular import prepare_breast_expression, prepare_expanded_molecular
from kexia_cancer.paths import METADATA_DIR, PROCESSED_DIR, RAW_DIR

modeling = pd.read_csv(PROCESSED_DIR / "breast_gdsc2_modeling.csv.gz", usecols=["SANGER_MODEL_ID"])
model_ids = sorted(modeling["SANGER_MODEL_ID"].dropna().unique())
expression, expression_meta = prepare_breast_expression(
    sorted(RAW_DIR.glob("rnaseq_merged_*.zip"))[-1],
    model_ids,
    PROCESSED_DIR / "breast_expression_log2_tpm.parquet",
    METADATA_DIR / "breast_expression_metadata.json",
)
expanded, expanded_meta = prepare_expanded_molecular(
    sorted(RAW_DIR.glob("mutations_summary_*.zip"))[-1],
    sorted(RAW_DIR.glob("cnv_summary_*.zip"))[-1],
    model_ids,
    PROCESSED_DIR / "breast_expanded_molecular.parquet",
    METADATA_DIR / "breast_expanded_molecular_metadata.json",
)
print(json.dumps({"expression": expression_meta, "expanded": expanded_meta}, indent=2))

