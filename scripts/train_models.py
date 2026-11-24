#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kexia_cancer.models.baselines import train_and_evaluate
from kexia_cancer.paths import MODEL_DIR, PROCESSED_DIR, REPORT_DIR, ensure_directories


if __name__ == "__main__":
    ensure_directories()
    dataset = pd.read_csv(PROCESSED_DIR / "breast_gdsc2_modeling.csv.gz", low_memory=False)
    report = train_and_evaluate(dataset, MODEL_DIR, REPORT_DIR)
    print(json.dumps(report, indent=2, default=str))

