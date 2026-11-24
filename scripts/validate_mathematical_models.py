#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kexia_cancer.evaluation.math_validation import validate_endpoint_models
from kexia_cancer.paths import PROCESSED_DIR, REPORT_DIR

raw = PROCESSED_DIR / "breast_gdsc2_raw_wells.csv.gz"
if not raw.exists():
    raise SystemExit("Raw breast wells are missing. Run prepare_breast_cancer.py after downloading gdsc2_raw.")
report = validate_endpoint_models(raw, PROCESSED_DIR, REPORT_DIR)
print(json.dumps(report, indent=2, default=str))

