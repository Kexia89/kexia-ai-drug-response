#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kexia_cancer.evaluation.math_audit import run_math_audit
from kexia_cancer.paths import PROCESSED_DIR, REPORT_DIR


if __name__ == "__main__":
    report = run_math_audit(
        PROCESSED_DIR / "breast_gdsc2_raw_wells.csv.gz",
        PROCESSED_DIR / "breast_gdsc2_normalized_dose_response.csv.gz",
        REPORT_DIR,
    )
    print(json.dumps(report, indent=2))
