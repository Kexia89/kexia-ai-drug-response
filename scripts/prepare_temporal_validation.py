#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kexia_cancer.data.temporal import prepare_temporal_counts
from kexia_cancer.evaluation.temporal_validation import validate_untreated_logistic
from kexia_cancer.paths import METADATA_DIR, PROCESSED_DIR, RAW_DIR, REPORT_DIR


if __name__ == "__main__":
    counts, metadata = prepare_temporal_counts(
        RAW_DIR / "Drug-Response-Dynamics-Model-Calibration",
        PROCESSED_DIR / "breast_temporal_doxorubicin_counts.csv.gz",
        METADATA_DIR / "breast_temporal_doxorubicin_metadata.json",
    )
    validation = validate_untreated_logistic(
        PROCESSED_DIR / "breast_temporal_doxorubicin_counts.csv.gz", REPORT_DIR
    )
    print(json.dumps({"dataset": metadata, "validation": validation}, indent=2))
