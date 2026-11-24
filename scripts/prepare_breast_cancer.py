#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kexia_cancer.data.prepare import prepare_dataset
from kexia_cancer.paths import PROCESSED_DIR, RAW_DIR, REPORT_DIR, ensure_directories


if __name__ == "__main__":
    ensure_directories()
    result = prepare_dataset(RAW_DIR, PROCESSED_DIR, REPORT_DIR)
    print(json.dumps(result, indent=2, default=str))

