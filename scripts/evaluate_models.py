#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "artifacts" / "reports" / "model_evaluation.json"
if not path.exists():
    raise SystemExit("No evaluation exists. Run: python scripts/train_models.py")
report = json.loads(path.read_text(encoding="utf-8"))
print(f"Selected model: {report['selected_model']} | target: {report['target']}")
for name, result in report["protocols"].items():
    m = result["metrics"]
    s = result["split"]
    print(
        f"{name}: n={s['test_rows']}, MAE={m['mae']:.4f}, RMSE={m['rmse']:.4f}, "
        f"R2={m['r2']:.4f}, Pearson={m['pearson']:.4f}, Spearman={m['spearman']:.4f}"
    )
print("Full metrics and bootstrap intervals:", path)

