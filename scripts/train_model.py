from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kexia_portfolio.modeling import train_artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame = pd.read_csv(args.input, low_memory=False)
    artifact, predictions, report = train_artifact(frame, fraction=args.test_fraction, seed=args.seed)
    output = ROOT / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output / "drug_response_model.joblib")
    predictions.to_csv(output / "predictions.csv", index=False)
    (output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
