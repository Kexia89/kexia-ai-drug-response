from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kexia_portfolio.plots import save_error_plot, save_prediction_plot


if __name__ == "__main__":
    artifacts = ROOT / "artifacts"
    predictions = pd.read_csv(artifacts / "predictions.csv")
    figures = artifacts / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    save_prediction_plot(predictions, figures / "observed_vs_predicted.html")
    save_error_plot(predictions, figures / "error_by_cell.html")
    print(figures)
