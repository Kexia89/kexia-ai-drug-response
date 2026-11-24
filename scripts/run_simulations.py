#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import json
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kexia_cancer.paths import FIGURE_DIR, REPORT_DIR, ensure_directories
from kexia_cancer.simulation.models import (
    ResistanceParameters,
    SinglePopulationParameters,
    resistance_sensitivity_samples,
    sensitivity_ensemble,
    simulate_single,
)

ensure_directories()
untreated = simulate_single(
    SinglePopulationParameters(concentration_uM=0.0, kmax_per_hour=0.0), label="Simulación hipotética no calibrada"
)
untreated["scenario"] = "sin intervención"
treated = simulate_single(SinglePopulationParameters(), label="Simulación hipotética no calibrada")
treated["scenario"] = "intervención hipotética"
result = __import__("pandas").concat([untreated, treated], ignore_index=True)
result.to_csv(REPORT_DIR / "conceptual_simulation.csv", index=False)
figure = px.line(result, x="time_hours", y="population", color="scenario", title="Simulación hipotética no calibrada")
figure.write_html(FIGURE_DIR / "conceptual_simulation.html", include_plotlyjs="cdn")
bands = sensitivity_ensemble(
    SinglePopulationParameters(),
    {
        "proliferation_rate_per_hour": (0.015, 0.045),
        "kmax_per_hour": (0.01, 0.08),
        "ec50_uM": (0.2, 5.0),
    },
    samples=200,
)
bands.to_csv(REPORT_DIR / "single_population_sensitivity_bands.csv", index=False)
samples = resistance_sensitivity_samples(
    ResistanceParameters(),
    {
        "proliferation_rate_per_hour": (0.015, 0.045),
        "kmax_per_hour": (0.01, 0.08),
        "ec50_uM": (0.2, 5.0),
        "initial_resistant_fraction": (0.0, 0.2),
        "resistant_kmax_per_hour": (0.0, 0.02),
        "resistant_ec50_uM": (2.0, 30.0),
    },
    samples=250,
)
samples.to_csv(REPORT_DIR / "resistance_sensitivity_samples.csv.gz", index=False, compression="gzip")
parameter_names = [
    "proliferation_rate_per_hour", "kmax_per_hour", "ec50_uM", "initial_resistant_fraction",
    "resistant_kmax_per_hour", "resistant_ec50_uM",
]
correlations = {
    parameter: {
        outcome: float(spearmanr(samples[parameter], samples[outcome]).statistic)
        for outcome in ["final_population", "final_resistant_fraction"]
    }
    for parameter in parameter_names
}
(REPORT_DIR / "simulation_sensitivity_summary.json").write_text(
    json.dumps({"method": "uniform global sampling; Spearman rank correlation", "samples": 250, "correlations": correlations}, indent=2),
    encoding="utf-8",
)
print(REPORT_DIR / "conceptual_simulation.csv")
print(FIGURE_DIR / "conceptual_simulation.html")

