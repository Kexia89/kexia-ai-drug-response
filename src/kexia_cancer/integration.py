from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from kexia_cancer.simulation.models import SinglePopulationParameters, simulate_single


def evaluate_pair(model_path: Path, data: pd.DataFrame, cell_id: str, drug_id: int) -> dict:
    rows = data[(data["SANGER_MODEL_ID"] == cell_id) & (data["DRUG_ID"] == drug_id)]
    if rows.empty:
        raise KeyError("The requested verified cell-compound pair is absent from the processed feature table")
    row = rows.iloc[[0]]
    artifact = joblib.load(model_path)
    prediction = artifact.predict(row).iloc[0].to_dict()
    observation = {
        "LN_IC50": float(row.iloc[0]["LN_IC50"]),
        "AUC": float(row.iloc[0]["AUC"]),
        "RMSE": float(row.iloc[0]["RMSE"]),
        "provenance": "experimental observation: GDSC2 72 h fitted viability curve",
    }
    return {
        "cell_id": cell_id,
        "drug_id": int(drug_id),
        "observation": observation,
        "prediction": {**prediction, "target": artifact.target, "provenance": "AI prediction"},
        "dynamic_mapping": None,
        "warning": "LN_IC50 is not converted to a killing rate. Set ODE pharmacodynamic parameters explicitly or calibrate them with independent time-course evidence.",
    }


def conceptual_simulation(params: SinglePopulationParameters, duration_hours: float) -> pd.DataFrame:
    return simulate_single(params, duration_hours=duration_hours, label="Simulación hipotética no calibrada")

