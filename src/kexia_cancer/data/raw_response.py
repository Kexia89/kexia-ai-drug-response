from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

SINGLE_AGENT_TAG = re.compile(r"^L\d+-D\d+-S$")


def normalize_gdsc2_raw_wells(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize raw intensities using within-plate blank and vehicle controls.

    Blank B wells contain medium/no cells. NC-1 wells are DMSO-treated cells.
    Values are not clipped: observations outside [0, 1] remain visible for QC.
    DMSO tag rows are excluded because they can duplicate a drug-treated position.
    """
    usecols = [
        "BARCODE", "SCAN_ID", "CELL_LINE_NAME", "SANGER_MODEL_ID", "DURATION",
        "POSITION", "TAG", "DRUG_ID", "CONC", "INTENSITY",
    ]
    raw = pd.read_csv(path, usecols=usecols, low_memory=False)
    raw["INTENSITY"] = pd.to_numeric(raw["INTENSITY"], errors="coerce")
    plate_keys = ["BARCODE", "SCAN_ID"]
    blank = raw[raw["TAG"].eq("B")].groupby(plate_keys)["INTENSITY"].median().rename("blank_median")
    vehicle = raw[raw["TAG"].eq("NC-1")].groupby(plate_keys)["INTENSITY"].median().rename("vehicle_median")
    controls = pd.concat([blank, vehicle], axis=1).reset_index()
    controls["dynamic_range"] = controls["vehicle_median"] - controls["blank_median"]
    controls["valid_controls"] = (
        controls[["blank_median", "vehicle_median"]].notna().all(axis=1) & (controls["dynamic_range"] > 0)
    )
    treatments = raw[
        raw["TAG"].astype(str).map(lambda x: bool(SINGLE_AGENT_TAG.match(x)))
        & raw["DRUG_ID"].notna()
        & raw["CONC"].notna()
    ].copy()
    treatments = treatments.merge(controls, on=plate_keys, how="left", validate="many_to_one")
    treatments = treatments[treatments["valid_controls"].fillna(False)].copy()
    treatments["viability_fraction"] = (
        (treatments["INTENSITY"] - treatments["blank_median"]) / treatments["dynamic_range"]
    )
    grouped = (
        treatments.groupby(
            ["SANGER_MODEL_ID", "CELL_LINE_NAME", "DRUG_ID", "CONC", "DURATION"], as_index=False
        )["viability_fraction"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "viability_mean", "std": "viability_sd", "count": "replicates"})
    )
    qc = pd.DataFrame(
        {
            "plates_total": [raw[plate_keys].drop_duplicates().shape[0]],
            "plates_with_valid_controls": [int(controls["valid_controls"].sum())],
            "single_agent_wells_normalized": [len(treatments)],
            "fraction_below_zero": [float((treatments["viability_fraction"] < 0).mean())],
            "fraction_above_one": [float((treatments["viability_fraction"] > 1).mean())],
            "normalization": ["(intensity - median blank B)/(median vehicle NC-1 - median blank B), within plate"],
        }
    )
    return grouped, qc

