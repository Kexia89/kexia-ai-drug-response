from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


CONCENTRATION = re.compile(r"(?:Dox\s*)?(\d+(?:\.\d+)?)\s*nM", re.IGNORECASE)
WELL = re.compile(r"\(([A-H]\d+)\)\s*$", re.IGNORECASE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cell_line(filename: str) -> str:
    if filename.startswith("BT474"):
        return "BT-474"
    if filename.startswith("MCF7"):
        return "MCF7"
    return "MDA-MB-231"


def prepare_temporal_counts(repository: Path, output_path: Path, metadata_path: Path) -> tuple[pd.DataFrame, dict]:
    raw_dir = repository / "Raw Data"
    rows = []
    files = []
    for workbook in sorted(raw_dir.glob("*.xlsx")):
        raw = pd.read_excel(workbook, sheet_name="Raw Data", header=None)
        header_positions = raw.index[raw.iloc[:, 0].astype(str).eq("Date Time")]
        if len(header_positions) != 1:
            continue
        header_row = int(header_positions[0])
        headers = raw.iloc[header_row].tolist()
        data = raw.iloc[header_row + 1:].copy()
        data.columns = headers
        elapsed = pd.to_numeric(data.iloc[:, 1], errors="coerce")
        for column in data.columns[2:]:
            label = str(column)
            match = CONCENTRATION.search(label)
            untreated = bool(re.search(r"untreated|control|0\s*nM", label, re.IGNORECASE))
            # In the single-dose concentration workbooks, columns without a Dox
            # token are the explicitly described untreated wells.
            if workbook.stem.endswith("Drug_Concentration") and match is None:
                untreated = True
            concentration = float(match.group(1)) if match else (0.0 if untreated else np.nan)
            well_match = WELL.search(label)
            values = pd.to_numeric(data[column], errors="coerce")
            valid = elapsed.notna() & values.notna()
            for time, count in zip(elapsed[valid], values[valid]):
                rows.append({
                    "source_file": workbook.name, "cell_line": _cell_line(workbook.name),
                    "experiment": workbook.stem.split("_", 1)[1], "elapsed_hours": float(time),
                    "cell_count": float(count), "doxorubicin_nM": concentration,
                    "well": well_match.group(1).upper() if well_match else None,
                    "condition_label": label, "measurement": "Green Object Count (1/Well)",
                })
        files.append({"name": workbook.name, "bytes": workbook.stat().st_size, "sha256": _sha256(workbook)})
    counts = pd.DataFrame(rows)
    zero_fraction = counts.groupby(["source_file", "elapsed_hours"])["cell_count"].transform(lambda x: float((x == 0).mean()))
    counts["all_conditions_zero_at_time"] = zero_fraction.eq(1.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(output_path, index=False, compression="gzip")
    commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_repository": "https://github.com/brocklab/Drug-Response-Dynamics-Model-Calibration",
        "source_commit": commit, "article_doi": "10.1371/journal.pcbi.1009104",
        "biostudies_accession": "S-EPMC9004764",
        "rows": len(counts), "files": files, "cell_lines": sorted(counts["cell_line"].unique()),
        "elapsed_hours_range": [float(counts["elapsed_hours"].min()), float(counts["elapsed_hours"].max())],
        "concentration_unit": "nM", "measurement_unit": "Green Object Count per well",
        "zero_handling": "Values are retained. Entire time points with zero across all conditions are flagged as likely acquisition gaps.",
        "license_status": (
            "No LICENSE file detected in the source repository. The associated PLOS article is CC BY; "
            "repository/data reuse terms require independent review before redistribution."
        ),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return counts, metadata

