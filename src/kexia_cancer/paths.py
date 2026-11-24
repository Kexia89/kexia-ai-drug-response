from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
METADATA_DIR = ROOT / "data" / "metadata"
MODEL_DIR = ROOT / "artifacts" / "models"
REPORT_DIR = ROOT / "artifacts" / "reports"
FIGURE_DIR = ROOT / "artifacts" / "figures"


def ensure_directories() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, METADATA_DIR, MODEL_DIR, REPORT_DIR, FIGURE_DIR):
        path.mkdir(parents=True, exist_ok=True)

