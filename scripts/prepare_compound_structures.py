#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kexia_cancer.features.chemical import build_descriptors_from_raw_chembl, resolve_chembl_structures
from kexia_cancer.paths import METADATA_DIR, PROCESSED_DIR, RAW_DIR

annotations = pd.read_csv(sorted(RAW_DIR.glob("screened_compounds*.csv"))[-1])
modeling = pd.read_csv(PROCESSED_DIR / "breast_gdsc2_modeling.csv.gz", usecols=["DRUG_ID"])
ids = set(modeling["DRUG_ID"].astype(int))
annotations = annotations[annotations["DRUG_ID"].astype(int).isin(ids)].copy()
annotations = annotations.sort_values("SCREENING_SITE", key=lambda x: x.ne("SANGER")).drop_duplicates("DRUG_ID")
raw_payloads = sorted(METADATA_DIR.glob("chembl_molecules_batch_*.json")) + sorted(
    METADATA_DIR.glob("chembl_inchikey_batch_*.json")
)
if raw_payloads:
    frame, manifest = build_descriptors_from_raw_chembl(
        annotations,
        raw_payloads,
        PROCESSED_DIR / "compound_rdkit_descriptors.csv",
        METADATA_DIR / "chemical_structure_manifest.json",
    )
else:
    frame, manifest = resolve_chembl_structures(
        annotations,
        PROCESSED_DIR / "compound_rdkit_descriptors.csv",
        METADATA_DIR / "chemical_structure_manifest.json",
        METADATA_DIR / "chembl_response_cache.json",
    )
print(json.dumps(manifest, indent=2))

