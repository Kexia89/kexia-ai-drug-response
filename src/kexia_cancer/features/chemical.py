from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"


def _request_json(session: requests.Session, url: str, params: dict | None = None, retries: int = 4) -> dict:
    for attempt in range(retries):
        response = session.get(url, params=params, timeout=(20, 60), headers={"Accept": "application/json"})
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"ChEMBL request failed after {retries} attempts: {url}")


def _lookup(session: requests.Session, chembl_id: str | None, inchi_key: str | None) -> tuple[dict | None, str]:
    if chembl_id:
        url = f"{CHEMBL_API}/molecule/{chembl_id}.json"
        try:
            return _request_json(session, url), url
        except requests.HTTPError as exc:
            if exc.response is None or exc.response.status_code != 404:
                raise
    if inchi_key:
        url = f"{CHEMBL_API}/molecule.json"
        payload = _request_json(
            session,
            url,
            params={"molecule_structures__standard_inchi_key": inchi_key, "limit": 5},
        )
        molecules = payload.get("molecules", [])
        if len(molecules) == 1:
            return molecules[0], f"{url}?molecule_structures__standard_inchi_key={inchi_key}"
    return None, ""


def _descriptors(smiles: str) -> dict | None:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
    return {
        "canonical_smiles_rdkit": Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True),
        "murcko_scaffold": scaffold or pd.NA,
        "desc_mol_wt": Descriptors.MolWt(molecule),
        "desc_logp": Crippen.MolLogP(molecule),
        "desc_tpsa": rdMolDescriptors.CalcTPSA(molecule),
        "desc_hbd": Lipinski.NumHDonors(molecule),
        "desc_hba": Lipinski.NumHAcceptors(molecule),
        "desc_rotatable_bonds": Lipinski.NumRotatableBonds(molecule),
        "desc_aromatic_rings": Lipinski.NumAromaticRings(molecule),
        "desc_ring_count": Lipinski.RingCount(molecule),
        "desc_fraction_csp3": rdMolDescriptors.CalcFractionCSP3(molecule),
        "desc_heavy_atoms": Lipinski.HeavyAtomCount(molecule),
        "desc_qed": QED.qed(molecule),
    }


def resolve_chembl_structures(
    compounds: pd.DataFrame,
    output_csv: Path,
    manifest_path: Path,
    cache_path: Path,
) -> tuple[pd.DataFrame, dict]:
    """Resolve official ChEMBL structures and accept only exact InChIKey matches."""
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    session = requests.Session()
    records, failures = [], []
    for row in compounds.drop_duplicates("DRUG_ID").itertuples(index=False):
        drug_id = int(row.DRUG_ID)
        chembl_id = None if pd.isna(getattr(row, "CHEMBL_ID", None)) else str(row.CHEMBL_ID).strip()
        inchi_key = None if pd.isna(getattr(row, "INCHI_KEY", None)) else str(row.INCHI_KEY).strip()
        cache_key = f"{chembl_id or ''}|{inchi_key or ''}"
        try:
            if cache_key in cache:
                molecule, source_url = cache[cache_key].get("molecule"), cache[cache_key].get("source_url", "")
            else:
                molecule, source_url = _lookup(session, chembl_id, inchi_key)
                cache[cache_key] = {"molecule": molecule, "source_url": source_url}
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
            if not molecule:
                failures.append({"DRUG_ID": drug_id, "reason": "not_found_in_chembl"})
                continue
            structures = molecule.get("molecule_structures") or {}
            returned_key = structures.get("standard_inchi_key")
            smiles = structures.get("canonical_smiles")
            if inchi_key and returned_key != inchi_key:
                failures.append({"DRUG_ID": drug_id, "reason": "inchi_key_mismatch", "returned": returned_key})
                continue
            descriptor_values = _descriptors(smiles) if smiles else None
            if descriptor_values is None:
                failures.append({"DRUG_ID": drug_id, "reason": "missing_or_invalid_smiles"})
                continue
            records.append(
                {
                    "DRUG_ID": drug_id,
                    "DRUG_NAME": row.DRUG_NAME,
                    "source_chembl_id": molecule.get("molecule_chembl_id"),
                    "source_inchi_key": returned_key,
                    "source_url": source_url,
                    "inchi_key_verified": bool(not inchi_key or returned_key == inchi_key),
                    "chembl_canonical_smiles": smiles,
                    **descriptor_values,
                }
            )
        except Exception as exc:
            failures.append({"DRUG_ID": drug_id, "reason": type(exc).__name__, "detail": str(exc)})
    frame = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "ChEMBL REST API, EMBL-EBI",
        "api_base": CHEMBL_API,
        "verification": "Exact standard InChIKey match when a CMP InChIKey was supplied; no inferred structures accepted.",
        "input_compounds": int(compounds["DRUG_ID"].nunique()),
        "resolved_structures": int(frame["DRUG_ID"].nunique()) if not frame.empty else 0,
        "unique_scaffolds": int(frame["murcko_scaffold"].nunique()) if not frame.empty else 0,
        "failures": failures,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return frame, manifest


def build_descriptors_from_raw_chembl(
    compounds: pd.DataFrame,
    raw_files: list[Path],
    output_csv: Path,
    manifest_path: Path,
) -> tuple[pd.DataFrame, dict]:
    """Build verified descriptors from raw payloads fetched by the ChEMBL skill client."""
    molecules: dict[str, dict] = {}
    by_key: dict[str, dict] = {}
    for raw_file in raw_files:
        payload = json.loads(raw_file.read_text(encoding="utf-8"))
        for molecule in payload.get("molecules", []):
            chembl_id = molecule.get("molecule_chembl_id")
            structures = molecule.get("molecule_structures") or {}
            inchi_key = structures.get("standard_inchi_key")
            if chembl_id:
                molecules[chembl_id] = molecule
            if inchi_key:
                by_key[inchi_key] = molecule
    records, failures = [], []
    for row in compounds.drop_duplicates("DRUG_ID").itertuples(index=False):
        drug_id = int(row.DRUG_ID)
        chembl_id = None if pd.isna(getattr(row, "CHEMBL_ID", None)) else str(row.CHEMBL_ID).strip()
        inchi_key = None if pd.isna(getattr(row, "INCHI_KEY", None)) else str(row.INCHI_KEY).strip()
        molecule = molecules.get(chembl_id or "") or by_key.get(inchi_key or "")
        if molecule is None:
            failures.append({"DRUG_ID": drug_id, "reason": "not_found_in_fetched_chembl_records"})
            continue
        structures = molecule.get("molecule_structures") or {}
        returned_key = structures.get("standard_inchi_key")
        smiles = structures.get("canonical_smiles")
        if inchi_key and returned_key != inchi_key:
            failures.append({"DRUG_ID": drug_id, "reason": "inchi_key_mismatch", "returned": returned_key})
            continue
        descriptor_values = _descriptors(smiles) if smiles else None
        if descriptor_values is None:
            failures.append({"DRUG_ID": drug_id, "reason": "missing_or_invalid_smiles"})
            continue
        resolved_id = molecule.get("molecule_chembl_id")
        records.append(
            {
                "DRUG_ID": drug_id,
                "DRUG_NAME": row.DRUG_NAME,
                "source_chembl_id": resolved_id,
                "source_inchi_key": returned_key,
                "source_url": f"https://www.ebi.ac.uk/chembl/explore/compound/{resolved_id}",
                "inchi_key_verified": bool(not inchi_key or returned_key == inchi_key),
                "chembl_canonical_smiles": smiles,
                **descriptor_values,
            }
        )
    frame = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "ChEMBL REST API, EMBL-EBI",
        "api_base": CHEMBL_API,
        "raw_payloads": [
            (
                path.resolve().relative_to(Path(__file__).resolve().parents[3]).as_posix()
                if path.resolve().is_relative_to(Path(__file__).resolve().parents[3])
                else path.resolve().as_posix()
            )
            for path in raw_files
        ],
        "verification": "Exact standard InChIKey match when a CMP InChIKey was supplied; no inferred structures accepted.",
        "input_compounds": int(compounds["DRUG_ID"].nunique()),
        "resolved_structures": int(frame["DRUG_ID"].nunique()) if not frame.empty else 0,
        "unique_scaffolds": int(frame["murcko_scaffold"].nunique()) if not frame.empty else 0,
        "failures": failures,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return frame, manifest

