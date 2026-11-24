from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def prepare_breast_expression(
    rnaseq_zip: Path,
    model_ids: list[str],
    output_parquet: Path,
    metadata_path: Path,
) -> tuple[pd.DataFrame, dict]:
    """Extract the official merged log2(TPM+1) matrix for verified Sanger model IDs."""
    member = next(
        name for name in zipfile.ZipFile(rnaseq_zip).namelist() if "rsem_tpm" in name.lower() and name.lower().endswith(".csv")
    )
    with zipfile.ZipFile(rnaseq_zip) as archive:
        with archive.open(member) as handle:
            header = pd.read_csv(handle, nrows=0).columns.tolist()
        available = [model_id for model_id in model_ids if model_id in header]
        missing = sorted(set(model_ids) - set(available))
        usecols = header[:3] + available
        with archive.open(member) as handle:
            raw = pd.read_csv(handle, usecols=usecols, low_memory=False)
    model_names = dict(zip(available, raw.iloc[0][available].astype(str)))
    data_sources = dict(zip(available, raw.iloc[1][available].astype(str)))
    genes = raw.iloc[3:].copy()
    genes = genes.rename(columns={header[0]: "gene_symbol", header[1]: "ensembl_gene_id", header[2]: "gene_id"})
    genes = genes[genes["ensembl_gene_id"].notna()].copy()
    genes["feature_id"] = (
        "EXP_" + genes["gene_symbol"].fillna("NA").astype(str) + "|" + genes["ensembl_gene_id"].astype(str)
    )
    values = genes[available].apply(pd.to_numeric, errors="coerce").astype("float32")
    values.index = genes["feature_id"]
    duplicate_feature_rows = int(values.index.duplicated(keep=False).sum())
    duplicate_feature_ids = int(values.index[values.index.duplicated(keep=False)].nunique())
    if duplicate_feature_ids:
        values = values.groupby(level=0, sort=False).median()
    matrix = values.T
    matrix.index.name = "SANGER_MODEL_ID"
    matrix = matrix.reset_index()
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(output_parquet, index=False)
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_archive": (
            rnaseq_zip.resolve().relative_to(Path(__file__).resolve().parents[3]).as_posix()
            if rnaseq_zip.resolve().is_relative_to(Path(__file__).resolve().parents[3])
            else rnaseq_zip.resolve().as_posix()
        ),
        "source_member": member,
        "measurement": "RSEM TPM reported by Sanger DepMap as log2(TPM+1)",
        "join_identifier": "SANGER_MODEL_ID",
        "requested_models": len(model_ids),
        "available_models": len(available),
        "missing_models": missing,
        "genes": int(matrix.shape[1] - 1),
        "duplicate_feature_rows": duplicate_feature_rows,
        "duplicate_feature_ids": duplicate_feature_ids,
        "duplicate_feature_handling": "Median log2(TPM+1) across duplicate feature identifiers.",
        "model_names": model_names,
        "data_sources": data_sources,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return matrix, metadata


def prepare_expanded_molecular(
    mutations_zip: Path,
    cnv_zip: Path,
    model_ids: list[str],
    output_parquet: Path,
    metadata_path: Path,
) -> tuple[pd.DataFrame, dict]:
    """Create all-driver mutation and CNV matrices without response-informed filtering."""
    with zipfile.ZipFile(mutations_zip) as archive:
        member = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
        with archive.open(member) as handle:
            mutations = pd.read_csv(handle, low_memory=False)
    mutations = mutations[
        mutations["model_id"].isin(model_ids)
        & mutations["cancer_driver"].astype(str).str.lower().isin(["t", "true", "1"])
        & mutations["gene_symbol"].notna()
    ].copy()
    mutation_models = set(mutations["model_id"])
    mutations["value"] = 1.0
    mut = mutations.pivot_table(index="model_id", columns="gene_symbol", values="value", aggfunc="max")
    mut = mut.reindex(model_ids).fillna(0.0).add_prefix("MUTX_")

    with zipfile.ZipFile(cnv_zip) as archive:
        member = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
        with archive.open(member) as handle:
            cnv = pd.read_csv(handle, low_memory=False)
    cnv = cnv[cnv["model_id"].isin(model_ids) & cnv["symbol"].notna()].copy()
    cnv["total_copy_number"] = pd.to_numeric(cnv["total_copy_number"], errors="coerce")
    cnv_models = set(cnv["model_id"])
    cn = cnv.pivot_table(index="model_id", columns="symbol", values="total_copy_number", aggfunc="median")
    cn = cn.reindex(model_ids).add_prefix("CNX_")

    matrix = pd.concat([mut, cn], axis=1)
    matrix.insert(0, "SANGER_MODEL_ID", matrix.index)
    matrix["mutation_profile_available"] = matrix["SANGER_MODEL_ID"].isin(mutation_models).astype(float)
    matrix["cnv_profile_available"] = matrix["SANGER_MODEL_ID"].isin(cnv_models).astype(float)
    matrix = matrix.reset_index(drop=True)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(output_parquet, index=False)
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "join_identifier": "SANGER_MODEL_ID",
        "models": len(model_ids),
        "mutation_driver_features": int(len(mut.columns)),
        "cnv_driver_features": int(len(cn.columns)),
        "mutation_models_available": len(mutation_models),
        "cnv_models_available": len(cnv_models),
        "selection": "All genes present in official driver summaries; no response-informed gene filtering.",
        "duplicate_handling": "Mutation=max binary across records; CNV=median total copy number across sources.",
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return matrix, metadata

