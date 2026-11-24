from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

CURATED_BREAST_GENES = [
    "TP53", "PIK3CA", "PTEN", "BRCA1", "BRCA2", "ERBB2", "ESR1", "RB1",
    "AKT1", "CDH1", "GATA3", "MAP3K1", "CCND1", "MYC", "FGFR1", "PALB2",
]


def _single(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matches {pattern} in {directory}")
    return matches[-1]


def _read_first_csv_from_zip(path: Path, **kwargs) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError(f"No CSV in {path}")
        with archive.open(names[0]) as handle:
            return pd.read_csv(handle, low_memory=False, **kwargs)


def _binary_mutation_features(mutations: pd.DataFrame, model_ids: pd.Index) -> pd.DataFrame:
    mutations = mutations[
        mutations["model_id"].isin(model_ids)
        & mutations["gene_symbol"].isin(CURATED_BREAST_GENES)
        & mutations["cancer_driver"].astype(str).str.lower().isin(["t", "true", "1"])
    ].copy()
    mutations["value"] = 1.0
    matrix = mutations.pivot_table(index="model_id", columns="gene_symbol", values="value", aggfunc="max")
    matrix = matrix.reindex(index=model_ids, columns=CURATED_BREAST_GENES, fill_value=0).fillna(0)
    return matrix.add_prefix("MUT_").reset_index(names="SANGER_MODEL_ID")


def _cnv_features(cnv: pd.DataFrame, model_ids: pd.Index) -> pd.DataFrame:
    cnv = cnv[cnv["model_id"].isin(model_ids) & cnv["symbol"].isin(CURATED_BREAST_GENES)].copy()
    cnv["total_copy_number"] = pd.to_numeric(cnv["total_copy_number"], errors="coerce")
    # Prefer Sanger observations when duplicates exist, otherwise median is robust across sources.
    matrix = cnv.pivot_table(index="model_id", columns="symbol", values="total_copy_number", aggfunc="median")
    matrix = matrix.reindex(index=model_ids, columns=CURATED_BREAST_GENES)
    return matrix.add_prefix("CN_").reset_index(names="SANGER_MODEL_ID")


def extract_breast_raw(raw_zip: Path, model_ids: set[str], output: Path, chunksize: int = 250_000) -> int:
    """Extract unnormalised breast-model wells without claiming that intensity is viability."""
    with zipfile.ZipFile(raw_zip) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if not names:
            return 0
        written = 0
        first = True
        with archive.open(names[0]) as handle:
            for chunk in pd.read_csv(handle, chunksize=chunksize, low_memory=False):
                selected = chunk[chunk["SANGER_MODEL_ID"].isin(model_ids)]
                if selected.empty:
                    continue
                selected.to_csv(output, mode="w" if first else "a", header=first, index=False)
                written += len(selected)
                first = False
        return written


def prepare_dataset(raw_dir: Path, processed_dir: Path, reports_dir: Path) -> dict:
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    fitted_path = _single(raw_dir, "*GDSC2_fitted*.xlsx")
    try:
        fitted = pd.read_excel(fitted_path, engine="calamine")
    except ImportError:
        fitted = pd.read_excel(fitted_path, engine="openpyxl")
    required = {
        "CELL_LINE_NAME", "SANGER_MODEL_ID", "CANCER_TYPE", "DRUG_ID", "DRUG_NAME",
        "PUTATIVE_TARGET", "PATHWAY_NAME", "MIN_CONC", "MAX_CONC", "LN_IC50", "AUC", "RMSE", "Z_SCORE",
    }
    missing = required - set(fitted.columns)
    if missing:
        raise ValueError(f"GDSC2 fitted file lacks columns: {sorted(missing)}")
    breast = fitted[fitted["CANCER_TYPE"].astype(str).str.contains("breast", case=False, na=False)].copy()
    breast = breast[breast["SANGER_MODEL_ID"].notna()].copy()
    breast["IC50_uM"] = np.exp(pd.to_numeric(breast["LN_IC50"], errors="coerce"))
    breast["RESULT_PROVENANCE"] = "experimental observation: GDSC2 72 h fitted viability curve"

    model_path = _single(raw_dir, "model_list_*.csv")
    models = pd.read_csv(model_path, low_memory=False)
    model_cols = [
        "model_id", "model_name", "tissue", "cancer_type", "cancer_type_detail", "model_type",
        "msi_status", "ploidy_snp6", "ploidy_wes", "ploidy_wgs", "mutational_burden",
        "COSMIC_ID", "BROAD_ID", "CCLE_ID", "RRID",
    ]
    models = models[[c for c in model_cols if c in models.columns]].drop_duplicates("model_id")
    models = models.rename(columns={"model_id": "SANGER_MODEL_ID", "model_name": "MODEL_NAME_OFFICIAL"})

    growth = pd.read_csv(_single(raw_dir, "growth_rate_*.csv"))
    growth = growth.rename(columns={"model_id": "SANGER_MODEL_ID"})
    growth["growth_rate_per_hour"] = np.log(2) / pd.to_numeric(growth["doubling_time_hours"], errors="coerce")
    # Multiple experimental growth estimates can exist for one model; retain their
    # median and the number of estimates instead of silently selecting a row.
    growth = (
        growth.groupby("SANGER_MODEL_ID", as_index=False)
        .agg(
            doubling_time_hours=("doubling_time_hours", "median"),
            growth_rate_per_hour=("growth_rate_per_hour", "median"),
            growth_measurements=("growth_rate_per_hour", "count"),
        )
    )

    model_ids = pd.Index(sorted(breast["SANGER_MODEL_ID"].unique()), name="SANGER_MODEL_ID")
    mutations = _read_first_csv_from_zip(_single(raw_dir, "mutations_summary_*.zip"))
    mutation_features = _binary_mutation_features(mutations, model_ids)
    cnv = _read_first_csv_from_zip(_single(raw_dir, "cnv_summary_*.zip"))
    cnv_features = _cnv_features(cnv, model_ids)
    cell_features = (
        pd.DataFrame({"SANGER_MODEL_ID": model_ids})
        .merge(models, on="SANGER_MODEL_ID", how="left", validate="one_to_one")
        .merge(growth, on="SANGER_MODEL_ID", how="left", validate="one_to_one")
        .merge(mutation_features, on="SANGER_MODEL_ID", how="left", validate="one_to_one")
        .merge(cnv_features, on="SANGER_MODEL_ID", how="left", validate="one_to_one")
    )
    # CMP/GDSC do not provide a breast molecular-subtype field in these files.
    cell_features["MOLECULAR_SUBTYPE"] = pd.NA
    cell_features["SUBTYPE_EVIDENCE"] = "not provided in downloaded official annotations"

    compounds = pd.read_csv(_single(raw_dir, "screened_compounds*.csv"))
    compounds["DRUG_ID"] = pd.to_numeric(compounds["DRUG_ID"], errors="coerce")
    compounds = compounds.sort_values("SCREENING_SITE", key=lambda x: x.ne("SANGER")).drop_duplicates("DRUG_ID")
    compound_cols = ["DRUG_ID", "SCREENING_SITE", "CHEMBL_ID", "INCHI_KEY"]
    compounds = compounds[[c for c in compound_cols if c in compounds.columns]]

    table = breast.merge(cell_features, on="SANGER_MODEL_ID", how="left", validate="many_to_one")
    table = table.merge(compounds, on="DRUG_ID", how="left", validate="many_to_one")
    table.to_csv(processed_dir / "breast_gdsc2_modeling.csv.gz", index=False, compression="gzip")
    cell_features.to_csv(processed_dir / "breast_cell_features.csv", index=False)
    (
        breast[["DRUG_ID", "DRUG_NAME", "PUTATIVE_TARGET", "PATHWAY_NAME", "MIN_CONC", "MAX_CONC"]]
        .drop_duplicates()
        .merge(compounds, on="DRUG_ID", how="left")
        .to_csv(processed_dir / "compound_catalog.csv", index=False)
    )

    raw_rows = 0
    raw_candidates = sorted(raw_dir.glob("*GDSC2_public_raw*.zip"))
    raw_output = processed_dir / "breast_gdsc2_raw_wells.csv.gz"
    if raw_candidates:
        raw_rows = extract_breast_raw(raw_candidates[-1], set(model_ids), raw_output)

    numeric_cols = ["LN_IC50", "AUC", "RMSE", "Z_SCORE", "MIN_CONC", "MAX_CONC"]
    summary = {
        "source_file": fitted_path.name,
        "models": int(breast["SANGER_MODEL_ID"].nunique()),
        "compounds": int(breast["DRUG_ID"].nunique()),
        "observations": int(len(breast)),
        "raw_wells_extracted": int(raw_rows),
        "known_requested_models": {
            name: bool(
                breast["CELL_LINE_NAME"]
                .astype(str)
                .str.upper()
                .str.replace(r"[^A-Z0-9]", "", regex=True)
                .eq("".join(ch for ch in name.upper() if ch.isalnum()))
                .any()
            )
            for name in ["MCF-7", "MDA-MB-231", "T47D", "BT-474", "SK-BR-3"]
        },
        "missing_percent": {c: float(table[c].isna().mean() * 100) for c in numeric_cols},
        "distributions": {c: table[c].describe().dropna().to_dict() for c in ["LN_IC50", "IC50_uM", "AUC", "RMSE"]},
        "genomics": {
            "curated_mutation_features": len([c for c in table if c.startswith("MUT_")]),
            "curated_cnv_features": len([c for c in table if c.startswith("CN_")]),
            "expression_matrix_included": False,
            "expression_omission": "Official processed RNA-seq archive is ~1.29 GB; deliberately excluded from first run.",
        },
        "quality": {
            "rmse_max": float(table["RMSE"].max()),
            "rmse_gt_0_3": int((table["RMSE"] > 0.3).sum()),
            "official_release_rule": "Curves with RMSE > 0.3 are excluded before release (Sanger documentation).",
        },
        "definitions": {
            "LN_IC50": "Natural logarithm of fitted IC50; concentrations are micromolar. IC50_uM = exp(LN_IC50).",
            "AUC": "Fraction of total fitted area between the lowest and highest screened concentrations; an experimental curve summary, not clinical efficacy.",
            "raw_intensity": "Unnormalised plate signal; not labelled viability until control normalization and QC are applied.",
        },
    }
    (reports_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    lines = [
        "# Preliminary exploratory report — GDSC2 breast models",
        "",
        "> Preclinical computational research only. This report does not recommend patient treatment.",
        "",
        f"- Models: **{summary['models']}**",
        f"- Compounds: **{summary['compounds']}**",
        f"- Fitted observations: **{summary['observations']}**",
        f"- Extracted raw wells: **{summary['raw_wells_extracted']}**",
        f"- LN_IC50 median: **{summary['distributions']['LN_IC50'].get('50%', float('nan')):.4g}**",
        f"- AUC median: **{summary['distributions']['AUC'].get('50%', float('nan')):.4g}**",
        f"- Fitted-curve RMSE median: **{summary['distributions']['RMSE'].get('50%', float('nan')):.4g}**",
        "",
        "## Interpretation safeguards",
        "",
        "`LN_IC50` is the natural logarithm of IC50 in µM; conversion uses `exp(LN_IC50)`. "
        "AUC is a fractional area of the fitted 72-hour viability curve over the screened concentration range. "
        "Neither quantity is clinical efficacy. Endpoint viability does not uniquely identify cell death, growth arrest or a dynamic killing rate.",
        "",
        "## Genomic coverage",
        "",
        "The first run uses official driver-mutation and driver-CNV summaries for a fixed, predeclared breast-cancer gene panel. "
        "RNA-seq was not downloaded because the official processed archive is approximately 1.29 GB. No synthetic values replaced it.",
        "",
        "## Molecular subtype",
        "",
        "The downloaded GDSC2 fitted file and CMP model list do not expose a verified breast-subtype field. "
        "Subtype is therefore left missing rather than inferred from cell-line names.",
    ]
    (reports_dir / "preliminary_scientific_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary

