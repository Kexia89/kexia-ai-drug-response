#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "artifacts" / "reports"
METADATA = ROOT / "data" / "metadata"


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    view = frame[columns].copy()
    for column in view.select_dtypes(include="number"):
        view[column] = view[column].map(lambda value: f"{value:.4f}")
    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in view.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *rows])


def main() -> None:
    comparison = pd.read_csv(REPORTS / "phase2_model_comparison.csv")
    repeated = pd.read_csv(REPORTS / "phase2_repeated_group_validation_summary.csv")
    scaffold = pd.read_csv(REPORTS / "phase2_scaffold_evaluation.csv")
    intervals = pd.read_csv(REPORTS / "phase2_interval_calibration.csv")
    audit = json.loads((REPORTS / "phase2_dataset_audit.json").read_text(encoding="utf-8"))
    math = json.loads((REPORTS / "phase2_mathematical_audit.json").read_text(encoding="utf-8"))
    temporal = json.loads((REPORTS / "phase2_temporal_validation.json").read_text(encoding="utf-8"))
    expression = json.loads((METADATA / "breast_expression_metadata.json").read_text(encoding="utf-8"))
    chemical = json.loads((METADATA / "chemical_structure_manifest.json").read_text(encoding="utf-8"))

    b = comparison[(comparison["protocol"] == "B_unseen_cells") & ~comparison["variant"].str.startswith("ablate_")]
    b = b[b["variant"] != "chemical_descriptors"]
    ablations = comparison[comparison["variant"].str.startswith("ablate_")].copy()
    baseline_by_protocol = comparison[comparison["variant"] == "current"].set_index("protocol")["r2"]
    ablations["delta_r2_vs_audit_baseline"] = ablations.apply(
        lambda row: row["r2"] - baseline_by_protocol[row["protocol"]], axis=1
    )
    repeated_cells = repeated[repeated["protocol"] == "repeated_unseen_cells"]
    repeated_drugs = repeated[repeated["protocol"] == "repeated_unseen_chemical_identities"]
    best_cells = repeated_cells[repeated_cells["variant"] != "group_mean_reference"].sort_values("r2_mean", ascending=False).iloc[0]
    best_cells_overall = repeated_cells.sort_values("r2_mean", ascending=False).iloc[0]
    best_scaffold = scaffold.sort_values("r2", ascending=False).iloc[0]

    content = f"""# Kexia Cancer Research AI — Informe de auditoría científica Fase 2

Fecha de ejecución: 17 de noviembre de 2026. Uso exclusivo de investigación preclínica computacional.

## Dictamen ejecutivo

- La separación original por `SANGER_MODEL_ID` fue estricta para células nuevas y el preprocesamiento se ajustó dentro de cada entrenamiento. No se encontraron duplicados exactos, pares célula–`DRUG_ID` duplicados ni `NLME_CURVE_ID` duplicados.
- La separación original por `DRUG_ID` no equivalía a separación por identidad química: cuatro InChIKeys aparecieron en entrenamiento y prueba con identificadores distintos. Fase 2 usa InChIKey/ChEMBL verificado como grupo.
- La selección y calibración conformal originales usaban filas aleatorias dentro del conjunto de desarrollo. Esto no contamina el test final, pero sí es optimista para cambios de célula/compuesto y no garantiza cobertura bajo esos desplazamientos.
- `LN_IC50` está correctamente tratado como `ln(IC50 en µM)` y la conversión documentada es `IC50_uM = exp(LN_IC50)`.
- En el test fijo de células nuevas, RNA-seq alcanzó R² {b.loc[b['variant'].eq('expression_pca'), 'r2'].iloc[0]:.4f}; el combinado logró el menor MAE ({b.loc[b['variant'].eq('combined'), 'mae'].iloc[0]:.4f}). La conclusión definitiva se basa en las repeticiones por grupos, no en una única partición.
- En validación repetida de células no vistas, la referencia simple por media del compuesto fue superior (R² {best_cells_overall['r2_mean']:.4f} ± {best_cells_overall['r2_sd']:.4f}). Entre modelos aprendidos, `{best_cells['variant']}` fue el mejor con R² {best_cells['r2_mean']:.4f} ± {best_cells['r2_sd']:.4f}. La combinación molecular no demostró mejora robusta.
- En andamiajes químicos no vistos, `{best_scaffold['variant']}` obtuvo R² {best_scaffold['r2']:.4f}; los intervalos bootstrap por andamiaje se solapan, por lo que la evidencia de mejora estructural es prometedora pero no concluyente.

## 1. Integridad y fugas

- Filas: {audit['rows']:,}; duplicados exactos: {audit['exact_duplicate_rows']}; pares duplicados: {audit['duplicate_cell_drug_rows']}; curvas duplicadas: {audit['duplicate_curve_ids']}.
- Nombres con múltiples `DRUG_ID`: {len(audit['drug_names_with_multiple_ids'])}. Identidades químicas con múltiples IDs: {len(audit['chemical_identities_with_multiple_drug_ids'])}.
- Solapamientos químicos del split original: {', '.join(audit['original_compound_split_identity_overlap'])}.
- La imputación, codificación y escalado originales están dentro de `Pipeline`/`ColumnTransformer` y se ajustan sólo con filas de entrenamiento.
- Para expresión, selección por varianza, escalado y PCA se vuelven a ajustar usando únicamente células de entrenamiento. Para mutación/CNV ampliadas, imputación y `SelectKBest` están dentro del pipeline.

## 2. Test fijo de células no vistas

El estimador estandarizado de ablación usa el mismo esquema HistGradientBoosting y 120 iteraciones máximas para todas las variantes. Es una réplica de auditoría comparable internamente; no sustituye las métricas históricas del artefacto original de 300 iteraciones.

{markdown_table(b, ['variant', 'mae', 'rmse', 'r2', 'pearson', 'spearman', 'test_rows'])}

## 3. Ablaciones

{markdown_table(ablations, ['protocol', 'variant', 'mae', 'rmse', 'r2', 'delta_r2_vs_audit_baseline'])}

`MIN_CONC` y `MAX_CONC` aportan información predictiva importante, sobre todo juntas para compuestos nuevos. Esto puede reflejar el dominio experimental de la curva y no biología causal. `PUTATIVE_TARGET` aporta mucho en células nuevas, pero poco en identidades químicas nuevas en este split. `SCREENING_SITE` no cambió el resultado porque no presenta variación útil en este subconjunto.

## 4. Validación repetida por grupos

{markdown_table(repeated, ['protocol', 'variant', 'repeats', 'mae_mean', 'mae_sd', 'rmse_mean', 'rmse_sd', 'r2_mean', 'r2_sd'])}

Los tests independientes fijos no se utilizaron para elegir hiperparámetros. Las tres repeticiones se ejecutaron dentro del conjunto de desarrollo y reservaron grupos completos.

## 5. Estructuras químicas verificadas

- Estructuras resueltas: {chemical.get('resolved_structures', 'N/D')}; andamiajes Murcko: {chemical.get('unique_scaffolds', 'N/D')}.
- No se inventaron SMILES. Los conflictos de InChIKey se rechazaron y permanecen en el registro de fallos.

{markdown_table(scaffold, ['variant', 'mae', 'rmse', 'r2', 'pearson', 'spearman', 'test_rows'])}

## 6. Calibración de intervalos

{markdown_table(intervals, ['protocol', 'calibration_group', 'half_width', 'empirical_coverage', 'mean_interval_width', 'test_groups'])}

La cobertura fue inferior al 95% en células, identidades químicas y especialmente en el escenario doblemente no visto. Los intervalos son marginales y no deben interpretarse como incertidumbre individual confiable.

## 7. Matrices moleculares

- RNA-seq: {expression['available_models']}/{expression['requested_models']} modelos, {expression['genes']:,} genes, escala oficial `log2(TPM+1)`.
- Modelo ausente: {', '.join(expression['missing_models'])}. No se fabricó expresión; se conserva como faltante.
- Duplicados de característica: {expression['duplicate_feature_ids']} identificador, agregado por mediana y documentado.
- Matriz ampliada: 161 genes de mutación conductora y 769 características CNV, unidas sólo por `SANGER_MODEL_ID`.
- No existe subtipo molecular oficial verificable para estas 52 líneas en los archivos descargados; no se realizó análisis por subtipo inventado.

## 8. Auditoría matemática

- Pozos normalizados >1: {100 * math['normalization']['fraction_above_one']:.2f}%; no se recortaron.
- Correlación Spearman entre fracción >1 y rango dinámico: {math['normalization']['spearman_fraction_above_one']['dynamic_range']:.3f}.
- Curvas Hill auditadas: {math['hill_endpoint_models']['curves_audited']}; EC50 dentro del rango: {100 * math['hill_endpoint_models']['fraction_ec50_inside_tested_range']:.1f}%.
- MAE de extrapolación a la máxima concentración: {math['hill_endpoint_models']['extrapolation_mae']:.4f}.
- Condición mediana de covarianza: {math['hill_endpoint_models']['covariance_condition_median']:.3g}; SD mediana leave-one-out de log10(EC50): {math['hill_endpoint_models']['leave_one_out_log10_ec50_sd_median']:.3f}.

## 9. Evidencia temporal independiente

Se incorporaron conteos longitudinales públicos de BT-474, MCF7 y MDA-MB-231. La prueba cronológica 70/30 del crecimiento logístico sin intervención produjo NRMSE mediana {temporal['median_test_nrmse']:.3f}. Es un rendimiento insuficiente para declarar validado el crecimiento logístico fuera de muestra. Las dinámicas farmacológicas no se validaron: los experimentos usan pulsos, dosis seriales y retardos de muerte/recuperación ausentes en la ODE actual.

## 10. Problemas científicos abiertos

1. Sólo 52 células limitan la estabilidad de PCA, selección de genes y análisis por subtipo.
2. Concentraciones y anotaciones del ensayo dominan parte de la predicción; su utilidad no equivale a mecanismo molecular causal.
3. Sólo 210/295 compuestos tienen estructura exacta resuelta; el análisis químico puede sufrir sesgo de disponibilidad.
4. Los intervalos no alcanzan cobertura nominal bajo desplazamientos de dominio.
5. La función Hill de viabilidad a 72 h no identifica una tasa de muerte ni parámetros dinámicos.
6. El modelo temporal actual debe incorporar pulsos, arresto y retardos antes de confrontarse con los datos longitudinales descargados.

## Reproducibilidad

Ejecute `prepare_phase2_molecular.py`, `prepare_compound_structures.py`, `run_phase2_experiments.py`, `run_phase2_repeated_validation.py`, `run_phase2_scaffold_evaluation.py`, `audit_mathematical_models.py`, `prepare_temporal_validation.py`, `train_phase2_candidate.py` y finalmente `pytest`.
"""
    (REPORTS / "phase2_model_audit.md").write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
