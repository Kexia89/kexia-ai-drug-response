# Kexia Cancer Research AI — Informe de auditoría científica Fase 2

Fecha de ejecución: 17 de noviembre de 2026. Uso exclusivo de investigación preclínica computacional.

## Dictamen ejecutivo

- La separación original por `SANGER_MODEL_ID` fue estricta para células nuevas y el preprocesamiento se ajustó dentro de cada entrenamiento. No se encontraron duplicados exactos, pares célula–`DRUG_ID` duplicados ni `NLME_CURVE_ID` duplicados.
- La separación original por `DRUG_ID` no equivalía a separación por identidad química: cuatro InChIKeys aparecieron en entrenamiento y prueba con identificadores distintos. Fase 2 usa InChIKey/ChEMBL verificado como grupo.
- La selección y calibración conformal originales usaban filas aleatorias dentro del conjunto de desarrollo. Esto no contamina el test final, pero sí es optimista para cambios de célula/compuesto y no garantiza cobertura bajo esos desplazamientos.
- `LN_IC50` está correctamente tratado como `ln(IC50 en µM)` y la conversión documentada es `IC50_uM = exp(LN_IC50)`.
- En el test fijo de células nuevas, RNA-seq alcanzó R² 0.6351; el combinado logró el menor MAE (1.2504). La conclusión definitiva se basa en las repeticiones por grupos, no en una única partición.
- En validación repetida de células no vistas, la referencia simple por media del compuesto fue superior (R² 0.7475 ± 0.0428). Entre modelos aprendidos, `current` fue el mejor con R² 0.6812 ± 0.0357. La combinación molecular no demostró mejora robusta.
- En andamiajes químicos no vistos, `chemical_descriptors` obtuvo R² 0.3842; los intervalos bootstrap por andamiaje se solapan, por lo que la evidencia de mejora estructural es prometedora pero no concluyente.

## 1. Integridad y fugas

- Filas: 13,388; duplicados exactos: 0; pares duplicados: 0; curvas duplicadas: 0.
- Nombres con múltiples `DRUG_ID`: 9. Identidades químicas con múltiples IDs: 9.
- Solapamientos químicos del split original: INCHIKEY:CYOHGALHFOKKQC-UHFFFAOYSA-N, INCHIKEY:KSERXGMCDHOLSS-LJQANCHMSA-N, INCHIKEY:RJURFGZVJUQBHK-IIXSONLDSA-N, INCHIKEY:ZDZOTLJHXYCWBA-VCVYQWHSSA-N.
- La imputación, codificación y escalado originales están dentro de `Pipeline`/`ColumnTransformer` y se ajustan sólo con filas de entrenamiento.
- Para expresión, selección por varianza, escalado y PCA se vuelven a ajustar usando únicamente células de entrenamiento. Para mutación/CNV ampliadas, imputación y `SelectKBest` están dentro del pipeline.

## 2. Test fijo de células no vistas

El estimador estandarizado de ablación usa el mismo esquema HistGradientBoosting y 120 iteraciones máximas para todas las variantes. Es una réplica de auditoría comparable internamente; no sustituye las métricas históricas del artefacto original de 300 iteraciones.

| variant | mae | rmse | r2 | pearson | spearman | test_rows |
|---|---|---|---|---|---|---|
| current | 1.3092 | 1.7115 | 0.6044 | 0.7843 | 0.7098 | 2131.0000 |
| expanded_molecular | 1.2661 | 1.6522 | 0.6313 | 0.7978 | 0.7334 | 2131.0000 |
| expression_pca | 1.2644 | 1.6438 | 0.6351 | 0.8034 | 0.7406 | 2131.0000 |
| combined | 1.2504 | 1.6551 | 0.6300 | 0.7959 | 0.7257 | 2131.0000 |

## 3. Ablaciones

| protocol | variant | mae | rmse | r2 | delta_r2_vs_audit_baseline |
|---|---|---|---|---|---|
| B_unseen_cells | ablate_MIN_CONC | 1.3370 | 1.7416 | 0.5903 | -0.0140 |
| B_unseen_cells | ablate_MAX_CONC | 1.3310 | 1.7341 | 0.5939 | -0.0105 |
| B_unseen_cells | ablate_SCREENING_SITE | 1.3092 | 1.7115 | 0.6044 | 0.0000 |
| B_unseen_cells | ablate_PUTATIVE_TARGET | 1.5081 | 1.9364 | 0.4936 | -0.1108 |
| B_unseen_cells | ablate_PATHWAY_NAME | 1.3267 | 1.7169 | 0.6019 | -0.0025 |
| B_unseen_cells | ablate_MIN_CONC+MAX_CONC | 1.3915 | 1.8098 | 0.5576 | -0.0468 |
| B_unseen_cells | ablate_SCREENING_SITE+PUTATIVE_TARGET+PATHWAY_NAME | 1.6107 | 2.0502 | 0.4323 | -0.1721 |
| C_unseen_chemical_identities | ablate_MIN_CONC | 1.5752 | 2.0955 | 0.4097 | -0.0039 |
| C_unseen_chemical_identities | ablate_MAX_CONC | 1.5882 | 2.0975 | 0.4085 | -0.0050 |
| C_unseen_chemical_identities | ablate_SCREENING_SITE | 1.5812 | 2.0886 | 0.4136 | 0.0000 |
| C_unseen_chemical_identities | ablate_PUTATIVE_TARGET | 1.5935 | 2.0887 | 0.4135 | -0.0000 |
| C_unseen_chemical_identities | ablate_PATHWAY_NAME | 1.5840 | 2.0912 | 0.4121 | -0.0015 |
| C_unseen_chemical_identities | ablate_MIN_CONC+MAX_CONC | 1.9913 | 2.7094 | 0.0131 | -0.4004 |
| C_unseen_chemical_identities | ablate_SCREENING_SITE+PUTATIVE_TARGET+PATHWAY_NAME | 1.6344 | 2.1332 | 0.3882 | -0.0253 |

`MIN_CONC` y `MAX_CONC` aportan información predictiva importante, sobre todo juntas para compuestos nuevos. Esto puede reflejar el dominio experimental de la curva y no biología causal. `PUTATIVE_TARGET` aporta mucho en células nuevas, pero poco en identidades químicas nuevas en este split. `SCREENING_SITE` no cambió el resultado porque no presenta variación útil en este subconjunto.

## 4. Validación repetida por grupos

| protocol | variant | repeats | mae_mean | mae_sd | rmse_mean | rmse_sd | r2_mean | r2_sd |
|---|---|---|---|---|---|---|---|---|
| repeated_unseen_cells | combined | 3.0000 | 1.1826 | 0.1834 | 1.5300 | 0.1955 | 0.6757 | 0.0424 |
| repeated_unseen_cells | current | 3.0000 | 1.1790 | 0.1624 | 1.5169 | 0.1808 | 0.6812 | 0.0357 |
| repeated_unseen_cells | group_mean_reference | 3.0000 | 1.0325 | 0.1770 | 1.3497 | 0.1985 | 0.7475 | 0.0428 |
| repeated_unseen_chemical_identities | chemical_descriptors | 3.0000 | 1.5981 | 0.1218 | 2.0390 | 0.1638 | 0.4758 | 0.0259 |
| repeated_unseen_chemical_identities | current | 3.0000 | 1.5829 | 0.2314 | 2.0468 | 0.2974 | 0.4720 | 0.0923 |
| repeated_unseen_chemical_identities | group_mean_reference | 3.0000 | 2.1126 | 0.1258 | 2.7504 | 0.1845 | 0.0456 | 0.0312 |

Los tests independientes fijos no se utilizaron para elegir hiperparámetros. Las tres repeticiones se ejecutaron dentro del conjunto de desarrollo y reservaron grupos completos.

## 5. Estructuras químicas verificadas

- Estructuras resueltas: 210; andamiajes Murcko: 191.
- No se inventaron SMILES. Los conflictos de InChIKey se rechazaron y permanecen en el registro de fallos.

| variant | mae | rmse | r2 | pearson | spearman | test_rows |
|---|---|---|---|---|---|---|
| current | 1.6258 | 2.2173 | 0.2916 | 0.5725 | 0.6444 | 1940.0000 |
| chemical_descriptors | 1.5928 | 2.0673 | 0.3842 | 0.6766 | 0.7005 | 1940.0000 |

## 6. Calibración de intervalos

| protocol | calibration_group | half_width | empirical_coverage | mean_interval_width | test_groups |
|---|---|---|---|---|---|
| A_known_entities_new_pairs | SANGER_MODEL_ID | 3.3616 | 0.9765 | 6.7232 | 52.0000 |
| B_unseen_cells | SANGER_MODEL_ID | 2.9503 | 0.9033 | 5.9006 | 10.0000 |
| C_unseen_chemical_identities | CHEMICAL_IDENTITY | 3.5035 | 0.9202 | 7.0070 | 57.0000 |
| D_unseen_cells_and_chemical_identities | SANGER_MODEL_ID | 2.8555 | 0.7901 | 5.7111 | 10.0000 |

La cobertura fue inferior al 95% en células, identidades químicas y especialmente en el escenario doblemente no visto. Los intervalos son marginales y no deben interpretarse como incertidumbre individual confiable.

## 7. Matrices moleculares

- RNA-seq: 51/52 modelos, 41,144 genes, escala oficial `log2(TPM+1)`.
- Modelo ausente: SIDM01261. No se fabricó expresión; se conserva como faltante.
- Duplicados de característica: 1 identificador, agregado por mediana y documentado.
- Matriz ampliada: 161 genes de mutación conductora y 769 características CNV, unidas sólo por `SANGER_MODEL_ID`.
- No existe subtipo molecular oficial verificable para estas 52 líneas en los archivos descargados; no se realizó análisis por subtipo inventado.

## 8. Auditoría matemática

- Pozos normalizados >1: 31.23%; no se recortaron.
- Correlación Spearman entre fracción >1 y rango dinámico: 0.031.
- Curvas Hill auditadas: 120; EC50 dentro del rango: 70.8%.
- MAE de extrapolación a la máxima concentración: 0.1811.
- Condición mediana de covarianza: 6.09e+06; SD mediana leave-one-out de log10(EC50): 0.383.

## 9. Evidencia temporal independiente

Se incorporaron conteos longitudinales públicos de BT-474, MCF7 y MDA-MB-231. La prueba cronológica 70/30 del crecimiento logístico sin intervención produjo NRMSE mediana 0.669. Es un rendimiento insuficiente para declarar validado el crecimiento logístico fuera de muestra. Las dinámicas farmacológicas no se validaron: los experimentos usan pulsos, dosis seriales y retardos de muerte/recuperación ausentes en la ODE actual.

## 10. Problemas científicos abiertos

1. Sólo 52 células limitan la estabilidad de PCA, selección de genes y análisis por subtipo.
2. Concentraciones y anotaciones del ensayo dominan parte de la predicción; su utilidad no equivale a mecanismo molecular causal.
3. Sólo 210/295 compuestos tienen estructura exacta resuelta; el análisis químico puede sufrir sesgo de disponibilidad.
4. Los intervalos no alcanzan cobertura nominal bajo desplazamientos de dominio.
5. La función Hill de viabilidad a 72 h no identifica una tasa de muerte ni parámetros dinámicos.
6. El modelo temporal actual debe incorporar pulsos, arresto y retardos antes de confrontarse con los datos longitudinales descargados.

## Reproducibilidad

Ejecute `prepare_phase2_molecular.py`, `prepare_compound_structures.py`, `run_phase2_experiments.py`, `run_phase2_repeated_validation.py`, `run_phase2_scaffold_evaluation.py`, `audit_mathematical_models.py`, `prepare_temporal_validation.py`, `train_phase2_candidate.py` y finalmente `pytest`.
