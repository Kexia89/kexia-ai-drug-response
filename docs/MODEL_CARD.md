# Ficha de modelos

## Modelo principal

- **Tarea:** regresión de `LN_IC50 = ln(IC50 en µM)` para pares línea celular–compuesto.
- **Algoritmo:** `HistGradientBoostingRegressor`, elegido por MAE de calibración frente a Ridge y Random Forest.
- **Datos:** 13,388 observaciones GDSC2 de 52 líneas celulares de cáncer de mama y 295 compuestos.
- **Entradas:** rango de concentración, ploidía, carga mutacional, crecimiento, mutaciones/CNV curadas y anotaciones categóricas del compuesto/ensayo.
- **Preprocesamiento:** imputación, estandarización y one-hot dentro de `sklearn.Pipeline`; se ajusta sólo con entrenamiento.
- **Artefacto generado localmente:** `artifacts/models/gdsc2_breast_ln_ic50.joblib` (no redistribuido en esta entrega).
- **Uso previsto:** generación de hipótesis computacionales preclínicas dentro del dominio representado.

El rendimiento se degrada de R² 0.7773 para pares nuevos de entidades conocidas a R² 0.3211 cuando célula y compuesto son simultáneamente nuevos. Por ello no debe usarse como predictor universal.

## Candidato combinado de Fase 2

- **Artefacto generado localmente:** `artifacts/models/gdsc2_breast_phase2_combined_candidate.joblib` (no redistribuido en esta entrega).
- **Características añadidas:** RNA-seq `log2(TPM+1)`, PCA ajustado sólo en células de entrenamiento, mutación/CNV ampliadas y descriptores químicos verificados.
- **Estado:** experimental; no reemplaza el modelo principal.

En tres repeticiones de células no vistas, el modelo combinado obtuvo R² `0.6757 ± 0.0424`, el modelo actual `0.6812 ± 0.0357` y una referencia simple por media del compuesto `0.7475 ± 0.0428`. No hay mejora robusta demostrada.

## Red neuronal two-tower

`src/kexia_cancer/models/deep.py` implementa codificadores separados de célula y compuesto, pérdida Huber, AdamW, dropout, weight decay, early stopping y semillas reproducibles. No existe un checkpoint verificado de la primera corrida y no se reporta rendimiento de producción.

## Usos fuera de alcance

- decisiones de tratamiento o diagnóstico;
- estimación de respuesta de pacientes;
- selección automática de dosis;
- extrapolación a tumores, toxicidad o farmacocinética clínica;
- sustitución de validación experimental independiente.

## Incertidumbre y sesgos

- Los intervalos conformales son marginales y pierden cobertura bajo cambio de dominio.
- La muestra de 52 células limita estabilidad molecular.
- Estructuras químicas exactas están disponibles para 210/295 compuestos, lo que puede introducir sesgo de disponibilidad.
- `MIN_CONC` y `MAX_CONC` son predictivas pero reflejan diseño experimental y no necesariamente biología causal.
- Los artefactos `joblib` pueden ejecutar código al cargarse: cárgalos sólo desde una copia confiable y con versiones compatibles.

## Trazabilidad

La semilla por defecto es 42. Las métricas agregadas, intervalos, tamaños de split y auditorías están en `artifacts/reports/`. Las tablas fila por fila se regeneran localmente. La procedencia de datos está en `data/metadata/download_manifest.json`.

