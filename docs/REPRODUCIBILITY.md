# Reproducibilidad

## Entorno

- Python 3.11 (verificado con 3.11.9).
- `requirements.txt`: rangos compatibles.
- `requirements-lock.txt`: versiones observadas en la corrida local verificada.
- Semilla por defecto: 42, en `configs/default.yaml`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e .
python -m pytest
```

Si el pin exacto de PyTorch no está disponible para su plataforma, use `requirements.txt` o instale la compilación CPU/CUDA indicada por PyTorch y luego instale el resto.

## Pipeline completo

| Orden | Comando | Producto principal |
|---:|---|---|
| 1 | `python scripts/download_datasets.py` | `data/raw/`, manifiesto SHA-256 |
| 2 | `python scripts/prepare_breast_cancer.py` | tabla GDSC2 y características celulares |
| 3 | `python scripts/prepare_phase2_molecular.py` | expresión y molecular ampliado |
| 4 | `python scripts/prepare_compound_structures.py` | estructuras verificadas y RDKit |
| 5 | `python scripts/train_models.py` | modelo base `.joblib` |
| 6 | `python scripts/evaluate_models.py` | métricas y predicciones base |
| 7 | `python scripts/validate_mathematical_models.py` | curvas endpoint retenidas |
| 8 | `python scripts/run_phase2_experiments.py` | comparación molecular/química |
| 9 | `python scripts/run_phase2_repeated_validation.py` | validación repetida por grupos |
| 10 | `python scripts/run_phase2_scaffold_evaluation.py` | holdout de Murcko |
| 11 | `python scripts/audit_mathematical_models.py` | identificabilidad/extrapolación |
| 12 | `python scripts/prepare_temporal_validation.py` | prueba cronológica logística |
| 13 | `python scripts/train_phase2_candidate.py` | candidato combinado |
| 14 | `python scripts/run_simulations.py` | simulaciones y sensibilidad |

Los scripts pueden tardar y requerir varios GB. Para inspeccionar la resolución del catálogo sin descargar, use:

```powershell
python scripts/download_datasets.py --list
```

## Verificación mínima sin descargar datos

```powershell
python -m pytest
```

Las 16 pruebas cubren descubrimiento de descargas, splits, humo del modelo, Fase 2 y simulaciones matemáticas.

## Aplicación

Después de preparar los datos o usando una copia local ya preparada:

```powershell
streamlit run app/main.py
```

## Qué debe coincidir

El paralelismo, BLAS y hardware pueden producir diferencias pequeñas de punto flotante. Deben conservarse la estructura de splits, conteos de filas/grupos, orden de magnitud de métricas y conclusiones cualitativas. Use los JSON/CSV en `artifacts/reports/` como referencia y no sólo una métrica aislada.

## Artefactos entrenados

Los `.joblib` no se redistribuyen por prudencia respecto de los términos upstream. `train_models.py` y `train_phase2_candidate.py` los generan localmente. No cargue archivos `joblib` no confiables.

