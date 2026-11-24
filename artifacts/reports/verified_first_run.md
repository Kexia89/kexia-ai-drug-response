# Kexia Cancer Research AI — verified first run

**Execution date:** 13 November 2026  
**Scope:** public GDSC2 breast-cancer cell lines; preclinical computation only.

## Data and quality

- 52 breast models, 295 compounds and 13,388 fitted GDSC2 observations.
- 2,192,303 raw breast-model rows retained; 252,415 single-agent wells normalized.
- 2,389/2,389 represented plates had a positive blank-to-vehicle dynamic range.
- Release curve RMSE: median 0.07830, maximum 0.29961; no value exceeded 0.3.
- MCF7, MDA-MB-231, T47D and BT-474 were found; SK-BR-3 was not found.
- No verified breast-subtype field was present; no subtype was inferred.
- Processed RNA-seq was deliberately omitted (~1.29 GB). No synthetic replacement was used.

## Predictive model

HistGradientBoosting was selected over Ridge and Random Forest on an internal
calibration partition. Inputs combine fixed-panel driver mutation/CNV values,
growth/model measurements, concentration range, target and pathway. `LN_IC50`
is the target.

| Evaluation | MAE | RMSE | R² | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|
| Known entities, unseen pairs | 0.9180 | 1.2285 | 0.7773 | 0.8820 | 0.8334 |
| Unseen cells | 1.3561 | 1.7582 | 0.5825 | 0.7743 | 0.6979 |
| Unseen compounds | 1.4576 | 2.0117 | 0.4549 | 0.6917 | 0.6141 |
| Unseen cells and compounds | 1.7670 | 2.3770 | 0.3211 | 0.5736 | 0.4519 |

The declining performance under entity shift is the central first scientific
finding. It argues against treating random pair splits as sufficient evidence
of generalization. Full bootstrap intervals and split entity counts are in
`model_evaluation.json`.

## Endpoint mathematical validation

Within-plate normalized viability was not clipped. Interleaved concentrations
were held out from 250 real cell–compound curves (631 held-out points/model).

| Model | MAE | RMSE |
|---|---:|---:|
| 3-parameter Hill | 0.08642 | 0.11918 |
| Hill, zero bottom | 0.08487 | 0.11495 |
| log-linear comparator | 0.08695 | 0.11732 |

The models were similar; the more flexible Hill form did not improve held-out
RMSE in this sampled evaluation. This is an endpoint viability interpolation
test, not evidence that the ODE describes temporal cell counts.

## Dynamic simulation status

The logistic/Hill and sensitive/resistant ODEs are implemented, solve with
`solve_ivp`, enforce nonnegative inputs and pass numerical edge-case tests.
The public breast raw subset used in this run has one endpoint duration, so a
dynamic death/growth-rate model is not identifiable. UI trajectories therefore
remain labelled **“Simulación hipotética no calibrada”**. No fitted IC50 was
converted directly into an ODE killing EC50.

A 250-sample global sensitivity run was executed over explicitly hypothetical
uniform ranges. Final population was most rank-correlated with sensitive-cell
proliferation (Spearman 0.607), `kmax` (-0.470) and EC50 (0.460). These numbers
describe the chosen exploratory ranges, not calibrated biology; all samples and
the complete correlation table are saved with the reports.

## Verification

- `python -m pytest`: 10 passed.
- Python byte compilation: passed.
- Streamlit headless startup: HTTP 200 on the local test port.
- Download manifest: seven resources with expected byte sizes and SHA-256 hashes.

## Next evidence required

The most valuable next steps are train-only RNA-seq feature selection, verified
chemical structures and scaffold-aware validation, an independent cell-line
screen, and genuine multi-timepoint cell counts for dynamic parameter
identifiability.

