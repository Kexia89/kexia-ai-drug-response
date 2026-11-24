# Preliminary exploratory report — GDSC2 breast models

> Preclinical computational research only. This report does not recommend patient treatment.

- Models: **52**
- Compounds: **295**
- Fitted observations: **13388**
- Extracted raw wells: **2192303**
- LN_IC50 median: **3.479**
- AUC median: **0.9517**
- Fitted-curve RMSE median: **0.0783**

## Interpretation safeguards

`LN_IC50` is the natural logarithm of IC50 in µM; conversion uses `exp(LN_IC50)`. AUC is a fractional area of the fitted 72-hour viability curve over the screened concentration range. Neither quantity is clinical efficacy. Endpoint viability does not uniquely identify cell death, growth arrest or a dynamic killing rate.

## Genomic coverage

The first run uses official driver-mutation and driver-CNV summaries for a fixed, predeclared breast-cancer gene panel. RNA-seq was not downloaded because the official processed archive is approximately 1.29 GB. No synthetic values replaced it.

## Molecular subtype

The downloaded GDSC2 fitted file and CMP model list do not expose a verified breast-subtype field. Subtype is therefore left missing rather than inferred from cell-line names.
