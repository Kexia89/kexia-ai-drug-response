from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


def viability_hill(concentration_uM: np.ndarray, bottom: float, ec50_uM: float, hill: float) -> np.ndarray:
    concentration = np.maximum(np.asarray(concentration_uM, dtype=float), 0)
    return bottom + (1 - bottom) / (1 + np.power(concentration / ec50_uM, hill))


@dataclass(frozen=True)
class DoseResponseFit:
    bottom: float
    ec50_uM: float
    hill: float
    covariance: np.ndarray
    status: str = "endpoint viability calibrated; dynamic model only partially calibrated"


def fit_endpoint_viability(concentration_uM: np.ndarray, viability_fraction: np.ndarray) -> DoseResponseFit:
    """Fit a bounded 3-parameter endpoint viability curve.

    This EC50 belongs to an endpoint viability function. It is intentionally not
    returned as a death-rate EC50 and must not be inserted into an ODE without an
    explicit additional modelling assumption.
    """
    x = np.asarray(concentration_uM, dtype=float)
    y = np.asarray(viability_fraction, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x >= 0)
    x, y = x[mask], y[mask]
    if len(x) < 5 or np.unique(x).size < 4:
        raise ValueError("At least five finite observations over four concentrations are required")
    positive = x[x > 0]
    guess_ec50 = float(np.median(positive)) if len(positive) else 1.0
    params, covariance = curve_fit(
        viability_hill,
        x,
        y,
        p0=[max(0.0, min(0.8, float(np.min(y)))), guess_ec50, 1.0],
        bounds=([0.0, 1e-12, 0.05], [1.2, max(1e6, float(np.max(x)) * 1e4), 8.0]),
        maxfev=50_000,
    )
    return DoseResponseFit(float(params[0]), float(params[1]), float(params[2]), covariance)

