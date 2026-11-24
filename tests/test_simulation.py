from __future__ import annotations

import numpy as np

from kexia_cancer.simulation.calibration import fit_endpoint_viability, viability_hill
from kexia_cancer.simulation.models import (
    ResistanceParameters,
    SinglePopulationParameters,
    hill_effect,
    simulate_resistance,
    simulate_single,
)


def test_zero_concentration_matches_zero_drug_effect() -> None:
    common = dict(initial_population=1000, proliferation_rate_per_hour=0.02, carrying_capacity=1e6, ec50_uM=1, hill=1)
    at_zero = simulate_single(SinglePopulationParameters(**common, concentration_uM=0, kmax_per_hour=1), duration_hours=72)
    no_effect = simulate_single(SinglePopulationParameters(**common, concentration_uM=100, kmax_per_hour=0), duration_hours=72)
    np.testing.assert_allclose(at_zero.population, no_effect.population, rtol=1e-8, atol=1e-6)


def test_population_is_nonnegative_under_extreme_effect() -> None:
    result = simulate_single(
        SinglePopulationParameters(initial_population=1e6, kmax_per_hour=10, concentration_uM=1e6),
        duration_hours=1000,
    )
    assert np.all(result.population >= 0)
    assert np.isfinite(result.population).all()


def test_resistance_components_sum_to_total_and_are_nonnegative() -> None:
    result = simulate_resistance(ResistanceParameters(initial_resistant_fraction=0.2), duration_hours=48)
    np.testing.assert_allclose(result.sensitive + result.resistant, result.population)
    assert (result[["sensitive", "resistant", "population"]] >= 0).all().all()


def test_reproducible_deterministic_solution() -> None:
    params = SinglePopulationParameters()
    a = simulate_single(params)
    b = simulate_single(params)
    np.testing.assert_array_equal(a.population.to_numpy(), b.population.to_numpy())


def test_hill_boundaries() -> None:
    assert hill_effect(0, 0.2, 1, 2) == 0
    assert np.isclose(hill_effect(1, 0.2, 1, 2), 0.1)
    assert hill_effect(1e12, 0.2, 1, 2) <= 0.2


def test_endpoint_curve_fit_recovers_synthetic_parameters() -> None:
    concentrations = np.logspace(-3, 2, 20)
    viability = viability_hill(concentrations, 0.15, 0.8, 1.4)
    fit = fit_endpoint_viability(concentrations, viability)
    assert np.isclose(fit.bottom, 0.15, atol=1e-3)
    assert np.isclose(fit.ec50_uM, 0.8, atol=1e-3)
    assert np.isclose(fit.hill, 1.4, atol=1e-3)
    assert "partially calibrated" in fit.status

