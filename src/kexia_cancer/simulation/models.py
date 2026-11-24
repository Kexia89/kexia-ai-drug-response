from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class SinglePopulationParameters:
    initial_population: float = 1e5
    proliferation_rate_per_hour: float = 0.03
    carrying_capacity: float = 1e7
    concentration_uM: float = 1.0
    kmax_per_hour: float = 0.04
    ec50_uM: float = 1.0
    hill: float = 1.0


@dataclass(frozen=True)
class ResistanceParameters(SinglePopulationParameters):
    initial_resistant_fraction: float = 0.01
    resistant_proliferation_rate_per_hour: float = 0.025
    resistant_kmax_per_hour: float = 0.005
    resistant_ec50_uM: float = 10.0
    transition_rate_per_hour: float = 0.0


def hill_effect(concentration_uM: float | np.ndarray, kmax_per_hour: float, ec50_uM: float, hill: float) -> np.ndarray:
    concentration = np.maximum(np.asarray(concentration_uM, dtype=float), 0.0)
    if kmax_per_hour < 0 or ec50_uM <= 0 or hill <= 0:
        raise ValueError("kmax must be non-negative; EC50 and Hill coefficient must be positive")
    numerator = np.power(concentration, hill)
    return kmax_per_hour * numerator / (np.power(ec50_uM, hill) + numerator)


def _validate_common(initial: float, rate: float, capacity: float, duration: float) -> None:
    if initial < 0 or rate < 0 or capacity <= 0 or duration <= 0:
        raise ValueError("Population/rates must be non-negative and capacity/duration positive")


def simulate_single(
    params: SinglePopulationParameters,
    duration_hours: float = 240,
    points: int = 300,
    concentration: Callable[[float], float] | None = None,
    label: str = "Simulación hipotética no calibrada",
) -> pd.DataFrame:
    _validate_common(params.initial_population, params.proliferation_rate_per_hour, params.carrying_capacity, duration_hours)
    cfun = concentration or (lambda _t: params.concentration_uM)

    def rhs(t: float, y: np.ndarray) -> list[float]:
        n = max(float(y[0]), 0.0)
        kill = float(hill_effect(cfun(t), params.kmax_per_hour, params.ec50_uM, params.hill))
        return [params.proliferation_rate_per_hour * n * (1 - n / params.carrying_capacity) - kill * n]

    times = np.linspace(0, duration_hours, points)
    solution = solve_ivp(rhs, (0, duration_hours), [params.initial_population], t_eval=times, method="LSODA", rtol=1e-8, atol=1e-6)
    if not solution.success:
        raise RuntimeError(solution.message)
    population = np.maximum(solution.y[0], 0.0)
    return pd.DataFrame(
        {
            "time_hours": solution.t,
            "population": population,
            "concentration_uM": [cfun(t) for t in solution.t],
            "simulation_label": label,
            "model": "single logistic + Hill effect",
        }
    )


def simulate_resistance(
    params: ResistanceParameters,
    duration_hours: float = 240,
    points: int = 300,
    label: str = "Simulación hipotética no calibrada",
) -> pd.DataFrame:
    _validate_common(params.initial_population, params.proliferation_rate_per_hour, params.carrying_capacity, duration_hours)
    if not 0 <= params.initial_resistant_fraction <= 1 or params.transition_rate_per_hour < 0:
        raise ValueError("Resistant fraction must be in [0,1] and transition rate non-negative")
    sensitive0 = params.initial_population * (1 - params.initial_resistant_fraction)
    resistant0 = params.initial_population * params.initial_resistant_fraction

    def rhs(_t: float, y: np.ndarray) -> list[float]:
        sensitive, resistant = np.maximum(y, 0.0)
        crowding = max(0.0, 1 - (sensitive + resistant) / params.carrying_capacity)
        ks = float(hill_effect(params.concentration_uM, params.kmax_per_hour, params.ec50_uM, params.hill))
        kr = float(hill_effect(params.concentration_uM, params.resistant_kmax_per_hour, params.resistant_ec50_uM, params.hill))
        transition = params.transition_rate_per_hour * sensitive
        return [
            params.proliferation_rate_per_hour * sensitive * crowding - ks * sensitive - transition,
            params.resistant_proliferation_rate_per_hour * resistant * crowding - kr * resistant + transition,
        ]

    times = np.linspace(0, duration_hours, points)
    solution = solve_ivp(rhs, (0, duration_hours), [sensitive0, resistant0], t_eval=times, method="LSODA", rtol=1e-8, atol=1e-6)
    if not solution.success:
        raise RuntimeError(solution.message)
    sensitive, resistant = np.maximum(solution.y, 0.0)
    return pd.DataFrame(
        {
            "time_hours": solution.t,
            "sensitive": sensitive,
            "resistant": resistant,
            "population": sensitive + resistant,
            "concentration_uM": params.concentration_uM,
            "simulation_label": label,
            "model": "competitive sensitive/resistant logistic + Hill effects",
        }
    )


def sensitivity_ensemble(
    base: SinglePopulationParameters,
    ranges: dict[str, tuple[float, float]],
    samples: int = 200,
    duration_hours: float = 240,
    seed: int = 42,
) -> pd.DataFrame:
    """Propagate uncertainty in unidentified dynamic parameters into trajectory bands."""
    rng = np.random.default_rng(seed)
    trajectories = []
    for sample in range(samples):
        values = asdict(base)
        for name, (low, high) in ranges.items():
            values[name] = float(rng.uniform(low, high))
        trajectory = simulate_single(
            SinglePopulationParameters(**values), duration_hours=duration_hours, label="Simulación parcialmente calibrada"
        )
        trajectory["sample"] = sample
        trajectories.append(trajectory)
    stacked = pd.concat(trajectories, ignore_index=True)
    return (
        stacked.groupby("time_hours")["population"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .rename(columns={0.025: "lower_95", 0.5: "median", 0.975: "upper_95"})
        .reset_index()
    )


def resistance_sensitivity_samples(
    base: ResistanceParameters,
    ranges: dict[str, tuple[float, float]],
    samples: int = 200,
    duration_hours: float = 240,
    seed: int = 42,
) -> pd.DataFrame:
    """Global uniform sensitivity sample for proliferation, drug effect and resistance."""
    rng = np.random.default_rng(seed)
    records = []
    for sample in range(samples):
        values = asdict(base)
        drawn = {}
        for name, (low, high) in ranges.items():
            drawn[name] = values[name] = float(rng.uniform(low, high))
        result = simulate_resistance(ResistanceParameters(**values), duration_hours=duration_hours)
        final = result.iloc[-1]
        records.append(
            {
                "sample": sample,
                **drawn,
                "final_population": float(final["population"]),
                "final_resistant_fraction": float(final["resistant"] / final["population"]) if final["population"] > 0 else 0.0,
            }
        )
    return pd.DataFrame(records)

