from __future__ import annotations

import numpy as np


def incidence_from_susceptibles(S: np.ndarray) -> np.ndarray:
    """Compute per-step incidence from a susceptible trajectory.

    Parameters
    ----------
    S:
        Susceptible compartment values of length T+1.

    Returns
    -------
    np.ndarray
        Incidence per step of length T, defined as max(S[t] - S[t+1], 0).

    Notes
    -----
    For the deterministic SEIR Euler solver, S is non-increasing for reasonable dt,
    so this corresponds to new infections per time step.
    """

    S = np.asarray(S, dtype=float)
    if S.ndim != 1 or S.shape[0] < 2:
        raise ValueError("S must be a 1D array of length >= 2")

    inc = S[:-1] - S[1:]
    return np.maximum(inc, 0.0)


def expected_reported_cases(*, incidence: np.ndarray, reporting_rate: float) -> np.ndarray:
    """Simple observation model: expected reported cases per step.

    E[y_t] = rho * incidence_t

    Parameters
    ----------
    incidence:
        Array of non-negative incidence values.
    reporting_rate:
        rho in [0, 1].
    """

    if not (0.0 <= reporting_rate <= 1.0):
        raise ValueError("reporting_rate must be in [0, 1]")

    incidence = np.asarray(incidence, dtype=float)
    if np.any(incidence < 0):
        raise ValueError("incidence must be non-negative")

    return reporting_rate * incidence


def sample_poisson_reports(
    *,
    expected: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample reported cases using a Poisson observation model."""

    expected = np.asarray(expected, dtype=float)
    if np.any(expected < 0):
        raise ValueError("expected must be non-negative")

    if rng is None:
        rng = np.random.default_rng()

    return rng.poisson(lam=expected)
