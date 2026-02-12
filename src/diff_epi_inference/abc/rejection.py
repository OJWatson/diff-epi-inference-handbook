from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class ABCRejectionResult:
    """Output of an ABC rejection run."""

    thetas: Array
    distances: Array
    n_trials: int


def abc_rejection(
    prior_sample: Callable[[np.random.Generator], Array],
    simulate: Callable[[Array, np.random.Generator], Array],
    distance: Callable[[Array, Array], float],
    y_obs: Array,
    *,
    epsilon: float,
    n_accept: int,
    max_trials: int,
    summary: Callable[[Array], Array] | None = None,
    rng: np.random.Generator | None = None,
) -> ABCRejectionResult:
    """ABC rejection sampler.

    Draw parameters ``theta`` from a prior, simulate pseudo-data, and accept draws
    with summary-distance below a threshold ``epsilon``.

    Args:
        prior_sample: Function ``prior_sample(rng) -> theta``.
        simulate: Function ``simulate(theta, rng) -> y_sim``.
        distance: Function ``distance(s_sim, s_obs) -> scalar``.
        y_obs: Observed data (array-like).
        epsilon: Acceptance threshold (non-negative).
        n_accept: Number of accepted samples to return.
        max_trials: Maximum number of prior draws/simulations.
        summary: Optional summary function applied to both observed and simulated data.
        rng: Optional NumPy RNG.

    Returns:
        ABCRejectionResult containing accepted thetas, their distances, and the
        total number of trials performed.

    Notes:
        This is a minimal baseline intended for pedagogy. It is not optimised.
    """

    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and non-negative")
    if n_accept <= 0:
        raise ValueError("n_accept must be positive")
    if max_trials <= 0:
        raise ValueError("max_trials must be positive")

    rng = np.random.default_rng() if rng is None else rng

    y_obs = np.asarray(y_obs)
    summarise = (lambda x: x) if summary is None else summary
    s_obs = np.asarray(summarise(y_obs))

    accepted_thetas: list[Array] = []
    accepted_distances: list[float] = []

    n_trials = 0
    while len(accepted_thetas) < n_accept and n_trials < max_trials:
        n_trials += 1

        theta = np.asarray(prior_sample(rng), dtype=float)
        y_sim = np.asarray(simulate(theta, rng))
        s_sim = np.asarray(summarise(y_sim))

        d = float(distance(s_sim, s_obs))
        if d <= epsilon:
            accepted_thetas.append(theta)
            accepted_distances.append(d)

    if len(accepted_thetas) < n_accept:
        raise RuntimeError(
            "ABC rejection did not reach n_accept within max_trials: "
            f"accepted={len(accepted_thetas)}, n_accept={n_accept}, max_trials={max_trials}"
        )

    thetas = np.stack(accepted_thetas, axis=0)
    distances = np.asarray(accepted_distances, dtype=float)

    return ABCRejectionResult(thetas=thetas, distances=distances, n_trials=n_trials)
