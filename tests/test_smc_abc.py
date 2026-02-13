from __future__ import annotations

import numpy as np
import pytest

from diff_epi_inference.abc import smc_abc


def _normal_logpdf(x: float, *, mu: float, sigma: float) -> float:
    x = float(x)
    mu = float(mu)
    sigma = float(sigma)
    return -0.5 * np.log(2.0 * np.pi * sigma * sigma) - 0.5 * ((x - mu) / sigma) ** 2


def test_smc_abc_shapes_weights_and_thresholds() -> None:
    rng = np.random.default_rng(0)

    # Toy model: theta ~ N(0, 1), y | theta ~ N(theta, 1)
    def prior_sample(rng: np.random.Generator) -> np.ndarray:
        return np.array([rng.normal(0.0, 1.0)])

    def prior_logpdf(theta: np.ndarray) -> float:
        return _normal_logpdf(theta[0], mu=0.0, sigma=1.0)

    def simulate(theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return np.array([rng.normal(theta[0], 1.0)])

    def summary(y: np.ndarray) -> np.ndarray:
        return y

    def distance(s_sim: np.ndarray, s_obs: np.ndarray) -> float:
        return float(np.abs(s_sim[0] - s_obs[0]))

    sigma_kernel = 0.5

    def perturb(theta_prev: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return np.array([rng.normal(theta_prev[0], sigma_kernel)])

    def perturb_logpdf(theta_prop: np.ndarray, theta_prev: np.ndarray) -> float:
        return _normal_logpdf(theta_prop[0], mu=theta_prev[0], sigma=sigma_kernel)

    y_obs = np.array([0.25])
    eps = np.array([2.0, 1.0])

    res = smc_abc(
        prior_sample=prior_sample,
        prior_logpdf=prior_logpdf,
        simulate=simulate,
        distance=distance,
        y_obs=y_obs,
        epsilons=eps,
        n_particles=40,
        max_trials_per_round=20_000,
        perturb=perturb,
        perturb_logpdf=perturb_logpdf,
        summary=summary,
        rng=rng,
    )

    assert res.thetas.shape == (2, 40, 1)
    assert res.distances.shape == (2, 40)
    assert res.weights.shape == (2, 40)
    assert np.allclose(res.epsilons, eps)

    # All accepted particles satisfy the round-specific threshold.
    assert np.all(res.distances[0] <= eps[0] + 1e-12)
    assert np.all(res.distances[1] <= eps[1] + 1e-12)

    # Weights are normalised per round.
    assert np.all(res.weights >= 0)
    assert np.allclose(np.sum(res.weights, axis=1), np.ones(2))


def test_smc_abc_rejects_increasing_epsilons() -> None:
    rng = np.random.default_rng(0)

    def prior_sample(rng: np.random.Generator) -> np.ndarray:
        return np.array([0.0])

    def prior_logpdf(theta: np.ndarray) -> float:
        return 0.0

    def simulate(theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return np.array([0.0])

    def distance(s_sim: np.ndarray, s_obs: np.ndarray) -> float:
        return 0.0

    def perturb(theta_prev: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return theta_prev

    def perturb_logpdf(theta_prop: np.ndarray, theta_prev: np.ndarray) -> float:
        return 0.0

    with pytest.raises(ValueError, match="non-increasing"):
        smc_abc(
            prior_sample=prior_sample,
            prior_logpdf=prior_logpdf,
            simulate=simulate,
            distance=distance,
            y_obs=np.array([0.0]),
            epsilons=np.array([0.1, 0.2]),
            n_particles=2,
            max_trials_per_round=10,
            perturb=perturb,
            perturb_logpdf=perturb_logpdf,
            rng=rng,
        )
