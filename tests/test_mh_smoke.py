import numpy as np

from diff_epi_inference.mcmc.mh import random_walk_metropolis_hastings


def test_random_walk_mh_standard_normal_smoke() -> None:
    rng = np.random.default_rng(0)

    def log_prob(x: np.ndarray) -> float:
        # Standard normal up to an additive constant.
        return float(-0.5 * np.sum(x**2))

    res = random_walk_metropolis_hastings(
        log_prob,
        np.array([0.0]),
        proposal_std=1.0,
        n_steps=4000,
        rng=rng,
    )

    assert 0.05 < res.accept_rate < 0.9

    samples = res.chain[1000:, 0]  # burn-in
    mean = float(np.mean(samples))
    var = float(np.var(samples))

    assert abs(mean) < 0.1
    assert 0.8 < var < 1.2
