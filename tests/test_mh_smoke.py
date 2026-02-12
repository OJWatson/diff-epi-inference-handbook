import numpy as np

from diff_epi_inference.mcmc.mh import random_walk_metropolis_hastings


def test_random_walk_mh_shapes_and_reproducible() -> None:
    def log_prob(x: np.ndarray) -> float:
        return float(-0.5 * np.sum(x**2))

    x0 = np.array([0.2, -0.1])
    n_steps = 25

    res1 = random_walk_metropolis_hastings(
        log_prob,
        x0,
        proposal_std=0.3,
        n_steps=n_steps,
        rng=np.random.default_rng(123),
    )
    res2 = random_walk_metropolis_hastings(
        log_prob,
        x0,
        proposal_std=0.3,
        n_steps=n_steps,
        rng=np.random.default_rng(123),
    )

    assert res1.chain.shape == (n_steps + 1, *x0.shape)
    assert res1.accepted.shape == (n_steps,)
    assert res1.chain.dtype == float
    assert res1.accepted.dtype == bool

    np.testing.assert_allclose(res1.chain, res2.chain)
    np.testing.assert_array_equal(res1.accepted, res2.accepted)


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
