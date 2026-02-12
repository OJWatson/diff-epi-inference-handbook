import numpy as np
import pytest

from diff_epi_inference.mcmc.hmc import hamiltonian_monte_carlo


def test_hmc_shapes_and_reproducible() -> None:
    def log_prob(x: np.ndarray) -> float:
        # Standard normal up to an additive constant.
        return float(-0.5 * np.sum(x**2))

    x0 = np.array([0.2, -0.1])
    n_steps = 25

    res1 = hamiltonian_monte_carlo(
        log_prob,
        x0,
        n_steps=n_steps,
        step_size=0.2,
        n_leapfrog=5,
        grad_eps=1e-4,
        rng=np.random.default_rng(123),
    )
    res2 = hamiltonian_monte_carlo(
        log_prob,
        x0,
        n_steps=n_steps,
        step_size=0.2,
        n_leapfrog=5,
        grad_eps=1e-4,
        rng=np.random.default_rng(123),
    )

    assert res1.chain.shape == (n_steps + 1, *x0.shape)
    assert res1.accepted.shape == (n_steps,)
    assert res1.chain.dtype == float
    assert res1.accepted.dtype == bool

    np.testing.assert_allclose(res1.chain, res2.chain)
    np.testing.assert_array_equal(res1.accepted, res2.accepted)


def test_hmc_input_validation() -> None:
    def log_prob(x: np.ndarray) -> float:
        return float(-0.5 * np.sum(x**2))

    x0 = np.array([0.0])

    with pytest.raises(ValueError, match="n_steps must be positive"):
        hamiltonian_monte_carlo(log_prob, x0, n_steps=0, step_size=0.1, n_leapfrog=1)

    with pytest.raises(ValueError, match="step_size must be positive"):
        hamiltonian_monte_carlo(log_prob, x0, n_steps=1, step_size=0.0, n_leapfrog=1)

    with pytest.raises(ValueError, match="n_leapfrog must be positive"):
        hamiltonian_monte_carlo(log_prob, x0, n_steps=1, step_size=0.1, n_leapfrog=0)

    with pytest.raises(ValueError, match="grad_eps must be positive"):
        hamiltonian_monte_carlo(
            log_prob,
            x0,
            n_steps=1,
            step_size=0.1,
            n_leapfrog=1,
            grad_eps=0.0,
        )


def test_hmc_standard_normal_smoke() -> None:
    rng = np.random.default_rng(0)

    def log_prob(x: np.ndarray) -> float:
        # Standard normal up to an additive constant.
        return float(-0.5 * np.sum(x**2))

    res = hamiltonian_monte_carlo(
        log_prob,
        np.array([0.0]),
        n_steps=2000,
        step_size=0.2,
        n_leapfrog=5,
        grad_eps=1e-4,
        rng=rng,
    )

    assert 0.2 < res.accept_rate <= 1.0

    samples = res.chain[500:, 0]  # burn-in
    mean = float(np.mean(samples))
    var = float(np.var(samples))

    assert abs(mean) < 0.1
    assert 0.8 < var < 1.2
