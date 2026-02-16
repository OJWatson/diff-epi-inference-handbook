import numpy as np
import pytest


def _have_jax() -> bool:
    try:
        import jax  # noqa: F401

        return True
    except ModuleNotFoundError:
        return False


@pytest.mark.skipif(not _have_jax(), reason="optional dependency jax not installed")
def test_toy_relaxed_sir_simulation_shapes_and_finite():
    from diff_epi_inference.examples.toy_diff_epi_jax import simulate_toy_relaxed_sir

    path = simulate_toy_relaxed_sir(
        seed=0,
        beta=0.35,
        gamma=0.15,
        s0=90.0,
        i0=10.0,
        r0=0.0,
        steps=25,
        temperature=0.7,
    )

    assert path.s.shape == (26,)
    assert path.i.shape == (26,)
    assert path.r.shape == (26,)
    assert path.new_infections.shape == (25,)
    assert path.new_recoveries.shape == (25,)

    assert np.all(np.isfinite(path.s))
    assert np.all(np.isfinite(path.i))
    assert np.all(np.isfinite(path.r))
    assert np.all(np.isfinite(path.new_infections))
    assert np.all(np.isfinite(path.new_recoveries))


@pytest.mark.skipif(not _have_jax(), reason="optional dependency jax not installed")
def test_toy_grad_fit_reduces_loss():
    from diff_epi_inference.examples.toy_diff_epi_jax import (
        fit_beta_by_gradient_descent,
        simulate_toy_relaxed_sir,
    )

    steps = 30
    true_beta = 0.4
    gamma = 0.2

    path = simulate_toy_relaxed_sir(
        seed=123,
        beta=true_beta,
        gamma=gamma,
        s0=80.0,
        i0=20.0,
        r0=0.0,
        steps=steps,
        temperature=0.6,
    )
    y_obs = np.asarray(path.new_infections)

    res = fit_beta_by_gradient_descent(
        y_obs=y_obs,
        seed=123,
        beta_init=0.15,
        gamma=gamma,
        s0=80.0,
        i0=20.0,
        r0=0.0,
        steps=steps,
        temperature=0.6,
        lr=0.25,
        iters=40,
    )

    assert res.losses.shape == (40,)
    assert np.isfinite(res.beta_hat)

    # Smoke criterion: loss should go down substantially.
    assert float(res.losses[-1]) < float(res.losses[0]) * 0.8
