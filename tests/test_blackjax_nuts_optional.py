import numpy as np
import pytest


def test_blackjax_nuts_standard_normal_smoke():
    pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("blackjax")

    from diff_epi_inference.mcmc.nuts_blackjax import run_blackjax_nuts

    def log_prob_standard_normal(x):
        # log N(x; 0, 1) up to additive constant
        return -0.5 * jnp.sum(x**2)

    res = run_blackjax_nuts(
        log_prob_standard_normal,
        x0=np.array([3.0]),
        num_warmup=200,
        num_samples=500,
        seed=0,
    )

    chain = res.chain[:, 0]

    # Very loose checks: this is a smoke test, and only runs when optional deps exist.
    assert np.isfinite(res.accept_rate)
    assert 0.2 < res.accept_rate < 1.0
    assert abs(float(np.mean(chain))) < 0.25
    assert 0.7 < float(np.std(chain)) < 1.3
