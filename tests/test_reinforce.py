import numpy as np

from diff_epi_inference.gradients import reinforce_grad_logit_bernoulli


def test_reinforce_grad_matches_analytic_for_identity_function():
    # f(z)=z, z~Bern(p), p=sigmoid(logit)
    # E[z]=p, dp/dlogit = p(1-p)
    logit = 0.3
    p = 1.0 / (1.0 + np.exp(-logit))
    grad_true = p * (1.0 - p)

    rng = np.random.default_rng(0)
    res = reinforce_grad_logit_bernoulli(lambda z: z, logit=logit, rng=rng, n_samples=50_000)

    assert np.isfinite(res.grad)
    assert abs(res.grad - grad_true) < 3e-3


def test_reinforce_baseline_none_and_mean_are_consistent():
    # Using a baseline should not change the *expected* gradient.
    # With finite samples, we just check both are close to the analytic value.
    logit = -0.7
    p = 1.0 / (1.0 + np.exp(-logit))

    # f(z)=2z+0.1 => E[f]=2p+0.1 => d/dlogit E[f] = 2 * p(1-p)
    grad_true = 2.0 * p * (1.0 - p)

    rng = np.random.default_rng(1)
    f = lambda z: 2.0 * z + 0.1

    res_none = reinforce_grad_logit_bernoulli(
        f, logit=logit, rng=rng, n_samples=80_000, baseline="none"
    )

    rng = np.random.default_rng(2)
    res_mean = reinforce_grad_logit_bernoulli(
        f, logit=logit, rng=rng, n_samples=80_000, baseline="mean"
    )

    assert abs(res_none.grad - grad_true) < 4e-3
    assert abs(res_mean.grad - grad_true) < 4e-3
