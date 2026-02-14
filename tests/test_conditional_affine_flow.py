import numpy as np

from diff_epi_inference.flows import ConditionalAffineDiagNormal


def test_conditional_affine_diag_normal_fit_and_logprob_smoke():
    rng = np.random.default_rng(0)

    n = 500
    k = 3
    d = 2

    # Synthetic contexts.
    C = rng.normal(size=(n, k))

    # Ground truth conditional affine parameters.
    b_true = np.array([0.2, -0.1])
    W_true = np.array([[0.5, -0.2], [0.1, 0.3], [-0.4, 0.0]])
    sigma_true = np.array([0.7, 0.4])

    mu = C @ W_true + b_true
    X = mu + rng.normal(size=(n, d)) * sigma_true

    flow = ConditionalAffineDiagNormal.fit_closed_form(contexts=C, thetas=X)

    # Mean should be close in an MSE sense.
    mu_hat = flow.mean(C)
    mse = float(np.mean((mu_hat - mu) ** 2))
    assert mse < 0.05

    # Log prob finite and has right shape.
    lp = flow.log_prob(X, C)
    assert lp.shape == (n,)
    assert np.all(np.isfinite(lp))


def test_conditional_affine_diag_normal_sample_shape():
    rng = np.random.default_rng(1)
    flow = ConditionalAffineDiagNormal(
        b=np.array([0.0, 1.0]),
        W=np.zeros((2, 2)),
        log_sigma=np.log(np.array([1.0, 2.0])),
    )

    s = flow.sample(np.array([0.2, -0.3]), n=123, rng=rng)
    assert s.shape == (123, 2)
