import numpy as np
import pytest

from diff_epi_inference import SEIRParams
from diff_epi_inference.flows import ConditionalAffineDiagNormal
from diff_epi_inference.pipeline import simulate_seir_and_report_stochastic


def test_modern_sbi_end_to_end_golden_outputs() -> None:
    """Golden checks for the offline deterministic Modern SBI end-to-end chapter.

    This test intentionally mirrors `book/modern-sbi-end-to-end.qmd` and asserts key
    printed quantities are stable. If this test changes, update the chapter (and
    its frozen outputs) together.
    """

    # Global seed(s) for determinism.
    rng_obs = np.random.default_rng(0)
    rng_train = np.random.default_rng(1)
    rng_post = np.random.default_rng(2)
    rng_ppc = np.random.default_rng(3)

    steps = 80
    reporting_rate = 0.25

    params_fixed = SEIRParams(beta=0.3, sigma=1 / 4.0, gamma=1 / 6.0)

    def simulate_y(beta: float, *, rng: np.random.Generator) -> np.ndarray:
        ds = simulate_seir_and_report_stochastic(
            params=SEIRParams(
                beta=float(beta),
                sigma=params_fixed.sigma,
                gamma=params_fixed.gamma,
            ),
            s0=10_000,
            e0=3,
            i0=2,
            r0=0,
            dt=1.0,
            steps=steps,
            reporting_rate=reporting_rate,
            rng=rng,
        )
        return ds.y.astype(float)

    def summary(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        peak_t = int(np.argmax(y))
        return np.array([np.sum(y), np.max(y), peak_t], dtype=float)

    # Synthetic observed dataset.
    beta_true = 0.35
    y_obs = simulate_y(beta_true, rng=rng_obs)
    s_obs = summary(y_obs)

    assert s_obs.tolist() == [1138.0, 59.0, 77.0]

    # Prior and simulation budget.
    logbeta_prior_mean = float(np.log(0.3))
    logbeta_prior_sd = 0.35

    def prior_sample_logbeta(rng: np.random.Generator) -> float:
        return float(rng.normal(loc=logbeta_prior_mean, scale=logbeta_prior_sd))

    def make_dataset(n_sims: int, *, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        S = np.zeros((n_sims, 3), dtype=float)
        logb = np.zeros((n_sims,), dtype=float)
        for i in range(n_sims):
            lb = prior_sample_logbeta(rng)
            beta = float(np.exp(lb))
            y = simulate_y(beta, rng=rng)
            S[i] = summary(y)
            logb[i] = lb
        return S, logb

    S_train, logb_train = make_dataset(600, rng=rng_train)
    S_test, logb_test = make_dataset(200, rng=np.random.default_rng(9))

    assert S_train.shape == (600, 3)
    assert S_test.shape == (200, 3)

    # Closed-form conditional posterior fit.
    flow = ConditionalAffineDiagNormal.fit_closed_form(
        contexts=S_train,
        thetas=logb_train[:, None],
    )

    mu_test = flow.mean(S_test)[:, 0]
    rmse = float(np.sqrt(np.mean((mu_test - logb_test) ** 2)))
    sigma_hat = float(np.exp(flow.log_sigma[0]))

    assert sigma_hat == pytest.approx(0.13899648761901903, rel=1e-10, abs=1e-12)
    assert rmse == pytest.approx(0.13457752613127882, rel=1e-10, abs=1e-12)

    # Posterior sampling for the observation.
    logb_post = flow.sample(s_obs, n=5_000, rng=rng_post)[:, 0]
    beta_post = np.exp(logb_post)

    assert float(beta_post.mean()) == pytest.approx(0.3545349552914927, rel=1e-10, abs=1e-12)
    assert float(np.quantile(beta_post, 0.1)) == pytest.approx(
        0.29254098493715425, rel=1e-10, abs=1e-12
    )
    assert float(np.quantile(beta_post, 0.9)) == pytest.approx(
        0.4189537197061017, rel=1e-10, abs=1e-12
    )

    # PPC in summary space.
    n_ppc = 120
    S_ppc = np.zeros((n_ppc, 3), dtype=float)
    for i in range(n_ppc):
        beta_i = float(beta_post[i])
        y_i = simulate_y(beta_i, rng=rng_ppc)
        S_ppc[i] = summary(y_i)

    q10 = np.quantile(S_ppc, 0.1, axis=0)
    q90 = np.quantile(S_ppc, 0.9, axis=0)

    assert q10.tolist() == pytest.approx([179.2, 15.8, 61.9], rel=1e-10, abs=1e-12)
    assert q90.tolist() == pytest.approx([1992.9, 97.0, 79.0], rel=1e-10, abs=1e-12)
