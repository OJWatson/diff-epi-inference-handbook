import numpy as np
import pytest


def test_blackjax_nuts_seir_beta_only_coverage_smoke():
    """Optional coverage smoke test for the BlackJAX NUTS SEIR beta-only posterior.

    This is intentionally lightweight and only runs when optional deps exist.
    It is *not* full SBC; it is a quick sanity check that nominal 90% intervals
    are not wildly miscalibrated.
    """

    pytest.importorskip("jax")
    pytest.importorskip("blackjax")

    from diff_epi_inference import (
        SEIRParams,
        discrete_gamma_delay_pmf,
        expected_reported_cases_delayed,
        incidence_from_susceptibles,
        sample_nbinom_reports,
        simulate_seir_euler,
    )
    from diff_epi_inference.mcmc.nuts_blackjax import run_blackjax_nuts
    from diff_epi_inference.models.seir_jax_beta_only import make_log_post_logbeta_jax

    # --- Fixed settings (match the running example defaults used in the book) ---
    sigma_fixed = 1 / 5
    gamma_fixed = 1 / 7

    s0, e0, i0, r0 = 999.0, 0.0, 1.0, 0.0

    dt = 0.2
    steps = 200

    reporting_rate = 0.3
    w = discrete_gamma_delay_pmf(shape=2.0, scale=1.0, max_delay=20)
    dispersion = 20.0

    # Prior on log(beta)
    logbeta_prior_mean = float(np.log(0.5))
    logbeta_prior_sd = 0.5

    # (JAX implementation lives in diff_epi_inference.models.seir_jax_beta_only)

    rng = np.random.default_rng(2026)

    beta_grid = [0.4, 0.5, 0.6, 0.7]
    contained = []

    for beta_true in beta_grid:
        params_true = SEIRParams(beta=float(beta_true), sigma=sigma_fixed, gamma=gamma_fixed)
        out_true = simulate_seir_euler(
            params=params_true,
            s0=s0,
            e0=e0,
            i0=i0,
            r0=r0,
            dt=dt,
            steps=steps,
        )
        inc_true = incidence_from_susceptibles(out_true["S"])
        mu_true = expected_reported_cases_delayed(
            incidence=inc_true,
            reporting_rate=reporting_rate,
            delay_pmf=w,
        )
        y_obs = sample_nbinom_reports(expected=mu_true, dispersion=dispersion, rng=rng)

        log_post = make_log_post_logbeta_jax(
            y_obs=y_obs,
            w_delay_pmf=w,
            sigma=sigma_fixed,
            gamma=gamma_fixed,
            s0=s0,
            e0=e0,
            i0=i0,
            r0=r0,
            dt=dt,
            steps=steps,
            reporting_rate=reporting_rate,
            dispersion=dispersion,
            logbeta_prior_mean=logbeta_prior_mean,
            logbeta_prior_sd=logbeta_prior_sd,
        )

        res = run_blackjax_nuts(
            log_post,
            x0=np.array([np.log(0.3)]),
            num_warmup=250,
            num_samples=500,
            seed=123,
        )

        assert np.isfinite(res.accept_rate)
        assert 0.2 < res.accept_rate < 1.0

        beta_draws = np.exp(res.chain[:, 0])
        q05, q95 = map(float, np.quantile(beta_draws, [0.05, 0.95]))
        contained.append(q05 <= beta_true <= q95)

    # A very loose, low-power sanity check: we should not be catastrophically miscalibrated.
    coverage = float(np.mean(contained))
    assert 0.25 <= coverage <= 1.0
