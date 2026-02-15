import numpy as np
import pytest


def test_meanfield_vi_seir_beta_only_smoke():
    """Optional smoke test: mean-field Gaussian VI runs and improves the ELBO."""

    pytest.importorskip("jax")

    from diff_epi_inference import SEIRParams
    from diff_epi_inference.models.seir_jax_beta_only import make_log_post_logbeta_jax
    from diff_epi_inference.observation import (
        discrete_gamma_delay_pmf,
        expected_reported_cases_delayed,
        incidence_from_susceptibles,
        sample_nbinom_reports,
    )
    from diff_epi_inference.seir import simulate_seir_euler
    from diff_epi_inference.vi import fit_meanfield_gaussian_vi_jax

    # --- Fixed settings: keep this tiny so the test stays fast ---
    sigma_fixed = 1 / 5
    gamma_fixed = 1 / 7

    s0, e0, i0, r0 = 999.0, 0.0, 1.0, 0.0

    dt = 0.5
    steps = 60

    reporting_rate = 0.3
    w = discrete_gamma_delay_pmf(shape=2.0, scale=1.0, max_delay=15)
    dispersion = 20.0

    logbeta_prior_mean = float(np.log(0.5))
    logbeta_prior_sd = 0.5

    rng = np.random.default_rng(2026)
    beta_true = 0.6

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

    res = fit_meanfield_gaussian_vi_jax(
        log_post,
        dim=1,
        seed=0,
        num_steps=400,
        lr=5e-2,
        num_mc_samples=32,
        init_mean=np.array([np.log(0.3)]),
        init_log_std=np.array([np.log(0.5)]),
    )

    assert np.isfinite(res.elbo_history[-1])

    # ELBO should generally improve from the start.
    assert float(res.elbo_history[-1]) > float(res.elbo_history[0])

    beta_vi = float(np.exp(res.mean[0]))

    # Very loose sanity check: estimate should land in a plausible range.
    assert 0.1 < beta_vi < 2.0
