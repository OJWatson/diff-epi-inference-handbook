import numpy as np
import pytest


def test_blackjax_nuts_seir_beta_only_coverage_smoke():
    """Optional coverage smoke test for the BlackJAX NUTS SEIR beta-only posterior.

    This is intentionally light-weight and only runs when optional deps exist.
    It is *not* full SBC; it is a quick sanity check that nominal 90% intervals
    are not wildly miscalibrated.
    """

    pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("blackjax")

    import jax
    from jax.scipy.special import gammaln

    from diff_epi_inference import (
        SEIRParams,
        discrete_gamma_delay_pmf,
        expected_reported_cases_delayed,
        incidence_from_susceptibles,
        sample_nbinom_reports,
        simulate_seir_euler,
    )
    from diff_epi_inference.mcmc.nuts_blackjax import run_blackjax_nuts

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

    # JAX constants
    w_jax = jnp.asarray(w, dtype=float)
    w_jax = w_jax / jnp.sum(w_jax)

    def simulate_seir_euler_jax(beta: jax.Array) -> jax.Array:
        beta = jnp.asarray(beta)
        sigma = jnp.asarray(sigma_fixed)
        gamma = jnp.asarray(gamma_fixed)

        dt_j = jnp.asarray(dt)
        steps_j = int(steps)

        def step_fn(state, _):
            s, e, i, r = state
            n = s + e + i + r
            inf_flow = beta * s * i / n
            inc_flow = sigma * e
            rec_flow = gamma * i
            s1 = s - dt_j * inf_flow
            e1 = e + dt_j * (inf_flow - inc_flow)
            i1 = i + dt_j * (inc_flow - rec_flow)
            r1 = r + dt_j * rec_flow
            return (s1, e1, i1, r1), s1

        init = (jnp.asarray(s0), jnp.asarray(e0), jnp.asarray(i0), jnp.asarray(r0))
        (_, _, _, _), s_hist = jax.lax.scan(step_fn, init, xs=None, length=steps_j)

        # Prepend s0 so we have length steps+1 like the NumPy solver.
        s_all = jnp.concatenate([jnp.asarray([s0]), s_hist], axis=0)
        return s_all

    def nbinom_logpmf_jax(k: jax.Array, mu: jax.Array, dispersion_: float) -> jax.Array:
        r = jnp.asarray(float(dispersion_))
        p = r / (r + mu)
        logp = jnp.log(p)
        log1mp = jnp.log1p(-p)
        kf = k.astype(float)
        out = (
            gammaln(kf + r)
            - gammaln(r)
            - gammaln(kf + 1.0)
            + r * logp
            + kf * log1mp
        )

        zero_mu = mu == 0
        out = jnp.where(zero_mu & (kf == 0), 0.0, out)
        out = jnp.where(zero_mu & (kf > 0), -jnp.inf, out)
        return out

    def make_log_post_logbeta_jax(y_obs_local: np.ndarray):
        y_obs_jax = jnp.asarray(y_obs_local)

        def _log_post(position: jax.Array) -> jax.Array:
            logbeta = position[0]
            beta = jnp.exp(logbeta)

            lp = -0.5 * ((logbeta - logbeta_prior_mean) / logbeta_prior_sd) ** 2

            s_all = simulate_seir_euler_jax(beta)
            inc = jnp.maximum(s_all[:-1] - s_all[1:], 0.0)

            base = reporting_rate * inc
            mu = jnp.convolve(base, w_jax, mode="full")[: base.shape[0]]

            ll = jnp.sum(nbinom_logpmf_jax(y_obs_jax, mu, dispersion))
            return lp + ll

        return _log_post

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

        log_post = make_log_post_logbeta_jax(y_obs)

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
