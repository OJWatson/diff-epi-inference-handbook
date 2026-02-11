import numpy as np

from diff_epi_inference import (
    SEIRParams,
    discrete_gamma_delay_pmf,
    expected_reported_cases_delayed,
    incidence_from_susceptibles,
    nbinom_loglik,
    sample_nbinom_reports,
    simulate_seir_euler,
)


def test_delay_nbinom_loglik_finite_and_shapes():
    out = simulate_seir_euler(
        params=SEIRParams(beta=0.6, sigma=1 / 5, gamma=1 / 7),
        s0=999.0,
        e0=0.0,
        i0=1.0,
        r0=0.0,
        dt=0.2,
        steps=100,
    )
    inc = incidence_from_susceptibles(out["S"])

    w = discrete_gamma_delay_pmf(shape=2.0, scale=1.0, max_delay=30)
    mu = expected_reported_cases_delayed(incidence=inc, reporting_rate=0.25, delay_pmf=w)

    rng = np.random.default_rng(123)
    y = sample_nbinom_reports(expected=mu, dispersion=10.0, rng=rng)

    assert inc.shape == mu.shape == y.shape
    assert np.all(mu >= 0)
    assert np.all(y >= 0)

    ll = nbinom_loglik(y=y, mu=mu, dispersion=10.0)
    assert np.isfinite(ll)
