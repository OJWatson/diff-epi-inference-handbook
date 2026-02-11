import numpy as np
import pytest

from diff_epi_inference import (
    SEIRParams,
    expected_reported_cases,
    incidence_from_susceptibles,
    sample_poisson_reports,
    simulate_seir_euler,
)


def test_incidence_from_susceptibles_matches_drop_in_S():
    out = simulate_seir_euler(
        params=SEIRParams(beta=0.6, sigma=1 / 5, gamma=1 / 7),
        s0=999.0,
        e0=0.0,
        i0=1.0,
        r0=0.0,
        dt=0.2,
        steps=50,
    )
    inc = incidence_from_susceptibles(out["S"])

    assert inc.shape == (50,)
    assert np.all(inc >= 0)

    # By definition: incidence = max(S[t] - S[t+1], 0)
    assert np.allclose(inc, np.maximum(out["S"][:-1] - out["S"][1:], 0.0))


def test_expected_reported_cases_scaling_and_bounds():
    inc = np.array([0.0, 1.0, 2.5])

    mu = expected_reported_cases(incidence=inc, reporting_rate=0.3)
    assert np.allclose(mu, 0.3 * inc)

    with pytest.raises(ValueError):
        expected_reported_cases(incidence=inc, reporting_rate=-0.1)
    with pytest.raises(ValueError):
        expected_reported_cases(incidence=inc, reporting_rate=1.1)


def test_sample_poisson_reports_shape_and_nonnegative():
    rng = np.random.default_rng(0)
    expected = np.array([0.0, 1.0, 10.0])
    y = sample_poisson_reports(expected=expected, rng=rng)

    assert y.shape == expected.shape
    assert np.issubdtype(y.dtype, np.integer)
    assert np.all(y >= 0)
