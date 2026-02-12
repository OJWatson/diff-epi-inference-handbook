"""NumPy implementation of the *minimal* SEIR beta-only posterior used in the book.

This helper exists to reduce drift between:

- the MH/HMC beta-only examples in `book/classical-baselines.qmd`, and
- the (in-book) beta-only calibration smoke tests.

It is intentionally scoped to *only* inferring ``beta`` via ``log_beta = log(beta)``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from diff_epi_inference import (
    SEIRParams,
    expected_reported_cases_delayed,
    incidence_from_susceptibles,
    nbinom_loglik,
    simulate_seir_euler,
)


def make_log_post_logbeta_numpy(
    *,
    y_obs: np.ndarray,
    w_delay_pmf: np.ndarray,
    sigma: float,
    gamma: float,
    s0: float,
    e0: float,
    i0: float,
    r0: float,
    dt: float,
    steps: int,
    reporting_rate: float,
    dispersion: float,
    logbeta_prior_mean: float,
    logbeta_prior_sd: float,
) -> Callable[[np.ndarray], float]:
    """Build a NumPy log posterior for SEIR with *only* beta inferred.

    The returned callable has signature ``log_post(position)`` where
    ``position.shape == (1,)`` and ``position[0] == log(beta)``.

    Notes
    -----
    - The delay PMF is passed through to ``expected_reported_cases_delayed``.
      Unlike the JAX version, we do not normalise it here.
    - Normalisation constants for the Gaussian prior are dropped.
    """

    y_obs = np.asarray(y_obs)
    w = np.asarray(w_delay_pmf, dtype=float)

    sigma = float(sigma)
    gamma = float(gamma)
    s0 = float(s0)
    e0 = float(e0)
    i0 = float(i0)
    r0 = float(r0)
    dt = float(dt)
    steps = int(steps)
    reporting_rate = float(reporting_rate)
    dispersion = float(dispersion)
    logbeta_prior_mean = float(logbeta_prior_mean)
    logbeta_prior_sd = float(logbeta_prior_sd)

    def log_post(position: np.ndarray) -> float:
        position = np.asarray(position, dtype=float)
        if position.shape != (1,):
            raise ValueError("position must have shape (1,)")

        logbeta = float(position[0])
        beta = float(np.exp(logbeta))

        lp = -0.5 * ((logbeta - logbeta_prior_mean) / logbeta_prior_sd) ** 2

        params = SEIRParams(beta=beta, sigma=sigma, gamma=gamma)
        out = simulate_seir_euler(
            params=params,
            s0=s0,
            e0=e0,
            i0=i0,
            r0=r0,
            dt=dt,
            steps=steps,
        )

        inc = incidence_from_susceptibles(out["S"])
        mu = expected_reported_cases_delayed(
            incidence=inc,
            reporting_rate=reporting_rate,
            delay_pmf=w,
        )

        ll = nbinom_loglik(y=y_obs, mu=mu, dispersion=dispersion)
        return float(lp + ll)

    return log_post
