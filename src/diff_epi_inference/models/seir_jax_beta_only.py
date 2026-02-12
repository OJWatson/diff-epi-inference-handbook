"""JAX reimplementation of the *minimal* SEIR beta-only posterior used in the book.

This module exists to reduce drift between:

- the optional BlackJAX NUTS demo in `book/classical-baselines.qmd`, and
- the optional calibration/coverage smoke test
  `tests/test_blackjax_nuts_seir_calibration_optional.py`.

It is intentionally scoped to the **beta-only** likelihood:

- SEIR dynamics via Euler steps (S, E, I, R)
- incidence from susceptibles
- delayed reporting via discrete convolution
- Negative Binomial observation model

JAX is an *optional* dependency. To keep base installs working, this module does
not import JAX at import time; the public helpers import JAX lazily.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def make_log_post_logbeta_jax(
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
) -> Callable[[Any], Any]:
    """Build a JAX-traceable log posterior for SEIR with *only* beta inferred.

    The returned callable has signature `log_post(position)` where
    `position.shape == (1,)` and `position[0] == log(beta)`.

    Parameters
    ----------
    y_obs:
        Observed reported cases at each time step (length `steps`).
    w_delay_pmf:
        Discrete delay PMF used for reporting convolution. Does not need to be
        normalised.
    sigma, gamma:
        SEIR fixed rates.
    s0, e0, i0, r0:
        Initial conditions.
    dt, steps:
        Euler step size and number of steps.
    reporting_rate:
        Fraction of incident infections that become reported cases.
    dispersion:
        Negative Binomial dispersion (size) parameter.
    logbeta_prior_mean, logbeta_prior_sd:
        Gaussian prior parameters on log(beta).

    Returns
    -------
    log_post:
        JAX function returning a scalar log posterior.

    Notes
    -----
    - We mirror the NumPy running-example implementation closely, but keep the
      JAX path deliberately minimal.
    - This is not a general-purpose model API.
    """

    import jax
    import jax.numpy as jnp
    from jax.scipy.special import gammaln

    y_obs_jax = jnp.asarray(y_obs)

    w_jax = jnp.asarray(w_delay_pmf, dtype=float)
    w_jax = w_jax / jnp.sum(w_jax)

    sigma_j = jnp.asarray(sigma)
    gamma_j = jnp.asarray(gamma)

    dt_j = jnp.asarray(dt)
    steps_j = int(steps)

    s0_j = jnp.asarray(s0)
    e0_j = jnp.asarray(e0)
    i0_j = jnp.asarray(i0)
    r0_j = jnp.asarray(r0)

    dispersion_j = float(dispersion)

    def simulate_seir_euler_s_path(beta: jax.Array) -> jax.Array:
        beta = jnp.asarray(beta)

        def step_fn(state, _):
            s, e, i, r = state
            n = s + e + i + r
            inf_flow = beta * s * i / n
            inc_flow = sigma_j * e
            rec_flow = gamma_j * i
            s1 = s - dt_j * inf_flow
            e1 = e + dt_j * (inf_flow - inc_flow)
            i1 = i + dt_j * (inc_flow - rec_flow)
            r1 = r + dt_j * rec_flow
            return (s1, e1, i1, r1), s1

        init = (s0_j, e0_j, i0_j, r0_j)
        (_, _, _, _), s_hist = jax.lax.scan(step_fn, init, xs=None, length=steps_j)

        # Prepend s0 so we have length steps+1 like the NumPy solver.
        return jnp.concatenate([jnp.asarray([s0]), s_hist], axis=0)

    def nbinom_logpmf(k: jax.Array, mu: jax.Array) -> jax.Array:
        r = jnp.asarray(dispersion_j)
        p = r / (r + mu)
        logp = jnp.log(p)
        log1mp = jnp.log1p(-p)
        kf = k.astype(float)
        out = gammaln(kf + r) - gammaln(r) - gammaln(kf + 1.0) + r * logp + kf * log1mp

        zero_mu = mu == 0
        out = jnp.where(zero_mu & (kf == 0), 0.0, out)
        out = jnp.where(zero_mu & (kf > 0), -jnp.inf, out)
        return out

    def log_post(position: jax.Array) -> jax.Array:
        logbeta = position[0]
        beta = jnp.exp(logbeta)

        lp = -0.5 * ((logbeta - logbeta_prior_mean) / logbeta_prior_sd) ** 2

        s_all = simulate_seir_euler_s_path(beta)
        inc = jnp.maximum(s_all[:-1] - s_all[1:], 0.0)

        base = reporting_rate * inc
        mu = jnp.convolve(base, w_jax, mode="full")[: base.shape[0]]

        ll = jnp.sum(nbinom_logpmf(y_obs_jax, mu))
        return lp + ll

    return log_post
