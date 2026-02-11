from __future__ import annotations

import math

import numpy as np


def loggamma(x: np.ndarray | float) -> np.ndarray | float:
    """Vectorized log-gamma using math.lgamma."""

    if np.isscalar(x):
        return math.lgamma(float(x))
    x = np.asarray(x, dtype=float)
    return np.vectorize(math.lgamma, otypes=[float])(x)


def nbinom_logpmf(*, k: np.ndarray, mu: np.ndarray, dispersion: float) -> np.ndarray:
    """Negative binomial log-PMF with mean/dispersion parameterisation.

    We use the NB2 style parameterisation:

      Var[Y] = mu + mu^2 / dispersion

    where dispersion > 0. As dispersion -> +inf, this approaches a Poisson.

    Parameters
    ----------
    k:
        Observations (non-negative integers).
    mu:
        Mean parameter (non-negative).
    dispersion:
        Positive dispersion parameter.

    Returns
    -------
    np.ndarray
        Elementwise log-probabilities.
    """

    if dispersion <= 0:
        raise ValueError("dispersion must be > 0")

    k = np.asarray(k)
    mu = np.asarray(mu, dtype=float)

    if np.any(mu < 0):
        raise ValueError("mu must be non-negative")
    if np.any(k < 0):
        raise ValueError("k must be non-negative")

    # Support broadcasting
    k = k.astype(float)

    r = float(dispersion)
    # Convert to (r, p) parameterisation where:
    #   P(Y=k) = C(k+r-1,k) (1-p)^k p^r
    # with mean mu = r(1-p)/p -> p = r/(r+mu)
    p = r / (r + mu)

    # Handle mu==0 safely: then p==1, log(1-p)=-inf; but k must be 0 to be valid.
    # We compute in a way that yields 0 for k==0, -inf for k>0.
    logp = np.log(p)
    log1mp = np.log1p(-p)

    out = loggamma(k + r) - loggamma(r) - loggamma(k + 1.0) + r * logp + k * log1mp

    # Fix mu==0 edge case to avoid NaNs from 0 * -inf
    zero_mu = mu == 0
    if np.any(zero_mu):
        out = np.where(zero_mu & (k == 0), 0.0, out)
        out = np.where(zero_mu & (k > 0), -np.inf, out)

    return out
