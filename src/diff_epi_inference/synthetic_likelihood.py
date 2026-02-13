from __future__ import annotations

from collections.abc import Callable

import numpy as np


def estimate_summary_gaussian(
    simulate_summary_fn: Callable[[np.random.Generator], np.ndarray],
    *,
    n_sims: int,
    rng: np.random.Generator,
    cov_jitter: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a Gaussian approximation to summary statistics.

    This is a small utility for synthetic-likelihood methods, where we model
    summary statistics as approximately Gaussian:

        s(y) | theta  ~  N(mu(theta), Sigma(theta)).

    The user provides a callable which simulates once (at fixed theta, captured
    by closure) and returns the summary vector.

    Parameters
    ----------
    simulate_summary_fn:
        Callable taking an RNG and returning a 1D summary vector (shape (d,)).
    n_sims:
        Number of simulator replicates to use.
    rng:
        Random number generator.
    cov_jitter:
        Non-negative diagonal jitter added to the empirical covariance.

    Returns
    -------
    (mu, cov):
        Empirical mean and (jittered) covariance of the summaries.
    """

    if n_sims <= 0:
        raise ValueError("n_sims must be positive")
    if cov_jitter < 0:
        raise ValueError("cov_jitter must be non-negative")

    sims = [np.asarray(simulate_summary_fn(rng), dtype=float) for _ in range(n_sims)]
    sims = np.asarray(sims, dtype=float)

    if sims.ndim != 2:
        raise ValueError("simulate_summary_fn must return a 1D vector")

    mu = np.mean(sims, axis=0)
    cov = np.cov(sims, rowvar=False)
    cov = np.atleast_2d(cov)
    cov = cov + float(cov_jitter) * np.eye(cov.shape[0])

    return mu, cov


def mvn_logpdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """Log N(x | mean, cov) for a (possibly) multivariate normal.

    Uses a stable solve + log-determinant. Returns -inf if cov is not
    positive-definite.
    """

    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)

    if x.shape != mean.shape:
        raise ValueError("x and mean must have the same shape")

    d = x.size
    diff = x - mean

    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return -np.inf

    sol = np.linalg.solve(cov, diff)
    quad = float(diff @ sol)

    return float(-0.5 * (d * np.log(2 * np.pi) + logdet + quad))
