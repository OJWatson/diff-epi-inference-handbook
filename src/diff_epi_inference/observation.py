from __future__ import annotations

import math

import numpy as np

from .stats import nbinom_logpmf


def incidence_from_susceptibles(S: np.ndarray) -> np.ndarray:
    """Compute per-step incidence from a susceptible trajectory.

    Parameters
    ----------
    S:
        Susceptible compartment values of length T+1.

    Returns
    -------
    np.ndarray
        Incidence per step of length T, defined as max(S[t] - S[t+1], 0).

    Notes
    -----
    For the deterministic SEIR Euler solver, S is non-increasing for reasonable dt,
    so this corresponds to new infections per time step.
    """

    S = np.asarray(S, dtype=float)
    if S.ndim != 1 or S.shape[0] < 2:
        raise ValueError("S must be a 1D array of length >= 2")

    inc = S[:-1] - S[1:]
    return np.maximum(inc, 0.0)


def expected_reported_cases(*, incidence: np.ndarray, reporting_rate: float) -> np.ndarray:
    """Simple observation model: expected reported cases per step.

    E[y_t] = rho * incidence_t

    Parameters
    ----------
    incidence:
        Array of non-negative incidence values.
    reporting_rate:
        rho in [0, 1].
    """

    if not (0.0 <= reporting_rate <= 1.0):
        raise ValueError("reporting_rate must be in [0, 1]")

    incidence = np.asarray(incidence, dtype=float)
    if np.any(incidence < 0):
        raise ValueError("incidence must be non-negative")

    return reporting_rate * incidence


def sample_poisson_reports(
    *,
    expected: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample reported cases using a Poisson observation model."""

    expected = np.asarray(expected, dtype=float)
    if np.any(expected < 0):
        raise ValueError("expected must be non-negative")

    if rng is None:
        rng = np.random.default_rng()

    return rng.poisson(lam=expected)


def discrete_gamma_delay_pmf(*, shape: float, scale: float, max_delay: int) -> np.ndarray:
    """A lightweight discrete delay distribution using a Gamma(shape, scale).

    Returns probabilities for delays d=0..max_delay inclusive, normalised to sum to 1.

    Notes
    -----
    This is a pragmatic helper intended for runnable examples. It uses a midpoint
    approximation of the Gamma PDF evaluated at (d + 0.5).
    """

    if shape <= 0 or scale <= 0:
        raise ValueError("shape and scale must be > 0")
    if max_delay < 0:
        raise ValueError("max_delay must be >= 0")

    d = np.arange(max_delay + 1, dtype=float)
    x = d + 0.5

    # Gamma pdf: x^{k-1} exp(-x/theta) / (Gamma(k) theta^k)
    logpdf = (shape - 1.0) * np.log(x) - x / scale - math.lgamma(shape) - shape * np.log(scale)
    w = np.exp(logpdf)
    w = w / w.sum()
    return w


def apply_delay(*, incidence: np.ndarray, delay_pmf: np.ndarray) -> np.ndarray:
    """Convolve incidence with a discrete delay PMF.

    incidence has length T; delay_pmf has length D.
    Returns expected events of length T (truncated to observed window).
    """

    incidence = np.asarray(incidence, dtype=float)
    delay_pmf = np.asarray(delay_pmf, dtype=float)

    if incidence.ndim != 1:
        raise ValueError("incidence must be 1D")
    if delay_pmf.ndim != 1 or delay_pmf.shape[0] < 1:
        raise ValueError("delay_pmf must be 1D with length >= 1")
    if np.any(incidence < 0):
        raise ValueError("incidence must be non-negative")
    if np.any(delay_pmf < 0):
        raise ValueError("delay_pmf must be non-negative")

    if delay_pmf.sum() <= 0:
        raise ValueError("delay_pmf must have positive mass")

    delay_pmf = delay_pmf / delay_pmf.sum()

    full = np.convolve(incidence, delay_pmf, mode="full")
    return full[: incidence.shape[0]]


def expected_reported_cases_delayed(
    *,
    incidence: np.ndarray,
    reporting_rate: float,
    delay_pmf: np.ndarray,
) -> np.ndarray:
    """Delayed + under-reported expected case counts.

    Let i_t be incidence (new infections per step). We model expected reports as:

      mu_t = rho * sum_{d>=0} i_{t-d} * w_d

    where w_d is a discrete delay PMF.
    """

    base = expected_reported_cases(incidence=incidence, reporting_rate=reporting_rate)
    return apply_delay(incidence=base, delay_pmf=delay_pmf)


def nbinom_loglik(*, y: np.ndarray, mu: np.ndarray, dispersion: float) -> float:
    """Sum of NB log-likelihoods for observed counts."""

    y = np.asarray(y)
    mu = np.asarray(mu, dtype=float)

    if y.shape != mu.shape:
        raise ValueError("y and mu must have the same shape")

    ll = nbinom_logpmf(k=y, mu=mu, dispersion=dispersion)
    return float(np.sum(ll))


def sample_nbinom_reports(
    *,
    expected: np.ndarray,
    dispersion: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample reported cases using an NB2 observation model."""

    if dispersion <= 0:
        raise ValueError("dispersion must be > 0")

    expected = np.asarray(expected, dtype=float)
    if np.any(expected < 0):
        raise ValueError("expected must be non-negative")

    if rng is None:
        rng = np.random.default_rng()

    r = float(dispersion)
    p = r / (r + expected)
    # numpy uses n, p with mean n(1-p)/p.
    return rng.negative_binomial(n=r, p=p)
