from __future__ import annotations

import numpy as np

from .dataset import TimeSeriesDataset
from .observation import (
    discrete_gamma_delay_pmf,
    expected_reported_cases_delayed,
    incidence_from_susceptibles,
    sample_nbinom_reports,
)
from .seir import SEIRParams, simulate_seir_euler, simulate_seir_stochastic_tau_leap


def reported_cases_dataset_from_S(
    *,
    t: np.ndarray,
    S: np.ndarray,
    reporting_rate: float,
    delay_shape: float = 2.0,
    delay_scale: float = 1.0,
    max_delay: int = 20,
    dispersion: float = 20.0,
    rng: np.random.Generator | None = None,
    name: str = "reported_cases",
) -> TimeSeriesDataset:
    """Build a TimeSeriesDataset of reported cases from a susceptible trajectory.

    The observation model is:
      i_t = max(S_t - S_{t+1}, 0)
      mu_t = delay( rho * i_t )
      y_t ~ NegBin(mu_t, dispersion)

    Notes
    -----
    If t has length T+1 and S has length T+1, the returned dataset has length T
    with time points t[1:]
    """

    t = np.asarray(t, dtype=float)
    S = np.asarray(S)

    if t.ndim != 1 or S.ndim != 1 or t.shape[0] != S.shape[0]:
        raise ValueError("t and S must be 1D arrays of the same length")
    if t.shape[0] < 2:
        raise ValueError("t/S must have length >= 2")

    inc = incidence_from_susceptibles(S.astype(float))
    w = discrete_gamma_delay_pmf(shape=delay_shape, scale=delay_scale, max_delay=max_delay)
    mu = expected_reported_cases_delayed(incidence=inc, reporting_rate=reporting_rate, delay_pmf=w)
    y = sample_nbinom_reports(expected=mu, dispersion=dispersion, rng=rng)

    return TimeSeriesDataset(t=t[1:], y=y, name=name)


def simulate_seir_and_report_deterministic(
    *,
    params: SEIRParams,
    s0: float,
    e0: float,
    i0: float,
    r0: float,
    dt: float,
    steps: int,
    reporting_rate: float,
    rng: np.random.Generator | None = None,
    name: str = "reported_cases_deterministic",
) -> TimeSeriesDataset:
    """Deterministic SEIR (Euler) + shared observation model -> dataset."""

    out = simulate_seir_euler(
        params=params,
        s0=s0,
        e0=e0,
        i0=i0,
        r0=r0,
        dt=dt,
        steps=steps,
    )

    return reported_cases_dataset_from_S(
        t=out["t"],
        S=out["S"],
        reporting_rate=reporting_rate,
        rng=rng,
        name=name,
    )


def simulate_seir_and_report_stochastic(
    *,
    params: SEIRParams,
    s0: int,
    e0: int,
    i0: int,
    r0: int,
    dt: float,
    steps: int,
    reporting_rate: float,
    rng: np.random.Generator | None = None,
    name: str = "reported_cases_stochastic",
) -> TimeSeriesDataset:
    """Stochastic SEIR (tau-leap) + shared observation model -> dataset."""

    out = simulate_seir_stochastic_tau_leap(
        params=params,
        s0=s0,
        e0=e0,
        i0=i0,
        r0=r0,
        dt=dt,
        steps=steps,
        rng=rng,
    )

    return reported_cases_dataset_from_S(
        t=out["t"],
        S=out["S"],
        reporting_rate=reporting_rate,
        rng=rng,
        name=name,
    )
