"""Minimal companion code for the Diff-Epi Inference Handbook."""

from .dataset import TimeSeriesDataset, from_cases
from .observation import (
    apply_delay,
    discrete_gamma_delay_pmf,
    expected_reported_cases,
    expected_reported_cases_delayed,
    incidence_from_susceptibles,
    nbinom_loglik,
    sample_nbinom_reports,
    sample_poisson_reports,
)
from .pipeline import (
    reported_cases_dataset_from_S,
    simulate_seir_and_report_deterministic,
    simulate_seir_and_report_stochastic,
)
from .plotting import plot_timeseries
from .seir import SEIRParams, simulate_seir_euler, simulate_seir_stochastic_tau_leap

__all__ = [
    "SEIRParams",
    "simulate_seir_euler",
    "simulate_seir_stochastic_tau_leap",
    "incidence_from_susceptibles",
    "expected_reported_cases",
    "sample_poisson_reports",
    "discrete_gamma_delay_pmf",
    "apply_delay",
    "expected_reported_cases_delayed",
    "sample_nbinom_reports",
    "nbinom_loglik",
    "TimeSeriesDataset",
    "from_cases",
    "plot_timeseries",
    "reported_cases_dataset_from_S",
    "simulate_seir_and_report_deterministic",
    "simulate_seir_and_report_stochastic",
]
