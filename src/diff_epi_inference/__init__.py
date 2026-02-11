"""Minimal companion code for the Diff-Epi Inference Handbook."""

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
from .seir import SEIRParams, simulate_seir_euler

__all__ = [
    "SEIRParams",
    "simulate_seir_euler",
    "incidence_from_susceptibles",
    "expected_reported_cases",
    "sample_poisson_reports",
    "discrete_gamma_delay_pmf",
    "apply_delay",
    "expected_reported_cases_delayed",
    "sample_nbinom_reports",
    "nbinom_loglik",
]
