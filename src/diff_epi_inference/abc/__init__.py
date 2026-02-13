"""Likelihood-free (ABC) baselines.

This package contains minimal, pedagogical implementations of approximate Bayesian
computation algorithms used in the handbook.
"""

from .rejection import ABCRejectionResult, abc_rejection
from .smc import SMCABCResult, smc_abc, systematic_resample

__all__ = [
    "ABCRejectionResult",
    "SMCABCResult",
    "abc_rejection",
    "smc_abc",
    "systematic_resample",
]
