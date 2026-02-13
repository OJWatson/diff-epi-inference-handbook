"""Likelihood-free (ABC) baselines.

This package contains minimal, pedagogical implementations of approximate Bayesian
computation algorithms used in the handbook.
"""

from .rejection import ABCRejectionResult, abc_rejection
from .smc import systematic_resample

__all__ = ["ABCRejectionResult", "abc_rejection", "systematic_resample"]
