"""Likelihood-free (ABC) baselines.

This package contains minimal, pedagogical implementations of approximate Bayesian
computation algorithms used in the handbook.
"""

from .rejection import ABCRejectionResult, abc_rejection

__all__ = ["ABCRejectionResult", "abc_rejection"]
