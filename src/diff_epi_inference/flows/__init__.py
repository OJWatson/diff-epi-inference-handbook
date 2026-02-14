"""Normalising-flow style density models.

This module intentionally starts with very small, dependency-light components.
The goal is pedagogical (for the handbook) and to provide runnable baselines.

The first implemented flow is a conditional diagonal affine Gaussian:

  theta = mu(context) + sigma(context) * z,   z ~ N(0, I)

which is a valid conditional normalising flow with tractable log density.
"""

from .conditional_affine import ConditionalAffineDiagNormal

__all__ = ["ConditionalAffineDiagNormal"]
