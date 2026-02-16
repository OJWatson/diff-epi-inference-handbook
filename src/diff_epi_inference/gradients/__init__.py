"""Gradient estimators and relaxations.

This package collects small, dependency-light implementations of:

- score-function / REINFORCE estimators for discrete latent variables
- continuous relaxations (e.g. Gumbel-Softmax)

The implementations are intentionally minimal and are meant for pedagogy and
small-scale experiments.
"""

from .reinforce import ReinforceResult, reinforce_grad_logit_bernoulli
from .relaxations import gumbel_softmax

__all__ = [
    "ReinforceResult",
    "reinforce_grad_logit_bernoulli",
    "gumbel_softmax",
]
