from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class ReinforceResult:
    """Result container for a REINFORCE gradient estimate."""

    grad: float
    baseline: float
    mean_f: float
    p: float
    n_samples: int


def reinforce_grad_logit_bernoulli(
    f: Callable[[np.ndarray], np.ndarray],
    *,
    logit: float,
    rng: np.random.Generator,
    n_samples: int,
    baseline: Literal["none", "mean"] = "mean",
) -> ReinforceResult:
    """Estimate d/dlogit E[f(z)] where z ~ Bernoulli(sigmoid(logit)).

    Uses the score-function / REINFORCE identity:

      d/dφ E[f(z)] = E[f(z) * d/dφ log p(z; φ)].

    For a Bernoulli parameterised by a logit φ, we have:

      d/dφ log p(z; φ) = z - sigmoid(φ).

    Parameters
    ----------
    f:
        A function mapping samples z with shape (n_samples,) to values with the
        same shape.
    logit:
        Bernoulli logit φ.
    rng:
        NumPy random generator.
    n_samples:
        Number of Monte Carlo samples.
    baseline:
        Control variate. "mean" subtracts the sample mean of f(z).

    Returns
    -------
    ReinforceResult
        grad is the Monte Carlo estimate.

    Notes
    -----
    - This estimator is unbiased for any baseline b that does not depend on z.
    - Using the sample mean as a baseline is common and tends to reduce variance.
    """

    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    p = float(sigmoid(logit))
    z = rng.binomial(n=1, p=p, size=n_samples).astype(float)

    fz = np.asarray(f(z))
    if fz.shape != z.shape:
        raise ValueError(f"f(z) must return shape {z.shape}, got {fz.shape}")

    score = z - p

    if baseline == "none":
        b = 0.0
    elif baseline == "mean":
        b = float(np.mean(fz))
    else:
        raise ValueError("baseline must be 'none' or 'mean'")

    grad_hat = float(np.mean((fz - b) * score))

    return ReinforceResult(
        grad=grad_hat,
        baseline=b,
        mean_f=float(np.mean(fz)),
        p=p,
        n_samples=n_samples,
    )
