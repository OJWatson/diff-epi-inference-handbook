from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class MHResult:
    """Output of a Metropolis–Hastings run."""

    chain: Array
    accepted: Array

    @property
    def accept_rate(self) -> float:
        return float(np.mean(self.accepted))


def random_walk_metropolis_hastings(
    log_prob_fn: Callable[[Array], float],
    x0: Array,
    *,
    proposal_std: float | Array = 1.0,
    n_steps: int,
    rng: np.random.Generator | None = None,
) -> MHResult:
    """Random-walk Metropolis–Hastings sampler.

    This is a minimal baseline implementation intended for pedagogical use.

    Args:
        log_prob_fn: Function proportional to the log-density of the target distribution.
        x0: Initial state; can be a scalar-shaped array or a 1D vector.
        proposal_std: Standard deviation of the isotropic Gaussian proposal.
            May be a float or an array broadcastable to the state shape.
        n_steps: Number of MH transitions.
        rng: Optional NumPy RNG.

    Returns:
        MHResult with fields:
          - chain: array of shape (n_steps + 1, *x0.shape)
          - accepted: boolean array of shape (n_steps,)

    Notes:
        Uses a symmetric proposal: x' = x + eps, eps ~ N(0, proposal_std^2).
    """

    if n_steps <= 0:
        raise ValueError("n_steps must be positive")

    rng = np.random.default_rng() if rng is None else rng

    x0 = np.asarray(x0, dtype=float)
    chain = np.empty((n_steps + 1, *x0.shape), dtype=float)
    accepted = np.zeros(n_steps, dtype=bool)

    x = x0
    lp = float(log_prob_fn(x))
    chain[0] = x

    prop_std = np.asarray(proposal_std, dtype=float)

    if not np.all(np.isfinite(prop_std)):
        raise ValueError("proposal_std must be finite")
    if np.any(prop_std <= 0):
        raise ValueError("proposal_std must be positive")

    try:
        prop_std = np.broadcast_to(prop_std, x.shape)
    except ValueError as e:
        raise ValueError("proposal_std must be broadcastable to x0.shape") from e

    for t in range(n_steps):
        proposal = x + rng.normal(loc=0.0, scale=prop_std, size=x.shape)
        lp_prop = float(log_prob_fn(proposal))

        log_alpha = lp_prop - lp
        if np.log(rng.random()) < log_alpha:
            x = proposal
            lp = lp_prop
            accepted[t] = True

        chain[t + 1] = x

    return MHResult(chain=chain, accepted=accepted)
