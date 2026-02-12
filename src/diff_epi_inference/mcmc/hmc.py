from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class HMCResult:
    """Output of a Hamiltonian Monte Carlo run."""

    chain: Array
    accepted: Array
    step_size: float
    n_leapfrog: int

    @property
    def accept_rate(self) -> float:
        return float(np.mean(self.accepted))


def _finite_difference_grad(
    log_prob_fn: Callable[[Array], float],
    x: Array,
    *,
    eps: float,
) -> Array:
    """Central finite-difference gradient for small pedagogical examples.

    Notes:
        This is *not* intended for performance. It exists so the repository can ship an HMC
        baseline without adding heavy autodiff dependencies.
    """

    x = np.asarray(x, dtype=float)
    grad = np.empty_like(x, dtype=float)

    for i in range(x.size):
        e = np.zeros_like(x, dtype=float)
        e.flat[i] = eps
        lp_plus = float(log_prob_fn(x + e))
        lp_minus = float(log_prob_fn(x - e))
        grad.flat[i] = (lp_plus - lp_minus) / (2.0 * eps)

    return grad


def hamiltonian_monte_carlo(
    log_prob_fn: Callable[[Array], float],
    x0: Array,
    *,
    n_steps: int,
    step_size: float,
    n_leapfrog: int,
    grad_eps: float = 1e-4,
    rng: np.random.Generator | None = None,
) -> HMCResult:
    """Minimal Hamiltonian Monte Carlo (HMC) sampler.

    This implementation uses a finite-difference gradient (central differences) to avoid
    adding heavyweight autodiff dependencies.

    Args:
        log_prob_fn: Function proportional to the log-density of the target distribution.
        x0: Initial state; can be a scalar-shaped array or a 1D vector.
        n_steps: Number of HMC transitions.
        step_size: Leapfrog step size ("epsilon").
        n_leapfrog: Number of leapfrog steps per proposal.
        grad_eps: Finite difference step size for gradient estimation.
        rng: Optional NumPy RNG.

    Returns:
        HMCResult with fields:
          - chain: array of shape (n_steps + 1, *x0.shape)
          - accepted: boolean array of shape (n_steps,)

    Notes:
        - Uses an isotropic unit mass matrix.
        - For real models, prefer autodiff + NUTS (e.g. NumPyro/BlackJAX), or at least an
          analytic gradient.
    """

    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if n_leapfrog <= 0:
        raise ValueError("n_leapfrog must be positive")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    if grad_eps <= 0.0:
        raise ValueError("grad_eps must be positive")

    rng = np.random.default_rng() if rng is None else rng

    x0 = np.asarray(x0, dtype=float)
    chain = np.empty((n_steps + 1, *x0.shape), dtype=float)
    accepted = np.zeros(n_steps, dtype=bool)

    x = x0
    lp = float(log_prob_fn(x))
    chain[0] = x

    for t in range(n_steps):
        p0 = rng.normal(loc=0.0, scale=1.0, size=x.shape)
        x_prop = np.array(x, copy=True)
        p_prop = np.array(p0, copy=True)

        # Leapfrog integrator.
        grad_lp = _finite_difference_grad(log_prob_fn, x_prop, eps=grad_eps)
        p_prop = p_prop + 0.5 * step_size * grad_lp

        for _ in range(n_leapfrog):
            x_prop = x_prop + step_size * p_prop
            grad_lp = _finite_difference_grad(log_prob_fn, x_prop, eps=grad_eps)
            p_prop = p_prop + step_size * grad_lp

        # Undo the extra half-step.
        p_prop = p_prop - 0.5 * step_size * grad_lp

        # Make proposal symmetric.
        p_prop = -p_prop

        lp_prop = float(log_prob_fn(x_prop))

        h0 = -lp + 0.5 * float(np.sum(p0**2))
        h1 = -lp_prop + 0.5 * float(np.sum(p_prop**2))

        log_alpha = -(h1 - h0)
        if np.log(rng.random()) < log_alpha:
            x = x_prop
            lp = lp_prop
            accepted[t] = True

        chain[t + 1] = x

    return HMCResult(chain=chain, accepted=accepted, step_size=step_size, n_leapfrog=n_leapfrog)
