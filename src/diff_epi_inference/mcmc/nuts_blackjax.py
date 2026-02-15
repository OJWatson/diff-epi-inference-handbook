"""Optional NUTS baseline via BlackJAX.

The handbook aims to keep the core library dependency-light. For a full-strength gradient-based
MCMC baseline (autodiff + NUTS), we rely on *optional* JAX/BlackJAX dependencies.

This module is safe to import without JAX installed: imports happen inside functions.

References:
  - https://github.com/blackjax-devs/blackjax
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class BlackJAXNUTSResult:
    """Result of a BlackJAX NUTS run.

    Attributes:
        chain: Array of shape (num_samples, dim) containing post-warmup draws.
        accept_rate: Mean acceptance probability reported by the kernel.
    """

    chain: Array
    accept_rate: float


def run_blackjax_nuts(
    log_prob_fn: Callable[[object], object],
    x0: Array,
    *,
    num_warmup: int = 500,
    num_samples: int = 1_000,
    seed: int = 0,
) -> BlackJAXNUTSResult:
    """Run NUTS using BlackJAX window adaptation.

    Args:
        log_prob_fn: Python callable returning log density at a NumPy array position.
            The function must be compatible with JAX tracing; in practice this means it should
            be implemented using ``jax.numpy`` if you want gradients.
        x0: Initial state as a 1D NumPy array.
        num_warmup: Number of warmup / adaptation steps.
        num_samples: Number of post-warmup samples to draw.
        seed: PRNG seed.

    Returns:
        BlackJAXNUTSResult containing the chain and mean acceptance probability.

    Raises:
        ImportError: if JAX/BlackJAX are not installed.
    """

    try:
        import blackjax
        import jax
        import jax.numpy as jnp
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Optional dependency missing. Install with `pip install -e .[nuts]` "
            "to enable BlackJAX NUTS."
        ) from e

    if num_warmup <= 0:
        raise ValueError("num_warmup must be positive")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    x0 = np.asarray(x0, dtype=float)
    if x0.ndim != 1:
        raise ValueError("x0 must be a 1D array")

    # Wrap the user log-prob so it is JAX-friendly.
    # Users should implement log_prob_fn using jax.numpy so gradients are available.
    def logdensity(position: jax.Array) -> jax.Array:
        return jnp.asarray(log_prob_fn(position))

    rng_key = jax.random.PRNGKey(seed)
    initial_position = jnp.asarray(x0)

    # Window adaptation chooses step size and mass matrix.
    adaptation = blackjax.window_adaptation(
        blackjax.nuts,
        logdensity,
        target_acceptance_rate=0.8,
    )

    rng_key, adapt_key, sample_key = jax.random.split(rng_key, 3)
    (state, parameters), _ = adaptation.run(adapt_key, initial_position, num_warmup)

    kernel = blackjax.nuts(logdensity, **parameters).step

    def one_step(carry, _):
        state, key = carry
        key, subkey = jax.random.split(key)
        state, info = kernel(subkey, state)

        # BlackJAX changed this field name across versions.
        # - older: acceptance_probability
        # - newer: acceptance_rate
        try:
            acc = info.acceptance_probability
        except AttributeError:  # pragma: no cover
            acc = info.acceptance_rate

        return (state, key), (state.position, acc)

    (state, _), (positions, acc_probs) = jax.lax.scan(
        one_step,
        (state, sample_key),
        xs=None,
        length=num_samples,
    )

    chain = np.asarray(positions)
    accept_rate = float(np.mean(np.asarray(acc_probs)))
    return BlackJAXNUTSResult(chain=chain, accept_rate=accept_rate)
