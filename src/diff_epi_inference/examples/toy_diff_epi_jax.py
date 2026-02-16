"""Toy *differentiable epidemiology* case study (JAX, relaxed stochastic transitions).

This module is intentionally small and self-contained: it provides a miniature
SIR-like simulator where the *discrete* transition counts are replaced by a
smooth relaxation so that gradients can flow end-to-end.

The goal is pedagogical (M6.4): demonstrate the pattern

    simulator -> loss/likelihood -> grad -> gradient-based fit

without claiming this is a faithful epidemic model.

JAX is an optional dependency. Public helpers import JAX lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _logit(p: Any) -> Any:
    """Numerically stable logit for p in (0, 1)."""

    import jax.numpy as jnp

    p = jnp.clip(p, 1e-6, 1.0 - 1e-6)
    return jnp.log(p) - jnp.log1p(-p)


def _relaxed_binomial_count(
    key: Any,
    *,
    n: Any,
    p: Any,
    temperature: float,
) -> Any:
    """Relaxed Binomial(n, p) draw in [0, n] using a Binary Concrete sample.

    We use a *single* relaxed Bernoulli sample to represent the fraction of
    successes, then scale by n. This is a coarse but very cheap relaxation.

    Notes
    -----
    If you need a closer approximation, you could sum many relaxed Bernoullis,
    but that is outside the scope of the toy case study.
    """

    import jax
    import jax.numpy as jnp

    u = jax.random.uniform(key, shape=(), minval=1e-6, maxval=1.0 - 1e-6)
    logistic_noise = jnp.log(u) - jnp.log1p(-u)

    y = jax.nn.sigmoid((_logit(p) + logistic_noise) / float(temperature))
    return jnp.clip(n * y, 0.0, n)


@dataclass(frozen=True)
class ToySIRPath:
    s: np.ndarray
    i: np.ndarray
    r: np.ndarray
    new_infections: np.ndarray
    new_recoveries: np.ndarray


def _simulate_toy_relaxed_sir_jax(
    *,
    seed: int,
    beta: Any,
    gamma: Any,
    s0: float,
    i0: float,
    r0: float,
    steps: int,
    temperature: float,
):
    """JAX-only simulator returning JAX arrays.

    This is separated from the public NumPy-returning wrapper so that we can
    differentiate through the simulator without accidentally converting tracers
    to NumPy arrays.
    """

    import jax
    import jax.numpy as jnp

    # beta may be a JAX tracer when differentiating through the simulator.
    beta_j = jnp.asarray(beta, dtype=float)
    gamma_j = jnp.asarray(gamma, dtype=float)

    if steps <= 0:
        raise ValueError("steps must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    key = jax.random.PRNGKey(int(seed))

    s0_j = jnp.asarray(s0, dtype=float)
    i0_j = jnp.asarray(i0, dtype=float)
    r0_j = jnp.asarray(r0, dtype=float)

    def step_fn(carry, _):
        s, i, r, key = carry
        key_inf, key_rec, key_next = jax.random.split(key, 3)

        n = s + i + r
        p_inf = 1.0 - jnp.exp(-beta_j * i / jnp.maximum(n, 1e-6))
        p_rec = 1.0 - jnp.exp(-gamma_j)

        new_inf = _relaxed_binomial_count(key_inf, n=s, p=p_inf, temperature=temperature)
        new_rec = _relaxed_binomial_count(key_rec, n=i, p=p_rec, temperature=temperature)

        s1 = s - new_inf
        i1 = i + new_inf - new_rec
        r1 = r + new_rec

        return (s1, i1, r1, key_next), (s1, i1, r1, new_inf, new_rec)

    init = (s0_j, i0_j, r0_j, key)
    (_, _, _, _), outs = jax.lax.scan(step_fn, init, xs=None, length=int(steps))

    s_hist, i_hist, r_hist, new_inf, new_rec = outs

    s_all = jnp.concatenate([jnp.asarray([s0_j]), s_hist], axis=0)
    i_all = jnp.concatenate([jnp.asarray([i0_j]), i_hist], axis=0)
    r_all = jnp.concatenate([jnp.asarray([r0_j]), r_hist], axis=0)

    return s_all, i_all, r_all, new_inf, new_rec


def simulate_toy_relaxed_sir(
    *,
    seed: int,
    beta: float,
    gamma: float,
    s0: float,
    i0: float,
    r0: float,
    steps: int,
    temperature: float = 0.5,
) -> ToySIRPath:
    """Simulate a toy relaxed SIR trajectory.

    Returns NumPy arrays; use the internal JAX helper for differentiation.
    """

    import jax

    temperature_f = float(temperature)
    s_all, i_all, r_all, new_inf, new_rec = _simulate_toy_relaxed_sir_jax(
        seed=seed,
        beta=beta,
        gamma=gamma,
        s0=s0,
        i0=i0,
        r0=r0,
        steps=steps,
        temperature=temperature_f,
    )

    s_all, i_all, r_all, new_inf, new_rec = jax.device_get(
        (s_all, i_all, r_all, new_inf, new_rec)
    )

    return ToySIRPath(
        s=np.asarray(s_all),
        i=np.asarray(i_all),
        r=np.asarray(r_all),
        new_infections=np.asarray(new_inf),
        new_recoveries=np.asarray(new_rec),
    )


@dataclass(frozen=True)
class ToyGradFitResult:
    beta_init: float
    beta_hat: float
    losses: np.ndarray


def fit_beta_by_gradient_descent(
    *,
    y_obs: np.ndarray,
    seed: int,
    beta_init: float,
    gamma: float,
    s0: float,
    i0: float,
    r0: float,
    steps: int,
    temperature: float = 0.5,
    lr: float = 0.1,
    iters: int = 50,
) -> ToyGradFitResult:
    """Fit beta by differentiating through the relaxed simulator.

    We use a simple squared-error loss between observed *incidence* and the
    simulator's relaxed incidence path.

    Returns
    -------
    ToyGradFitResult
        Includes the loss trace for smoke-testing and plotting.
    """

    import jax
    import jax.numpy as jnp

    y_obs = np.asarray(y_obs, dtype=float)
    if y_obs.shape != (steps,):
        raise ValueError(f"y_obs must have shape ({steps},)")

    def loss_fn(logbeta):
        beta = jnp.exp(logbeta)
        _, _, _, new_inf, _ = _simulate_toy_relaxed_sir_jax(
            seed=seed,
            beta=beta,
            gamma=gamma,
            s0=s0,
            i0=i0,
            r0=r0,
            steps=steps,
            temperature=float(temperature),
        )
        y_hat = jnp.asarray(new_inf)
        # Normalise to make the scale somewhat independent of the run length
        return jnp.mean((y_hat - jnp.asarray(y_obs)) ** 2)

    grad_fn = jax.grad(loss_fn)

    logbeta = jnp.log(jnp.asarray(beta_init, dtype=float))
    losses = []
    for _ in range(int(iters)):
        loss_val = loss_fn(logbeta)
        g = grad_fn(logbeta)
        losses.append(loss_val)
        logbeta = logbeta - float(lr) * g

    beta_hat = float(np.exp(np.asarray(logbeta)))

    return ToyGradFitResult(
        beta_init=float(beta_init),
        beta_hat=beta_hat,
        losses=np.asarray(jax.device_get(jnp.stack(losses))),
    )
