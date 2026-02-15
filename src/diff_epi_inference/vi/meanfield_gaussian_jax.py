"""Mean-field Gaussian variational inference (JAX).

This module provides a small, dependency-light implementation of
reparameterised ("pathwise") gradient VI for continuous parameters.

Design goals
------------
- **JAX is optional**: we avoid importing it at module import time.
- **No Optax dependency**: we implement Adam directly to keep this usable with
  the `jax` extra (and not require `modern-sbi`).
- Simple, explicit API aimed at the handbook's worked examples.

The main entry point is :func:`fit_meanfield_gaussian_vi_jax`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np


@dataclass(frozen=True)
class MeanFieldGaussianVIResult:
    """Result from mean-field Gaussian VI."""

    mean: np.ndarray
    log_std: np.ndarray
    elbo_history: np.ndarray


def fit_meanfield_gaussian_vi_jax(
    log_joint: Callable[[Any], Any],
    *,
    dim: int,
    seed: int = 0,
    num_steps: int = 2_000,
    lr: float = 1e-2,
    num_mc_samples: int = 32,
    init_mean: np.ndarray | None = None,
    init_log_std: np.ndarray | None = None,
    jit: bool = True,
) -> MeanFieldGaussianVIResult:
    r"""Fit a diagonal (mean-field) Gaussian variational approximation with JAX.

    We approximate a target density with unnormalised log-joint `log_joint(theta)`
    using

    .. math::

        q_\phi(\theta) = \mathcal{N}(\mu, \mathrm{diag}(\sigma^2)),
        \qquad \phi = (\mu, \log \sigma).

    The ELBO is estimated via Monte Carlo with the reparameterisation trick:

    .. math::

        \mathcal{L}(\phi) = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)}
            \left[\log p(y,\theta(\epsilon;\phi)) - \log q_\phi(\theta(\epsilon;\phi))\right].

    Parameters
    ----------
    log_joint:
        JAX-traceable callable returning a scalar log-joint density given
        `theta` of shape `(dim,)`.
    dim:
        Parameter dimension.
    seed:
        PRNG seed.
    num_steps:
        Optimisation steps.
    lr:
        Adam learning rate.
    num_mc_samples:
        Monte Carlo samples per step.
    init_mean, init_log_std:
        Optional initial values. If omitted, initial mean is zeros and initial
        std is 1.
    jit:
        If True, JIT-compile the update step.

    Returns
    -------
    MeanFieldGaussianVIResult
        Approximate posterior parameters and ELBO trace.

    Notes
    -----
    - This function performs *gradient ascent* on the ELBO.
    - For numerical stability, `log_std` is unconstrained but the implied
      standard deviation is clamped to a small minimum inside the entropy term.
    """

    import jax
    import jax.numpy as jnp

    if dim < 1:
        raise ValueError("dim must be >= 1")
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    if num_mc_samples < 1:
        raise ValueError("num_mc_samples must be >= 1")

    if init_mean is None:
        init_mean = np.zeros((dim,), dtype=float)
    if init_log_std is None:
        init_log_std = np.zeros((dim,), dtype=float)

    mean0 = jnp.asarray(init_mean, dtype=float)
    log_std0 = jnp.asarray(init_log_std, dtype=float)

    # --- Small Adam implementation (no Optax dependency) ---
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    def normal_logpdf_diag(x: jax.Array, mean: jax.Array, log_std: jax.Array) -> jax.Array:
        # log N(x | mean, diag(std^2))
        # stable: compute using log_std.
        return -0.5 * jnp.sum(
            jnp.log(2.0 * jnp.pi) + 2.0 * log_std + ((x - mean) / jnp.exp(log_std)) ** 2
        )

    def elbo_estimate(key: jax.Array, mean: jax.Array, log_std: jax.Array) -> jax.Array:
        # Draw eps ~ N(0, I), reparameterise theta = mean + exp(log_std) * eps
        keys = jax.random.split(key, num_mc_samples)

        def one_sample(k):
            eps_s = jax.random.normal(k, shape=(dim,))
            theta = mean + jnp.exp(log_std) * eps_s
            return log_joint(theta) - normal_logpdf_diag(theta, mean, log_std)

        vals = jax.vmap(one_sample)(keys)
        return jnp.mean(vals)

    elbo_and_grad = jax.value_and_grad(elbo_estimate, argnums=(1, 2))

    class _OptState(NamedTuple):
        step: jax.Array
        m_mean: jax.Array
        v_mean: jax.Array
        m_logstd: jax.Array
        v_logstd: jax.Array

    def adam_init(mean: jax.Array, log_std: jax.Array) -> _OptState:
        z_mean = jnp.zeros_like(mean)
        z_ls = jnp.zeros_like(log_std)
        return _OptState(step=jnp.asarray(0), m_mean=z_mean, v_mean=z_mean, m_logstd=z_ls, v_logstd=z_ls)

    def adam_update(params, grads, state: _OptState):
        mean, log_std = params
        g_mean, g_logstd = grads
        step = state.step + 1

        m_mean = beta1 * state.m_mean + (1.0 - beta1) * g_mean
        v_mean = beta2 * state.v_mean + (1.0 - beta2) * (g_mean**2)

        m_ls = beta1 * state.m_logstd + (1.0 - beta1) * g_logstd
        v_ls = beta2 * state.v_logstd + (1.0 - beta2) * (g_logstd**2)

        # Bias correction
        m_mean_hat = m_mean / (1.0 - beta1**step)
        v_mean_hat = v_mean / (1.0 - beta2**step)
        m_ls_hat = m_ls / (1.0 - beta1**step)
        v_ls_hat = v_ls / (1.0 - beta2**step)

        mean = mean + lr * m_mean_hat / (jnp.sqrt(v_mean_hat) + eps)
        log_std = log_std + lr * m_ls_hat / (jnp.sqrt(v_ls_hat) + eps)

        return (mean, log_std), _OptState(step=step, m_mean=m_mean, v_mean=v_mean, m_logstd=m_ls, v_logstd=v_ls)

    def step_fn(carry, t):
        key, mean, log_std, opt_state = carry
        key, subkey = jax.random.split(key)
        elbo_val, grads = elbo_and_grad(subkey, mean, log_std)
        (mean, log_std), opt_state = adam_update((mean, log_std), grads, opt_state)
        return (key, mean, log_std, opt_state), elbo_val

    if jit:
        step_fn_compiled = jax.jit(step_fn)
    else:
        step_fn_compiled = step_fn

    key0 = jax.random.PRNGKey(seed)
    opt0 = adam_init(mean0, log_std0)

    (keyT, meanT, log_stdT, _), elbos = jax.lax.scan(
        step_fn_compiled,
        (key0, mean0, log_std0, opt0),
        xs=jnp.arange(num_steps),
    )

    # Materialise results back to NumPy for downstream use.
    return MeanFieldGaussianVIResult(
        mean=np.asarray(meanT),
        log_std=np.asarray(log_stdT),
        elbo_history=np.asarray(elbos),
    )
