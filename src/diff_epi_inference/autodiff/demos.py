"""Runnable autodiff demos (JAX).

These helpers are used by the Quarto book chapters.

Design goals:
- Run in a few seconds.
- Deterministic outputs.
- Keep imports local so the core package stays NumPy-only.

All functions raise ImportError if JAX is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _require_jax():
    try:
        import jax  # noqa: F401
        import jax.numpy as jnp  # noqa: F401

        return jax, jnp
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ImportError(
            "JAX is required for autodiff demos. Install with `uv pip install -e '.[jax]'`."
        ) from e


@dataclass(frozen=True)
class ScalarGradDemoResult:
    x_grid: np.ndarray
    f_vals: np.ndarray
    grad_vals: np.ndarray


def scalar_function_and_grad_demo(n: int = 200) -> ScalarGradDemoResult:
    """Demo 1: tiny scalar function and its gradient.

    Returns arrays suitable for plotting.
    """
    jax, jnp = _require_jax()

    n = int(n)
    if n < 3:
        raise ValueError("n must be >= 3")

    def f(x):
        # Smooth scalar-to-scalar function with non-trivial gradient.
        return jnp.sin(x) * x**2 + 0.1 * jnp.cos(3.0 * x)

    df = jax.grad(f)

    x = np.linspace(-2.0, 2.0, n, dtype=float)
    fx = np.asarray(jax.vmap(f)(jnp.asarray(x)))
    dfx = np.asarray(jax.vmap(df)(jnp.asarray(x)))

    return ScalarGradDemoResult(x_grid=x, f_vals=fx, grad_vals=dfx)


@dataclass(frozen=True)
class ForwardModeDemoResult:
    x0: float
    jvp: np.ndarray
    y: np.ndarray


def forward_mode_jvp_demo(m: int = 50, x0: float = 0.3) -> ForwardModeDemoResult:
    """Demo 2: vector-valued function with 1D input where forward-mode is natural.

    We compute JVP: d/dx f(x) at x0, producing an m-vector.
    """
    jax, jnp = _require_jax()

    m = int(m)
    if m < 1:
        raise ValueError("m must be >= 1")

    a = jnp.linspace(0.5, 3.0, m)

    def f(x_scalar: jnp.ndarray) -> jnp.ndarray:
        # R -> R^m
        return jnp.sin(a * x_scalar) + 0.1 * (a * x_scalar) ** 2

    x = jnp.asarray(float(x0))
    # Tangent direction v=1 gives df/dx.
    y, jvp = jax.jvp(f, (x,), (jnp.asarray(1.0),))

    return ForwardModeDemoResult(x0=float(x0), jvp=np.asarray(jvp), y=np.asarray(y))


@dataclass(frozen=True)
class ReverseModeDemoResult:
    dim: int
    theta0: np.ndarray
    loss0: float
    grad0: np.ndarray
    grad_norm: float


def reverse_mode_vjp_demo(dim: int = 200, seed: int = 0) -> ReverseModeDemoResult:
    """Demo 3: scalar loss with many parameters where reverse-mode is a good fit.

    We compute grad(loss)(theta) for a moderately-sized vector theta.
    """
    jax, jnp = _require_jax()

    dim = int(dim)
    if dim < 1:
        raise ValueError("dim must be >= 1")

    key = jax.random.PRNGKey(int(seed))
    theta0 = jax.random.normal(key, shape=(dim,)) * 0.2

    def loss(theta: jnp.ndarray) -> jnp.ndarray:
        # Smooth scalar objective.
        return jnp.mean((jnp.sin(theta) + 0.1 * theta) ** 2)

    g = jax.grad(loss)
    loss0 = float(loss(theta0))
    grad0 = np.asarray(g(theta0))
    grad_norm = float(np.linalg.norm(grad0))

    return ReverseModeDemoResult(
        dim=dim,
        theta0=np.asarray(theta0),
        loss0=loss0,
        grad0=grad0,
        grad_norm=grad_norm,
    )
