from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diff_epi_inference.seir import (
    SEIRParams,
    simulate_seir_euler,
    simulate_seir_stochastic_tau_leap,
)


def finite_difference_central(f, x: float, eps: float) -> float:
    """Central finite-difference derivative estimate.

    Notes
    -----
    This is intentionally tiny (used in book demos).
    """

    if eps <= 0:
        raise ValueError("eps must be > 0")
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


@dataclass(frozen=True)
class DiscontinuousFiniteDiffResult:
    x0: float
    eps: np.ndarray
    fd_estimates: np.ndarray


def discontinuous_finite_difference_instability_demo(
    *, x0: float = 0.0, eps_list: list[float] | None = None
) -> DiscontinuousFiniteDiffResult:
    """Show how finite differences behave badly on a discontinuity.

    We use a Heaviside step function. The derivative is undefined at x=0,
    and finite-difference estimates explode as eps -> 0.
    """

    if eps_list is None:
        eps_list = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003]

    def step(u: float) -> float:
        return float(u >= 0.0)

    eps = np.asarray(eps_list, dtype=float)
    fd = np.asarray([finite_difference_central(step, x0, e) for e in eps], dtype=float)
    return DiscontinuousFiniteDiffResult(x0=float(x0), eps=eps, fd_estimates=fd)


@dataclass(frozen=True)
class SmoothJaxGradResult:
    x0: float
    f0: float
    grad0: float
    grad0_analytic: float


def smooth_function_jax_grad_demo(*, x0: float = 0.3) -> SmoothJaxGradResult:
    """Show that autodiff gives a stable gradient for a smooth function.

    Requires JAX as an optional dependency.
    """

    try:
        import jax
        import jax.numpy as jnp
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "JAX is required for this demo. Install with `uv pip install -e '.[jax]'`."
        ) from e

    def f(x):
        return jnp.sin(x) + 0.1 * x**2

    grad_f = jax.grad(f)

    x0_j = jnp.asarray(x0)
    f0 = float(f(x0_j))
    g0 = float(grad_f(x0_j))

    # d/dx sin(x) + 0.1 x^2 = cos(x) + 0.2 x
    g0_true = float(np.cos(float(x0)) + 0.2 * float(x0))

    return SmoothJaxGradResult(x0=float(x0), f0=f0, grad0=g0, grad0_analytic=g0_true)


@dataclass(frozen=True)
class TauLeapNonDiffResult:
    betas: np.ndarray
    final_size: np.ndarray
    fd_eps: float
    fd_estimates: np.ndarray


def tau_leap_seir_nondifferentiability_demo(
    *,
    beta0: float = 0.25,
    span: float = 0.06,
    n: int = 61,
    fd_eps: float = 1e-3,
    seed: int = 0,
) -> TauLeapNonDiffResult:
    """Illustrate that a stochastic tau-leap SEIR sample path is not differentiable.

    We run a *single* random seed for each beta value, and plot an objective
    (final epidemic size: R_T) as a function of beta.

    The mapping beta -> R_T is piecewise-constant / jumpy because the simulator
    returns integers with discrete random transitions.

    The finite-difference estimates therefore tend to be zero punctuated by spikes.
    """

    if n < 5:
        raise ValueError("n must be >= 5")

    betas = np.linspace(beta0 - span / 2.0, beta0 + span / 2.0, n, dtype=float)

    params_template = dict(sigma=0.2, gamma=0.1)
    init = dict(s0=990, e0=5, i0=5, r0=0)

    dt = 1.0
    steps = 60

    final_size = np.empty_like(betas)
    for j, beta in enumerate(betas):
        rng = np.random.default_rng(seed)
        params = SEIRParams(beta=float(beta), **params_template)
        out = simulate_seir_stochastic_tau_leap(
            params=params,
            dt=dt,
            steps=steps,
            rng=rng,
            **init,
        )
        final_size[j] = float(out["R"][-1])

    # Finite-difference derivative estimate across the beta grid.
    fd = np.full_like(betas, np.nan, dtype=float)
    for j, beta in enumerate(betas):
        def obj(b: float) -> float:
            rng = np.random.default_rng(seed)
            params = SEIRParams(beta=float(b), **params_template)
            out = simulate_seir_stochastic_tau_leap(
                params=params,
                dt=dt,
                steps=steps,
                rng=rng,
                **init,
            )
            return float(out["R"][-1])

        if (beta - fd_eps) <= 0:
            continue
        fd[j] = finite_difference_central(obj, beta, fd_eps)

    return TauLeapNonDiffResult(
        betas=betas,
        final_size=final_size,
        fd_eps=float(fd_eps),
        fd_estimates=fd,
    )


@dataclass(frozen=True)
class DifferentiableSeirJaxResult:
    beta0: float
    loss0: float
    grad_beta0: float


def differentiable_seir_euler_jax_grad_demo(*, beta0: float = 0.25) -> DifferentiableSeirJaxResult:
    """Autodiff through a deterministic SEIR Euler simulator (optional JAX).

    We define a tiny smooth loss: L(beta) = (I_T(beta) - target)^2.

    Requires JAX as an optional dependency.
    """

    try:
        import jax
        import jax.numpy as jnp
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "JAX is required for this demo. Install with `uv pip install -e '.[jax]'`."
        ) from e

    sigma = 0.2
    gamma = 0.1
    s0, e0, i0, r0 = 990.0, 5.0, 5.0, 0.0
    dt = 0.25
    steps = 240
    target = 50.0

    def loss(beta):
        beta = jnp.asarray(beta)

        def step_fn(state, _):
            s, e, i, r = state
            n = s + e + i + r
            inf_flow = beta * s * i / n
            inc_flow = sigma * e
            rec_flow = gamma * i
            s1 = s - dt * inf_flow
            e1 = e + dt * (inf_flow - inc_flow)
            i1 = i + dt * (inc_flow - rec_flow)
            r1 = r + dt * rec_flow
            return (s1, e1, i1, r1), i1

        init = (s0, e0, i0, r0)
        (_, _, _, _), i_hist = jax.lax.scan(step_fn, init, xs=None, length=int(steps))
        i_T = i_hist[-1]
        return (i_T - target) ** 2

    g = jax.grad(loss)

    beta0_j = jnp.asarray(beta0)
    loss0 = float(loss(beta0_j))
    grad0 = float(g(beta0_j))

    return DifferentiableSeirJaxResult(beta0=float(beta0), loss0=loss0, grad_beta0=grad0)


@dataclass(frozen=True)
class DeterministicSeirVsTauLeapResult:
    t: np.ndarray
    i_det: np.ndarray
    i_tau: np.ndarray


def deterministic_vs_tau_leap_demo(*, seed: int = 0) -> DeterministicSeirVsTauLeapResult:
    """Convenience demo: compare deterministic Euler vs stochastic tau-leap trajectories."""

    params = SEIRParams(beta=0.25, sigma=0.2, gamma=0.1)

    det = simulate_seir_euler(
        params=params,
        s0=990.0,
        e0=5.0,
        i0=5.0,
        r0=0.0,
        dt=0.5,
        steps=160,
    )

    tau = simulate_seir_stochastic_tau_leap(
        params=params,
        s0=990,
        e0=5,
        i0=5,
        r0=0,
        dt=0.5,
        steps=160,
        rng=np.random.default_rng(seed),
    )

    return DeterministicSeirVsTauLeapResult(
        t=det["t"],
        i_det=det["I"],
        i_tau=tau["I"].astype(float),
    )
