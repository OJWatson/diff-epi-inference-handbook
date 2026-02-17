from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SEIRParams:
    """Parameters for a simple deterministic SEIR model.

    Notes
    -----
    This is intentionally lightweight: no external ODE solvers.
    """

    beta: float  # transmission rate
    sigma: float  # incubation rate (E->I)
    gamma: float  # recovery rate (I->R)


def simulate_seir_euler(
    *,
    params: SEIRParams,
    s0: float,
    e0: float,
    i0: float,
    r0: float,
    dt: float,
    steps: int,
) -> dict[str, np.ndarray]:
    """Simulate SEIR using a forward Euler discretisation.

    Returns arrays for t, S, E, I, R.
    """

    if steps < 1:
        raise ValueError("steps must be >= 1")
    if dt <= 0:
        raise ValueError("dt must be > 0")

    t = np.arange(steps + 1, dtype=float) * dt
    s = np.empty(steps + 1, dtype=float)
    e = np.empty(steps + 1, dtype=float)
    i = np.empty(steps + 1, dtype=float)
    r = np.empty(steps + 1, dtype=float)

    s[0], e[0], i[0], r[0] = s0, e0, i0, r0

    for k in range(steps):
        n = s[k] + e[k] + i[k] + r[k]
        if n <= 0:
            raise ValueError("population must stay positive")

        inf_flow = params.beta * s[k] * i[k] / n
        inc_flow = params.sigma * e[k]
        rec_flow = params.gamma * i[k]

        s[k + 1] = s[k] - dt * inf_flow
        e[k + 1] = e[k] + dt * (inf_flow - inc_flow)
        i[k + 1] = i[k] + dt * (inc_flow - rec_flow)
        r[k + 1] = r[k] + dt * rec_flow

    return {"t": t, "S": s, "E": e, "I": i, "R": r}


def simulate_seir_stochastic_tau_leap(
    *,
    params: SEIRParams,
    s0: int,
    e0: int,
    i0: int,
    r0: int,
    dt: float,
    steps: int,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Simulate a simple *stochastic* SEIR model using tau-leaping.

    This provides a paired non-differentiable variant for the running example.

    Transitions over each time step are sampled as:

    - S -> E: Binom(S, 1 - exp(-beta * I/N * dt))
    - E -> I: Binom(E, 1 - exp(-sigma * dt))
    - I -> R: Binom(I, 1 - exp(-gamma * dt))

    Returns integer arrays for S, E, I, R (and float t).
    """

    if steps < 1:
        raise ValueError("steps must be >= 1")
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if min(s0, e0, i0, r0) < 0:
        raise ValueError("initial compartments must be >= 0")

    if rng is None:
        rng = np.random.default_rng()

    t = np.arange(steps + 1, dtype=float) * dt
    s = np.empty(steps + 1, dtype=int)
    e = np.empty(steps + 1, dtype=int)
    i = np.empty(steps + 1, dtype=int)
    r = np.empty(steps + 1, dtype=int)

    s[0], e[0], i[0], r[0] = int(s0), int(e0), int(i0), int(r0)

    for k in range(steps):
        n = s[k] + e[k] + i[k] + r[k]
        if n <= 0:
            raise ValueError("population must stay positive")

        # Clamp probabilities into [0, 1] for numerical safety.
        p_se = 1.0 - np.exp(-params.beta * (i[k] / n) * dt)
        p_ei = 1.0 - np.exp(-params.sigma * dt)
        p_ir = 1.0 - np.exp(-params.gamma * dt)

        p_se = float(np.clip(p_se, 0.0, 1.0))
        p_ei = float(np.clip(p_ei, 0.0, 1.0))
        p_ir = float(np.clip(p_ir, 0.0, 1.0))

        d_se = rng.binomial(n=s[k], p=p_se)
        d_ei = rng.binomial(n=e[k], p=p_ei)
        d_ir = rng.binomial(n=i[k], p=p_ir)

        s[k + 1] = s[k] - d_se
        e[k + 1] = e[k] + d_se - d_ei
        i[k + 1] = i[k] + d_ei - d_ir
        r[k + 1] = r[k] + d_ir

    return {"t": t, "S": s, "E": e, "I": i, "R": r}
