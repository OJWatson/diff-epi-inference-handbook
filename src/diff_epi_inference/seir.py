from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SEIRParams:
    """Parameters for a simple deterministic SEIR model.

    Notes
    -----
    This is intentionally lightweight for M0: no external ODE solvers.
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
