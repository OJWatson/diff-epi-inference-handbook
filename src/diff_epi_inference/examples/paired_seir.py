from __future__ import annotations

import numpy as np

from ..dataset import TimeSeriesDataset
from ..pipeline import simulate_seir_and_report_deterministic, simulate_seir_and_report_stochastic
from ..seir import SEIRParams


def make_paired_seir_reported_datasets(
    *,
    seed_det: int = 0,
    seed_sto: int = 1,
    steps: int = 60,
    dt: float = 0.2,
    reporting_rate: float = 0.3,
) -> tuple[TimeSeriesDataset, TimeSeriesDataset]:
    """Tiny paired deterministic + stochastic SEIR example.

    Both simulators feed the *same* observation model (reporting + delay + NB noise)
    via the shared pipeline helpers.

    Returns
    -------
    (ds_det, ds_sto)
        Two TimeSeriesDataset objects on the same time grid.
    """

    params = SEIRParams(beta=0.6, sigma=1 / 5, gamma=1 / 7)

    ds_det = simulate_seir_and_report_deterministic(
        params=params,
        s0=999.0,
        e0=0.0,
        i0=1.0,
        r0=0.0,
        dt=dt,
        steps=steps,
        reporting_rate=reporting_rate,
        rng=np.random.default_rng(seed_det),
        name="reported (deterministic)",
    )

    ds_sto = simulate_seir_and_report_stochastic(
        params=params,
        s0=999,
        e0=0,
        i0=1,
        r0=0,
        dt=dt,
        steps=steps,
        reporting_rate=reporting_rate,
        rng=np.random.default_rng(seed_sto),
        name="reported (stochastic)",
    )

    return ds_det, ds_sto
