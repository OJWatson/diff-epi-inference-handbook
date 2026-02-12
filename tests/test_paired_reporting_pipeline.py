import numpy as np

from diff_epi_inference import (
    SEIRParams,
    simulate_seir_and_report_deterministic,
    simulate_seir_and_report_stochastic,
)


def test_paired_deterministic_and_stochastic_pipeline_shapes_and_time_match():
    params = SEIRParams(beta=0.6, sigma=1 / 5, gamma=1 / 7)

    rng_det = np.random.default_rng(0)
    rng_sto = np.random.default_rng(1)

    ds_det = simulate_seir_and_report_deterministic(
        params=params,
        s0=999.0,
        e0=0.0,
        i0=1.0,
        r0=0.0,
        dt=0.2,
        steps=50,
        reporting_rate=0.3,
        rng=rng_det,
        name="det",
    )

    ds_sto = simulate_seir_and_report_stochastic(
        params=params,
        s0=999,
        e0=0,
        i0=1,
        r0=0,
        dt=0.2,
        steps=50,
        reporting_rate=0.3,
        rng=rng_sto,
        name="sto",
    )

    assert ds_det.t.shape == ds_det.y.shape == (50,)
    assert ds_sto.t.shape == ds_sto.y.shape == (50,)

    # same time grid
    assert np.allclose(ds_det.t, ds_sto.t)

    assert np.all(ds_det.y >= 0)
    assert np.all(ds_sto.y >= 0)
