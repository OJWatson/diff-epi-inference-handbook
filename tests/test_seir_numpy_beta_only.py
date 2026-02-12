from __future__ import annotations

import numpy as np
import pytest

from diff_epi_inference.models.seir_numpy_beta_only import make_log_post_logbeta_numpy


def test_make_log_post_logbeta_numpy_returns_finite_scalar() -> None:
    steps = 30

    log_post = make_log_post_logbeta_numpy(
        y_obs=np.zeros(steps, dtype=int),
        w_delay_pmf=np.array([1.0], dtype=float),
        sigma=1 / 5.0,
        gamma=1 / 7.0,
        s0=999.0,
        e0=0.0,
        i0=1.0,
        r0=0.0,
        dt=1.0,
        steps=steps,
        reporting_rate=0.5,
        dispersion=10.0,
        logbeta_prior_mean=float(np.log(0.3)),
        logbeta_prior_sd=0.5,
    )

    lp = log_post(np.array([np.log(0.25)], dtype=float))
    assert isinstance(lp, float)
    assert np.isfinite(lp)


@pytest.mark.parametrize(
    "position",
    [
        np.array(0.0),
        np.array([0.0, 1.0]),
        np.zeros((1, 1)),
    ],
)
def test_make_log_post_logbeta_numpy_validates_position_shape(position: np.ndarray) -> None:
    log_post = make_log_post_logbeta_numpy(
        y_obs=np.zeros(5, dtype=int),
        w_delay_pmf=np.array([1.0], dtype=float),
        sigma=1 / 5.0,
        gamma=1 / 7.0,
        s0=99.0,
        e0=0.0,
        i0=1.0,
        r0=0.0,
        dt=1.0,
        steps=5,
        reporting_rate=0.5,
        dispersion=10.0,
        logbeta_prior_mean=0.0,
        logbeta_prior_sd=1.0,
    )

    with pytest.raises(ValueError, match=r"position must have shape \(1,\)"):
        _ = log_post(position)
