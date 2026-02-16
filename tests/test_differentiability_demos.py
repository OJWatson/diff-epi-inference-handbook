import numpy as np

from diff_epi_inference.differentiability.demos import (
    discontinuous_finite_difference_instability_demo,
    finite_difference_central,
    tau_leap_seir_nondifferentiability_demo,
)


def test_finite_difference_central_requires_positive_eps():
    def f(x: float) -> float:
        return x**2

    try:
        finite_difference_central(f, 0.0, 0.0)
    except ValueError as e:
        assert "eps" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_discontinuous_demo_explodes_as_eps_shrinks():
    res = discontinuous_finite_difference_instability_demo(
        x0=0.0, eps_list=[1.0, 0.1, 0.01]
    )
    assert res.eps.shape == (3,)
    assert res.fd_estimates.shape == (3,)

    # For a step function at the discontinuity, the central FD estimate is ~1/(2*eps).
    assert res.fd_estimates[2] > res.fd_estimates[1] > res.fd_estimates[0]
    assert np.isfinite(res.fd_estimates).all()


def test_tau_leap_demo_is_jumpy_and_fd_has_spikes():
    res = tau_leap_seir_nondifferentiability_demo(beta0=0.25, span=0.02, n=31, fd_eps=1e-3, seed=0)
    assert res.betas.shape == (31,)
    assert res.final_size.shape == (31,)
    assert res.fd_estimates.shape == (31,)

    # Final size should take (mostly) integer-ish values.
    assert np.allclose(res.final_size, np.round(res.final_size))

    # We expect at least one jump in the objective across betas for a fixed seed.
    assert np.any(np.diff(res.final_size) != 0.0)

    # FD estimates should include zeros (flat regions) and at least one large magnitude spike.
    finite_fd = res.fd_estimates[np.isfinite(res.fd_estimates)]
    assert np.any(np.isclose(finite_fd, 0.0))
    assert np.nanmax(np.abs(finite_fd)) >= 10.0
