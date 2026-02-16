import numpy as np
import pytest


def _have_jax() -> bool:
    try:
        import jax  # noqa: F401

        return True
    except ModuleNotFoundError:
        return False


@pytest.mark.skipif(not _have_jax(), reason="optional dependency jax not installed")
def test_scalar_function_and_grad_demo_shapes_and_finite():
    from diff_epi_inference.autodiff.demos import scalar_function_and_grad_demo

    res = scalar_function_and_grad_demo(n=50)
    assert res.x_grid.shape == (50,)
    assert res.f_vals.shape == (50,)
    assert res.grad_vals.shape == (50,)
    assert np.all(np.isfinite(res.f_vals))
    assert np.all(np.isfinite(res.grad_vals))


@pytest.mark.skipif(not _have_jax(), reason="optional dependency jax not installed")
def test_forward_mode_jvp_demo_shapes_and_finite():
    from diff_epi_inference.autodiff.demos import forward_mode_jvp_demo

    res = forward_mode_jvp_demo(m=17, x0=0.25)
    assert res.y.shape == (17,)
    assert res.jvp.shape == (17,)
    assert np.all(np.isfinite(res.y))
    assert np.all(np.isfinite(res.jvp))


@pytest.mark.skipif(not _have_jax(), reason="optional dependency jax not installed")
def test_reverse_mode_vjp_demo_shapes_and_finite():
    from diff_epi_inference.autodiff.demos import reverse_mode_vjp_demo

    res = reverse_mode_vjp_demo(dim=33, seed=0)
    assert res.theta0.shape == (33,)
    assert res.grad0.shape == (33,)
    assert np.isfinite(res.loss0)
    assert np.isfinite(res.grad_norm)
    assert np.all(np.isfinite(res.grad0))
