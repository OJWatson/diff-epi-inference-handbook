import numpy as np

from diff_epi_inference import SEIRParams, simulate_seir_euler


def test_running_example_outputs_shape_and_time_monotonic():
    """A tiny smoke test mirroring the book/running-example.qmd code path."""

    params = SEIRParams(beta=0.6, sigma=1 / 5, gamma=1 / 7)
    out = simulate_seir_euler(
        params=params,
        s0=999.0,
        e0=0.0,
        i0=1.0,
        r0=0.0,
        dt=0.2,
        steps=200,
    )

    for k in ["t", "S", "E", "I", "R"]:
        assert k in out
        assert isinstance(out[k], np.ndarray)
        assert out[k].shape == (201,)
        assert np.all(np.isfinite(out[k]))

    assert np.all(np.diff(out["t"]) > 0)

    # Closed-population invariant
    n = out["S"] + out["E"] + out["I"] + out["R"]
    assert np.allclose(n, n[0])
