import numpy as np

from diff_epi_inference import SEIRParams, simulate_seir_euler


def test_simulate_seir_euler_conserves_population():
    params = SEIRParams(beta=0.5, sigma=0.2, gamma=0.1)
    out = simulate_seir_euler(params=params, s0=990, e0=5, i0=5, r0=0, dt=0.1, steps=200)

    n0 = out["S"][0] + out["E"][0] + out["I"][0] + out["R"][0]
    nT = out["S"][-1] + out["E"][-1] + out["I"][-1] + out["R"][-1]

    assert np.isfinite(nT)
    assert abs(nT - n0) < 1e-6
