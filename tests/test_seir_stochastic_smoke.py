import numpy as np

from diff_epi_inference import SEIRParams, simulate_seir_stochastic_tau_leap


def test_stochastic_tau_leap_conserves_population_and_nonnegative():
    rng = np.random.default_rng(0)
    out = simulate_seir_stochastic_tau_leap(
        params=SEIRParams(beta=0.6, sigma=1 / 5, gamma=1 / 7),
        s0=999,
        e0=0,
        i0=1,
        r0=0,
        dt=0.2,
        steps=200,
        rng=rng,
    )

    S, E, I_comp, R = out["S"], out["E"], out["I"], out["R"]

    assert S.dtype.kind in {"i", "u"}
    assert S.shape == (201,)
    assert np.all(S >= 0)
    assert np.all(E >= 0)
    assert np.all(I_comp >= 0)
    assert np.all(R >= 0)

    n = S + E + I_comp + R
    assert np.all(n == n[0])
