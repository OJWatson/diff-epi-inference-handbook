import numpy as np

from diff_epi_inference.gradients import gumbel_softmax


def test_gumbel_softmax_outputs_on_simplex():
    rng = np.random.default_rng(0)
    logits = np.array([0.0, 1.0, -0.5])

    y = gumbel_softmax(logits, temperature=0.7, rng=rng, hard=False)

    assert y.shape == logits.shape
    assert np.all(y >= 0.0)
    assert abs(float(np.sum(y)) - 1.0) < 1e-12


def test_gumbel_softmax_hard_is_one_hot():
    rng = np.random.default_rng(1)
    logits = np.array([0.0, 1.0, -0.5, 2.0])

    y = gumbel_softmax(logits, temperature=1.0, rng=rng, hard=True)

    assert y.shape == logits.shape
    assert set(np.unique(y)).issubset({0.0, 1.0})
    assert int(np.sum(y)) == 1
