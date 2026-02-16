import numpy as np

from diff_epi_inference.gradients import bernoulli_concrete


def test_bernoulli_concrete_shape_and_bounds() -> None:
    rng = np.random.default_rng(0)
    logit = np.array([0.0, 1.0, -2.0])

    y = bernoulli_concrete(logit, temperature=0.7, rng=rng)
    assert y.shape == logit.shape
    assert np.all(y >= 0.0)
    assert np.all(y <= 1.0)


def test_bernoulli_concrete_hard_is_binary() -> None:
    rng = np.random.default_rng(1)
    logit = np.linspace(-2.0, 2.0, 5)

    y = bernoulli_concrete(logit, temperature=0.7, rng=rng, hard=True)
    assert set(np.unique(y)).issubset({0.0, 1.0})
