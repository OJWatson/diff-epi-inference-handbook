import numpy as np
import pytest

from diff_epi_inference.abc import systematic_resample


def test_systematic_resample_reproducible() -> None:
    w = np.array([0.1, 0.3, 0.6])

    idx1 = systematic_resample(w, n_samples=20, rng=np.random.default_rng(0))
    idx2 = systematic_resample(w, n_samples=20, rng=np.random.default_rng(0))

    assert idx1.shape == (20,)
    assert np.array_equal(idx1, idx2)
    assert np.all((0 <= idx1) & (idx1 < 3))


def test_systematic_resample_matches_weights_approximately() -> None:
    # This is a stochastic test, but systematic resampling has low variance.
    w = np.array([0.2, 0.8])

    idx = systematic_resample(w, n_samples=10_000, rng=np.random.default_rng(1))
    p_hat = float(np.mean(idx == 1))

    assert 0.77 <= p_hat <= 0.83


@pytest.mark.parametrize(
    ("weights", "n_samples"),
    [
        (np.array([]), None),
        (np.array([[1.0, 2.0]]), None),
        (np.array([np.nan, 1.0]), None),
        (np.array([1.0, -1.0]), None),
        (np.array([0.0, 0.0]), None),
        (np.array([1.0, 1.0]), 0),
    ],
)
def test_systematic_resample_input_validation(weights: np.ndarray, n_samples: int | None) -> None:
    with pytest.raises(ValueError):
        systematic_resample(weights, n_samples=n_samples)
