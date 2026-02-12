import numpy as np
import pytest

from diff_epi_inference.abc import abc_rejection


def test_abc_rejection_shapes_and_reproducibility() -> None:
    rng0 = np.random.default_rng(123)

    theta_true = np.array([0.7])
    y_obs = theta_true + rng0.normal(loc=0.0, scale=0.1, size=(50,))

    def prior_sample(rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(low=0.0, high=1.0, size=(1,))

    def simulate(theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return theta[0] + rng.normal(loc=0.0, scale=0.1, size=(50,))

    def summary(y: np.ndarray) -> np.ndarray:
        return np.array([float(np.mean(y))])

    def distance(s_sim: np.ndarray, s_obs: np.ndarray) -> float:
        return float(np.abs(s_sim[0] - s_obs[0]))

    eps = 0.02

    res1 = abc_rejection(
        prior_sample,
        simulate,
        distance,
        y_obs,
        epsilon=eps,
        n_accept=20,
        max_trials=50_000,
        summary=summary,
        rng=np.random.default_rng(0),
    )

    res2 = abc_rejection(
        prior_sample,
        simulate,
        distance,
        y_obs,
        epsilon=eps,
        n_accept=20,
        max_trials=50_000,
        summary=summary,
        rng=np.random.default_rng(0),
    )

    assert res1.thetas.shape == (20, 1)
    assert res1.distances.shape == (20,)
    assert np.all(res1.distances <= eps)

    # Deterministic given the same RNG seed.
    assert np.array_equal(res1.thetas, res2.thetas)
    assert np.array_equal(res1.distances, res2.distances)


@pytest.mark.parametrize(
    ("epsilon", "n_accept", "max_trials"),
    [
        (-1.0, 10, 100),
        (0.1, 0, 100),
        (0.1, 10, 0),
    ],
)
def test_abc_rejection_input_validation(epsilon: float, n_accept: int, max_trials: int) -> None:
    def prior_sample(rng: np.random.Generator) -> np.ndarray:
        return np.array([0.0])

    def simulate(theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return np.array([0.0])

    def distance(s_sim: np.ndarray, s_obs: np.ndarray) -> float:
        return 0.0

    with pytest.raises(ValueError):
        abc_rejection(
            prior_sample,
            simulate,
            distance,
            y_obs=np.array([0.0]),
            epsilon=epsilon,
            n_accept=n_accept,
            max_trials=max_trials,
        )


def test_abc_rejection_raises_if_not_enough_accepts() -> None:
    def prior_sample(rng: np.random.Generator) -> np.ndarray:
        return rng.normal(size=(1,))

    def simulate(theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return theta

    def distance(s_sim: np.ndarray, s_obs: np.ndarray) -> float:
        return float(np.abs(s_sim[0] - s_obs[0]))

    with pytest.raises(RuntimeError):
        abc_rejection(
            prior_sample,
            simulate,
            distance,
            y_obs=np.array([0.0]),
            epsilon=0.0,
            n_accept=5,
            max_trials=3,
            rng=np.random.default_rng(0),
        )
