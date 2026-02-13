import numpy as np
import pytest

from diff_epi_inference.synthetic_likelihood import estimate_summary_gaussian, mvn_logpdf


def test_mvn_logpdf_matches_univariate_formula() -> None:
    x = np.array([1.5])
    mean = np.array([0.5])
    cov = np.array([[2.0]])

    got = mvn_logpdf(x, mean, cov)

    # Univariate normal logpdf
    var = 2.0
    expected = float(
        -0.5
        * (
            np.log(2.0 * np.pi)
            + np.log(var)
            + ((float(x[0]) - float(mean[0])) ** 2) / var
        )
    )

    assert np.isfinite(got)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


def test_mvn_logpdf_returns_minus_inf_for_non_pd_cov() -> None:
    x = np.array([0.0, 0.0])
    mean = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.0], [0.0, -1.0]])

    assert mvn_logpdf(x, mean, cov) == -np.inf


def test_mvn_logpdf_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="x and mean must have the same shape"):
        mvn_logpdf(np.array([0.0, 1.0]), np.array([0.0]), np.eye(2))


def test_estimate_summary_gaussian_matches_numpy_mean_cov_plus_jitter() -> None:
    samples = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 4.0],
            [0.0, 2.0, 5.0],
            [3.0, 0.0, 2.0],
        ],
        dtype=float,
    )

    it = iter(samples)

    def simulate_summary(_rng: np.random.Generator) -> np.ndarray:
        return next(it)

    jitter = 0.123
    mu, cov = estimate_summary_gaussian(
        simulate_summary,
        n_sims=samples.shape[0],
        rng=np.random.default_rng(0),
        cov_jitter=jitter,
    )

    expected_mu = np.mean(samples, axis=0)
    expected_cov = np.cov(samples, rowvar=False)
    expected_cov = np.atleast_2d(expected_cov) + jitter * np.eye(expected_cov.shape[0])

    np.testing.assert_allclose(mu, expected_mu, rtol=0, atol=1e-12)
    np.testing.assert_allclose(cov, expected_cov, rtol=0, atol=1e-12)


def test_estimate_summary_gaussian_validates_args() -> None:
    def simulate_summary(rng: np.random.Generator) -> np.ndarray:
        return rng.normal(size=2)

    with pytest.raises(ValueError, match="n_sims must be positive"):
        estimate_summary_gaussian(simulate_summary, n_sims=0, rng=np.random.default_rng(0))

    with pytest.raises(ValueError, match="cov_jitter must be non-negative"):
        estimate_summary_gaussian(
            simulate_summary,
            n_sims=1,
            rng=np.random.default_rng(0),
            cov_jitter=-1e-6,
        )


def test_estimate_summary_gaussian_rejects_non_vector_summaries() -> None:
    def simulate_bad(_rng: np.random.Generator) -> np.ndarray:
        return np.zeros((2, 2))

    with pytest.raises(ValueError, match="must return a 1D vector"):
        estimate_summary_gaussian(simulate_bad, n_sims=2, rng=np.random.default_rng(0))
