import numpy as np
import pytest

from diff_epi_inference.plotting import (
    plot_distribution_comparison,
    plot_loss_curve,
    plot_rank_histogram,
    plot_sensitivity_ranges,
    plot_series_comparison,
    plot_summary_intervals,
)


def test_plot_loss_curve_validates_and_returns_fig_ax():
    with pytest.raises(ValueError, match="length >= 2"):
        plot_loss_curve(np.array([1.0]))

    fig, ax = plot_loss_curve(np.array([3.0, 2.0, 1.0]))
    assert fig is ax.figure
    assert len(ax.lines) == 1
    fig.clf()


def test_plot_series_comparison_shape_validation():
    t = np.arange(5.0)
    y = np.arange(5.0)
    with pytest.raises(ValueError, match="same shape"):
        plot_series_comparison(t=t, observed=y, fitted=np.arange(4.0))


def test_plot_rank_histogram_validation_and_output():
    with pytest.raises(ValueError, match=">= 2"):
        plot_rank_histogram(np.array([0, 1]), n_ranks=1)
    with pytest.raises(ValueError, match="must be in"):
        plot_rank_histogram(np.array([0, 3]), n_ranks=3)

    fig, ax = plot_rank_histogram(np.array([0, 1, 2, 1, 0]), n_ranks=3)
    assert fig is ax.figure
    fig.clf()


def test_plot_summary_intervals_and_sensitivity_ranges_validate_shapes():
    observed = np.array([1.0, 2.0, 3.0])
    draws = np.ones((10, 3))
    fig, ax = plot_summary_intervals(observed=observed, draws=draws, labels=["a", "b", "c"])
    assert fig is ax.figure
    fig.clf()

    with pytest.raises(ValueError, match="shape"):
        plot_summary_intervals(observed=observed, draws=np.ones((10, 2)), labels=["a", "b", "c"])

    fig, ax = plot_sensitivity_ranges(
        labels=["x", "y"],
        low=np.array([0.9, 1.1]),
        base=np.array([1.0, 1.3]),
        high=np.array([1.2, 1.5]),
    )
    assert fig is ax.figure
    fig.clf()

    with pytest.raises(ValueError, match="same shape"):
        plot_sensitivity_ranges(
            labels=["x", "y"],
            low=np.array([0.9]),
            base=np.array([1.0, 1.3]),
            high=np.array([1.2, 1.5]),
        )


def test_plot_distribution_comparison_validation_and_output():
    with pytest.raises(ValueError, match="non-empty"):
        plot_distribution_comparison(
            a=np.array([]),
            b=np.array([1.0]),
            label_a="a",
            label_b="b",
        )

    fig, ax = plot_distribution_comparison(
        a=np.array([0.0, 1.0, 2.0]),
        b=np.array([0.5, 1.5, 2.5]),
        label_a="posterior",
        label_b="approximation",
    )
    assert fig is ax.figure
    fig.clf()
