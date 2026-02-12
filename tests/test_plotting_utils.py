import matplotlib.pyplot as plt
import numpy as np
import pytest

from diff_epi_inference import TimeSeriesDataset, plot_timeseries
from diff_epi_inference.plotting.mcmc import autocorr_1d, plot_acf


def test_plot_timeseries_returns_fig_ax():
    ds = TimeSeriesDataset(t=np.array([0.0, 1.0, 2.0]), y=np.array([0, 1, 1]), name="y")
    fig, ax = plot_timeseries(ds)

    assert fig is ax.figure
    assert len(ax.lines) == 1
    x, y = ax.lines[0].get_data()
    assert np.allclose(x, ds.t)
    assert np.allclose(y, ds.y)

    # Avoid GUI resource warnings in test runners
    fig.clf()


def test_autocorr_1d_validates_inputs():
    with pytest.raises(ValueError, match="x must be 1D"):
        autocorr_1d(np.zeros((2, 2)), max_lag=1)

    with pytest.raises(ValueError, match="max_lag must be >= 0"):
        autocorr_1d(np.zeros(3), max_lag=-1)

    with pytest.raises(ValueError, match="x must be non-empty"):
        autocorr_1d(np.array([]), max_lag=0)


def test_autocorr_1d_known_series_sanity():
    # Alternating +/-1 has a known biased ACF when mean is exactly 0 (even length).
    x = np.tile([1.0, -1.0], 50)  # N=100
    n = x.size

    acf = autocorr_1d(x, max_lag=2)
    assert acf.shape == (3,)
    assert np.isclose(acf[0], 1.0)
    assert np.isclose(acf[1], -(n - 1) / n)
    assert np.isclose(acf[2], (n - 2) / n)

    # Constant series => zero variance after centering; defined here as all ones.
    acf_const = autocorr_1d(np.ones(10), max_lag=3)
    assert np.allclose(acf_const, np.ones(4))


def test_plot_acf_validates_kind():
    fig, ax = plt.subplots(figsize=(4, 2))
    try:
        with pytest.raises(ValueError, match="kind must be 'stem' or 'bar'"):
            plot_acf(np.arange(10.0), ax=ax, kind="nope")  # type: ignore[arg-type]
    finally:
        fig.clf()


def test_plot_acf_bar_returns_fig_ax_and_plots_expected_bars():
    max_lag = 7
    fig, ax = plot_acf(np.arange(100.0), max_lag=max_lag, kind="bar")

    assert fig is ax.figure
    # Matplotlib `bar` draws one Rectangle patch per bar.
    assert len(ax.patches) == max_lag + 1

    # Avoid GUI resource warnings in test runners
    fig.clf()
