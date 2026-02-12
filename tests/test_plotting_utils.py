import numpy as np

from diff_epi_inference import TimeSeriesDataset, plot_timeseries


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
