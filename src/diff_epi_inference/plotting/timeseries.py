from __future__ import annotations

import matplotlib.pyplot as plt

from ..dataset import TimeSeriesDataset


def plot_timeseries(ds: TimeSeriesDataset, *, ax=None, label: str | None = None):
    """Plot a TimeSeriesDataset.

    Returns (fig, ax).
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3))
    else:
        fig = ax.figure

    if label is None:
        label = ds.name if ds.name else "y"

    ax.plot(ds.t, ds.y, label=label)
    ax.set_xlabel("time")
    ax.set_ylabel("y")
    if label:
        ax.legend()

    return fig, ax
