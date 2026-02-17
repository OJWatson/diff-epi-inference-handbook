from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

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


def plot_seir_compartments(
    *,
    t: np.ndarray,
    S: np.ndarray,
    E: np.ndarray,
    I_series: np.ndarray,
    R: np.ndarray,
    ax=None,
    title: str = "SEIR compartments",
):
    """Plot SEIR compartment trajectories on a shared axis."""

    t = np.asarray(t, dtype=float)
    S = np.asarray(S, dtype=float)
    E = np.asarray(E, dtype=float)
    I_series = np.asarray(I_series, dtype=float)
    R = np.asarray(R, dtype=float)

    if not (t.shape == S.shape == E.shape == I_series.shape == R.shape):
        raise ValueError("t, S, E, I_series, R must all have the same shape")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    ax.plot(t, S, label="S")
    ax.plot(t, E, label="E")
    ax.plot(t, I_series, label="I")
    ax.plot(t, R, label="R")
    ax.set_xlabel("time")
    ax.set_ylabel("population")
    ax.set_title(title)
    ax.legend()
    return fig, ax


def plot_observation_overlay(
    *,
    t: np.ndarray,
    observed: np.ndarray,
    expected: np.ndarray | None = None,
    observed_label: str = "observed",
    expected_label: str = "expected",
    ax=None,
    title: str = "Observation model",
):
    """Plot observed series with an optional expected trajectory overlay."""

    t = np.asarray(t, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if t.shape != observed.shape:
        raise ValueError("t and observed must have the same shape")

    if expected is not None:
        expected = np.asarray(expected, dtype=float)
        if expected.shape != observed.shape:
            raise ValueError("expected must have same shape as observed")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3))
    else:
        fig = ax.figure

    if expected is not None:
        ax.plot(t, expected, label=expected_label, lw=1.4)
    ax.plot(t, observed, label=observed_label, alpha=0.8, lw=1.2, color="black")
    ax.set_xlabel("time")
    ax.set_ylabel("cases / step")
    ax.set_title(title)
    ax.legend()
    return fig, ax


def plot_two_timeseries(
    *,
    t: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    label1: str,
    label2: str,
    ax=None,
    title: str = "Two time series",
    ylabel: str = "value",
):
    """Plot two aligned time series for direct comparison."""

    t = np.asarray(t, dtype=float)
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    if not (t.shape == y1.shape == y2.shape):
        raise ValueError("t, y1, y2 must have the same shape")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3))
    else:
        fig = ax.figure

    ax.plot(t, y1, label=label1, lw=1.2)
    ax.plot(t, y2, label=label2, lw=1.2, alpha=0.8)
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return fig, ax


def plot_ppc_overlay(
    *,
    t: np.ndarray,
    observed: np.ndarray,
    draws: np.ndarray,
    ax=None,
    title: str = "Posterior predictive overlay",
    max_draws: int | None = None,
):
    """Overlay posterior predictive draws against observed data."""

    t = np.asarray(t, dtype=float)
    observed = np.asarray(observed, dtype=float)
    draws = np.asarray(draws, dtype=float)

    if observed.shape != t.shape:
        raise ValueError("observed and t must have the same shape")
    if draws.ndim != 2 or draws.shape[1] != t.shape[0]:
        raise ValueError("draws must be shape (n_draws, len(t))")

    if max_draws is not None:
        if max_draws < 1:
            raise ValueError("max_draws must be >= 1")
        draws = draws[:max_draws]

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3))
    else:
        fig = ax.figure

    for y in draws:
        ax.plot(t, y, color="C0", alpha=0.25, lw=1.0)
    ax.plot(t, observed, color="black", lw=2.0, label="observed")
    ax.set_xlabel("time")
    ax.set_ylabel("reported cases")
    ax.set_title(title)
    ax.legend()
    return fig, ax
