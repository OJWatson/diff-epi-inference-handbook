from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curve(
    values: np.ndarray,
    *,
    ax=None,
    title: str = "Optimisation trajectory",
    ylabel: str = "objective",
):
    """Plot a 1D optimisation history such as loss or ELBO."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("values must be a 1D array with length >= 2")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 3))
    else:
        fig = ax.figure

    ax.plot(values, lw=1.8, color="C0")
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig, ax


def plot_series_comparison(
    *,
    t: np.ndarray,
    observed: np.ndarray,
    fitted: np.ndarray,
    observed_label: str = "observed",
    fitted_label: str = "fitted",
    title: str = "Observed vs fitted",
    ylabel: str = "value",
    ax=None,
):
    """Plot observed and fitted series on a common axis."""

    t = np.asarray(t, dtype=float)
    observed = np.asarray(observed, dtype=float)
    fitted = np.asarray(fitted, dtype=float)
    if not (t.shape == observed.shape == fitted.shape):
        raise ValueError("t, observed, and fitted must have the same shape")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3))
    else:
        fig = ax.figure

    ax.plot(t, observed, color="black", lw=2.0, label=observed_label)
    ax.plot(t, fitted, color="C1", lw=1.6, label=fitted_label)
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    return fig, ax


def plot_rank_histogram(
    ranks: np.ndarray,
    *,
    n_ranks: int,
    ax=None,
    title: str = "SBC rank histogram",
):
    """Plot an SBC-style rank histogram."""

    ranks = np.asarray(ranks)
    if ranks.ndim != 1 or ranks.size < 1:
        raise ValueError("ranks must be a non-empty 1D array")
    if n_ranks < 2:
        raise ValueError("n_ranks must be >= 2")
    if np.any(ranks < 0) or np.any(ranks > (n_ranks - 1)):
        raise ValueError("ranks must be in [0, n_ranks - 1]")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 3))
    else:
        fig = ax.figure

    bins = np.arange(n_ranks + 1) - 0.5
    ax.hist(ranks, bins=bins, color="C0", alpha=0.8)
    ax.set_xlabel("rank")
    ax.set_ylabel("count")
    ax.set_title(title)
    return fig, ax


def plot_distribution_comparison(
    *,
    a: np.ndarray,
    b: np.ndarray,
    label_a: str,
    label_b: str,
    bins: int = 40,
    ax=None,
    title: str = "Distribution comparison",
    xlabel: str = "value",
):
    """Overlay two 1D sample distributions as density histograms."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.size < 1 or b.size < 1:
        raise ValueError("a and b must be non-empty 1D arrays")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 3))
    else:
        fig = ax.figure

    ax.hist(a, bins=bins, density=True, alpha=0.45, label=label_a, color="C0")
    ax.hist(b, bins=bins, density=True, alpha=0.45, label=label_b, color="C1")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(loc="best")
    return fig, ax


def plot_summary_intervals(
    *,
    observed: np.ndarray,
    draws: np.ndarray,
    labels: list[str],
    q_low: float = 0.1,
    q_high: float = 0.9,
    ax=None,
    title: str = "Observed summaries vs posterior predictive intervals",
):
    """Compare observed summaries against posterior predictive intervals."""

    observed = np.asarray(observed, dtype=float)
    draws = np.asarray(draws, dtype=float)

    if observed.ndim != 1:
        raise ValueError("observed must be 1D")
    if draws.ndim != 2 or draws.shape[1] != observed.shape[0]:
        raise ValueError("draws must have shape (n_draws, len(observed))")
    if len(labels) != observed.shape[0]:
        raise ValueError("labels must have one entry per summary")
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError("quantiles must satisfy 0 <= q_low < q_high <= 1")

    ql = np.quantile(draws, q_low, axis=0)
    qm = np.quantile(draws, 0.5, axis=0)
    qh = np.quantile(draws, q_high, axis=0)
    x = np.arange(observed.shape[0], dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3.5))
    else:
        fig = ax.figure

    ax.vlines(x, ql, qh, color="C0", lw=4, alpha=0.6, label=f"q{int(100*q_low)}-q{int(100*q_high)}")
    ax.scatter(x, qm, color="C0", s=30, zorder=3, label="median")
    ax.scatter(x, observed, color="black", marker="x", s=55, zorder=4, label="observed")
    ax.set_xticks(x, labels)
    ax.set_ylabel("summary value")
    ax.set_title(title)
    ax.legend(loc="best")
    return fig, ax


def plot_sensitivity_ranges(
    *,
    labels: list[str],
    low: np.ndarray,
    base: np.ndarray,
    high: np.ndarray,
    ax=None,
    title: str = "Sensitivity ranges",
):
    """Plot low/base/high scenario summaries as horizontal intervals."""

    low = np.asarray(low, dtype=float)
    base = np.asarray(base, dtype=float)
    high = np.asarray(high, dtype=float)

    if not (low.ndim == base.ndim == high.ndim == 1):
        raise ValueError("low, base, and high must be 1D arrays")
    if not (low.shape == base.shape == high.shape):
        raise ValueError("low, base, and high must have the same shape")
    if len(labels) != low.shape[0]:
        raise ValueError("labels must have one entry per element")

    y = np.arange(low.shape[0], dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3.5))
    else:
        fig = ax.figure

    ax.hlines(y, low, high, color="C1", lw=4, alpha=0.7, label="range")
    ax.scatter(base, y, color="black", marker="o", s=35, label="base")
    ax.set_yticks(y, labels)
    ax.set_xlabel("value")
    ax.set_title(title)
    ax.legend(loc="best")
    return fig, ax
