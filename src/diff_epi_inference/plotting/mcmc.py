from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np


def autocorr_1d(x: np.ndarray, *, max_lag: int) -> np.ndarray:
    """Compute the (biased) autocorrelation function for a 1D series.

    Returns an array of length max_lag+1 with acf[0] == 1.
    """

    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be 1D")
    if max_lag < 0:
        raise ValueError("max_lag must be >= 0")
    if x.size == 0:
        raise ValueError("x must be non-empty")

    x = x - float(np.mean(x))
    denom = float(np.sum(x * x))
    if denom == 0.0:
        return np.ones(max_lag + 1)

    # Full autocorrelation via convolution; take non-negative lags.
    c = np.correlate(x, x, mode="full")
    c = c[c.size // 2 : c.size // 2 + (max_lag + 1)]
    return c / denom


def plot_trace(
    x: np.ndarray,
    *,
    ax=None,
    label: str | None = None,
    true_value: float | None = None,
    title: str | None = None,
):
    """Simple trace plot for a 1D chain."""

    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("x must be 1D")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3))
    else:
        fig = ax.figure

    ax.plot(x, lw=0.7, label=label)
    if true_value is not None:
        ax.axhline(true_value, color="black", ls="--", lw=1.0, label="true")

    ax.set_xlabel("iteration")
    ax.set_ylabel("value")
    if title is not None:
        ax.set_title(title)
    if label is not None or true_value is not None:
        ax.legend()

    return fig, ax


def plot_acf(
    x: np.ndarray,
    *,
    max_lag: int = 60,
    ax=None,
    title: str | None = None,
    kind: Literal["stem", "bar"] = "stem",
):
    """Autocorrelation plot for a 1D chain."""

    acf = autocorr_1d(x, max_lag=max_lag)
    lags = np.arange(acf.size)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 2.5))
    else:
        fig = ax.figure

    if kind == "stem":
        ax.stem(lags, acf, basefmt=" ")
    elif kind == "bar":
        ax.bar(lags, acf, width=1.0)
    else:
        raise ValueError("kind must be 'stem' or 'bar'")

    ax.set_xlabel("lag")
    ax.set_ylabel("ACF")
    ax.set_ylim(-1.0, 1.0)
    if title is not None:
        ax.set_title(title)

    return fig, ax


def plot_trace_and_acf(
    x: np.ndarray,
    *,
    max_lag: int = 60,
    true_value: float | None = None,
    title: str | None = None,
):
    """Convenience helper: trace + ACF side-by-side."""

    fig, axes = plt.subplots(1, 2, figsize=(9, 3), constrained_layout=True)
    plot_trace(x, ax=axes[0], true_value=true_value, title=title)
    plot_acf(x, ax=axes[1], max_lag=max_lag, title="ACF")
    return fig, axes
