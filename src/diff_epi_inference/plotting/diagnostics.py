from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def ess(weights: np.ndarray) -> float:
    """Effective sample size (ESS) for 1D importance weights.

    Uses the standard definition ESS = 1 / sum(w^2) for *normalised* weights.
    """

    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError("weights must be 1D")
    if w.size == 0:
        raise ValueError("weights must be non-empty")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")

    s = float(np.sum(w))
    if s <= 0.0:
        raise ValueError("weights must sum to a positive value")

    w = w / s
    return float(1.0 / np.sum(w * w))


def plot_hist_1d(
    x: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    true_value: float | None = None,
    bins: int = 30,
    density: bool = True,
    ax=None,
    title: str | None = None,
    xlabel: str | None = None,
):
    """Plot a 1D histogram with an optional true-value marker.

    Returns (fig, ax).
    """

    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be 1D")

    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != x.shape:
            raise ValueError("weights must have the same shape as x")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
    else:
        fig = ax.figure

    ax.hist(x, bins=bins, weights=weights, density=density, alpha=0.7, color="C0")
    if true_value is not None:
        ax.axvline(true_value, color="black", ls="--", lw=1.2, label="true")
        ax.legend()

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.set_ylabel("density" if density else "count")
    if title is not None:
        ax.set_title(title)

    return fig, ax
