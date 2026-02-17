from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimeSeriesDataset:
    """A minimal standard dataset format for the handbook.

    This is intentionally small and dependency-free (NumPy only).

    Fields
    ------
    t:
        1D array of time points of length T.
    y:
        1D array of observations (e.g. reported cases) of length T.
    name:
        Optional human-friendly name.
    """

    t: np.ndarray
    y: np.ndarray
    name: str = ""

    def __post_init__(self) -> None:
        t = np.asarray(self.t, dtype=float)
        y = np.asarray(self.y)

        if t.ndim != 1 or y.ndim != 1:
            raise ValueError("t and y must be 1D arrays")
        if t.shape[0] != y.shape[0]:
            raise ValueError("t and y must have the same length")
        if t.shape[0] < 1:
            raise ValueError("dataset must have at least one observation")
        if not np.all(np.isfinite(t)):
            raise ValueError("t must be finite")
        if not np.all(np.diff(t) > 0):
            raise ValueError("t must be strictly increasing")

        # Basic expectations for count observations
        if np.issubdtype(y.dtype, np.floating):
            if not np.all(np.isfinite(y)):
                raise ValueError("y must be finite")
        if np.any(y < 0):
            raise ValueError("y must be non-negative")

        object.__setattr__(self, "t", t)
        object.__setattr__(self, "y", y)


def from_cases(
    *,
    y: np.ndarray,
    dt: float = 1.0,
    t0: float = 0.0,
    name: str = "",
) -> TimeSeriesDataset:
    """Construct a dataset from regularly-spaced case counts."""

    if dt <= 0:
        raise ValueError("dt must be > 0")

    y = np.asarray(y)
    t = t0 + dt * np.arange(y.shape[0], dtype=float)
    return TimeSeriesDataset(t=t, y=y, name=name)
