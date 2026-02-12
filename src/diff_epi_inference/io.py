from __future__ import annotations

from pathlib import Path

import numpy as np

from .dataset import TimeSeriesDataset


def save_timeseries_npz(ds: TimeSeriesDataset, path: str | Path) -> None:
    """Save a TimeSeriesDataset to a .npz file."""

    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("path must have .npz suffix")

    np.savez_compressed(path, t=ds.t, y=ds.y, name=np.array(ds.name, dtype=object))


def load_timeseries_npz(path: str | Path) -> TimeSeriesDataset:
    """Load a TimeSeriesDataset from a .npz file saved by save_timeseries_npz."""

    path = Path(path)
    with np.load(path, allow_pickle=True) as z:
        t = z["t"]
        y = z["y"]
        name = str(z["name"].item()) if "name" in z.files else ""

    return TimeSeriesDataset(t=t, y=y, name=name)
