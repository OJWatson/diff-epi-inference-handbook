from __future__ import annotations

import numpy as np

Array = np.ndarray


def systematic_resample(
    weights: Array,
    *,
    n_samples: int | None = None,
    rng: np.random.Generator | None = None,
) -> Array:
    """Systematic resampling for a weighted particle set.

    This is a small utility used by SMC-style algorithms (e.g. SMC-ABC).

    Args:
        weights: Non-negative (unnormalised is ok) weights of shape ``(n_particles,)``.
        n_samples: Number of resampled indices to return. Defaults to ``len(weights)``.
        rng: Optional NumPy random number generator.

    Returns:
        Integer indices into the original particle array, shape ``(n_samples,)``.

    Notes:
        The algorithm draws a single uniform random offset and places equally spaced
        points on ``[0, 1)`` to select indices via the cumulative weight function.
        Compared to multinomial resampling it has lower variance.
    """

    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError("weights must be a 1D array")
    if w.size == 0:
        raise ValueError("weights must be non-empty")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")

    total = float(np.sum(w))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("weights must sum to a positive finite value")

    n = int(w.size if n_samples is None else n_samples)
    if n <= 0:
        raise ValueError("n_samples must be positive")

    rng = np.random.default_rng() if rng is None else rng

    w = w / total
    cdf = np.cumsum(w)
    # Guard against numerical issues so searchsorted always returns in-bounds.
    cdf[-1] = 1.0

    u0 = float(rng.random()) / n
    u = u0 + (np.arange(n, dtype=float) / n)
    idx = np.searchsorted(cdf, u, side="left")
    return idx.astype(int)
