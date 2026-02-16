from __future__ import annotations

import numpy as np


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x)
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def gumbel_softmax(
    logits: np.ndarray,
    *,
    temperature: float,
    rng: np.random.Generator,
    hard: bool = False,
) -> np.ndarray:
    """Sample from the Gumbel-Softmax (Concrete) distribution.

    Parameters
    ----------
    logits:
        Array of unnormalised log-probabilities (..., K).
    temperature:
        Positive temperature. Lower -> closer to one-hot.
    rng:
        NumPy RNG.
    hard:
        If True, return a one-hot sample using argmax (non-differentiable).
        If False, return the soft sample on the simplex.

    Returns
    -------
    np.ndarray
        Sample with same shape as logits.

    Notes
    -----
    This is a *relaxation* of sampling from a categorical distribution.
    In autodiff frameworks, you would typically use the soft sample for
    backpropagation; a popular heuristic is the straight-through estimator,
    which uses a hard sample in the forward pass and the soft sample in the
    backward pass.
    """

    logits = np.asarray(logits, dtype=float)
    if logits.ndim < 1:
        raise ValueError("logits must have at least 1 dimension")
    if not np.isfinite(logits).all():
        raise ValueError("logits must be finite")

    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    # Gumbel(0, 1) noise: -log(-log(U))
    u = rng.uniform(low=1e-6, high=1.0 - 1e-6, size=logits.shape)
    g = -np.log(-np.log(u))

    y_soft = _softmax((logits + g) / temperature, axis=-1)

    if not hard:
        return y_soft

    # Hard one-hot (argmax)
    k = y_soft.shape[-1]
    idx = np.argmax(y_soft, axis=-1)
    y_hard = np.eye(k, dtype=float)[idx]
    return y_hard


def bernoulli_concrete(
    logit: float | np.ndarray,
    *,
    temperature: float,
    rng: np.random.Generator,
    hard: bool = False,
) -> np.ndarray:
    """Sample a Binary Concrete (relaxed Bernoulli) random variable.

    This is the K=2 special case of :func:`gumbel_softmax`, returned as a scalar
    in [0, 1] corresponding to the "1" class probability.

    Parameters
    ----------
    logit:
        Bernoulli logit(s). Any shape is allowed.
    temperature:
        Positive temperature. Lower -> closer to {0, 1}.
    rng:
        NumPy RNG.
    hard:
        If True, threshold the soft sample at 0.5.

    Returns
    -------
    np.ndarray
        Array with same shape as logit, values in [0, 1].

    Notes
    -----
    In an autodiff framework, a common heuristic is the straight-through (ST)
    estimator: use the hard sample in the forward pass but backprop through the
    soft sample.
    """

    logit = np.asarray(logit, dtype=float)
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if not np.isfinite(logit).all():
        raise ValueError("logit must be finite")

    # Two Gumbel noises: equivalent to Logistic noise on the logit.
    u1 = rng.uniform(low=1e-6, high=1.0 - 1e-6, size=logit.shape)
    u0 = rng.uniform(low=1e-6, high=1.0 - 1e-6, size=logit.shape)
    g1 = -np.log(-np.log(u1))
    g0 = -np.log(-np.log(u0))

    y_soft = 1.0 / (1.0 + np.exp(-(logit + g1 - g0) / temperature))

    if not hard:
        return y_soft

    return (y_soft >= 0.5).astype(float)
