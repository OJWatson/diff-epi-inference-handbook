from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class SMCABCResult:
    """Output of an SMC-ABC run."""

    thetas: Array
    distances: Array
    weights: Array
    epsilons: Array


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


def smc_abc(
    prior_sample: Callable[[np.random.Generator], Array],
    prior_logpdf: Callable[[Array], float],
    simulate: Callable[[Array, np.random.Generator], Array],
    distance: Callable[[Array, Array], float],
    y_obs: Array,
    *,
    epsilons: Array,
    n_particles: int,
    max_trials_per_round: int,
    perturb: Callable[[Array, np.random.Generator], Array],
    perturb_logpdf: Callable[[Array, Array], float],
    summary: Callable[[Array], Array] | None = None,
    rng: np.random.Generator | None = None,
) -> SMCABCResult:
    """Sequential Monte Carlo ABC (SMC-ABC) sampler.

    This is a minimal, pedagogical implementation of the standard SMC-ABC scheme:

    - Round 0: sample from the prior until ``n_particles`` draws satisfy
      ``distance(summary(y_sim), summary(y_obs)) <= epsilons[0]``.
    - Round t>0: propose by resampling particles from the previous population
      (using weights), perturb with a kernel, and accept proposals under the
      tighter threshold ``epsilons[t]``.
    - Importance weights are updated as:

      ``w_t(i) ∝ prior(theta_t(i)) / Σ_j w_{t-1}(j) K(theta_t(i) | theta_{t-1}(j))``.

    Args:
        prior_sample: Function ``prior_sample(rng) -> theta``.
        prior_logpdf: Function ``prior_logpdf(theta) -> scalar`` (up to a constant).
        simulate: Function ``simulate(theta, rng) -> y_sim``.
        distance: Function ``distance(s_sim, s_obs) -> scalar``.
        y_obs: Observed data (array-like).
        epsilons: Non-increasing ABC thresholds for each round, shape ``(n_rounds,)``.
        n_particles: Number of particles per round.
        max_trials_per_round: Maximum number of proposals to try per round.
        perturb: Function ``perturb(theta_prev, rng) -> theta_prop``.
        perturb_logpdf: Function ``perturb_logpdf(theta_prop, theta_prev) -> scalar``.
        summary: Optional summary function applied to both observed and simulated data.
        rng: Optional NumPy RNG.

    Returns:
        SMCABCResult with arrays:
          - thetas: shape ``(n_rounds, n_particles, theta_dim)``
          - distances: shape ``(n_rounds, n_particles)``
          - weights: shape ``(n_rounds, n_particles)`` (normalised per-round)
          - epsilons: copy of input thresholds, shape ``(n_rounds,)``

    Notes:
        This baseline is intentionally simple and not optimised.
    """

    rng = np.random.default_rng() if rng is None else rng

    eps = np.asarray(epsilons, dtype=float)
    if eps.ndim != 1 or eps.size == 0:
        raise ValueError("epsilons must be a non-empty 1D array")
    if not np.all(np.isfinite(eps)) or np.any(eps < 0):
        raise ValueError("epsilons must be finite and non-negative")
    if np.any(np.diff(eps) > 0):
        raise ValueError("epsilons must be non-increasing")

    if n_particles <= 0:
        raise ValueError("n_particles must be positive")
    if max_trials_per_round <= 0:
        raise ValueError("max_trials_per_round must be positive")

    y_obs = np.asarray(y_obs)
    summarise = (lambda x: x) if summary is None else summary
    s_obs = np.asarray(summarise(y_obs))

    n_rounds = int(eps.size)
    thetas: list[Array] = []
    distances: list[Array] = []
    weights: list[Array] = []

    prev_thetas: Array | None = None
    prev_w: Array | None = None

    for t in range(n_rounds):
        accepted_thetas: list[Array] = []
        accepted_distances: list[float] = []
        accepted_logw_unnorm: list[float] = []

        n_trials = 0
        while len(accepted_thetas) < n_particles and n_trials < max_trials_per_round:
            n_trials += 1

            if t == 0:
                theta_prop = np.asarray(prior_sample(rng), dtype=float)
                logw = 0.0
            else:
                assert prev_thetas is not None
                assert prev_w is not None
                idx = systematic_resample(prev_w, n_samples=1, rng=rng)[0]
                theta_prev = np.asarray(prev_thetas[idx], dtype=float)
                theta_prop = np.asarray(perturb(theta_prev, rng), dtype=float)

                log_prior = float(prior_logpdf(theta_prop))
                log_k = np.array(
                    [float(perturb_logpdf(theta_prop, th)) for th in prev_thetas],
                    dtype=float,
                )
                if not np.all(np.isfinite(log_k)):
                    raise ValueError("perturb_logpdf must return finite values")

                # denom = Σ_j w_{t-1}(j) * K(theta | theta_{t-1,j})
                m = float(np.max(log_k))
                denom = float(np.sum(prev_w * np.exp(log_k - m)) * np.exp(m))
                if not np.isfinite(denom) or denom <= 0.0:
                    # If the proposal kernel has effectively zero support at theta_prop.
                    continue
                logw = log_prior - float(np.log(denom))

            y_sim = np.asarray(simulate(theta_prop, rng))
            s_sim = np.asarray(summarise(y_sim))
            d = float(distance(s_sim, s_obs))

            if d <= float(eps[t]):
                accepted_thetas.append(theta_prop)
                accepted_distances.append(d)
                accepted_logw_unnorm.append(logw)

        if len(accepted_thetas) < n_particles:
            raise RuntimeError(
                "SMC-ABC did not reach n_particles within max_trials_per_round: "
                f"round={t}, accepted={len(accepted_thetas)}, n_particles={n_particles}, "
                f"max_trials_per_round={max_trials_per_round}"
            )

        theta_arr = np.stack(accepted_thetas, axis=0)
        dist_arr = np.asarray(accepted_distances, dtype=float)

        logw_arr = np.asarray(accepted_logw_unnorm, dtype=float)
        if not np.all(np.isfinite(logw_arr)):
            raise ValueError("weights became non-finite")
        logw_arr = logw_arr - float(np.max(logw_arr))
        w_arr = np.exp(logw_arr)
        w_arr = w_arr / float(np.sum(w_arr))

        thetas.append(theta_arr)
        distances.append(dist_arr)
        weights.append(w_arr)

        prev_thetas = theta_arr
        prev_w = w_arr

    return SMCABCResult(
        thetas=np.stack(thetas, axis=0),
        distances=np.stack(distances, axis=0),
        weights=np.stack(weights, axis=0),
        epsilons=eps.copy(),
    )
