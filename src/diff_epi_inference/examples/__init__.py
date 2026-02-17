"""Runnable micro-examples used by the handbook.

These are kept minimal and dependency-free so they can be imported in tests.
"""

from .paired_seir import make_paired_seir_reported_datasets

# Optional JAX-based differentiable epidemiology toy example.
# Imported lazily by tests/docs; we keep it out of __all__ if JAX is missing.
try:  # pragma: no cover
    from .toy_diff_epi_jax import (
        ToyGradFitResult,
        ToySIRPath,
        fit_beta_by_gradient_descent,
        simulate_toy_relaxed_sir,
    )

    _HAVE_TOY_JAX = True
except ModuleNotFoundError:  # pragma: no cover
    _HAVE_TOY_JAX = False

__all__ = ["make_paired_seir_reported_datasets"]
if _HAVE_TOY_JAX:
    __all__ += [
        "ToyGradFitResult",
        "ToySIRPath",
        "fit_beta_by_gradient_descent",
        "simulate_toy_relaxed_sir",
    ]
