"""Variational inference utilities.

JAX implementations are optional and imported lazily inside functions.
"""

from .meanfield_gaussian_jax import fit_meanfield_gaussian_vi_jax

__all__ = [
    "fit_meanfield_gaussian_vi_jax",
]
