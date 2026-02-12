"""Runnable micro-examples used by the handbook.

These are kept minimal and dependency-free so they can be imported in tests.
"""

from .paired_seir import make_paired_seir_reported_datasets

__all__ = ["make_paired_seir_reported_datasets"]
