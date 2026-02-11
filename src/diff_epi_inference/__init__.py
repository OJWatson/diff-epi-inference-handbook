"""Minimal companion code for the Diff-Epi Inference Handbook."""

from .seir import SEIRParams, simulate_seir_euler

__all__ = ["SEIRParams", "simulate_seir_euler"]
