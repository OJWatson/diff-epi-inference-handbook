"""MCMC utilities.

This subpackage contains small, dependency-light reference implementations of classical
Markov chain Monte Carlo algorithms used as baselines in the handbook.
"""

from .hmc import HMCResult, hamiltonian_monte_carlo
from .mh import MHResult, random_walk_metropolis_hastings

__all__ = [
    "HMCResult",
    "MHResult",
    "hamiltonian_monte_carlo",
    "random_walk_metropolis_hastings",
]
