"""Experimental monotone search on the entire regularized training objective.

The direction remains stochastic. This is not classical Armijo because
the full directional derivative is not evaluated.
"""
from .armijo import backtrack


def full_backtracking(objective, params, gradient, triplets,
                      alpha_initial, beta, max_backtracks):
    """Return accepted step and total loss calls, including the initial loss."""
    current = objective.loss(params, triplets)
    alpha, calls = backtrack(objective, params, gradient, triplets, current,
                             alpha_initial, beta, max_backtracks)
    return alpha, calls + 1
