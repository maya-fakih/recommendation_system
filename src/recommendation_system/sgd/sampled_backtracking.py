"""Exploratory search on a separate random evaluation batch.

Sample without replacement independently of the gradient batch; overlap
is possible. Keep the evaluation batch fixed for every trial in one search.
"""
from .full_backtracking import full_backtracking


def sampled_backtracking(objective, params, gradient, triplets, rng,
                         loss_batch_size, alpha_initial, beta, max_backtracks):
    """Return accepted step and loss calls on a fixed independent sample."""
    if not 1 <= loss_batch_size <= len(triplets):
        raise ValueError("Loss batch size must be within the training set size")
    indices = rng.choice(len(triplets), size=loss_batch_size, replace=False)
    return full_backtracking(objective, params, gradient, triplets[indices],
                             alpha_initial, beta, max_backtracks)
