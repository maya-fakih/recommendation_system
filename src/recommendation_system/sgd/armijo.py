"""Mini-batch Armijo search; see Nocedal and Wright, Numerical Optimization,
2nd ed., Chapter 3, for the classical sufficient-decrease condition.
"""
import numpy as np
from ..model import Parameters


def trial_parameters(params, gradient, alpha):
    """Return a new iterate without modifying the current arrays."""
    return Parameters(**{key: value - alpha * getattr(gradient, key)
                         for key, value in vars(params).items()})


def backtrack(objective, params, gradient, samples, current_loss,
              alpha_initial, beta, max_backtracks, slope=0.0):
    """Return (accepted step or None, number of trial loss evaluations).

    A positive slope imposes sufficient decrease; zero requires strict decrease.
    The evaluation samples stay fixed throughout the search.
    """
    alpha = alpha_initial
    evaluations = 0
    for _ in range(max_backtracks + 1):
        if alpha <= 0:
            break
        evaluations += 1
        try:
            loss = objective.loss(trial_parameters(params, gradient, alpha), samples)
        except FloatingPointError:
            alpha *= beta
            continue
        if loss < current_loss and loss <= current_loss - alpha * slope:
            return alpha, evaluations
        alpha *= beta
    return None, evaluations


def armijo(objective, params, gradient, batch, batch_loss,
           alpha_initial, beta, c, max_backtracks):
    """Require sufficient decrease on the gradient batch itself."""
    norm_squared = sum(float(np.sum(a ** 2)) for a in vars(gradient).values())
    if not np.isfinite(norm_squared) or norm_squared == 0:
        return None, 0
    return backtrack(objective, params, gradient, batch, batch_loss,
                     alpha_initial, beta, max_backtracks, c * norm_squared)
