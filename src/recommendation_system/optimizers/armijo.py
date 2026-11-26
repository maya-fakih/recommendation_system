# Armijo backtracking line search and gradient descent implementation.

import numpy as np
from time import perf_counter
from dataclasses import dataclass

from .gradient_descent import (
    gradient_step,
    GDResult,
)
from ..constants import DEFAULT_TOLERANCE, DEFAULT_ALPHA, DEFAULT_C, DEFAULT_BETA, DEFAULT_MAX_BACKTRACKS

@dataclass
class ArmijoResult(GDResult):
    function_evaluations: int
    backtracking_steps: int
    step_sizes: list

def armijo_backtracking(
    loss_function,
    x: np.ndarray,
    loss: float,
    gradient: np.ndarray,
    alpha_0: float = DEFAULT_ALPHA,
    c: float = DEFAULT_C,
    beta: float = DEFAULT_BETA,
    max_backtracks: int = DEFAULT_MAX_BACKTRACKS,
):
    """Find a step size satisfying the Armijo condition."""

    if not np.isfinite(alpha_0) or alpha_0 <= 0:
        raise ValueError("Initial step size must be finite and positive")

    if not 0 < c < 1:
        raise ValueError("Armijo parameter c must be between 0 and 1")

    if not 0 < beta < 1:
        raise ValueError("Beta must be between 0 and 1")

    if max_backtracks < 0:
        raise ValueError("Maximum backtracks must be nonnegative")

    alpha = alpha_0

    gradient_norm_squared = float(
        np.sum(gradient ** 2)
    )

    function_evaluations = 0

    for backtracks in range(max_backtracks + 1):

        # Candidate point
        x_trial = gradient_step(
            x,
            gradient,
            alpha,
        )

        # Evaluate the candidate objective
        function_evaluations += 1
        trial_loss = loss_function(x_trial)

        # Armijo sufficient-decrease condition
        armijo_bound = (
            loss - c * alpha * gradient_norm_squared
        )

        if (
            np.isfinite(trial_loss)
            and np.isfinite(x_trial).all()
            and trial_loss <= armijo_bound
        ):
            return (
                alpha,
                x_trial,
                function_evaluations,
                backtracks,
            )

        # Reduce the step size
        alpha *= beta

    raise FloatingPointError(
        "Armijo backtracking failed to find an acceptable step"
    )

def gradient_descent_armijo(
    loss_function,
    loss_and_grad,
    x0: np.ndarray,
    alpha_0: float = DEFAULT_ALPHA,
    c: float = DEFAULT_C,
    beta: float = DEFAULT_BETA,
    tolerance: float = DEFAULT_TOLERANCE,
    max_iterations: int = 100,
    max_backtracks: int = DEFAULT_MAX_BACKTRACKS,
) -> ArmijoResult:
    """Minimize a differentiable function using Armijo GD."""

    # Validate hyperparameters
    if not np.isfinite(alpha_0) or alpha_0 <= 0:
        raise ValueError("Initial step size must be finite and positive")

    if not 0 < c < 1:
        raise ValueError("Armijo parameter c must be between 0 and 1")

    if not 0 < beta < 1:
        raise ValueError("Beta must be between 0 and 1")

    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Tolerance must be finite and positive")

    if max_iterations < 1:
        raise ValueError("Maximum iterations must be positive")

    if max_backtracks < 0:
        raise ValueError("Maximum backtracks must be nonnegative")

    start_time = perf_counter()

    # Copy initial parameters
    x = np.array(x0, dtype=np.float64, copy=True)

    history = []
    step_sizes = []

    gradient_evaluations = 0
    function_evaluations = 0
    total_backtracking_steps = 0

    converged = False

    for iteration in range(max_iterations + 1):

        # Evaluate objective and gradient
        gradient_evaluations += 1
        function_evaluations += 1

        loss, gradient = loss_and_grad(x)

        gradient = np.asarray(gradient)

        if gradient.shape != x.shape:
            raise ValueError("Invalid gradient shape")

        # Detect numerical instability
        if (
            not np.isfinite(loss)
            or not np.isfinite(gradient).all()
            or not np.isfinite(x).all()
        ):
            raise FloatingPointError(
                "Armijo GD encountered non-finite values"
            )

        # Calculate gradient norm
        gradient_norm = float(np.linalg.norm(gradient))

        # Record convergence metrics
        history.append({
            "iteration": iteration,
            "loss": float(loss),
            "gradient_norm": gradient_norm,
            "gradient_evaluations": gradient_evaluations,
            "function_evaluations": function_evaluations,
            "backtracking_steps": total_backtracking_steps,
            "elapsed_seconds": perf_counter() - start_time,
        })

        # Check convergence
        if gradient_norm <= tolerance:
            converged = True
            break

        # Check maximum number of updates
        if iteration == max_iterations:
            break

        # Find an acceptable step size
        alpha, x_trial, evaluations, backtracks = (
            armijo_backtracking(
                loss_function=loss_function,
                x=x,
                loss=loss,
                gradient=gradient,
                alpha_0=alpha_0,
                c=c,
                beta=beta,
                max_backtracks=max_backtracks,
            )
        )

        # Update computational counters
        function_evaluations += evaluations
        total_backtracking_steps += backtracks

        # Accept the new parameters
        x = x_trial

        # Record the accepted step size
        step_sizes.append(alpha)

    elapsed_seconds = perf_counter() - start_time

    return ArmijoResult(
        x=x,
        loss=float(loss),
        gradient_norm=gradient_norm,
        iterations=iteration,
        gradient_evaluations=gradient_evaluations,
        elapsed_seconds=elapsed_seconds,
        converged=converged,
        history=history,
        function_evaluations=function_evaluations,
        backtracking_steps=total_backtracking_steps,
        step_sizes=step_sizes,
    )
