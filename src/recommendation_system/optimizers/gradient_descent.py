# Shared fixed-step gradient descent implementation.
# This can be used accross multiple optimizers, including BCD

import numpy as np
from time import perf_counter
from dataclasses import dataclass

# This rate can be changed later
DEFAULT_TOLERANCE = 1e-6

@dataclass
class GDResult:
    x: np.ndarray
    loss: float
    gradient_norm: float
    iterations: int
    gradient_evaluations: int
    elapsed_seconds: float
    converged: bool
    history: list

def gradient_step(
    x: np.ndarray,
    gradient: np.ndarray,
    learning_rate: float,
) -> np.ndarray:
    """Perform one gradient descent update."""

    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")

    if x.shape != gradient.shape:
        raise ValueError("Parameters and gradient must have the same shape")

    # x_(k+1) = x_k - alpha * gradient
    return x - learning_rate * gradient


def gradient_descent(
    loss_and_grad,
    x0: np.ndarray,
    learning_rate: float = 0.01,
    tolerance: float = DEFAULT_TOLERANCE,
    max_iterations: int = 100,
) -> GDResult:
    """Minimize a differentiable function using fixed-step GD.

    loss_and_grad:
        Callable returning (objective_value, gradient).

    x0:
        Initial parameters.

    learning_rate:
        Fixed step size.

    tolerance:
        Gradient norm stopping tolerance.

    max_iterations:
        Maximum number of parameter updates.
    """

    # Validate hyperparameters
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")

    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Tolerance must be finite and positive")

    if max_iterations < 1:
        raise ValueError("Maximum iterations must be positive")

    start_time = perf_counter()

    # Copy to avoid modifying the original parameters
    x = np.array(x0, dtype=np.float64, copy=True)

    history = []
    gradient_evaluations = 0
    converged = False

    for iteration in range(max_iterations + 1):

        # Evaluate objective and gradient
        gradient_evaluations += 1
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
                "Gradient descent encountered non-finite values"
            )

        # Calculate gradient norm
        gradient_norm = float(np.linalg.norm(gradient))

        # Record convergence metrics
        history.append({
            "iteration": iteration,
            "loss": float(loss),
            "gradient_norm": gradient_norm,
            "gradient_evaluations": gradient_evaluations,
            "elapsed_seconds": perf_counter() - start_time,
        })

        # Check convergence
        if gradient_norm <= tolerance:
            converged = True
            break

        # Check maximum number of updates
        if iteration == max_iterations:
            break

        # Perform gradient descent update
        x = gradient_step(
            x,
            gradient,
            learning_rate,
        )

    elapsed_seconds = perf_counter() - start_time

    return GDResult(
        x=x,
        loss=float(loss),
        gradient_norm=gradient_norm,
        iterations=iteration,
        gradient_evaluations=gradient_evaluations,
        elapsed_seconds=elapsed_seconds,
        converged=converged,
        history=history,
    )
