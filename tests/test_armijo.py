
import numpy as np

from recommendation_system.optimizers.armijo import (
    gradient_descent_armijo,
)

def loss_function(x):

    return (x[0] - 3)**2 + (x[1] + 2)**2

def loss_and_grad(x):

    loss = loss_function(x)

    gradient = np.array([
        2 * (x[0] - 3),
        2 * (x[1] + 2),
    ])

    return loss, gradient

x0 = np.array([0.0, 0.0])

result = gradient_descent_armijo(
    loss_function=loss_function,
    loss_and_grad=loss_and_grad,
    x0=x0,
    alpha_0=1.0,
    c=1e-4,
    beta=0.5,
    tolerance=1e-6,
    max_iterations=1000,
)

print("Initial point:", x0)
print("Final point:", result.x)
print("Final objective:", result.loss)
print("Gradient norm:", result.gradient_norm)
print("Iterations:", result.iterations)
print("Gradient evaluations:", result.gradient_evaluations)
print("Function evaluations:", result.function_evaluations)
print("Backtracking steps:", result.backtracking_steps)
print("Accepted step sizes:", result.step_sizes)
print("Duration:", result.elapsed_seconds)
print("Converged:", result.converged)


# ==========================================================
# VERIFY RESULTS
# ==========================================================

exact_minimum = np.array([3.0, -2.0])

error = np.linalg.norm(
    result.x - exact_minimum
)

print("Distance to exact minimum:", error)

# Check that the objective is non-increasing
loss_history = np.array([
    entry["loss"]
    for entry in result.history
])

assert np.all(np.diff(loss_history) <= 1e-12)

assert result.converged
assert error < 1e-6

print("\nARMIJO GRADIENT DESCENT TEST PASSED")
