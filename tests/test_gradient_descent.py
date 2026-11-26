
import numpy as np

from recommendation_system.optimizers.gradient_descent import (
    gradient_descent,
)

def loss_and_grad(x):

    loss = (x[0] - 3)**2 + (x[1] + 2)**2

    gradient = np.array([
        2 * (x[0] - 3),
        2 * (x[1] + 2),
    ])

    return loss, gradient

x0 = np.array([0.0, 0.0])

result = gradient_descent(
    loss_and_grad=loss_and_grad,
    x0=x0,
    learning_rate=0.1,
    tolerance=1e-6,
    max_iterations=1000,
)

print("Initial point:", x0)
print("Final point:", result.x)
print("Final objective:", result.loss)
print("Gradient norm:", result.gradient_norm)
print("Iterations:", result.iterations)
print("Gradient evaluations:", result.gradient_evaluations)
print("Duration:", result.elapsed_seconds)
print("Converged:", result.converged)

error = np.linalg.norm(
    result.x - np.array([3.0, -2.0])
)

print("Distance to exact minimum:", error)

assert result.converged
assert error < 1e-6

print("\nGRADIENT DESCENT TEST PASSED")
