
import numpy as np
from copy import deepcopy

from recommendation_system.data import load_data
from recommendation_system.model import HybridMatrixFactorization
from recommendation_system.objective import PairwiseObjective
from recommendation_system.experiments import initialize

data = load_data("data")

model = HybridMatrixFactorization(
    data.Z,
    data.q_index,
)

objective = PairwiseObjective(
    model,
    lambda_f=0.001,
    lambda_w=0.001,
)

params, _ = initialize(data, seed=616)

triplets = data.train_triplets

rng = np.random.default_rng(616)

h = 1e-5
tolerance = 1e-6

def relative_error(analytical, numerical):

    return (
        np.linalg.norm(analytical - numerical)
        / max(
            np.linalg.norm(analytical),
            np.linalg.norm(numerical),
            1e-12,
        )
    )


def flatten_QWb(Q, W, b):

    return np.concatenate([
        Q.ravel(),
        W.ravel(),
        b.ravel(),
    ])


def report(name, error):

    status = "PASS" if error < tolerance else "FAIL"

    print(f"{name}: {error:.6e} [{status}]")

    return error < tolerance

print("\nTEST 1: Hessian of P")

direction_P = rng.normal(size=params.P.shape)
direction_P /= np.linalg.norm(direction_P)

# Analytical Hessian
H_P = objective.hessian_P(params, triplets)

# Multiply each customer Hessian by its direction
analytical_P = np.einsum(
    "uij,uj->ui",
    H_P,
    direction_P,
)

# Perturb only P
params_plus = deepcopy(params)
params_minus = deepcopy(params)

params_plus.P += h * direction_P
params_minus.P -= h * direction_P

# Calculate gradients at P + hv and P - hv
_, grad_plus = objective.loss_and_grad(
    params_plus, triplets
)

_, grad_minus = objective.loss_and_grad(
    params_minus, triplets
)

# Numerical Hessian-vector product
numerical_P = (
    grad_plus.P - grad_minus.P
) / (2 * h)

error_P = relative_error(
    analytical_P,
    numerical_P,
)

passed_P = report("Relative error", error_P)

print("Hessian shape:", H_P.shape)

print("\nTEST 2: Hessian-vector product of P")

Hv_P = objective.hessian_vector_P(
    params,
    triplets,
    direction_P,
)

error_Hv_P = relative_error(
    Hv_P,
    numerical_P,
)

passed_Hv_P = report(
    "Relative error",
    error_Hv_P,
)

# Compare the two analytical implementations
error_consistency = relative_error(
    analytical_P,
    Hv_P,
)

passed_consistency = report(
    "Explicit Hessian vs direct HVP",
    error_consistency,
)

print("\nTEST 3: Hessian-vector product of Q, W, b")

direction_Q = rng.normal(size=params.Q.shape)
direction_W = rng.normal(size=params.W.shape)
direction_b = rng.normal(size=params.b.shape)

# Normalize the joint direction
norm = np.linalg.norm(
    flatten_QWb(
        direction_Q,
        direction_W,
        direction_b,
    )
)

direction_Q /= norm
direction_W /= norm
direction_b /= norm

# Analytical Hessian-vector product
Hv_Q, Hv_W, Hv_b = objective.hessian_vector_QWb(
    params,
    triplets,
    direction_Q,
    direction_W,
    direction_b,
)

# Perturb only Q, W and b
params_plus = deepcopy(params)
params_minus = deepcopy(params)

params_plus.Q += h * direction_Q
params_plus.W += h * direction_W
params_plus.b += h * direction_b

params_minus.Q -= h * direction_Q
params_minus.W -= h * direction_W
params_minus.b -= h * direction_b

# P remains fixed

_, grad_plus = objective.loss_and_grad(
    params_plus, triplets
)

_, grad_minus = objective.loss_and_grad(
    params_minus, triplets
)

# Numerical Hessian-vector products
numerical_Q = (
    grad_plus.Q - grad_minus.Q
) / (2 * h)

numerical_W = (
    grad_plus.W - grad_minus.W
) / (2 * h)

numerical_b = (
    grad_plus.b - grad_minus.b
) / (2 * h)

# Individual relative errors
passed_Q = report(
    "Relative error Q",
    relative_error(Hv_Q, numerical_Q),
)

passed_W = report(
    "Relative error W",
    relative_error(Hv_W, numerical_W),
)

passed_b = report(
    "Relative error b",
    relative_error(Hv_b, numerical_b),
)

# Joint relative error
analytical_C = flatten_QWb(
    Hv_Q, Hv_W, Hv_b
)

numerical_C = flatten_QWb(
    numerical_Q, numerical_W, numerical_b
)

error_C = relative_error(
    analytical_C,
    numerical_C,
)

passed_C = report(
    "Joint relative error",
    error_C,
)

print("\nTEST 4: Hessian properties of P")

symmetry_error = np.max(
    np.abs(H_P - H_P.transpose(0, 2, 1))
)

eigenvalues = np.linalg.eigvalsh(H_P)

min_eigenvalue = np.min(eigenvalues)
max_eigenvalue = np.max(eigenvalues)

print(f"Symmetry error: {symmetry_error:.6e}")
print(f"Minimum eigenvalue: {min_eigenvalue:.6e}")
print(f"Maximum eigenvalue: {max_eigenvalue:.6e}")

passed_symmetry = symmetry_error < tolerance
passed_PD = min_eigenvalue > 0

print("Symmetric:", passed_symmetry)
print("Positive definite:", passed_PD)

print("\nTEST 5: Positive curvature of Q, W, b")

direction_C = flatten_QWb(
    direction_Q,
    direction_W,
    direction_b,
)

curvature_C = direction_C @ analytical_C

passed_curvature = curvature_C > 0

print(f"v^T H v: {curvature_C:.6e}")
print("Positive curvature:", passed_curvature)

all_passed = all([
    passed_P,
    passed_Hv_P,
    passed_consistency,
    passed_Q,
    passed_W,
    passed_b,
    passed_C,
    passed_symmetry,
    passed_PD,
    passed_curvature,
])

print("\n" + "=" * 50)

if all_passed:
    print("ALL HESSIAN TESTS PASSED")
else:
    print("SOME HESSIAN TESTS FAILED")

print("=" * 50)

assert all_passed, "Hessian verification failed"
