"""Pairwise logistic objective and analytic gradients."""

import numpy as np

from .model import HybridMatrixFactorization, Parameters


class PairwiseObjective:
    """Evaluate the shared regularized objective on ordinary-product triplets."""

    def __init__(self, model: HybridMatrixFactorization, lambda_f: float, lambda_w: float):
        if not (np.isfinite(lambda_f) and np.isfinite(lambda_w)
                and lambda_f > 0 and lambda_w > 0):
            raise ValueError("Regularization strengths must be finite and positive")
        self.model = model
        self.lambda_f = lambda_f
        self.lambda_w = lambda_w

    def _batch(self, params, triplets):
        if triplets.ndim != 2 or triplets.shape[1] != 3 or len(triplets) == 0:
            raise ValueError("Expected a nonempty array of (user, positive, comparison) rows")
        users, positive, comparison = triplets.T
        i, j = self.model.q_index[positive], self.model.q_index[comparison]
        if np.any((i < 0) | (j < 0)):
            raise ValueError("Objective triplets must contain ordinary products")
        # Compute the positive-minus-comparison score directly for each triplet.
        text_difference = self.model.Z[positive] - self.model.Z[comparison]
        product_difference = params.Q[i] - params.Q[j] + text_difference @ params.W.T
        user_vectors = params.P[users]
        margin = np.sum(user_vectors * product_difference, axis=1) + params.b[i] - params.b[j]
        return margin, users, i, j, user_vectors, product_difference, text_difference

    def data_loss(self, params: Parameters, triplets: np.ndarray) -> float:
        """Return the mean unregularized pairwise loss."""
        margin = self._batch(params, triplets)[0]
        return float(np.logaddexp(0, -margin).mean())

    def penalty(self, params: Parameters) -> float:
        """Return the L2 penalty over all trainable arrays."""
        factor_norm = sum(np.sum(array ** 2) for array in (params.P, params.Q, params.b))
        return float((self.lambda_f * factor_norm + self.lambda_w * np.sum(params.W ** 2)) / 2)

    def loss(self, params: Parameters, triplets: np.ndarray) -> float:
        """Evaluate the regularized objective without a gradient."""
        return self.data_loss(params, triplets) + self.penalty(params)

    def loss_and_grad(
        self, params: Parameters, triplets: np.ndarray,
    ) -> tuple[float, Parameters]:
        """Return the regularized batch loss and gradients shaped like the parameters."""
        margin, users, i, j, user_vectors, product_difference, text_difference = (
            self._batch(params, triplets)
        )
        # The derivative of softplus(-margin), divided by the batch size.
        weight = -np.exp(-np.logaddexp(0, margin)) / len(triplets)
        weighted_users = weight[:, None] * user_vectors
        gradients = Parameters(
            self.lambda_f * params.P,
            self.lambda_f * params.Q,
            self.lambda_w * params.W + weighted_users.T @ text_difference,
            self.lambda_f * params.b,
        )
        # Repeated IDs must accumulate every contribution from the batch.
        np.add.at(gradients.P, users, weight[:, None] * product_difference)
        np.add.at(gradients.Q, i, weighted_users)
        np.add.at(gradients.Q, j, -weighted_users)
        np.add.at(gradients.b, i, weight)
        np.add.at(gradients.b, j, -weight)
        return float(np.logaddexp(0, -margin).mean()) + self.penalty(params), gradients

# We need the Hessian only for the Newtown iterations that we are willing to test for BCD
    @staticmethod
    def _curvature(margin: np.ndarray) -> np.ndarray:
        """Calculate the second derivative of the logistic loss."""

        # rho = 1 / (1 + exp(Delta))
        rho = np.exp(-np.logaddexp(0, margin))

        # Numerically stable calculation of 1 - rho
        one_minus_rho = np.exp(-np.logaddexp(0, -margin))

        # omega = rho * (1 - rho)
        return rho * one_minus_rho

    def hessian_P(
        self,
        params: Parameters,
        triplets: np.ndarray,
    ) -> np.ndarray:
        """Calculate the independent Hessian for each customer.

        Q, W and b are fixed.
        Output shape: (n_users, k, k).
        """

        (
            margin,
            users,
            _,
            _,
            _,
            product_difference,
            _,
        ) = self._batch(params, triplets)

        n_triplets = len(triplets)
        n_users, k = params.P.shape

        # Logistic curvature divided by N
        omega = self._curvature(margin) / n_triplets

        # Initialize Hessians with regularization: lambda_f * I
        hessians = np.zeros((n_users, k, k), dtype=np.float64)
        hessians[:] = self.lambda_f * np.eye(k)

        # Calculate the Hessian for each customer
        for u in np.unique(users):

            mask = users == u

            A = product_difference[mask]
            weights = omega[mask]

            # H_u = sum(omega * a_ij * a_ij^T) / N
            #       + lambda_f * I
            hessians[u] += A.T @ (weights[:, None] * A)

        return hessians

    def hessian_vector_P(
        self,
        params: Parameters,
        triplets: np.ndarray,
        direction: np.ndarray,
    ) -> np.ndarray:
        """Calculate H_P @ direction without constructing H_P."""

        if direction.shape != params.P.shape:
            raise ValueError("Direction must have the same shape as P")

        (
            margin,
            users,
            _,
            _,
            _,
            product_difference,
            _,
        ) = self._batch(params, triplets)

        omega = self._curvature(margin) / len(triplets)

        # eta = a_ij^T * v_u
        eta = np.sum(
            product_difference * direction[users],
            axis=1,
        )

        # Regularization contribution
        result = self.lambda_f * direction

        # Data-loss contribution
        contributions = (
            (omega * eta)[:, None] * product_difference
        )

        # Accumulate contributions for each customer
        np.add.at(result, users, contributions)

        return result

    def hessian_vector_QWb(
        self,
        params: Parameters,
        triplets: np.ndarray,
        direction_Q: np.ndarray,
        direction_W: np.ndarray,
        direction_b: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the joint Hessian-vector product for Q, W, b.

        P is fixed.
        Mixed derivatives between Q, W and b are included.
        The full Hessian is not explicitly constructed.
        """

        if (
            direction_Q.shape != params.Q.shape
            or direction_W.shape != params.W.shape
            or direction_b.shape != params.b.shape
        ):
            raise ValueError("Invalid Block 2 direction shapes")

        (
            margin,
            _,
            i,
            j,
            user_vectors,
            _,
            text_difference,
        ) = self._batch(params, triplets)

        n_triplets = len(triplets)

        # Logistic curvature divided by N
        omega = self._curvature(margin) / n_triplets

        # Directional difference in product representations
        directional_difference = (
            direction_Q[i]
            - direction_Q[j]
            + text_difference @ direction_W.T
        )

        # eta = h_ij^T * v
        eta = (
            np.sum(
                user_vectors * directional_difference,
                axis=1,
            )
            + direction_b[i]
            - direction_b[j]
        )

        # Combined Hessian-vector weight
        weight = omega * eta
        weighted_users = weight[:, None] * user_vectors

        # Initialize with regularization contributions
        Hv_Q = self.lambda_f * direction_Q
        Hv_W = self.lambda_w * direction_W
        Hv_b = self.lambda_f * direction_b

        # Hessian-vector contributions for Q
        np.add.at(Hv_Q, i, weighted_users)
        np.add.at(Hv_Q, j, -weighted_users)

        # Hessian-vector contributions for W
        Hv_W += weighted_users.T @ text_difference

        # Hessian-vector contributions for b
        np.add.at(Hv_b, i, weight)
        np.add.at(Hv_b, j, -weight)

        return Hv_Q, Hv_W, Hv_b
