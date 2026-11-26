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
