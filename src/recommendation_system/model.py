"""Shared scoring model for hybrid matrix factorization."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Parameters:
    """Trainable arrays for users, ordinary products, and text projection."""

    P: np.ndarray  # One k-dimensional row per user
    Q: np.ndarray  # One k-dimensional row per ordinary product
    W: np.ndarray  # Shape (k, text_feature_dimension)
    b: np.ndarray  # One bias per ordinary product


class HybridMatrixFactorization:
    """Score products using user preferences, interactions, and product text."""

    def __init__(self, Z: np.ndarray, q_index: np.ndarray):
        """
        Store the fixed text features and interaction-row mapping.

        Both arrays are indexed by product ID. For a cold product, its
        q_index is -1 and it has no learned interaction vector or bias.
        """
        self.Z = Z
        self.q_index = q_index

    def scores(
        self,
        params: Parameters,
        user_id: int,
        product_ids: np.ndarray,
    ) -> np.ndarray:
        """Return ranking scores for one user, in the given product order."""
        user = params.P[user_id]
        product_vectors = self.Z[product_ids] @ params.W.T
        interaction_rows = self.q_index[product_ids]

        ordinary = interaction_rows >= 0
        ordinary_rows = interaction_rows[ordinary]
        product_vectors[ordinary] += params.Q[ordinary_rows]

        scores = product_vectors @ user
        scores[ordinary] += params.b[ordinary_rows]
        return scores
