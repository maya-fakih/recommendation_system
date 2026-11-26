"""Checks for the model, objective, evaluator, and common experiment runner."""

import unittest

import numpy as np

from recommendation_system.data import Dataset
from recommendation_system.evaluation import evaluate, top_ten
from recommendation_system.experiments import initialize, run, selection_key
from recommendation_system.model import HybridMatrixFactorization, Parameters
from recommendation_system.objective import PairwiseObjective
from recommendation_system.optimizer import Optimizer, TrainingState, shuffled_batches


class SharedTests(unittest.TestCase):
    def setUp(self):
        self.Z = np.array([[1., 0.], [0., 1.], [1., 1.]])
        # Swap ordinary factor rows and include a text-only product.
        self.q_index = np.array([1, 0, -1])
        self.model = HybridMatrixFactorization(self.Z, self.q_index)
        self.params = Parameters(
            np.array([[2., 3.], [0.1, 0.2]]), np.array([[4., 1.], [2., 5.]]),
            np.diag([2., 3.]), np.array([7., 11.]),
        )
        self.triplets = np.array([[0, 0, 1], [0, 0, 1], [0, 1, 0]])
        self.objective = PairwiseObjective(self.model, 0.01, 0.02)

    def test_scores_with_cold_products_and_permuted_rows(self):
        scores = self.model.scores(self.params, 0, np.array([2, 0, 1]))
        np.testing.assert_allclose(scores, [13., 34., 27.])

    def test_gradient_against_central_differences(self):
        loss, gradient = self.objective.loss_and_grad(self.params, self.triplets)
        self.assertAlmostEqual(loss, self.objective.loss(self.params, self.triplets))
        scores = self.model.scores(self.params, 0, np.array([0, 1]))
        margins = np.array([scores[0] - scores[1], scores[0] - scores[1],
                            scores[1] - scores[0]])
        self.assertAlmostEqual(self.objective.data_loss(self.params, self.triplets),
                               np.logaddexp(0, -margins).mean())
        # Check every analytic derivative against a centered perturbation.
        eps = 1e-5
        for name, values in vars(self.params).items():
            for index in np.ndindex(values.shape):
                original = values[index]
                values[index] = original + eps
                upper = self.objective.loss(self.params, self.triplets)
                values[index] = original - eps
                lower = self.objective.loss(self.params, self.triplets)
                values[index] = original
                self.assertAlmostEqual(getattr(gradient, name)[index],
                                       (upper - lower) / (2 * eps), places=7)

    def test_ranking_ties_and_review_filter(self):
        np.testing.assert_array_equal(
            top_ten(np.array([5, 2, 0]), np.array([1., 1., 0.])), [2, 5, 0]
        )
        data = Dataset(
            self.Z, self.q_index, {"ordinary": np.arange(12)}, {0: {0}},
            {}, self.triplets, self.triplets, {("ordinary", "validation"): {0: {2, 11}}}, 1,
        )
        result = evaluate(data, "ordinary", "validation", lambda user, ids: np.zeros(len(ids)))
        self.assertEqual(result.n_users, 1)
        self.assertAlmostEqual(result.recall_at_10, 0.5)
        self.assertAlmostEqual(result.ndcg_at_10, 1 / (1 + np.log2(3)))

    def test_batches_keep_the_last_partial_batch(self):
        triplets = np.arange(513 * 3).reshape(-1, 3)
        batches = list(shuffled_batches(triplets, np.random.default_rng(616)))
        self.assertEqual([len(batch) for batch in batches], [256, 256, 1])
        np.testing.assert_array_equal(np.sort(np.concatenate(batches)[:, 0]), triplets[:, 0])

    def test_repeatable_initialization_and_counted_training(self):
        class TwoGradientOptimizer(Optimizer):
            def train(self, objective, initial, triplets, shuffle_rng):
                objective.loss_and_grad(initial, triplets)
                objective.loss(initial, triplets)
                objective.loss_and_grad(initial, triplets)
                yield TrainingState(initial, 1)
                # Mutate after yielding to check that the checkpoint owns a copy.
                initial.P[:] = 123

        data = Dataset(
            self.Z, self.q_index, {"ordinary": np.array([0, 1])}, {}, {},
            self.triplets, self.triplets, {("ordinary", "validation"): {0: {1}}}, 2,
        )
        first, first_shuffle = initialize(data, 616)
        second, second_shuffle = initialize(data, 616)
        np.testing.assert_array_equal(first.P, second.P)
        np.testing.assert_array_equal(first_shuffle.permutation(100), second_shuffle.permutation(100))
        result = run(TwoGradientOptimizer(), self.objective, data, seed=616, work_interval=4)
        self.assertIsNone(result.failure)
        self.assertEqual(result.gradient_work, 6)
        self.assertEqual(result.selected.gradient_work, 6)
        np.testing.assert_array_equal(result.selected_params.P, first.P)
        self.assertEqual(selection_key([result])[0], result.selected.validation.ndcg_at_10)
