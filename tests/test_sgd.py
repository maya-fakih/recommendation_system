"""Deterministic correctness checks; no dataset or training experiment required."""
import unittest
from pathlib import Path

import numpy as np

from recommendation_system.model import Parameters, HybridMatrixFactorization
from recommendation_system.objective import PairwiseObjective
from recommendation_system.optimizer import CountedObjective
from recommendation_system.sgd.sgd import SGD, LineSearchSettings
from recommendation_system.sgd.armijo import armijo, trial_parameters
from recommendation_system.sgd.full_backtracking import full_backtracking
from recommendation_system.sgd.sampled_backtracking import sampled_backtracking
from recommendation_system.sgd.config import read_yaml, build_sgd


class RecordingObjective(CountedObjective):
    def __init__(self, objective):
        super().__init__(objective)
        self.loss_batches = []
        self.gradient_batches = []

    def loss(self, params, triplets):
        self.loss_batches.append(triplets.copy())
        return super().loss(params, triplets)

    def loss_and_grad(self, params, triplets):
        self.gradient_batches.append(triplets.copy())
        return super().loss_and_grad(params, triplets)


class TestSGD(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(616)
        self.params = Parameters(*(rng.normal(0, .1, shape)
                                   for shape in [(2, 2), (3, 2), (2, 4), (3,)]))
        self.triplets = np.array([[0, 0, 1], [0, 0, 2], [1, 1, 2],
                                  [1, 1, 0], [0, 0, 1]])
        self.objective = PairwiseObjective(
            HybridMatrixFactorization(rng.normal(size=(3, 4)), np.arange(3)),
            .01, .02)

    def test_gradient_finite_difference(self):
        _, grad = self.objective.loss_and_grad(self.params, self.triplets)
        h = 1e-6
        for key, array in vars(self.params).items():
            for index in np.ndindex(array.shape):
                old = array[index]
                array[index] = old + h
                plus = self.objective.loss(self.params, self.triplets)
                array[index] = old - h
                minus = self.objective.loss(self.params, self.triplets)
                array[index] = old
                self.assertAlmostEqual((plus-minus)/(2*h),
                                       getattr(grad, key)[index], places=6)

    def test_fixed_step(self):
        counted = RecordingObjective(self.objective)
        state = next(SGD(.1, 2, 2).train(
            counted, self.params, self.triplets, np.random.default_rng(3)))
        _, grad = self.objective.loss_and_grad(self.params, counted.gradient_batches[0])
        expected = trial_parameters(self.params, grad, .1)
        for key, array in vars(expected).items():
            np.testing.assert_allclose(getattr(state.params, key), array)
        self.assertEqual(counted.gradient_work, 2)
        self.assertEqual(state.updates, 1)

    def test_armijo_condition(self):
        batch = self.triplets[:2]
        loss, grad = self.objective.loss_and_grad(self.params, batch)
        counted = RecordingObjective(self.objective)
        alpha, calls = armijo(counted, self.params, grad, batch, loss, 10., .5, .0001, 30)
        self.assertIsNotNone(alpha)
        norm2 = sum(np.sum(a*a) for a in vars(grad).values())
        self.assertLessEqual(self.objective.loss(trial_parameters(self.params, grad, alpha), batch),
                             loss - .0001 * alpha * norm2)
        self.assertEqual(calls, len(counted.loss_batches))
        self.assertEqual(counted.gradient_work, 0)

    def test_full_search_and_rejection(self):
        loss, grad = self.objective.loss_and_grad(self.params, self.triplets)
        counted = RecordingObjective(self.objective)
        alpha, calls = full_backtracking(counted, self.params, grad, self.triplets, 1., .5, 30)
        self.assertIsNotNone(alpha)
        self.assertLess(self.objective.loss(trial_parameters(self.params, grad, alpha), self.triplets), loss)
        self.assertEqual(calls, len(counted.loss_batches))
        uphill = Parameters(**{k: -a for k, a in vars(grad).items()})
        alpha, calls = full_backtracking(counted, self.params, uphill, self.triplets, 1e-4, .5, 3)
        self.assertIsNone(alpha)
        self.assertEqual(calls, 5)

    def test_sampled_search_keeps_batch_fixed(self):
        _, grad = self.objective.loss_and_grad(self.params, self.triplets[:2])
        counted = RecordingObjective(self.objective)
        alpha, calls = sampled_backtracking(counted, self.params, grad, self.triplets,
                                             np.random.default_rng(4), 4, 10., .5, 30)
        self.assertEqual(calls, len(counted.loss_batches))
        batch = counted.loss_batches[0]
        self.assertEqual(len(batch), 4)
        for other in counted.loss_batches:
            np.testing.assert_array_equal(other, batch)
        if alpha is not None:
            self.assertLess(self.objective.loss(trial_parameters(self.params, grad, alpha), batch),
                            self.objective.loss(self.params, batch))
        self.assertEqual(counted.gradient_work, 0)

    def test_all_modes_work_and_reproducibility(self):
        baseline_batches = None
        for mode in ("disabled", "minibatch", "sampled", "full"):
            with self.subTest(mode=mode):
                opt = SGD(.1, 2, 10, LineSearchSettings(mode, 1., .5, .0001, 10, 4))
                counted = RecordingObjective(self.objective)
                states = list(opt.train(counted, self.params, self.triplets, np.random.default_rng(7)))
                self.assertEqual(counted.gradient_work, 10)
                self.assertEqual([len(b) for b in counted.gradient_batches], [2, 2, 1]*2)
                self.assertEqual(states[-1].data_passes, 2)
                self.assertEqual(states[-1].diagnostics["loss_work"],
                                 sum(map(len, counted.loss_batches)))
                self.assertEqual(states[-1].updates + states[-1].diagnostics["rejected_steps"], 6)
                if baseline_batches is None:
                    baseline_batches = counted.gradient_batches
                for a, b in zip(baseline_batches, counted.gradient_batches):
                    np.testing.assert_array_equal(a, b)
                again = list(opt.train(CountedObjective(self.objective), self.params,
                                       self.triplets, np.random.default_rng(7)))
                for key, array in vars(states[-1].params).items():
                    np.testing.assert_array_equal(array, getattr(again[-1].params, key))

    def test_yaml(self):
        root = Path(__file__).resolve().parents[1]
        common = read_yaml(root / "configs/common.yaml")
        settings = read_yaml(root / "configs/sgd.yaml")
        opt = build_sgd(common, settings)
        self.assertEqual(opt.batch_size, common["training"]["batch_size"])
        self.assertEqual(opt.learning_rate, settings["learning_rate"])
        self.assertEqual(opt.line_search.mode, settings["line_search"]["mode"])


if __name__ == "__main__":
    unittest.main()
