"""Training interface shared by SGD, SAM, SALR, and block coordinate descent."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .model import Parameters
from .objective import PairwiseObjective


BATCH_SIZE = 256


def shuffled_batches(
    triplets: np.ndarray, rng: np.random.Generator,
) -> Iterator[np.ndarray]:
    """Shuffle one pass of training triplets and keep the final partial batch."""
    order = rng.permutation(len(triplets))
    for start in range(0, len(order), BATCH_SIZE):
        yield triplets[order[start:start + BATCH_SIZE]]


class CountedObjective:
    """Count triplets in every gradient call, including probes and trial steps."""

    def __init__(self, objective: PairwiseObjective):
        self._objective = objective
        self.gradient_work = 0

    def loss(self, params: Parameters, triplets: np.ndarray) -> float:
        loss = self._objective.loss(params, triplets)
        if not np.isfinite(loss):
            raise FloatingPointError("Non-finite objective loss")
        return loss

    def loss_and_grad(self, params: Parameters, triplets: np.ndarray) -> tuple[float, Parameters]:
        # Charge work before evaluation so failed probes still count.
        self.gradient_work += len(triplets)
        loss, gradient = self._objective.loss_and_grad(params, triplets)
        if not np.isfinite(loss) or any(
            not np.isfinite(array).all() for array in vars(gradient).values()
        ):
            raise FloatingPointError("Non-finite loss or gradient")
        return loss, gradient


@dataclass
class TrainingState:
    """One accepted iterate, with optimizer-specific diagnostics if applicable."""

    params: Parameters
    updates: int
    data_passes: float | None = None
    diagnostics: dict[str, float] | None = None


class Optimizer(ABC):
    """Subclasses implement updates and stopping with their own settings."""

    @abstractmethod
    def train(
        self,
        objective: CountedObjective,
        initial: Parameters,
        triplets: np.ndarray,
        shuffle_rng: np.random.Generator,
    ) -> Iterator[TrainingState]:
        """Yield each accepted update or block solve. Use the objective for every gradient call."""
        raise NotImplementedError
