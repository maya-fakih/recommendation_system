"""Plain mini-batch SGD, following Goodfellow et al., Deep Learning, Ch. 8.
https://www.deeplearningbook.org/contents/optimization.html
"""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from ..model import Parameters
from ..optimizer import CountedObjective, Optimizer, TrainingState


@dataclass
class SGD(Optimizer):
    """Fixed learning-rate SGD with an explicit gradient-work budget."""

    learning_rate: float
    batch_size: int
    max_gradient_work: int

    def __post_init__(self) -> None:
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Expected a positive finite learning rate")
        if self.batch_size < 1 or self.max_gradient_work < 1:
            raise ValueError("Batch size and work budget must be positive")

    def train(
        self,
        objective: CountedObjective,
        initial: Parameters,
        triplets: np.ndarray,
        shuffle_rng: np.random.Generator,
    ) -> Iterator[TrainingState]:
        if len(triplets) == 0:
            raise ValueError("Training triplets must not be empty")

        params = initial
        updates = 0
        processed = 0
        while processed < self.max_gradient_work:
            order = shuffle_rng.permutation(len(triplets))
            for start in range(0, len(order), self.batch_size):
                batch = triplets[order[start:start + self.batch_size]]
                if processed + len(batch) > self.max_gradient_work:
                    return
                _, gradient = objective.loss_and_grad(params, batch)
                params = Parameters(
                    P=params.P - self.learning_rate * gradient.P,
                    Q=params.Q - self.learning_rate * gradient.Q,
                    W=params.W - self.learning_rate * gradient.W,
                    b=params.b - self.learning_rate * gradient.b,
                )
                updates += 1
                processed += len(batch)
                yield TrainingState(params, updates, processed / len(triplets))
