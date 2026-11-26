"""Sharpness-aware minimization with a plain SGD outer step."""

from dataclasses import dataclass

import numpy as np

from .model import Parameters
from .optimizer import CountedObjective, Optimizer


@dataclass
class SAM(Optimizer):
    """Configure the mini-batch learning rate, L2 radius, and epoch count."""

    learning_rate: float = 0.1
    radius: float = 0.05
    epochs: int = 3

    def __post_init__(self):
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Expected a finite, positive learning rate")
        if not np.isfinite(self.radius) or self.radius < 0:
            raise ValueError("Expected a finite, nonnegative radius")
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, (int, np.integer)):
            raise ValueError("Expected an integer epoch count")
        if self.epochs < 1:
            raise ValueError("Expected at least one epoch")

    def _perturb(self, params: Parameters, gradient: Parameters) -> Parameters:
        # Normalize all four blocks together to get a single L2 perturbation.
        norm = np.sqrt(sum(np.vdot(block, block) for block in vars(gradient).values()))
        if norm == 0 or self.radius == 0:
            return params

        return Parameters(**{
            name: block + self.radius * (getattr(gradient, name) / norm)
            for name, block in vars(params).items()
        })

    def _step(
        self, objective: CountedObjective, params: Parameters, batch: np.ndarray,
    ) -> Parameters:
        _, gradient = objective.loss_and_grad(params, batch)
        perturbed = self._perturb(params, gradient)
        if perturbed is not params:
            _, gradient = objective.loss_and_grad(perturbed, batch)

        # Apply the gradient at the perturbed point to the original iterate.
        return Parameters(**{
            name: block - self.learning_rate * getattr(gradient, name)
            for name, block in vars(params).items()
        })
