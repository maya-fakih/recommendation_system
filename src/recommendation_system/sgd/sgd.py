"""Plain mini-batch SGD and optional exploratory line searches.

SGD reference: Goodfellow et al., Deep Learning, Chapter 8.
https://www.deeplearningbook.org/contents/optimization.html
"""
from collections.abc import Iterator
from dataclasses import dataclass
import numpy as np

from ..model import Parameters
from ..optimizer import CountedObjective, Optimizer, TrainingState
from .armijo import armijo, trial_parameters
from .full_backtracking import full_backtracking
from .sampled_backtracking import sampled_backtracking


@dataclass
class LineSearchSettings:
    mode: str
    alpha_initial: float
    beta: float
    c: float
    max_backtracks: int
    loss_batch_size: int

    def __post_init__(self):
        if self.mode not in ("disabled", "minibatch", "full", "sampled"):
            raise ValueError("Unknown line-search mode")
        if not np.isfinite(self.alpha_initial) or self.alpha_initial <= 0:
            raise ValueError("Expected positive initial step")
        if not 0 < self.beta < 1 or not 0 < self.c < 1:
            raise ValueError("beta and c must lie in (0, 1)")
        if self.max_backtracks < 0 or self.loss_batch_size < 1:
            raise ValueError("Invalid search limits")


@dataclass
class SGD(Optimizer):
    """Fixed-rate SGD by default; optional search settings choose trial steps."""
    learning_rate: float
    batch_size: int
    max_gradient_work: int
    line_search: LineSearchSettings | None = None

    def __post_init__(self):
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Expected a positive finite learning rate")
        if self.batch_size < 1 or self.max_gradient_work < 1:
            raise ValueError("Batch size and work budget must be positive")

    def train(self, objective: CountedObjective, initial: Parameters,
              triplets: np.ndarray, shuffle_rng: np.random.Generator
              ) -> Iterator[TrainingState]:
        if len(triplets) == 0:
            raise ValueError("Training triplets must not be empty")
        settings = self.line_search
        mode = settings.mode if settings else "disabled"
        if mode == "sampled" and settings.loss_batch_size > len(triplets):
            raise ValueError("Loss batch exceeds training set size")
        # Copy the RNG state so evaluation sampling does not change batch order.
        from copy import deepcopy
        evaluation_rng = deepcopy(shuffle_rng)
        # Advance only the copied stream to separate it from gradient shuffling.
        evaluation_rng = np.random.default_rng(
            evaluation_rng.integers(0, 2**63, dtype=np.int64)
        )
        params = initial
        updates = processed = rejected = loss_work = 0
        order = None
        offset = 0
        resume = getattr(self, "resume_state", None)
        if resume is not None:
            params = resume["params"]
            updates, processed, rejected, loss_work = resume["counts"]
            order, offset = resume["order"], resume["offset"]
            shuffle_rng.bit_generator.state = resume["shuffle"]
            evaluation_rng.bit_generator.state = resume["evaluation"]
            objective.gradient_work = processed
        while processed < self.max_gradient_work:
            if order is None:
                order = shuffle_rng.permutation(len(triplets))
                offset = 0
            for start in range(offset, len(order), self.batch_size):
                batch = triplets[order[start:start + self.batch_size]]
                if processed + len(batch) > self.max_gradient_work:
                    return
                batch_loss, gradient = objective.loss_and_grad(params, batch)
                processed += len(batch)
                alpha = self.learning_rate
                calls = 0
                if mode != "disabled":
                    args = (settings.alpha_initial, settings.beta)
                    if mode == "minibatch":
                        alpha, calls = armijo(objective, params, gradient, batch,
                            batch_loss, *args, settings.c, settings.max_backtracks)
                        loss_work += calls * len(batch)
                    elif mode == "full":
                        alpha, calls = full_backtracking(objective, params, gradient,
                            triplets, *args, settings.max_backtracks)
                        loss_work += calls * len(triplets)
                    else:
                        alpha, calls = sampled_backtracking(objective, params, gradient,
                            triplets, evaluation_rng, settings.loss_batch_size,
                            *args, settings.max_backtracks)
                        loss_work += calls * settings.loss_batch_size
                if alpha is None:
                    rejected += 1
                else:
                    params = trial_parameters(params, gradient, alpha)
                    updates += 1
                # A rejected step retains the last accepted parameters. Report
                # work even then, so checkpoints and rejection counts remain visible.
                self.resume_state = {
                    "params": params, "counts": (updates, processed, rejected, loss_work),
                    "order": order, "offset": start + len(batch),
                    "shuffle": shuffle_rng.bit_generator.state,
                    "evaluation": evaluation_rng.bit_generator.state,
                }
                yield TrainingState(params, updates, processed / len(triplets), {
                    "learning_rate": float(alpha or 0),
                    "accepted": float(alpha is not None),
                    "rejected_steps": float(rejected),
                    "loss_work": float(loss_work),
                    "search_loss_calls": float(calls),
                })
            order = None
            offset = 0
