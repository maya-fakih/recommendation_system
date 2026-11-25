"""Shared initialization, gradient-work monitoring, and checkpoint selection."""

from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .data import Dataset
from .evaluation import RankingMetrics, evaluate
from .model import Parameters
from .objective import PairwiseObjective
from .optimizer import CountedObjective, Optimizer


SEEDS = (616, 617, 618)


def initialize(data: Dataset, seed: int) -> tuple[Parameters, np.random.Generator]:
    """Give each method the same starting arrays and an independent shuffle stream."""
    init_seed, shuffle_seed = np.random.SeedSequence(seed).spawn(2)
    rng = np.random.default_rng(init_seed)
    n_products = np.count_nonzero(data.q_index >= 0)
    params = Parameters(
        rng.normal(0, 0.01, (data.n_users, 32)),
        rng.normal(0, 0.01, (n_products, 32)),
        rng.normal(0, 0.01, (32, data.Z.shape[1])),
        np.zeros(n_products),
    )
    return params, np.random.default_rng(shuffle_seed)


@dataclass
class Checkpoint:
    gradient_work: int
    updates: int
    data_passes: float | None
    elapsed_seconds: float
    training_seconds: float
    train_loss: float
    validation_loss: float
    penalty: float
    validation: RankingMetrics


@dataclass
class RunResult:
    checkpoints: list[Checkpoint]
    selected: Checkpoint | None
    selected_params: Parameters | None
    elapsed_seconds: float
    training_seconds: float
    gradient_work: int
    diagnostics: list[tuple[int, int, dict[str, float]]]
    failure: str | None


def run(
    optimizer: Optimizer,
    objective: PairwiseObjective,
    data: Dataset,
    *,
    seed: int,
    work_interval: int,
) -> RunResult:
    """Monitor the first accepted state at or after each work threshold."""
    if work_interval < 1:
        raise ValueError("Expected a positive gradient-work interval")
    start = perf_counter()
    initial, shuffle_rng = initialize(data, seed)
    counted = CountedObjective(objective)
    checkpoints = []
    diagnostics = []
    selected = selected_params = None
    next_work = work_interval
    monitoring_seconds = 0.0
    failure = None

    try:
        for state in optimizer.train(counted, initial, data.train_triplets, shuffle_rng):
            params = state.params
            if any(not np.isfinite(array).all() for array in vars(params).values()):
                raise FloatingPointError("Non-finite accepted parameters")
            if state.diagnostics is not None:
                diagnostics.append((counted.gradient_work, state.updates, dict(state.diagnostics)))
            if counted.gradient_work < next_work:
                continue
            checkpoint_start = perf_counter()
            train_loss = objective.data_loss(params, data.train_triplets)
            validation_loss = objective.data_loss(params, data.validation_triplets)
            penalty = objective.penalty(params)
            if not np.isfinite([train_loss, validation_loss, penalty]).all():
                raise FloatingPointError("Non-finite checkpoint loss")
            metrics = evaluate(
                data, "ordinary", "validation",
                lambda user, ids: objective.model.scores(params, user, ids),
            )
            snapshot = Checkpoint(
                counted.gradient_work, state.updates, state.data_passes,
                checkpoint_start - start, checkpoint_start - start - monitoring_seconds,
                train_loss, validation_loss, penalty, metrics,
            )
            checkpoints.append(snapshot)
            if selected is None or metrics.ndcg_at_10 > selected.validation.ndcg_at_10:
                selected, selected_params = snapshot, deepcopy(params)
            monitoring_seconds += perf_counter() - checkpoint_start
            next_work = (counted.gradient_work // work_interval + 1) * work_interval
    except FloatingPointError as error:
        failure = str(error)
    elapsed = perf_counter() - start
    return RunResult(
        checkpoints, selected if failure is None else None,
        selected_params if failure is None else None, elapsed, elapsed - monitoring_seconds,
        counted.gradient_work, diagnostics, failure,
    )


def selection_key(results: list[RunResult]) -> tuple[float, float]:
    """Rank complete run groups by mean validation NDCG, then lower validation loss."""
    if not results or any(result.selected is None for result in results):
        raise ValueError("Every run must have a valid selected checkpoint")
    return (
        float(np.mean([result.selected.validation.ndcg_at_10 for result in results])),
        -float(np.mean([result.selected.validation_loss for result in results])),
    )
