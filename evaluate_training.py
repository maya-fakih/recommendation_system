"""Run shared training comparisons and save checkpoint histories and figures."""

import argparse
from dataclasses import asdict, is_dataclass
from importlib import import_module
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np

from recommendation_system.data import load_data
from recommendation_system.experiments import SEEDS, run
from recommendation_system.optimizer import Optimizer
from recommendation_system.model import HybridMatrixFactorization
from recommendation_system.objective import PairwiseObjective


ROOT = Path(__file__).resolve().parent


def save_run(directory, method, seed, optimizer, result, settings):
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{method}-{seed}"
    history = {
        "method": method,
        "seed": seed,
        "optimizer": f"{type(optimizer).__module__}:{type(optimizer).__qualname__}",
        "optimizer_settings": asdict(optimizer) if is_dataclass(optimizer) else None,
        "settings": settings,
        "checkpoints": [asdict(checkpoint) for checkpoint in result.checkpoints],
        "selected": asdict(result.selected) if result.selected is not None else None,
        "elapsed_seconds": result.elapsed_seconds,
        "training_seconds": result.training_seconds,
        "gradient_work": result.gradient_work,
        "diagnostics": [
            {"gradient_work": work, "updates": updates, "values": values}
            for work, updates, values in result.diagnostics
        ],
        "failure": result.failure,
    }
    checkpoint_path = directory / f"{stem}.npz"
    if result.selected_params is None:
        checkpoint_path.unlink(missing_ok=True)
    else:
        np.savez_compressed(checkpoint_path, **vars(result.selected_params))
    (directory / f"{stem}.json").write_text(
        json.dumps(history, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )


def plot_runs(runs, directory):
    directory.mkdir(parents=True, exist_ok=True)
    loss_figure, loss_axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True, layout="constrained")
    ndcg_figure, ndcg_axis = plt.subplots(figsize=(9, 4), layout="constrained")
    rate_figure, rate_axis = plt.subplots(figsize=(9, 4), layout="constrained")

    for index, ((method, seed), result) in enumerate(runs.items()):
        label = f"{method.upper()} · {seed}"
        color = f"C{index % 10}"
        work = [point.gradient_work for point in result.checkpoints]
        if work:
            loss_axes[0].plot(work, [p.train_loss for p in result.checkpoints],
                              label=label, color=color, marker=".")
            loss_axes[1].plot(work, [p.validation_loss for p in result.checkpoints],
                              label=label, color=color, marker=".")
            ndcg_axis.plot(work, [p.validation.ndcg_at_10 for p in result.checkpoints],
                           label=label, color=color, marker=".")
        rates = [(work, values["learning_rate"]) for work, _, values in result.diagnostics
                 if method == "salr" and "learning_rate" in values]
        if rates:
            rate_axis.plot(*zip(*rates), label=label, color=color)

    for axis, label in zip(
        (*loss_axes, ndcg_axis, rate_axis),
        ("Training loss", "Validation loss", "Validation NDCG@10", "SALR learning rate"),
    ):
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend(ncol=3, fontsize="small")
        else:
            axis.text(0.5, 0.5, "No observations", ha="center", transform=axis.transAxes)
    for axis in (loss_axes[1], ndcg_axis, rate_axis):
        axis.set_xlabel("Gradient work (triplets)")
    for figure, filename in (
        (loss_figure, "loss.png"), (ndcg_figure, "validation_ndcg.png"),
        (rate_figure, "salr_learning_rate.png"),
    ):
        figure.savefig(directory / filename, dpi=150)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--optimizer", action="append", metavar="NAME=MODULE:FACTORY", required=True,
        help="Import a zero-argument factory for an optimizer. Repeat to compare methods.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--lambda-f", type=float, default=0.01)
    parser.add_argument("--lambda-w", type=float, default=0.01)
    parser.add_argument("--work-interval", type=int, default=1024)
    args = parser.parse_args()

    factories = {}
    for spec in args.optimizer:
        name, _, entry = spec.partition("=")
        module, _, attribute = entry.partition(":")
        if not name or not name.isascii() or not name.replace("-", "").isalnum():
            parser.error(f"Invalid optimizer name in {spec!r}")
        if not module or not attribute or name in factories:
            parser.error(f"Expected a unique NAME=MODULE:FACTORY, got {spec!r}")
        try:
            factories[name] = getattr(import_module(module), attribute)
        except (ImportError, AttributeError) as error:
            parser.error(str(error))
        if not callable(factories[name]):
            parser.error(f"{entry!r} is not callable")

    data = load_data(args.data)
    objective = PairwiseObjective(
        HybridMatrixFactorization(data.Z, data.q_index), args.lambda_f, args.lambda_w,
    )
    settings = {"lambda_f": args.lambda_f, "lambda_w": args.lambda_w,
                "work_interval": args.work_interval}
    runs = {}
    for method, factory in factories.items():
        for seed in dict.fromkeys(args.seeds):
            optimizer = factory()
            if not isinstance(optimizer, Optimizer):
                parser.error(f"The factory for {method!r} must return an Optimizer instance")
            result = run(optimizer, objective, data, seed=seed, work_interval=args.work_interval)
            save_run(args.results / "training_data", method, seed, optimizer, result, settings)
            runs[method, seed] = result
            print(f"{method.upper()} {seed} · {len(result.checkpoints)} checkpoints"
                  f" · {result.failure or 'completed'}", flush=True)

    plot_runs(runs, args.results / "training_imgs")
    return int(any(result.selected is None for result in runs.values()))


if __name__ == "__main__":
    raise SystemExit(main())
