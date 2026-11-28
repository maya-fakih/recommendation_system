"""Run SGD using YAML settings and the shared evaluation pipeline."""
import argparse
from pathlib import Path

from evaluate_training import save_run, plot_runs
from recommendation_system.data import load_data
from recommendation_system.experiments import run
from recommendation_system.model import HybridMatrixFactorization
from recommendation_system.objective import PairwiseObjective
from recommendation_system.sgd.config import read_yaml, build_sgd

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common", type=Path, default=ROOT / "configs/common.yaml")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sgd.yaml")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--results", type=Path, default=ROOT / "results/sgd")
    args = parser.parse_args()
    common, settings = read_yaml(args.common), read_yaml(args.config)
    optimizer = build_sgd(common, settings)
    if common["evaluation"]["top_k"] != 10:
        parser.error("Shared evaluation currently supports top_k=10 only")
    data = load_data(args.data)
    objective = PairwiseObjective(
        HybridMatrixFactorization(data.Z, data.q_index),
        **common["regularization"],
    )
    mode = optimizer.line_search.mode
    name = "sgd" if mode == "disabled" else f"sgd-{mode}"
    runs = {}
    for seed in dict.fromkeys(common["training"]["seeds"]):
        optimizer = build_sgd(common, settings)
        result = run(optimizer, objective, data, seed=seed,
                     work_interval=common["training"]["work_interval"],
                     model_settings=common["model"])
        save_run(args.results / "training_data", name, seed, optimizer, result,
                 {"common": common, "sgd": settings})
        runs[name, seed] = result
        print(f"{name} {seed}: {result.failure or 'completed'}", flush=True)
    plot_runs(runs, args.results / name / "training_imgs")
    return int(any(result.selected is None for result in runs.values()))


if __name__ == "__main__":
    raise SystemExit(main())
