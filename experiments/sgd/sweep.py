"""Run a reproducible SGD sweep, skipping completed trials on restart."""
import argparse
import json
from copy import deepcopy
from pathlib import Path
from time import perf_counter

from evaluate_training import save_run, plot_runs
from recommendation_system.data import load_data
from recommendation_system.experiments import run
from recommendation_system.model import HybridMatrixFactorization
from recommendation_system.objective import PairwiseObjective
from recommendation_system.sgd.config import read_yaml, build_sgd

ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/sgd/sweep")
    parser.add_argument("--common", type=Path, default=ROOT / "configs/common.yaml")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sgd.yaml")
    parser.add_argument("--sweep", type=Path, default=ROOT / "configs/sgd_experiments.yaml")
    args = parser.parse_args()
    common = read_yaml(args.common)
    base = read_yaml(args.config)
    sweep = read_yaml(args.sweep)
    common["regularization"] = sweep["regularization"]
    data = load_data(ROOT / "data")
    objective = PairwiseObjective(HybridMatrixFactorization(data.Z, data.q_index),
                                  **common["regularization"])
    rows = []
    configurations = [(f"sgd-lr-{rate:g}", rate, "disabled") for rate in sweep["learning_rates"]]
    configurations += [(f"sgd-{mode}", sweep["alpha_initial"], mode) for mode in sweep.get("adaptive_modes", [])]
    for name, rate, mode in configurations:
        settings = deepcopy(base)
        settings["learning_rate"] = rate
        settings["max_gradient_work"] = sweep["epochs"] * len(data.train_triplets)
        settings["line_search"]["mode"] = mode
        if mode != "disabled":
            settings["line_search"]["alpha_initial"] = rate
        if "gradient_tolerance" in sweep:
            settings["gradient_tolerance"] = sweep["gradient_tolerance"]
            settings["max_gradient_work"] *= 2
        directory = args.results / name
        config = {"common": common, "sgd": settings}
        optimizer = build_sgd(common, settings)
        for seed in common["training"]["seeds"]:
            history = directory / "training_data" / f"{name}-{seed}.json"
            model = history.with_suffix(".npz")
            if history.exists():
                record = json.loads(history.read_text())
                if record["settings"] != config:
                    raise ValueError(f"Configuration changed: use a new results directory ({history})")
                if record["failure"] or (record["selected"] is not None and model.exists()):
                    print(f"{name} {seed}: already recorded", flush=True)
                    rows.append(record)
                    continue
            print(f"{name} {seed}: training", flush=True)
            optimizer = build_sgd(common, settings)
            result = run(optimizer, objective, data, seed=seed,
                         work_interval=common["training"]["work_interval"],
                         model_settings=common["model"])
            save_run(directory / "training_data", name, seed, optimizer, result, config)
            plot_runs({(name, seed): result}, directory / f"plots-{seed}")
            record = json.loads(history.read_text())
            rows.append(record)
            print(f"{name} {seed}: {record['failure'] or record['selected']}", flush=True)
    summary = []
    for name, rate, mode in configurations:
        group = [r for r in rows if r["method"] == name]
        valid = len(group) == len(common["training"]["seeds"]) and all(r["selected"] is not None for r in group)
        scores = [r["selected"]["validation"]["ndcg_at_10"] for r in group] if valid else []
        summary.append({"method": name, "learning_rate": rate, "mode": mode, "mean_ndcg": sum(scores)/len(scores) if scores else None,
                        "elapsed_seconds": sum(r["elapsed_seconds"] for r in group)})
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__":
    main()
