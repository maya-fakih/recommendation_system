# Repository layout

- `src/recommendation_system/`: shared model, objective, data, optimizer interface, and independent optimizer packages (sgd, sam, salr, bcd).
- `experiments/`: executable experiment modules, organized by optimizer.
  - `sgd/train.py`, `sgd/sweep.py`; equivalent modules for SAM, SALR and BCD.
  - `common/`: shared checkpoint persistence, evaluation, plotting and statistics.
  - `compare.py`: eventual cross-method comparison.
- `configs/`: YAML settings, including common regularization and seeds.
- `tests/`: unit and integration tests.
- `data/`: downloaded dataset; use make setup_data.
- `results/`: generated outputs, separated by optimizer/configuration/seed.
- `Makefile`: user-facing commands, invoking modules from repository root.

This is the target layout, not a claim that every module already exists. Keep optimizer implementations independent; experiment modules may use shared utilities. Use fresh optimizer instances per seed. Keep latest resumable state separate from best validation model. Do not use test data for tuning. Preserve existing entry points until callers are migrated. Coordinate common regularization and comparison budgets before final evaluation.
