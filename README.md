# Product recommendation

ENMG 616 project on hybrid matrix factorization for product recommendation.

## Setup

From the repository root, with Python 3.10+, Make, curl, and unzip installed:

```sh
make install
make setup_data
source ../recommendation-system-venv/bin/activate
python -m unittest discover -s tests -v
```

`make install` creates `../recommendation-system-venv` and installs the editable
package and dependencies declared in `pyproject.toml`. `make setup_data` downloads
the published `course-data-v1` archive and places its eight dataset files in
`data/`. Rerunning it refreshes those files.

## Training

See [TASKS.md](TASKS.md) for optimizer ownership and references. Build the shared
model from `load_data("data")`, construct a `PairwiseObjective`, then pass it and
an optimizer instance to `recommendation_system.experiments.run`. Use the same
seeds and gradient-work checkpoint interval when comparing optimizers.
