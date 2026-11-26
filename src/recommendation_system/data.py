"""Read the supplied data and align text features by product ID."""

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _rows(path):
    with path.open(encoding="utf-8", newline="") as file:
        yield from csv.DictReader(file)


@dataclass
class Dataset:
    Z: np.ndarray
    q_index: np.ndarray
    catalogs: dict[str, np.ndarray]
    reviewed: dict[int, set[int]]
    positives: dict[int, set[int]]
    train_triplets: np.ndarray
    validation_triplets: np.ndarray
    relevant: dict[tuple[str, str], dict[int, set[int]]]
    n_users: int


def load_data(directory: str | Path, *, include_test: bool = False) -> Dataset:
    """Load prepared inputs. Opt in to test relevance only for final evaluation."""
    directory = Path(directory)
    products = sorted(_rows(directory / "products.csv"),
                      key=lambda row: int(row["product_id"]))
    # Product IDs index text features directly, even when archive rows arrive out of order.
    product_ids = np.array([int(row["product_id"]) for row in products])
    if not np.array_equal(product_ids, np.arange(len(products))):
        raise ValueError("Expected contiguous product IDs starting at zero")

    catalogs = defaultdict(list)
    for product_id, row in enumerate(products):
        catalogs[row["catalog"]].append(product_id)
    q_index = np.array([int(row["q_index"]) for row in products])
    with np.load(directory / "product_features.npz") as features:
        feature_ids = features["product_ids"]
        order = np.argsort(feature_ids)
        if not np.array_equal(feature_ids[order], product_ids):
            raise ValueError("Text feature IDs do not match the product catalog")
        Z = features["Z"][order]

    reviewed = defaultdict(set)
    positives = defaultdict(set)
    for row in _rows(directory / "train.csv"):
        user, product = int(row["user_id"]), int(row["product_id"])
        reviewed[user].add(product)
        if int(row["rating"]) >= 4:
            positives[user].add(product)

    # Keep test relevance out of training and validation runs by default.
    relevant = defaultdict(lambda: defaultdict(set))
    for row in _rows(directory / "evaluation.csv"):
        if row["split"] == "test" and not include_test:
            continue
        key = row["task"], row["split"]
        relevant[key][int(row["user_id"])].add(int(row["relevant_product_id"]))

    def triplets(filename):
        columns = ("user_id", "positive_id", "comparison_id")
        return np.array([tuple(int(row[column]) for column in columns)
                         for row in _rows(directory / filename)], dtype=int).reshape(-1, 3)

    return Dataset(
        Z, q_index, {name: np.array(ids) for name, ids in catalogs.items()},
        dict(reviewed), dict(positives), triplets("train_triplets.csv"),
        triplets("validation_triplets.csv"),
        {key: dict(users) for key, users in relevant.items()}, max(reviewed) + 1,
    )
