"""Read experiment settings without coupling the optimizer to YAML."""
from pathlib import Path
import yaml
from .sgd import SGD, LineSearchSettings


def read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        settings = yaml.safe_load(stream)
    if not isinstance(settings, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return settings


def build_sgd(common: dict, settings: dict) -> SGD:
    """Construct an optimizer from the shared and SGD-specific mappings."""
    if common["training"]["drop_last"]:
        raise ValueError("SGD requires retaining the final partial batch")
    return SGD(
        learning_rate=settings["learning_rate"],
        batch_size=common["training"]["batch_size"],
        max_gradient_work=settings["max_gradient_work"],
        gradient_tolerance=settings.get("gradient_tolerance"),
        line_search=LineSearchSettings(**settings["line_search"]),
    )
