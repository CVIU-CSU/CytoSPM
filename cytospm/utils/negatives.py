"""Negative (non-diagnostic) PentaCyto classes excluded from the base metric.

The list lives in ``data/texts/tct_ngc_dev30_negatives.json``.
"""
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEV30_NEGATIVES_JSON = _REPO_ROOT / "data" / "texts" / "tct_ngc_dev30_negatives.json"


def load_dev30_negative_classes(path: str | Path | None = None) -> list[str]:
    """Return the negative class names (optionally from another JSON)."""
    p = Path(path) if path is not None else DEV30_NEGATIVES_JSON
    with open(p, encoding="utf-8") as f:
        return list(json.load(f)["negative_class_names"])
