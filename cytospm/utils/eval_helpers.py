"""Helpers shared by the evaluation scripts."""
from __future__ import annotations

import json
from pathlib import Path



def resolve_ann_file(data_root: str, ann_file: str) -> Path:
    """Join a relative annotation path onto ``data_root``."""
    ann_path = Path(ann_file)
    if ann_path.is_absolute():
        return ann_path
    return Path(data_root) / ann_path


def derive_metainfo(ann_path) -> tuple[tuple[str, ...], int]:
    """Return ``(class_names_in_cat_id_order, num_classes)`` from a COCO ann JSON."""
    with open(ann_path, "r") as f:
        data = json.load(f)
    cats = sorted(data["categories"], key=lambda c: c["id"])
    return tuple(c["name"] for c in cats), len(cats)
