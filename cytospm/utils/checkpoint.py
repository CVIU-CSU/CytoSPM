from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Optional

import torch

try:  # Keep imports usable without the training environment.
    from mmengine.runner.checkpoint import CheckpointLoader
except ImportError:  # pragma: no cover - exercised only in a minimal env
    CheckpointLoader = None  # type: ignore[assignment,misc]


SAFE_CHECKPOINT_PREFIX = "safe://"
TRUSTED_WEDETECT_PREFIX = "trusted-wedetect://"
# Hugging Face revision and SHA256 of the WeDetect-Tiny initialization weight.
WEDETECT_HF_REVISION = "3b9d0670cbf300d0b48f900ae87a6c4785d88939"
WEDETECT_SHA256 = (
    "e1c881120c3ea51b09242f16a7da66f62423f47adabe7ffd2814ad002d2ac62c"
)


def _uri_path(filename: str, prefix: str) -> str:
    """Strip a local-only URI prefix and normalize the resulting path."""
    if not isinstance(filename, str) or not filename.startswith(prefix):
        raise ValueError(f"expected a {prefix!r} checkpoint URI, got {filename!r}")
    path = filename[len(prefix):]
    if not path or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", path):
        raise ValueError(f"{prefix} accepts a local path, got {filename!r}")
    return os.path.expanduser(path)


def _load_safe_local(filename: str, map_location: Any = "cpu") -> Any:
    path = _uri_path(filename, SAFE_CHECKPOINT_PREFIX)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"safe checkpoint does not exist: {path}")
    return torch.load(path, map_location=map_location, weights_only=True)


def _load_trusted_wedetect(filename: str, map_location: Any = "cpu") -> Any:
    """Load the WeDetect-Tiny initialization checkpoint after a SHA256 check.

    This checkpoint stores MMEngine objects and cannot be read with
    ``weights_only=True``, so its content hash is verified first.
    """
    path = _uri_path(filename, TRUSTED_WEDETECT_PREFIX)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"trusted WeDetect checkpoint does not exist: {path}")
    with open(path, "rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
        got = digest.hexdigest()
        if got != WEDETECT_SHA256:
            raise RuntimeError(
                "refusing trusted WeDetect checkpoint with unexpected SHA256: "
                f"{path}\nexpected {WEDETECT_SHA256}\ngot      {got}\n"
                f"fetch revision {WEDETECT_HF_REVISION} with tools/fetch_checkpoints.sh"
            )
        stream.seek(0)
        return torch.load(stream, map_location=map_location, weights_only=False)


if CheckpointLoader is not None:
    CheckpointLoader.register_scheme(
        SAFE_CHECKPOINT_PREFIX, _load_safe_local, force=True
    )
    CheckpointLoader.register_scheme(
        TRUSTED_WEDETECT_PREFIX, _load_trusted_wedetect, force=True
    )


def make_safe_checkpoint_uri(checkpoint: str | os.PathLike[str]) -> str:
    """Return a ``safe://`` URI (tensor-only loading) for a local path."""
    value = os.fspath(checkpoint)
    if value.startswith(SAFE_CHECKPOINT_PREFIX):
        return value
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", value):
        raise ValueError(
            "safe checkpoint loading only accepts local paths; download and "
            f"verify {value!r} before evaluation"
        )
    return SAFE_CHECKPOINT_PREFIX + value


def resolve_eval_checkpoint(
    checkpoint: Optional[str | os.PathLike[str]],
    work_dir: str | Path | None = None,
    *,
    allow_unsafe_checkpoint: bool = False,
) -> str:
    """Resolve an evaluation checkpoint, loading it tensor-only by default.

    Set ``allow_unsafe_checkpoint`` only for full training checkpoints you
    created yourself (they contain Python objects).
    """
    if checkpoint is None:
        if work_dir is None:
            raise ValueError("work_dir is required when checkpoint is omitted")
        resolved = resolve_latest_checkpoint(None, work_dir)
    else:
        resolved = os.fspath(checkpoint)
    if allow_unsafe_checkpoint:
        return resolved
    return make_safe_checkpoint_uri(resolved)


def resolve_latest_checkpoint(
    checkpoint: Optional[str],
    work_dir: str | Path,
) -> str:
    """Return ``checkpoint`` if given, else the newest checkpoint in
    ``work_dir`` (``best_*`` first, then ``epoch_*``)."""
    if checkpoint:
        return checkpoint

    work_path = Path(work_dir)
    for pattern in (
        "best_coco_overall_macro_mAP_epoch_*.pth",
        "best_coco_bbox_mAP_epoch_*.pth",
        "epoch_*.pth",
    ):
        candidates = sorted(
            work_path.glob(pattern),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return str(candidates[0])
    raise FileNotFoundError(
        f"No checkpoint found under {work_path}; pass --checkpoint explicitly."
    )
