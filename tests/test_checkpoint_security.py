"""Tests for safe checkpoint loading and the checkpoint download script."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import torch

from cytospm.utils.checkpoint import (
    SAFE_CHECKPOINT_PREFIX,
    make_safe_checkpoint_uri,
    resolve_eval_checkpoint,
)


class _UnsafeObject:
    """Object rejected by the weights-only unpickler."""


def _save(path: Path, value) -> None:
    torch.save(value, path)


def test_safe_uri_loader_accepts_tensor_release_and_rejects_python_object(tmp_path):
    from mmengine.runner.checkpoint import CheckpointLoader

    safe = tmp_path / "release.pth"
    _save(safe, {"state_dict": {"weight": torch.ones(2)}})
    loaded = CheckpointLoader.load_checkpoint(
        make_safe_checkpoint_uri(safe), map_location="cpu", logger=None
    )
    assert torch.equal(loaded["state_dict"]["weight"], torch.ones(2))

    unsafe = tmp_path / "unsafe.pth"
    _save(unsafe, {"state_dict": {"payload": _UnsafeObject()}})
    with pytest.raises((RuntimeError, pickle_error_types())):
        CheckpointLoader.load_checkpoint(
            make_safe_checkpoint_uri(unsafe), map_location="cpu", logger=None
        )


def pickle_error_types():
    import pickle

    return pickle.UnpicklingError


def test_eval_resolution_is_safe_by_default(tmp_path):
    path = tmp_path / "epoch_1.pth"
    assert resolve_eval_checkpoint(path) == SAFE_CHECKPOINT_PREFIX + str(path)
    assert resolve_eval_checkpoint(path, allow_unsafe_checkpoint=True) == str(path)
    with pytest.raises(ValueError):
        make_safe_checkpoint_uri("https://example.invalid/model.pth")


def test_fetch_script_fails_closed_for_unknown_or_corrupt_files(tmp_path):
    script = Path(__file__).parents[1] / "tools" / "fetch_checkpoints.sh"
    unknown = subprocess.run(
        ["bash", str(script)],
        cwd=script.parents[1],
        env={
            **os.environ,
            "CKPT_FILES": "not-registered.pth",
            "CKPT_DIR": str(tmp_path),
        },
        text=True,
        capture_output=True,
    )
    assert unknown.returncode != 0
    assert "no SHA256 registered" in unknown.stderr + unknown.stdout

    corrupt_dir = tmp_path / "corrupt"
    corrupt_dir.mkdir()
    (corrupt_dir / "wedetect_tiny.pth").write_bytes(b"not-a-checkpoint")
    corrupt = subprocess.run(
        ["bash", str(script)],
        cwd=script.parents[1],
        env={
            **os.environ,
            "CKPT_FILES": "wedetect_tiny.pth",
            "CKPT_DIR": str(corrupt_dir),
        },
        text=True,
        capture_output=True,
    )
    assert corrupt.returncode != 0
    assert "SHA256 mismatch" in corrupt.stderr + corrupt.stdout
