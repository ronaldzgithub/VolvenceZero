"""Unit tests for portable PEFT checkpoint path resolution."""

from __future__ import annotations

import pathlib

from volvence_zero.substrate import (
    peft_checkpoint_root,
    resolve_peft_checkpoint_dir,
)


def test_empty_input_maps_to_empty() -> None:
    assert resolve_peft_checkpoint_dir("") == ""
    assert resolve_peft_checkpoint_dir("   ") == ""


def test_existing_absolute_path_returned_resolved(tmp_path) -> None:
    checkpoint = tmp_path / "einstein" / "abc123"
    checkpoint.mkdir(parents=True)
    assert resolve_peft_checkpoint_dir(str(checkpoint)) == str(
        checkpoint.resolve()
    )


def test_root_env_overrides_default(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VZ_PEFT_CHECKPOINT_ROOT", str(tmp_path))
    assert peft_checkpoint_root() == tmp_path


def test_relative_path_resolved_under_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VZ_PEFT_CHECKPOINT_ROOT", str(tmp_path))
    (tmp_path / "einstein" / "abc123").mkdir(parents=True)
    resolved = resolve_peft_checkpoint_dir("einstein/abc123")
    assert resolved == str((tmp_path / "einstein" / "abc123").resolve())


def test_absolute_path_rerooted_via_anchor(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VZ_PEFT_CHECKPOINT_ROOT", str(tmp_path))
    (tmp_path / "einstein" / "abc123").mkdir(parents=True)
    baked = "/other/machine/.local/peft-checkpoints/einstein/abc123"
    resolved = resolve_peft_checkpoint_dir(baked)
    assert resolved == str((tmp_path / "einstein" / "abc123").resolve())


def test_unresolvable_path_returned_verbatim_for_failloud(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("VZ_PEFT_CHECKPOINT_ROOT", str(tmp_path))
    # No directory exists anywhere; the resolver returns the input so
    # activate_peft_adapter raises a loud FileNotFoundError (never a
    # silent degrade to the hook path).
    missing = "/nowhere/peft-checkpoints/ghost/deadbeef"
    assert resolve_peft_checkpoint_dir(missing) == str(
        pathlib.Path(missing)
    )
