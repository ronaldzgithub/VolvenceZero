# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from companion_bench.msc_corpus import load_msc_manifest, load_msc_split


def _write_fixture(root: Path) -> Path:
    dialogue = root / "msc_dialogue" / "session_4"
    dialogue.mkdir(parents=True)
    row = {
        "metadata": {"initial_data_id": "train:fixture", "session_id": 3},
        "init_personas": [["I like tea."], ["I like books."]],
        "previous_dialogs": [
            {
                "dialog": [{"text": f"session {index} hello"}, {"text": "hello back"}],
                "time_num": index,
                "time_unit": "days",
                "time_back": f"{index} days ago",
            }
            for index in range(1, 4)
        ],
        "dialog": [{"text": "latest hello"}, {"text": "latest reply"}],
    }
    path = dialogue / "train.txt"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return root


def test_packaged_manifest_freezes_official_split_hashes() -> None:
    manifest = load_msc_manifest()
    assert manifest["archive_sha256"] == (
        "e640e37cf4317cd09fc02a4cd57ef130a185f23635f4003b0cee341ffcb45e60"
    )
    assert manifest["splits"]["heldout"]["conversation_count"] == 501
    assert manifest["license_policy"].startswith("noncommercial-research-only")


def test_loader_normalizes_multisession_dyad_without_mutable_refs(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path)
    dyads, audit = load_msc_split(root, split="train", strict=False)
    assert len(dyads) == 1
    dyad = dyads[0]
    assert dyad.dyad_id == "train:fixture"
    assert len(dyad.sessions) == 4
    assert dyad.utterance_count == 8
    assert dyad.sessions[0].utterances[0].speaker == "speaker_1"
    assert audit.minimum_session_count == audit.maximum_session_count == 4
    assert audit.dropped_empty_utterance_count == 0
    assert not audit.verified


def test_loader_rejects_unknown_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="split must be one of"):
        load_msc_split(tmp_path, split="dev")


def test_loader_strict_mode_rejects_unfrozen_bytes(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path)
    with pytest.raises(ValueError, match="file SHA-256 mismatch"):
        load_msc_split(root, split="train", strict=True)
