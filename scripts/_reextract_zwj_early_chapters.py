"""One-shot operator script: re-extract 张无忌 pre-birth chapters (ch-0..ch-6)
with the v2 subjective-anchor prompt and merge into the candidate ledger.

The first pass (prompt v1) misattributed 张君宝/张翠山 viewpoint scenes to
张无忌 for chapters before his birth. Only those chapters are re-run so the
already-correct ch-7..ch-39 extractions are kept as-is.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, "examples")

from bake_zhang_wuji_live_through import _ExternalJsonRuntimeAdapter

from lifeform_domain_character import (
    ChapterLiveThroughLedger,
    build_zhang_wuji_profile,
    read_ledger_json,
    read_text_with_detected_encoding,
    split_source_chapters,
    write_ledger_json,
)
from lifeform_domain_character.extraction import extract_chapter_ledger_candidate
from lifeform_service.openai_compat_client import build_client_from_env

REDO = ("ch-0", "ch-1", "ch-2", "ch-3", "ch-4", "ch-5", "ch-6")
LEDGER_PATH = pathlib.Path(
    "artifacts/character-live-through/zhang_wuji.candidate_ledger.json"
)


def main() -> int:
    client = build_client_from_env()
    if client is None:
        raise ValueError("PROTOCOL_LLM_API_KEY not configured")
    text, _encoding, source_sha = read_text_with_detected_encoding(
        pathlib.Path("data/novels/倚天屠龙记.TXT")
    )
    chapters = split_source_chapters(text)
    redo_chapters = tuple(ch for ch in chapters if ch.chapter_id in REDO)
    assert len(redo_chapters) == len(REDO)

    profile = build_zhang_wuji_profile()
    candidate = extract_chapter_ledger_candidate(
        chapters=redo_chapters,
        llm_runtime=_ExternalJsonRuntimeAdapter(client),
        character_id=profile.profile_id,
        character_name=profile.character_name,
        source_title=profile.source_title,
        source_sha256=source_sha,
    )
    if candidate.failed_chapters:
        raise RuntimeError(f"re-extraction failed: {candidate.failed_chapters}")

    existing = read_ledger_json(LEDGER_PATH)
    by_id = {ch.chapter_id: ch for ch in existing.chapters}
    for ch in candidate.chapters:
        old = by_id[ch.chapter_id]
        print(
            f"{ch.chapter_id}: {old.coverage.value} -> {ch.coverage.value} "
            f"(scenes {len(old.scenes)}->{len(ch.scenes)}, "
            f"events {len(old.semantic_events)}->{len(ch.semantic_events)}, "
            f"facts {len(old.known_facts)}->{len(ch.known_facts)})"
        )
        by_id[ch.chapter_id] = ch
    merged = ChapterLiveThroughLedger(
        character_id=existing.character_id,
        source_title=existing.source_title,
        source_sha256=existing.source_sha256,
        chapters=tuple(
            sorted(by_id.values(), key=lambda item: item.chapter_index)
        ),
        reviewed_by=existing.reviewed_by,
    )
    write_ledger_json(merged, LEDGER_PATH)
    print(f"merged ledger written to {LEDGER_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
