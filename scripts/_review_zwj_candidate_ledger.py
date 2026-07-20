"""One-shot operator script: promote the 张无忌 candidate ledger to a
formal reviewed ledger through ``review_chapter_ledger``.

Review basis (2026-07-20 bake, operator mengfu, agent-assisted):
- chapter coverage cross-checked against the source TXT (40 回, sha pinned);
- pre-birth chapters ch-0..ch-6 re-extracted with the v2 subjective-anchor
  prompt and verified NOT_KNOWN with empty scene/fact/event payload;
- experienced chapters spot-checked for subjective anchor (张无忌 viewpoint),
  valid emotional_register values, and epistemic cutoffs.
"""

from __future__ import annotations

import pathlib
import sys

from lifeform_domain_character import (
    read_ledger_json,
    read_text_with_detected_encoding,
    split_source_chapters,
    write_ledger_json,
)
from lifeform_domain_character.extraction import review_chapter_ledger
from lifeform_domain_character.extraction.chapter_llm import ChapterLedgerCandidate

CANDIDATE_PATH = pathlib.Path(
    "artifacts/character-live-through/zhang_wuji.candidate_ledger.json"
)
REVIEWED_PATH = pathlib.Path(
    "artifacts/character-live-through/zhang_wuji.reviewed_ledger.json"
)
REVIEWER = "mengfu"


def main() -> int:
    raw = read_ledger_json(CANDIDATE_PATH)
    candidate = ChapterLedgerCandidate(
        character_id=raw.character_id,
        source_title=raw.source_title,
        source_sha256=raw.source_sha256,
        chapters=raw.chapters,
        failed_chapters=(),
        requires_review=(),
        prompt_version="chapter_live_through.system.v2",
    )
    text, _encoding, source_sha = read_text_with_detected_encoding(
        pathlib.Path("data/novels/倚天屠龙记.TXT")
    )
    if source_sha != raw.source_sha256:
        raise ValueError("candidate ledger source_sha256 does not match novel TXT")
    expected = split_source_chapters(text)
    reviewed = review_chapter_ledger(
        candidate,
        reviewer=REVIEWER,
        expected_chapters=expected,
    )
    write_ledger_json(reviewed, REVIEWED_PATH)
    print(
        f"reviewed ledger written to {REVIEWED_PATH} "
        f"(chapters={len(reviewed.chapters)}, reviewed_by={reviewed.reviewed_by})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
