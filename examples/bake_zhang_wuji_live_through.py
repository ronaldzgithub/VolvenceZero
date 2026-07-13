"""Operator CLI for 张无忌 chapter live-through bake.

Stages:
  extract       build scaffold; with LLM responses, write a candidate ledger
                (a real --reviewer turns it into a formal reviewed ledger)
  review-check  validate a reviewed ledger against the source TXT
  replay       run chapter live-through with disk-backed progress/resume
  evaluate     run lightweight static not-known coverage checks
  save         replay (or resume) and save a LifeformTemplate v2

Kernel state (memory / semantic / vitals) persists under --state-dir so a
resumed run continues from real lived state, not an empty brain.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import sys
from typing import Any

from lifeform_domain_character import (
    ChapterCoverageKind,
    ChapterLiveThroughDriver,
    ChapterLiveThroughLedger,
    build_character_lifeform,
    build_review_scaffold,
    build_zhang_wuji_profile,
    read_ledger_json,
    read_text_with_detected_encoding,
    save_lifeform_template,
    split_source_chapters,
    write_ledger_json,
)
from lifeform_domain_character.extraction import (
    extract_chapter_ledger_candidate,
    review_chapter_ledger,
)
from volvence_zero.memory import (
    FileSystemPersistenceBackend,
    PersistenceBackend,
    build_default_memory_store,
)
from volvence_zero.owner_hydration import OwnerPersistenceSnapshot
from lifeform_service.openai_compat_client import (
    build_client_from_env,
    describe_active_provider,
)

_PLACEHOLDER_REVIEWER = "operator-review-required"


class _FixtureChapterRuntime:
    def __init__(self, response_dir: pathlib.Path) -> None:
        self._response_dir = response_dir

    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        del max_new_tokens, temperature
        match = re.search(r"Chapter id: (ch-\d+)", prompt)
        if match is None:
            raise ValueError("fixture runtime could not find Chapter id in prompt")
        path = self._response_dir / f"{match.group(1)}.json"
        return path.read_text(encoding="utf-8")


class _ExternalJsonRuntimeAdapter:
    def __init__(self, client: Any) -> None:
        self._client = client

    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        del max_new_tokens, temperature
        payload = self._client.complete_json(
            system_prompt=(
                "You are an extraction worker. Return only one JSON object "
                "that conforms to the schema embedded in the user prompt."
            ),
            user_prompt=prompt,
        )
        return json.dumps(payload, ensure_ascii=False)


def _load_owner_snapshot(
    backend: PersistenceBackend,
    owner_name: str,
) -> OwnerPersistenceSnapshot | None:
    loaded = backend.load_checkpoint(key=f"owner_hydration/{owner_name}")
    if loaded is None:
        return None
    data, _version = loaded
    payload = json.loads(data.decode("utf-8"))
    return OwnerPersistenceSnapshot(
        owner_name=str(payload["owner_name"]),
        schema_version=int(payload["schema_version"]),
        payload=dict(payload["payload"]),
        description=str(payload.get("description", "")),
    )


def _source_chapters(novel_path: pathlib.Path):
    text, _encoding, _sha = read_text_with_detected_encoding(novel_path)
    return split_source_chapters(text)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("extract", "review-check", "replay", "evaluate", "save", "all"),
        default="extract",
    )
    parser.add_argument("--novel", type=pathlib.Path, default=pathlib.Path("data/novels/倚天屠龙记.TXT"))
    parser.add_argument(
        "--scaffold-output",
        type=pathlib.Path,
        default=pathlib.Path("artifacts/character-live-through/zhang_wuji.review_scaffold.json"),
    )
    parser.add_argument("--reviewed-ledger", type=pathlib.Path, default=None)
    parser.add_argument(
        "--candidate-response-dir",
        type=pathlib.Path,
        default=None,
        help="Directory containing per-chapter LLM JSON responses named ch-<n>.json.",
    )
    parser.add_argument(
        "--external-llm",
        choices=("auto", "required", "off"),
        default="auto",
        help=(
            "Use PROTOCOL_LLM_* OpenAI-compatible external LLM for extraction "
            "when no --candidate-response-dir is supplied. 'required' fails "
            "loudly if the client is not configured."
        ),
    )
    parser.add_argument(
        "--candidate-output",
        type=pathlib.Path,
        default=pathlib.Path("artifacts/character-live-through/zhang_wuji.candidate_ledger.json"),
    )
    parser.add_argument(
        "--progress",
        type=pathlib.Path,
        default=pathlib.Path("artifacts/character-live-through/zhang_wuji.replay_progress.jsonl"),
    )
    parser.add_argument(
        "--state-dir",
        type=pathlib.Path,
        default=pathlib.Path("artifacts/character-live-through/zhang_wuji.kernel_state"),
        help="Disk-backed kernel state (memory/semantic/vitals) so resume continues real lived state.",
    )
    parser.add_argument(
        "--template-output",
        type=pathlib.Path,
        default=pathlib.Path("artifacts/lifeform-templates/zhang_wuji/zhang-wuji-live-through.json"),
    )
    parser.add_argument("--reviewer", default="operator-review-required")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def _stage_extract(args: argparse.Namespace) -> int:
    profile = build_zhang_wuji_profile()
    scaffold = build_review_scaffold(
        novel_path=args.novel,
        character_id=profile.profile_id,
        source_title=profile.source_title,
        reviewed_by="operator-review-required",
    )
    write_ledger_json(scaffold, args.scaffold_output)
    print(f"[scaffold] wrote {args.scaffold_output}")
    if args.candidate_response_dir is not None:
        llm_runtime = _FixtureChapterRuntime(args.candidate_response_dir)
        print(f"[extract] using fixture responses from {args.candidate_response_dir}")
    elif args.external_llm == "off":
        print("[extract] external LLM disabled; scaffold only")
        return 0
    else:
        active_provider = describe_active_provider()
        client = build_client_from_env()
        if client is None:
            message = (
                "[extract] no PROTOCOL_LLM_API_KEY; scaffold only "
                f"(provider={active_provider['provider']} "
                f"model={active_provider['model'] or '<unset>'})"
            )
            if args.external_llm == "required":
                raise ValueError(message)
            print(message)
            return 0
        llm_runtime = _ExternalJsonRuntimeAdapter(client)
        print(
            "[extract] using external LLM "
            f"provider={active_provider['provider']} "
            f"model={active_provider['model']} "
            f"base_url={active_provider['base_url']}"
        )
    chapters = _source_chapters(args.novel)
    candidate = extract_chapter_ledger_candidate(
        chapters=chapters,
        llm_runtime=llm_runtime,
        character_id=profile.profile_id,
        character_name=profile.character_name,
        source_title=profile.source_title,
        source_sha256=scaffold.source_sha256,
    )
    if candidate.failed_chapters:
        print(f"[extract] FAILED chapters (re-run or review manually): {candidate.failed_chapters}")
    if candidate.requires_review:
        print(f"[extract] chapters flagged for extra review: {candidate.requires_review}")
    if args.reviewer == _PLACEHOLDER_REVIEWER:
        # No accountable reviewer yet: persist an UNREVIEWED candidate.
        # review-check rejects the placeholder, so this file cannot slip
        # into replay/save without a real review pass.
        unreviewed = ChapterLiveThroughLedger(
            character_id=candidate.character_id,
            source_title=candidate.source_title,
            source_sha256=candidate.source_sha256,
            chapters=candidate.chapters,
            reviewed_by=_PLACEHOLDER_REVIEWER,
        )
        write_ledger_json(unreviewed, args.candidate_output)
        print(f"[candidate] wrote UNREVIEWED candidate ledger {args.candidate_output}")
        print("[candidate] pass --reviewer <your-id> to produce a formal reviewed ledger")
        return 0
    reviewed = review_chapter_ledger(
        candidate,
        reviewer=args.reviewer,
        expected_chapters=chapters,
    )
    write_ledger_json(reviewed, args.candidate_output)
    print(f"[candidate] wrote reviewed ledger {args.candidate_output}")
    return 0


def _load_reviewed_ledger(path: pathlib.Path | None):
    if path is None:
        raise ValueError("--reviewed-ledger is required for this stage")
    return read_ledger_json(path)


def _stage_review_check(args: argparse.Namespace) -> int:
    ledger = _load_reviewed_ledger(args.reviewed_ledger)
    if not ledger.reviewed_by.strip() or ledger.reviewed_by == _PLACEHOLDER_REVIEWER:
        raise ValueError("reviewed ledger must have a real reviewed_by value")
    if tuple(ch.chapter_index for ch in ledger.chapters) != tuple(
        sorted(ch.chapter_index for ch in ledger.chapters)
    ):
        raise ValueError("reviewed ledger chapters are not ordered")
    # Cross-check against the source TXT: same file hash, full chapter
    # coverage, and each chapter's evidence pinned to its chapter hash.
    _text, _encoding, source_sha256 = read_text_with_detected_encoding(args.novel)
    if ledger.source_sha256 != source_sha256:
        raise ValueError(
            "reviewed ledger source_sha256 does not match --novel file: "
            f"ledger={ledger.source_sha256} file={source_sha256}"
        )
    source = _source_chapters(args.novel)
    expected_ids = tuple(ch.chapter_id for ch in source)
    actual_ids = tuple(ch.chapter_id for ch in ledger.chapters)
    if actual_ids != expected_ids:
        raise ValueError(
            "reviewed ledger chapter coverage mismatch vs source TXT: "
            f"expected {len(expected_ids)} chapters {expected_ids[:3]}..., "
            f"got {len(actual_ids)} chapters {actual_ids[:3]}..."
        )
    source_by_id = {ch.chapter_id: ch for ch in source}
    for chapter in ledger.chapters:
        expected_sha = source_by_id[chapter.chapter_id].text_sha256
        if f"sha256:{expected_sha}" not in chapter.evidence_locator:
            raise ValueError(
                f"chapter {chapter.chapter_id} evidence_locator is not pinned "
                f"to the source chapter hash sha256:{expected_sha}"
            )
    print(
        "[review-check] ok "
        f"chapters={len(ledger.chapters)} reviewed_by={ledger.reviewed_by} "
        f"source_sha256={source_sha256[:12]}..."
    )
    return 0


def _run_replay(args: argparse.Namespace):
    profile = build_zhang_wuji_profile()
    ledger = _load_reviewed_ledger(args.reviewed_ledger)
    if ledger.reviewed_by == _PLACEHOLDER_REVIEWER:
        raise ValueError(
            "refusing to replay an UNREVIEWED candidate ledger; run the "
            "review pass first (extract --reviewer <your-id>)"
        )
    progress_has_records = args.progress.exists() and args.progress.stat().st_size > 0
    if not args.resume:
        # Fresh run: stale kernel state must not hydrate into the new
        # life (the driver separately resets the progress file).
        if args.state_dir.exists():
            shutil.rmtree(args.state_dir)
    backend = FileSystemPersistenceBackend(base_dir=str(args.state_dir))
    memory_store = build_default_memory_store(persistence_backend=backend)
    if args.resume:
        loaded = memory_store.load_from_backend()
        if progress_has_records and not loaded:
            raise ValueError(
                "resume requested and progress records exist, but no kernel "
                f"memory state was found under {args.state_dir}; the resumed "
                "template would be missing lived chapters. Delete "
                f"{args.progress} to restart from scratch instead."
            )
    bundle = build_character_lifeform(profile, memory_store=memory_store)
    report = ChapterLiveThroughDriver().run_ledger(
        ledger=ledger,
        lifeform=bundle.lifeform,
        progress_path=args.progress,
        resume=args.resume,
    )
    if not memory_store.save_to_backend():
        raise RuntimeError("failed to persist kernel memory state after replay")
    return ledger, memory_store, backend, report


def _stage_replay(args: argparse.Namespace) -> int:
    _ledger, _memory_store, _backend, report = _run_replay(args)
    print(
        "[replay] ok "
        f"chapters={report.chapters_processed} pe={report.total_pe_signal:.4f}"
    )
    return 0


def _stage_evaluate(args: argparse.Namespace) -> int:
    ledger = _load_reviewed_ledger(args.reviewed_ledger)
    not_known = [ch for ch in ledger.chapters if ch.coverage is ChapterCoverageKind.NOT_KNOWN]
    leaks = [ch.chapter_id for ch in not_known if ch.known_facts or ch.semantic_events or ch.scenes]
    if leaks:
        raise ValueError(f"not-known chapters carry forbidden payload: {leaks!r}")
    print(f"[evaluate] ok not_known={len(not_known)}")
    return 0


def _stage_save(args: argparse.Namespace) -> int:
    ledger, memory_store, backend, report = _run_replay(args)
    owner_snapshots = tuple(
        snap
        for snap in (_load_owner_snapshot(backend, "semantic_state"),)
        if snap is not None
    )
    output_path = args.template_output
    result = save_lifeform_template(
        profile=build_zhang_wuji_profile(),
        template_id=output_path.stem,
        output_dir=output_path.parent,
        memory_store=memory_store,
        vitals_drive_levels=report.final_vitals,
        source_arc_id="chapter-live-through-ledger",
        replay_provenance=(
            f"chapter-live-through:{ledger.source_sha256}:"
            f"chapters={report.chapters_processed}:"
            f"experienced={report.experienced_chapters}:"
            f"learned={report.learned_chapters}:"
            f"not_known={report.not_known_chapters}"
        ),
        overwrite_existing=args.force,
        preserve_memory=True,
        owner_hydration_snapshots=owner_snapshots,
    )
    print(f"[template] wrote {result.template_path}")
    print("Use in browser chat:")
    print("  VERTICAL=zhang_wuji")
    print(f"  ZHANG_WUJI_TEMPLATE_PATH={result.template_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    # "all" omits the standalone replay stage: save runs the replay
    # itself (with progress/resume), so including both would live the
    # same life twice in one invocation.
    stages = (
        ("extract", "review-check", "evaluate", "save")
        if args.stage == "all"
        else (args.stage,)
    )
    for stage in stages:
        if stage == "extract":
            _stage_extract(args)
        elif stage == "review-check":
            _stage_review_check(args)
        elif stage == "replay":
            _stage_replay(args)
        elif stage == "evaluate":
            _stage_evaluate(args)
        elif stage == "save":
            _stage_save(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
