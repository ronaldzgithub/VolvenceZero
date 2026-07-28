#!/usr/bin/env python3
"""Independent, read-only behavioral-fidelity evaluation for Zhang Wuji."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from lifeform_domain_character import (
    behavior_fidelity_capture_from_dict,
    behavior_fidelity_report_from_dict,
    build_character_lifeform,
    build_scene_behavior_fidelity_inputs,
    build_zhang_wuji_profile,
    capture_behavior_fidelity_async,
    compare_behavior_fidelity_reports,
    give_birth,
    read_ledger_json,
    review_behavior_fidelity,
    reviewed_behavior_fidelity_assessment_from_dict,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("capture", "review", "compare"),
        required=True,
    )
    parser.add_argument(
        "--reviewed-ledger",
        type=Path,
        default=Path(
            "artifacts/character-live-through/"
            "zhang_wuji.reviewed_ledger.json"
        ),
    )
    parser.add_argument("--chapter-id", default="ch-11")
    parser.add_argument("--arm-id", default="baked")
    parser.add_argument("--template", type=Path, default=None)
    parser.add_argument("--capture", type=Path, default=None)
    parser.add_argument("--assessment", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--reference-output", type=Path, default=None)
    parser.add_argument("--baked-report", type=Path, default=None)
    parser.add_argument("--cold-report", type=Path, default=None)
    parser.add_argument("--comparison-output", type=Path, default=None)
    parser.add_argument(
        "--profile-answer-holdout-passed",
        action="store_true",
        help=(
            "Permit a learned-advantage claim only after a separate "
            "profile-answer holdout audit has passed."
        ),
    )
    return parser


def _chapter_inputs(args: argparse.Namespace):
    ledger = read_ledger_json(args.reviewed_ledger)
    matches = tuple(
        chapter
        for chapter in ledger.chapters
        if chapter.chapter_id == args.chapter_id
    )
    if len(matches) != 1:
        raise ValueError(
            f"chapter id {args.chapter_id!r} matched {len(matches)} chapters"
        )
    chapter = matches[0]
    if len(chapter.scenes) != 1:
        raise ValueError(
            "minimal behavior fidelity command requires exactly one scene, "
            f"got {len(chapter.scenes)}"
        )
    return build_scene_behavior_fidelity_inputs(
        character_id=ledger.character_id,
        scene=chapter.scenes[0],
        reviewed_by=chapter.reviewed_by,
    )


def _capture(args: argparse.Namespace) -> int:
    if args.capture is None:
        raise ValueError("--capture is required for capture stage")
    stimulus, reference = _chapter_inputs(args)
    if args.template is not None:
        before = _file_sha256(args.template)
        bundle = give_birth(args.template)
        profile = None
    else:
        profile = build_zhang_wuji_profile()
        before = _text_sha256(repr(profile))
        bundle = build_character_lifeform(profile)
    capture = asyncio.run(
        capture_behavior_fidelity_async(
            stimulus=stimulus,
            lifeform=bundle.lifeform,
            arm_id=args.arm_id,
            source_state_sha256_before=before,
            source_state_sha256_after=before,
        )
    )
    after = (
        _file_sha256(args.template)
        if args.template is not None
        else _text_sha256(repr(profile))
    )
    capture = replace(
        capture,
        source_state_sha256_after=after,
        source_state_unchanged=before == after,
    )
    _write_json(args.capture, asdict(capture))
    if args.reference_output is not None:
        _write_json(
            args.reference_output,
            {
                "stimulus": asdict(stimulus),
                "reference": asdict(reference),
            },
        )
    print(
        "[behavior-fidelity:capture] "
        f"arm={capture.arm_id} "
        f"candidate_sha256={capture.candidate_response_sha256} "
        f"source_unchanged={capture.source_state_unchanged} "
        f"capture={args.capture}"
    )
    return 0


def _review(args: argparse.Namespace) -> int:
    if args.capture is None or args.assessment is None or args.report is None:
        raise ValueError(
            "--capture, --assessment, and --report are required for review"
        )
    _stimulus, reference = _chapter_inputs(args)
    capture = behavior_fidelity_capture_from_dict(_read_json(args.capture))
    assessment = reviewed_behavior_fidelity_assessment_from_dict(
        _read_json(args.assessment)
    )
    report = review_behavior_fidelity(
        capture=capture,
        reference=reference,
        assessment=assessment,
    )
    _write_json(args.report, asdict(report))
    print(
        "[behavior-fidelity:review] "
        f"arm={report.arm_id} "
        f"score={report.overall_score:.3f} "
        f"passed={report.behavior_fidelity_passed} "
        f"claim={report.claim_status} "
        f"report={args.report}"
    )
    return 0


def _compare(args: argparse.Namespace) -> int:
    if (
        args.baked_report is None
        or args.cold_report is None
        or args.comparison_output is None
    ):
        raise ValueError(
            "--baked-report, --cold-report, and --comparison-output "
            "are required for compare"
        )
    comparison = compare_behavior_fidelity_reports(
        baked=behavior_fidelity_report_from_dict(
            _read_json(args.baked_report)
        ),
        cold=behavior_fidelity_report_from_dict(
            _read_json(args.cold_report)
        ),
        profile_answer_holdout_passed=(
            args.profile_answer_holdout_passed
        ),
    )
    _write_json(args.comparison_output, asdict(comparison))
    print(
        "[behavior-fidelity:compare] "
        f"baked={comparison.baked_score:.3f} "
        f"cold={comparison.cold_score:.3f} "
        f"delta={comparison.baked_minus_cold:.3f} "
        f"advantage={comparison.learned_behavior_advantage} "
        f"report={args.comparison_output}"
    )
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.stage == "capture":
        return _capture(args)
    if args.stage == "review":
        return _review(args)
    return _compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
