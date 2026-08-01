#!/usr/bin/env python3
"""Loop-external candidate validator for the companion playbook overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import jsonschema


REPO_ROOT = Path(__file__).resolve().parents[1]
for _source_root in sorted((REPO_ROOT / "packages").glob("*/src")):
    sys.path.insert(0, str(_source_root))

from lifeform_domain_emogpt import (  # noqa: E402
    load_companion_playbook_overlay,
    resolve_companion_package_overlay,
)
from volvence_zero.runtime import WiringLevel  # noqa: E402


class CompanionOverlayValidationError(RuntimeError):
    """Raised when a candidate violates schema or owner constraints."""


def validate_candidate(path: Path) -> dict[str, object]:
    candidate_path = path.expanduser().resolve()
    schema_path = (
        REPO_ROOT
        / "packages"
        / "lifeform-domain-emogpt"
        / "src"
        / "lifeform_domain_emogpt"
        / "schemas"
        / "companion_playbook_overlay.schema.json"
    )
    try:
        candidate_raw = json.loads(candidate_path.read_text(encoding="utf-8"))
        schema_raw = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompanionOverlayValidationError(f"cannot read candidate/schema: {exc}") from exc
    try:
        jsonschema.Draft202012Validator(schema_raw).validate(candidate_raw)
    except jsonschema.ValidationError as exc:
        location = ".".join(str(part) for part in exc.absolute_path) or "<root>"
        raise CompanionOverlayValidationError(
            f"candidate schema violation at {location}: {exc.message}"
        ) from exc
    try:
        overlay = load_companion_playbook_overlay(candidate_path)
        resolution = resolve_companion_package_overlay(
            wiring_level=WiringLevel.SHADOW,
            overlay_path=candidate_path,
        )
    except ValueError as exc:
        raise CompanionOverlayValidationError(f"owner validation failed: {exc}") from exc
    return {
        "schema_version": "companion-playbook-overlay-candidate-validation.v1",
        "status": "PASS",
        "overlay_id": overlay.overlay_id,
        "content_sha256": overlay.content_sha256,
        "candidate_rule_count": len(resolution.candidate_rules),
        "live_rule_count": len(resolution.live_rules),
        "wiring_level": resolution.wiring_level.value,
        "applied": resolution.applied,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate one companion playbook overlay candidate in SHADOW"
    )
    parser.add_argument("candidate_path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = validate_candidate(args.candidate_path)
    except CompanionOverlayValidationError as exc:
        print(f"companion overlay validator: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
