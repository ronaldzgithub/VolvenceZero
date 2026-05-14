#!/usr/bin/env python3
# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Trusted Runner skeleton (debt #57 SHADOW).

Walks the credential lifecycle described in
``docs/external/companion-bench-trusted-runner-protocol.md`` end-to-end
in ``--dry-run`` mode without touching real secrets, real APIs, or
the held-out scenario set. The point is to lock the protocol surface
so the ACTIVE wire-up (real public-key crypto + secrets vault +
transcript cleanup cron + billing) drops in without re-debating the
contract.

Lifecycle covered (--dry-run):

1. Validate ``submission.encrypted.json`` envelope schema (debt #57 §3.1)
2. Print VZ public-key fingerprint placeholder (debt #57 §3.1)
3. Simulate credential ingest → ``secrets_vault_log.jsonl`` append-only
4. Simulate scoring run → produce ``verdict.json`` shape (verdict +
   per-axis scores; transcript intentionally absent)
5. Simulate transcript cleanup → write ``transcript_deletion_log.jsonl``
6. Simulate credential purge (debt #57 §3.2 — 30-day cadence)

Real implementation (debt #57 ACTIVE) will replace each step with the
corresponding real action; the I/O contract here stays.

Usage::

    python scripts/companion_bench/trusted_runner_skeleton.py \\
        --dry-run \\
        --encrypted-submission path/to/submission.encrypted.json \\
        --output-dir artifacts/trusted_runner/<submission_id>/
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import pathlib
import sys


_REQUIRED_ENVELOPE_FIELDS = (
    "submission_id",
    "system_name",
    "model_identifier",
    "base_url",
    "api_key_ciphertext",
    "system_prompt",
    "generation_config",
    "leaderboard_category",
)

# Public key fingerprint placeholder. Real fingerprint lands when
# debt #57 ACTIVE generates the actual VZ public key (committed to
# docs/external/companion-bench-trusted-runner-pubkey.asc).
_PLACEHOLDER_PUBKEY_FINGERPRINT = "placeholder/sha256:0000000000000000"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trusted_runner_skeleton")
    p.add_argument(
        "--encrypted-submission",
        type=pathlib.Path,
        required=True,
        help="Path to submission.encrypted.json (envelope; ciphertext stays sealed).",
    )
    p.add_argument(
        "--output-dir",
        type=pathlib.Path,
        required=True,
        help="Per-submission artifacts root.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Required in SHADOW. Real ACTIVE wire-up will accept "
            "--no-dry-run once VZ public key + secrets vault + cleanup "
            "cron are wired."
        ),
    )
    return p


def _validate_envelope(payload: dict) -> None:
    """Enforce the §3.1 schema. Fail-loud on missing fields."""
    missing = [k for k in _REQUIRED_ENVELOPE_FIELDS if k not in payload]
    if missing:
        raise ValueError(
            f"submission envelope missing required fields {sorted(missing)}; "
            f"see docs/external/companion-bench-trusted-runner-protocol.md §3.1"
        )
    if not isinstance(payload["api_key_ciphertext"], str) or not payload[
        "api_key_ciphertext"
    ]:
        raise ValueError("api_key_ciphertext must be a non-empty base64 string")
    cat = payload["leaderboard_category"]
    if cat not in {"open-weight", "closed-api", "bespoke"}:
        raise ValueError(
            f"leaderboard_category must be one of "
            f"['open-weight', 'closed-api', 'bespoke']; got {cat!r}"
        )


def _stage_credential_ingest(
    *,
    output_dir: pathlib.Path,
    payload: dict,
    pubkey_fingerprint: str,
) -> pathlib.Path:
    """Append an audit row to ``secrets_vault_log.jsonl`` (placeholder)."""
    log_path = output_dir / "secrets_vault_log.jsonl"
    entry = {
        "timestamp_iso": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "submission_id": payload["submission_id"],
        "stage": "ingest",
        "pubkey_fingerprint": pubkey_fingerprint,
        "ciphertext_sha256": hashlib.sha256(
            payload["api_key_ciphertext"].encode("utf-8")
        ).hexdigest(),
        "shadow_dry_run": True,
    }
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return log_path


def _emit_verdict_skeleton(
    *,
    output_dir: pathlib.Path,
    payload: dict,
) -> pathlib.Path:
    """Per protocol §2.2: trusted runner returns verdict + per-axis only."""
    verdict_path = output_dir / "verdict.json"
    skeleton = {
        "scaffold_status": "SHADOW",
        "submission_id": payload["submission_id"],
        "model_identifier": payload["model_identifier"],
        "leaderboard_category": payload["leaderboard_category"],
        "axis_means": {axis: None for axis in ("A1", "A2", "A3", "A4", "A5", "A6")},
        "transcript_returned": False,
        "transcript_deletion_recorded": True,
        "verdict": "skeleton_dry_run",
        "notes": (
            "SHADOW dry-run skeleton; no real scoring performed. "
            "ACTIVE will populate axis_means + verdict, while keeping "
            "transcript_returned=False per protocol §2.2."
        ),
    }
    verdict_path.write_text(
        json.dumps(skeleton, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return verdict_path


def _stage_transcript_cleanup(
    *,
    output_dir: pathlib.Path,
    payload: dict,
) -> pathlib.Path:
    """Append a transcript-deletion ledger row even in dry-run.

    Real ACTIVE deletes ``artifacts/companion_bench_runs/<sid>/`` files;
    skeleton just records intent so the ledger schema can be locked.
    """
    log_path = output_dir / "transcript_deletion_log.jsonl"
    entry = {
        "timestamp_iso": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "submission_id": payload["submission_id"],
        "stage": "transcript_cleanup",
        "deleted_file_count": 0,
        "deleted_file_sha256_set": [],
        "shadow_dry_run": True,
    }
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return log_path


def _stage_credential_purge(
    *,
    output_dir: pathlib.Path,
    payload: dict,
) -> pathlib.Path:
    """Per §3.2: credentials live ≤ 30 days from ingest. Skeleton records intent."""
    log_path = output_dir / "secrets_vault_log.jsonl"
    purge_at = _dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(days=30)
    entry = {
        "timestamp_iso": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "submission_id": payload["submission_id"],
        "stage": "purge_scheduled",
        "purge_at_iso": purge_at.isoformat(),
        "shadow_dry_run": True,
    }
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return log_path


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.dry_run:
        sys.stderr.write(
            "trusted_runner_skeleton: SHADOW only supports --dry-run; "
            "ACTIVE wire-up lands with debt #57 ACTIVE. "
            "Re-run with --dry-run.\n"
        )
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.encrypted_submission.exists():
        sys.stderr.write(
            f"encrypted submission not found: {args.encrypted_submission}\n"
        )
        return 2
    payload = json.loads(args.encrypted_submission.read_text(encoding="utf-8"))
    _validate_envelope(payload)

    print(f"VZ public-key fingerprint: {_PLACEHOLDER_PUBKEY_FINGERPRINT}")

    ingest_log = _stage_credential_ingest(
        output_dir=args.output_dir,
        payload=payload,
        pubkey_fingerprint=_PLACEHOLDER_PUBKEY_FINGERPRINT,
    )
    print(f"credential ingest logged → {ingest_log}")

    verdict = _emit_verdict_skeleton(
        output_dir=args.output_dir, payload=payload
    )
    print(f"verdict skeleton → {verdict}")

    cleanup = _stage_transcript_cleanup(
        output_dir=args.output_dir, payload=payload
    )
    print(f"transcript cleanup logged → {cleanup}")

    purge = _stage_credential_purge(
        output_dir=args.output_dir, payload=payload
    )
    print(f"credential purge scheduled → {purge}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
