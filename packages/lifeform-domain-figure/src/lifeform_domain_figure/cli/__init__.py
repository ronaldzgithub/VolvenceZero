"""Operator-facing CLI for the figure vertical bake / gate / rollback pipeline.

Why this exists:

The F1-F6 wheel ships pure Python functions for each step (compile a
profile + corpus into a :class:`FigureArtifactBundle`, contrastive
steering bake, persona LoRA bake, OFFLINE gate apply, pool register).
Until this module landed, every operator action required hand-written
Python boilerplate — there was no auditable CLI surface, no on-disk
bundle persistence, no rollback story. known-debts #23 closes that
gap.

Subcommands:

* ``bake-bundle``    — compile a reviewed profile + corpus into a
                       fresh bundle and persist it.
* ``bake-steering``  — bake a contrastive steering set on top of an
                       existing bundle, route through the OFFLINE
                       :class:`ModificationGate`, persist the new
                       bundle.
* ``bake-lora``      — bake a synthetic persona LoRA artifact on top
                       of an existing bundle, route through the
                       OFFLINE gate, register in the
                       :class:`PersonaLoRAPool`, persist the new
                       bundle.
* ``rollback``       — restore a previous bundle id as the active
                       persona pool record. Append-only: writes a
                       new audit row pointing at the prior record.
* ``list``           — enumerate persisted bundles for one or all
                       figures.

Exit code semantics (consumed by ``scripts/figure_demo_einstein.sh``):

* ``0`` — action succeeded.
* ``1`` — CLI argument / parsing error.
* ``2`` — :class:`GateDecision.BLOCK` from the OFFLINE gate. The
          audit row is still written; ``block_reasons`` carry the
          structured rejection list.
* ``3`` — I/O / schema error (missing bundle, corrupt manifest,
          integrity hash mismatch, etc.).

R8 / R10 / R15 invariants:

* The CLI never reaches into figure-wheel internal modules: every
  bake / apply / register call goes through the public surface
  re-exported from :mod:`lifeform_domain_figure.__init__`. The
  static contract test
  ``tests/contracts/test_figure_cli_uses_only_public_surface.py``
  enforces this with an AST scan.
* No subcommand bypasses
  :func:`apply_steering_through_gate` or
  :func:`apply_persona_lora_through_gate`. The ``--evaluation-snapshot``
  flag is mandatory on every gate-driven subcommand and accepts
  either a JSON path or the literal ``default-clean`` (developer
  shortcut, intentionally explicit so it never silently engages).
* Every CLI invocation that mutates persisted state writes one
  :class:`FigureBakeAuditRecord` capturing the gate decision, the
  prior bundle / record ids, and the rollback evidence the operator
  supplied.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence

from lifeform_domain_figure.cli import _commands


# Family memorial figure ids look like ``family_<cuid>`` — cuids are
# 24-25 lowercase alphanumeric. We allow anything reasonable here and
# defer the strict format check to the bake-worker / control plane,
# but reject obviously-bogus inputs (whitespace, path separators) so
# argparse fails loudly at the boundary.
_BUILTIN_FIGURES = frozenset(("einstein", "lu_xun"))
_FAMILY_ID_RE = re.compile(r"^family_[A-Za-z0-9_-]{4,128}$")
# Generic persona slugs (Myriad historical figures, digital-employee
# personas, ...) become bundle directory names, so the validator only
# admits path-safe identifiers; the schema check happens at profile
# load time (a --profile-json file is mandatory for these slugs).
_GENERIC_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def _validate_figure_arg(value: str) -> str:
    """argparse type=...: accept built-ins + dynamic figure ids.

    Static ``choices=`` cannot express the dynamic family-memorial /
    generic-persona patterns, so we hand-roll a validator that mirrors
    the original constraint set plus the dynamic paths. Generic slugs
    pass the shape check here and are gated on ``--profile-json`` /
    ``--profile-file`` presence at command time.
    """

    if value in _BUILTIN_FIGURES:
        return value
    if _FAMILY_ID_RE.match(value):
        return value
    if _GENERIC_SLUG_RE.match(value):
        return value
    raise argparse.ArgumentTypeError(
        f"--figure {value!r} is not recognised; expected one of "
        f"{sorted(_BUILTIN_FIGURES)}, a string matching family_<id>, "
        f"or a path-safe persona slug (with --profile-json)."
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argparse parser.

    Exposed so tests can introspect / drive subcommands programmatically
    without spawning a subprocess.
    """

    parser = argparse.ArgumentParser(
        prog="figure-bake",
        description=(
            "Operator CLI for the figure-vertical bake / gate / "
            "rollback pipeline (known-debts #23)."
        ),
    )
    parser.add_argument(
        "--bundle-root",
        default="data/figure_bundles",
        help=(
            "Directory root for persisted FigureArtifactBundle "
            "directories. Default: data/figure_bundles."
        ),
    )
    parser.add_argument(
        "--audit-root",
        default="data/figure_audit",
        help=(
            "Directory root for the bake / gate audit log. "
            "Default: data/figure_audit."
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    bake_bundle = subparsers.add_parser(
        "bake-bundle",
        help="Compile a reviewed profile + corpus into a fresh bundle.",
    )
    bake_bundle.add_argument(
        "--figure",
        required=True,
        type=_validate_figure_arg,
        help=(
            "Figure id. Built-in: 'einstein' or 'lu_xun' (reviewer-"
            "curated profile + synthetic corpus). Dynamic: ids "
            "starting with 'family_' / 'myriad_' load their dedicated "
            "JSON schemas from --profile-file; ANY other path-safe "
            "slug (e.g. 'libai') loads the generic persona schema "
            "from --profile-json (see "
            "lifeform_domain_figure.profiles.generic)."
        ),
    )
    bake_bundle.add_argument(
        "--profile-file",
        "--profile-json",
        dest="profile_file",
        default=None,
        help=(
            "Path to a JSON profile descriptor. Required for every "
            "non-built-in figure. family_* / myriad_* ids use their "
            "dedicated schemas (lifeform_domain_figure.profiles."
            "family / .myriad); any other slug uses the generic "
            "persona schema (lifeform_domain_figure.profiles.generic, "
            "compatible with Myriad seed/figures/<slug>/profile.json)."
        ),
    )
    bake_bundle.add_argument(
        "--corpus-mode",
        default="synthetic",
        choices=("synthetic", "curated"),
        help=(
            "Corpus source. 'synthetic' uses the wheel's reviewer-"
            "paraphrased placeholder corpus (default; SHADOW-safe). "
            "'curated' (Wave J closure) walks an L1 cleaning store "
            "+ a curator-staged metadata JSONL to assemble a real "
            "Figure*Source tuple — requires --cleaning-root and "
            "--curated-metadata-file; optionally --verification-root "
            "+ --require-verification-pass when an L2 ledger exists."
        ),
    )
    bake_bundle.add_argument(
        "--cleaning-root",
        default=None,
        help=(
            "Cleaning store root (the same directory L0 crawler "
            "writes raw bytes into and figure_clean writes cleaned "
            "text into). Required when --corpus-mode=curated."
        ),
    )
    bake_bundle.add_argument(
        "--curated-metadata-file",
        default=None,
        help=(
            "JSONL of CuratedSourceMetadata records keyed by "
            "raw_sha256. Required when --corpus-mode=curated."
        ),
    )
    bake_bundle.add_argument(
        "--verification-root",
        default=None,
        help=(
            "Optional verification ledger root (typically the same "
            "directory as --cleaning-root; figure_verify run-batch "
            "writes <root>/verification/<sha>/checks.jsonl). When "
            "supplied AND --require-verification-pass is set, the "
            "bundle gate refuses sources without all-PASS verdicts."
        ),
    )
    bake_bundle.add_argument(
        "--require-verification-pass",
        action="store_true",
        help=(
            "Run build_figure_artifact_bundle with "
            "require_verification_pass=True; requires "
            "--verification-root. The OFFLINE gate refuses bundle "
            "build if any source has any non-PASS axis."
        ),
    )
    bake_bundle.add_argument(
        "--time-window-id",
        default=None,
        help=(
            "Optional TimeWindowedView id to apply to the profile "
            "before compile."
        ),
    )

    bake_steering = subparsers.add_parser(
        "bake-steering",
        help=(
            "Bake a contrastive steering set on an existing bundle "
            "and route through the OFFLINE gate."
        ),
    )
    bake_steering.add_argument(
        "--figure",
        required=True,
        choices=("einstein",),
        help=(
            "Reviewed profile identifier. Steering currently only "
            "ships an Einstein contrast set; lu_xun support is "
            "tracked under known-debts #27. **family_<id>** dynamic "
            "profiles (used by the VolvenceDeploy family-memorial "
            "product) are NOT supported here — family memorials "
            "ship with L1 / L3 / L4 only; L2 contrastive steering "
            "requires reviewer-curated contrast sets the family "
            "does not have."
        ),
    )
    bake_steering.add_argument(
        "--bundle",
        required=True,
        help="Source bundle id to attach steering to.",
    )
    bake_steering.add_argument(
        "--evaluation-snapshot",
        required=True,
        help=(
            "Path to a JSON-serialised EvaluationSnapshot, or the "
            "literal 'default-clean' for the developer-mode clean "
            "snapshot (do NOT use 'default-clean' for production "
            "promotions)."
        ),
    )
    bake_steering.add_argument(
        "--rollback-evidence",
        required=True,
        help=(
            "Non-empty operator-supplied evidence string the gate "
            "logs as the rollback target identifier. Required by "
            "the OFFLINE gate (R10)."
        ),
    )
    bake_steering.add_argument(
        "--validation-delta",
        type=float,
        default=0.05,
        help="Override the gate proposal validation_delta (default 0.05).",
    )
    bake_steering.add_argument(
        "--capacity-cost",
        type=float,
        default=0.20,
        help="Override the gate proposal capacity_cost (default 0.20).",
    )
    bake_steering.add_argument(
        "--use-real-residual",
        action="store_true",
        help=(
            "If set, derive contrastive residuals from a live "
            "OpenWeightResidualRuntime hidden-state capture (debt #21 "
            "closure). Requires transformers + torch; otherwise the "
            "bake falls back to the hashing-embedding coordinate "
            "system (default behaviour)."
        ),
    )
    bake_steering.add_argument(
        "--real-residual-model-id",
        default="sshleifer/tiny-gpt2",
        help=(
            "HuggingFace model id used when --use-real-residual is "
            "set. Default: sshleifer/tiny-gpt2 (CPU smoke-friendly)."
        ),
    )
    bake_steering.add_argument(
        "--real-residual-layer-index",
        type=int,
        default=0,
        help=(
            "Layer index at which to capture the contrastive "
            "residual when --use-real-residual is set. Default: 0."
        ),
    )

    bake_lora = subparsers.add_parser(
        "bake-lora",
        help=(
            "Bake a persona LoRA artifact on an existing bundle and "
            "route through the OFFLINE gate."
        ),
    )
    bake_lora.add_argument(
        "--figure",
        required=True,
        choices=("einstein",),
        help=(
            "Reviewed profile identifier. lu_xun corpus is tracked "
            "under known-debts #27. **family_<id>** dynamic profiles "
            "(used by the VolvenceDeploy family-memorial product) "
            "are NOT supported here — family memorials ship with "
            "L1 / L3 / L4 only; persona LoRA requires a curated "
            "training plan + GPU substrate the family pipeline "
            "does not provision."
        ),
    )
    bake_lora.add_argument(
        "--bundle",
        required=True,
        help="Source bundle id to attach the LoRA artifact to.",
    )
    bake_lora.add_argument(
        "--corpus-mode",
        default="synthetic",
        choices=("synthetic", "curated"),
        help=(
            "LoRA training corpus source. 'synthetic' (default; "
            "backward-compatible) re-derives envelopes from the "
            "wheel's reviewer-paraphrased synthetic corpus. "
            "'curated' (Wave N closure) reads the same L1 cleaning "
            "store + curator metadata that produced a curated bundle "
            "so the LoRA trains on real corpus matching the bundle's "
            "domain_package; requires --cleaning-root and "
            "--curated-metadata-file."
        ),
    )
    bake_lora.add_argument(
        "--cleaning-root",
        default=None,
        help=(
            "L1 cleaning store root (only consumed when "
            "--corpus-mode=curated)."
        ),
    )
    bake_lora.add_argument(
        "--curated-metadata-file",
        default=None,
        help=(
            "JSONL of CuratedSourceMetadata records keyed by "
            "raw_sha256 (only consumed when --corpus-mode=curated)."
        ),
    )
    bake_lora.add_argument(
        "--backend",
        default="synthetic",
        choices=("synthetic", "peft"),
        help=(
            "Bake backend. 'synthetic' is the deterministic CPU "
            "default (hash-derived delta, byte-for-byte reproducible "
            "across machines, for SHADOW deployments and tests). "
            "'peft' runs a real PEFT LoRA training loop on a frozen "
            "HuggingFace base; requires peft + transformers + torch "
            "to be installed (``pip install vz-runtime[torch]``)."
        ),
    )
    bake_lora.add_argument(
        "--rank",
        type=int,
        default=8,
        help="LoRA rank for the training plan (default 8).",
    )
    bake_lora.add_argument(
        "--target-layer-index",
        type=int,
        default=0,
        help="LoRA target layer index (default 0).",
    )
    bake_lora.add_argument(
        "--peft-model-id",
        default="sshleifer/tiny-gpt2",
        help=(
            "HuggingFace model id of the frozen base (only consumed "
            "when --backend peft). Default: sshleifer/tiny-gpt2 "
            "(small enough for CPU smoke tests)."
        ),
    )
    bake_lora.add_argument(
        "--peft-target-modules",
        default="c_attn",
        help=(
            "Comma-separated PEFT target_modules (only consumed when "
            "--backend peft). Default: c_attn (matches GPT-2-style "
            "attention blocks)."
        ),
    )
    bake_lora.add_argument(
        "--peft-alpha",
        type=int,
        default=16,
        help="PEFT lora_alpha (only consumed when --backend peft).",
    )
    bake_lora.add_argument(
        "--peft-dropout",
        type=float,
        default=0.0,
        help="PEFT lora_dropout (only consumed when --backend peft).",
    )
    bake_lora.add_argument(
        "--peft-max-steps",
        type=int,
        default=20,
        help=(
            "Hard cap on optimizer steps per epoch when --backend peft "
            "(default 20; CPU smoke tests fit in seconds at this cap)."
        ),
    )
    bake_lora.add_argument(
        "--peft-device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Torch device when --backend peft (default cpu).",
    )
    bake_lora.add_argument(
        "--evaluation-snapshot",
        required=True,
        help=(
            "Path to a JSON-serialised EvaluationSnapshot, or the "
            "literal 'default-clean' for the developer-mode clean "
            "snapshot."
        ),
    )
    bake_lora.add_argument(
        "--rollback-evidence",
        required=True,
        help=(
            "Non-empty operator-supplied evidence string the gate "
            "logs as the rollback target identifier."
        ),
    )
    bake_lora.add_argument(
        "--validation-delta",
        type=float,
        default=0.05,
        help="Override the gate proposal validation_delta (default 0.05).",
    )
    bake_lora.add_argument(
        "--capacity-cost",
        type=float,
        default=0.30,
        help="Override the gate proposal capacity_cost (default 0.30).",
    )
    # R4-7: optional presence-service LoRA fingerprint registration.
    # When --presence-persona is set (and PRESENCE_BASE_URL +
    # PRESENCE_INTERNAL_SECRET are configured, via env or the flags
    # below), a successful applied bake POSTs the LoRA fingerprint to
    # presence so it knows which LoRA a persona uses. Fire-and-forget:
    # weights stay in the bundle; presence only records the fingerprint.
    bake_lora.add_argument(
        "--presence-persona",
        default=None,
        help=(
            "Presence persona identifier to register this LoRA "
            "fingerprint against: either a presence DB id or "
            "'<app-slug>:<external-ref>'. Omit to skip presence "
            "registration."
        ),
    )
    bake_lora.add_argument(
        "--presence-base-url",
        default=None,
        help="Override PRESENCE_BASE_URL for the LoRA fingerprint POST.",
    )
    bake_lora.add_argument(
        "--presence-internal-secret",
        default=None,
        help="Override PRESENCE_INTERNAL_SECRET for the LoRA POST.",
    )
    bake_lora.add_argument(
        "--presence-lora-layer",
        default="persona",
        help="LoRA layer tag registered with presence (default persona).",
    )

    rollback = subparsers.add_parser(
        "rollback",
        help=(
            "Restore a previously-baked bundle id as the active "
            "persona pool record. Append-only: writes a new audit "
            "row, never deletes prior bundle / audit files."
        ),
    )
    rollback.add_argument(
        "--figure",
        required=True,
        help="Figure identifier under which the rollback runs.",
    )
    rollback.add_argument(
        "--to-bundle",
        required=True,
        help="Target bundle id to restore (must already exist on disk).",
    )
    rollback.add_argument(
        "--rollback-evidence",
        required=True,
        help=(
            "Non-empty operator-supplied justification the audit "
            "row records (e.g. 'reverting after eval regression on "
            "rank-8 bake')."
        ),
    )

    list_cmd = subparsers.add_parser(
        "list",
        help="List persisted bundles under --bundle-root.",
    )
    list_cmd.add_argument(
        "--figure",
        default=None,
        help="Optional figure id filter; defaults to 'all figures'.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Top-level CLI entry point.

    Returns the exit code; the ``__main__`` shim turns this into the
    process exit status. Tests drive subcommands by passing
    ``argv`` directly.
    """

    parser = build_parser()
    if argv is None:
        argv = sys.argv[1:]
    args = parser.parse_args(list(argv))

    handler = {
        "bake-bundle": _commands.cmd_bake_bundle,
        "bake-steering": _commands.cmd_bake_steering,
        "bake-lora": _commands.cmd_bake_lora,
        "rollback": _commands.cmd_rollback,
        "list": _commands.cmd_list,
    }.get(args.command)
    if handler is None:
        parser.error(f"unknown command {args.command!r}")
        return 1
    return handler(args)


__all__ = [
    "build_parser",
    "main",
]
