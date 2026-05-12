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
import sys
from collections.abc import Sequence

from lifeform_domain_figure.cli import _commands


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
        choices=("einstein", "lu_xun"),
        help="Reviewed profile identifier.",
    )
    bake_bundle.add_argument(
        "--corpus-mode",
        default="synthetic",
        choices=("synthetic",),
        help=(
            "Corpus source. 'synthetic' uses the wheel's reviewer-"
            "paraphrased placeholder corpus; 'curated' is reserved "
            "for the V2 archive fetcher work (known-debts #19) and "
            "is not yet wired."
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
            "tracked under known-debts #27."
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
            "under known-debts #27."
        ),
    )
    bake_lora.add_argument(
        "--bundle",
        required=True,
        help="Source bundle id to attach the LoRA artifact to.",
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
