"""Subcommand handlers for ``python -m lifeform_domain_figure.cli``.

Each handler returns the process exit code (see
:mod:`lifeform_domain_figure.cli.__init__` for the exit-code
semantics). Handlers route every mutation through the wheel's public
surface so the CLI can never become a "second editor" of figure-
internal state (R8): all bundle compilation goes through
:func:`build_figure_artifact_bundle`, all gate apply goes through
:func:`apply_steering_through_gate` /
:func:`apply_persona_lora_through_gate`, all pool registration goes
through :func:`register_bundle_persona_lora`. The static contract
test ``tests/contracts/test_figure_cli_uses_only_public_surface.py``
enforces this with an AST scan.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable

from volvence_zero.credit.gate import GateDecision
from volvence_zero.substrate import PersonaLoRAPool, default_persona_lora_pool

from lifeform_domain_figure import (
    FigureBundleInputs,
    FigureCorpusSourceBundle,
    SyntheticLoRABakeBackend,
    apply_persona_lora_through_gate,
    apply_steering_through_gate,
    bake_steering_set,
    build_einstein_contrast_set,
    build_einstein_profile,
    build_figure_artifact_bundle,
    build_figure_ingestion_envelope,
    build_lora_training_plan,
    build_lu_xun_profile,
    build_steering_training_plan,
    list_figure_bundles,
    load_figure_bundle,
    save_figure_bundle,
    synthetic_einstein_corpus,
)
from lifeform_domain_figure.audit import (
    FigureBakeAction,
    FigureGateDecisionLabel,
    build_audit_record,
    find_previous_audit_for_bundle,
    write_audit,
)
from lifeform_domain_figure.cli._eval_snapshot_loader import (
    load_evaluation_snapshot,
)


# Exit code constants (mirrored in the package docstring)
EXIT_OK = 0
EXIT_USAGE = 1
EXIT_GATE_BLOCK = 2
EXIT_IO_OR_SCHEMA = 3


def cmd_bake_bundle(args: argparse.Namespace) -> int:
    """Compile a profile + corpus into a fresh bundle, persist + audit."""

    profile_factory, corpus_factory = _resolve_figure_factories(args.figure)
    if profile_factory is None or corpus_factory is None:
        print(
            f"figure {args.figure!r} is not yet wired for bake-bundle "
            f"(see known-debts.md #27 for lu_xun corpus follow-up)",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if args.corpus_mode != "synthetic":
        print(
            f"corpus-mode {args.corpus_mode!r} is reserved for the V2 "
            f"archive fetcher work (known-debts.md #19) and not yet "
            f"wired",
            file=sys.stderr,
        )
        return EXIT_USAGE

    profile = profile_factory()
    papers, letters, lectures, notebooks = corpus_factory()
    corpus_bundle = FigureCorpusSourceBundle(
        figure_id=args.figure,
        papers=papers,
        letters=letters,
        lectures=lectures,
        notebooks=notebooks,
    )
    envelope_set = build_figure_ingestion_envelope(
        corpus_bundle, uploader="lifeform_domain_figure.cli:bake-bundle"
    )
    inputs = FigureBundleInputs(
        profile=profile,
        envelopes=envelope_set.envelopes,
        time_window_id=args.time_window_id,
    )
    try:
        bundle = build_figure_artifact_bundle(inputs)
        bundle_dir = save_figure_bundle(bundle, root_dir=args.bundle_root)
    except (OSError, ValueError) as exc:
        print(f"bake-bundle failed: {exc}", file=sys.stderr)
        return EXIT_IO_OR_SCHEMA

    audit = build_audit_record(
        action=FigureBakeAction.BAKE_BUNDLE,
        figure_id=bundle.figure_id,
        bundle_id=bundle.bundle_id,
        previous_bundle_id="absent",
        gate_decision=FigureGateDecisionLabel.NA,
        corpus_mode=args.corpus_mode,
    )
    audit_path = write_audit(audit, root_dir=args.audit_root)

    print(json.dumps(
        {
            "action": audit.action.value,
            "bundle_id": bundle.bundle_id,
            "figure_id": bundle.figure_id,
            "bundle_dir": str(bundle_dir),
            "audit_path": str(audit_path),
            "audit_id": audit.audit_id,
        },
        indent=2,
        sort_keys=True,
    ))
    return EXIT_OK


def cmd_bake_steering(args: argparse.Namespace) -> int:
    """Bake steering on an existing bundle, route through OFFLINE gate."""

    if args.figure != "einstein":
        print(
            f"figure {args.figure!r} not yet wired for bake-steering "
            f"(only Einstein has a reviewed contrast set; lu_xun is "
            f"tracked under known-debts.md #27)",
            file=sys.stderr,
        )
        return EXIT_USAGE

    try:
        snapshot = load_evaluation_snapshot(args.evaluation_snapshot)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(
            f"failed to load evaluation snapshot from "
            f"{args.evaluation_snapshot!r}: {exc}",
            file=sys.stderr,
        )
        return EXIT_IO_OR_SCHEMA

    try:
        base_bundle = load_figure_bundle(
            root_dir=args.bundle_root,
            bundle_id=args.bundle,
            figure_id=args.figure,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"failed to load source bundle: {exc}", file=sys.stderr)
        return EXIT_IO_OR_SCHEMA

    contrast_plan = build_steering_training_plan(build_einstein_contrast_set())
    steering = bake_steering_set(contrast_plan)

    result = apply_steering_through_gate(
        base_bundle=base_bundle,
        steering=steering,
        evaluation_snapshot=snapshot,
        validation_delta=args.validation_delta,
        capacity_cost=args.capacity_cost,
        rollback_evidence=args.rollback_evidence,
    )
    return _finalise_gate_apply(
        args=args,
        result_bundle=result.bundle,
        base_bundle=base_bundle,
        decision=result.gate.decision,
        block_reasons=result.gate.block_reasons,
        applied=result.applied,
        action=FigureBakeAction.BAKE_STEERING,
        backend_id="steering-cpu-contrastive-v1",
        record_id=None,
        previous_record_id="absent",
    )


def cmd_bake_lora(args: argparse.Namespace) -> int:
    """Bake a persona LoRA on an existing bundle, route through OFFLINE gate."""

    if args.figure != "einstein":
        print(
            f"figure {args.figure!r} not yet wired for bake-lora "
            f"(only Einstein has a reviewed corpus; lu_xun is "
            f"tracked under known-debts.md #27)",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if args.backend != "synthetic":
        print(
            f"backend {args.backend!r} not yet wired (PEFT backend is "
            f"tracked under known-debts.md #18)",
            file=sys.stderr,
        )
        return EXIT_USAGE

    try:
        snapshot = load_evaluation_snapshot(args.evaluation_snapshot)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(
            f"failed to load evaluation snapshot from "
            f"{args.evaluation_snapshot!r}: {exc}",
            file=sys.stderr,
        )
        return EXIT_IO_OR_SCHEMA

    try:
        base_bundle = load_figure_bundle(
            root_dir=args.bundle_root,
            bundle_id=args.bundle,
            figure_id=args.figure,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"failed to load source bundle: {exc}", file=sys.stderr)
        return EXIT_IO_OR_SCHEMA

    # Re-derive the corpus envelopes deterministically from the figure
    # id; the bundle does not persist envelopes (R15: envelopes are
    # input, the bundle's integrity hash already pins what they
    # produced).
    profile_factory, corpus_factory = _resolve_figure_factories(args.figure)
    if profile_factory is None or corpus_factory is None:
        print(
            f"internal: figure {args.figure!r} not in factory map",
            file=sys.stderr,
        )
        return EXIT_USAGE
    papers, letters, lectures, notebooks = corpus_factory()
    envelope_set = build_figure_ingestion_envelope(
        FigureCorpusSourceBundle(
            figure_id=args.figure,
            papers=papers,
            letters=letters,
            lectures=lectures,
            notebooks=notebooks,
        ),
        uploader="lifeform_domain_figure.cli:bake-lora",
    )
    plan = build_lora_training_plan(
        figure_id=args.figure,
        envelopes=envelope_set.envelopes,
        rank=args.rank,
        target_layer_index=args.target_layer_index,
    )
    artifact = SyntheticLoRABakeBackend().bake(plan)

    pool = default_persona_lora_pool()
    result = apply_persona_lora_through_gate(
        base_bundle=base_bundle,
        artifact=artifact,
        evaluation_snapshot=snapshot,
        pool=pool,
        validation_delta=args.validation_delta,
        capacity_cost=args.capacity_cost,
        rollback_evidence=args.rollback_evidence,
    )
    return _finalise_gate_apply(
        args=args,
        result_bundle=result.bundle,
        base_bundle=base_bundle,
        decision=result.gate.decision,
        block_reasons=result.gate.block_reasons,
        applied=result.applied,
        action=FigureBakeAction.BAKE_LORA,
        backend_id=artifact.backend_id,
        record_id=result.record_id,
        previous_record_id=result.previous_record_id,
    )


def cmd_rollback(args: argparse.Namespace) -> int:
    """Restore ``--to-bundle`` as the active bundle / pool record.

    R15 append-only: prior bundle / audit files are preserved on
    disk; the rollback writes a new ``ROLLBACK`` audit row that
    names the previous record id, so the audit log reads as a
    full timeline (no edits).
    """

    if not args.rollback_evidence.strip():
        print(
            "rollback: --rollback-evidence must be non-empty",
            file=sys.stderr,
        )
        return EXIT_USAGE

    try:
        target_bundle = load_figure_bundle(
            root_dir=args.bundle_root,
            bundle_id=args.to_bundle,
            figure_id=args.figure,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"rollback: target bundle not found: {exc}", file=sys.stderr)
        return EXIT_IO_OR_SCHEMA

    pool = default_persona_lora_pool()
    previous_record_id = (
        pool.lookup(args.figure).record_id
        if pool.has(args.figure)
        else "absent"
    )

    new_record_id: str | None = None
    if target_bundle.lora is not None:
        if pool.has(args.figure):
            pool.deregister(args.figure)
        prior_audit = find_previous_audit_for_bundle(
            root_dir=args.audit_root,
            figure_id=args.figure,
            bundle_id=args.to_bundle,
        )
        new_record_id = pool.register(
            figure_id=target_bundle.lora.figure_id,
            source_bundle_id=target_bundle.bundle_id,
            backend_id=target_bundle.lora.backend_id,
            training_plan_hash=target_bundle.lora.training_plan_hash,
            adapter_layers=target_bundle.lora.adapter_layers,
            parameter_count=target_bundle.lora.parameter_count,
            description=(
                target_bundle.lora.description
                + (
                    f" (rollback to bundle "
                    f"{target_bundle.bundle_id}; prior_audit="
                    f"{prior_audit.audit_id[:12] if prior_audit else 'absent'})"
                )
            ),
        )

    audit = build_audit_record(
        action=FigureBakeAction.ROLLBACK,
        figure_id=args.figure,
        bundle_id=target_bundle.bundle_id,
        previous_bundle_id="rolled-back",
        record_id=new_record_id,
        previous_record_id=previous_record_id,
        gate_decision=FigureGateDecisionLabel.NA,
        rollback_evidence=args.rollback_evidence,
    )
    audit_path = write_audit(audit, root_dir=args.audit_root)

    print(json.dumps(
        {
            "action": audit.action.value,
            "figure_id": args.figure,
            "restored_bundle_id": target_bundle.bundle_id,
            "previous_record_id": previous_record_id,
            "new_record_id": new_record_id,
            "audit_path": str(audit_path),
            "audit_id": audit.audit_id,
        },
        indent=2,
        sort_keys=True,
    ))
    return EXIT_OK


def cmd_list(args: argparse.Namespace) -> int:
    """Enumerate persisted bundles for one or all figures."""

    try:
        manifests = list_figure_bundles(
            root_dir=args.bundle_root,
            figure_id=args.figure,
        )
    except (OSError, ValueError) as exc:
        print(f"list failed: {exc}", file=sys.stderr)
        return EXIT_IO_OR_SCHEMA

    rendered = [
        {
            "figure_id": m.figure_id,
            "bundle_id": m.bundle_id,
            "profile_version": m.profile_version,
            "version_window": list(m.version_window),
            "integrity_hash": m.integrity_hash,
            "created_at_iso": m.created_at_iso,
            "steering_present": m.steering_present,
            "lora_present": m.lora_present,
            "bundle_dir": str(m.bundle_dir),
        }
        for m in manifests
    ]
    print(json.dumps({"bundles": rendered, "count": len(rendered)}, indent=2, sort_keys=True))
    return EXIT_OK


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_FigureFactories = tuple[Callable[[], object] | None, Callable[[], object] | None]


def _resolve_figure_factories(figure: str) -> _FigureFactories:
    """Map ``--figure`` to (profile_factory, corpus_factory) callables.

    Returns ``(None, None)`` when the figure id has no shipped corpus
    yet (lu_xun, tracked under known-debts.md #27). Profile factories
    exist for both Einstein and Lu Xun, but the corpus side is what
    the CLI gates on — bake-bundle requires both.
    """

    if figure == "einstein":
        return (build_einstein_profile, synthetic_einstein_corpus)
    if figure == "lu_xun":
        # Profile available, corpus pending #27
        return (build_lu_xun_profile, None)
    return (None, None)


def _finalise_gate_apply(
    *,
    args: argparse.Namespace,
    result_bundle,
    base_bundle,
    decision: GateDecision,
    block_reasons: tuple[str, ...],
    applied: bool,
    action: FigureBakeAction,
    backend_id: str,
    record_id: str | None,
    previous_record_id: str,
) -> int:
    """Persist + audit the outcome of a gate-apply subcommand.

    On ``GateDecision.ALLOW``: save the new bundle, write an
    ``ALLOW`` audit row, return :data:`EXIT_OK`.
    On ``GateDecision.BLOCK``: write a ``BLOCK`` audit row carrying
    the structured ``block_reasons`` tuple, leave the persisted
    bundle dir unchanged, return :data:`EXIT_GATE_BLOCK`.
    """

    if applied and decision is GateDecision.ALLOW:
        try:
            bundle_dir = save_figure_bundle(
                result_bundle, root_dir=args.bundle_root
            )
        except (OSError, ValueError) as exc:
            print(f"{action.value}: persist failed: {exc}", file=sys.stderr)
            return EXIT_IO_OR_SCHEMA
        audit = build_audit_record(
            action=action,
            figure_id=result_bundle.figure_id,
            bundle_id=result_bundle.bundle_id,
            previous_bundle_id=base_bundle.bundle_id,
            record_id=record_id,
            previous_record_id=previous_record_id,
            gate_decision=FigureGateDecisionLabel.ALLOW,
            rollback_evidence=args.rollback_evidence,
            validation_delta=args.validation_delta,
            capacity_cost=args.capacity_cost,
            backend_id=backend_id,
        )
        audit_path = write_audit(audit, root_dir=args.audit_root)
        print(json.dumps(
            {
                "action": audit.action.value,
                "applied": True,
                "decision": "ALLOW",
                "bundle_id": result_bundle.bundle_id,
                "previous_bundle_id": base_bundle.bundle_id,
                "record_id": record_id,
                "previous_record_id": previous_record_id,
                "bundle_dir": str(bundle_dir),
                "audit_path": str(audit_path),
                "audit_id": audit.audit_id,
            },
            indent=2,
            sort_keys=True,
        ))
        return EXIT_OK

    audit = build_audit_record(
        action=action,
        figure_id=base_bundle.figure_id,
        bundle_id=base_bundle.bundle_id,
        previous_bundle_id="absent",
        record_id=None,
        previous_record_id=previous_record_id,
        gate_decision=FigureGateDecisionLabel.BLOCK,
        block_reasons=block_reasons,
        rollback_evidence=args.rollback_evidence,
        validation_delta=args.validation_delta,
        capacity_cost=args.capacity_cost,
        backend_id=backend_id,
    )
    audit_path = write_audit(audit, root_dir=args.audit_root)
    print(json.dumps(
        {
            "action": audit.action.value,
            "applied": False,
            "decision": "BLOCK",
            "bundle_id": base_bundle.bundle_id,
            "block_reasons": list(block_reasons),
            "audit_path": str(audit_path),
            "audit_id": audit.audit_id,
        },
        indent=2,
        sort_keys=True,
    ), file=sys.stderr)
    return EXIT_GATE_BLOCK


__all__ = [
    "EXIT_GATE_BLOCK",
    "EXIT_IO_OR_SCHEMA",
    "EXIT_OK",
    "EXIT_USAGE",
    "cmd_bake_bundle",
    "cmd_bake_lora",
    "cmd_bake_steering",
    "cmd_list",
    "cmd_rollback",
]
