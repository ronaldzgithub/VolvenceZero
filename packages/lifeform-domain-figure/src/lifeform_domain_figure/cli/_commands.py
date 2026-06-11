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
import pathlib
import sys
from collections.abc import Callable

from volvence_zero.credit.gate import GateDecision
from volvence_zero.substrate import default_persona_lora_pool

from lifeform_domain_figure import (
    FigureBundleInputs,
    FigureCorpusSourceBundle,
    PEFTLoRABakeBackend,
    PEFTLoRAConfig,
    SyntheticLoRABakeBackend,
    VerificationGateError,
    VerificationLedger,
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
    load_curated_corpus_from_cleaning_store,
    load_family_profile_file,
    load_figure_bundle,
    load_generic_profile_file,
    load_myriad_profile_file,
    save_figure_bundle,
    synthetic_corpus_from_profile,
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

    profile_factory, _corpus_factory = _resolve_figure_factories(
        args.figure, profile_file=getattr(args, "profile_file", None)
    )
    if profile_factory is None:
        print(
            f"figure {args.figure!r} is not yet wired for bake-bundle "
            f"(see known-debts.md #27 for lu_xun corpus follow-up). "
            f"Dynamic family memorial profiles require both "
            f"--figure family_<id> and --profile-file <path.json>; "
            f"myriad personas require --figure myriad_<id> and "
            f"--profile-file <path.json>.",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if args.corpus_mode == "synthetic":
        return _cmd_bake_bundle_synthetic(args=args, profile_factory=profile_factory)
    if args.corpus_mode == "curated":
        return _cmd_bake_bundle_curated(args=args, profile_factory=profile_factory)
    print(
        f"corpus-mode {args.corpus_mode!r} unknown; expected one of "
        f"('synthetic', 'curated')",
        file=sys.stderr,
    )
    return EXIT_USAGE


def _cmd_bake_bundle_synthetic(*, args, profile_factory) -> int:
    """Synthetic-corpus path: legacy SHADOW behaviour, unchanged."""

    _profile_factory, corpus_factory = _resolve_figure_factories(
        args.figure, profile_file=getattr(args, "profile_file", None)
    )
    if corpus_factory is None:
        print(
            f"figure {args.figure!r} has no synthetic corpus factory wired "
            f"(Einstein and generic --profile-json personas ship one; "
            f"family_* / myriad_* memorial corpora must use "
            f"--corpus-mode curated). See known-debts.md #27.",
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
        corpus_bundle, uploader="lifeform_domain_figure.cli:bake-bundle:synthetic"
    )
    inputs = FigureBundleInputs(
        profile=profile,
        envelopes=envelope_set.envelopes,
        time_window_id=args.time_window_id,
    )
    return _finalise_bake_bundle(args=args, inputs=inputs)


def _cmd_bake_bundle_curated(*, args, profile_factory) -> int:
    """Curated-corpus path: walk L1 cleaning store + curator metadata file."""

    if not args.cleaning_root or not args.curated_metadata_file:
        print(
            "bake-bundle --corpus-mode curated requires both --cleaning-root "
            "and --curated-metadata-file",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if args.require_verification_pass and not args.verification_root:
        print(
            "bake-bundle --require-verification-pass requires "
            "--verification-root pointing at the L2 ledger",
            file=sys.stderr,
        )
        return EXIT_USAGE
    cleaning_root = pathlib.Path(args.cleaning_root)
    metadata_file = pathlib.Path(args.curated_metadata_file)
    try:
        curated = load_curated_corpus_from_cleaning_store(
            cleaning_root=cleaning_root,
            figure_id=args.figure,
            metadata_file=metadata_file,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"bake-bundle (curated): {exc}", file=sys.stderr)
        return EXIT_IO_OR_SCHEMA
    profile = profile_factory()
    corpus_bundle = FigureCorpusSourceBundle(
        figure_id=args.figure,
        papers=curated.papers,
        letters=curated.letters,
        lectures=curated.lectures,
        notebooks=curated.notebooks,
    )
    envelope_set = build_figure_ingestion_envelope(
        corpus_bundle, uploader="lifeform_domain_figure.cli:bake-bundle:curated"
    )
    verification_ledger = None
    if args.verification_root:
        verification_ledger = VerificationLedger(
            pathlib.Path(args.verification_root)
        )
    inputs = FigureBundleInputs(
        profile=profile,
        envelopes=envelope_set.envelopes,
        time_window_id=args.time_window_id,
        provenance_records=curated.provenance_records,
        verification_ledger=verification_ledger,
        require_verification_pass=bool(args.require_verification_pass),
    )
    return _finalise_bake_bundle(args=args, inputs=inputs, source_count_summary=curated.source_count_by_kind)


def _finalise_bake_bundle(
    *,
    args,
    inputs: FigureBundleInputs,
    source_count_summary: dict | None = None,
) -> int:
    """Shared persistence + audit tail used by both corpus modes."""

    try:
        bundle = build_figure_artifact_bundle(inputs)
        bundle_dir = save_figure_bundle(bundle, root_dir=args.bundle_root)
    except VerificationGateError as exc:
        # OFFLINE gate refusal: write an audit record + non-zero exit
        # so the curator sees exactly which axes failed.
        audit = build_audit_record(
            action=FigureBakeAction.BAKE_BUNDLE,
            figure_id=args.figure,
            bundle_id=f"figure-bundle:{args.figure}:gate-blocked",
            previous_bundle_id="absent",
            gate_decision=FigureGateDecisionLabel.BLOCK,
            block_reasons=tuple(str(exc).splitlines()),
            corpus_mode=args.corpus_mode,
        )
        audit_path = write_audit(audit, root_dir=args.audit_root)
        print(
            json.dumps(
                {
                    "action": audit.action.value,
                    "decision": "BLOCK",
                    "block_reason_summary": str(exc).splitlines()[0],
                    "audit_path": str(audit_path),
                    "audit_id": audit.audit_id,
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return EXIT_GATE_BLOCK
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

    payload = {
        "action": audit.action.value,
        "bundle_id": bundle.bundle_id,
        "figure_id": bundle.figure_id,
        "bundle_dir": str(bundle_dir),
        "audit_path": str(audit_path),
        "audit_id": audit.audit_id,
        "provenance_fingerprint": bundle.provenance_fingerprint,
    }
    if source_count_summary is not None:
        payload["source_count_by_kind"] = dict(source_count_summary)
    print(json.dumps(payload, indent=2, sort_keys=True))
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

    if getattr(args, "use_real_residual", False):
        try:
            runtime = _load_real_residual_runtime(
                model_id=args.real_residual_model_id,
            )
        except (ImportError, RuntimeError) as exc:
            print(
                "bake-steering --use-real-residual: could not load real "
                f"substrate runtime: {exc}",
                file=sys.stderr,
            )
            return EXIT_IO_OR_SCHEMA
        contrast_plan = build_steering_training_plan(
            build_einstein_contrast_set(),
            substrate_runtime=runtime,
            layer_index=args.real_residual_layer_index,
        )
        backend_id = "steering-real-residual-v1"
    else:
        contrast_plan = build_steering_training_plan(
            build_einstein_contrast_set()
        )
        backend_id = "steering-cpu-contrastive-v1"
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
        backend_id=backend_id,
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
    if args.backend not in ("synthetic", "peft"):
        print(
            f"backend {args.backend!r} unknown; expected 'synthetic' or 'peft'",
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

    corpus_mode = getattr(args, "corpus_mode", "synthetic")
    if corpus_mode == "curated":
        # Wave N: derive LoRA training envelopes from the same L1
        # cleaning store + curator metadata that produced the
        # curated bundle. Otherwise the LoRA would be trained on
        # synthetic_einstein_corpus() while bundle says "curated"
        # — a silent disagreement that defeats the whole point.
        if not args.cleaning_root or not args.curated_metadata_file:
            print(
                "bake-lora --corpus-mode curated requires both "
                "--cleaning-root and --curated-metadata-file",
                file=sys.stderr,
            )
            return EXIT_USAGE
        try:
            curated = load_curated_corpus_from_cleaning_store(
                cleaning_root=pathlib.Path(args.cleaning_root),
                figure_id=args.figure,
                metadata_file=pathlib.Path(args.curated_metadata_file),
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"bake-lora (curated): {exc}", file=sys.stderr)
            return EXIT_IO_OR_SCHEMA
        envelope_set = build_figure_ingestion_envelope(
            FigureCorpusSourceBundle(
                figure_id=args.figure,
                papers=curated.papers,
                letters=curated.letters,
                lectures=curated.lectures,
                notebooks=curated.notebooks,
            ),
            uploader="lifeform_domain_figure.cli:bake-lora:curated",
        )
    else:
        # Legacy synthetic path — re-derive envelopes deterministically
        # from the figure id; the bundle does not persist envelopes
        # (R15: envelopes are input, the bundle's integrity hash
        # already pins what they produced).
        profile_factory, corpus_factory = _resolve_figure_factories(
            args.figure, profile_file=getattr(args, "profile_file", None)
        )
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
            uploader="lifeform_domain_figure.cli:bake-lora:synthetic",
        )
    plan = build_lora_training_plan(
        figure_id=args.figure,
        envelopes=envelope_set.envelopes,
        rank=args.rank,
        target_layer_index=args.target_layer_index,
    )
    if args.backend == "peft":
        try:
            artifact = _bake_with_peft(args=args, plan=plan)
        except ImportError as exc:
            print(
                "bake-lora --backend peft requires peft + transformers + torch. "
                f"{exc}",
                file=sys.stderr,
            )
            return EXIT_IO_OR_SCHEMA
        except RuntimeError as exc:
            print(f"bake-lora --backend peft: {exc}", file=sys.stderr)
            return EXIT_IO_OR_SCHEMA
    else:
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
    # R4-7: fire-and-forget presence LoRA fingerprint registration on a
    # successful applied bake. Weights stay in the bundle; presence only
    # learns which fingerprint a persona uses. Never affects the exit
    # code — a presence outage must not fail a model promotion.
    if result.applied and getattr(args, "presence_persona", None):
        _maybe_register_lora_with_presence(args=args, artifact=artifact)
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
            peft_checkpoint_dir=getattr(
                target_bundle.lora, "peft_checkpoint_dir", ""
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


def _resolve_figure_factories(
    figure: str,
    *,
    profile_file: str | pathlib.Path | None = None,
) -> _FigureFactories:
    """Map ``--figure`` to (profile_factory, corpus_factory) callables.

    Returns ``(None, None)`` when the figure id has no shipped corpus
    yet (lu_xun, tracked under known-debts.md #27). Profile factories
    exist for both Einstein and Lu Xun, but the corpus side is what
    the CLI gates on — bake-bundle requires both.

    Dynamic family-memorial path (U4 / family-memorial enabler): any
    ``figure`` starting with ``family_`` is a per-memorial profile
    loaded from ``profile_file``. We return a profile factory that
    closes over the loaded profile dict and ``None`` for the corpus
    factory — family memorials are CURATED-CORPUS ONLY (the corpus is
    the family's uploaded text, not a reviewer-curated synthesizer
    output).
    """

    if figure == "einstein":
        return (build_einstein_profile, synthetic_einstein_corpus)
    if figure == "lu_xun":
        # Profile available, corpus pending #27
        return (build_lu_xun_profile, None)
    if figure.startswith("family_"):
        if profile_file is None:
            print(
                f"figure {figure!r} requires --profile-file pointing at "
                "a family memorial profile JSON",
                file=sys.stderr,
            )
            return (None, None)
        loaded_path = pathlib.Path(profile_file)
        profile = load_family_profile_file(loaded_path)
        if profile.profile_id != figure:
            raise ValueError(
                f"--figure {figure!r} does not match profile_id "
                f"{profile.profile_id!r} in {loaded_path}; the bake-worker "
                "is the only authorised producer of family profile files "
                "and must keep these two ids in lock-step."
            )
        return (lambda profile=profile: profile, None)
    if figure.startswith("myriad_"):
        # D-myriad-1: operator-configured generic figure personas. Like
        # the family path these are CURATED-CORPUS ONLY — the corpus is
        # the operator's uploaded text, not a reviewer-curated synthesizer
        # output — so we return a profile factory and ``None`` for corpus.
        if profile_file is None:
            print(
                f"figure {figure!r} requires --profile-file pointing at "
                "a myriad persona profile JSON",
                file=sys.stderr,
            )
            return (None, None)
        loaded_path = pathlib.Path(profile_file)
        profile = load_myriad_profile_file(loaded_path)
        if profile.profile_id != figure:
            raise ValueError(
                f"--figure {figure!r} does not match profile_id "
                f"{profile.profile_id!r} in {loaded_path}; the myriad "
                "operator tool is the only authorised producer of myriad "
                "profile files and must keep these two ids in lock-step."
            )
        return (lambda profile=profile: profile, None)
    if profile_file is not None:
        # Generic persona path (D-myriad-1 / D-MY-1): any other slug
        # loads the Myriad-seed-compatible JSON schema. Unlike the
        # family / myriad branches it ALSO gets a synthetic corpus
        # factory — a deterministic CPU-only smoke corpus derived
        # from the profile itself — so ``--corpus-mode synthetic``
        # works for any slug with zero crawling. ``--corpus-mode
        # curated`` remains available through the shared path.
        loaded_path = pathlib.Path(profile_file)
        profile = load_generic_profile_file(loaded_path, expected_slug=figure)
        return (
            lambda profile=profile: profile,
            lambda profile=profile: synthetic_corpus_from_profile(profile),
        )
    print(
        f"figure {figure!r} requires --profile-json (or --profile-file) "
        "pointing at a generic persona profile JSON (see "
        "lifeform_domain_figure.profiles.generic for the schema)",
        file=sys.stderr,
    )
    return (None, None)


def _load_real_residual_runtime(*, model_id: str):
    """Construct an :class:`OpenWeightResidualRuntime` for steering bake.

    Defers the import of ``vz-substrate.residual_backend`` to call
    time so the CLI module loads fine when transformers / torch
    are absent. Raises :class:`ImportError` with a precise install
    hint when the real path is requested but the optional deps
    are not installed.
    """

    try:
        from volvence_zero.substrate.residual_backend import (
            TransformersOpenWeightResidualRuntime,
        )
    except ImportError as exc:
        raise ImportError(
            "real-residual steering bake requires transformers + torch. "
            "Install via ``pip install vz-substrate[hf]`` (and torch). "
            f"Underlying error: {exc}"
        ) from exc
    return TransformersOpenWeightResidualRuntime(
        model_id=model_id,
        device="cpu",
    )


def _bake_with_peft(*, args: argparse.Namespace, plan):
    """Construct a :class:`PEFTLoRABakeBackend` from CLI args + bake.

    Kept as a small helper so :func:`cmd_bake_lora` reads as one
    linear flow regardless of backend. The backend ctor itself
    raises ``ImportError`` if peft is not installed, which the
    caller surfaces as :data:`EXIT_IO_OR_SCHEMA`.
    """

    target_modules = tuple(
        module.strip()
        for module in str(args.peft_target_modules).split(",")
        if module.strip()
    )
    if not target_modules:
        raise RuntimeError(
            "--peft-target-modules must list at least one module name"
        )
    config = PEFTLoRAConfig(
        target_modules=target_modules,
        rank=args.rank,
        alpha=args.peft_alpha,
        dropout=args.peft_dropout,
    )
    backend = PEFTLoRABakeBackend(
        model_id=args.peft_model_id,
        peft_config=config,
        runtime_device=args.peft_device,
        max_steps=args.peft_max_steps,
    )
    return backend.bake(plan)


def _maybe_register_lora_with_presence(
    *,
    args: argparse.Namespace,
    artifact: object,
) -> None:
    """R4-7: best-effort POST of the baked LoRA fingerprint to presence.

    Resolves base url + secret from the CLI flags, then env. Skips
    silently when either is missing. Never raises — import + network
    failures are swallowed so a presence outage cannot fail a bake.
    """

    import os

    base_url = (
        getattr(args, "presence_base_url", None)
        or os.environ.get("PRESENCE_BASE_URL")
        or ""
    ).strip()
    secret = (
        getattr(args, "presence_internal_secret", None)
        or os.environ.get("PRESENCE_INTERNAL_SECRET")
        or ""
    ).strip()
    if not base_url or not secret:
        return
    try:
        from lifeform_domain_figure import (
            build_registration_from_artifact,
            register_lora_into_presence,
        )

        registration = build_registration_from_artifact(
            artifact=artifact,  # type: ignore[arg-type]
            persona_identifier=str(args.presence_persona),
            license_label=f"figure-bake:{getattr(artifact, 'figure_id', 'unknown')}",
            layer=getattr(args, "presence_lora_layer", "persona"),
        )
        register_lora_into_presence(
            registration=registration,
            presence_base_url=base_url,
            internal_secret=secret,
        )
    except Exception as exc:  # noqa: BLE001 - fire-and-forget by design
        print(
            f"bake-lora: presence LoRA registration skipped ({exc})",
            file=sys.stderr,
        )


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
