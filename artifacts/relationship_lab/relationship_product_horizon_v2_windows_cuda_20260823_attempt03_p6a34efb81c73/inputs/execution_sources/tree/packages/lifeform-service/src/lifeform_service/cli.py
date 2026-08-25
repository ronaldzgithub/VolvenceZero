"""``lifeform-serve`` CLI \u2014 start the HTTP service.

Default vertical is ``companion`` (loaded from ``lifeform-domain-emogpt``)
when present. Other verticals can be selected with ``--vertical NAME``;
``--list-verticals`` prints what is available in the current install.

Substrate sharing
-----------------

By default the service runs in ``--substrate-mode synthetic`` which uses
the lightweight in-process synthetic substrate \u2014 no GPU, no model
weights, fine for tests and demos. For production on one GPU server,
pass ``--substrate-mode hf-shared --substrate-model-id Qwen/...`` and
the service loads ONE Qwen model at startup and shares it across every
session. The model is eagerly loaded before the listener binds, so the
first ``POST /v1/sessions/{id}/turns`` does not pay the model-load
latency.

Concurrency model: aiohttp runs a single-threaded asyncio event loop;
``runtime.generate(...)`` is sync and blocks the loop for its duration.
Concurrent sessions therefore serialise naturally on the model. If you
ever introduce ``run_in_executor`` for parallel decoding, you MUST also
add a ``threading.Lock`` around the runtime \u2014 the current default does
not need one because there is no parallelism in the inference path.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING

from aiohttp import web
from volvence_zero.runtime import WiringLevel

from lifeform_service.app import create_app
from lifeform_service.alpha import AlphaServiceConfig, load_alpha_users
from lifeform_service.companion_evidence_profile import (
    COMPANION_EVIDENCE_PROFILE_NAMES,
    MSC_RUNTIME_COLLECTOR,
    MSC_RUNTIME_PROFILE_NAMES,
    MSC_STEERING_SHADOW_COLLECTOR,
    resolve_companion_evidence_profile,
    write_companion_evidence_profile_attestation,
)
from lifeform_service.character_packages import (
    load_character_runtime_assets,
    write_character_runtime_stack_attestation,
)
from lifeform_service.steering_activation import (
    load_steering_activation_authorization,
)
from lifeform_service.verticals import (
    default_vertical_name,
    discover_companion_ablation_verticals,
    discover_verticals,
)

if TYPE_CHECKING:
    from lifeform_service.character_packages import CharacterRuntimeAssets
    from volvence_zero.runtime import WiringLevel
    from volvence_zero.substrate import CommonAdapterBundle, OpenWeightResidualRuntime
    from volvence_zero.steering_contracts import SteeringArtifactBundle


_LOG = logging.getLogger("lifeform-serve")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lifeform-serve",
        description=(
            "Start a multi-tenant HTTP service that exposes a Volvence Zero "
            "lifeform from a chosen ``lifeform-domain-*`` vertical. Default "
            "deployment uses one shared synthetic substrate; for one-GPU "
            "production servers pass --substrate-mode hf-shared."
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host (default 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Bind port (default 8765).",
    )
    parser.add_argument(
        "--vertical",
        default=None,
        help=(
            "Vertical name to host. Defaults to the first installed vertical. "
            "Use --list-verticals to inspect the current install."
        ),
    )
    parser.add_argument(
        "--ablation-bundle",
        action="store_true",
        help=(
            "Host the reviewed Companion Bench ablation vertical bundle in one "
            "process. Mutually exclusive with --vertical; OpenAI-compat callers "
            "select the track with X-Compat-Vertical or ?vertical=."
        ),
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=256,
        help="Cap on concurrently live sessions before LRU eviction (default 256).",
    )
    parser.add_argument(
        "--idle-eviction-seconds",
        type=float,
        default=1800.0,
        help=(
            "Sessions idle longer than this many seconds are auto-closed. "
            "Pass 0 (or negative) to disable idle eviction."
        ),
    )
    parser.add_argument(
        "--substrate-mode",
        choices=("synthetic", "hf-shared"),
        default="synthetic",
        help=(
            "synthetic: no GPU, no model weights (default; safe for tests). "
            "hf-shared: load ONE Qwen-style model at startup and share it "
            "across every session. Requires the [hf] extras of vz-substrate."
        ),
    )
    parser.add_argument(
        "--substrate-model-id",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help=("HF model id to load when --substrate-mode=hf-shared (default Qwen/Qwen2.5-1.5B-Instruct)."),
    )
    parser.add_argument(
        "--substrate-model-source",
        default=None,
        help=(
            "Optional local HF snapshot path while preserving the declared "
            "--substrate-model-id lineage. Evidence profiles use this to pin "
            "the exact frozen weight directory."
        ),
    )
    parser.add_argument(
        "--substrate-device",
        default="auto",
        help="Torch device for the shared HF runtime (auto / cpu / cuda / cuda:0 / ...).",
    )
    parser.add_argument(
        "--substrate-model-dtype",
        choices=("float16", "bfloat16", "float32"),
        default=None,
        help=(
            "Optional explicit frozen model load dtype. Evidence runners use "
            "this to bind numerical stability to their preregistered lineage."
        ),
    )
    parser.add_argument(
        "--substrate-local-files-only",
        action="store_true",
        help="Forbid HF Hub network fetches (use only the local cache).",
    )
    parser.add_argument(
        "--substrate-layer-indices",
        type=int,
        nargs="+",
        default=None,
        help="Explicit frozen residual hook layers for evidence capture.",
    )
    parser.add_argument(
        "--substrate-activation-width",
        type=int,
        default=8,
        help="Residual readout width per selected layer (default 8).",
    )
    parser.add_argument(
        "--substrate-max-length",
        type=int,
        default=None,
        help=(
            "Explicit substrate input-token limit. The ordinary service keeps "
            "the runtime default when omitted; MSC evidence requires this and "
            "fails instead of truncating."
        ),
    )
    parser.add_argument(
        "--substrate-expected-weights-sha256",
        default="",
        help="Fail startup unless the local frozen weights match this SHA-256.",
    )
    parser.add_argument(
        "--msc-temporal-n-z",
        type=int,
        choices=(3, 16, 64, 256),
        default=None,
        help=(
            "Evidence-only temporal controller capacity for the MSC runtime "
            "collector. Required with that profile; rejected elsewhere."
        ),
    )
    parser.add_argument(
        "--steering-artifact-bundle",
        type=Path,
        default=None,
        help=(
            "Model-bound SteeringArtifactBundle. Evidence use is limited to "
            "the MSC SHADOW collector; production use additionally requires "
            "the exact B3 manifest, activation plan, rollout step, and for "
            "step >1 the immediately preceding canary receipt."
        ),
    )
    parser.add_argument(
        "--steering-promotion-manifest",
        type=Path,
        default=None,
        help=(
            "B3 steering-promotion artifact_manifest.json that binds the "
            "candidate bundle and activation plan."
        ),
    )
    parser.add_argument(
        "--steering-activation-plan",
        type=Path,
        default=None,
        help="B3 steering-activation-plan.v3 artifact.",
    )
    parser.add_argument(
        "--steering-activation-step",
        type=int,
        default=None,
        help=(
            "One-based authorized rollout step to apply. Each successive "
            "service rollout advances by exactly one frozen plan step."
        ),
    )
    parser.add_argument(
        "--steering-previous-activation-receipt",
        type=Path,
        default=None,
        help=(
            "Required for B3 rollout step >1. Must be the immediately "
            "preceding healthy canary receipt; forbidden for step 1."
        ),
    )
    parser.add_argument(
        "--common-adapter-bundle",
        type=Path,
        default=None,
        help=(
            "Path to one OFFLINE-gated ACTIVE CommonAdapterBundle. The bundle "
            "is loaded process-wide and must match --substrate-model-id."
        ),
    )
    parser.add_argument(
        "--character-package-manifest",
        type=Path,
        action="append",
        default=[],
        help=("Path to an admitted CharacterPackageManifest. Repeat to load multiple per-session character choices."),
    )
    parser.add_argument(
        "--character-package-mode",
        choices=("disabled", "shadow", "active"),
        default="shadow",
        help="Default wiring for --character-package-manifest entries.",
    )
    parser.add_argument(
        "--character-package-wiring",
        action="append",
        default=[],
        metavar="CHARACTER_ID=MODE",
        help=("Per-character disabled/shadow/active override. Repeat for more than one character."),
    )
    parser.add_argument(
        "--list-verticals",
        action="store_true",
        help="Print the verticals discovered in this install and exit.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default INFO).",
    )
    parser.add_argument(
        "--alpha-enabled",
        action="store_true",
        help="Enable closed-alpha service mode with required user identity.",
    )
    parser.add_argument(
        "--alpha-users-file",
        default=None,
        help="JSON list or {'users': [...]} allowlist for closed-alpha users.",
    )
    parser.add_argument(
        "--memory-scope-root-dir",
        default=None,
        help="Root directory for per-user scoped memory in alpha mode.",
    )
    parser.add_argument(
        "--evidence-root-dir",
        default=None,
        help=(
            "Root directory for closed-alpha session, typed dialogue-outcome, "
            "and deletion evidence bundles."
        ),
    )
    parser.add_argument(
        "--relationship-intelligence",
        action="store_true",
        help=(
            "Enable the P4 closed-alpha relationship turn/outcome/followup "
            "routes. Action advisories remain SHADOW."
        ),
    )
    parser.add_argument(
        "--relationship-outcome-typing-qualification",
        default=None,
        metavar="PATH",
        help=(
            "Frozen content-hashed qualification artifact for real-user "
            "relationship outcome typing. Without a passing artifact, "
            "outcomes are collection-only."
        ),
    )
    parser.add_argument(
        "--relationship-training-candidate-root-dir",
        default=None,
        help=(
            "Separate offline root for per-outcome opt-in relationship "
            "training candidates; must differ from --evidence-root-dir."
        ),
    )
    parser.add_argument(
        "--allow-evidence-time-override",
        action="store_true",
        help=(
            "Allow observed_at_ms on continuity-metrics for isolated "
            "virtual-calendar evidence runs. Never enable in product service."
        ),
    )
    parser.add_argument(
        "--companion-evidence-profile",
        choices=COMPANION_EVIDENCE_PROFILE_NAMES,
        default=None,
        help=(
            "Evidence-only matched-intervention startup profile. Requires "
            "the companion vertical, closed-alpha isolation, a dedicated "
            "evidence root, virtual-calendar mode, hf-shared, and local "
            "model files. Never use in product service."
        ),
    )
    parser.add_argument(
        "--companion-playbook-overlay-mode",
        choices=("disabled", "shadow"),
        default="disabled",
        help=(
            "Operator-controlled companion playbook overlay rollout. "
            "disabled does not read the asset; shadow validates the candidate "
            "before startup while keeping live behavior on the baseline. "
            "ACTIVE is intentionally unavailable at this service boundary."
        ),
    )
    parser.add_argument(
        "--companion-playbook-overlay-path",
        type=Path,
        default=None,
        help=(
            "Optional companion overlay candidate path for shadow validation. "
            "Omit to use the immutable asset shipped by lifeform-domain-emogpt."
        ),
    )
    parser.add_argument(
        "--service-version",
        default="closed-alpha-v0",
        help="Service version returned in alpha responses.",
    )
    parser.add_argument(
        "--policy-version",
        default="alpha-policy-v0",
        help="Policy version returned in alpha responses.",
    )
    parser.add_argument(
        "--require-alpha-preflight",
        action="store_true",
        help="Run closed-alpha preflight before binding the service.",
    )
    parser.add_argument(
        "--enable-openai-compat",
        action="store_true",
        help=(
            "Mount the OpenAI Chat Completions compatible router "
            "(POST /v1/chat/completions) on this app. The existing "
            "/v1/models route is owned by lifeform-service and is "
            "unaffected by this flag. Used by external benchmark "
            "harnesses (EQ-Bench 3, EmpathyBench, OpenAI Python "
            "client) — see lifeform-openai-compat wheel + "
            "docs/external/. Requires the lifeform-openai-compat "
            "wheel to be installed. Default: off (existing "
            "/v1/sessions/{id}/turns API only)."
        ),
    )
    parser.add_argument(
        "--openai-compat-api-key-env",
        default="LIFEFORM_LOCAL_API_KEY",
        help=(
            "Environment variable containing the Bearer API key required by "
            "the OpenAI-compatible /v1/chat/completions route when "
            "--enable-openai-compat is set (default LIFEFORM_LOCAL_API_KEY)."
        ),
    )
    parser.add_argument(
        "--protocol-approved-dir",
        default=None,
        help=(
            "Directory to persist approved protocols as JSON. When "
            "set, the protocol uptake service mirrors every "
            "approved candidate to '<dir>/<protocol_id>.json' and "
            "exposes the GET/POST/DELETE /v1/protocols/library "
            "routes so the chat UI can pick which persisted "
            "protocols to activate across restarts. Default: None "
            "(library mode disabled; approved protocols evaporate "
            "on restart)."
        ),
    )
    return parser


def _maybe_build_protocol_uptake_service(args: argparse.Namespace):
    """Construct a ProtocolUptakeService when persistence is requested.

    Returns ``None`` when ``--protocol-approved-dir`` is not set — in
    that case the CLI keeps the legacy single-vertical behavior of
    not mounting the protocol routes at all (matches pre-persistence
    behavior of this CLI; the richer ``start_browser_chat_qwen``
    scripts do their own wiring).

    Returns a configured :class:`ProtocolUptakeService` with the
    persistence store wired so approved protocols are mirrored to
    disk and library routes work.
    """

    approved_dir = (args.protocol_approved_dir or "").strip() or None
    if approved_dir is None:
        return None
    from pathlib import Path

    from lifeform_service.protocol_persistence import ProtocolPersistenceStore
    from lifeform_service.protocol_uptake import (
        ProtocolUptakeConfig,
        ProtocolUptakeService,
    )

    store = ProtocolPersistenceStore(Path(approved_dir))
    service = ProtocolUptakeService(
        config=ProtocolUptakeConfig(),
        persistence=store,
    )
    persisted = store.list_all()
    if persisted:
        _LOG.info(
            "protocol library: discovered %d persisted protocol(s) in %s "
            "(use POST /v1/protocols/library/<id>/load to activate)",
            len(persisted),
            approved_dir,
        )
    else:
        _LOG.info(
            "protocol library: empty (no .json files under %s)",
            approved_dir,
        )
    return service


def _character_wiring_overrides(
    values: list[str],
) -> dict[str, "WiringLevel"]:
    from volvence_zero.runtime import WiringLevel

    overrides: dict[str, WiringLevel] = {}
    for value in values:
        character_id, separator, mode = value.strip().partition("=")
        if not separator or not character_id or not mode:
            raise ValueError("--character-package-wiring must use CHARACTER_ID=MODE.")
        if character_id in overrides:
            raise ValueError(f"duplicate --character-package-wiring id {character_id!r}.")
        try:
            overrides[character_id] = WiringLevel(mode)
        except ValueError as exc:
            raise ValueError(
                f"--character-package-wiring MODE must be disabled, shadow, or active; got {mode!r}."
            ) from exc
    return overrides


def _load_admitted_character_stack(
    args: argparse.Namespace,
) -> tuple["CommonAdapterBundle | None", "CharacterRuntimeAssets | None"]:
    from volvence_zero.runtime import WiringLevel
    from volvence_zero.substrate import CommonAdapterBundle

    bundle_path = args.common_adapter_bundle
    manifest_paths = tuple(args.character_package_manifest)
    if bundle_path is None:
        if manifest_paths:
            raise ValueError("--character-package-manifest requires --common-adapter-bundle.")
        if args.character_package_wiring:
            raise ValueError("--character-package-wiring requires --character-package-manifest.")
        return None, None
    if args.substrate_mode != "hf-shared":
        raise ValueError("--common-adapter-bundle requires --substrate-mode hf-shared.")
    common_bundle = CommonAdapterBundle.from_json(bundle_path.read_text(encoding="utf-8"))
    common_bundle.require_active()
    if common_bundle.base_model_id != args.substrate_model_id:
        raise ValueError("common adapter bundle base_model_id does not match --substrate-model-id.")
    if not manifest_paths:
        if args.character_package_wiring:
            raise ValueError("--character-package-wiring requires --character-package-manifest.")
        return common_bundle, None
    assets = load_character_runtime_assets(
        common_adapter_bundle_path=bundle_path,
        manifest_paths=manifest_paths,
        wiring_by_character=_character_wiring_overrides(args.character_package_wiring),
        default_wiring=WiringLevel(args.character_package_mode),
    )
    if assets.common_adapter_bundle.bundle_id != common_bundle.bundle_id:
        raise ValueError("character manifest loader resolved a different common adapter.")
    return assets.common_adapter_bundle, assets


def _build_shared_substrate(
    args: argparse.Namespace,
    *,
    common_adapter_bundle: "CommonAdapterBundle | None" = None,
    character_runtime_assets: "CharacterRuntimeAssets | None" = None,
) -> "OpenWeightResidualRuntime | None":
    """Construct the service-wide shared substrate runtime.

    Returns ``None`` for ``--substrate-mode synthetic`` so the vertical
    factory falls back to its default per-session synthetic runtime
    (which is cheap; no shared state needed).
    """
    if args.substrate_mode == "synthetic":
        _LOG.info("substrate_mode=synthetic; sessions will use per-session synthetic runtimes")
        return None

    if args.substrate_mode == "hf-shared":
        from volvence_zero.substrate import build_transformers_runtime_with_fallback

        profile = (
            resolve_companion_evidence_profile(args.companion_evidence_profile)
            if args.companion_evidence_profile is not None
            else None
        )
        allow_live_mutation = bool(profile is not None and profile.allow_single_session_live_substrate_mutation)

        _LOG.info(
            "substrate_mode=hf-shared; eagerly loading model_id=%s on device=%s (local_files_only=%s)",
            args.substrate_model_id,
            args.substrate_device,
            args.substrate_local_files_only,
        )
        runtime = build_transformers_runtime_with_fallback(
            model_id=args.substrate_model_id,
            model_source=args.substrate_model_source,
            device=args.substrate_device,
            model_dtype=args.substrate_model_dtype,
            local_files_only=args.substrate_local_files_only,
            layer_indices=(
                tuple(args.substrate_layer_indices)
                if args.substrate_layer_indices is not None
                else None
            ),
            activation_width=args.substrate_activation_width,
            max_length=(
                args.substrate_max_length
                if args.substrate_max_length is not None
                else 64
            ),
            fail_on_truncation=(
                (
                    profile is not None
                    and profile.name in MSC_RUNTIME_PROFILE_NAMES
                )
                or args.steering_activation_step is not None
            ),
            expected_model_weights_sha256=(
                args.substrate_expected_weights_sha256
            ),
            # hf-shared means "serve THIS model". A silent builtin-fallback
            # substitute would violate the substrate contract (and poison
            # same-substrate benchmark runs), so load failures must raise.
            fallback_mode="deny",
            # Sharing requires frozen weights; explicit kwargs make the
            # invariant impossible to mis-configure. ``create_app``
            # double-checks via _enforce_frozen_for_sharing.
            allow_live_substrate_mutation=allow_live_mutation,
            common_adapter_bundle=common_adapter_bundle,
            character_prefix_registry=(
                character_runtime_assets.prefix_registry if character_runtime_assets is not None else None
            ),
        )
        _LOG.info(
            "shared substrate ready: model_id=%s runtime_origin=%s dtype=%s",
            getattr(runtime, "model_id", "?"),
            getattr(runtime, "runtime_origin", "?"),
            getattr(runtime, "model_dtype", "unknown"),
        )
        return runtime

    raise ValueError(f"Unknown --substrate-mode {args.substrate_mode!r}")


def _load_steering_bundle(
    path: Path | None,
) -> tuple["SteeringArtifactBundle | None", str | None]:
    if path is None:
        return None, None
    from volvence_zero.steering_contracts import SteeringArtifactBundle

    payload = path.read_bytes()
    bundle = SteeringArtifactBundle.from_json(payload.decode("utf-8"))
    return bundle, hashlib.sha256(payload).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.ablation_bundle and args.vertical:
        print("--ablation-bundle cannot be combined with --vertical", file=sys.stderr)
        return 1
    if args.substrate_activation_width < 1:
        print("--substrate-activation-width must be positive", file=sys.stderr)
        return 1
    if args.substrate_max_length is not None and args.substrate_max_length < 1:
        print("--substrate-max-length must be positive", file=sys.stderr)
        return 1
    expected_weights = args.substrate_expected_weights_sha256
    if expected_weights and (
        len(expected_weights) != 64
        or any(character not in "0123456789abcdef" for character in expected_weights)
    ):
        print(
            "--substrate-expected-weights-sha256 must be lowercase SHA-256",
            file=sys.stderr,
        )
        return 1
    overlay_shadow_requested = args.companion_playbook_overlay_mode == "shadow"
    if args.companion_playbook_overlay_path is not None and not overlay_shadow_requested:
        print(
            "--companion-playbook-overlay-path requires "
            "--companion-playbook-overlay-mode shadow",
            file=sys.stderr,
        )
        return 1
    if overlay_shadow_requested:
        overlay_errors = []
        if args.ablation_bundle:
            overlay_errors.append("--ablation-bundle is not allowed")
        if args.vertical not in (None, "companion"):
            overlay_errors.append("--vertical must be companion")
        if args.companion_evidence_profile is not None:
            overlay_errors.append("--companion-evidence-profile is not allowed")
        if overlay_errors:
            print(
                "--companion-playbook-overlay-mode shadow rejected: "
                + "; ".join(overlay_errors),
                file=sys.stderr,
            )
            return 1
    steering_activation_values = (
        args.steering_promotion_manifest,
        args.steering_activation_plan,
        args.steering_activation_step,
    )
    steering_activation_requested = any(
        value is not None for value in steering_activation_values
    ) or args.steering_previous_activation_receipt is not None
    steering_activation_complete = all(
        value is not None for value in steering_activation_values
    )
    if args.companion_evidence_profile is not None:
        evidence_profile_errors = []
        if args.ablation_bundle:
            evidence_profile_errors.append("--ablation-bundle is not allowed")
        if args.vertical not in (None, "companion"):
            evidence_profile_errors.append("--vertical must be companion")
        if not args.alpha_enabled:
            evidence_profile_errors.append("--alpha-enabled is required")
        if args.evidence_root_dir is None:
            evidence_profile_errors.append("--evidence-root-dir is required")
        if not args.allow_evidence_time_override:
            evidence_profile_errors.append("--allow-evidence-time-override is required")
        if args.substrate_mode != "hf-shared":
            evidence_profile_errors.append("--substrate-mode must be hf-shared")
        if not args.substrate_local_files_only:
            evidence_profile_errors.append("--substrate-local-files-only is required")
        profile = resolve_companion_evidence_profile(args.companion_evidence_profile)
        if profile.allow_single_session_live_substrate_mutation and args.max_sessions != 1:
            evidence_profile_errors.append("mutable evidence profile requires --max-sessions 1")
        if profile.name in MSC_RUNTIME_PROFILE_NAMES:
            if args.msc_temporal_n_z is None:
                evidence_profile_errors.append(
                    "MSC runtime collector requires --msc-temporal-n-z"
                )
            if not args.substrate_expected_weights_sha256:
                evidence_profile_errors.append(
                    "MSC runtime collector requires --substrate-expected-weights-sha256"
                )
            if args.substrate_layer_indices is None:
                evidence_profile_errors.append(
                    "MSC runtime collector requires --substrate-layer-indices"
                )
            if args.substrate_max_length is None:
                evidence_profile_errors.append(
                    "MSC runtime collector requires --substrate-max-length"
                )
            if args.max_sessions != 1:
                evidence_profile_errors.append(
                    "MSC runtime collector requires --max-sessions 1"
                )
        if profile.name == MSC_STEERING_SHADOW_COLLECTOR:
            if args.steering_artifact_bundle is None:
                evidence_profile_errors.append(
                    "MSC steering SHADOW collector requires "
                    "--steering-artifact-bundle"
                )
        elif args.steering_artifact_bundle is not None:
            evidence_profile_errors.append(
                "--steering-artifact-bundle is only valid for the MSC "
                "steering SHADOW collector"
            )
        if steering_activation_requested:
            evidence_profile_errors.append(
                "B3 ACTIVE rollout arguments are forbidden in evidence profiles"
            )
        if evidence_profile_errors:
            print(
                "--companion-evidence-profile rejected: " + "; ".join(evidence_profile_errors),
                file=sys.stderr,
            )
            return 1
    elif args.msc_temporal_n_z is not None:
        print(
            "--msc-temporal-n-z requires --companion-evidence-profile "
            f"{MSC_RUNTIME_COLLECTOR}",
            file=sys.stderr,
        )
        return 1
    elif args.steering_artifact_bundle is not None:
        activation_errors = []
        if not steering_activation_complete:
            activation_errors.append(
                "--steering-promotion-manifest, --steering-activation-plan, "
                "and --steering-activation-step are all required"
            )
        if args.ablation_bundle:
            activation_errors.append("--ablation-bundle is not allowed")
        if args.vertical not in (None, "companion"):
            activation_errors.append("--vertical must be companion")
        if args.substrate_mode != "hf-shared":
            activation_errors.append("--substrate-mode must be hf-shared")
        if not args.substrate_local_files_only:
            activation_errors.append("--substrate-local-files-only is required")
        if args.substrate_max_length is None:
            activation_errors.append("--substrate-max-length is required")
        if args.substrate_model_dtype is None:
            activation_errors.append("--substrate-model-dtype is required")
        steering_env_overrides = tuple(
            name
            for name in (
                "VZ_STEERING_SENSOR",
                "VZ_STEERING_EXECUTOR",
                "VZ_STEERING_GATE",
                "VZ_STEERING_SHADOW_HOOK",
                "VZ_STEERING_UNGATED_ACTION",
                "VZ_SEMANTIC_PROPOSAL_CHANNEL",
            )
            if os.environ.get(name, "").strip()
        )
        if steering_env_overrides:
            activation_errors.append(
                "B3-authorized rollout forbids steering/semantic overrides: "
                + ", ".join(steering_env_overrides)
            )
        if activation_errors:
            print(
                "B3 steering rollout rejected: " + "; ".join(activation_errors),
                file=sys.stderr,
            )
            return 1
    elif steering_activation_requested:
        print(
            "B3 steering rollout arguments require --steering-artifact-bundle",
            file=sys.stderr,
        )
        return 1

    try:
        steering_bundle, steering_bundle_sha256 = _load_steering_bundle(
            args.steering_artifact_bundle
        )
    except (OSError, UnicodeDecodeError, TypeError, ValueError) as exc:
        print(f"Failed to load steering artifact bundle: {exc}", file=sys.stderr)
        return 1
    steering_authorization = None
    if steering_bundle is not None and steering_activation_requested:
        if steering_bundle_sha256 is None:  # pragma: no cover - loader invariant
            raise RuntimeError("loaded steering bundle lacks its SHA-256")
        try:
            steering_authorization = load_steering_activation_authorization(
                bundle=steering_bundle,
                bundle_sha256=steering_bundle_sha256,
                promotion_manifest=args.steering_promotion_manifest,
                activation_plan=args.steering_activation_plan,
                rollout_step=args.steering_activation_step,
                substrate_model_id=args.substrate_model_id,
                substrate_expected_weights_sha256=(
                    args.substrate_expected_weights_sha256
                ),
                substrate_layer_indices=tuple(args.substrate_layer_indices or ()),
                substrate_activation_width=args.substrate_activation_width,
                substrate_model_dtype=args.substrate_model_dtype,
                substrate_max_length=args.substrate_max_length,
                previous_activation_receipt=(
                    args.steering_previous_activation_receipt
                ),
            )
        except (OSError, TypeError, ValueError) as exc:
            print(f"Failed to authorize B3 steering rollout: {exc}", file=sys.stderr)
            return 1
        _LOG.info(
            "B3 steering rollout authorized: bundle_id=%s step=%d prefix=%s "
            "manifest_sha256=%s plan_sha256=%s previous_receipt_sha256=%s",
            steering_authorization.candidate_bundle_id,
            steering_authorization.rollout_step,
            ",".join(steering_authorization.eligible_prefix),
            steering_authorization.manifest_sha256,
            steering_authorization.activation_plan_sha256,
            steering_authorization.previous_receipt_sha256,
        )
    if steering_bundle is not None:
        if steering_bundle.reader.model_id != args.substrate_model_id:
            print(
                "steering bundle model_id does not match --substrate-model-id",
                file=sys.stderr,
            )
            return 1
        if (
            steering_bundle.reader.model_weights_sha256
            != args.substrate_expected_weights_sha256
        ):
            print(
                "steering bundle weights do not match "
                "--substrate-expected-weights-sha256",
                file=sys.stderr,
            )
            return 1
        if (
            args.substrate_layer_indices is None
            or steering_bundle.reader.layer_index
            not in args.substrate_layer_indices
        ):
            print(
                "--substrate-layer-indices must include the steering layer",
                file=sys.stderr,
            )
            return 1
        if args.substrate_activation_width != steering_bundle.reader.residual_width:
            print(
                "--substrate-activation-width must match the steering bundle",
                file=sys.stderr,
            )
            return 1

    try:
        discovered = (
            discover_companion_ablation_verticals()
            if args.ablation_bundle
            else discover_verticals(
                companion_evidence_profile=args.companion_evidence_profile,
                companion_evidence_temporal_n_z=args.msc_temporal_n_z,
                companion_steering_bundle=steering_bundle,
                companion_steering_rollout_config=(
                    steering_authorization.rollout_config
                    if steering_authorization is not None
                    else None
                ),
                companion_steering_rollout_max_new_tokens=(
                    steering_authorization.generation_max_new_tokens
                    if steering_authorization is not None
                    else None
                ),
                companion_steering_rollout_temperature=(
                    steering_authorization.generation_temperature
                    if steering_authorization is not None
                    else None
                ),
                companion_steering_semantic_proposal_channel=(
                    steering_authorization.semantic_proposal_channel
                    if steering_authorization is not None
                    else None
                ),
                companion_playbook_overlay_wiring=WiringLevel(
                    args.companion_playbook_overlay_mode
                ),
                companion_playbook_overlay_path=(
                    args.companion_playbook_overlay_path
                ),
            )
        )
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.list_verticals:
        if not discovered:
            print("No verticals available. Install lifeform-domain-emogpt or another vertical.")
            return 1
        for name, spec in discovered.items():
            print(
                f"{name}\ttemporal_bootstrap={spec.has_temporal_bootstrap}\t"
                f"regime_bootstrap={spec.has_regime_bootstrap}"
            )
        return 0

    if not discovered:
        print(
            "No verticals available. Install lifeform-domain-emogpt or another "
            "lifeform-domain-* before running the service.",
            file=sys.stderr,
        )
        return 1

    name = "companion" if args.ablation_bundle else (args.vertical or default_vertical_name())
    if name not in discovered:
        print(
            f"Unknown vertical {name!r}. Available: {sorted(discovered.keys())}",
            file=sys.stderr,
        )
        return 1

    spec = discovered[name]
    idle = args.idle_eviction_seconds if args.idle_eviction_seconds > 0 else None
    try:
        alpha_users = load_alpha_users(args.alpha_users_file)
    except Exception as exc:
        print(f"Failed to load --alpha-users-file: {exc}", file=sys.stderr)
        return 1
    try:
        alpha_config = AlphaServiceConfig(
            enabled=args.alpha_enabled,
            memory_scope_root_dir=args.memory_scope_root_dir,
            evidence_root_dir=args.evidence_root_dir,
            service_version=args.service_version,
            policy_version=args.policy_version,
            alpha_users=alpha_users,
            # D6 (#alpha-reload): remember the source file so the running
            # service can hot-reload the allow-list (endpoint / SIGHUP).
            alpha_users_path=args.alpha_users_file,
            allow_evidence_time_override=args.allow_evidence_time_override,
            relationship_intelligence_enabled=args.relationship_intelligence,
            relationship_outcome_typing_qualification_path=(
                args.relationship_outcome_typing_qualification
            ),
            relationship_training_candidate_root_dir=(
                args.relationship_training_candidate_root_dir
            ),
        )
    except ValueError as exc:
        print(f"Invalid closed-alpha relationship configuration: {exc}", file=sys.stderr)
        return 1
    if args.alpha_enabled and args.memory_scope_root_dir is None:
        print("--alpha-enabled requires --memory-scope-root-dir", file=sys.stderr)
        return 1
    if args.relationship_intelligence and not args.alpha_enabled:
        print("--relationship-intelligence requires --alpha-enabled", file=sys.stderr)
        return 1
    if args.relationship_intelligence and args.evidence_root_dir is None:
        print("--relationship-intelligence requires --evidence-root-dir", file=sys.stderr)
        return 1
    if args.require_alpha_preflight:
        from lifeform_evolution.closed_alpha_preflight import (
            format_closed_alpha_preflight_report,
            run_closed_alpha_preflight,
        )

        preflight_root = args.evidence_root_dir or "artifacts/lifeform_service_alpha"
        report = run_closed_alpha_preflight(
            artifacts_dir=f"{preflight_root}/preflight",
            scope_root_dir=f"{preflight_root}/preflight_scope",
        )
        print(format_closed_alpha_preflight_report(report))
        if not report.passed:
            return 1

    try:
        common_adapter_bundle, character_runtime_assets = _load_admitted_character_stack(args)
        substrate_runtime = _build_shared_substrate(
            args,
            common_adapter_bundle=common_adapter_bundle,
            character_runtime_assets=character_runtime_assets,
        )
    except ModuleNotFoundError as exc:
        print(
            f"--substrate-mode {args.substrate_mode} requires optional deps: {exc}\n"
            f"Install with: pip install 'vz-substrate[hf]'",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Failed to build shared substrate runtime: {exc}", file=sys.stderr)
        return 1

    if args.companion_evidence_profile is not None:
        try:
            attestation_path = write_companion_evidence_profile_attestation(
                output_dir=Path(args.evidence_root_dir),
                profile=resolve_companion_evidence_profile(args.companion_evidence_profile),
                substrate_model_id=args.substrate_model_id,
                substrate_device=args.substrate_device,
                substrate_model_dtype=getattr(
                    substrate_runtime,
                    "model_dtype",
                    "",
                ),
                temporal_n_z=args.msc_temporal_n_z,
                steering_bundle_id=(
                    steering_bundle.bundle_id
                    if steering_bundle is not None
                    else None
                ),
                steering_bundle_sha256=steering_bundle_sha256,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            print(
                f"Failed to attest --companion-evidence-profile: {exc}",
                file=sys.stderr,
            )
            return 1
        _LOG.info(
            "companion evidence profile=%s attestation=%s",
            args.companion_evidence_profile,
            attestation_path,
        )

    if common_adapter_bundle is not None and args.evidence_root_dir is not None:
        try:
            stack_attestation_path = write_character_runtime_stack_attestation(
                output_dir=Path(args.evidence_root_dir),
                common_adapter_bundle=common_adapter_bundle,
                character_runtime_assets=character_runtime_assets,
                substrate_model_id=args.substrate_model_id,
                substrate_device=args.substrate_device,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            print(
                f"Failed to attest admitted character runtime stack: {exc}",
                file=sys.stderr,
            )
            return 1
        _LOG.info(
            "character runtime stack attestation=%s",
            stack_attestation_path,
        )

    protocol_uptake_service = _maybe_build_protocol_uptake_service(args)

    app_kwargs = {
        "max_sessions": args.max_sessions,
        "idle_eviction_seconds": idle,
        "substrate_runtime": substrate_runtime,
        "alpha_config": alpha_config,
        "protocol_uptake_service": protocol_uptake_service,
        "companion_evidence_profile": args.companion_evidence_profile,
        "allow_evidence_single_session_mutation": bool(
            args.companion_evidence_profile is not None
            and resolve_companion_evidence_profile(
                args.companion_evidence_profile
            ).allow_single_session_live_substrate_mutation
        ),
        "character_runtime_assets": character_runtime_assets,
    }
    if args.ablation_bundle:
        app = create_app(
            verticals=discovered,
            default_vertical=name,
            **app_kwargs,
        )
    else:
        app = create_app(
            vertical=spec,
            **app_kwargs,
        )
    if args.enable_openai_compat:
        # Deferred import: keeps lifeform-openai-compat an optional dep
        # (it is in the workspace but not in lifeform-service's pyproject
        # dependencies — the inverse direction is correct).
        try:
            from lifeform_openai_compat import add_openai_routes
        except ImportError as exc:
            print(
                f"--enable-openai-compat requires the lifeform-openai-compat "
                f"wheel: {exc}\n"
                f"Install with: pip install -e packages/lifeform-openai-compat",
                file=sys.stderr,
            )
            return 1
        api_key_env = args.openai_compat_api_key_env.strip()
        if not api_key_env:
            print(
                "--enable-openai-compat requires a non-empty --openai-compat-api-key-env name",
                file=sys.stderr,
            )
            return 1
        api_key = os.environ.get(api_key_env, "").strip()
        if not api_key:
            print(
                f"--enable-openai-compat requires env var {api_key_env} to contain the local OpenAI-compatible API key",
                file=sys.stderr,
            )
            return 1
        add_openai_routes(app, api_keys=(api_key,))
    print(
        (
            f"[lifeform-serve] ablation_bundle={','.join(discovered.keys())}  default_vertical={name}  "
            if args.ablation_bundle
            else (
                f"[lifeform-serve] vertical={spec.name}  "
                f"temporal_bootstrap={spec.has_temporal_bootstrap}  "
                f"regime_bootstrap={spec.has_regime_bootstrap}  "
            )
        )
        + f"substrate_mode={args.substrate_mode}"
        + (f"  alpha_enabled={args.alpha_enabled}" if args.alpha_enabled else "")
        + (f"  openai_compat=on(auth_env={args.openai_compat_api_key_env})" if args.enable_openai_compat else "")
        + (f"  model_id={args.substrate_model_id}" if args.substrate_mode == "hf-shared" else "")
        + (
            f"  evidence_profile={args.companion_evidence_profile}"
            if args.companion_evidence_profile is not None
            else ""
        )
        + (
            f"  companion_playbook_overlay={args.companion_playbook_overlay_mode}"
            if overlay_shadow_requested
            else ""
        )
        + (
            "  steering_rollout="
            f"step-{steering_authorization.rollout_step}:"
            f"{','.join(steering_authorization.eligible_prefix)}"
            if steering_authorization is not None
            else ""
        )
    )
    print(f"[lifeform-serve] listening on http://{args.host}:{args.port}")
    web.run_app(app, host=args.host, port=args.port, print=lambda *_: None)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
