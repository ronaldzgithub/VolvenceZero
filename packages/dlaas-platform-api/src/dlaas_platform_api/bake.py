"""aiohttp routes for the multi-angle bake plane.

Endpoint summary::

    POST /dlaas/v1/bake                  — submit a multi-angle bake run
    GET  /dlaas/v1/bake                  — list runs (tenant-scoped)
    GET  /dlaas/v1/bake/{run_id}         — run + per-angle job states
    GET  /dlaas/v1/bake/{run_id}/events  — SSE progress stream (monitor)
    POST /dlaas/v1/bake/{run_id}/cancel  — request cancellation
    GET  /dlaas/v1/bake/{run_id}/result  — per-angle produced templates

A bake takes shared **raw materials** and fans them out into N
**angles** (author / 诠释者 interpreter / 角色 character). Each angle is
its own job with the proven per-stage progress
(``staging → cleaning → verifying → baking → registering → done``) and
produces one template. The run aggregate rolls the angle states up.

Layering (R8 / R12 / R15): this is platform **orchestration**, not a
second owner of the baked bundle. Each angle runs three explicit seams:
解析 (analysis) via the third-party LLM plane turns raw materials into a
structured profile; 吸收 (absorption) via ``VzBakeAngleCompiler`` (VZ,
never an LLM) compiles that profile into a real figure/character
artifact; register via ``RegistryBakeArtifactRegistrar`` registers the
bundle + mints the template + advances the persona lifecycle to
``pretrained``. The default runner is ``third_party_llm``
(``VZ_BAKE_RUNNER``); ``synthetic`` is the explicit GPU-free CI/dev
fallback. Failures are captured as a FAILED angle with a typed reason —
never a silent success.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.resources
import json
import logging
import os
import pathlib
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from aiohttp import web

from dlaas_platform_contracts import (
    BakeAngle,
    BakeAngleJob,
    BakeAngleKind,
    BakeAngleResult,
    BakeAngleStatus,
    BakeContractError,
    BakeRequest,
    BakeRun,
    BakeRunEvent,
    BakeRunStatus,
    RawMaterial,
    ThirdPartyLlmJsonRequest,
    rollup_run_status,
)
from dlaas_platform_registry import (
    REGISTRY_APP_KEY,
    Registry,
    require_control_plane_or_service,
    require_tenant_auth,
)
from dlaas_platform_api.third_party_llm import (
    build_third_party_llm_config_from_env,
    complete_json_with_config,
)

_LOG = logging.getLogger("dlaas_platform_api.bake")

BAKE_BUNDLE_APP_KEY = "dlaas_bake_bundle"

#: Stages the orchestrator walks each angle through (observable order).
_PROGRESS_STAGES: tuple[BakeAngleStatus, ...] = (
    BakeAngleStatus.STAGING,
    BakeAngleStatus.CLEANING,
    BakeAngleStatus.VERIFYING,
    BakeAngleStatus.BAKING,
    BakeAngleStatus.REGISTERING,
)

#: Verticals each angle kind routes to (for the result/runner).
_FIGURE_ANGLES: frozenset[BakeAngleKind] = frozenset(
    {BakeAngleKind.AUTHOR, BakeAngleKind.INTERPRETER}
)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _stage_dwell_seconds() -> float:
    """Pause between observable stages so SSE clients see progress.

    Kept tiny by default; a real runner does meaningful work per stage
    and this dwell becomes negligible. Configurable for tests.
    """

    raw = os.environ.get("VZ_BAKE_STAGE_DWELL_SEC", "0.05").strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.05


# ---------------------------------------------------------------------------
# Runner: what actually turns one angle into a template
# ---------------------------------------------------------------------------


class BakeAngleRunner(Protocol):
    """Bakes one angle from shared materials into a template pointer."""

    async def run(
        self,
        *,
        run: BakeRun,
        angle: BakeAngle,
        materials: Sequence[RawMaterial],
        shared_profile: Mapping[str, Any],
    ) -> BakeAngleResult: ...


class SyntheticBakeAngleRunner:
    """No-GPU runner: deterministic template/bundle ids per angle.

    Default runner for CI / dev / environments without the figure or
    character compiler. It does not bake real weights; it produces
    stable, content-addressed pointers so the monitor + result + (gated)
    lifecycle can be exercised end-to-end.
    """

    backend_id = "synthetic"

    async def run(
        self,
        *,
        run: BakeRun,
        angle: BakeAngle,
        materials: Sequence[RawMaterial],
        shared_profile: Mapping[str, Any],
    ) -> BakeAngleResult:
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "source_ref": run.source_ref,
                    "corpus_mode": run.corpus_mode,
                    "angle": angle.to_json(),
                    "material_count": len(materials),
                    "shared_profile_keys": sorted(shared_profile.keys()),
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        short = fingerprint[:16]
        is_figure = angle.kind in _FIGURE_ANGLES
        prefix = "figure-bundle" if is_figure else "character-template"
        bundle_id = f"{prefix}:{angle.slug}:{short}"
        template_id = f"tpl_{angle.kind.value}_{angle.slug}_{short[:12]}"
        return BakeAngleResult(
            angle_kind=angle.kind,
            angle_slug=angle.slug,
            template_id=template_id,
            figure_artifact_id=bundle_id if is_figure else "",
            bundle_id=bundle_id,
            lifecycle_stage="pretrained",
            integrity_hash=fingerprint,
        )


# ---------------------------------------------------------------------------
# 解析 (analysis) -> 吸收 (absorption) -> register: three explicit seams
#
# Hard boundary (per product law): the third-party LLM ONLY does 解析 /
# decomposition (turn raw materials = 饲料 into a structured profile). All
# 吸收 / learning (compiling the profile into a figure/character artifact,
# replay/teach, registration, lifecycle) goes through VZ + the control
# plane and NEVER through an LLM. What substrate VZ uses underneath is the
# platform's own config and invisible to this layer.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompiledArtifact:
    """Output of the VZ 吸收 stage for one angle (pre-registration)."""

    angle_kind: BakeAngleKind
    angle_slug: str
    is_figure: bool
    bundle_id: str
    figure_artifact_id: str
    integrity_hash: str
    figure_id: str
    display_name: str
    bundle_root: str


@dataclass(frozen=True)
class RegisteredArtifact:
    """Output of the control-plane register stage."""

    template_id: str
    lifecycle_stage: str
    note: str = ""


class BakeAngleCompiler(Protocol):
    """吸收 / absorption: structured profile -> VZ artifact. No LLM here."""

    def compile(
        self,
        *,
        run: BakeRun,
        angle: BakeAngle,
        profile_json: Mapping[str, Any],
        materials: Sequence[RawMaterial],
    ) -> CompiledArtifact: ...


class VzBakeAngleCompiler:
    """Default 吸收 stage backed by the VZ figure/character compilers.

    This stage MUST NOT call any LLM: the 解析 stage already produced the
    structured profile (the 饲料). Here VZ compiles it into a real
    on-disk artifact. Substrate choice is VZ's own config.
    """

    def compile(
        self,
        *,
        run: BakeRun,
        angle: BakeAngle,
        profile_json: Mapping[str, Any],
        materials: Sequence[RawMaterial],
    ) -> CompiledArtifact:
        if angle.kind in _FIGURE_ANGLES:
            return self._compile_figure(run, angle, profile_json)
        return self._compile_character(run, angle, profile_json, materials)

    def _compile_figure(
        self,
        run: BakeRun,
        angle: BakeAngle,
        profile_json: Mapping[str, Any],
    ) -> CompiledArtifact:
        from lifeform_domain_figure.bundle_io import save_figure_bundle
        from lifeform_domain_figure.compiler import (
            FigureBundleInputs,
            build_figure_artifact_bundle,
        )
        from lifeform_domain_figure.profiles.generic import (
            build_generic_profile_from_json,
        )
        from lifeform_domain_figure.sample_corpus import (
            synthetic_corpus_from_profile,
        )

        profile_payload = _normalise_figure_profile(angle, profile_json)
        profile = build_generic_profile_from_json(
            profile_payload, expected_slug=angle.slug
        )
        envelopes = synthetic_corpus_from_profile(profile)
        bundle = build_figure_artifact_bundle(
            FigureBundleInputs(profile=profile, envelopes=envelopes)
        )
        root = _bundle_root() / "figure-bundles"
        save_figure_bundle(bundle, root_dir=root)
        return CompiledArtifact(
            angle_kind=angle.kind,
            angle_slug=angle.slug,
            is_figure=True,
            bundle_id=bundle.bundle_id,
            figure_artifact_id=bundle.bundle_id,
            integrity_hash=bundle.integrity_hash,
            figure_id=profile.profile_id,
            display_name=profile.figure_name,
            bundle_root=str(root),
        )

    def _compile_character(
        self,
        run: BakeRun,
        angle: BakeAngle,
        profile_json: Mapping[str, Any],
        materials: Sequence[RawMaterial],
    ) -> CompiledArtifact:
        from lifeform_domain_character.lifeform_builder import (
            build_character_lifeform,
        )
        from lifeform_domain_character.template_save import (
            save_lifeform_template,
        )

        profile = _character_profile_from_json(
            angle=angle, run=run, payload=profile_json
        )
        excerpt = "\n\n".join(
            m.text for m in materials if m.kind == "text" and m.text.strip()
        )[:8000]
        build_character_lifeform(
            profile,
            novel_excerpt=excerpt or None,
            rare_heavy_enabled=False,
        )
        template_id = _template_id(run=run, angle=angle)
        saved = save_lifeform_template(
            profile=profile,
            template_id=template_id,
            output_dir=_bundle_root() / "character-templates" / run.run_id,
            overwrite_existing=True,
            preserve_memory=True,
        )
        digest = hashlib.sha256(saved.template_path.read_bytes()).hexdigest()
        return CompiledArtifact(
            angle_kind=angle.kind,
            angle_slug=angle.slug,
            is_figure=False,
            bundle_id=f"character-template:{profile.profile_id}:{digest[:16]}",
            figure_artifact_id="",
            integrity_hash=digest,
            figure_id=profile.profile_id,
            display_name=profile.character_name,
            bundle_root="",
        )


def _bake_register_enabled() -> bool:
    return os.environ.get("VZ_BAKE_REGISTER", "1").strip() not in ("0", "false", "False")


class BakeArtifactRegistrar(Protocol):
    """Register a compiled artifact through the VZ control plane."""

    async def register(
        self, *, run: BakeRun, angle: BakeAngle, compiled: CompiledArtifact
    ) -> RegisteredArtifact: ...


class NullBakeArtifactRegistrar:
    """No control-plane registration: write-only artifacts (dev/test)."""

    async def register(
        self, *, run: BakeRun, angle: BakeAngle, compiled: CompiledArtifact
    ) -> RegisteredArtifact:
        return RegisteredArtifact(
            template_id=_template_id(run=run, angle=angle),
            lifecycle_stage="",
            note="registration_disabled",
        )


class RegistryBakeArtifactRegistrar:
    """Real registration: figure-bundle store + template mint + lifecycle.

    Still 吸收/control-plane, never LLM. Figure bundles register into the
    runtime ``FigureBundleStore`` (no tenant needed) so wake/bind can
    resolve them. Template mint + persona lifecycle ``pretrained`` are
    tenant-scoped governance; when the run has no tenant (operator bake
    without tenant context) they are deferred (recorded as a note), not
    failed — the bundle is still registered and adoptable.
    """

    def __init__(self, registry: Registry) -> None:
        self._registry = registry

    async def register(
        self, *, run: BakeRun, angle: BakeAngle, compiled: CompiledArtifact
    ) -> RegisteredArtifact:
        template_id = _template_id(run=run, angle=angle)
        if compiled.is_figure and compiled.bundle_root:
            # 吸收 -> runtime store so figure_artifact_id is resolvable.
            from lifeform_service.bundle_root_scanner import (
                scan_and_register_bundles,
            )

            scan_and_register_bundles(
                compiled.bundle_root, figure_id=compiled.figure_id or None
            )
        if not run.tenant_id:
            return RegisteredArtifact(
                template_id=template_id,
                lifecycle_stage="",
                note="bundle_registered; template/lifecycle deferred (no tenant)",
            )
        return await self._mint_and_advance(
            run=run, template_id=template_id, compiled=compiled
        )

    async def _mint_and_advance(
        self, *, run: BakeRun, template_id: str, compiled: CompiledArtifact
    ) -> RegisteredArtifact:
        from dlaas_platform_registry import (
            PersonaLifecycleStore,
            TemplateNotFound,
            TemplateStore,
        )
        from dlaas_platform_registry import (  # typed errors
            PersonaLifecycleConflict,
        )
        from dlaas_platform_registry.tenants import TenantNotFound, TenantStore
        from dlaas_platform_contracts import (
            LifecycleTransitionError,
            PersonaLifecycleStage,
        )

        # Template mint has a tenant_id FK. If the caller passed a tenant
        # that is not provisioned in this registry, do NOT hard-fail the
        # angle — defer template/lifecycle (the figure bundle is already
        # registered and resolvable) and record why. This keeps an
        # operator who set a tenant env without provisioning the tenant
        # row in the same safe state as the no-tenant path.
        try:
            await TenantStore(self._registry).get(run.tenant_id)
        except TenantNotFound:
            return RegisteredArtifact(
                template_id=template_id,
                lifecycle_stage="",
                note=(
                    f"bundle_registered; template/lifecycle deferred "
                    f"(tenant {run.tenant_id!r} not provisioned)"
                ),
            )

        templates = TemplateStore(self._registry)
        lifecycles = PersonaLifecycleStore(self._registry)
        # Persist the bake angle (大师/旁观者/人物) + provenance as a
        # first-class field so a management console can categorize souls
        # without parsing the template_id string. `angle` is the SSOT a
        # consumer reads; template_id prefix remains a fallback.
        bake_meta = {
            "angle": compiled.angle_kind.value,
            "angle_slug": compiled.angle_slug,
            "source_ref": run.source_ref,
            "bake_run_id": run.run_id,
            "app_id": run.app_id,
        }
        try:
            await templates.get(template_id)
        except TemplateNotFound:
            await templates.create(
                tenant_id=run.tenant_id,
                template_name=compiled.display_name or compiled.angle_slug,
                domain="figure" if compiled.is_figure else "character",
                runtime_template_id=run.runtime_template_id,
                figure_artifact_id=compiled.figure_artifact_id,
                template_id=template_id,
                persona_spec={"bake": bake_meta},
            )
        try:
            await lifecycles.create(
                template_id=template_id,
                tenant_id=run.tenant_id,
                display_name=compiled.display_name,
                app_id=run.app_id,
                notes=(
                    f"bake angle={bake_meta['angle']}; "
                    f"source_ref={run.source_ref}; run={run.run_id}"
                ),
                actor="bake",
            )
        except PersonaLifecycleConflict:
            pass
        record = await lifecycles.get_by_template(template_id)
        if record.stage is PersonaLifecycleStage.DRAFT:
            try:
                record = await lifecycles.advance(
                    lifecycle_id=record.lifecycle_id,
                    target=PersonaLifecycleStage.PRETRAINED,
                    evidence={"figure_bundle_id": compiled.bundle_id},
                    actor="bake",
                )
            except LifecycleTransitionError as exc:
                return RegisteredArtifact(
                    template_id=template_id,
                    lifecycle_stage=record.stage.value,
                    note=f"lifecycle_advance_skipped: {exc}",
                )
        return RegisteredArtifact(
            template_id=template_id, lifecycle_stage=record.stage.value
        )


class ThirdPartyLlmBakeAngleRunner:
    """Real runner: 解析 via third-party LLM, 吸收 + register via VZ.

    Pipeline per angle:
      1. 解析  — `_extract_profile` calls the third-party LLM JSON plane
                 to turn raw materials (饲料) into a structured profile.
      2. 吸收  — `BakeAngleCompiler` (VZ) compiles the profile into a real
                 figure/character artifact. No LLM here.
      3. register — `BakeArtifactRegistrar` registers the bundle + mints
                 the template + advances the persona lifecycle.

    The three seams are injectable so tests can exercise the orchestration
    (and the 解析/吸收 boundary) without the heavy VZ stack.
    """

    backend_id = "third_party_llm"

    def __init__(
        self,
        *,
        config: Any | None = None,
        compiler: BakeAngleCompiler | None = None,
        registrar: BakeArtifactRegistrar | None = None,
    ) -> None:
        self._config = config or build_third_party_llm_config_from_env()
        self._compiler = compiler or VzBakeAngleCompiler()
        self._registrar = registrar or NullBakeArtifactRegistrar()

    async def run(
        self,
        *,
        run: BakeRun,
        angle: BakeAngle,
        materials: Sequence[RawMaterial],
        shared_profile: Mapping[str, Any],
    ) -> BakeAngleResult:
        if not materials and not shared_profile:
            raise RuntimeError(
                "third_party_llm runner requires raw_materials or shared_profile"
            )
        # 1. 解析 (LLM): raw materials -> structured profile.
        profile_json = await self._extract_profile(
            run=run,
            angle=angle,
            materials=materials,
            shared_profile=shared_profile,
        )
        # 2. 吸收 (VZ, no LLM): compile profile -> artifact. Off-loop.
        compiled = await asyncio.to_thread(
            self._compiler.compile,
            run=run,
            angle=angle,
            profile_json=profile_json,
            materials=materials,
        )
        # 3. register (control plane): bundle store + template + lifecycle.
        if _bake_register_enabled():
            registered = await self._registrar.register(
                run=run, angle=angle, compiled=compiled
            )
        else:
            registered = RegisteredArtifact(
                template_id=_template_id(run=run, angle=angle),
                lifecycle_stage="",
                note="registration_disabled",
            )
        return BakeAngleResult(
            angle_kind=compiled.angle_kind,
            angle_slug=compiled.angle_slug,
            template_id=registered.template_id,
            figure_artifact_id=compiled.figure_artifact_id,
            bundle_id=compiled.bundle_id,
            lifecycle_stage=registered.lifecycle_stage,
            integrity_hash=compiled.integrity_hash,
        )

    async def _extract_profile(
        self,
        *,
        run: BakeRun,
        angle: BakeAngle,
        materials: Sequence[RawMaterial],
        shared_profile: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        schema_name = _schema_name_for_angle(angle)
        response = await complete_json_with_config(
            config=self._config,
            llm_request=ThirdPartyLlmJsonRequest(
                system_prompt=_read_package_text(
                    "prompts/" + _prompt_name_for_angle(angle)
                ),
                user_prompt=_render_bake_user_prompt(
                    run=run,
                    angle=angle,
                    materials=materials,
                    shared_profile=shared_profile,
                ),
                schema=_read_package_json("schemas/" + schema_name),
                schema_name=schema_name.replace(".schema.json", ""),
                metadata={
                    "dlaas.bake.run_id": run.run_id,
                    "dlaas.bake.angle": angle.angle_id,
                    "dlaas.bake.source_ref": run.source_ref,
                },
            ),
        )
        return response.content


def default_bake_runner(
    *, registrar: BakeArtifactRegistrar | None = None
) -> BakeAngleRunner:
    """Select the runner from ``VZ_BAKE_RUNNER`` (default third_party_llm).

    Production default routes 解析 through the third-party LLM and 吸收 +
    register through VZ. ``synthetic`` is the explicit GPU-free CI/dev
    fallback (deterministic ids, no LLM, no registration). When
    third_party_llm is selected but the provider is unconfigured, each
    angle fails loud with ``third_party_llm_unconfigured``.
    """

    choice = os.environ.get("VZ_BAKE_RUNNER", "third_party_llm").strip().lower()
    if choice == "synthetic":
        return SyntheticBakeAngleRunner()
    if choice in ("", "third_party_llm"):
        return ThirdPartyLlmBakeAngleRunner(registrar=registrar)
    _LOG.warning(
        "VZ_BAKE_RUNNER=%r is not a built-in runner; a deployment must "
        "inject it via attach_bake_routes(runner=...). Falling back to "
        "third_party_llm.",
        choice,
    )
    return ThirdPartyLlmBakeAngleRunner(registrar=registrar)


# ---------------------------------------------------------------------------
# Store + orchestrator
# ---------------------------------------------------------------------------


class InMemoryBakeStore:
    """Process-local store of bake runs, angle jobs and event logs.

    Holds a tiny pub/sub: each run has a set of subscriber queues so the
    SSE endpoint streams live progress without polling.
    """

    def __init__(self) -> None:
        self._runs: dict[str, BakeRun] = {}
        self._jobs: dict[str, list[BakeAngleJob]] = {}
        self._events: dict[str, list[BakeRunEvent]] = {}
        self._subscribers: dict[str, set[asyncio.Queue]] = {}
        self._cancelled: set[str] = set()

    def create_run(
        self, *, run: BakeRun, jobs: Sequence[BakeAngleJob]
    ) -> None:
        self._runs[run.run_id] = run
        self._jobs[run.run_id] = list(jobs)
        self._events[run.run_id] = []
        self._subscribers[run.run_id] = set()

    def get_run(self, run_id: str) -> BakeRun | None:
        return self._runs.get(run_id)

    def list_runs(self, *, tenant_id: str = "") -> tuple[BakeRun, ...]:
        runs = sorted(
            self._runs.values(), key=lambda r: r.created_at_ms, reverse=True
        )
        if tenant_id:
            runs = [r for r in runs if r.tenant_id == tenant_id]
        return tuple(runs)

    def list_jobs(self, run_id: str) -> tuple[BakeAngleJob, ...]:
        return tuple(self._jobs.get(run_id, ()))

    def set_run_status(self, run_id: str, status: BakeRunStatus) -> BakeRun:
        from dataclasses import replace

        run = self._runs[run_id]
        run = replace(run, status=status, updated_at_ms=_now_ms())
        self._runs[run_id] = run
        return run

    def update_job(self, run_id: str, job: BakeAngleJob) -> None:
        jobs = self._jobs.setdefault(run_id, [])
        for idx, existing in enumerate(jobs):
            if existing.angle_job_id == job.angle_job_id:
                jobs[idx] = job
                return
        jobs.append(job)

    def get_result(self, run_id: str) -> tuple[BakeAngleResult, ...]:
        return tuple(
            job.result for job in self._jobs.get(run_id, ()) if job.result
        )

    # --- cancellation -----------------------------------------------------

    def mark_cancelled(self, run_id: str) -> None:
        self._cancelled.add(run_id)

    def is_cancelled(self, run_id: str) -> bool:
        return run_id in self._cancelled

    # --- pub/sub ----------------------------------------------------------

    def append_event(self, event: BakeRunEvent) -> None:
        self._events.setdefault(event.run_id, []).append(event)
        for queue in tuple(self._subscribers.get(event.run_id, ())):
            queue.put_nowait(event)

    def events(self, run_id: str) -> tuple[BakeRunEvent, ...]:
        return tuple(self._events.get(run_id, ()))

    def subscribe(self, run_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(run_id, set()).add(queue)
        return queue

    def unsubscribe(self, run_id: str, queue: asyncio.Queue) -> None:
        subs = self._subscribers.get(run_id)
        if subs and queue in subs:
            subs.discard(queue)

    def close_stream(self, run_id: str) -> None:
        """Signal all subscribers the run is terminal (sentinel ``None``)."""

        for queue in tuple(self._subscribers.get(run_id, ())):
            queue.put_nowait(None)


class BakeRunExecutor:
    """Runs each angle of a submitted bake, emitting progress events."""

    def __init__(
        self,
        *,
        store: InMemoryBakeStore,
        runner: BakeAngleRunner | None = None,
        registrar: BakeArtifactRegistrar | None = None,
    ) -> None:
        self._store = store
        self._runner = runner or default_bake_runner(registrar=registrar)
        self._tasks: set[asyncio.Task] = set()

    @property
    def store(self) -> InMemoryBakeStore:
        return self._store

    def submit(self, run_id: str) -> None:
        task = asyncio.ensure_future(self._process(run_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def stop(self) -> None:
        for task in tuple(self._tasks):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # noqa: BLE001 - shutdown best-effort
                _LOG.warning("bake executor shutdown raised: %s", exc)
        self._tasks.clear()

    def _emit(
        self,
        *,
        run_id: str,
        event_kind: str,
        run_status: BakeRunStatus,
        angle_id: str = "",
        angle_status: BakeAngleStatus | None = None,
        message: str = "",
    ) -> None:
        self._store.append_event(
            BakeRunEvent(
                event_kind=event_kind,
                run_id=run_id,
                run_status=run_status,
                angle_id=angle_id,
                angle_status=angle_status,
                message=message,
                recorded_at_ms=_now_ms(),
            )
        )

    async def _process(self, run_id: str) -> None:
        run = self._store.get_run(run_id)
        if run is None:
            return
        run = self._store.set_run_status(run_id, BakeRunStatus.RUNNING)
        self._emit(
            run_id=run_id,
            event_kind="run",
            run_status=run.status,
            message=f"bake run started ({len(self._store.list_jobs(run_id))} angles)",
        )
        for job in self._store.list_jobs(run_id):
            try:
                await self._process_angle(run_id, job)
            except asyncio.CancelledError:
                self._fail_job(run_id, job, "executor cancelled")
                raise
            except Exception as exc:  # noqa: BLE001 - capture per-angle
                _LOG.exception(
                    "bake angle %s failed in run %s: %s",
                    job.angle.angle_id,
                    run_id,
                    exc,
                )
                self._fail_job(run_id, job, f"unexpected error: {exc}")
        statuses = [j.status for j in self._store.list_jobs(run_id)]
        final = rollup_run_status(statuses)
        run = self._store.set_run_status(run_id, final)
        self._emit(
            run_id=run_id,
            event_kind="done",
            run_status=final,
            message=f"bake run finished: {final.value}",
        )
        self._store.close_stream(run_id)

    async def _process_angle(self, run_id: str, job: BakeAngleJob) -> None:
        run = self._store.get_run(run_id)
        assert run is not None
        if self._store.is_cancelled(run_id):
            cancelled = job.with_status(
                BakeAngleStatus.CANCELLED,
                error="run cancelled before angle started",
                updated_at_ms=_now_ms(),
            )
            self._store.update_job(run_id, cancelled)
            self._emit(
                run_id=run_id,
                event_kind="angle",
                run_status=BakeRunStatus.RUNNING,
                angle_id=job.angle.angle_id,
                angle_status=BakeAngleStatus.CANCELLED,
            )
            return
        dwell = _stage_dwell_seconds()
        current = job
        for stage in _PROGRESS_STAGES:
            if self._store.is_cancelled(run_id):
                self._fail_job(
                    run_id,
                    current,
                    "run cancelled",
                    status=BakeAngleStatus.CANCELLED,
                )
                return
            current = current.with_status(stage, updated_at_ms=_now_ms())
            self._store.update_job(run_id, current)
            self._emit(
                run_id=run_id,
                event_kind="angle",
                run_status=BakeRunStatus.RUNNING,
                angle_id=job.angle.angle_id,
                angle_status=stage,
                message=f"{job.angle.angle_id}: {stage.value}",
            )
            if dwell:
                await asyncio.sleep(dwell)
        materials = run_materials_for(run, job.angle, self._store)
        result = await self._runner.run(
            run=run,
            angle=job.angle,
            materials=materials,
            shared_profile=_run_shared_profile(run, self._store),
        )
        done = current.with_status(
            BakeAngleStatus.DONE,
            result=result,
            updated_at_ms=_now_ms(),
            log_tail=f"baked template {result.template_id}",
        )
        self._store.update_job(run_id, done)
        self._emit(
            run_id=run_id,
            event_kind="angle",
            run_status=BakeRunStatus.RUNNING,
            angle_id=job.angle.angle_id,
            angle_status=BakeAngleStatus.DONE,
            message=f"{job.angle.angle_id}: template {result.template_id}",
        )

    def _fail_job(
        self,
        run_id: str,
        job: BakeAngleJob,
        reason: str,
        *,
        status: BakeAngleStatus = BakeAngleStatus.FAILED,
    ) -> None:
        failed = job.with_status(status, error=reason, updated_at_ms=_now_ms())
        self._store.update_job(run_id, failed)
        self._emit(
            run_id=run_id,
            event_kind="error" if status is BakeAngleStatus.FAILED else "angle",
            run_status=BakeRunStatus.RUNNING,
            angle_id=job.angle.angle_id,
            angle_status=status,
            message=f"{job.angle.angle_id}: {reason}",
        )


# ---------------------------------------------------------------------------
# Real-runner helpers
# ---------------------------------------------------------------------------


def _schema_name_for_angle(angle: BakeAngle) -> str:
    if angle.kind is BakeAngleKind.CHARACTER:
        return "bake_character_profile.schema.json"
    if angle.kind is BakeAngleKind.INTERPRETER:
        return "bake_interpreter_profile.schema.json"
    return "bake_author_profile.schema.json"


def _prompt_name_for_angle(angle: BakeAngle) -> str:
    if angle.kind is BakeAngleKind.CHARACTER:
        return "bake_character_profile.system.md"
    if angle.kind is BakeAngleKind.INTERPRETER:
        return "bake_interpreter_profile.system.md"
    return "bake_author_profile.system.md"


def _read_package_text(relative_path: str) -> str:
    return (
        importlib.resources.files("dlaas_platform_api")
        .joinpath(relative_path)
        .read_text(encoding="utf-8")
    )


def _read_package_json(relative_path: str) -> Mapping[str, Any]:
    raw = (
        importlib.resources.files("dlaas_platform_api")
        .joinpath(relative_path)
        .read_text(encoding="utf-8")
    )
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"{relative_path} must contain a JSON object")
    return data


def _render_bake_user_prompt(
    *,
    run: BakeRun,
    angle: BakeAngle,
    materials: Sequence[RawMaterial],
    shared_profile: Mapping[str, Any],
) -> str:
    material_lines: list[str] = []
    for idx, material in enumerate(materials[:20]):
        if material.kind == "text":
            body = material.text[:4000]
        else:
            body = f"{material.kind}:{material.ref}"
        material_lines.append(
            f"[material {idx:02d} kind={material.kind} media={material.media_kind}]\n{body}"
        )
    return "\n\n".join(
        [
            f"source_ref: {run.source_ref}",
            f"angle: {angle.kind.value}:{angle.slug}",
            f"display_name: {angle.display_name or angle.slug}",
            "angle_config_json:",
            json.dumps(angle.to_json(), ensure_ascii=False, sort_keys=True),
            "shared_profile_json:",
            json.dumps(dict(shared_profile), ensure_ascii=False, sort_keys=True)[:12000],
            "raw_materials:",
            "\n\n".join(material_lines) or "(none)",
        ]
    )


def _normalise_figure_profile(
    angle: BakeAngle, payload: Mapping[str, Any]
) -> dict[str, Any]:
    out = dict(payload)
    out["slug"] = _safe_slug(str(out.get("slug") or angle.slug))
    if out["slug"] != angle.slug:
        out["slug"] = angle.slug
    out["display_name"] = str(
        out.get("display_name") or angle.display_name or angle.slug
    )
    out.setdefault("vocation", "interpreter" if angle.kind is BakeAngleKind.INTERPRETER else "author")
    out.setdefault("domain_coverage_seed", ["source-material", angle.kind.value])
    boundary = out.get("boundary_priors")
    if not isinstance(boundary, dict):
        boundary = {}
    boundary.setdefault(
        "out_of_scope_topics",
        [
            "unsupported claims outside the supplied materials",
            "private facts not present in the corpus",
        ],
    )
    out["boundary_priors"] = boundary
    evidence = out.get("evidence_sources")
    if isinstance(evidence, list):
        normalised: list[dict[str, str]] = []
        for idx, entry in enumerate(evidence):
            if isinstance(entry, dict):
                title = str(entry.get("title") or f"source {idx}")
                provenance = str(entry.get("provenance") or entry.get("summary") or "")
            else:
                title = str(entry)
                provenance = str(entry)
            normalised.append({"title": title, "provenance": provenance})
        out["evidence_sources"] = normalised
    else:
        out["evidence_sources"] = [
            {"title": f"{angle.slug} supplied bake materials", "provenance": "raw_materials"}
        ]
    return out


def _character_profile_from_json(
    *, angle: BakeAngle, run: BakeRun, payload: Mapping[str, Any]
):
    from lifeform_domain_character.profile import (
        CharacterBoundaryPrior,
        CharacterKnowledgeSeed,
        CharacterSoulProfile,
    )

    name = str(
        payload.get("displayName")
        or payload.get("display_name")
        or angle.display_name
        or angle.slug
    )
    description = str(payload.get("description") or f"Character profile for {name}.")
    brief = str(payload.get("brief") or description[:240] or name)
    knowledge = (
        CharacterKnowledgeSeed(
            seed_id=f"{angle.slug}-knowledge-00",
            domain="source-material",
            title=f"{name} source grounding",
            summary=brief,
            snippet=description[:500],
            evidence_locator=run.source_ref,
            confidence=0.7,
            evidence_strength="medium",
            topic_tags=("bake", "character"),
        ),
    )
    boundary = (
        CharacterBoundaryPrior(
            boundary_id=f"{angle.slug}-scope-boundary",
            regime_id=None,
            trigger_reasons=("outside_source_material",),
            answer_depth_limit_hint="refuse_or_clarify",
            clarification_required=True,
            refer_out_required=False,
            blocked_topics=("unsupported private facts",),
            required_disclaimers=("stay within supplied source material",),
            confidence=1.0,
            description=(
                "The character must stay inside supplied materials and "
                "refuse unsupported factual claims."
            ),
        ),
    )
    return CharacterSoulProfile(
        profile_id=_safe_slug(angle.slug),
        character_name=name,
        source_title=run.source_ref,
        version="0.1.0",
        reviewed_by="third-party-llm-bake",
        source_uri=run.source_ref,
        description=description,
        knowledge_seeds=knowledge,
        signature_cases=(),
        strategy_priors=(),
        boundary_priors=boundary,
        drive_priors=(),
        target_contexts=("character-companion", "fictional-roleplay"),
    )


def _safe_slug(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
    out = out.strip("_-")
    return out or "angle"


def _bundle_root() -> pathlib.Path:
    return pathlib.Path(os.environ.get("VZ_BAKE_BUNDLE_ROOT", ".dlaas-bake")).resolve()


def _template_id(*, run: BakeRun, angle: BakeAngle) -> str:
    digest = hashlib.sha256(f"{run.run_id}:{angle.angle_id}".encode("utf-8")).hexdigest()
    return f"tpl_{angle.kind.value}_{_safe_slug(angle.slug)}_{digest[:12]}"


# Request-derived material/profile lookups kept here so the executor can
# access the original request without storing it twice. The request is
# stashed on the bundle when the run is created.


def run_materials_for(
    run: BakeRun, angle: BakeAngle, store: InMemoryBakeStore
) -> tuple[RawMaterial, ...]:
    request = _REQUESTS.get(run.run_id)
    if request is None:
        return ()
    return request.materials_for(angle)


def _run_shared_profile(
    run: BakeRun, store: InMemoryBakeStore
) -> Mapping[str, Any]:
    request = _REQUESTS.get(run.run_id)
    return dict(request.shared_profile) if request else {}


# Process-local map of run_id -> original BakeRequest (raw materials are
# not part of the persisted run aggregate to keep it lean).
_REQUESTS: dict[str, BakeRequest] = {}


class BakeBundle:
    """Container the api wheel reads to dispatch bake state."""

    __slots__ = ("store", "executor")

    def __init__(
        self,
        *,
        runner: BakeAngleRunner | None = None,
        registrar: BakeArtifactRegistrar | None = None,
    ) -> None:
        self.store = InMemoryBakeStore()
        self.executor = BakeRunExecutor(
            store=self.store, runner=runner, registrar=registrar
        )


def attach_bake_routes(
    app: web.Application,
    *,
    registry: Registry,
    runner: BakeAngleRunner | None = None,
    registrar: BakeArtifactRegistrar | None = None,
) -> web.Application:
    if REGISTRY_APP_KEY not in app:
        raise ValueError(
            "attach_bake_routes requires app[REGISTRY_APP_KEY] "
            "(dlaas_platform_api.build_dlaas_app handles this)."
        )
    # Default registrar registers bundles + mints templates + advances the
    # persona lifecycle through the same registry the control plane uses.
    bundle = BakeBundle(
        runner=runner,
        registrar=registrar or RegistryBakeArtifactRegistrar(registry),
    )
    app[BAKE_BUNDLE_APP_KEY] = bundle

    async def _on_cleanup(_app: web.Application) -> None:
        await bundle.executor.stop()

    app.on_cleanup.append(_on_cleanup)

    R = app.router
    R.add_post("/dlaas/v1/bake", _handle_submit)
    R.add_get("/dlaas/v1/bake", _handle_list)
    R.add_get("/dlaas/v1/bake/{run_id}/events", _handle_events)
    R.add_get("/dlaas/v1/bake/{run_id}/result", _handle_result)
    R.add_post("/dlaas/v1/bake/{run_id}/cancel", _handle_cancel)
    R.add_get("/dlaas/v1/bake/{run_id}", _handle_get)
    return app


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def _handle_submit(request: web.Request) -> web.Response:
    actor = await _authorize(request)
    if isinstance(actor, web.Response):
        return actor
    actor_label, actor_tenant_id = actor
    data = await _read_json(request)
    try:
        bake_request = BakeRequest.from_json(data)
    except BakeContractError as exc:
        return _error(400, "invalid_bake_request", str(exc))
    # Tenant credentials are scoped to their own tenant; operators may
    # bake on behalf of an explicit tenant_id in the body.
    tenant_id = bake_request.tenant_id
    if actor_tenant_id:
        if tenant_id and tenant_id != actor_tenant_id:
            return _error(
                403,
                "tenant_mismatch",
                "authenticated tenant cannot bake for another tenant",
            )
        tenant_id = actor_tenant_id
    bundle = _bundle(request)
    run_id, run = submit_bake_run(request.app, bake_request, tenant_id=tenant_id)
    _LOG.info(
        "bake run %s submitted by %s: %d angles from %s",
        run_id,
        actor_label,
        len(bake_request.angles),
        bake_request.source_ref,
    )
    return web.json_response(
        {
            "status": "ok",
            "run_id": run_id,
            "run": run.to_json(jobs=bundle.store.list_jobs(run_id)),
        },
        status=202,
    )


def submit_bake_run(
    app: web.Application,
    bake_request: BakeRequest,
    *,
    tenant_id: str | None = None,
) -> tuple[str, BakeRun]:
    """Create + queue a bake run in-process (the HTTP submit path's core).

    Extracted so other parts of the platform-api workflow (e.g. the
    cultivation induct step) can submit a bake run without an HTTP round
    trip. Requires :func:`attach_bake_routes` to have bound the bake bundle
    to ``app``; raises ``KeyError`` otherwise (a wiring contract violation
    that must fail loudly). ``tenant_id`` overrides the request's tenant
    when given (operator/system callers).
    """

    bundle: BakeBundle = app[BAKE_BUNDLE_APP_KEY]
    run_id = f"bake_{uuid.uuid4().hex[:16]}"
    now = _now_ms()
    run = BakeRun(
        run_id=run_id,
        source_ref=bake_request.source_ref,
        status=BakeRunStatus.QUEUED,
        tenant_id=tenant_id if tenant_id is not None else bake_request.tenant_id,
        app_id=bake_request.app_id,
        corpus_mode=bake_request.corpus_mode,
        runtime_template_id=bake_request.runtime_template_id,
        notes=bake_request.notes,
        created_at_ms=now,
        updated_at_ms=now,
    )
    jobs = [
        BakeAngleJob(
            angle_job_id=f"{run_id}:{angle.angle_id}",
            run_id=run_id,
            angle=angle,
            status=BakeAngleStatus.QUEUED,
            updated_at_ms=now,
        )
        for angle in bake_request.angles
    ]
    bundle.store.create_run(run=run, jobs=jobs)
    _REQUESTS[run_id] = bake_request
    bundle.executor.submit(run_id)
    return run_id, run


async def _handle_list(request: web.Request) -> web.Response:
    actor = await _authorize(request)
    if isinstance(actor, web.Response):
        return actor
    _actor_label, actor_tenant_id = actor
    bundle = _bundle(request)
    runs = bundle.store.list_runs(tenant_id=actor_tenant_id)
    return web.json_response(
        {"status": "ok", "runs": [r.to_json() for r in runs]}
    )


async def _handle_get(request: web.Request) -> web.Response:
    resolved = await _resolve_authorized_run(request)
    if isinstance(resolved, web.Response):
        return resolved
    bundle, run = resolved
    return web.json_response(
        {
            "status": "ok",
            "run": run.to_json(jobs=bundle.store.list_jobs(run.run_id)),
        }
    )


async def _handle_result(request: web.Request) -> web.Response:
    resolved = await _resolve_authorized_run(request)
    if isinstance(resolved, web.Response):
        return resolved
    bundle, run = resolved
    results = bundle.store.get_result(run.run_id)
    return web.json_response(
        {
            "status": "ok",
            "run_id": run.run_id,
            "run_status": run.status.value,
            "templates": [r.to_json() for r in results],
        }
    )


async def _handle_cancel(request: web.Request) -> web.Response:
    resolved = await _resolve_authorized_run(request)
    if isinstance(resolved, web.Response):
        return resolved
    bundle, run = resolved
    if run.status in (
        BakeRunStatus.DONE,
        BakeRunStatus.FAILED,
        BakeRunStatus.PARTIAL,
        BakeRunStatus.CANCELLED,
    ):
        return _error(
            409,
            "run_terminal",
            f"run {run.run_id} is already {run.status.value}",
        )
    bundle.store.mark_cancelled(run.run_id)
    return web.json_response(
        {"status": "ok", "run_id": run.run_id, "cancel_requested": True}
    )


async def _handle_events(request: web.Request) -> web.StreamResponse:
    resolved = await _resolve_authorized_run(request)
    if isinstance(resolved, web.Response):
        return resolved
    bundle, run = resolved
    store = bundle.store
    response = web.StreamResponse(
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
    await response.prepare(request)
    queue = store.subscribe(run.run_id)
    try:
        # Replay backlog so a late subscriber still sees full progress.
        for event in store.events(run.run_id):
            await _write_sse(response, event)
        # If the run already finished before we subscribed, close now.
        current = store.get_run(run.run_id)
        if current is not None and current.status in (
            BakeRunStatus.DONE,
            BakeRunStatus.FAILED,
            BakeRunStatus.PARTIAL,
            BakeRunStatus.CANCELLED,
        ):
            await _write_sse_comment(response, "stream-end")
            return response
        while True:
            event = await queue.get()
            if event is None:
                await _write_sse_comment(response, "stream-end")
                break
            await _write_sse(response, event)
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    finally:
        store.unsubscribe(run.run_id, queue)
    return response


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _bundle(request: web.Request) -> BakeBundle:
    return request.app[BAKE_BUNDLE_APP_KEY]


async def _authorize(
    request: web.Request,
) -> tuple[str, str] | web.Response:
    """Return ``(actor_label, tenant_id)``; tenant_id == "" for operators."""

    headers = request.headers
    if "X-Control-Plane-Secret" in headers or "X-Service-Secret" in headers:
        require_control_plane_or_service(request)
        return "operator", ""
    tenant = await require_tenant_auth(request)
    return f"tenant:{tenant.tenant_id}", tenant.tenant_id


async def _resolve_authorized_run(
    request: web.Request,
) -> tuple[BakeBundle, BakeRun] | web.Response:
    actor = await _authorize(request)
    if isinstance(actor, web.Response):
        return actor
    _actor_label, actor_tenant_id = actor
    bundle = _bundle(request)
    run_id = request.match_info["run_id"]
    run = bundle.store.get_run(run_id)
    if run is None:
        return _error(404, "run_not_found", run_id)
    if actor_tenant_id and run.tenant_id and run.tenant_id != actor_tenant_id:
        return _error(
            403,
            "tenant_mismatch",
            f"run {run_id} is owned by another tenant",
        )
    return bundle, run


async def _write_sse(response: web.StreamResponse, event: BakeRunEvent) -> None:
    payload = json.dumps(event.to_json(), ensure_ascii=False)
    await response.write(
        f"event: {event.event_kind}\ndata: {payload}\n\n".encode("utf-8")
    )


async def _write_sse_comment(response: web.StreamResponse, comment: str) -> None:
    await response.write(f": {comment}\n\n".encode("utf-8"))


async def _read_json(request: web.Request) -> Mapping[str, Any]:
    if not request.body_exists:
        raise _bad_request("missing_body", "Body required")
    text = await request.text()
    if not text.strip():
        raise _bad_request("missing_body", "Empty body")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _bad_request("invalid_json", f"Body is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise _bad_request("invalid_envelope", "Top-level body must be a JSON object")
    return data


def _bad_request(code: str, detail: str) -> web.HTTPBadRequest:
    return web.HTTPBadRequest(
        text=json.dumps({"status": "error", "error": code, "detail": detail}),
        content_type="application/json",
    )


def _error(status: int, code: str, detail: str) -> web.Response:
    return web.json_response(
        {"status": "error", "error": code, "detail": detail}, status=status
    )


__all__ = [
    "BAKE_BUNDLE_APP_KEY",
    "BakeAngleCompiler",
    "BakeAngleRunner",
    "BakeArtifactRegistrar",
    "BakeBundle",
    "BakeRunExecutor",
    "CompiledArtifact",
    "InMemoryBakeStore",
    "NullBakeArtifactRegistrar",
    "RegisteredArtifact",
    "RegistryBakeArtifactRegistrar",
    "SyntheticBakeAngleRunner",
    "ThirdPartyLlmBakeAngleRunner",
    "VzBakeAngleCompiler",
    "submit_bake_run",
    "attach_bake_routes",
    "default_bake_runner",
]
