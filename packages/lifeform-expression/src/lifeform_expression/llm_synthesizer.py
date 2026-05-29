"""LLM-backed grounded synthesizer.

Composes:

* The kernel's ``LLMResponseSynthesizer`` for the actual generation call
  (it already plumbs system prompt + chat messages + decoding constraints
  + control codes through the substrate runtime).
* Our ``PromptPlanner`` to add a structured plan on top, attached to the
  rationale field for downstream auditing.

This is the "real LLM" path of the lifeform expression layer. The plain
``GroundedResponseSynthesizer`` (in ``response_synthesizer.py``) is the
deterministic / template path used by the synthetic substrate and most
tests; this class is what a product wires up when the brain runs against
an actual HF runtime.

Why subclass ``LLMResponseSynthesizer`` rather than rebuild the prompt
plumbing here:

* The kernel already has ``build_system_prompt`` / ``build_chat_messages``
  that consume the assembly + context. Re-implementing that in the
  lifeform layer would duplicate kernel logic and become a hidden second
  owner of prompt assembly (R8 violation).
* Our value-add is the **plan** — section ordering / question budget /
  intent — exposed in a structured rationale tail. The plan is also
  available as ``synthesize.last_plan`` for product code that wants to
  surface the audit trail (e.g. evaluation, dashboards).

Construct with ``LifeformLLMResponseSynthesizer(runtime=...)`` and pass it
to ``Lifeform(response_synthesizer=...)``.
"""

from __future__ import annotations

import contextlib
from collections import deque
from dataclasses import replace
from typing import Any

from volvence_zero.agent.response import (
    AgentResponse,
    LLMResponseSynthesizer,
    ResponseContext,
)
from volvence_zero.application.runtime import ResponseAssemblySnapshot

from lifeform_expression.grounded_decoder import (
    GroundedDecoder,
    GroundedDecoderConfig,
)
from lifeform_expression.prompt_planner import PromptPlan, PromptPlanner
from lifeform_expression.scope_refuser import (
    CoveragePolicy,
    ScopeRefuser,
    ScopeRefuserConfig,
)
from lifeform_expression.style_prior_injector import StylePriorInjector


# Default conversation history budget. Six turns ≈ 12 chat messages,
# which is small enough not to blow up the substrate context window
# but large enough to give a 1-3B base model the raw cues a short
# follow-up like "然后呢" or a single-token user utterance needs.
_DEFAULT_HISTORY_TURNS = 6
# Per-message character cap used when injecting prior turns. Long
# assistant turns (e.g. structured replies) get trimmed in the middle
# so that the head and tail are both visible to the model.
_HISTORY_MESSAGE_CHAR_BUDGET = 600


class LifeformLLMResponseSynthesizer(LLMResponseSynthesizer):
    """LLM synthesizer that records a structured plan per turn.

    Per-instance state (``_history`` and ``_last_plan``) is **not**
    safe to share across concurrent sessions. ``Lifeform`` clones
    one synthesizer per session via :meth:`clone_for_session` so
    each conversation gets its own ring buffer.

    The optional ``figure_bundle`` parameter binds a frozen
    :class:`lifeform_domain_figure.FigureArtifactBundle` to this
    synthesizer. When set, the figure-vertical L1 / L3 / L4
    enforcement layers (style prior injection, grounded decoding,
    scope refusal) consume the bundle through the duck-typed
    contract surface — the synthesizer itself never imports
    ``lifeform_domain_figure``, so the wheel-boundary direction is
    preserved.
    """

    def __init__(
        self,
        *,
        runtime: Any,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        planner: PromptPlanner | None = None,
        history_turns: int = _DEFAULT_HISTORY_TURNS,
        figure_bundle: object | None = None,
        persona_lora_enabled: bool = True,
        persona_lora_pool: object | None = None,
    ) -> None:
        super().__init__(
            runtime=runtime,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        self._planner = planner or PromptPlanner()
        self._last_plan: PromptPlan | None = None
        self._history_turns = max(0, history_turns)
        self._history: deque[tuple[str, str]] = deque(maxlen=self._history_turns)
        self._figure_bundle = figure_bundle
        # Adapter policy gate (R10 / substrate adapter_policy): when the
        # adopting contract's substrate profile forbids persona LoRA,
        # the SessionManager binds this False so the per-turn activation
        # is skipped even if a record exists in the pool. Defaults True
        # so the standalone / dev path keeps the additive behaviour.
        self._persona_lora_enabled = persona_lora_enabled
        # Per-tenant scoped persona-LoRA pool. When the SessionManager
        # binds a figure bundle for a specific ai_id it passes its own
        # scoped pool so two tenants that adopt different bundles for the
        # same figure_id never collide (the global default pool is
        # last-register-wins). ``None`` falls back to the process-wide
        # default pool, preserving standalone behaviour.
        self._persona_lora_pool = persona_lora_pool

    @property
    def planner(self) -> PromptPlanner:
        return self._planner

    @property
    def last_plan(self) -> PromptPlan | None:
        """The most recent ``PromptPlan`` produced for inspection / audit.

        Returns ``None`` until the first ``synthesize`` call.
        """
        return self._last_plan

    @property
    def history_turns(self) -> int:
        return self._history_turns

    @property
    def figure_bundle(self) -> object | None:
        return self._figure_bundle

    @property
    def persona_lora_enabled(self) -> bool:
        return self._persona_lora_enabled

    @property
    def persona_lora_pool(self) -> object | None:
        return self._persona_lora_pool

    def with_figure_bundle(
        self, bundle: object | None
    ) -> "LifeformLLMResponseSynthesizer":
        """Return a clone bound to ``bundle`` (or unbound when ``None``).

        The figure bundle is shared by reference across the clone —
        it is frozen, so this is safe; new sessions get their own
        history ring buffer through :meth:`clone_for_session`.
        """

        clone = self.clone_for_session()
        clone._figure_bundle = bundle  # noqa: SLF001 — internal reassignment
        return clone

    def with_persona_lora_enabled(
        self, enabled: bool
    ) -> "LifeformLLMResponseSynthesizer":
        """Return a clone whose per-turn persona-LoRA gate is ``enabled``.

        Used by the SessionManager to enforce the adopting contract's
        substrate ``adapter_policy`` at the activation site: when the
        policy forbids LoRA, the bound synthesizer never activates an
        adapter even if a record is present in the pool.
        """

        clone = self.clone_for_session()
        clone._persona_lora_enabled = enabled  # noqa: SLF001
        clone._figure_bundle = self._figure_bundle  # noqa: SLF001
        return clone

    def with_persona_lora_pool(
        self, pool: object | None
    ) -> "LifeformLLMResponseSynthesizer":
        """Return a clone whose persona-LoRA activation consults ``pool``.

        The SessionManager passes its per-ai_id scoped pool so persona
        LoRA lookups are isolated per tenant. ``None`` restores the
        process-wide default pool fallback.
        """

        clone = self.clone_for_session()
        clone._persona_lora_pool = pool  # noqa: SLF001
        clone._figure_bundle = self._figure_bundle  # noqa: SLF001
        return clone

    def clone_for_session(self) -> "LifeformLLMResponseSynthesizer":
        """Return a session-local clone with an empty history buffer.

        ``runtime`` and ``planner`` are shared by reference (the
        substrate runtime is intentionally process-wide; the planner
        is stateless across turns). The figure bundle is also shared
        by reference because it is frozen. Only the ring buffer and
        the ``last_plan`` cache are independent so concurrent
        sessions do not see each other's turns.
        """

        clone = type(self)(
            runtime=self._runtime,
            max_new_tokens=self._max_new_tokens,
            temperature=self._temperature,
            planner=self._planner,
            history_turns=self._history_turns,
            figure_bundle=self._figure_bundle,
            persona_lora_enabled=self._persona_lora_enabled,
            persona_lora_pool=self._persona_lora_pool,
        )
        return clone

    def synthesize(
        self,
        *,
        context: ResponseContext,
        assembly: ResponseAssemblySnapshot | None = None,
    ) -> AgentResponse:
        plan = self._planner.plan(context=context, assembly=assembly)
        self._last_plan = plan
        history_turns = self._snapshot_history()
        if history_turns and not context.prior_turns:
            context = replace(context, prior_turns=history_turns)

        bundle = self._figure_bundle
        # L4 (Wave F closure): pre-generation scope check. When the
        # bundle has a coverage_map and the user query is out of
        # scope (and policy is STRICT_REFUSE), short-circuit with
        # the reviewer-curated refusal text and skip the LLM call
        # entirely. SOFT_DISCLAIM emits a disclaimer that we add
        # to the rationale and let the LLM run; PASSTHROUGH leaves
        # everything unchanged.
        scope_directive = _evaluate_scope(
            bundle=bundle, query=context.user_input
        )
        if scope_directive is not None and scope_directive.should_refuse:
            refusal = AgentResponse(
                text=scope_directive.refusal_text,
                regime_id=context.regime_id if hasattr(context, "regime_id") else "",
                abstract_action="refuse_out_of_scope",
                rationale=(
                    f"L4 scope refusal: {scope_directive.coverage_decision}; "
                    f"{scope_directive.rationale}"
                ),
                rationale_tags=("l4_scope_refusal",),
            )
            self._record_turn(
                user_text=context.user_input,
                assistant_text=refusal.text,
            )
            return _attach_plan_rationale(refusal, plan)

        # L1 (Wave F closure): stash style-hint summary as a rationale
        # tag so audit logs can see what voice prior the synthesizer
        # was conditioned on. Real logit-bias plumbing requires a
        # tokenizer adapter at the runtime layer; for now the rationale
        # tag is the load-bearing surface that future runtime extensions
        # (e.g., HF generate logit_bias=...) read.
        style_tag = _style_hint_tag(bundle=bundle)

        # Wave D + Wave F bridge: when the figure_id resolves to a
        # registered persona LoRA in the default pool, activate it
        # over the runtime forward path for the lifetime of the
        # super().synthesize() call. The frozen base is never
        # mutated; the activate context restores the pre-call
        # forward state on exit (R2 + R15).
        with _maybe_activate_persona_lora(
            bundle=bundle,
            runtime=self._runtime,
            enabled=self._persona_lora_enabled,
            pool=self._persona_lora_pool,
        ):
            response = super().synthesize(context=context, assembly=assembly)

        # L3 (Wave F closure): post-generation grounding verify. The
        # decoder reports unsupported assertions; we attach the
        # verdict's rationale to the response audit trail. The
        # synthesizer does NOT mutate the generated text — that's
        # the synthesizer-vs-grounded-decoder boundary: text is the
        # LLM's output; the verdict is metadata. Future packets can
        # opt into hard refusal by raising at this point.
        #
        # U6 (family-memorial enabler): we now also surface the
        # structured ``EvidencePointer`` records (not just the count
        # in the rationale tag) so the OpenAI-compat bridge can write
        # them into an ``event: evidence`` SSE frame. The UI in
        # apps/family-memorial renders each pointer as a clickable
        # citation card. Pointers are JSON-safe dicts here to avoid
        # pulling lifeform-domain-figure types into AgentResponse.
        grounded_tag, evidence_pointers = _grounded_verify(
            bundle=bundle, text=response.text
        )
        scope_disclaimer_tag = _scope_disclaimer_tag(scope_directive)

        self._record_turn(
            user_text=context.user_input,
            assistant_text=response.text,
        )
        enforced_response = _attach_enforcement_tags(
            response,
            tags=tuple(
                tag
                for tag in (style_tag, grounded_tag, scope_disclaimer_tag)
                if tag
            ),
            evidence_pointers=evidence_pointers,
        )
        return _attach_plan_rationale(enforced_response, plan)

    def _snapshot_history(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._history)

    def _record_turn(self, *, user_text: str, assistant_text: str) -> None:
        if self._history_turns <= 0:
            return
        if not user_text.strip() and not assistant_text.strip():
            return
        self._history.append(
            (
                _trim_history_message(user_text),
                _trim_history_message(assistant_text),
            )
        )


def _trim_history_message(text: str) -> str:
    compact = text.strip()
    if len(compact) <= _HISTORY_MESSAGE_CHAR_BUDGET:
        return compact
    head_budget = _HISTORY_MESSAGE_CHAR_BUDGET // 2
    tail_budget = _HISTORY_MESSAGE_CHAR_BUDGET - head_budget - 5
    return f"{compact[:head_budget]}…{compact[-tail_budget:]}"


def _evaluate_scope(*, bundle: Any, query: str):
    """Run the L4 ScopeRefuser when the bundle has a coverage map.

    Returns ``None`` when there is no bundle, the bundle has no
    coverage map, or the query string is empty — keeping the legacy
    behaviour for synthesizers without a figure bundle attached.

    The bundle's ``retrieval_index`` (debt #39 retrieval-augmented
    floor) is injected into the refuser so an in-corpus query that
    misses every static centroid but cosine-matches a real corpus
    chunk gets lifted to IN_DOMAIN instead of being L4-refused — the
    Wave K Einstein "equivalence principle / postulate / theory"
    false-refuse fix. When the bundle has no retrieval index the
    refuser falls back to centroid-only classification, identical to
    the pre-wiring behaviour.
    """

    if bundle is None:
        return None
    coverage_map = getattr(bundle, "coverage_map", None)
    if coverage_map is None or not query.strip():
        return None
    retrieval_index = getattr(bundle, "retrieval_index", None)
    refuser = ScopeRefuser(
        coverage_map,
        config=ScopeRefuserConfig(policy=CoveragePolicy.STRICT_REFUSE),
        retrieval_index=retrieval_index,
    )
    return refuser.evaluate(query=query)


def _style_hint_tag(*, bundle: Any) -> str:
    """Return a single rationale tag summarising the L1 style prior.

    Empty when no bundle / style prior is attached.
    """

    if bundle is None:
        return ""
    style_prior = getattr(bundle, "style_prior", None)
    if style_prior is None:
        return ""
    injector = StylePriorInjector(style_prior)
    hint = injector.style_hint()
    if not hint.top_term_examples:
        return f"l1_style_prior=figure:{hint.figure_id}"
    sample = ",".join(hint.top_term_examples[:3])
    return f"l1_style_prior=figure:{hint.figure_id};terms:{sample}"


def _grounded_verify(
    *, bundle: Any, text: str
) -> tuple[str, tuple[dict, ...]]:
    """Run the L3 GroundedDecoder against the generated text.

    Returns ``(tag, pointer_dicts)``:
      * ``tag`` is a single rationale tag summarising the verdict
        (``passed`` count + unsupported count + evidence count) — same
        string as the legacy ``_grounded_verify_tag`` produced; empty
        when there is no bundle, no retrieval index, or empty text.
      * ``pointer_dicts`` is a tuple of JSON-safe dicts that mirror
        the typed :class:`EvidencePointer` records from
        :meth:`GroundedDecoder.verify_with_pointers` (debt #24 / U6).
        Each entry carries ``raw_locator`` / ``chunk_id`` /
        ``source_envelope_id`` plus structured fields when the
        locator parsed (``locator_kind``, ``document_id``,
        ``paragraph_index``, ``offset_start``, ``offset_end``,
        ``language``, ``sender_id``, ``recipient_id``, ``date_iso``,
        ``venue_id``, ``volume``, ``page``). The dict shape avoids
        pulling lifeform-domain-figure types into the vz-runtime
        ``AgentResponse`` contract (R8 / wheel-boundary).
    """

    if bundle is None or not text.strip():
        return ("", ())
    retrieval_index = getattr(bundle, "retrieval_index", None)
    if retrieval_index is None:
        return ("", ())
    decoder = GroundedDecoder(
        retrieval_index, config=GroundedDecoderConfig()
    )
    verdict, pointers = decoder.verify_with_pointers(text=text)
    tag = (
        f"l3_grounded_verify=passed:{int(verdict.passed)};"
        f"unsupported:{len(verdict.unsupported_assertions)};"
        f"evidence:{len(verdict.evidence_pointers)}"
    )
    pointer_dicts = tuple(_evidence_pointer_to_dict(p) for p in pointers)
    return (tag, pointer_dicts)


def _evidence_pointer_to_dict(pointer: Any) -> dict:
    """Convert an ``EvidencePointer`` to a JSON-safe dict.

    Field-by-field copy via documented public attrs (no
    ``__dict__`` / no ``asdict`` to keep the contract narrow even
    when ``EvidencePointer`` grows new fields upstream — adding a
    field here is an explicit decision).
    """

    return {
        "raw_locator": pointer.raw_locator,
        "chunk_id": pointer.chunk_id,
        "source_envelope_id": pointer.source_envelope_id,
        "locator_kind": pointer.locator_kind,
        "document_id": pointer.document_id,
        "paragraph_index": pointer.paragraph_index,
        "offset_start": pointer.offset_start,
        "offset_end": pointer.offset_end,
        "language": pointer.language,
        "sender_id": pointer.sender_id,
        "recipient_id": pointer.recipient_id,
        "date_iso": pointer.date_iso,
        "venue_id": pointer.venue_id,
        "volume": pointer.volume,
        "page": pointer.page,
        "rendered": pointer.rendered,
    }


def _scope_disclaimer_tag(directive) -> str:
    """Render a rationale tag for soft-disclaim scope verdicts.

    Empty when the directive is None or carries no disclaimers.
    """

    if directive is None:
        return ""
    if directive.should_refuse:
        # Already handled upstream via short-circuit refusal
        return ""
    if not directive.disclaimers:
        return ""
    return (
        f"l4_scope_disclaimer={directive.coverage_decision};"
        f"count:{len(directive.disclaimers)}"
    )


@contextlib.contextmanager
def _maybe_activate_persona_lora(
    *, bundle: Any, runtime: Any, enabled: bool = True, pool: Any = None
):
    """Activate the persona LoRA on ``runtime`` when the bundle pins one.

    Looks up the figure_id in the process-wide default
    :class:`PersonaLoRAPool`; when a record exists AND the runtime
    is :class:`LoRAAwareResidualRuntime`, calls
    ``pool.activate(figure_id, runtime=runtime)`` for the lifetime
    of the context. Falls through (no-op) when:

    * the adopting contract's adapter_policy disabled LoRA (``enabled``
      is False),
    * no bundle attached,
    * bundle has no figure_id,
    * pool has no record for figure_id,
    * runtime does not satisfy the activation Protocol.

    All fallbacks preserve the legacy synthesize() behaviour
    bit-identically — the activation is purely additive.
    """

    if not enabled:
        yield
        return
    if bundle is None:
        yield
        return
    figure_id = getattr(bundle, "figure_id", "") or ""
    if not figure_id:
        yield
        return
    try:
        from volvence_zero.substrate import (
            LoRAAwareResidualRuntime,
            default_persona_lora_pool,
        )
    except ImportError:
        yield
        return
    # A SessionManager-scoped pool (per ai_id) isolates tenants; fall
    # back to the process-wide default pool for the standalone path.
    active_pool = pool if pool is not None else default_persona_lora_pool()
    if not active_pool.has(figure_id):
        yield
        return
    if not isinstance(runtime, LoRAAwareResidualRuntime):
        yield
        return
    with active_pool.activate(figure_id, runtime=runtime):
        yield


def _attach_enforcement_tags(
    response: AgentResponse,
    *,
    tags: tuple[str, ...],
    evidence_pointers: tuple[dict, ...] = (),
) -> AgentResponse:
    """Append L1 / L3 / L4 enforcement tags to a response's rationale tags.

    ``evidence_pointers`` (U6) replaces any pre-existing pointers on
    the response with the freshly-computed structured list from the
    L3 grounded verify. Empty tuple is treated as "no pointers to
    attach"; existing pointers (if any from upstream paths) survive.
    """

    if not tags and not evidence_pointers:
        return response
    merged = list(response.rationale_tags)
    seen = set(merged)
    for tag in tags:
        if tag and tag not in seen:
            merged.append(tag)
            seen.add(tag)
    return AgentResponse(
        text=response.text,
        regime_id=response.regime_id,
        abstract_action=response.abstract_action,
        rationale=response.rationale,
        rationale_tags=tuple(merged),
        evidence_pointers=(
            evidence_pointers if evidence_pointers else response.evidence_pointers
        ),
    )


def _attach_plan_rationale(response: AgentResponse, plan: PromptPlan) -> AgentResponse:
    rationale = response.rationale or ""
    if rationale and not rationale.endswith("."):
        rationale += "."
    plan_tag = (
        f" Plan: intent={plan.intent.value};"
        f" sections={','.join(s.value for s in plan.sections)};"
        f" q={plan.question_budget}."
    )
    merged: list[str] = []
    seen: set[str] = set()
    for tag in tuple(response.rationale_tags) + tuple(plan.rationale_tags):
        if tag and tag not in seen:
            seen.add(tag)
            merged.append(tag)
    plan_summary_tag = (
        "plan="
        f"intent:{plan.intent.value};"
        f"sections:{','.join(s.value for s in plan.sections)};"
        f"q:{plan.question_budget}"
    )
    if plan_summary_tag not in seen:
        merged.append(plan_summary_tag)
    return AgentResponse(
        text=response.text,
        regime_id=response.regime_id,
        abstract_action=response.abstract_action,
        rationale=(rationale + plan_tag).strip(),
        rationale_tags=tuple(merged),
        # U6: preserve structured evidence pointers across rationale
        # merge so the OpenAI-compat bridge can write them to the
        # ``event: evidence`` SSE frame downstream.
        evidence_pointers=response.evidence_pointers,
    )
