"""Wave O.2 — context managers for the three ablation conditions.

Each condition yields a :class:`LifeformLLMResponseSynthesizer`
configured to exercise a specific layer of the figure-vertical
stack:

* :data:`PersonaCondition.RAW` — synthesizer with ``figure_bundle=None``.
  No L1 (style hint) / L3 (grounded decoder) / L4 (scope refusal)
  enforcement. No persona-LoRA activation. Pure base LLM.

* :data:`PersonaCondition.BUNDLE` — synthesizer with
  ``figure_bundle=bundle``. L1 + L3 + L4 enforcement runs but the
  persona LoRA is **temporarily deregistered** from the default
  pool so the synthesizer's auto-activation path falls through.
  The pool record is restored on exit.

* :data:`PersonaCondition.BUNDLE_LORA` — synthesizer with
  ``figure_bundle=bundle``. L1 + L3 + L4 enforcement runs and the
  persona LoRA is **active** (the synthesizer's auto-activate
  path triggers ``pool.activate``). This is the production
  configuration.

Same ``runtime`` is reused across all three conditions to keep
Qwen / tokenizer load cost amortised across the verification run.
The runtime base ``state_dict`` must be byte-identical before /
after each context (R2 frozen base; covered by Wave D contracts).
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

from lifeform_domain_figure.persona_runtime_surface import (
    temporarily_deregister_pool_record,
)
from lifeform_domain_figure.verification.persona.records import PersonaCondition


@contextlib.contextmanager
def with_condition(
    *,
    condition: PersonaCondition,
    runtime: Any,
    bundle: Any,
) -> Iterator[Any]:
    """Yield a synthesizer wired for ``condition``.

    ``bundle`` MUST carry the figure_id whose persona LoRA is
    expected to be in the default pool (for ``BUNDLE_LORA``) or
    expected to be deregistered (for ``BUNDLE``).
    """

    # Imported here so this module stays importable without
    # ``lifeform-expression`` resolved yet at the top-level.
    from lifeform_expression import LifeformLLMResponseSynthesizer

    if condition is PersonaCondition.RAW:
        synth = LifeformLLMResponseSynthesizer(
            runtime=runtime, figure_bundle=None
        )
        yield synth
        return

    figure_id = getattr(bundle, "figure_id", "") or ""
    if not figure_id:
        raise ValueError(
            "with_condition: bundle must carry a non-empty figure_id "
            "for BUNDLE / BUNDLE_LORA conditions"
        )

    if condition is PersonaCondition.BUNDLE:
        with temporarily_deregister_pool_record(figure_id=figure_id):
            synth = LifeformLLMResponseSynthesizer(
                runtime=runtime, figure_bundle=bundle
            )
            yield synth
        return

    if condition is PersonaCondition.BUNDLE_LORA:
        synth = LifeformLLMResponseSynthesizer(
            runtime=runtime, figure_bundle=bundle
        )
        yield synth
        return

    raise ValueError(f"with_condition: unknown condition {condition!r}")

__all__ = ["with_condition"]
