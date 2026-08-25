"""Temporal-owner Top-K policy over immutable conditioning-bank snapshots."""

from __future__ import annotations

from collections.abc import Sequence

from volvence_zero.conditioning_bank_contracts import (
    ConditioningBankSnapshot,
    ConditioningRouterDecision,
)
from volvence_zero.semantic_embedding import semantic_topic_similarity

TOPK_SEMANTIC_ROUTER_VERSION = "topk-semantic.v1"


def select_conditioning_banks(
    *,
    user_input: str,
    banks: Sequence[ConditioningBankSnapshot],
    k: int,
) -> ConditioningRouterDecision:
    """Score injectable banks and publish a deterministic temporal decision."""

    if k < 1:
        raise ValueError("select_conditioning_banks k must be >= 1.")
    scored: list[tuple[str, float]] = []
    for bank in (item for item in banks if item.is_injectable):
        if not bank.rendered_statement:
            raise ValueError(
                "select_conditioning_banks requires a non-empty "
                "rendered_statement on every injectable candidate; "
                f"{bank.bank_type.value!r} published none."
            )
        relevance = semantic_topic_similarity(
            user_input,
            bank.rendered_statement,
        )
        score = max(
            0.0,
            min(1.0, relevance * bank.confidence * bank.freshness),
        )
        scored.append((bank.bank_type.value, score))
    scored.sort(key=lambda item: item[0])
    ranked = sorted(scored, key=lambda item: (-item[1], item[0]))
    selected = tuple(sorted(bank for bank, _ in ranked[:k]))
    return ConditioningRouterDecision(
        router_version=TOPK_SEMANTIC_ROUTER_VERSION,
        k=k,
        selected_bank_set=selected,
        scores=tuple(scored),
        description=(
            f"Top-{k} semantic routing over {len(scored)} injectable "
            f"candidate(s): selected "
            f"{'+'.join(selected) if selected else 'none'}; "
            + " ".join(f"{bank}={score:.3f}" for bank, score in scored)
        ),
    )


__all__ = [
    "TOPK_SEMANTIC_ROUTER_VERSION",
    "select_conditioning_banks",
]
