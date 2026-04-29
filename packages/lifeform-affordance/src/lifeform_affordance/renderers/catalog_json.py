"""Full-catalog JSON renderer.

Emits a JSON-serialisable dict listing every descriptor with every
field. Intended for management UIs, external dashboards, and audit
snapshots. Unlike ``render_openai_tools`` this does NOT filter by
kind, and unlike ``render_markdown`` this does NOT truncate \u2014 you
get the whole thing.

Output shape:

    {
        "count": 3,
        "affordances": [
            {
                "name": "read_file",
                "kind": "tool",
                "version": "0.1.0",
                ...
                "cost_model": {
                    "latency_class": "fast",
                    "monetary_class": "free",
                    "rate_limit_per_minute": null,
                },
                "safety_model": {
                    "requires_user_confirmation": false,
                    ...
                },
            },
            ...
        ],
    }

Stable: sorts by registration order (matching ``descriptors``
iteration order) so the catalog is reproducible.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from volvence_zero.affordance import AffordanceDescriptor


def render_catalog_json(
    descriptors: Iterable[AffordanceDescriptor],
    *,
    include_excluded_from_runtime_selection: bool = True,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for d in descriptors:
        if (
            not include_excluded_from_runtime_selection
            and d.excluded_from_runtime_selection
        ):
            continue
        records.append(
            {
                "name": d.name,
                "kind": d.kind.value,
                "version": d.version,
                "display_name": d.display_name,
                "description": d.description,
                "when_to_use": d.when_to_use,
                "when_not_to_use": d.when_not_to_use,
                "parameters_schema": dict(d.parameters_schema),
                "output_schema": dict(d.output_schema),
                "cost_model": {
                    "latency_class": d.cost_model.latency_class.value,
                    "monetary_class": d.cost_model.monetary_class.value,
                    "rate_limit_per_minute": d.cost_model.rate_limit_per_minute,
                },
                "safety_model": {
                    "requires_user_confirmation": d.safety_model.requires_user_confirmation,
                    "irreversible": d.safety_model.irreversible,
                    "requires_consent_grant": list(d.safety_model.requires_consent_grant),
                    "blocked_in_regimes": list(d.safety_model.blocked_in_regimes),
                    "audit_required": d.safety_model.audit_required,
                },
                "preconditions": list(d.preconditions),
                "affordance_tags": list(d.affordance_tags),
                "examples": list(d.examples),
                "source_path": d.source_path,
                "excluded_from_runtime_selection": d.excluded_from_runtime_selection,
            }
        )
    return {"count": len(records), "affordances": records}


__all__ = ["render_catalog_json"]
