"""Packet 7.3: APIInjectionUptake — convert a JSON / dict
specification of a behavior protocol into a
``BehaviorProtocolCandidate`` directly (no LLM extraction step).

Use case: an external orchestrator / DLaaS platform passes a
fully-typed protocol spec via REST / RPC; we validate and wrap
it in a candidate so the same R10 ModificationGate review path
applies."""

from __future__ import annotations

from lifeform_protocol_runtime.api_injection_uptake.injection import (
    inject_protocol_from_payload,
)

__all__ = ["inject_protocol_from_payload"]
