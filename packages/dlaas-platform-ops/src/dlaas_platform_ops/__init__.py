"""DLaaS platform-tier operations.

Public exports:

* :class:`PauseStore` — per-session pause state for operator
  takeover. Slice 5.1.
* :func:`evaluate_session` / :class:`HandoffDecision` — handoff
  trigger driven by the kernel's ``rupture_state`` snapshot.
  Slice 5.2.
* :class:`LedgerBroker` / :class:`LedgerEvent` — admin SSE event
  source. Slice 5.3.
* :class:`OpsBundle` / :func:`attach_ops_routes` — aiohttp wiring
  helper that registers every ops endpoint.
* :class:`OutboundScheduler` / :func:`evaluate_outbound` /
  :class:`OutboundCadenceConfig` / :class:`OutboundLedger` /
  :class:`OutboundDecision` / :class:`OutboundFollowupRequest`
  (W3-B) — proactive followup scheduler for LTV / private-domain
  verticals. Reads ``relationship_state`` snapshot through the
  public Brain facade and emits typed DLaaS command requests.
"""

from __future__ import annotations

from dlaas_platform_ops.handoff_trigger import HandoffDecision, evaluate_session
from dlaas_platform_ops.ledger import LedgerBroker, LedgerEvent
from dlaas_platform_ops.outbound_scheduler import (
    OutboundCadenceConfig,
    OutboundDecision,
    OutboundFollowupRequest,
    OutboundLedger,
    OutboundScheduler,
    evaluate_outbound,
)
from dlaas_platform_ops.pause_state import (
    OperatorMessage,
    PauseStore,
    operator_takeover_response_body,
)
from dlaas_platform_ops.routes import (
    OPS_BUNDLE_APP_KEY,
    OpsBundle,
    attach_ops_routes,
)

__all__ = (
    "HandoffDecision",
    "LedgerBroker",
    "LedgerEvent",
    "OPS_BUNDLE_APP_KEY",
    "OperatorMessage",
    "OpsBundle",
    "OutboundCadenceConfig",
    "OutboundDecision",
    "OutboundFollowupRequest",
    "OutboundLedger",
    "OutboundScheduler",
    "PauseStore",
    "attach_ops_routes",
    "evaluate_outbound",
    "evaluate_session",
    "operator_takeover_response_body",
)
