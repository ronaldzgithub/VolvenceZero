"""Governance demo driver utilities.

Standalone scripts that drive a running ``lifeform-service`` with a
``companion_bench``-backed user simulator so the chat UI's Governance
panel has signal to render across multi-session arcs.

The driver intentionally lives under ``scripts/`` rather than inside a
wheel because it spans license boundaries: it consumes both the
Apache 2.0 ``companion-bench`` public API (FSM + LLM utterance backend)
and the proprietary ``lifeform-service`` HTTP routes. ``scripts/`` is
the workspace's neutral application-layer space — wheel-internal code
must NOT import from here.
"""
