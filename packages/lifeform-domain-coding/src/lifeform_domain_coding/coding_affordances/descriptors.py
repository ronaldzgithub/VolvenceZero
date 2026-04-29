"""Coding vertical affordance descriptors (Gap 1 slice 2b).

Three read-only TOOL-kind descriptors:

* ``read_file`` \u2014 UTF-8 text read of one file under the sandbox.
* ``list_dir`` \u2014 list file + subdirectory names in one directory.
* ``grep`` \u2014 case-sensitive substring search across workspace files.

All three require the ``filesystem_read`` consent grant. All three
are blocked in ``casual_social`` and ``emotional_support`` regimes
because a pair-programmer in those conversational modes should be
listening, not rummaging through code.

Descriptor ``when_to_use`` / ``when_not_to_use`` strings are
purposefully rich (\u2265 50 chars enforced by ``AffordanceDescriptor.__post_init__``)
\u2014 these are the LLM-facing selection hints that turn into
``render_openai_tools`` descriptions at prompt time.
"""

from __future__ import annotations

from lifeform_affordance import (
    AffordanceCost,
    AffordanceDescriptor,
    AffordanceKind,
    AffordanceLatencyClass,
    AffordanceMonetaryClass,
    AffordanceSafety,
)


CONSENT_FILESYSTEM_READ = "filesystem_read"
"""Canonical consent grant name for reading workspace files.

The lifeform host that wires the coding invoker must ensure this
grant is in ``granted_consents`` for a user / tenant before any
of these affordances can actually invoke. Descriptor authors should
prefer this exact string over ad-hoc variants so multiple coding
tools stack on one permission check.
"""


# Regimes that should NOT surface filesystem tools. A pair-programmer
# in an emotionally-charged conversation or light chitchat has no
# business opening files \u2014 the bias here is conservative.
_CODING_BLOCKED_REGIMES: tuple[str, ...] = (
    "casual_social",
    "emotional_support",
    "repair_and_deescalation",
)


_READ_FILE_DESCRIPTOR = AffordanceDescriptor(
    name="read_file",
    kind=AffordanceKind.TOOL,
    version="0.1.0",
    display_name="Read file",
    description="Read a UTF-8 text file from the workspace and return its content.",
    when_to_use=(
        "Use when the lifeform needs the exact text content of a code, "
        "test, or configuration file before reasoning about it. Prefer "
        "this over guessing what's inside the file from memory."
    ),
    when_not_to_use=(
        "Do not use on binary files (PDFs, images, compiled artefacts); "
        "the call will error. Do not use to enumerate large directories "
        "\u2014 that is what list_dir is for. Avoid on files larger than "
        "a few hundred KB to respect the max_bytes budget."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
            },
            "max_bytes": {
                "type": "integer",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    output_schema={
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "truncated": {"type": "boolean"},
            "byte_count": {"type": "integer"},
            "resolved_path": {"type": "string"},
        },
        "required": ["content", "truncated", "byte_count", "resolved_path"],
    },
    cost_model=AffordanceCost(
        latency_class=AffordanceLatencyClass.FAST,
        monetary_class=AffordanceMonetaryClass.FREE,
    ),
    safety_model=AffordanceSafety(
        requires_user_confirmation=False,
        irreversible=False,
        requires_consent_grant=(CONSENT_FILESYSTEM_READ,),
        blocked_in_regimes=_CODING_BLOCKED_REGIMES,
        audit_required=False,
    ),
    preconditions=("scene.is_open",),
    affordance_tags=("read", "filesystem", "code"),
    examples=(
        "read_file(path='packages/lifeform-core/src/lifeform_core/vitals.py')",
        "read_file(path='tests/test_regime_identity.py', max_bytes=16384)",
    ),
    source_path="lifeform_domain_coding.coding_affordances.descriptors:read_file",
)


_LIST_DIR_DESCRIPTOR = AffordanceDescriptor(
    name="list_dir",
    kind=AffordanceKind.TOOL,
    version="0.1.0",
    display_name="List directory",
    description=(
        "List file and subdirectory names in one directory under the workspace."
    ),
    when_to_use=(
        "Use when the lifeform needs to know what files exist under a "
        "given directory before deciding which to read. Good for "
        "orienting in an unfamiliar part of the codebase, for building "
        "a grep scope, or for confirming a file's existence."
    ),
    when_not_to_use=(
        "Do not use to enumerate the entire repository \u2014 set a "
        "sensible subdirectory. Do not use as a substitute for grep "
        "when you already know the substring pattern you want to find."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    output_schema={
        "type": "object",
        "properties": {
            "entries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "kind": {"type": "string"},
                    },
                    "required": ["name", "kind"],
                },
            },
            "resolved_path": {"type": "string"},
        },
        "required": ["entries", "resolved_path"],
    },
    cost_model=AffordanceCost(
        latency_class=AffordanceLatencyClass.INSTANT,
        monetary_class=AffordanceMonetaryClass.FREE,
    ),
    safety_model=AffordanceSafety(
        requires_user_confirmation=False,
        irreversible=False,
        requires_consent_grant=(CONSENT_FILESYSTEM_READ,),
        blocked_in_regimes=_CODING_BLOCKED_REGIMES,
        audit_required=False,
    ),
    preconditions=("scene.is_open",),
    affordance_tags=("list", "filesystem", "code"),
    examples=(
        "list_dir(path='packages/lifeform-core/src/lifeform_core')",
        "list_dir(path='.')",
    ),
    source_path="lifeform_domain_coding.coding_affordances.descriptors:list_dir",
)


_GREP_DESCRIPTOR = AffordanceDescriptor(
    name="grep",
    kind=AffordanceKind.TOOL,
    version="0.1.0",
    display_name="Grep workspace",
    description=(
        "Case-sensitive substring search through workspace text files."
    ),
    when_to_use=(
        "Use when the lifeform needs to locate every occurrence of a "
        "specific symbol, phrase, or code pattern across the workspace "
        "\u2014 e.g. 'where is this function called', 'which tests "
        "reference this fixture'. Cheap enough to run speculatively."
    ),
    when_not_to_use=(
        "Do not use for fuzzy / semantic search \u2014 this is literal "
        "case-sensitive substring match. Do not use for regex patterns; "
        "the pattern is treated as plain text. For large codebases "
        "consider narrowing with ``subdir`` to respect the max-results "
        "budget rather than paging through partial results."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
            },
            "subdir": {
                "type": "string",
            },
            "max_results": {
                "type": "integer",
            },
        },
        "required": ["pattern"],
        "additionalProperties": False,
    },
    output_schema={
        "type": "object",
        "properties": {
            "matches": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "line_number": {"type": "integer"},
                        "line": {"type": "string"},
                    },
                    "required": ["path", "line_number", "line"],
                },
            },
            "truncated": {"type": "boolean"},
            "scanned_files": {"type": "integer"},
        },
        "required": ["matches", "truncated", "scanned_files"],
    },
    cost_model=AffordanceCost(
        latency_class=AffordanceLatencyClass.FAST,
        monetary_class=AffordanceMonetaryClass.FREE,
    ),
    safety_model=AffordanceSafety(
        requires_user_confirmation=False,
        irreversible=False,
        requires_consent_grant=(CONSENT_FILESYSTEM_READ,),
        blocked_in_regimes=_CODING_BLOCKED_REGIMES,
        audit_required=False,
    ),
    preconditions=("scene.is_open",),
    affordance_tags=("search", "filesystem", "code"),
    examples=(
        "grep(pattern='def run_turn', subdir='packages')",
        "grep(pattern='AffordanceRegistry', max_results=200)",
    ),
    source_path="lifeform_domain_coding.coding_affordances.descriptors:grep",
)


CODING_AFFORDANCE_DESCRIPTORS: tuple[AffordanceDescriptor, ...] = (
    _READ_FILE_DESCRIPTOR,
    _LIST_DIR_DESCRIPTOR,
    _GREP_DESCRIPTOR,
)


__all__ = [
    "CODING_AFFORDANCE_DESCRIPTORS",
    "CONSENT_FILESYSTEM_READ",
]
