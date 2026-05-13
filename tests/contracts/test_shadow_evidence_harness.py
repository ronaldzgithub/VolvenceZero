"""Shadow evidence harness template contract test (architecture-uplift B4).

Verifies that:

- ``scripts/run_shadow_evidence_template.py`` exists and exposes a
  parametrised CLI (``--baseline``, ``--candidate``, ``--case-limit``,
  ``--output-dir``)
- The harness imports its building blocks from
  ``volvence_zero.agent`` (no internal-only imports — must stay on the
  public dialogue benchmark surface).

We do NOT execute the harness here (it spins up the full dialogue
benchmark — too heavy for contract tests). Behavioural validation lives
in dialogue / scripted-dialogue test suites; here we only pin the contract
surface so future refactors do not silently break the SHADOW-evidence
flow.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
HARNESS_PATH = REPO_ROOT / "scripts" / "run_shadow_evidence_template.py"


def test_shadow_evidence_template_script_exists() -> None:
    assert HARNESS_PATH.is_file(), (
        f"B4 deliverable missing: {HARNESS_PATH} should be the parametrised "
        f"shadow-evidence harness template."
    )


def _parse_harness() -> ast.Module:
    return ast.parse(HARNESS_PATH.read_text(encoding="utf-8"), filename=str(HARNESS_PATH))


def test_harness_defines_required_cli_arguments() -> None:
    tree = _parse_harness()
    src = ast.unparse(tree)
    # Quick structural assertions; not a full argparse model — that level
    # of fidelity would be over-engineered for a template-stability check.
    for arg in ("--baseline", "--candidate", "--case-limit", "--output-dir"):
        assert arg in src, f"harness must expose CLI flag {arg}"


def test_harness_imports_only_public_agent_surface() -> None:
    """Public surface contract: harness must consume the dialogue benchmark
    through ``volvence_zero.agent``'s exported names, not private modules."""
    tree = _parse_harness()
    bad_imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith("volvence_zero") and "._" in module:
                bad_imports.append(module)
    assert not bad_imports, (
        "shadow-evidence harness imports private modules:\n"
        + "\n".join(f"  - {m}" for m in bad_imports)
    )


def test_harness_writes_dated_markdown_brief() -> None:
    """Harness emits a markdown brief named ``<candidate>-shadow-evidence-<date>.md``;
    contract test pins the filename pattern so docs stay discoverable."""
    src = HARNESS_PATH.read_text(encoding="utf-8")
    assert "{candidate}-shadow-evidence-{date}.md" in src or (
        "candidate" in src and "shadow-evidence" in src and "date" in src
    ), (
        "harness must produce a dated <candidate>-shadow-evidence-<YYYY-MM-DD>.md brief"
    )


def test_harness_writes_raw_json_artifact() -> None:
    """Per-profile metric_means JSON is the durable raw artifact."""
    src = HARNESS_PATH.read_text(encoding="utf-8")
    assert "per_profile_metric_means" in src
