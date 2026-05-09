"""Input normalization for LLM-produced JSON payloads.

Both ``LLMToMProposalRuntime`` and ``LLMCommonGroundProposalRuntime``
ask their text provider for "a JSON array, no markdown" and then call
``json.loads`` on the raw output. In practice, common open-weight chat
models (Qwen 2.5 / Llama 3 / Mistral instruct, etc.) ignore the "no
markdown" hint and wrap their reply in a fenced code block:

    ```json
    [...]
    ```

The strict parser then fails on the leading triple-backtick and the
runtime emits zero typed proposals — even when the model produced a
perfectly valid payload inside the fence. The 2026-05-09 cross-session
probe diagnosed this at scale: 100% of Qwen 1.5B proposals were
fence-wrapped and 100% were rejected as ``parse_error``.

This module exposes one helper, ``strip_code_fence``, that the two
runtime parsers call before ``json.loads``. The helper is conservative:
it removes a single outer fence (with or without an info string), and
leaves any content that does not start with a fence untouched. It is
NOT a markdown parser; it does not handle nested fences or code blocks
that contain literal triple-backticks. Those edge cases are not
observed in current LLM outputs and would risk masking genuine
malformed payloads.

This is **input normalization**, not **keyword-driven logic** (which the
project's no-keyword-matching rule prohibits): the helper does not
choose a behaviour based on the content; it removes a known
serialization wrapper before the strict typed parser runs. The strict
schema validation downstream is unchanged.
"""

from __future__ import annotations


def strip_code_fence(text: str) -> str:
    """Return ``text`` with a single outer triple-backtick fence removed.

    Tolerates an optional info string (e.g. ``json``, ``JSON``) on the
    opening fence and trailing whitespace / newline before the closing
    fence. Returns ``text`` unchanged when it does not start with
    ``"```"`` after stripping outer whitespace.

    The function intentionally does NOT:

    * unwrap nested fences (returns the outermost fence's content as-is);
    * handle alternative fence syntax (``~~~``);
    * strip prose prefixes like ``"Here is the JSON:"`` — those are
      genuine schema violations the strict parser should continue to
      reject so the diagnostic sink can surface them.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    body = stripped[3:]
    newline_index = body.find("\n")
    if newline_index == -1:
        return text
    info = body[:newline_index].strip()
    if info and not info.isalnum():
        return text
    inner = body[newline_index + 1 :]
    closing_index = inner.rfind("```")
    if closing_index == -1:
        return text
    return inner[:closing_index].strip()


__all__ = ["strip_code_fence"]
