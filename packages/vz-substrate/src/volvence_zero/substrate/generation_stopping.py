"""Generation-time stopping helpers owned by the substrate runtime."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any


def complete_json_object_prefix(text: str) -> str | None:
    """Return the first complete JSON root object prefix, if one exists.

    Leading JSON whitespace is retained. Braces and brackets inside strings do
    not affect nesting, including when a quote or backslash is escaped. A
    structurally closed but invalid object is not accepted.
    """

    root_start = 0
    while root_start < len(text) and text[root_start] in " \t\r\n":
        root_start += 1
    if root_start >= len(text) or text[root_start] != "{":
        return None

    stack: list[str] = []
    in_string = False
    escaped = False
    for index in range(root_start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            expected = "{" if char == "}" else "["
            if not stack or stack.pop() != expected:
                return None
            if not stack:
                candidate = text[: index + 1]
                try:
                    decoded = json.loads(candidate)
                except (TypeError, ValueError):
                    return None
                return candidate if isinstance(decoded, dict) else None
    return None


class CompleteJsonObjectStoppingCriteria:
    """Transformers-compatible criterion for one generated JSON object."""

    def __init__(
        self,
        *,
        prompt_length: int,
        decode_generated_ids: Callable[[Any], str],
    ) -> None:
        self._prompt_length = prompt_length
        self._decode_generated_ids = decode_generated_ids

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
        del scores, kwargs
        if int(input_ids.shape[0]) != 1:
            return False
        generated_ids = input_ids[0, self._prompt_length :]
        generated_text = self._decode_generated_ids(generated_ids)
        return complete_json_object_prefix(generated_text) is not None
