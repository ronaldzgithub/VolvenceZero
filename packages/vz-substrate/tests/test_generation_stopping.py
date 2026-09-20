from __future__ import annotations

from volvence_zero.substrate.generation_stopping import (
    CompleteJsonObjectStoppingCriteria,
    complete_json_object_prefix,
)


class _SingleSequenceIds:
    def __init__(self, values: list[int]) -> None:
        self._values = values
        self.shape = (1, len(values))

    def __getitem__(self, key: tuple[int, slice]) -> list[int]:
        row, item_slice = key
        assert row == 0
        return self._values[item_slice]


def test_complete_json_object_prefix_accepts_nested_objects_and_arrays() -> None:
    text = '  {"outer":{"items":[1,{"done":true}]}} trailing'

    assert complete_json_object_prefix(text) == (
        '  {"outer":{"items":[1,{"done":true}]}}'
    )


def test_complete_json_object_prefix_ignores_braces_inside_strings() -> None:
    text = '{"line":"literal } and { plus \\\"quote\\\" and \\\\ slash","ok":true}'

    assert complete_json_object_prefix(text) == text


def test_complete_json_object_prefix_rejects_truncated_json() -> None:
    assert complete_json_object_prefix('{"outer":{"items":[1,2]}') is None


def test_complete_json_object_prefix_requires_object_at_completion_start() -> None:
    assert complete_json_object_prefix('plain text {"later":true}') is None
    assert complete_json_object_prefix('[{"array":true}]') is None


def test_stopping_criteria_waits_for_complete_object_not_truncated_prefix() -> None:
    criterion = CompleteJsonObjectStoppingCriteria(
        prompt_length=2,
        decode_generated_ids=lambda token_ids: "".join(
            chr(token_id) for token_id in token_ids
        ),
    )

    truncated = _SingleSequenceIds([1, 2, *map(ord, '{"nested":{"x":1}')])
    complete = _SingleSequenceIds([1, 2, *map(ord, '{"nested":{"x":1}}')])

    assert criterion(truncated, scores=None) is False
    assert criterion(complete, scores=None) is True
