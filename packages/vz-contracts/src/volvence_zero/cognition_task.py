"""Immutable request-level contracts for bounded cognition tasks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CognitionTaskKind(str, Enum):
    """The explicitly requested cognitive operation for one turn."""

    CHOOSE_OBSERVABLE_ACTION = "choose_observable_action"


class CognitionRequiredReadout(str, Enum):
    """Owner-published readouts a cognition task must produce."""

    RESPONSE_ACTION_REALIZATION = "response_action_realization"


@dataclass(frozen=True)
class CognitionTaskContract:
    """Typed task intent that is separate from the perceived event text.

    ``user_input`` remains the world/perception evidence consumed by memory
    and CaseMemory.  This contract says what bounded operation must be
    performed over that evidence; it never supplies an action itself.
    """

    kind: CognitionTaskKind
    required_readouts: tuple[CognitionRequiredReadout, ...] = (
        CognitionRequiredReadout.RESPONSE_ACTION_REALIZATION,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.kind, CognitionTaskKind):
            raise TypeError("cognition task kind must be CognitionTaskKind")
        if not isinstance(self.required_readouts, tuple):
            raise TypeError("cognition task required_readouts must be a tuple")
        if any(
            not isinstance(readout, CognitionRequiredReadout)
            for readout in self.required_readouts
        ):
            raise TypeError(
                "cognition task required_readouts must contain "
                "CognitionRequiredReadout values"
            )
        if len(set(self.required_readouts)) != len(self.required_readouts):
            raise ValueError("cognition task required_readouts must be unique")
        if (
            self.kind is CognitionTaskKind.CHOOSE_OBSERVABLE_ACTION
            and CognitionRequiredReadout.RESPONSE_ACTION_REALIZATION
            not in self.required_readouts
        ):
            raise ValueError(
                "choose_observable_action requires the "
                "response_action_realization readout"
            )


class CognitionTaskContractError(RuntimeError):
    """Base error for a cognition task that cannot satisfy its contract."""


class MissingRequiredCognitionReadoutError(CognitionTaskContractError):
    """A task reached expression without its required owner readout."""


__all__ = [
    "CognitionRequiredReadout",
    "CognitionTaskContract",
    "CognitionTaskContractError",
    "CognitionTaskKind",
    "MissingRequiredCognitionReadoutError",
]
