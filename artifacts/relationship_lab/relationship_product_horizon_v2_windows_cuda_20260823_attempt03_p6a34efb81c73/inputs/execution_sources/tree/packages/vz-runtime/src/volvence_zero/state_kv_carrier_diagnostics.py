"""Mechanism gates for the State-KV prefix carrier (gates A and B).

The identification lane (``state_kv_identification``) asks whether two users
receive different answers. It has come back ``insufficient_data`` for every
carrier so far, and its wrong-user control sits at chance -- which leaves two
very different diagnoses indistinguishable:

* the attention never reads the state slots (a wiring / geometry problem), or
* it reads them and there is no state information inside (a training problem).

These two gates separate those. They are strictly weaker than the
identification claims: passing them licenses "the carrier is live", never "the
model recognises different people", and never anything about context
engineering being unnecessary.

**Why total slot attention is not a gate.** Measured on Qwen2.5-0.5B, a
zero-content prefix draws 0.347 of the final query's attention and a random
prefix 0.339, against a uniform expectation of 0.16 -- near-zero attention
logits absorb mass real tokens do not compete for. A gate on slot mass is
passed by a zero tensor, so mass is recorded as context and asserted on
nowhere. The gates below are calibrated against the random-content control
rather than against constants chosen after seeing the trained artifact.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from volvence_zero.state_kv_identification import ClaimResult, ClaimState

__all__ = [
    "CARRIER_DIAGNOSTICS_SCHEMA_VERSION",
    "PROBE_R2_FLOOR",
    "CarrierDiagnosticsVerdict",
    "evaluate_slot_attention_read",
    "evaluate_state_linearly_readable",
    "build_carrier_diagnostics_verdict",
]

CARRIER_DIAGNOSTICS_SCHEMA_VERSION = "state-kv-carrier-diagnostics.v1"

_CLAIM_ATTENTION = "claim_slot_attention_read"
_CLAIM_READABLE = "claim_state_linearly_readable"

CLAIM_NAMES: tuple[str, ...] = (_CLAIM_ATTENTION, _CLAIM_READABLE)

# Declared before the trained artifact was measured. A held-out mean R^2 of
# 0.1 is deliberately modest: this gate asks whether the state is present at
# all, not whether it is cleanly decodable.
PROBE_R2_FLOOR = 0.1


def evaluate_slot_attention_read(
    *,
    learned_nonuniformity: Sequence[float],
    control_nonuniformity: Sequence[float],
    state_spread: float,
    sentence_spread: float,
) -> ClaimResult:
    """Gate A: the slots are differentiated, and their attention tracks state.

    Two sub-conditions, both relative to a control rather than to a constant:

    * **A2** -- within-prefix ``max/min`` attention must exceed the
      random-content control's, in a majority of layers. A zero-content prefix
      scores exactly 1.0 here (identical slots are indistinguishable), so this
      asks whether the model tells the slots apart at all.
    * **A3** -- the slot-attention profile must vary more across *states* (same
      probe sentence) than across *sentences* (same state). A constant bias
      varies with neither; a channel that merely reacts to the input varies
      only with the latter.
    """

    if len(learned_nonuniformity) != len(control_nonuniformity):
        return ClaimResult(
            name=_CLAIM_ATTENTION,
            state=ClaimState.INSUFFICIENT_DATA,
            detail=(
                "learned and control profiles cover different layer counts "
                f"({len(learned_nonuniformity)} vs "
                f"{len(control_nonuniformity)})"
            ),
        )
    if not learned_nonuniformity:
        return ClaimResult(
            name=_CLAIM_ATTENTION,
            state=ClaimState.INSUFFICIENT_DATA,
            detail="no attention profile recorded",
        )
    layers = len(learned_nonuniformity)
    beaten = [
        index
        for index in range(layers)
        if learned_nonuniformity[index] > control_nonuniformity[index]
    ]
    differentiated = len(beaten) * 2 > layers
    modulated = state_spread > sentence_spread

    detail = (
        f"differentiated in {len(beaten)}/{layers} layers "
        f"(needs >{layers // 2}); state spread {state_spread:.5f} vs "
        f"sentence spread {sentence_spread:.5f}"
    )
    if differentiated and modulated:
        return ClaimResult(
            name=_CLAIM_ATTENTION, state=ClaimState.PASS, detail=detail
        )
    reasons = []
    if not differentiated:
        reasons.append(
            "slots not differentiated beyond the random-content control "
            "(degenerate bias)"
        )
    if not modulated:
        reasons.append(
            "slot attention tracks the probe sentence at least as much as the "
            "state"
        )
    return ClaimResult(
        name=_CLAIM_ATTENTION,
        state=ClaimState.FAIL,
        detail=f"{'; '.join(reasons)}: {detail}",
    )


def evaluate_state_linearly_readable(
    *,
    held_out_r2: Mapping[int, float],
    shuffled_r2: Mapping[int, float],
    control_r2: Mapping[int, float],
    control_hidden_identical: bool,
) -> ClaimResult:
    """Gate B: the readout is linearly recoverable from the real tokens.

    The probe is fitted on the hidden state of *prompt* positions after
    prefill, never on the prefix tensors -- those are a deterministic function
    of the readout, so probing them would recover the generator, not the
    carrier.

    Three sub-conditions:

    * **B1** -- held-out-state mean ``R^2`` clears ``PROBE_R2_FLOOR`` in at
      least one layer.
    * **B2** -- the held-out score must clear the *shuffled-label null
      ceiling* by ``PROBE_R2_FLOOR``. An earlier form of this gate required the
      shuffled control to be ``<= 0`` and was wrong: the reported statistic is
      a maximum over layers (and over shuffle draws), and the maximum of a
      finite-sample ``R^2`` null is positively biased even when no signal
      exists. Measured on this substrate the null ceiling sits near 0.11 while
      the true-label score reaches 0.86, so comparing against zero would have
      failed the gate for a reason that has nothing to do with the carrier.
    * **B3** -- the same fit on the no-prefix control does not. This one is
      structural: the pure arms send byte-identical prompts, so without a
      prefix every state produces the *same* hidden state and no probe can do
      better than the mean. ``control_hidden_identical`` asserts that identity
      held, which is what makes B3 a check on the experiment rather than on
      the probe.
    """

    if not held_out_r2:
        return ClaimResult(
            name=_CLAIM_READABLE,
            state=ClaimState.INSUFFICIENT_DATA,
            detail="no probe fitted",
        )
    if not control_hidden_identical:
        # If the no-prefix hidden states differ across users, something other
        # than the prefix is carrying state and the probe cannot attribute.
        return ClaimResult(
            name=_CLAIM_READABLE,
            state=ClaimState.INSUFFICIENT_DATA,
            detail=(
                "no-prefix hidden states differ across states; the pure arm is "
                "not carrier-isolated and the probe cannot attribute"
            ),
        )
    best_layer = max(held_out_r2, key=lambda layer: held_out_r2[layer])
    best = held_out_r2[best_layer]
    shuffled_ceiling = max(shuffled_r2.values()) if shuffled_r2 else 0.0
    control_ceiling = max(control_r2.values()) if control_r2 else 0.0

    detail = (
        f"best held-out mean R2 {best:.4f} at layer {best_layer} "
        f"(floor {PROBE_R2_FLOOR}); shuffled-label null ceiling "
        f"{shuffled_ceiling:.4f}; no-prefix control ceiling "
        f"{control_ceiling:.4f}"
    )
    reasons = []
    if best <= PROBE_R2_FLOOR:
        reasons.append("held-out R2 does not clear the floor")
    if best - shuffled_ceiling < PROBE_R2_FLOOR:
        reasons.append(
            "held-out R2 does not clear the shuffled-label null ceiling"
        )
    if best - control_ceiling < PROBE_R2_FLOOR:
        reasons.append(
            "held-out R2 does not clear the no-prefix control (not the prefix)"
        )
    if reasons:
        return ClaimResult(
            name=_CLAIM_READABLE,
            state=ClaimState.FAIL,
            detail=f"{'; '.join(reasons)}: {detail}",
        )
    return ClaimResult(
        name=_CLAIM_READABLE, state=ClaimState.PASS, detail=detail
    )


@dataclass(frozen=True)
class CarrierDiagnosticsVerdict:
    """The artifact: two gate states plus the measurements behind them."""

    schema_version: str
    substrate_fingerprint: str
    prefix_artifact_id: str
    claims: tuple[ClaimResult, ...]
    slot_mass_report: Mapping[str, Sequence[float]]
    nonuniformity_report: Mapping[str, Sequence[float]]
    probe_report: Mapping[str, Mapping[int, float]]
    notes: tuple[str, ...] = ()

    def claim(self, name: str) -> ClaimResult:
        for result in self.claims:
            if result.name == name:
                return result
        raise KeyError(f"verdict carries no claim named {name!r}")

    @property
    def carrier_is_live(self) -> bool:
        return all(claim.state is ClaimState.PASS for claim in self.claims)

    def as_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "substrate_fingerprint": self.substrate_fingerprint,
            "prefix_artifact_id": self.prefix_artifact_id,
            "carrier_is_live": self.carrier_is_live,
            "claims": [claim.as_json_dict() for claim in self.claims],
            "slot_mass_report": {
                key: [round(v, 6) for v in values]
                for key, values in self.slot_mass_report.items()
            },
            "nonuniformity_report": {
                key: [round(v, 6) for v in values]
                for key, values in self.nonuniformity_report.items()
            },
            "probe_report": {
                key: {str(layer): round(value, 6) for layer, value in fits.items()}
                for key, fits in self.probe_report.items()
            },
            "notes": list(self.notes),
        }

    def to_json(self) -> str:
        return json.dumps(self.as_json_dict(), ensure_ascii=False, indent=2)


def build_carrier_diagnostics_verdict(
    *,
    substrate_fingerprint: str,
    prefix_artifact_id: str,
    attention_claim: ClaimResult,
    readable_claim: ClaimResult,
    slot_mass_report: Mapping[str, Sequence[float]],
    nonuniformity_report: Mapping[str, Sequence[float]],
    probe_report: Mapping[str, Mapping[int, float]],
) -> CarrierDiagnosticsVerdict:
    """Assemble the verdict, with the anti-overclaim notes attached."""

    if not substrate_fingerprint:
        raise ValueError(
            "carrier diagnostics require a substrate fingerprint: without it "
            "the measurements cannot be tied to one frozen substrate."
        )
    if not prefix_artifact_id:
        raise ValueError(
            "carrier diagnostics require the prefix artifact id: an "
            "unattributed measurement cannot be re-checked."
        )
    notes = [
        "Slot attention mass is reported, never asserted on: a zero-content "
        "prefix draws more than the uniform expectation on this substrate.",
        "These gates are mechanism-level. Passing both licenses 'the carrier "
        "is live', not identification, and not any claim about context "
        "engineering being unnecessary.",
    ]
    return CarrierDiagnosticsVerdict(
        schema_version=CARRIER_DIAGNOSTICS_SCHEMA_VERSION,
        substrate_fingerprint=substrate_fingerprint,
        prefix_artifact_id=prefix_artifact_id,
        claims=(attention_claim, readable_claim),
        slot_mass_report=slot_mass_report,
        nonuniformity_report=nonuniformity_report,
        probe_report=probe_report,
        notes=tuple(notes),
    )
