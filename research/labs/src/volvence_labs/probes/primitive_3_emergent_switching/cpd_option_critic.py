"""P3 Emergent Switching — CPD + Option-Critic probe (stage 1).

Hypothesis: Change-point detection (CPD) on prediction error spikes can
automatically discover option boundaries in a multi-room navigation task,
matching or exceeding hand-coded switching rules.

Cells:
- baseline (no_switching): flat policy, no option structure
- probe_on (cpd_pe): CPD on PE spikes triggers option termination
- probe_off (cpd_reward): CPD on reward shift only (weaker signal)
- counterfactual (oracle_hand_coded): hand-coded room boundaries (upper bound)

Eval: Option boundary F1 vs hand-annotated, reuse rate, switching entropy.
Environment: Synthetic multi-room grid (MiniGrid-style, no gym dependency for CI).

Paper: CPD + Option-Critic (2510.24988).
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from ...framework.probe import (
    BaseProbe,
    GateReport,
    PrimitiveTag,
    ProbeContext,
    ReadoutBundle,
    RunOutcome,
    register_probe,
)
from ...framework.wiring import AblationCell


def _generate_real_trajectory(seed: int, model_id: str = "sshleifer/tiny-gpt2", n_rooms: int = 4) -> dict:
    """Generate multi-room trajectory with real model PE signals.

    Uses model perplexity on sequential text segments as the PE signal.
    Topic shifts between segments simulate room boundaries.
    """
    try:
        from ...framework.runtime import get_model_runtime

        # Text segments representing different "rooms" (topics)
        room_texts = [
            [
                "The kitchen was warm and smelled of fresh bread.",
                "Pots and pans hung from the ceiling rack.",
                "A timer beeped on the counter near the stove.",
            ],
            [
                "The library was quiet except for turning pages.",
                "Tall shelves lined every wall from floor to ceiling.",
                "A reading lamp cast a warm circle of light.",
            ],
            [
                "Rain hammered against the greenhouse glass panels.",
                "Tropical plants thrived in the humid atmosphere.",
                "Water dripped steadily into the collection basin.",
            ],
            [
                "The workshop floor was covered in wood shavings.",
                "Power tools hung neatly on the pegboard wall.",
                "A half-finished chair sat clamped to the bench.",
            ],
            [
                "The server room hummed with cooling fans.",
                "Blinking lights indicated network activity.",
                "Cable bundles ran neatly through overhead trays.",
            ],
            [
                "The garden path wound between flowering bushes.",
                "Bees moved lazily from bloom to bloom.",
                "A stone fountain bubbled at the center.",
            ],
        ]

        rng = np.random.default_rng(seed)
        selected_rooms = rng.choice(len(room_texts), size=min(n_rooms, len(room_texts)), replace=False)

        rt = get_model_runtime(model_id, dtype="fp32")
        rt.load_model()

        all_pe = []
        true_boundaries = []
        position = 0

        for room_idx in selected_rooms:
            sentences = room_texts[room_idx]
            if position > 0:
                true_boundaries.append(position)

            for sent in sentences:
                result = rt.get_logits_for_text(sent, max_length=64)
                logits = result["logits"].numpy()
                input_ids = result["input_ids"].numpy()

                # Per-token cross-entropy (PE signal)
                if len(logits) > 1:
                    shifted_logits = logits[:-1]
                    targets = input_ids[1:]
                    max_l = shifted_logits.max(axis=-1, keepdims=True)
                    exp_l = np.exp(shifted_logits - max_l)
                    probs = exp_l / exp_l.sum(axis=-1, keepdims=True)
                    target_probs = probs[np.arange(len(targets)), targets]
                    ce = -np.log(np.clip(target_probs, 1e-8, 1.0))
                    all_pe.append(float(ce.mean()))
                else:
                    all_pe.append(0.0)
                position += 1

        pe_signal = np.array(all_pe, dtype=np.float32)
        seq_len = len(pe_signal)

        # Generate synthetic rewards aligned with PE (reward dips at boundaries)
        rewards = np.ones(seq_len, dtype=np.float32) * 0.5
        for b in true_boundaries:
            if b < seq_len:
                rewards[b] -= 0.5

        return {
            "pe": pe_signal.tolist(),
            "rewards": rewards.tolist(),
            "true_boundaries": true_boundaries,
            "n_rooms": n_rooms,
            "total_steps": seq_len,
            "steps_per_room": seq_len // max(n_rooms, 1),
            "model_id": model_id,
            "model_sha": rt.model_sha,
            "source": "real",
        }
    except Exception as e:
        result = _generate_multiroom_trajectory(seed=seed, n_rooms=n_rooms)
        result["source"] = "synthetic_fallback"
        result["fallback_reason"] = str(e)
        return result


def _generate_multiroom_trajectory(seed: int, n_rooms: int = 4, steps_per_room: int = 25) -> dict:
    """Generate a synthetic multi-room navigation trajectory.

    Simulates an agent moving through rooms. At room boundaries, the reward
    structure and PE characteristics change abruptly.
    """
    rng = np.random.default_rng(seed)
    total_steps = n_rooms * steps_per_room

    # Generate PE signal: low within rooms, spikes at boundaries
    pe = np.zeros(total_steps, dtype=np.float32)
    rewards = np.zeros(total_steps, dtype=np.float32)
    true_boundaries = []

    for room in range(n_rooms):
        start = room * steps_per_room
        end = start + steps_per_room

        # Within-room PE: low noise
        room_pe_base = rng.uniform(0.1, 0.5)
        pe[start:end] = room_pe_base + rng.standard_normal(steps_per_room).astype(np.float32) * 0.05

        # Room-specific reward structure
        room_reward_base = rng.uniform(-0.5, 1.0)
        rewards[start:end] = room_reward_base + rng.standard_normal(steps_per_room).astype(np.float32) * 0.1

        # Boundary spike (except at start)
        if room > 0:
            boundary = start
            pe[boundary] += rng.uniform(1.5, 3.0)  # PE spike at boundary
            rewards[boundary] += rng.uniform(-1.0, -0.5)  # reward dip at boundary
            true_boundaries.append(boundary)

    return {
        "pe": pe.tolist(),
        "rewards": rewards.tolist(),
        "true_boundaries": true_boundaries,
        "n_rooms": n_rooms,
        "steps_per_room": steps_per_room,
        "total_steps": total_steps,
    }


def _cpd_on_signal(signal: np.ndarray, threshold: float = 2.0, window: int = 5) -> list[int]:
    """Simple change-point detection: z-score spike detection.

    Returns list of detected boundary indices.
    """
    if len(signal) < window * 2:
        return []

    detected = []
    for i in range(window, len(signal) - window):
        left = signal[i - window:i]
        right = signal[i:i + window]
        left_mean = left.mean()
        left_std = max(left.std(), 1e-6)
        z_score = abs(signal[i] - left_mean) / left_std
        if z_score > threshold:
            # Avoid detecting multiple points in same region
            if not detected or (i - detected[-1]) > window:
                detected.append(i)

    return detected


def _boundary_f1(detected: list[int], true_boundaries: list[int], tolerance: int = 3) -> dict[str, float]:
    """Compute F1 score for boundary detection with tolerance window."""
    if not true_boundaries:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not detected:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # True positives: detected within tolerance of a true boundary
    tp = 0
    matched_true = set()
    for d in detected:
        for i, tb in enumerate(true_boundaries):
            if abs(d - tb) <= tolerance and i not in matched_true:
                tp += 1
                matched_true.add(i)
                break

    precision = tp / len(detected) if detected else 0.0
    recall = tp / len(true_boundaries) if true_boundaries else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def _switching_entropy(boundaries: list[int], total_steps: int) -> float:
    """Entropy of inter-switch intervals (higher = more uniform switching)."""
    if len(boundaries) < 2:
        return 0.0
    intervals = np.diff(sorted(boundaries))
    if intervals.sum() == 0:
        return 0.0
    probs = intervals / intervals.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


@register_probe
class CPDOptionCriticProbe(BaseProbe):
    id = "cpd-option-critic-v1"
    hypothesis = (
        "CPD on PE spikes automatically discovers option boundaries in multi-room "
        "navigation, matching or exceeding hand-coded switching rules."
    )
    primitive = PrimitiveTag.P3_EMERGENT_SWITCHING
    r_ids = ("R3",)

    def knobs(self) -> dict[str, list]:
        return {
            "cpd_threshold": [1.5, 2.0, 2.5],
            "cpd_window": [3, 5, 7],
            "use_real_model": [False, True],
            "model_id": ["sshleifer/tiny-gpt2"],
        }

    def default_inputs(self, seed: int) -> Any:
        return _generate_multiroom_trajectory(seed=seed)

    def real_inputs(self, seed: int, knobs: Mapping[str, Any]) -> Any:
        """Generate multi-room trajectory with real model PE signals.

        Uses model perplexity on sequential text segments as the PE signal.
        Room boundaries correspond to topic shifts in the text.
        """
        model_id = knobs.get("model_id", "sshleifer/tiny-gpt2")
        return _generate_real_trajectory(seed=seed, model_id=model_id)

    def run_cell(self, ctx: ProbeContext, knobs: Mapping[str, Any]) -> RunOutcome:
        inputs = ctx.inputs
        pe = np.array(inputs["pe"], dtype=np.float32)
        rewards = np.array(inputs["rewards"], dtype=np.float32)
        true_boundaries = inputs["true_boundaries"]
        total_steps = inputs["total_steps"]

        threshold = knobs.get("cpd_threshold", 2.0)
        window = knobs.get("cpd_window", 5)

        if ctx.cell == AblationCell.BASELINE:
            # No switching: no boundaries detected
            detected = []

        elif ctx.cell == AblationCell.PROBE_ON:
            # CPD on PE spikes
            detected = _cpd_on_signal(pe, threshold=threshold, window=window)

        elif ctx.cell == AblationCell.PROBE_OFF:
            # CPD on reward shift only (weaker signal)
            detected = _cpd_on_signal(rewards, threshold=threshold, window=window)

        elif ctx.cell == AblationCell.COUNTERFACTUAL:
            # Oracle: use true boundaries directly
            detected = list(true_boundaries)

        else:
            raise ValueError(f"unknown cell: {ctx.cell!r}")

        # Evaluate
        f1_metrics = _boundary_f1(detected, true_boundaries, tolerance=3)
        entropy = _switching_entropy(detected, total_steps)
        reuse_rate = 1.0 - len(detected) / max(total_steps, 1)

        readouts = ReadoutBundle(
            metrics={
                "boundary_f1": f1_metrics["f1"],
                "boundary_precision": f1_metrics["precision"],
                "boundary_recall": f1_metrics["recall"],
                "switching_entropy": entropy,
                "reuse_rate": reuse_rate,
                "n_detected": float(len(detected)),
                "n_true": float(len(true_boundaries)),
            },
            artifacts={
                "detected_boundaries": detected,
                "true_boundaries": true_boundaries,
                "f1_detail": f1_metrics,
            },
            tags={
                "cell": ctx.cell.value,
                "seed": ctx.seed,
                "n_rooms": inputs["n_rooms"],
                "cpd_threshold": threshold,
                "cpd_window": window,
            },
        )

        return RunOutcome(
            readouts=readouts,
            output={
                "cell": ctx.cell.value,
                "boundary_f1": f1_metrics["f1"],
                "n_detected": len(detected),
            },
        )

    def gate(self, outcomes: list[RunOutcome]) -> GateReport:
        if not outcomes:
            return GateReport(passed=False, reason="no outcomes", stats={})

        # Gate: cpd_pe (probe_on) should have F1 > 0.5 and beat no_switching (baseline)
        probe_on = [o for o in outcomes if o.readouts.tags.get("cell") == "probe_on"]
        baseline = [o for o in outcomes if o.readouts.tags.get("cell") == "baseline"]

        if not probe_on:
            return GateReport(passed=False, reason="no probe_on outcomes", stats={})

        mean_f1 = sum(o.readouts.metrics["boundary_f1"] for o in probe_on) / len(probe_on)
        b_f1 = sum(o.readouts.metrics["boundary_f1"] for o in baseline) / len(baseline) if baseline else 0.0

        passed = mean_f1 > 0.5 and mean_f1 > b_f1
        return GateReport(
            passed=passed,
            reason=f"CPD PE F1={mean_f1:.3f} vs baseline={b_f1:.3f} (threshold=0.5)",
            stats={"probe_on_f1": mean_f1, "baseline_f1": b_f1},
        )
