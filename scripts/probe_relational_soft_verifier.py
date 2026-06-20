"""Stage 0 deterministic probe for the relational soft verifier (debt #85).

Spec: ``docs/specs/relational-soft-verifier.md``.

This probe validates **machinery only**, never the central self-confirmation
question (that needs the external human anchor -- debt #51 -- and an LLM
runtime -- debt #10B). Concretely it drives the *real* PE-owner critic head
(``volvence_zero.prediction.error._PECriticHead``) over a scripted
relationship trajectory (rupture -> repair, with noise) inside a single
regime bucket, and reads out, per turn:

- ``r_soft`` : the epistemic (learnable) part of the ``relationship`` axis PE,
  i.e. the soft-verifier reward proposed by the spec (epistemic-only).
- per-source epistemic contribution across the 4 PE axes (composable-verifier
  plumbing + single-source-drift monitoring surface).
- ``rho_self`` : Pearson corr between ``r_soft`` and the verifier's own
  predicted error magnitude (``critic_predicted_magnitude``). High rho_self
  with no external anchor is exactly the self-confirmation risk the central
  experiment must rule out -- here we can only *measure* it, not refute it.

EXIT(0) (debt #85 Stage 0): non-empty JSON + epistemic ratio not identically
zero + rho_self computable. This proves the readout-only plumbing works and
that epistemic separation does not collapse on a structured trajectory.

Run:
    python scripts/probe_relational_soft_verifier.py
Output:
    artifacts/eq_uplift/relational_soft_verifier_shadow.json
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

from volvence_zero.prediction import PredictionActionContext, PredictionError
from volvence_zero.prediction.error import _PECriticHead

REGIME_ID = "companion"
AXES = ("task", "relationship", "regime", "action")
RELATIONSHIP_AXIS_INDEX = 1


@dataclass(frozen=True)
class TurnInput:
    """One scripted turn: signed relationship PE plus the other axes."""

    phase: str
    task: float
    relationship: float
    regime: float
    action: float

    def to_error(self) -> PredictionError:
        mags = (abs(self.task), abs(self.relationship), abs(self.regime), abs(self.action))
        magnitude = sum(mags) / len(mags)
        signed = -abs(self.relationship)
        return PredictionError(
            task_error=self.task,
            relationship_error=self.relationship,
            regime_error=self.regime,
            action_error=self.action,
            magnitude=magnitude,
            signed_reward=signed,
            description=f"probe:{self.phase}",
        )


def _scripted_trajectory() -> list[TurnInput]:
    """A relationship arc the soft verifier should find *learnable*.

    Phase A (stable): low relationship error, mild noise.
    Phase B (rupture): a trust rupture spikes relationship error (surprising).
    Phase C (repair): relationship error decays back down (becomes predictable).
    """

    turns: list[TurnInput] = []
    stable = [0.14, 0.18, 0.12, 0.20, 0.13, 0.17, 0.15, 0.16]
    for i, rel in enumerate(stable):
        turns.append(
            TurnInput(
                phase="stable",
                task=0.30 + 0.10 * ((i % 3) - 1),
                relationship=rel,
                regime=0.10,
                action=0.25 + 0.05 * (i % 2),
            )
        )
    for i, rel in enumerate((0.78, 0.82, 0.70)):
        turns.append(
            TurnInput(
                phase="rupture",
                task=0.30 + 0.10 * ((i % 3) - 1),
                relationship=rel,
                regime=0.12,
                action=0.25 + 0.05 * (i % 2),
            )
        )
    rel = 0.62
    for i in range(9):
        turns.append(
            TurnInput(
                phase="repair",
                task=0.30 + 0.10 * ((i % 3) - 1),
                relationship=round(rel, 4),
                regime=0.11,
                action=0.25 + 0.05 * (i % 2),
            )
        )
        rel = rel * 0.78 + 0.05
    return turns


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 1e-12 or syy <= 1e-12:
        return None
    return sxy / math.sqrt(sxx * syy)


def run_probe() -> dict:
    critic = _PECriticHead(decay=0.9)
    context = PredictionActionContext(regime_id=REGIME_ID, abstract_action_id="companion_turn")
    trajectory = _scripted_trajectory()

    per_turn: list[dict] = []
    r_soft_series: list[float] = []
    critic_pred_series: list[float] = []
    rel_abs_series: list[float] = []
    per_source_epistemic: dict[str, list[float]] = {axis: [] for axis in AXES}

    for turn_index, turn in enumerate(trajectory, start=1):
        decomposition = critic.update(
            error=turn.to_error(),
            action_context=context,
            timestamp_ms=turn_index,
        )
        rel_axis_name, rel_axis_aleatoric, rel_axis_epistemic = decomposition.per_axis[
            RELATIONSHIP_AXIS_INDEX
        ]
        assert rel_axis_name == "relationship", rel_axis_name

        r_soft = rel_axis_epistemic
        r_soft_series.append(r_soft)
        critic_pred_series.append(decomposition.critic_predicted_magnitude)
        rel_abs_series.append(abs(turn.relationship))
        for axis_name, _aleatoric, epistemic in decomposition.per_axis:
            per_source_epistemic[axis_name].append(epistemic)

        per_turn.append(
            {
                "turn": turn_index,
                "phase": turn.phase,
                "relationship_abs_error": round(abs(turn.relationship), 4),
                "r_soft_relationship_epistemic": round(r_soft, 4),
                "relationship_aleatoric": round(rel_axis_aleatoric, 4),
                "agg_epistemic": round(decomposition.epistemic_magnitude, 4),
                "agg_aleatoric": round(decomposition.aleatoric_magnitude, 4),
                "improvement_magnitude": round(decomposition.improvement_magnitude, 4),
                "critic_predicted_magnitude": round(decomposition.critic_predicted_magnitude, 4),
                "critic_gate_decision": decomposition.critic_gate_decision,
            }
        )

    mean_r_soft = sum(r_soft_series) / len(r_soft_series)
    mean_rel_abs = sum(rel_abs_series) / len(rel_abs_series)
    epistemic_ratio = (mean_r_soft / mean_rel_abs) if mean_rel_abs > 1e-9 else 0.0
    rho_self = _pearson(r_soft_series, critic_pred_series)

    per_source_mean_epistemic = {
        axis: round(sum(vals) / len(vals), 4) for axis, vals in per_source_epistemic.items()
    }

    def _phase_mean(phase: str) -> float:
        vals = [row["r_soft_relationship_epistemic"] for row in per_turn if row["phase"] == phase]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    phase_means = {p: _phase_mean(p) for p in ("stable", "rupture", "repair")}

    epistemic_not_collapsed = any(v > 0.0 for v in r_soft_series)
    rho_self_computable = rho_self is not None
    json_non_empty = len(per_turn) > 0
    exit0_pass = bool(epistemic_not_collapsed and rho_self_computable and json_non_empty)

    return {
        "probe": "relational_soft_verifier_stage0",
        "spec": "docs/specs/relational-soft-verifier.md",
        "debt": 85,
        "stage": 0,
        "scope_note": (
            "MACHINERY ONLY. Does NOT validate the central self-confirmation "
            "question (rho_ext). That needs the external human anchor (debt #51) "
            "and an LLM runtime (debt #10B)."
        ),
        "turns": len(per_turn),
        "regime_id": REGIME_ID,
        "metrics": {
            "rsv_epistemic_ratio": round(epistemic_ratio, 4),
            "rsv_rho_self": (round(rho_self, 4) if rho_self is not None else None),
            "mean_r_soft": round(mean_r_soft, 4),
            "mean_relationship_abs_error": round(mean_rel_abs, 4),
            "phase_mean_r_soft": phase_means,
            "per_source_mean_epistemic": per_source_mean_epistemic,
        },
        "exit0": {
            "epistemic_not_collapsed": epistemic_not_collapsed,
            "rho_self_computable": rho_self_computable,
            "json_non_empty": json_non_empty,
            "pass": exit0_pass,
        },
        "per_turn": per_turn,
    }


def main() -> None:
    result = run_probe()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(repo_root, "artifacts", "eq_uplift")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "relational_soft_verifier_shadow.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    m = result["metrics"]
    print("=== Stage 0 relational soft verifier probe (debt #85) ===")
    print(f"turns                : {result['turns']}")
    print(f"rsv_epistemic_ratio  : {m['rsv_epistemic_ratio']}")
    print(f"rsv_rho_self         : {m['rsv_rho_self']}")
    print(f"phase_mean_r_soft    : {m['phase_mean_r_soft']}")
    print(f"per_source_epistemic : {m['per_source_mean_epistemic']}")
    print(f"EXIT(0) pass         : {result['exit0']['pass']}  {result['exit0']}")
    print(f"written              : {out_path}")
    print(result["scope_note"])


if __name__ == "__main__":
    main()
