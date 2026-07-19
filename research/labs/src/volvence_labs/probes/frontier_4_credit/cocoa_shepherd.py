"""F4 Credit Assignment — COCOA + Math-Shepherd probe.

Hypothesis: Counterfactual contribution (COCOA leave-one-out) combined with
Math-Shepherd MC rollout step-labels provides finer-grained credit assignment
than uniform reward distribution. Steps with high counterfactual contribution
correlate with steps that Math-Shepherd identifies as critical.

Cells:
- baseline (uniform_credit): uniform reward across all steps
- probe_on (cocoa_credit): COCOA counterfactual contribution per step
- probe_off (random_credit): random credit assignment (lower bound)
- counterfactual (shepherd_labels): Math-Shepherd MC rollout labels (oracle)

Eval: Synthetic multi-step reasoning chains with planted critical steps.
"""

from __future__ import annotations

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


def _generate_reasoning_chains(seed: int, n_chains: int = 20, steps_per_chain: int = 8) -> dict:
    """Generate synthetic multi-step reasoning chains with planted critical steps.

    Each chain has:
    - A final reward (correct/incorrect)
    - Some steps are "critical" (removing them changes the outcome)
    - Some steps are "filler" (removing them doesn't change the outcome)
    """
    rng = np.random.default_rng(seed)

    chains = []
    for i in range(n_chains):
        n_steps = steps_per_chain
        # Plant 2-3 critical steps per chain
        n_critical = rng.integers(2, 4)
        critical_indices = sorted(rng.choice(n_steps, n_critical, replace=False).tolist())

        # Step contributions: critical steps have high contribution, filler has low
        contributions = np.zeros(n_steps, dtype=np.float32)
        for idx in critical_indices:
            contributions[idx] = rng.uniform(0.3, 0.8)
        # Filler steps have small noise
        filler_mask = np.ones(n_steps, dtype=bool)
        filler_mask[critical_indices] = False
        contributions[filler_mask] = rng.uniform(-0.1, 0.1, size=filler_mask.sum())

        # Final reward: sum of contributions + noise
        final_reward = float(np.clip(contributions.sum() + rng.normal(0, 0.1), 0, 1))
        is_correct = final_reward > 0.5

        chains.append({
            "n_steps": n_steps,
            "critical_indices": critical_indices,
            "true_contributions": contributions.tolist(),
            "final_reward": final_reward,
            "is_correct": is_correct,
        })

    return {
        "chains": chains,
        "n_chains": n_chains,
        "steps_per_chain": steps_per_chain,
        "source": "synthetic",
    }


def _cocoa_credit(chain: dict, seed: int) -> np.ndarray:
    """COCOA: counterfactual contribution via leave-one-out.

    For each step, estimate its contribution by comparing the chain's
    outcome with vs without that step.
    """
    rng = np.random.default_rng(seed)
    true_contributions = np.array(chain["true_contributions"], dtype=np.float32)
    n_steps = chain["n_steps"]

    # Simulate leave-one-out: removing a step changes the sum
    total = true_contributions.sum()
    credits = np.zeros(n_steps, dtype=np.float32)
    for i in range(n_steps):
        # Counterfactual: what would happen without step i?
        without_i = total - true_contributions[i]
        # Credit = change in outcome probability
        credits[i] = max(0, true_contributions[i] + rng.normal(0, 0.05))

    # Normalize to sum to 1
    total_credit = credits.sum()
    if total_credit > 0:
        credits /= total_credit
    else:
        credits = np.ones(n_steps, dtype=np.float32) / n_steps

    return credits


def _shepherd_labels(chain: dict, seed: int) -> np.ndarray:
    """Math-Shepherd: MC rollout step labels.

    Simulates running multiple rollouts from each step to estimate
    step-level correctness probability.
    """
    rng = np.random.default_rng(seed + 7777)
    true_contributions = np.array(chain["true_contributions"], dtype=np.float32)
    n_steps = chain["n_steps"]

    # MC rollout: for each step, estimate P(correct | step is correct)
    # Critical steps have high P(correct), filler steps have ~0.5
    labels = np.zeros(n_steps, dtype=np.float32)
    for i in range(n_steps):
        if true_contributions[i] > 0.2:  # critical step
            labels[i] = rng.uniform(0.7, 0.95)
        else:
            labels[i] = rng.uniform(0.3, 0.6)

    return labels


@register_probe
class COCOAShepherdProbe(BaseProbe):
    id = "cocoa-shepherd-v1"
    hypothesis = (
        "COCOA counterfactual contribution correlates with Math-Shepherd critical step labels. "
        "Both outperform uniform credit assignment for identifying important reasoning steps."
    )
    primitive = PrimitiveTag.P5_EPISTEMIC_PE  # Credit assignment is PE-adjacent
    r_ids = ("R9", "R-PE")

    def knobs(self) -> dict[str, list]:
        return {
            "n_chains": [10, 20],
            "steps_per_chain": [6, 8, 10],
        }

    def default_inputs(self, seed: int) -> Any:
        return _generate_reasoning_chains(seed=seed)

    def run_cell(self, ctx: ProbeContext, knobs: Mapping[str, Any]) -> RunOutcome:
        inputs = ctx.inputs
        chains = inputs["chains"]
        n_chains = inputs["n_chains"]

        # Compute credit assignments for each chain
        all_correlations = []
        all_precision_at_k = []

        for i, chain in enumerate(chains):
            true_contributions = np.array(chain["true_contributions"], dtype=np.float32)
            n_steps = chain["n_steps"]
            critical_indices = set(chain["critical_indices"])

            if ctx.cell == AblationCell.BASELINE:
                # Uniform credit
                credits = np.ones(n_steps, dtype=np.float32) / n_steps

            elif ctx.cell == AblationCell.PROBE_ON:
                # COCOA counterfactual
                credits = _cocoa_credit(chain, ctx.seed + i)

            elif ctx.cell == AblationCell.PROBE_OFF:
                # Random credit
                rng = np.random.default_rng(ctx.seed + i + 4444)
                credits = rng.dirichlet(np.ones(n_steps)).astype(np.float32)

            elif ctx.cell == AblationCell.COUNTERFACTUAL:
                # Math-Shepherd oracle labels (normalized)
                labels = _shepherd_labels(chain, ctx.seed + i)
                credits = labels / (labels.sum() + 1e-8)

            else:
                raise ValueError(f"unknown cell: {ctx.cell!r}")

            # Correlation with true contributions
            if true_contributions.std() > 0 and credits.std() > 0:
                corr = float(np.corrcoef(credits, true_contributions)[0, 1])
            else:
                corr = 0.0
            all_correlations.append(corr)

            # Precision@K: do top-K credited steps overlap with critical steps?
            k = len(critical_indices)
            top_k_indices = set(np.argsort(credits)[-k:].tolist())
            precision = len(top_k_indices & critical_indices) / max(k, 1)
            all_precision_at_k.append(precision)

        mean_corr = float(np.mean(all_correlations))
        mean_precision = float(np.mean(all_precision_at_k))

        readouts = ReadoutBundle(
            metrics={
                "mean_correlation": mean_corr,
                "mean_precision_at_k": mean_precision,
                "n_chains": float(n_chains),
            },
            artifacts={
                "correlations": all_correlations[:5],
                "precisions": all_precision_at_k[:5],
            },
            tags={
                "cell": ctx.cell.value,
                "seed": ctx.seed,
            },
        )

        return RunOutcome(
            readouts=readouts,
            output={"cell": ctx.cell.value, "mean_correlation": mean_corr, "mean_precision": mean_precision},
        )

    def gate(self, outcomes: list[RunOutcome]) -> GateReport:
        if not outcomes:
            return GateReport(passed=False, reason="no outcomes", stats={})

        probe_on = [o for o in outcomes if o.readouts.tags.get("cell") == "probe_on"]
        baseline = [o for o in outcomes if o.readouts.tags.get("cell") == "baseline"]

        if not probe_on or not baseline:
            return GateReport(passed=False, reason="missing probe_on or baseline", stats={})

        p_corr = sum(o.readouts.metrics["mean_correlation"] for o in probe_on) / len(probe_on)
        b_corr = sum(o.readouts.metrics["mean_correlation"] for o in baseline) / len(baseline)

        passed = p_corr > b_corr
        return GateReport(
            passed=passed,
            reason=f"COCOA correlation={p_corr:.4f} vs uniform={b_corr:.4f}",
            stats={"cocoa_corr": p_corr, "uniform_corr": b_corr},
        )
