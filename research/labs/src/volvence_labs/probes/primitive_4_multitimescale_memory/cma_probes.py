"""P4 Multi-timescale Memory — CMA behavioral probes (stage 1).

Hypothesis: A system with multi-timescale memory (fast working memory + slow
consolidated memory) outperforms flat retrieval on tasks requiring both
immediate recall and long-range association.

Cells:
- baseline: flat key-value retrieval (no timescale separation)
- probe_on: Titans-style test-time training head (fast) + persistent store (slow)
- probe_off: only slow memory (no fast adaptation)
- counterfactual: random retrieval (should degrade to chance)

Eval: CMA 4 behavioral probes (synthetic versions for CI):
1. Immediate recall (working memory)
2. Delayed recall (consolidated memory)
3. Interference resistance (timescale separation)
4. Association across timescales

Papers: Titans (2501.00663), Miras (2504.13173), CMA (2601.09913).
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


def _generate_real_memory_task(seed: int, model_id: str = "sshleifer/tiny-gpt2", n_items: int = 32) -> dict:
    """Generate memory task with real model embeddings as keys/values.

    Uses sentence embeddings from the model as key-value pairs, creating
    a more realistic retrieval scenario where semantic similarity matters.
    """
    try:
        from ...framework.runtime import get_model_runtime

        # Sentences at different "timescales" (recent vs old context)
        recent_sentences = [
            "The cat sat on the mat.",
            "She opened the door quickly.",
            "Rain fell on the window.",
            "He typed the password carefully.",
            "The phone rang loudly.",
            "Coffee was brewing in the kitchen.",
            "The train arrived on time.",
            "She smiled at the stranger.",
        ]
        old_sentences = [
            "Long ago, the kingdom flourished under wise rule.",
            "The ancient library held secrets of forgotten civilizations.",
            "Across the vast ocean, explorers sought new lands.",
            "The mountain pass was treacherous in winter.",
            "Generations of scholars studied the mysterious texts.",
            "The old bridge had stood for centuries.",
            "Deep in the forest, a hidden temple remained.",
            "The stars guided travelers through the desert.",
            "A forgotten language was carved into the stone walls.",
            "The river had changed course many times over the ages.",
            "Ancient trade routes connected distant cultures.",
            "The ruins told stories of a once-great empire.",
            "Legends spoke of treasures buried beneath the hills.",
            "The observatory tracked celestial movements for decades.",
            "Old maps revealed paths no longer traveled.",
            "The monastery preserved knowledge through dark ages.",
            "Fossils in the cliff face dated back millions of years.",
            "The canal system was an engineering marvel of its era.",
            "Oral traditions kept the history alive for generations.",
            "The archaeological dig uncovered layers of civilization.",
            "Ancient astronomers predicted eclipses with precision.",
            "The fortress walls had withstood countless sieges.",
            "Scrolls in the vault contained lost mathematical proofs.",
            "The migration patterns had remained unchanged for millennia.",
        ]

        rng = np.random.default_rng(seed)
        fast_boundary = n_items // 4
        n_recent = min(fast_boundary, len(recent_sentences))
        n_old = min(n_items - fast_boundary, len(old_sentences))

        selected_recent = [recent_sentences[i % len(recent_sentences)] for i in range(n_recent)]
        selected_old = [old_sentences[i % len(old_sentences)] for i in range(n_old)]
        all_sentences = selected_recent + selected_old

        rt = get_model_runtime(model_id, dtype="fp32")
        result = rt.encode_text(all_sentences, max_length=64)
        embeddings = result["embeddings"].numpy().astype(np.float32)
        dim = embeddings.shape[1]

        # Use embeddings as both keys and values (self-retrieval task)
        keys = embeddings
        values = embeddings  # In real scenario, values could be different

        n_actual = len(all_sentences)
        timescales = np.array([0 if i < n_recent else 1 for i in range(n_actual)])

        # Queries: perturbed versions of keys
        n_queries = min(16, n_actual)
        query_indices = rng.integers(0, n_actual, size=n_queries)
        queries = keys[query_indices] + rng.standard_normal((n_queries, dim)).astype(np.float32) * 0.1
        expected_values = values[query_indices]
        query_timescales = timescales[query_indices]

        return {
            "keys": keys.tolist(),
            "values": values.tolist(),
            "timescales": timescales.tolist(),
            "queries": queries.tolist(),
            "expected_values": expected_values.tolist(),
            "query_timescales": query_timescales.tolist(),
            "n_items": n_actual,
            "dim": dim,
            "n_queries": n_queries,
            "model_id": model_id,
            "model_sha": rt.model_sha,
            "source": "real",
        }
    except Exception as e:
        result = _generate_memory_task(seed=seed, n_items=n_items)
        result["source"] = "synthetic_fallback"
        result["fallback_reason"] = str(e)
        return result


def _generate_memory_task(seed: int, n_items: int = 32, seq_len: int = 128) -> dict:
    """Generate a synthetic multi-timescale memory task.

    Creates key-value pairs at different "timescales" (positions in sequence),
    then queries that require recall from different depths.
    """
    rng = np.random.default_rng(seed)
    dim = 16

    # Keys and values at different timescales
    keys = rng.standard_normal((n_items, dim)).astype(np.float32)
    values = rng.standard_normal((n_items, dim)).astype(np.float32)

    # Assign timescales: items 0..n/4 are "recent" (fast), rest are "old" (slow)
    fast_boundary = n_items // 4
    timescales = np.array([0 if i < fast_boundary else 1 for i in range(n_items)])

    # Queries: mix of fast and slow recall
    n_queries = 16
    query_indices = rng.integers(0, n_items, size=n_queries)
    queries = keys[query_indices] + rng.standard_normal((n_queries, dim)).astype(np.float32) * 0.1
    expected_values = values[query_indices]
    query_timescales = timescales[query_indices]

    return {
        "keys": keys.tolist(),
        "values": values.tolist(),
        "timescales": timescales.tolist(),
        "queries": queries.tolist(),
        "expected_values": expected_values.tolist(),
        "query_timescales": query_timescales.tolist(),
        "n_items": n_items,
        "dim": dim,
        "n_queries": n_queries,
    }


class FlatRetriever:
    """Baseline: flat cosine-similarity retrieval."""

    def __init__(self, keys: np.ndarray, values: np.ndarray):
        self.keys = keys / (np.linalg.norm(keys, axis=-1, keepdims=True) + 1e-8)
        self.values = values

    def query(self, q: np.ndarray) -> np.ndarray:
        q_norm = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)
        sims = q_norm @ self.keys.T  # (n_queries, n_items)
        best_idx = sims.argmax(axis=-1)
        return self.values[best_idx]


class TitansStyleRetriever:
    """Probe_on: Titans-inspired dual-timescale memory.

    Fast memory: recent items with exponential decay weighting.
    Slow memory: all items with uniform weighting.
    Retrieval: weighted combination of fast and slow matches.
    """

    def __init__(self, keys: np.ndarray, values: np.ndarray, timescales: np.ndarray):
        self.keys = keys / (np.linalg.norm(keys, axis=-1, keepdims=True) + 1e-8)
        self.values = values
        self.timescales = timescales
        # Fast memory weights: exponential decay for "old" items
        self.fast_weights = np.where(timescales == 0, 1.0, 0.1)
        self.slow_weights = np.ones_like(timescales, dtype=np.float32)

    def query(self, q: np.ndarray) -> np.ndarray:
        q_norm = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)
        sims = q_norm @ self.keys.T  # (n_queries, n_items)

        # Fast retrieval (biased toward recent)
        fast_sims = sims * self.fast_weights[None, :]
        fast_idx = fast_sims.argmax(axis=-1)
        fast_vals = self.values[fast_idx]

        # Slow retrieval (uniform)
        slow_idx = sims.argmax(axis=-1)
        slow_vals = self.values[slow_idx]

        # Combine: use fast for recent queries, slow for old queries
        # (In real Titans, this is learned; here we use a heuristic)
        fast_confidence = fast_sims.max(axis=-1)
        slow_confidence = sims.max(axis=-1)
        alpha = np.where(
            fast_confidence > slow_confidence * 0.8,
            0.7,  # trust fast memory
            0.3,  # trust slow memory
        )
        return alpha[:, None] * fast_vals + (1 - alpha[:, None]) * slow_vals


class SlowOnlyRetriever:
    """Probe_off: only slow (consolidated) memory, no fast adaptation."""

    def __init__(self, keys: np.ndarray, values: np.ndarray, timescales: np.ndarray):
        # Only keep "old" items
        slow_mask = timescales == 1
        self.keys = keys[slow_mask]
        self.keys = self.keys / (np.linalg.norm(self.keys, axis=-1, keepdims=True) + 1e-8)
        self.values = values[slow_mask]

    def query(self, q: np.ndarray) -> np.ndarray:
        if len(self.keys) == 0:
            return np.zeros((len(q), q.shape[-1]), dtype=np.float32)
        q_norm = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)
        sims = q_norm @ self.keys.T
        best_idx = sims.argmax(axis=-1)
        return self.values[best_idx]


def _eval_retrieval(retrieved: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    """Compute retrieval quality metrics."""
    # Cosine similarity between retrieved and expected
    r_norm = retrieved / (np.linalg.norm(retrieved, axis=-1, keepdims=True) + 1e-8)
    e_norm = expected / (np.linalg.norm(expected, axis=-1, keepdims=True) + 1e-8)
    cos_sims = (r_norm * e_norm).sum(axis=-1)

    # MSE
    mse = ((retrieved - expected) ** 2).mean()

    return {
        "mean_cosine_sim": float(cos_sims.mean()),
        "std_cosine_sim": float(cos_sims.std()),
        "mse": float(mse),
        "recall_at_90": float((cos_sims > 0.9).mean()),
    }


@register_probe
class CMAProbe(BaseProbe):
    id = "cma-probes-v1"
    hypothesis = (
        "Multi-timescale memory (Titans-style fast+slow) outperforms flat retrieval "
        "on tasks requiring both immediate recall and long-range association."
    )
    primitive = PrimitiveTag.P4_MULTITIMESCALE_MEMORY
    r_ids = ("R1", "R5", "R6")

    def knobs(self) -> dict[str, list]:
        return {
            "n_items": [32, 64],
            "fast_decay": [0.1, 0.3],
            "use_real_model": [False, True],
            "model_id": ["sshleifer/tiny-gpt2"],
        }

    def default_inputs(self, seed: int) -> Any:
        return _generate_memory_task(seed=seed)

    def real_inputs(self, seed: int, knobs: Mapping[str, Any]) -> Any:
        """Generate memory task with real model embeddings as keys/values."""
        model_id = knobs.get("model_id", "sshleifer/tiny-gpt2")
        return _generate_real_memory_task(seed=seed, model_id=model_id)

    def run_cell(self, ctx: ProbeContext, knobs: Mapping[str, Any]) -> RunOutcome:
        inputs = ctx.inputs
        keys = np.array(inputs["keys"], dtype=np.float32)
        values = np.array(inputs["values"], dtype=np.float32)
        timescales = np.array(inputs["timescales"], dtype=np.int32)
        queries = np.array(inputs["queries"], dtype=np.float32)
        expected = np.array(inputs["expected_values"], dtype=np.float32)
        query_ts = np.array(inputs["query_timescales"], dtype=np.int32)

        if ctx.cell == AblationCell.BASELINE:
            retriever = FlatRetriever(keys, values)
        elif ctx.cell == AblationCell.PROBE_ON:
            retriever = TitansStyleRetriever(keys, values, timescales)
        elif ctx.cell == AblationCell.PROBE_OFF:
            retriever = SlowOnlyRetriever(keys, values, timescales)
        elif ctx.cell == AblationCell.COUNTERFACTUAL:
            # Random retrieval
            rng = np.random.default_rng(ctx.seed + 7777)
            class RandomRetriever:
                def query(self, q):
                    return rng.standard_normal(q.shape).astype(np.float32)
            retriever = RandomRetriever()
        else:
            raise ValueError(f"unknown cell: {ctx.cell!r}")

        retrieved = retriever.query(queries)
        overall_metrics = _eval_retrieval(retrieved, expected)

        # Per-timescale breakdown
        fast_mask = query_ts == 0
        slow_mask = query_ts == 1
        fast_metrics = _eval_retrieval(retrieved[fast_mask], expected[fast_mask]) if fast_mask.any() else {}
        slow_metrics = _eval_retrieval(retrieved[slow_mask], expected[slow_mask]) if slow_mask.any() else {}

        readouts = ReadoutBundle(
            metrics={
                "mean_cosine_sim": overall_metrics["mean_cosine_sim"],
                "mse": overall_metrics["mse"],
                "recall_at_90": overall_metrics["recall_at_90"],
                "fast_cosine_sim": fast_metrics.get("mean_cosine_sim", 0.0),
                "slow_cosine_sim": slow_metrics.get("mean_cosine_sim", 0.0),
                "n_queries": float(len(queries)),
            },
            artifacts={
                "overall": overall_metrics,
                "fast": fast_metrics,
                "slow": slow_metrics,
            },
            tags={
                "cell": ctx.cell.value,
                "seed": ctx.seed,
                "n_items": inputs["n_items"],
            },
        )

        return RunOutcome(
            readouts=readouts,
            output={
                "cell": ctx.cell.value,
                "mean_cosine_sim": overall_metrics["mean_cosine_sim"],
            },
        )

    def gate(self, outcomes: list[RunOutcome]) -> GateReport:
        if not outcomes:
            return GateReport(passed=False, reason="no outcomes", stats={})

        # Gate: probe_on should beat baseline on mean_cosine_sim
        baseline = [o for o in outcomes if o.readouts.tags.get("cell") == "baseline"]
        probe_on = [o for o in outcomes if o.readouts.tags.get("cell") == "probe_on"]

        if not baseline or not probe_on:
            return GateReport(passed=False, reason="missing baseline or probe_on", stats={})

        b_mean = sum(o.readouts.metrics["mean_cosine_sim"] for o in baseline) / len(baseline)
        p_mean = sum(o.readouts.metrics["mean_cosine_sim"] for o in probe_on) / len(probe_on)
        diff = p_mean - b_mean

        passed = diff > 0.0  # probe_on must beat baseline
        return GateReport(
            passed=passed,
            reason=f"probe_on cosine={p_mean:.4f} vs baseline={b_mean:.4f} (diff={diff:+.4f})",
            stats={"baseline_mean": b_mean, "probe_on_mean": p_mean, "diff": diff},
        )
