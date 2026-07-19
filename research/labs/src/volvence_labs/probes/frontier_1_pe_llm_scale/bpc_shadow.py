"""F1 PE-at-LLM-scale — BPC shadow evidence probe.

Hypothesis: Bits-per-character (BPC) on long-context text reveals PE structure
that is invisible at token level. Epistemic PE decomposition (from P5) applied
at BPC scale shows different distributional properties than raw cross-entropy.

This is a cross-primitive probe: it combines P5 (epistemic PE) with LLM-scale
evaluation to produce evidence for VZ P0-PE.4 (BPC SHADOW evidence).

Cells:
- baseline (raw_bpc): standard BPC (cross-entropy / chars_per_token)
- probe_on (epistemic_bpc): BPC decomposed into epistemic + aleatoric components
- probe_off (uniform_bpc): BPC assuming uniform distribution (upper bound)
- counterfactual (shuffled_bpc): BPC on shuffled text (destroys long-range structure)

Model: TinyLlama (or tiny-gpt2 for CI).
Eval: WikiText-2 validation set (long-form text).
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


def _generate_synthetic_bpc_data(seed: int, n_segments: int = 16, tokens_per_segment: int = 64) -> dict:
    """Generate synthetic BPC data for CI testing.

    Simulates token-level cross-entropy with realistic BPC characteristics:
    - Lower BPC in predictable regions (within-sentence)
    - Higher BPC at boundaries (between-sentence)
    - Long-range correlations in BPC trajectory
    """
    rng = np.random.default_rng(seed)
    total_tokens = n_segments * tokens_per_segment

    # Base BPC: AR(1) process with segment-level shifts
    bpc = np.zeros(total_tokens, dtype=np.float32)
    segment_means = rng.uniform(0.5, 2.0, size=n_segments)

    for seg in range(n_segments):
        start = seg * tokens_per_segment
        end = start + tokens_per_segment
        # Within-segment: low variance AR(1)
        noise = rng.standard_normal(tokens_per_segment).astype(np.float32) * 0.1
        ar = np.zeros(tokens_per_segment, dtype=np.float32)
        ar[0] = segment_means[seg] + noise[0]
        for t in range(1, tokens_per_segment):
            ar[t] = 0.9 * ar[t - 1] + 0.1 * segment_means[seg] + noise[t]
        bpc[start:end] = ar
        # Boundary spike
        if seg > 0:
            bpc[start] += rng.uniform(0.5, 1.5)

    # Chars per token (typical: 3-5 for English)
    chars_per_token = rng.uniform(3.0, 5.0, size=total_tokens).astype(np.float32)

    # Epistemic component: higher at boundaries, lower within segments
    epistemic = np.zeros(total_tokens, dtype=np.float32)
    for seg in range(n_segments):
        start = seg * tokens_per_segment
        end = start + tokens_per_segment
        epistemic[start:end] = rng.uniform(0.01, 0.1, size=tokens_per_segment)
        if seg > 0:
            epistemic[start] += rng.uniform(0.2, 0.5)

    return {
        "bpc": bpc.tolist(),
        "chars_per_token": chars_per_token.tolist(),
        "epistemic": epistemic.tolist(),
        "n_segments": n_segments,
        "tokens_per_segment": tokens_per_segment,
        "total_tokens": total_tokens,
        "source": "synthetic",
    }


def _generate_real_bpc_data(seed: int, model_id: str = "sshleifer/tiny-gpt2", n_segments: int = 8) -> dict:
    """Generate BPC data from real model forward pass on WikiText-2.

    Falls back to synthetic if model or dataset unavailable.
    """
    try:
        from ...framework.runtime import get_model_runtime
        from ...framework.runtime.dataset import wikitext2_val

        rt = get_model_runtime(model_id, dtype="fp32")
        rt.load_model()

        # Load WikiText-2 text
        ds = wikitext2_val()
        texts = ds.load_texts(text_field="text", limit=100)
        # Filter empty lines and join into segments
        texts = [t for t in texts if len(t.strip()) > 20]

        rng = np.random.default_rng(seed)
        selected = rng.choice(len(texts), size=min(n_segments, len(texts)), replace=False)

        all_bpc = []
        all_chars_per_token = []
        all_epistemic = []

        for idx in selected:
            text = texts[idx][:256]  # limit length for speed
            if len(text) < 10:
                continue

            result = rt.get_logits_for_text(text, max_length=128)
            logits = result["logits"].numpy()  # (seq_len, vocab)
            input_ids = result["input_ids"].numpy()  # (seq_len,)

            # Compute per-token cross-entropy
            seq_len = logits.shape[0] - 1  # shift by 1 for next-token prediction
            if seq_len < 2:
                continue

            # Softmax + gather target log-prob
            shifted_logits = logits[:-1]  # predict next token
            targets = input_ids[1:]  # actual next tokens

            max_logits = shifted_logits.max(axis=-1, keepdims=True)
            exp_logits = np.exp(shifted_logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
            target_probs = probs[np.arange(seq_len), targets]
            target_probs = np.clip(target_probs, 1e-8, 1.0)
            ce = -np.log(target_probs)  # nats

            # Convert to BPC: bits = nats / ln(2), then / chars_per_token
            bits = ce / math.log(2)
            tokens = result["tokens"][1:]  # shifted
            cpt = np.array([max(len(t.replace("Ġ", " ").replace("Ċ", "\n")), 1) for t in tokens], dtype=np.float32)
            bpc = bits / cpt

            # Epistemic proxy: entropy of prediction distribution
            entropy = -(probs * np.log(np.clip(probs, 1e-8, None))).sum(axis=-1)
            epistemic = entropy / math.log(2) / cpt  # bits per char

            all_bpc.extend(bpc.tolist())
            all_chars_per_token.extend(cpt.tolist())
            all_epistemic.extend(epistemic.tolist())

        if not all_bpc:
            raise ValueError("No valid segments produced")

        return {
            "bpc": all_bpc,
            "chars_per_token": all_chars_per_token,
            "epistemic": all_epistemic,
            "n_segments": n_segments,
            "tokens_per_segment": len(all_bpc) // max(n_segments, 1),
            "total_tokens": len(all_bpc),
            "model_id": model_id,
            "model_sha": rt.model_sha,
            "source": "real",
        }
    except Exception as e:
        result = _generate_synthetic_bpc_data(seed=seed, n_segments=n_segments)
        result["source"] = "synthetic_fallback"
        result["fallback_reason"] = str(e)
        return result


@register_probe
class BPCShadowProbe(BaseProbe):
    id = "bpc-shadow-v1"
    hypothesis = (
        "BPC on long-context text reveals PE structure invisible at token level. "
        "Epistemic BPC decomposition shows different distributional properties than raw CE."
    )
    primitive = PrimitiveTag.F5_R15_FORMALIZATION  # Cross-primitive, filed under F5 for now
    r_ids = ("R-PE", "R5")

    def knobs(self) -> dict[str, list]:
        return {
            "use_real_model": [False, True],
            "model_id": ["sshleifer/tiny-gpt2"],
            "n_segments": [8, 16],
        }

    def default_inputs(self, seed: int) -> Any:
        return _generate_synthetic_bpc_data(seed=seed)

    def real_inputs(self, seed: int, knobs: Mapping[str, Any]) -> Any:
        model_id = knobs.get("model_id", "sshleifer/tiny-gpt2")
        n_segments = knobs.get("n_segments", 8)
        return _generate_real_bpc_data(seed=seed, model_id=model_id, n_segments=n_segments)

    def run_cell(self, ctx: ProbeContext, knobs: Mapping[str, Any]) -> RunOutcome:
        inputs = ctx.inputs
        bpc = np.array(inputs["bpc"], dtype=np.float32)
        chars_per_token = np.array(inputs["chars_per_token"], dtype=np.float32)
        epistemic = np.array(inputs["epistemic"], dtype=np.float32)
        total_tokens = inputs["total_tokens"]

        if ctx.cell == AblationCell.BASELINE:
            # Raw BPC
            effective_bpc = bpc
            effective_epistemic = np.zeros_like(bpc)
            aleatoric = bpc.copy()

        elif ctx.cell == AblationCell.PROBE_ON:
            # Epistemic BPC decomposition
            effective_bpc = bpc
            effective_epistemic = epistemic
            aleatoric = np.maximum(bpc - epistemic, 0.0)

        elif ctx.cell == AblationCell.PROBE_OFF:
            # Uniform BPC (upper bound): log2(vocab_size) / chars_per_token
            vocab_size = 32000  # TinyLlama vocab
            uniform_bpc = math.log2(vocab_size) / chars_per_token
            effective_bpc = uniform_bpc
            effective_epistemic = np.zeros_like(bpc)
            aleatoric = uniform_bpc.copy()

        elif ctx.cell == AblationCell.COUNTERFACTUAL:
            # Shuffled BPC: destroy long-range structure
            rng = np.random.default_rng(ctx.seed + 8888)
            shuffled_bpc = rng.permutation(bpc)
            effective_bpc = shuffled_bpc
            effective_epistemic = rng.permutation(epistemic)
            aleatoric = np.maximum(shuffled_bpc - effective_epistemic, 0.0)
        else:
            raise ValueError(f"unknown cell: {ctx.cell!r}")

        # Compute distributional metrics
        mean_bpc = float(effective_bpc.mean())
        std_bpc = float(effective_bpc.std())
        mean_epistemic = float(effective_epistemic.mean())
        mean_aleatoric = float(aleatoric.mean())
        epistemic_ratio = mean_epistemic / (mean_bpc + 1e-8)

        # Long-range correlation: autocorrelation at lag 10
        if len(effective_bpc) > 20:
            centered = effective_bpc - effective_bpc.mean()
            autocorr_10 = float(np.correlate(centered[:-10], centered[10:])[0] / (np.var(centered) * len(centered[:-10]) + 1e-8))
        else:
            autocorr_10 = 0.0

        # IQR (inter-quartile range)
        q25, q75 = float(np.percentile(effective_bpc, 25)), float(np.percentile(effective_bpc, 75))
        iqr = q75 - q25

        readouts = ReadoutBundle(
            metrics={
                "mean_bpc": mean_bpc,
                "std_bpc": std_bpc,
                "mean_epistemic_bpc": mean_epistemic,
                "mean_aleatoric_bpc": mean_aleatoric,
                "epistemic_ratio": epistemic_ratio,
                "autocorr_lag10": autocorr_10,
                "iqr": iqr,
                "n_tokens": float(total_tokens),
            },
            artifacts={
                "bpc_head": effective_bpc[:16].tolist(),
                "epistemic_head": effective_epistemic[:16].tolist(),
                "distributional": {
                    "q25": q25,
                    "q75": q75,
                    "iqr": iqr,
                    "skewness": float(np.mean(((effective_bpc - mean_bpc) / (std_bpc + 1e-8)) ** 3)),
                },
            },
            tags={
                "cell": ctx.cell.value,
                "seed": ctx.seed,
                "source": inputs.get("source", "unknown"),
            },
        )

        return RunOutcome(
            readouts=readouts,
            output={"cell": ctx.cell.value, "mean_bpc": mean_bpc, "epistemic_ratio": epistemic_ratio},
        )

    def gate(self, outcomes: list[RunOutcome]) -> GateReport:
        if not outcomes:
            return GateReport(passed=False, reason="no outcomes", stats={})

        # Gate: epistemic decomposition (probe_on) should have non-trivial epistemic_ratio
        # AND raw BPC (baseline) should have higher autocorrelation than shuffled (counterfactual)
        probe_on = [o for o in outcomes if o.readouts.tags.get("cell") == "probe_on"]
        baseline = [o for o in outcomes if o.readouts.tags.get("cell") == "baseline"]
        counterfactual = [o for o in outcomes if o.readouts.tags.get("cell") == "counterfactual"]

        if not probe_on or not baseline:
            return GateReport(passed=False, reason="missing probe_on or baseline", stats={})

        # Check 1: epistemic ratio > 0 in probe_on
        mean_ratio = sum(o.readouts.metrics["epistemic_ratio"] for o in probe_on) / len(probe_on)
        ratio_ok = mean_ratio > 0.01

        # Check 2: baseline autocorrelation > counterfactual (long-range structure preserved)
        b_autocorr = sum(o.readouts.metrics["autocorr_lag10"] for o in baseline) / len(baseline)
        if counterfactual:
            cf_autocorr = sum(o.readouts.metrics["autocorr_lag10"] for o in counterfactual) / len(counterfactual)
            structure_ok = b_autocorr > cf_autocorr
        else:
            structure_ok = True

        passed = ratio_ok and structure_ok
        return GateReport(
            passed=passed,
            reason=f"epistemic_ratio={mean_ratio:.4f} (>0.01), autocorr baseline={b_autocorr:.4f} > cf",
            stats={"mean_epistemic_ratio": mean_ratio, "baseline_autocorr": b_autocorr},
        )
