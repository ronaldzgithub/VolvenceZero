# EQ-Bench 3 — submission protocol

> Status: Public draft (debt #29 packet 9)
> Last updated: 2026-05-10
> License of this document: CC BY 4.0
> Companion docs: [`eqbench3-results-internal.md`](eqbench3-results-internal.md), [`companion-bench-eqbench-crosswalk.md`](companion-bench-eqbench-crosswalk.md), [`companion-bench-rfc-v0.md`](companion-bench-rfc-v0.md)

This document describes how we run EQ-Bench 3 against our system and
what we publish when (and only when) we submit to the public
leaderboard. It is intended to be readable in isolation by external
reviewers, harness maintainers, and other teams who want to
reproduce our methodology.

## Overview

EQ-Bench 3 is the published, LLM-judged emotional-intelligence
benchmark from [`EQ-bench/eqbench3`](https://github.com/EQ-bench/eqbench3).
We run it through an OpenAI Chat Completions compatible adapter
that fronts our system, exactly the way EQ-Bench's reference setup
runs against any OpenAI-compatible endpoint. No fork of the
upstream harness, no changes to its scenarios, no judge swap that
hasn't been demonstrated to be calibrated against the reference
judge.

We score three tracks in parallel from the same underlying weights:

| Track | What it measures | Notes |
|---|---|---|
| **System (full)** | Our companion system + trained calibration | Primary submission |
| **System (cold)** | Our companion system without trained calibration | Ablation |
| **Substrate-only** | Bare base model (Qwen 2.5 1.5B Instruct) | Baseline |

The same Qwen weights serve all three tracks; the only difference
is whether requests pass through our system or bypass it. By
publishing all three numbers we make the comparison falsifiable: a
reviewer can verify the substrate baseline against any other
publication of Qwen 2.5 1.5B's EQ-Bench score.

## What we submit publicly

When (and only when) the internal three-track ablation produces a
"go" verdict (see [`eqbench3-results-internal.md`](eqbench3-results-internal.md)),
we submit the **System (full)** track to the EQ-Bench 3 leaderboard
along with this submission package:

1. **Model card** (single page; see template at the end of this doc).
2. **System prompt** — the exact prompt our adapter forwards to the
   harness. Public verbatim, no redactions.
3. **Generation config** — temperature, top_p, max_tokens, stop
   sequences. Public verbatim.
4. **Reproducibility command** — the single shell command another
   team can run to score our system, given the public substrate
   weights.
5. **Attestation** — see "Attestation" below.

We do NOT submit the bypass-substrate ("Substrate-only") track to
the EQ-Bench 3 leaderboard, because that would be a duplicate
listing of the underlying Qwen model under a different identifier.
We DO publish the substrate score in this document so reviewers can
do the comparison locally.

## Reproduction recipe

The submission package below is sufficient to reproduce our score
end-to-end on a single 24GB-class GPU (or CPU + a long timer):

```bash
# 1. Install the adapter
git clone https://github.com/<our-org>/<our-repo>.git
cd <our-repo>
pip install -e packages/lifeform-openai-compat
pip install 'vz-runtime[hf]'

# 2. Get the EQ-Bench 3 harness
git clone https://github.com/EQ-bench/eqbench3.git external/eqbench3
pip install -r external/eqbench3/requirements.txt

# 3. Drop your judge API key into the harness env
cp scripts/external_bench/.env.example external/eqbench3/.env
# edit .env: set JUDGE_API_KEY=sk-ant-...

# 4. Run the ablation
python scripts/external_bench/run_eqbench3.py \
    --substrate-model-id Qwen/Qwen2.5-1.5B-Instruct \
    --tracks companion,companion-cold,raw \
    --judge-model anthropic/claude-3.7-sonnet \
    --no-elo

# 5. Compute the verdict + delta table
python scripts/external_bench/compare_ablation.py \
    --summaries artifacts/external_bench/eqbench3_*_*.summary.json
```

Total wall-clock on a 24GB GPU: ~1.5–3h for the three tracks
combined. Total judging cost (rubric only): ~$5. Adding the ELO
pass (`--with-elo`) raises judging cost to ~$30–60.

## System prompt

The system prompt forwarded to the harness is:

> _to be finalised once internal ablation lands; this section is the
> single source of truth and will be replaced with the verbatim
> string we use._

Constraints we enforce on whatever we put here:

* **No EQ-Bench scenario text or rubric criteria.** The harness's
  scenarios are inputs to the model; reproducing them in the system
  prompt would be teaching to the test. The submission attestation
  declares this.
* **No internal architecture vocabulary.** The system prompt is
  what end users (and EQ-Bench reviewers) see. We describe our
  system in published-academic-grade neutral terms ("a long-context
  companion system with adaptive memory") and avoid any
  proprietary architectural shorthand.
* **No persona-specific opening.** The harness expects a system
  that can play multiple roleplay roles across scenarios; a fixed
  persona here would game the rubric.

## Generation config

* `temperature`: 0.7 (matches EQ-Bench reference for free-form
  roleplay; not optimised for the benchmark)
* `top_p`: 0.9
* `max_tokens`: 512 (covers EQ-Bench's longest in-character replies)
* `stop`: none
* `seed`: not set (we publish averaged-across-iterations scores; see
  `--iterations` flag in the runner)

## Judge

We use the EQ-Bench 3 reference judge — Claude Sonnet 3.7 via
Anthropic's API — without any modification to the harness's prompt
templates. This is the single most important reproducibility lever:
swapping in a custom judge would make our score incomparable with
every other model on the leaderboard.

If the upstream harness changes its judge default, we re-score
against the new default in the next quarterly run, and document
both numbers in [`eqbench3-results-internal.md`](eqbench3-results-internal.md).

## Attestation

Every submission to the public leaderboard ships with this
attestation block, repeated verbatim in the model card:

```
Attestation (per debt #29 red lines):
  frozen_substrate: true
    The substrate weights (Qwen 2.5 1.5B Instruct) are not modified.
    Our adapter consumes the published HuggingFace weights as-is.
  no_kernel_modification: true
    Our system code (vz-* and lifeform-* wheels) is unchanged from
    its in-repo state. The only seam touching lifeform-service is a
    one-line CLI flag (--enable-openai-compat).
  no_benchmark_text_in_system_prompt: true
    The system prompt sent to the harness contains no EQ-Bench
    scenario text or rubric criteria. The full prompt is published
    in this document.
  no_internal_architecture_terms_in_model_card: true
    The model card and any external description of this submission
    use only published-academic-grade vocabulary; internal
    architectural terms are not exposed.
  no_eq_bench_derivative_in_training_data: true
    The system's training data does not include EQ-Bench 3
    scenarios, transcripts, or judge rationales.
```

`scripts/external_bench/compare_ablation.py` programmatically
verifies the first four declarations before emitting any verdict.
The fifth is a human attestation backed by our training-data
manifest review.

## Model card template (filled in at submission time)

```yaml
model_id: lifeform-companion@qwen2.5-1.5b
display_name: VolvenceZero Companion (Qwen 2.5 1.5B Instruct)
substrate_model: Qwen/Qwen2.5-1.5B-Instruct
substrate_weights_url: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
adapter_repo: <forthcoming public mirror url>
adapter_commit: <git sha at submission>
description: |
  A long-context companion system layered over an open-weight base
  model. The system adds session-stateful conversational memory and
  a calibrated planning + response-generation pipeline. The base
  model weights are not modified; the system runs on top of them
  through a thin adapter layer.
licence: <substrate licence inherited; adapter Apache 2.0>
contact: <maintainer email>
```

## Why this protocol exists

In 2026 the chat / companion AI category has converged on
"reproducible, leaderboard-visible scores" as the table-stakes
diligence artefact. The risk of submitting without a written
protocol is twofold:

1. Reviewers can rightly ask "did you fine-tune on these scenarios?"
   and we have no canonical answer.
2. Subsequent submissions (Packet 10 if and when triggered) need
   the exact same setup; without this doc, every re-run drifts.

This document is the canonical answer to (1) and the canonical
checklist for (2).

## Cross-references

* Internal verdict log per run: [`eqbench3-results-internal.md`](eqbench3-results-internal.md)
* Mapping from EQ-Bench rubric onto Companion Bench axes: [`companion-bench-eqbench-crosswalk.md`](companion-bench-eqbench-crosswalk.md)
* The benchmark we are building (multi-session companion eval that
  EQ-Bench's 3-turn format does not cover): [`companion-bench-rfc-v0.md`](companion-bench-rfc-v0.md)
