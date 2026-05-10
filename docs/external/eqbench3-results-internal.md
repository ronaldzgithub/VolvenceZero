# EQ-Bench 3 — three-track ablation results (internal)

> Status: **TEMPLATE — awaiting first run**
> Last updated: 2026-05-10
> Scope: debt #29 packet 7 verdict surface
> Audience: internal review (this doc may inform but is not the public submission package — see [`eqbench3-submission-protocol.md`](eqbench3-submission-protocol.md) for that)

This document is the **per-run record** for our EQ-Bench 3
three-track ablation. Each fresh experiment writes a new dated
section at the top, keeping older runs as an audit trail. The
template below is the structure every entry should follow; do not
delete the template after the first run — copy it.

The packet 7 plan is intentionally split:

* The **machinery** to produce a verdict (the `compare_ablation.py`
  script + structured `AblationVerdict` JSON) lands in code.
* The **decision** of whether to ship to public leaderboards
  (Packet 10) lands HERE, after a human reviews the verdict against
  the four debt #29 red lines and the publish-threshold gate.

## How to update this doc

1. Run the ablation:

   ```bash
   python scripts/external_bench/run_eqbench3.py \
       --substrate-model-id Qwen/Qwen2.5-1.5B-Instruct \
       --tracks companion,companion-cold,raw \
       --judge-model anthropic/claude-3.7-sonnet \
       --no-elo
   ```

2. Run the comparator + dump verdict JSON:

   ```bash
   python scripts/external_bench/compare_ablation.py \
       --summaries artifacts/external_bench/eqbench3_*_*.summary.json \
       --output artifacts/external_bench/verdict_$(date -u +%Y%m%dT%H%M%SZ).json
   ```

3. Add a new section to this file using the template below. Keep
   numbers from the verdict JSON; add narrative analysis in the
   "Diagnosis" / "Decision" subsections.

4. If the verdict says **`go`**, file Packet 8 (EmpathyBench) +
   Packet 9 (cross-walk doc) + Packet 10 (public submission) as
   follow-ups. If **`hold`**, log the diagnosis here and update
   debt #29 with the next-step decision.

## Red-line enforcement

Every run summary must affirm these four declarations in its
attestation block (the comparator refuses to emit a verdict
otherwise):

1. `frozen_substrate: true` — substrate weights were not modified
2. `no_kernel_modification: true` — no `vz-*` or `lifeform-*` code
   touched besides the debt #29 packet 5 one-line `cli.py` change
3. `no_benchmark_text_in_system_prompt: true` — system prompt does
   not contain any EQ-Bench scenario text or rubric criteria
4. `no_internal_architecture_terms_in_model_card: true` — public
   model card does not mention `NL` / `ETA` / `R-PE` / `regime` /
   `owner SSOT` / `F1-F6` or any other internal vocabulary

These are the four red lines from debt #29 修法 5.

## How to read the table below

| Track | What it measures |
|---|---|
| `companion` | Full lifeform pipeline + trained companion bootstraps |
| `companion-cold` | Full lifeform pipeline, **no** trained bootstraps |
| `raw` | Bare substrate (Qwen) — bypasses the lifeform entirely |

Two key deltas:

* **Pipeline contribution** = `companion` − `raw`. Tells us whether
  the lifeform stack adds or subtracts EQ score over the bare LLM.
* **Bootstrap contribution** = `companion` − `companion-cold`. Tells
  us how much the offline-trained calibration is worth on EQ.

Reference range for Qwen 2.5 1.5B Instruct on EQ-Bench 3 rubric is
roughly 50-65 (frontier models are 70-85). The default
`--publish-threshold` for "GO public" is 65 on the 1.5B substrate;
raise to ~70+ on 7B and ~75+ on 14B.

---

## Run log (newest first)

### TEMPLATE — copy this block for every new run

> **Status**: pending first run
> **Run timestamp**: _e.g. 2026-MM-DDTHHMMSSZ_
> **Substrate**: _e.g. Qwen/Qwen2.5-1.5B-Instruct_
> **Hardware**: _e.g. RTX 4090 24GB / CPU-only_
> **Judge**: anthropic/claude-3.7-sonnet
> **ELO**: rubric-only / with-elo
> **Verdict file**: `artifacts/external_bench/verdict_<timestamp>.json`

#### Per-track scores

| Track | Rubric (0-100) | Notes |
|---|---|---|
| companion | _xx.xx_ |  |
| companion-cold | _xx.xx_ |  |
| raw | _xx.xx_ |  |

#### Deltas

| Delta | Value | Interpretation |
|---|---|---|
| companion − raw (pipeline) | _±x.xx_ | _lifeform helps / hurts / no-effect_ |
| companion − cold (bootstrap) | _±x.xx_ | _bootstraps help / hurt / no-effect_ |

#### Recommendations (from `compare_ablation.py`)

_paste verdict.recommendations bullet list here_

#### Diagnosis (human analysis)

_If pipeline contribution is non-positive: investigate which expression-layer
or PromptPlanner constraint is reducing model expressivity. Common suspects:_

- _expression-layer rewriting may over-truncate emotional nuance_
- _PromptPlanner's "stage" template may be too directive for free-form roleplay_
- _refusal layer firing on EQ-Bench scenarios it should not (verify via
  per-turn rationale_tags surfaced in response headers)_

_If bootstrap contribution is non-positive: regime priors may over-bias for
the benchmark's specific scenario distribution._

#### Decision

- [ ] **GO** — proceed to packets 8 + 9 + 10 (EmpathyBench / cross-walk / public submission)
- [ ] **HOLD** — record baseline, file follow-up debt for diagnosis, do NOT submit publicly
- [ ] **INSUFFICIENT DATA** — re-run with corrected configuration

#### Follow-up debts opened (if any)

_e.g. "Debt #31: companion pipeline regression of -3.2 EQ-Bench rubric vs raw —
likely PromptPlanner stage-template constraint. Diagnose before next ablation run."_

---

## Open questions (from debt #29 修法 (2) decision tree)

These remain to be resolved by the actual run:

1. **Does our pipeline help or hurt EQ on Qwen 1.5B?** — pipeline_delta sign.
2. **Are the trained bootstraps paying off on EQ?** — bootstrap_delta sign.
3. **Is the absolute score competitive enough to publish?** — primary track ≥ threshold.
4. **Should we move to a 7B substrate?** — only if (1) and (2) say "yes" and absolute is below threshold.
