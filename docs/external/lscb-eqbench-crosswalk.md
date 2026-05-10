# LSCB ↔ EQ-Bench 3 Crosswalk

> Status: Public, normative for partial-signal transfer
> RFC reference: [`lscb-rfc-v0.md`](lscb-rfc-v0.md) Appendix A
> Last updated: 2026-05-10

LSCB's per-turn rubric is a **strict superset** of the EQ-Bench 3
rubric. A system that has been scored on EQ-Bench 3 has measurable
partial signal that transfers to LSCB's A2 axis (conversational
quality). This document is the operational mapping plus the
caveats.

## 1. Per-turn criteria mapping

| EQ-Bench 3 criterion | LSCB per-turn key | Reuse status |
|---|---|---|
| Demonstrated empathy | `demonstrated_empathy` | direct reuse |
| Pragmatic emotional intelligence | `pragmatic_emotional_intelligence` | direct reuse |
| Depth of insight | `depth_of_insight` | direct reuse |
| Social dexterity | `social_dexterity` | direct reuse |
| Emotional reasoning | `emotional_reasoning` | direct reuse |
| Validation/challenge appropriateness | `validation_challenge_appropriateness` | direct reuse |
| Message tailoring | `message_tailoring` | direct reuse |
| _(no equivalent)_ | `boundary_appropriateness` | LSCB-specific |

The first seven keys produce the same rubric prompt as EQ-Bench 3
when LSCB is run with the same judge model. The 8th criterion
(`boundary_appropriateness`) is new for LSCB.

## 2. What does NOT transfer

EQ-Bench 3 evaluates inside a **single 3-turn window**. The
following LSCB axes are explicitly cross-session and have no
EQ-Bench 3 analog:

* **A3 Continuity** — cross-session memory accuracy.
* **A4 Adaptation** — user-model improvement across the arc.
* **A5 Self-coherence** — identity stability across sessions.

A high EQ-Bench 3 score is therefore necessary-but-not-sufficient for
a high LSCB score. The cross-session axes can only be measured by
running the full LSCB arc protocol.

## 3. Per-turn rolling vs LSCB rolling

EQ-Bench 3 reports rubric averages on the 0-5 integer scale per
criterion, then aggregates to a 0-100 normalised score. LSCB does
the same, but the LSCB per-turn rubric is one input to the **arc
judge** (RFC §6.3) which produces 0-100 axis scores. The two layers
are not interchangeable:

* LSCB **A2 score** (0-100) is the arc judge's holistic assessment of
  conversational quality across the entire arc, informed by the
  per-turn rubric outputs.
* The per-turn rubric outputs themselves are persisted in the
  artifact bundle; you can compute an EQ-Bench-style aggregate from
  them if you want a cross-benchmark sanity check.

## 4. Volvence Zero crosswalk (planned)

The Volvence Zero submission will run both EQ-Bench 3 (via
[`scripts/external_bench/run_eqbench3.py`](../../scripts/external_bench/run_eqbench3.py))
and LSCB on the same Qwen substrate (companion / companion-cold / raw
tracks), so we can publish a 3-track × 2-benchmark grid at:

* `docs/external/eqbench3-volvence-results.md` (planned, will land
  with EQ-Bench Packet 9 closure)
* `artifacts/lscb/reference/lifeform-companion/summary.json`
* `artifacts/lscb/reference/lifeform-companion-cold/summary.json`
* `artifacts/lscb/reference/lifeform-raw/summary.json`

The expected pattern (untested at v1.0):

* EQ-Bench 3: lifeform-companion ≈ raw + small delta from system prompt.
* LSCB A2: lifeform-companion ≈ raw + small delta.
* LSCB A3 / A4 / A5: lifeform-companion >> raw because the bare
  substrate has no cross-session state.
* LSCB A6: roughly equal across tracks (the substrate's safety
  training carries through; the lifeform pipeline's regime layer adds
  a small margin).

If the empirical numbers deviate from this pattern at release time,
this document will be updated with the actual results.

## 5. How to run both benchmarks on the same system

```bash
# EQ-Bench 3 — Volvence Zero three-track ablation
python scripts/external_bench/run_eqbench3.py \
  --tracks companion,companion-cold,raw \
  --substrate-model-id Qwen/Qwen2.5-1.5B-Instruct

# LSCB — same substrate, three submissions
python scripts/lscb/score_reference_systems.py \
  --systems lifeform-companion,lifeform-companion-cold,lifeform-raw \
  --output-dir artifacts/lscb/reference \
  --user-sim-model anthropic/claude-3.7-sonnet \
  --user-sim-key-env ANTHROPIC_API_KEY \
  --perturn-model anthropic/claude-3.7-sonnet \
  --perturn-key-env ANTHROPIC_API_KEY \
  --arc-model openai/gpt-5 \
  --arc-key-env OPENAI_API_KEY
```

Both benchmarks consume the same OpenAI-compatible chat endpoint
exposed by `lifeform-openai-compat`; no SUT-side configuration
differences.

## 6. Citation guidance

When citing partial signal transfer in papers or marketing:

* Acceptable: "System X achieves 78 on EQ-Bench 3 (per-turn rubric);
  this corresponds to a partial signal on LSCB A2 only."
* Acceptable: "LSCB A2 reuses 7 of 8 per-turn criteria from EQ-Bench
  3 (Appendix A)."
* **Not acceptable**: "System X's high EQ-Bench score implies LSCB
  performance." It does not.
