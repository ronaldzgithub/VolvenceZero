# External chat benchmark harness — three-track ablation runner

> Status: Packet 6 of debt #29
> Last updated: 2026-05-10

This directory holds the launchers we use to score Volvence Zero on
external chat / EQ benchmarks (EQ-Bench 3, EmpathyBench, etc.). The
harnesses themselves are upstream open-source projects we clone into
`external/`; this directory adds the **three-track ablation
orchestration** that compares:

* **Track A — `companion`**: full lifeform pipeline (PromptPlanner +
  ResponseSynthesizer + memory + regime + adaptive controllers) with
  the pre-trained companion bootstraps loaded.
* **Track B — `companion-cold`**: same lifeform pipeline but without
  the companion bootstraps (regime / temporal artifacts disabled).
  Isolates the score contribution of the trained artifacts.
* **Track C — `raw substrate`**: bypass the lifeform entirely; run
  the same Qwen substrate weights via `mode=raw` on the OpenAI-compat
  router. Establishes the bare-LLM baseline.

The same Qwen model serves all three tracks — the only difference is
how requests are routed through (or around) the lifeform pipeline.
By comparing all three on the same scenarios, we get a measurable
delta for "what does our architecture add over the base LLM?"

## Prerequisites

1. Install the OpenAI-compat wheel:

   ```
   pip install -e packages/lifeform-openai-compat
   ```

2. Install vz-runtime's HF extras (for Qwen):

   ```
   pip install 'vz-runtime[hf]'
   ```

3. Clone the EQ-Bench 3 harness into `external/eqbench3/`:

   ```
   git clone https://github.com/EQ-bench/eqbench3.git external/eqbench3
   pip install -r external/eqbench3/requirements.txt
   ```

4. Copy `.env.example` to `external/eqbench3/.env` and fill in your
   judge API key. We default to Anthropic (Claude Sonnet 3.7) per
   the EQ-Bench reference setup.

## Running the three tracks

```
python scripts/external_bench/run_eqbench3.py \
    --substrate-model-id Qwen/Qwen2.5-1.5B-Instruct \
    --substrate-device auto \
    --tracks companion,companion-cold,raw \
    --judge-model anthropic/claude-3.7-sonnet \
    --no-elo \
    --threads 1
```

What this does:

1. Boots three `lifeform-serve` instances on three ports (one per
   track) with `--enable-openai-compat`. All three share the same
   Qwen weights (loaded once per process; the script serialises track
   runs to keep VRAM/RAM at single-model footprint by default — pass
   `--parallel-tracks` to override if you have the headroom).
2. Runs `external/eqbench3/eqbench3.py` against each track's port,
   pointing `--test-model` at the OpenAI-compat URL the script
   exposes.
3. Saves per-track results into `artifacts/external_bench/`:
   * `eqbench3_<track>_<timestamp>.json` — full transcripts + rubric
     scores from the upstream harness (private)
   * `eqbench3_<track>_<timestamp>.summary.json` — distilled scores
     + lifeform telemetry headers (regime / pe / abstract action) we
     correlate with rubric grades
4. Calls `compare_ablation.py` (Packet 7) to emit the cross-track
   delta table.

## Cost / time

| Component | Estimate |
|---|---|
| Inference (Qwen 1.5B, three tracks × 45 scenarios × 3 turns) | ~1.5–3h on a single 24GB GPU; 5–10× longer on CPU |
| Judge (rubric only, Claude Sonnet 3.7) | ~$1.50 / track × 3 = ~$5 |
| Judge (rubric + ELO) | ~$10–20 / track × 3 = ~$30–60 |

Default is **rubric-only** (`--no-elo`). Add `--with-elo` once
verdict #29 packet 7 confirms the rubric scores are competitive
enough to bother spending the extra ELO budget.

## Scope and red lines

This launcher does not:

* train, fine-tune, or modify any kernel weight (frozen substrate)
* mutate any contract (vz-* and lifeform-* untouched outside the
  one-line cli.py change in Packet 5)
* embed any EQ-Bench / EmpathyBench scenario text in the system
  prompt (red line per debt #29 修法 5 第 3 条)
* publish any internal architecture vocabulary in the model card
  (red line per debt #29 修法 5 第 2 条; see
  `docs/external/eqbench3-submission-protocol.md` for the public
  description we ship)

Every track in this runner produces an attestation block in the
output JSON declaring the above; `compare_ablation.py` refuses to
emit a verdict if any track is missing an attestation.

## Packet status

| Packet | Status | What it lands |
|---|---|---|
| Packet 6 (this) | landed | runner + .env.example + README + per-track config |
| Packet 7 | pending | cross-track diff + verdict + go/no-go gate |
| Packet 8 | pending | empathybench second track (parallel structure) |
| Packet 9 | pending | public submission protocol + Companion Bench cross-walk |
| Packet 10 | conditional | EQ-Bench leaderboard PR (only if Packet 7 verdict goes "go") |
