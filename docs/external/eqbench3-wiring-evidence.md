# EQ-Bench 3 Wiring Evidence — dry-run on synthetic vertical

> Status: Wiring layer validated; P10 actuation (real Qwen + real judge) is the next step.
> Date: 2026-05-11
> Related debts: [#29](../known-debts.md) (P10 actuation gate), [#34](../known-debts.md) (harness perf for full-tier runs).

This document records the end-to-end dry-run that proves the
EQ-Bench 3 harness can drive our `lifeform-openai-compat` adapter
without any kernel modification and without API spend.

## What was validated

1. Upstream harness cloned cleanly into `external/eqbench3/` (~30 MB, 359 files).
2. Upstream `requirements.txt` (8 light deps: requests, dotenv, tqdm,
   numpy, joblib, nltk, wordfreq, scipy, trueskill) installs in
   ~50 s on a stock conda environment.
3. `lifeform-serve --vertical companion --substrate-mode synthetic
   --enable-openai-compat` boots successfully on port 8770.
4. `GET /v1/health`, `GET /v1/models`, `POST /v1/chat/completions`
   all return 200 with the expected JSON shape.
5. `x-lifeform-*` telemetry headers (`mode`, `regime`,
   `abstract-action`, `pe-magnitude`, `rationale-tags`) appear on
   chat-completion responses so the ablation comparator can
   correlate per-turn EQ-Bench rubric scores against lifeform
   internal state.
6. `python external/eqbench3/eqbench3.py --no-rubric --no-elo
   --ignore-canonical ...` runs the full 45-scenario set through
   our adapter:
   - 45/45 scenarios completed (status: 26 `completed` + 19
     `scenario_completed`).
   - 26 multi-turn debriefs also passed.
   - Wall time on CPU + synthetic substrate: **14:42**.
   - Output artifact: `artifacts/external_bench/eqbench3_dry_run.runs.json`
     (1.17 MB, valid OpenAI schema throughout).

## Bugs surfaced and fixed during wiring

### 1. `TEST_API_URL` requires the full chat-completions path

eqbench3's `utils/api.py:25` uses the env var verbatim — it does
**not** append `/chat/completions`. Our previous
`.env.example` and `scripts/external_bench/run_eqbench3.py`
wrote just the `/v1` base URL, so the harness sent its POST to
`http://127.0.0.1:8770/v1` (→ 404) then fell back to
`https://api.openai.com/v1/chat/completions` (→ 401, missing key).

Fix landed in this packet:

* `scripts/external_bench/.env.example` — both `TEST_API_URL` and
  `JUDGE_API_URL` now include the full path.
* `scripts/external_bench/run_eqbench3.py` — `base_url` per-track
  construction appends `/chat/completions` explicitly (both
  `lifeform` and `raw` modes).

The 11 unit tests in
`packages/lifeform-openai-compat/tests/test_run_eqbench3_smoke.py`
still pass after the fix.

### 2. PowerShell `Out-File -Encoding utf8` writes a BOM

The BOM corrupts the first line of `.env` (turning `TEST_API_URL`
into `\ufeffTEST_API_URL`, which `python-dotenv` silently drops).

This is a per-OS pitfall not a code bug. The runner script avoids
this because it does not write `.env`; the user is expected to copy
`.env.example` manually. We added a note to the example file
warning Windows users to use `Out-File -Encoding utf8NoBOM` (or
just edit in a regular text editor) when adapting the template.

## Cosmetic, not fixed

* eqbench3's `print_summary_box` (line 191) emits `\u2011`
  (non-breaking hyphen) which crashes Windows `cp936/gbk` stdout
  encoding. Exit code is still 0 (the error is caught after the
  full run is persisted to JSON), but the terminal-rendered
  summary box is absent on Windows. Upstream issue, not blocking.

## What is NOT yet validated (P10 actuation gate)

1. Real Qwen 1.5B substrate (`--substrate-mode hf-shared
   --substrate-model-id Qwen/Qwen2.5-1.5B-Instruct`). The synthetic
   vertical produces formulaic responses; real EQ-Bench scoring
   needs the actual substrate.
2. Real judge (Anthropic Claude 3.7 Sonnet by default per
   `.env.example`). Without `--no-rubric` the harness POSTs each
   completed scenario to the judge URL for 8-criterion scoring.
3. ELO pairwise comparisons against canonical leaderboard models.
   Costs an additional ~$10-20 per track per
   `scripts/external_bench/run_eqbench3.py --with-elo`.
4. Full three-track ablation
   (`companion / companion-cold / raw`). Each track spawns its own
   `lifeform-serve` process; total time ~3 × inference wallclock.

P10 actuation cost (RFC §6.7 of LSCB applies here too, since the
same judge model is shared):

| Component | Estimate |
|---|---|
| Qwen 1.5B inference, 3 tracks × 45 scenarios | 1.5-3 h on a 24GB GPU |
| Rubric judge (Claude 3.7 Sonnet) | ~$1.50 / track × 3 ≈ $5 |
| With ELO (`--with-elo`) | ~$10-20 / track × 3 ≈ $30-60 |

Recommended trigger order:

1. Run a **single track** real-Qwen + real-judge to calibrate
   wallclock and cost.
2. If results match expectations, fan out to all three tracks
   sequentially (or in parallel if VRAM allows).
3. Call `scripts/external_bench/compare_ablation.py` on the three
   `.summary.json` files to get the structured GO / HOLD verdict.

## Reproduction recipe

```bash
# 1. Install upstream harness deps
pip install -r external/eqbench3/requirements.txt

# 2. Create .env (do NOT use PowerShell Out-File default encoding)
cp scripts/external_bench/.env.example external/eqbench3/.env
# … then edit external/eqbench3/.env to set JUDGE_API_KEY for the real run

# 3. Boot the adapter in one shell
python -m lifeform_service.cli \
  --host 127.0.0.1 --port 8770 \
  --vertical companion --substrate-mode synthetic \
  --enable-openai-compat --log-level INFO

# 4. Drive the harness in another shell
cd external/eqbench3
python eqbench3.py \
  --test-model lifeform-companion-synthetic \
  --model-name vz-dry-run \
  --judge-model anthropic/claude-3.7-sonnet \
  --runs-file ../../artifacts/external_bench/eqbench3_dry_run.runs.json \
  --elo-results-file ../../artifacts/external_bench/eqbench3_dry_run.elo.json \
  --threads 1 --iterations 1 \
  --no-rubric --no-elo --ignore-canonical
```

For real scoring drop `--no-rubric` (and optionally `--no-elo`),
set a live `JUDGE_API_KEY` in `external/eqbench3/.env`, and
substitute `--substrate-mode hf-shared --substrate-model-id
Qwen/Qwen2.5-1.5B-Instruct --substrate-device cuda` in the
`lifeform_service.cli` invocation.
