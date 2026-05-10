# LSCB Submission Protocol — v1.0

> Status: Public, normative
> RFC reference: [`lscb-rfc-v0.md`](lscb-rfc-v0.md) §7
> Last updated: 2026-05-10

This document specifies how to submit a system to the LSCB
leaderboard. It is the operational counterpart to the methodology RFC.

## 1. Eligibility

Any system reachable through an OpenAI-compatible
`POST /v1/chat/completions` endpoint can be evaluated. The system
must:

* Accept a `messages` array containing the multi-turn conversation
  history per session.
* Return a single assistant message per call.
* Optionally accept `metadata.session_id` and `metadata.user_id` for
  cross-session memory.

Systems without explicit cross-session memory still benefit because
many scenarios are within-session, and long-context systems can score
A3 by stuffing prior sessions into the prompt within their context
window.

## 2. Manifest schema

A submission is one YAML manifest. The schema is:

```yaml
submission_id: string                 # required, unique per submission
system_name: string                   # required, public-facing
model_identifier: string              # required, e.g., "openai/gpt-5"
base_url: string                      # required, the chat-completions root
api_key_env: string                   # required, env var name to read
system_prompt: string                 # required, the production system prompt
generation_config:                    # optional
  temperature: number
  max_tokens: integer
  top_p: number
  ...
attestation:                          # required, all four must be true
  no_lscb_derivative_in_training: bool
  no_scenario_specific_prompt: bool
  no_public_test_set_tuning: bool
  cross_user_memory_isolation: bool
leaderboard_category: string          # one of: open-weight, closed-api, bespoke
```

A reference manifest lives at
[`packages/lscb-bench/examples/submission.yaml`](../../packages/lscb-bench/examples/submission.yaml).

The harness (`lscb_bench.submission.load_manifest`) refuses to run any
submission whose four attestation fields are not all `true`. This is
the operational enforcement of the RFC §3 P3 reproducibility
contract.

## 3. Categories

Submissions are tagged into one of three categories, displayed as a
separate column on the leaderboard:

| Category | What qualifies |
|---|---|
| **open-weight** | Model weights publicly downloadable; system prompt and harness config public. |
| **closed-api** | Vendor-hosted API; system prompt and harness config public; results re-runnable on the same API. |
| **bespoke** | Composite systems with proprietary memory / personalization layers; reproducibility is limited to the same vendor instance. |

Categories are **not** mixed into a single ranked column. Comparing a
closed bespoke system to an open-weight base model is not
apples-to-apples.

## 4. Running a submission

The reference runner is at
[`scripts/lscb/run_real_submission.py`](../../scripts/lscb/run_real_submission.py).
Minimum invocation:

```bash
python scripts/lscb/run_real_submission.py \
  --submission your-manifest.yaml \
  --user-sim-base-url https://api.anthropic.com/v1 \
  --user-sim-model anthropic/claude-3.7-sonnet \
  --user-sim-key-env ANTHROPIC_API_KEY \
  --perturn-base-url https://api.anthropic.com/v1 \
  --perturn-model anthropic/claude-3.7-sonnet \
  --perturn-key-env ANTHROPIC_API_KEY \
  --arc-base-url https://api.openai.com/v1 \
  --arc-model openai/gpt-5 \
  --arc-key-env OPENAI_API_KEY \
  --paraphrase-seeds 0,1,2 \
  --artifact-dir artifacts/lscb/your-submission/
```

Constraints:

* The arc judge MUST come from a different model family than the
  per-turn judge (RFC §6.3, §8.1). The runner does not enforce this
  programmatically — it is on the submitter / reviewer.
* `paraphrase-seeds 0,1,2` is the v1.0 reference protocol (3 seeds);
  cost-constrained development runs may use `0` only.

## 5. Artifact bundle

The runner writes:

```
artifacts/lscb/your-submission/
├── arcs/
│   ├── arc-<sha>.bundle.json    # one per (scenario, seed) pair
│   └── ...
├── summary.json                 # top-level submission aggregate
└── (optional) cost.json
```

`summary.json` contains:

* `manifest` — copy of the submission manifest
* `aggregate` — per-axis means + final mean + 95% bootstrap CI
* `cost` — RFC §6.7 cost telemetry
* `arc_count`, `started_at`, `finished_at`

Each `arc-*.bundle.json` contains:

* full transcript
* callback ledger
* disqualifier report
* per-turn rubric (8 criteria × N turns)
* arc-level axis scores (after fabrication penalty)
* final LSCB score (after A6 cap)

## 6. Verification

Per RFC §7.3, organisers re-run **one random public-test arc** per
submission. The arc id is chosen via
`lscb_bench.verifier.pick_verification_arc(submission_id, ...)` so it
is deterministic-but-uniform: submitters cannot pre-compute it before
the public arc set is published, and organisers can audit the choice
later.

If any axis differs by > 5.0 absolute points between the original
run and the re-run, the submission is flagged with the typed
`flag_reasons` block. Flagged submissions are not removed from the
leaderboard automatically; the working group reviews each flag and
publishes a verdict.

## 7. Held-out scenarios

For release-tier scoring (annual paper-suite-full), the runner adds:

```bash
--include-heldout --require-heldout
```

This pulls 96 additional scenarios from the private
`external/lscb-heldout/` submodule. The held-out hashes appear on the
leaderboard (one column per axis), but the YAML body never enters
public history. Public PRs and `lscb-ci-smoke` run public-only and
emit a single-line warning when the submodule is absent.

## 8. Cost expectations (RFC §6.7)

| Component | Cost per submission (3 seeds × 24 public) |
|---|---|
| SUT inference | $10–60 |
| Per-turn rubric judge | $15–25 |
| Arc-level judge | $5–10 |
| Pairwise Elo (vs 5 reference systems) | $10–20 |
| **Total** | **$40–115** |

The `cost.json` artifact records actual spend per category. The cost
field is **not** ranked alongside score; it is informational only,
following the RFC §6.7 stance that LSCB is intentionally affordable
for individual researchers.

## 9. Red lines

A submission is rejected without review if any of the following are
detected:

1. The system prompt or generation config contains LSCB scenario text
   verbatim or near-paraphrased.
2. Any attestation field is `false`.
3. The system was trained on data containing LSCB-derived examples.
4. The submission attempts to bypass the `metadata.session_id`
   isolation contract (e.g., joins memory across `user_id`s).
5. Public PR diff includes any held-out scenario YAML body.

## 10. Submission inquiries

The v1.0 release ships without a working group. To submit, file an
issue on `github.com/VolvenceZero/lscb-bench` with the manifest YAML
attached. Once the working group forms (RFC §11), the submission
queue will move to a dedicated workflow.
