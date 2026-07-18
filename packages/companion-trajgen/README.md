# companion-trajgen

Synthetic trajectory generation for the **Relationship Representation
Standard** (`companion-standard`). Runs Companion Bench public scenarios
through the bench's own user simulator + arc runner and exports canonical
`InteractionTrajectory` documents with per-segment relationship-state labels.

## Label provenance (hard invariant)

Labels come from **generation-time FSM state only** — the scenario's
declared FSM action at each `(session, turn)` coordinate drives a
deterministic relationship-state walk (`companion_trajgen.labeler`).
Judge scores are never a label source (R12: evaluation is a read-only
readout, never a learning source). The standard's `LabelSource` enum
cannot even represent judge output.

## Held-out exclusion (structural)

This package must never import `companion_bench.heldout_loader`, and every
scenario load goes through `load_scenarios_dir(include_held_out=False)`.
Held-out scenarios stay out of any training set *by construction*, enforced
by `tests/contracts/test_companion_trajgen_boundaries.py`.

## Two generation modes

- `--mode fsm` — deterministic: fake user utterances + fake SUT replies,
  zero LLM cost, byte-reproducible. Structure (FSM probe placement, session
  gaps, labels) is exact; surface text is synthetic filler.
- `--mode llm` — the bench's `OpenAIUtteranceClient` generates user turns
  and a real OpenAI-compatible SUT endpoint produces assistant turns
  (same procurement conventions as a Companion Bench run).

## Usage

```bash
# deterministic batch over the 30 public scenarios, 3 seeds each
companion-trajgen generate --mode fsm --out-dir data/trajectories

# LLM mode against a real SUT
companion-trajgen generate --mode llm --out-dir data/trajectories \
  --sut-base-url http://127.0.0.1:8000/v1 --sut-model my-model \
  --sim-base-url https://openrouter.ai/api/v1 --sim-model qwen/qwen3-235b
```

Train/val splits are assigned **whole scenario families at a time**
(`--val-families F5,F6` by default); a family never straddles the split.
Every exported document is validated with the `companion_standard`
conformance kit before it is written.
