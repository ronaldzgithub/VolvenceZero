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

## Why LLM-rendered synthetic data is not distillation

A fair objection to `--mode llm`: "if an LLM writes the conversations,
isn't the encoder just learning what that LLM thinks relationships look
like?" It would be — if the LLM were the **teacher**. Here it is only
the **actor**:

- **The supervision signal never comes from the LLM.** Labels are
  emitted by the scenario FSM at generation time: the script *declares*
  that a rupture fires at session 2 turn 3 and that the user returns 21
  days later testing a repair. The label is the scriptwriter's ground
  truth, not a reader-model's judgment. Distillation would mean training
  on an LLM's *assessment* of a transcript; that path is closed at the
  type level — `LabelSource` has no judge member.
- **The LLM renders structure into surface text.** Given the FSM state
  ("you are a user whose companion forgot something important, deciding
  whether to give it a second chance"), the simulator produces realistic
  utterances. Swapping the simulator changes the prose, not the labels.
- **The encoder can beat its own renderer** not by knowing more, but by
  task form: a zero-shot LLM must re-read the whole multi-session
  trajectory and reason over weeks of structure in-context on every
  call; the encoder is a compressed single-forward readout of exactly
  that structure — long-horizon state, calibration, and orders of
  magnitude cheaper. (That comparison is release-gated: encoder weights
  ship only if they beat the zero-shot baseline on held-out families.)

### Honest limits — synthetic is a bootstrap, not the endgame

LLM-simulated users are not real users. Passing validation on synthetic
families says nothing about real-world validity; simulator style
homogeneity and model fingerprints are real artifacts. Model cards for
anything trained on this data must say so. Synthetic data is used
because the alternative does not exist yet: per-turn ground-truth
relationship labels on *real* data are essentially unobtainable (human
annotation of "trust at turn 12" is expensive and noisy), and the
regimes that matter most — rupture, repair, dormancy — are long-tail in
the wild but can be covered densely by script. The schema already
carries the endgame: `TrajectorySource.CONSENTED_FIRST_PARTY` is the
provenance tag for real, consented interaction data, which is where any
serious encoder must eventually get its edge.

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
