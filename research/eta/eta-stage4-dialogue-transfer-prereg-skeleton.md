# ETA Stage-4 dialogue-transfer preregistration (SKELETON — design only)

**Status:** contingent skeleton. This document is produced as part of the
LLM-transfer ladder but describes an experiment that is **not executed here and
must not be run until Stages 1–3 all pass**:

- Gate 1 (rate axis responds on the seeded proof corpus): PENDING authoritative
  run. The 2026-08-01 `kill-eta` verdict remains in force until it passes.
- Gate 2 (continued-pretrained LLM residual linearly carries the active
  subgoal): contingent on Gate 1.
- Gate 3 (rate-distortion gap reappears on the补课 frozen LLM): contingent on
  Gate 2.

Only if all three retain the thesis does this dialogue-transfer experiment get
finalised into a real preregistration and executed. Writing the skeleton now
fixes the intended design so it cannot be reverse-engineered from a curve later.

## Why dialogue is the last rung, not the first

The proof environment gives a subgoal ground-truth (which coloured location is
active), so Gate 3 can require boundary-F1 to be higher inside the gap. Real
dialogue has **no subgoal ground-truth**: there is no oracle "active
conversational subgoal" per turn. So the boundary-F1 leg of the criterion is
unavailable and the dialogue criterion must fall back to the two legs that do
not need a subgoal oracle.

## Proposed design (to be frozen only if Gates 1–3 pass)

- **Action space (`z_t` distortion target):** the next dialogue act / response,
  scored as expert-response NLL through the steered frozen model — the same
  `SteeredActionScorer` distortion used in the proof harness, with the response
  surface replacing the `move:<location>` surface.
- **Corpus:** the already-provisioned Multi-Session Chat (MSC) corpus
  (`data/external/msc/v0.1/`), with a session-level train/heldout split that is
  disjoint by conversation id (no session appears on both sides).
- **Substrate:** the Gate-3 substrate lineage — either the original Qwen (if
  Gate 3 retained on the domain-pretrained base, the dialogue rerun tests
  whether the same holds without domain pretraining) or a dialogue-domain
  continued-pretrained base produced by the same rare-heavy merge path.
- **Criterion (two legs, boundary-F1 dropped):**
  1. **Rate-distortion gap:** sweep alpha; the frozen arm must show the
     near-vertical gap (drop_share ≥ 0.5 over ≤ 0.25 of the rate span) and the
     joint validity-control arm must not. Reuse `assess_gap` unchanged.
  2. **Held-out generalization:** the gap-region operating point must reduce
     held-out (unseen-session) response NLL relative to the unsteered baseline,
     i.e. the compression `z_t` learns transfers to new conversations.
- **Validity control:** the joint arm remains mandatory; indistinguishable arms
  ⇒ `instrument-invalid`, no conclusion.
- **Verdict set:** `retain-eta-on-dialogue`, `retain-weak`, `kill-eta`,
  `instrument-invalid`, `inconclusive-joint-arm-gap`, `incomplete-sweep`.

## Prohibited (to be enforced by the real preregistration)

- Inventing a per-turn subgoal oracle to resurrect a boundary-F1 leg.
- Splitting train/heldout by turn instead of by conversation id.
- Selecting alpha, the gap thresholds, or the substrate lineage after seeing any
  dialogue curve.
- Running any of this before Gates 1–3 have all passed under their own frozen
  preregistrations.

## Deliverable of this skeleton

None beyond the design record. No corpus is rendered, no model is trained, no
sweep is run. When Stages 1–3 pass, this skeleton is converted into a frozen
JSON preregistration (mirroring `preregister_eta_rate_distortion.py`) with source
SHAs, and only then executed.
