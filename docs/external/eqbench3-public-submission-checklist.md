# EQ-Bench 3 — public-submission checklist (Packet 10)

> Status: **GATED — do not actuate until Packet 7 verdict is "go"**
> Last updated: 2026-05-10
> Companion docs: [`eqbench3-submission-protocol.md`](eqbench3-submission-protocol.md), [`eqbench3-results-internal.md`](eqbench3-results-internal.md)

This is the actuation checklist for publishing our score to
the public EQ-Bench 3 leaderboard at <https://eqbench.com>. It is
**deliberately a checklist, not an instruction set**: the actual
submission is a human-in-the-loop event, not an automated step,
because (a) public attribution requires explicit organizational
sign-off and (b) the upstream maintainer's review process is
manual. This document exists so the steps are pre-agreed and can
be executed quickly when the green light comes.

## Pre-conditions (must all be true)

- [ ] **Internal verdict is "go"**. The most recent run record in
      [`eqbench3-results-internal.md`](eqbench3-results-internal.md)
      was emitted by `compare_ablation.py` with `go_no_go: "go"`.
- [ ] **All four red-line attestations declared**. The verdict's
      attestation block has all four declarations True (frozen
      substrate / no kernel modification / no benchmark text in
      system prompt / no internal architecture terms in model card).
      `compare_ablation.py` refuses to emit a "go" without these,
      so this is automatic — but verify.
- [ ] **Reproduction recipe verified**. A second person on the team
      ran the [`eqbench3-submission-protocol.md#reproduction-recipe`](eqbench3-submission-protocol.md)
      end-to-end on a clean environment and reproduced the score
      within the harness's documented per-iteration variance
      (~0.75 rubric points stddev for repeated runs of the same
      model per upstream's `--iterations` calibration).
- [ ] **No-contamination attestation human-signed**. A reviewer has
      signed off that the system's training data does not include
      EQ-Bench 3 scenarios, transcripts, or judge rationales. This
      is the fifth attestation that `compare_ablation.py` cannot
      verify — it is a human declaration backed by training-data
      manifest review.
- [ ] **Public adapter mirror exists**. The
      `lifeform-openai-compat` wheel + `scripts/external_bench/`
      runner are reachable from a public URL the EQ-Bench
      maintainer can clone (org-internal source release, public
      repo mirror, or attached tarball).

## Submission steps (when all pre-conditions are true)

1. **Open a GitHub issue at <https://github.com/EQ-bench/eqbench3>**
   following the project's "submit a model" template (or the
   format in [`eqbench3-submission-protocol.md#model-card-template`](eqbench3-submission-protocol.md)).
   Per upstream's policy, formal leaderboard inclusion may require
   open-weight access to the substrate (Qwen 2.5 1.5B Instruct
   already qualifies — published on HuggingFace under Qwen's
   licence). If our adapter is closed-source, this issue should
   request that the **substrate score** be listed (it is fully
   reproducible) and we provide the **system score** as
   supplementary information for context. The maintainer decides
   what to publish.

2. **Attach the verdict JSON** produced by `compare_ablation.py`.
   This contains the per-track rubric scores, deltas, and
   attestation block.

3. **Attach the run logs** from `artifacts/external_bench/` for
   the run that produced the "go" verdict. These are the raw
   transcripts the harness recorded; reviewers can replay them.

4. **Cross-link [`eqbench3-submission-protocol.md`](eqbench3-submission-protocol.md)** as the canonical
   reproducibility document. This is the doc that will be cited in
   external press / fundraising materials referring to our score.

5. **Mirror the submission summary in our own publishing channels**
   only after the upstream issue is open. Do not lead with our
   marketing channels — that has historically caused disputes when
   numbers are later revised by the maintainer.

## Post-submission

- [ ] **Update [`eqbench3-results-internal.md`](eqbench3-results-internal.md)**
      with the upstream issue URL, submission date, and any
      maintainer questions / requested revisions.

- [ ] **Update debt #29** in [`docs/known-debts.md`](../known-debts.md)
      with the closure status. If accepted to the leaderboard:
      move the debt to the closed section. If pending review or
      rejected: keep the debt open with the new state in its
      "subsequent updates" section.

- [ ] **Add a comparison row to [`companion-bench-rfc-v0.md`](companion-bench-rfc-v0.md)
      Appendix A** so the RFC's compatibility statement has
      empirical evidence backing.

## Conditions that abort the submission flow

If any of the following becomes true after Packet 7 says "go" but
before submission is complete, halt the flow and re-evaluate:

* Upstream EQ-Bench 3 changes its judge model in a way that
  invalidates our previously-collected rubric scores.
* A material change ships to our adapter or kernel between the
  ablation run and the submission. Re-run before submitting.
* A reviewer raises a red-line concern (training-data contamination,
  benchmark text in system prompt, etc.) that was missed in the
  pre-condition pass. File a follow-up debt to address before any
  resubmission.

## Escalation

If the public submission is denied or revised by the upstream
maintainer in a way that materially changes our positioning,
treat that as an externally-attested signal and:

1. Open a follow-up debt with the diagnosis.
2. Update [`companion-bench-rfc-v0.md`](companion-bench-rfc-v0.md) §8 (validity threats) with the
   lesson learned.
3. Pause any external marketing referring to the EQ-Bench score
   until the diagnosis is closed.
