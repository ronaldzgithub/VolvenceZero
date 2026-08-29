# High-value research directions

These directions come from local Packet 2 source, v2/v3 reports, trajectories,
and the current owner contracts. They are engineering hypotheses, not measured
candidate facts. External literature lookup is optional and passive.

1. **Strict marker accounting** — replace the legacy append-after-cap behavior
   with a marker-inclusive hard cap. Expected signature: strict budget pass
   rises to 1.0 and margin improves with unchanged recall retention. Surface:
   truncation strategy. Risk: cutting the last recalled entry.
2. **Whole-line packing** — retain only complete owner-published lines. Expected
   signature: no partial evidence and lower tokens. Surface: overflow behavior.
   Risk: a single long high-value entry may be skipped.
3. **Failure-first recall ordering** — prioritize entries carrying the exact
   structured `outcome:fail` tag. Expected signature: failed-entry retention
   stays 1.0 at smaller budgets. Surface: recall ordering. Risk: pass examples
   that encode preservation constraints disappear.
4. **Strength-first ordering** — use owner-published strength without parsing
   natural-language content. Expected signature: stable recall retention with a
   smaller entry count. Surface: ordering. Risk: strength and future utility
   may be miscalibrated.
5. **Recency-first control** — test whether current ordering beats a transparent
   recency baseline. Expected signature: either a lower margin at equal
   retention or useful negative evidence. Surface: ordering. Risk: recurring
   old conventions are displaced.
6. **Generic-section quota** — reserve most of the cap for recalled experience
   and bound generic snapshot descriptions separately. Expected signature:
   lower mean tokens while failed-entry retention remains 1.0. Surface: generic
   budget fraction. Risk: useful plan/open-loop state is omitted.
7. **Owner-section ablation** — remove one named snapshot family at a time and
   measure the efficiency/retention Pareto surface. Surface: section list.
   Risk: replay retention cannot reveal downstream hand-quality loss, so every
   win remains formal-validation pending.
8. **Exact-line deduplication** — eliminate byte-identical owner surfaces only.
   Expected signature: non-negative margin improvement and unchanged evidence
   counts. Surface: deduplication. Risk: low impact if redundancy is semantic
   rather than exact.
9. **Budget ladder** — evaluate fixed caps from the current 3500 toward the
   smallest cap that preserves all recalled failure evidence. Surface: hard
   cap. Risk: overfitting to the public eight-chain corpus.
10. **Independent baseline control** — reproduce the legacy policy and verify
    the v3 ratio before trusting improvements. Surface: control/replication.
    Risk: current owner descriptions may have drifted since the historical run;
    drift must be reported, not normalized away.
11. **Cross-chain worst-case protection** — optimize worst-chain margin rather
    than only the pooled mean. Surface: portfolio selection. Risk: sacrificing
    broad efficiency for one idiosyncratic chain.
12. **Formal calibration candidate** — nominate only policies that improve
    replay efficiency across at least two mechanism families and survive
    ablation. Surface: PI contract. Risk: public replay correlation with a fresh
    coding hand is unknown until loop-external validation.

If a lookup suggests unavailable datasets, checkpoints, packages, APIs, or
runtime infrastructure, adapt the idea to the existing policy schema and local
replay. Record unavailable resources as notes; do not download or install them
during a run. Every source-backed proposal should record title, identifier or
URL, and how it changed the hypothesis.
