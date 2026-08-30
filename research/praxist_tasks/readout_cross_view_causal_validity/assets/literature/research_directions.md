# High-Value Research Directions

These directions are agenda context, not measured task truth.

1. **Repair the weakest view, not only the mean.** The frozen baseline is
   domain-local because the weakest language/view slice fails. Compare
   original-only training with coherent multi-view directions, but preserve
   held-out semantic-group separation and calibration.
2. **Separate readout quality from actuation geometry.** Late layers can have a
   strong future-token patch effect while weakly classifying the condition;
   earlier discriminative layers can patch in the wrong direction. Search for
   configurations that clear both gates rather than optimizing either alone.
3. **Replicate and ablate the layer-14 linear canary.** Initialization observed
   a development PASS exactly at the worst-view boundary. Change one dimension
   at a time, test neighboring layers and pooling, and retain failures that
   explain whether the result is stable or accidental.
4. **Use random and semantic falsifiers.** A matched random direction must not
   reproduce the target-logit effect. Diagnose any seed or layer where random
   separation vanishes instead of tuning the threshold after seeing outcomes.
5. **Treat J-Lens-like output directions honestly.** The task's proxy uses the
   frozen output-token difference. It can establish a causal upper-bound clue,
   not an open-vocabulary thought decoder. Compare it with learned and
   centroid directions without inflating its claim.
6. **Prefer constrained Pareto improvements.** Improve qualification margin,
   weakest-view accuracy, calibration, and reverse-effect rate together. A
   higher average cross-view score that damages the causal or random-control
   gate is not an improvement.

Primary local sources are the Volvence continual-learning field map, reusable
mechanisms, unsolved-problem register, evidence roadmap, negative-results
register, and source ledger under `research/continual-deep-learning-2026-08/`.
External literature is contextual only; it cannot override evaluator facts or
provision new models, data, services, or dependencies during this task.
