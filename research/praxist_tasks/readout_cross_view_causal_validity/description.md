# Cross-View Relationship Readout and Causal-Patch Research

Volvence currently has evidence that relationship conditions can be read from
frozen-model representations, but that evidence does not establish that the
readout survives changes in wording, question format, language, role order, or
counterfactual surface form. A same-view probe can therefore look accurate
while reading a format shortcut. Conversely, a discriminative direction can be
correlational and fail when it is patched back into the model.

This task freezes a small, balanced public-development corpus for two existing
relationship conditions: `agency_displacement` and `belonging_erasure`. Every
semantic group has six surface views. The evaluator fits only from the frozen
training groups, scores distinct evaluation groups, and measures same-view and
cross-view balanced accuracy, calibration, held-out Cohen's d, cross-view
identity retrieval, direction coherence, target-logit patch effect, and a
matched random-direction control.

The baseline linear reader is intentionally not a production artifact. Its
complete initialization run is `DOMAIN_LOCAL`: it retains same-view and causal
signal but fails the weakest cross-view requirement. An initialization-only
layer-14 linear canary reached the frozen development PASS boundary, proving
that the confirmed lane is mechanically reachable; that observation is public
development evidence and must be replicated or ablated by the research loop.

Praxist owns only development search, evidence retention, and lineage. The
formal protocol uses a separately authorized corpus and remains outside this
loop. Even a development `PASS` cannot publish a semantic snapshot, update a
reader owner, provide PE/credit, change `WiringLevel`, or authorize deployment.
