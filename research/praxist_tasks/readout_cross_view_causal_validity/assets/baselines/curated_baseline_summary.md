# Curated baseline

The frozen layer-12 linear reader retains a useful same-view representation and
a causal target-logit effect above the matched random direction, but it fails
the cross-view instrument claim. Its weakest view is Chinese open-form at
balanced accuracy `0.375`; calibration also misses the `0.22` Brier gate at
`0.2294666087`. The correct development exit is therefore `DOMAIN_LOCAL`, not
PASS and not a collapsed instrument.

The initialization layer-14 linear canary demonstrates that the task's
confirmed lane is mechanically reachable: same-view accuracy `0.875`,
cross-view accuracy `0.825`, Cohen's d `2.8227947219`, and causal effect
`0.0866762400`. Its minimum margin is nevertheless exactly zero because one
view sits on the `0.5` boundary and random-control separation is only
`0.0196564595`. Praxist should replicate, ablate neighboring layers and pooling,
and seek positive worst-gate headroom rather than treating this one visible run
as settled science.

Both artifacts are public-development evidence. Formal qualification must bind
a separately authorized corpus, exact model and reader lineage, frozen protocol,
and named-human review outside Praxist. No development score may become PE,
credit, ModificationGate evidence by itself, SHADOW authorization, or ACTIVE
wiring.
