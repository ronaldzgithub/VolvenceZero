# Readout Cross-View Causal Validity

This is a runnable Praxist task project for Packet P0 of the Volvence
continual-learning evidence roadmap. It compares bounded residual-reader
configurations on a frozen public-development corpus and reports both
cross-view discrimination and causal target-logit effects.

A candidate directory contains only `reader.json`. Praxist may vary the
declared reader family, residual layer, pooling, regularization, bounded dose,
seed, and view aggregation. It may not modify the corpus, model binding,
evaluator, thresholds, task contract, Volvence source, formal protocol, or
runtime wiring.

Run a non-ranking preliminary check:

```bash
"../../../.venv/bin/python" evaluations/cross_view/run.py \
  --variant-dir assets/baseline \
  --output-dir scratch/baseline-preliminary \
  --mode preliminary
```

Run the complete development protocol:

```bash
"../../../.venv/bin/python" evaluations/cross_view/run.py \
  --variant-dir assets/baseline \
  --output-dir scratch/baseline-complete \
  --mode complete
```

Neither command performs formal qualification or authorizes a Volvence
candidate, `ModificationGate`, SHADOW, ACTIVE, or production wiring.
