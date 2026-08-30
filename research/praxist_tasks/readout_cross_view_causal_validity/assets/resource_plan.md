# Resource Plan

## Frozen environment

- Host observed during initialization: Apple arm64, 10 logical CPUs, about
  25.8 GiB memory, with MPS available.
- Task runtime: repository `.venv` with Python, PyTorch, Transformers, NumPy,
  scikit-learn, and jsonschema already installed.
- Model: exact locally cached `Qwen/Qwen2.5-0.5B-Instruct` revision
  `7ae557604adf67be50417f59c2c2f167def9a775`; every required model and tokenizer
  file is SHA-256 checked before evaluation. Network download is disabled.
- Storage was near capacity during initialization, so the evaluator persists
  only one compact summary and never writes activation caches or model copies.

## Measured evaluator cost

- Complete linear layer-12 baseline: 48 training-view rows plus 48 held-out
  rows, two causal-patch passes, about 15 seconds wall time on MPS.
- Peak memory is dominated by one 0.5B-parameter float32 model. The central
  scheduler therefore permits one experiment at a time on this host.
- Preliminary still loads the same model and training split, so it is a wiring
  check rather than a proportional cost estimate.
- The close-grade planning estimate is one minute with a `1.5` safety factor;
  this leaves ample drain time inside the generation boundary.

## Scheduling and recovery

- Profile: `mps_readout`; accelerator pressure domains are accelerator, memory,
  and I/O; one peer may hold one evaluator process.
- Cohort: 4; generations: 3; 1.5-hour generation bound, leaving at least a
  30-minute margin beyond the maximum synthesis trigger.
- DIG runs only before generation zero writes. QD remains enabled in later
  generations so mechanism, layer, pooling, and falsifier diversity do not
  collapse into a numeric sweep.
- A completed Praxist generation is the only resume checkpoint. Forge binds
  pause/resume to the exact Request, Approval, and last event SHA; resume never
  crops task artifacts or uses `--force`.

## Known limits

- The synthetic public-development corpus is intentionally small and visible.
- It does not reuse the sealed relationship qualification corpus and cannot
  support a production claim.
- The `j_lens_like` family is a future-token unembedding proxy, not a full
  average-Jacobian J-Lens implementation; findings must preserve that name and
  limitation.
