# Baseline performance status

- Status: `public_development_complete; initialization_canary_complete`.
- Exact model: `Qwen/Qwen2.5-0.5B-Instruct` revision
  `7ae557604adf67be50417f59c2c2f167def9a775`, loaded locally with every
  required file SHA-256 checked.
- Baseline: `baseline_linear_l12_last_token`, complete 8/8 held-out groups and
  6/6 views, summary SHA-256
  `6af038b25e1a47165504a80a7530fb16dac7c028f5d86fe829f5bb3a38b02a10`.
- Baseline result: `DOMAIN_LOCAL`; qualification margin `-0.25`, same-view
  balanced accuracy `0.75`, cross-view `0.675`, weakest view `0.375`, held-out
  Cohen's d `1.1101207395`, causal effect `0.0495384534`, and matched-random
  separation `0.0599618753`.
- Initialization canary: `initialization_canary_linear_l14_last_token`, summary
  SHA-256 `cef8736f98c024c3fcd327ae96d2b9f868574700e411a51923fbd0221cb88d58`.
  It reached development `PASS` with qualification margin exactly `0.0`; its
  weakest-view accuracy exactly equals the frozen `0.5` threshold, so it is a
  replication/ablation lead rather than robust formal evidence.
- Neither result uses the sealed relationship qualification corpus, supplies a
  learning signal, publishes an owner snapshot, or authorizes runtime wiring.
