# Baseline performance status

- Status: `public_development_replay_complete; launch_canary_complete`.
- Source revision: `2f3d56b388f9cb2f4702aa7da2670e71fdcf5efd`.
- Frozen formal report SHA-256:
  `9ad76d2f48b32ab8a40bd5288b0dfd52814f6b79970275641a960ce51ace0e42`.
- Formal provenance baseline: 24 chains × 10 episodes × 3 arms; measured
  previously and not rerun by this task.
- Public task baseline: exact legacy policy replay over chains 0–7; no coding
  hand or provider call. Canonical summary SHA-256:
  `44d3ff70c9e931694220c702e7a8eb4ab6b2d56c4283628e0bbc00d6910f58b7`.
- Complete replay: 8/8 units, 72 post-first-episode contexts, context ratio
  `0.1233350276`, scaling margin `-0.0233350276`, selected-line retention
  `0.9720279720 / 0.9846153846`, strict-budget pass rate `0.0`.
- The failure is valid negative development evidence. It does not invalidate
  the historical coding-quality result and does not authorize deployment.
