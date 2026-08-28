# coding-lab Packet 3.6 episode-outcome gate

- run_id: `packet36_smoke_20260828a` (tier: development)
- pass rates: {'noop': 0.65625, 'always_on': 0.65625, 'random_gate': 0.65625, 'table_gate': 0.65625}
- timing_vs_noop: mean +0.0000, 5% lower +0.0000
- timing_vs_random: mean +0.0000, 5% lower +0.0000
- always_vs_noop: mean +0.0000, 5% lower +0.0000
- table_vs_always_report_only: mean +0.0000, 5% lower +0.0000
- verdicts: {'outcome_timing_gate': False, 'placement_gate': False, 'intervention_gate': False}
- mechanism: {'episodes_total': 128, 'opportunity_episodes': 64, 'steer_decided_episodes': 37, 'trigger_delivered_episodes': 32, 'trigger_delivery_rate': 0.8648648648648649, 'steered_pass_rate_wilson95': [0.598827, 0.866386]}

Claim boundary: action-level timing on oracle pass rate; residual Steerable not lifted.
