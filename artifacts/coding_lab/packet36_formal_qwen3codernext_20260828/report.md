# coding-lab Packet 3.6 episode-outcome gate

- run_id: `packet36_formal_qwen3codernext_20260828` (tier: formal)
- pass rates: {'noop': 0.43125, 'always_on': 0.45625, 'random_gate': 0.45625, 'table_gate': 0.45625}
- timing_vs_noop: mean +0.0250, 5% lower -0.0062
- timing_vs_random: mean +0.0000, 5% lower -0.0188
- always_vs_noop: mean +0.0250, 5% lower +0.0063
- table_vs_always_report_only: mean +0.0000, 5% lower -0.0188
- verdicts: {'outcome_timing_gate': False, 'placement_gate': False, 'intervention_gate': True}
- mechanism: {'episodes_total': 640, 'opportunity_episodes': 300, 'steer_decided_episodes': 178, 'trigger_delivered_episodes': 178, 'trigger_delivery_rate': 1.0, 'steered_pass_rate_wilson95': [0.935938, 0.987943]}

Claim boundary: action-level timing on oracle pass rate; residual Steerable not lifted.
