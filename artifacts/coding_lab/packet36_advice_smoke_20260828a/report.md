# coding-lab Packet 3.6 episode-outcome gate

- run_id: `packet36_advice_smoke_20260828a` (tier: development)
- pass rates: {'noop': 0.65625, 'always_on': 0.53125, 'random_gate': 0.5625, 'table_gate': 0.65625}
- timing_vs_noop: mean +0.0000, 5% lower +0.0000
- timing_vs_random: mean +0.0938, 5% lower +0.0312
- always_vs_noop: mean -0.1250, 5% lower -0.1875
- table_vs_always: mean +0.1250, 5% lower +0.0625
- verdicts: {'outcome_timing_gate': False, 'placement_gate': True, 'intervention_gate': False, 'avoidance_timing_gate': True}
- mechanism: {'episodes_total': 128, 'opportunity_episodes': 64, 'steer_decided_episodes': 32, 'trigger_delivered_episodes': 28, 'trigger_delivery_rate': 0.875, 'steered_pass_rate_wilson95': [0.364495, 0.691306], 'advice_drawn_by_action': {'investigate': 2, 'edit': 3, 'test': 5, 'submit': 6}, 'advice_executed_by_arm_action': {'noop': {'investigate': 0, 'edit': 0, 'test': 0, 'submit': 0}, 'always_on': {'investigate': 2, 'edit': 3, 'test': 5, 'submit': 6}, 'random_gate': {'investigate': 1, 'edit': 1, 'test': 3, 'submit': 4}, 'table_gate': {'investigate': 1, 'edit': 2, 'test': 4, 'submit': 0}}}

Claim boundary: action-level timing on oracle pass rate; residual Steerable not lifted.
