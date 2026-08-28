# coding-lab Packet 3.6 episode-outcome gate

- run_id: `packet36_v21_formal_qwen3codernext_20260828` (tier: formal)
- pass rates: {'noop': 0.4666666666666667, 'always_on': 0.29583333333333334, 'random_gate': 0.36666666666666664, 'table_gate': 0.4125}
- timing_vs_noop: mean -0.0542, 5% lower -0.0875
- timing_vs_random: mean +0.0458, 5% lower +0.0167
- always_vs_noop: mean -0.1708, 5% lower -0.2208
- table_vs_always: mean +0.1167, 5% lower +0.0792
- verdicts: {'outcome_timing_gate': False, 'placement_gate': True, 'intervention_gate': False, 'avoidance_timing_gate': True}
- mechanism: {'episodes_total': 960, 'opportunity_episodes': 960, 'steer_decided_episodes': 373, 'trigger_delivered_episodes': 372, 'trigger_delivery_rate': 0.9973190348525469, 'steered_pass_rate_wilson95': [0.390176, 0.49041], 'advice_drawn_by_action': {'investigate': 73, 'edit': 58, 'test': 57, 'submit': 52}, 'advice_executed_by_arm_action': {'noop': {'investigate': 0, 'edit': 0, 'test': 0, 'submit': 0}, 'always_on': {'investigate': 73, 'edit': 58, 'test': 57, 'submit': 52}, 'random_gate': {'investigate': 18, 'edit': 14, 'test': 15, 'submit': 18}, 'table_gate': {'investigate': 23, 'edit': 21, 'test': 24, 'submit': 0}}}

Claim boundary: action-level timing on oracle pass rate; residual Steerable not lifted.
