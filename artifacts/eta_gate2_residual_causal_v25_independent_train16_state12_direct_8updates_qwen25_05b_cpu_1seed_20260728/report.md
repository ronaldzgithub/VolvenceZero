# ETA Gate 2 残差因果证据

- 判定：`mechanism-supported`
- substrate：`Qwen/Qwen2.5-0.5B-Instruct`
- seeds：`[0]`
- identity 对 strongest control 的 strong-success 差：`0.000000`
- identity 对 development-heldout best control 的 continuation-NLL 改善：`-0.001057`
- identity 对 eval best control 的 continuation-NLL 改善：`-0.017923593521118164`
- identity 对 fresh confirmation best control 的 continuation-NLL 改善：`None`
- identity 对 zero 的真实 downstream-effect 差：`0.001706`
- mechanism gates：`{'identity_exact': True, 'zero_exact': True, 'shuffle_permutation_nonidentity': True, 'shuffle_informative_steps_nonidentity': True, 'reverse_exact': True, 'real_open_weight_backend': True, 'hook_coverage': True, 'fallback_rate_zero': True, 'prefix_intervention_protocol': True, 'identity_effect_is_measurable_vs_zero': True, 'heldout_identity_control_exposure': True, 'shared_policy_checkpoint_matched': True, 'continuation_pe_training_isolated_to_identity': True, 'real_residual_ssl_bootstrap_used': True, 'causal_action_head_updated': True, 'observation_state_separate_from_actuator': True, 'continuation_pe_updates_policy_only': True, 'continuation_counterfactual_grid_used': True}`
- causal gates：`{'eval_continuation_scores_available_all_arms': True, 'identity_eval_nll_beats_controls': False, 'fresh_confirmation_scores_available_all_arms': False, 'fresh_confirmation_split_locked': False, 'identity_confirmation_nll_beats_controls': False}`

本包只判 Gate 2 的残差注入机制与 matched-control 因果差。
它不声称 Gate 2 longitudinal 已完成，也不产生 thesis-retained。
