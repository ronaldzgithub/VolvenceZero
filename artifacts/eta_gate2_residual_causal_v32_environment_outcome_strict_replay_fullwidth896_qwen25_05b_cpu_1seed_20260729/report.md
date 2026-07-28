# ETA Gate 2 v32 环境结果严格门重判

- 判定：`mechanism-supported`
- promotion allowed：`False`
- selector injection allowed：`False`
- reachable solution evidence：`True`
- kill conditions：`['eval_environment_outcome_audit_positive', 'fresh_confirmation_environment_audit_available', 'fresh_confirmation_split_locked', 'confirmation_environment_outcome_audit_positive']`
- eval audit selected credit：`-0.000131504148`
- heldout audit selected credit：`0.000138136808`
- validation audit selected credit：`0.000015486499`
- mechanism gates：`{'identity_exact': True, 'zero_exact': True, 'shuffle_permutation_nonidentity': True, 'shuffle_informative_steps_nonidentity': True, 'reverse_exact': True, 'real_open_weight_backend': True, 'hook_coverage': True, 'fallback_rate_zero': True, 'prefix_intervention_protocol': True, 'identity_effect_is_measurable_vs_zero': True, 'heldout_identity_control_exposure': True, 'shared_policy_checkpoint_matched': True, 'continuation_pe_training_isolated_to_identity': True, 'real_residual_ssl_bootstrap_used': True, 'causal_action_head_updated': True, 'observation_state_separate_from_actuator': True, 'continuation_pe_updates_policy_only': True, 'continuation_counterfactual_grid_used': True, 'environment_outcome_target_active': True, 'environment_forward_observed': True, 'environment_outcome_reaches_pe_credit': True, 'self_nll_excluded_from_selector_target': True}`
- causal gates：`{'eval_environment_outcome_audit_positive': False, 'fresh_confirmation_environment_audit_available': False, 'fresh_confirmation_split_locked': False, 'confirmation_environment_outcome_audit_positive': False}`
- selector gates：`{'train_grouped_cv_predictions_available': True, 'frozen_eval_predictions_available': True, 'frozen_heldout_predictions_available': True, 'frozen_validation_predictions_available': True, 'train_independent_audit_available': True, 'validation_independent_audit_available': True, 'no_eval_updates_after_fit': True, 'selector_ready_for_shadow_injection': False}`
- signal gates：`{'train_counterfactual_grid_present': True, 'validation_counterfactual_grid_present': True, 'train_oracle_transfer_exceeds_permutation_null': True, 'validation_oracle_transfer_exceeds_permutation_null': True}`

本报告只读重放 v31 已完成的原始候选记录；未重新执行模型 forward，未修改任何候选 measurement、PE 或 credit。
v32 只收紧 gate：所有冻结分区 audit 必须为正，并要求 target oracle 在独立 audit 上超过 permutation null。
