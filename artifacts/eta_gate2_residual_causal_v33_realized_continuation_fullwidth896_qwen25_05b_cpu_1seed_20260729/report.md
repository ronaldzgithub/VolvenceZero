# ETA Gate 2 残差因果证据

- 判定：`mechanism-supported`
- substrate：`Qwen/Qwen2.5-0.5B-Instruct`
- seeds：`[0]`
- identity 对 strongest control 的 strong-success 差：`0.000000`
- identity 对 development-heldout best control 的 continuation-NLL 改善：`-0.001588`
- identity 对 eval best control 的 continuation-NLL 改善：`-0.01997445179865842`
- identity 对 fresh confirmation best control 的 continuation-NLL 改善：`None`
- identity 对 zero 的真实 downstream-effect 差：`0.002024`
- mechanism gates：`{'identity_exact': True, 'zero_exact': True, 'shuffle_permutation_nonidentity': True, 'shuffle_informative_steps_nonidentity': True, 'reverse_exact': True, 'real_open_weight_backend': True, 'hook_coverage': True, 'fallback_rate_zero': True, 'prefix_intervention_protocol': True, 'identity_effect_is_measurable_vs_zero': True, 'heldout_identity_control_exposure': True, 'shared_policy_checkpoint_matched': True, 'continuation_pe_training_isolated_to_identity': True, 'real_residual_ssl_bootstrap_used': True, 'causal_action_head_updated': True, 'observation_state_separate_from_actuator': True, 'continuation_pe_updates_policy_only': True, 'continuation_counterfactual_grid_used': True, 'environment_outcome_target_active': True, 'environment_forward_observed': True, 'environment_outcome_reaches_pe_credit': True, 'self_nll_excluded_from_selector_target': True}`
- causal gates：`{'eval_environment_outcome_audit_positive': False, 'fresh_confirmation_environment_audit_available': False, 'fresh_confirmation_split_locked': False, 'confirmation_environment_outcome_audit_positive': False}`
- selector gates：`{'train_grouped_cv_predictions_available': True, 'frozen_eval_predictions_available': True, 'frozen_heldout_predictions_available': True, 'frozen_validation_predictions_available': True, 'train_independent_audit_available': True, 'validation_independent_audit_available': True, 'no_eval_updates_after_fit': True, 'selector_ready_for_shadow_injection': False}`
- signal gates：`{'train_counterfactual_grid_present': True, 'validation_counterfactual_grid_present': True, 'train_oracle_transfer_exceeds_permutation_null': True, 'validation_oracle_transfer_exceeds_permutation_null': False}`
- reachable solution evidence（oracle 过置换零假设）：`False`
- oracle permutation-null diagnostics：`{'train': {'prefix_count': 180.0, 'observed_oracle_target_mean': 0.03710667623413934, 'transfer_oracle_audit_mean': 0.0037154134776857164, 'permutation_null_audit_mean': 0.001249365568763078, 'transfer_excess_over_null_mean': 0.0024660479089226384}, 'eval': {'prefix_count': 11.0, 'observed_oracle_target_mean': 0.02328140085393732, 'transfer_oracle_audit_mean': -0.01959150487726385, 'permutation_null_audit_mean': -0.017097421914092765, 'transfer_excess_over_null_mean': -0.002494082963171086}, 'heldout': {'prefix_count': 12.0, 'observed_oracle_target_mean': 0.02786421775817871, 'transfer_oracle_audit_mean': -0.004640102386474609, 'permutation_null_audit_mean': 0.0018179127664277044, 'transfer_excess_over_null_mean': -0.006458015152902313}, 'validation': {'prefix_count': 21.0, 'observed_oracle_target_mean': 0.045525607608613516, 'transfer_oracle_audit_mean': -0.002204168410528274, 'permutation_null_audit_mean': 0.0002051127421391473, 'transfer_excess_over_null_mean': -0.0024092811526674213}, 'confirmation': {'prefix_count': 0.0, 'observed_oracle_target_mean': 0.0, 'transfer_oracle_audit_mean': 0.0, 'permutation_null_audit_mean': 0.0, 'transfer_excess_over_null_mean': 0.0}}`
- selector shadow injection allowed：`False`

本包只判 Gate 2 的残差注入机制与 matched-control 因果差。
它不声称 Gate 2 longitudinal 已完成，也不产生 thesis-retained。
