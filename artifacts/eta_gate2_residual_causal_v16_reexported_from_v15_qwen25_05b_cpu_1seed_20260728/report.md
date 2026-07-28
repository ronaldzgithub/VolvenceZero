# ETA Gate 2 残差因果证据

- 判定：`mechanism-supported`
- substrate：`Qwen/Qwen2.5-0.5B-Instruct`
- seeds：`[0]`
- 训练更新：`20`
- identity 对 development-heldout best control 的 continuation-NLL 改善：`-0.010073`
- identity 对 eval best control 的 continuation-NLL 改善：`-0.022188`
- identity 对 fresh confirmation best control 的 continuation-NLL 改善：`None`
- mechanism gates：`{'identity_exact': True, 'zero_exact': True, 'shuffle_permutation_nonidentity': True, 'shuffle_informative_steps_nonidentity': True, 'reverse_exact': True, 'real_open_weight_backend': True, 'hook_coverage': True, 'fallback_rate_zero': True, 'prefix_intervention_protocol': True, 'identity_effect_is_measurable_vs_zero': True, 'heldout_identity_control_exposure': True, 'shared_policy_checkpoint_matched': True, 'continuation_pe_training_isolated_to_identity': True, 'real_residual_ssl_bootstrap_used': True, 'causal_action_head_updated': True, 'continuation_pe_updates_policy_only': True, 'continuation_counterfactual_grid_used': True}`
- causal gates：`{'eval_continuation_scores_available_all_arms': True, 'identity_eval_nll_beats_controls': False, 'fresh_confirmation_scores_available_all_arms': False, 'fresh_confirmation_split_locked': False, 'identity_confirmation_nll_beats_controls': False}`

本包从 v15 冻结 benchmark 机械重导出，仅按 v16 契约重算 split 聚合与 verdict；未重跑模型，raw predictions/outcomes 未变。
现有 heldout 已用于开发观察，不是 fresh confirmation；本包只支持 mechanism-supported。
