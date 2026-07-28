# ETA Gate 2 残差因果证据

- 判定：`mechanism-supported`
- substrate：`Qwen/Qwen2.5-0.5B-Instruct`
- seeds：`[0]`
- identity 对 strongest control 的 strong-success 差：`0.000000`
- identity 对 best control 的 continuation-NLL 改善：`-0.000163`
- identity 对 zero 的真实 downstream-effect 差：`0.000319`
- mechanism gates：`{'identity_exact': True, 'zero_exact': True, 'shuffle_permutation_nonidentity': True, 'shuffle_informative_steps_nonidentity': True, 'reverse_exact': True, 'real_open_weight_backend': True, 'hook_coverage': True, 'fallback_rate_zero': True, 'prefix_intervention_protocol': True, 'identity_effect_is_measurable_vs_zero': True, 'shared_policy_checkpoint_matched': True}`
- causal gates：`{'continuation_scores_available_all_arms': True, 'identity_continuation_nll_beats_controls': False}`

本包只判 Gate 2 的残差注入机制与 matched-control 因果差。
它不声称 Gate 2 longitudinal 已完成，也不产生 thesis-retained。
