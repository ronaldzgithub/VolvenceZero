# ETA Gate 2 残差因果证据

- 判定：`wiring-ready`
- substrate：`Qwen/Qwen2.5-0.5B-Instruct`
- seeds：`[0]`
- identity 对 strongest control 的 strong-success 差：`0.000000`
- identity 对 zero 的真实 downstream-effect 差：`-0.000188`
- mechanism gates：`{'identity_exact': True, 'zero_exact': True, 'shuffle_permutation_nonidentity': False, 'reverse_exact': True, 'real_open_weight_backend': True, 'hook_coverage': True, 'fallback_rate_zero': True, 'prefix_intervention_protocol': True}`
- causal gates：`{'identity_effect_beats_zero': False, 'identity_strong_success_beats_controls': False, 'identity_terminal_success_not_worse': True}`

本包只判 Gate 2 的残差注入机制与 matched-control 因果差。
它不声称 Gate 2 longitudinal 已完成，也不产生 thesis-retained。
