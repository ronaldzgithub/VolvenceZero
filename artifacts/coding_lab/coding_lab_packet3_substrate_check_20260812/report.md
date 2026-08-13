# Packet 3 前置 a：基底预检

- model: `Qwen/Qwen2.5-Coder-1.5B-Instruct` device=cpu dtype=float32
- overall: PASS
- recommendation: proceed with Qwen/Qwen2.5-Coder-1.5B-Instruct fp32 geometry (hidden=1536, injection_layer=13)

| verdict | value |
|---|---|
| disk_ok | True |
| blocks_resolved | True |
| width_matches | True |
| capture_ok | True |
| scorer_ok | True |
| fp32_fits | True |

| check | value |
|---|---|
| disk_free_gib | 29.73 |
| load_seconds | 30.4 |
| hidden_size | 1536 |
| parameter_count | 1543714304 |
| capture_seconds | 28.86 |
| capture_steps | 30 |
| capture_widths | [1536] |
| capture_layer_indices | [13, 14, 15] |
| scorer_seconds | 11.43 |
| scorer_injection_layer | 13 |
| baseline_action_nll | [1.4393, 1.8343] |
| peak_rss_gib | 5.99 |
