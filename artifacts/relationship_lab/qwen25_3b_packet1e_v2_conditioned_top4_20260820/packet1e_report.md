# Relationship Lab P1e v2 consumer qualification

- artifact_id: `232afebb56afb5e457af3d7ca4ccfc560cc417447defcb6d265263085fad8693`
- consumer_protocol_id: `5221909debd8b0248c83332589c2681270118dc54b7014654db2d627ca2fbd1e`
- dataset_fingerprint: `d8e002d6d529476bf29622d4872afb0b1d7fec9d9c2e5942ecb830c8428b660b`
- model_id: `qwen2.5-3b-instruct`
- gate0_passed: **true**
- baseline_accuracy: **0.500**
- verdict: **rewrite_public_evidence_contract**

| Arm | accuracy | pair flip |
|---|---:|---:|
| prompt-steelman | 0.625 | 0.250 |
| rag-steelman | 0.250 | 0.250 |
| structured-state | 0.625 | 0.250 |

## Required next action

Repair public evidence/readout clarity before adding PE learning or steering.

## Claim boundary

P1e reports strong-baseline qualification or scenario saturation on the public synthetic v2 development split. It does not prove Volvence advantage, Appendable/Readable/Learnable/Steerable, formal held-out superiority, or product value.
