# Relationship Lab P1g v3 consumer qualification

- artifact_id: 9d7f05b574bafb21641d22c766fe31c4656c09bf6f5e04493474eee6c694e3c8
- consumer_protocol_id: 8e08d488382442f364aae102d80c268c8c23927d547f64c1e79cb0a87f0f52c6
- source_p1f_report_artifact_id: a231e2096b2c4b5fcf3e8b36fd099d0955ce2e355e793d38f5ed8e87a047ecbd
- dataset_fingerprint: 35b8c46e6fd5810779aff38ed935d8c4f0741bf7d496d2e3eec85f93fbf2134f
- model_id: qwen2.5-3b-instruct
- gate0_passed: true
- baseline_accuracy: 0.500
- verdict: consumer_still_underqualified

| Arm | accuracy | pair flip |
|---|---:|---:|
| prompt-steelman | 0.750 | 0.500 |
| rag-steelman | 0.500 | 0.000 |
| structured-state | 0.500 | 0.500 |

## Required next action

Record that semantically legible public evidence did not qualify this frozen Qwen consumer. Do not tune on these outputs; design a new versioned consumer-training split before another attempt.

## Claim boundary

P1g reports same-substrate strong-baseline qualification or saturation on the public synthetic v3 development split. It does not prove Volvence advantage, human readability, Appendable/Readable/Learnable/Steerable, formal held-out superiority, or product value.
