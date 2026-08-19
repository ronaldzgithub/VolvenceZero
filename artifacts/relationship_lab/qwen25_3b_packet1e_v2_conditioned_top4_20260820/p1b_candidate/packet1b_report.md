# Relationship Lab P1b development calibration

- artifact_id: `073501caf5513ef5ed75872a748eae8eba0f708a5cea6ed22ab3acbc90f671a8`
- verdict: **baseline_underqualified**
- gate1_passed: **false**
- all_readouts_valid: **true**
- saturated_arms: `none`
- readout_prompt_sha256: `9687c5043029502b0787cd88758d06c1c0541338ed3c50068ad4824ce25fd4e5`
- readout_request_template_sha256: `55db60c37825da4999931345c1c973c2ea8fe54634a9508317284a02fd9bbde3`
- readout_schema_sha256: `afa852a2ee43b55cd10e761d5bb5b6a56b546aaa626f9291edb550e3220c40ab`
- compiler_version: `relationship-evidence-argmax.v1`

| Arm | valid | accuracy | pair flip | prompt tokens |
|---|---:|---:|---:|---:|
| prompt-steelman | 8/8 | 0.625 | 0.250 | 16483 |
| rag-steelman | 8/8 | 0.250 | 0.250 | 7715 |
| structured-state | 8/8 | 0.625 | 0.250 | 7627 |

P1b is development-only evidence. Formal preregistration and secret heldout remain closed.
