# Relationship Lab P1b development calibration

- artifact_id: `10d120f49b442803cccec53c534e8f3c868ee644c0674439ede000d8dedd3a87`
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
| prompt-steelman | 8/8 | 0.750 | 0.500 | 17184 |
| rag-steelman | 8/8 | 0.500 | 0.000 | 8416 |
| structured-state | 8/8 | 0.500 | 0.500 | 8328 |

P1b is development-only evidence. Formal preregistration and secret heldout remain closed.
