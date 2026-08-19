# Relationship Lab P1b development calibration

- artifact_id: `7eb361d5ed8d6ac8c82d12382cdf7eb9bd2edfb9e920ceee7f14e41956c09b36`
- verdict: **dataset_saturated**
- gate1_passed: **false**
- all_readouts_valid: **true**
- saturated_arms: `prompt-steelman, rag-steelman`
- readout_prompt_sha256: `02b13cb111d9a329729c1e91737f8c60186d20fa7cce9c7e60fb49b87bd62c52`
- readout_request_template_sha256: `e0c5a4c10f941ee9b91e637c8129540f53d10f56a677e9022ab1c28305478dc9`
- readout_schema_sha256: `afa852a2ee43b55cd10e761d5bb5b6a56b546aaa626f9291edb550e3220c40ab`
- compiler_version: `relationship-evidence-argmax.v1`

| Arm | valid | accuracy | pair flip | prompt tokens |
|---|---:|---:|---:|---:|
| prompt-steelman | 8/8 | 1.000 | 1.000 | 14230 |
| rag-steelman | 8/8 | 1.000 | 1.000 | 5398 |
| structured-state | 8/8 | 1.000 | 1.000 | 5246 |

P1b is development-only evidence. Formal preregistration and secret heldout remain closed.
