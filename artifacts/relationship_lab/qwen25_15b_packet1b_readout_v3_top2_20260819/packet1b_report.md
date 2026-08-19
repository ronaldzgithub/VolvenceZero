# Relationship Lab P1b development calibration

- artifact_id: `9cd149b1e8c3f74d54d0cbaf72c216edfdaeba2979829925bc50c8ac3d60c4e8`
- verdict: **baseline_underqualified**
- gate1_passed: **false**
- all_readouts_valid: **true**
- saturated_arms: `none`
- readout_prompt_sha256: `02b13cb111d9a329729c1e91737f8c60186d20fa7cce9c7e60fb49b87bd62c52`
- readout_request_template_sha256: `e0c5a4c10f941ee9b91e637c8129540f53d10f56a677e9022ab1c28305478dc9`
- readout_schema_sha256: `afa852a2ee43b55cd10e761d5bb5b6a56b546aaa626f9291edb550e3220c40ab`
- compiler_version: `relationship-evidence-argmax.v1`

| Arm | valid | accuracy | pair flip | prompt tokens |
|---|---:|---:|---:|---:|
| prompt-steelman | 8/8 | 0.250 | 0.000 | 14230 |
| rag-steelman | 8/8 | 0.500 | 0.000 | 5398 |
| structured-state | 8/8 | 0.500 | 0.000 | 5246 |

P1b is development-only evidence. Formal preregistration and secret heldout remain closed.
