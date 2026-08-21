# Relationship Lab P1i training-only consumer calibration

- Calibration protocol: `080c908d7824b25081c501abbb6e76a2405a16508dbc0fb9d119b831447eef40`
- Report artifact: `c9382c1817114ea3b01684375546cbab3618ed38b655c46630e61af9e9e8a79a`
- Frozen consumer protocol: `938af6073c31346058a6df40877a48e68a449ce63acf941a7455049530a310bf`
- Selected candidate: `conditioned_match_v1`
- Ranking: `conditioned_match_v1, latent_partition_v1, counterfactual_contrast_v1`
- v4 inputs/outputs observed: `0 / 0`
- Formal hidden test / P2: `closed / disabled`

## Candidate metrics

| Candidate | Arm | Valid | Accuracy | Pair flip |
|---|---|---:|---:|---:|
| conditioned_match_v1 | prompt-steelman | 12/12 | 0.750 | 0.500 |
| conditioned_match_v1 | rag-steelman | 12/12 | 0.583 | 0.167 |
| conditioned_match_v1 | structured-state | 12/12 | 0.500 | 0.333 |
| latent_partition_v1 | prompt-steelman | 12/12 | 0.583 | 0.167 |
| latent_partition_v1 | rag-steelman | 12/12 | 0.500 | 0.000 |
| latent_partition_v1 | structured-state | 12/12 | 0.500 | 0.333 |
| counterfactual_contrast_v1 | prompt-steelman | 12/12 | 0.500 | 0.000 |
| counterfactual_contrast_v1 | rag-steelman | 12/12 | 0.583 | 0.167 |
| counterfactual_contrast_v1 | structured-state | 12/12 | 0.500 | 0.333 |

## Claim boundary

P1i reports bounded prompt-consumer calibration on the already-observed v3 training package and freezes exactly one external baseline consumer. It is not v4 qualification, Volvence advantage, Readable evidence, PE/credit learning, steering, formal held-out evidence, product evidence, or a complete four-capability claim.
