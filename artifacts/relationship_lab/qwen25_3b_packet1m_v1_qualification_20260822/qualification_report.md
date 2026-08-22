# P1m first qualification

- report id: `9580ddffb7113d386f01137837be0335e0b82ff8d7cfc9e333e86f73d249fc56`
- verdict: `prompt_steelman_baseline_too_weak`
- qualification passed: `false`
- scenario versioning closed: `true`

| arm | valid | accuracy | accuracy Wilson lower | pair flip | flip Wilson lower |
|---|---:|---:|---:|---:|---:|
| prompt-steelman-forced-choice | 48/48 | 0.500 | 0.385 | 0.000 | 0.000 |
| rag-steelman-observational | 48/48 | 0.500 | 0.385 | 0.000 | 0.000 |
| structured-state-named-reader | 48/48 | 0.958 | 0.882 | 1.000 | 0.899 |

This report only decides whether the generated P1m development instrument has an informative strong-baseline range and a functioning structured-state path. Passing is not formal held-out evidence, Volvence advantage, or proof of Appendable, Readable, Learnable, Steerable, production ACTIVE, safety, or product value. Failure permanently closes scenario versioning for this frozen P1m recipe.
