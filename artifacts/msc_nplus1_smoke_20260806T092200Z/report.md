# MSC N+1 prediction research report

- Evidence level: `pilot`
- Thesis status: `INELIGIBLE_PILOT`
- Formal experiment executed: `false`
- N+1 target owner: `vz-substrate` / `latest-token-selected-layer-residual-l2.v1`
- N+1 target model: `Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775#857fff1d`
- R3 same-substrate context: `True`
- R4 complete Volvence runtime: `True`
- R5 temporal capacity: `False`
- Temporal-capacity exit: `INELIGIBLE_PILOT`
- Selected temporal_n_z: `3`
- Context truncation policy: `deny`
- Thesis exit: `INELIGIBLE_PILOT`
- Forward-head capacity exit: `INELIGIBLE_PILOT`
- Selected PE forward_head_n_z: `256`
- Longest-session cosine advantage vs long context: `0.003794`
- Token ratio: `1.184790`
- Latency ratio: `2.980832`
- Zero-norm heldout predictions: `0`
- Resume checkpoints: `complete`

The primary test is only Volvence versus the zero-truncation 
long_context control at session five. Stateless and 
summary_retrieval are matched-eligibility controls only.
