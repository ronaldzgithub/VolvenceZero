# Relationship Lab Gate 0 calibration

- artifact_id: `740efc4ac487cf45544b58dea128f2b3b31cb107233cc6428bfdbf50eaba7b2c`
- dataset_fingerprint: `35b8c46e6fd5810779aff38ed935d8c4f0741bf7d496d2e3eec85f93fbf2134f`
- machinery_ready: **true**
- gate0_passed: **true**

## Checks

- `mirrored_counterfactual` — **pass**: Every mirrored pair has byte-identical current input and opposite non-noop optimal actions.
- `reactive_action_effect` — **pass**: Selected actions physically reach the environment and cause a configured minimum typed-outcome effect.
- `environment_determinism` — **pass**: Same inputs settle identically; changing the action changes the content-addressed environment evidence.
- `sut_truth_leakage` — **pass**: SUT payloads contain public histories only; latent ids, preferred actions, profiles, pair ids, and future outcomes are absent.
- `decision_trace_contract` — **pass**: Pre-action bet and post-action settlement round-trip through one content-addressed frozen sidecar.
- `frozen_baseline_non_saturation` — **pass**: Frozen real-substrate stateless/raw baseline is inside the configured non-saturation ceiling.

## Claim boundary

P0 only establishes instrument and evidence-contract readiness. This report includes a frozen real-substrate baseline, but it is not formal hidden-test or four-capability evidence unless the preregistration and secret heldout are separately frozen.
