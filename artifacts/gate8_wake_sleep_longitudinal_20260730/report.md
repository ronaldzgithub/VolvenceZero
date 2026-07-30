# Gate 8 cross-session wake/sleep evidence

- status: `longitudinal-supported`
- mechanism passed: `True`
- longitudinal passed: `True`
- settled transitions: `6120` arm-transitions
- constructor restarts per arm/seed: `50`

## Primary effects

- `cold_start_loss_reduction_vs_no_sleep`: `0.567363`
- `callback_consistency_gain_vs_no_sleep`: `1.000000`
- `delayed_payoff_gain_vs_no_sleep`: `0.567363`
- `payoff_margin_vs_memory_only`: `0.167363`
- `payoff_margin_vs_policy_only`: `0.400000`
- `single_owner_payoff_margin`: `0.167363`
- `full_cold_start_loss`: `0.001099`
- `full_callback_consistency`: `1.000000`
- `full_delayed_payoff`: `0.998901`
- `maximum_full_owner_state_drift`: `0.467850`

## Claim boundary

- This packet tests deterministic next-session callback and temporal alignment on frozen real-substrate source signals. It does not provide human relationship-quality ground truth or authorize production promotion.
