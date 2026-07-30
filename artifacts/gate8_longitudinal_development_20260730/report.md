# Gate 8 cross-session wake/sleep evidence

- status: `development-diagnostic`
- mechanism passed: `False`
- longitudinal passed: `False`
- settled transitions: `240` arm-transitions
- constructor restarts per arm/seed: `50`

## Primary effects

- `cold_start_loss_reduction_vs_no_sleep`: `0.543472`
- `callback_consistency_gain_vs_no_sleep`: `1.000000`
- `delayed_payoff_gain_vs_no_sleep`: `0.543472`
- `payoff_margin_vs_memory_only`: `0.143472`
- `payoff_margin_vs_policy_only`: `0.400000`
- `single_owner_payoff_margin`: `0.143472`
- `full_cold_start_loss`: `0.025226`
- `full_callback_consistency`: `1.000000`
- `full_delayed_payoff`: `0.974774`
- `maximum_full_owner_state_drift`: `0.417603`

## Claim boundary

- This packet tests deterministic next-session callback and temporal alignment on frozen real-substrate source signals. It does not provide human relationship-quality ground truth or authorize production promotion.
