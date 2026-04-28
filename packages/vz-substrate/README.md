# vz-substrate

> R2 (Stable Substrate + Adaptive Controllers)

Frozen LLM substrate adapter, residual-activation capture, and the **bounded** self-modification surface used by the rare-heavy offline path.

## What it owns

| Surface | Notes |
|---|---|
| `SubstrateAdapter`, `OpenWeightResidualRuntime`, `SubstrateModule` | Frozen forward path; publishes `substrate` snapshot |
| `SubstrateSelfModModule`, `SubstrateSelfModSnapshot` | Default `SHADOW`; only the rare-heavy offline owner can flip it `ACTIVE` |
| `SyntheticOpenWeightResidualRuntime`, `build_transformers_runtime_with_fallback` | Default synthetic substrate; HF backends require `pip install vz-substrate[hf]` |

## Hard limits

- The default runtime is **frozen**. There is no token-level RL surface here — Internal RL (R4) lives in `vz-temporal` on `z_t`, not on tokens.
- `vz-substrate` does **not** assemble prompts. Prompt composition is the expression layer's job (`lifeform-expression` or `vz-runtime` adapter), not the substrate's.

## Install

```bash
pip install vz-substrate              # synthetic only
pip install "vz-substrate[hf]"        # + transformers/torch for HF backends
```
