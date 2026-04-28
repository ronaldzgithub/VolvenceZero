# lifeform-domain-emogpt

Vertical for the **relationship-aware companion** archetype — what was historically called EmoGPT.

A vertical is **data + light glue** that compiles into the kernel's owner snapshots:

| Asset | Compiles into kernel surface |
|---|---|
| `DomainExperiencePackage` (knowledge / cases / playbooks / boundary hints) | `vz-application.domain_knowledge / case_memory / strategy_playbook / boundary_policy` |
| Regime priors (built into the package) | `vz-cognition.regime` warm-start |
| Scenario packs (`scenarios/*.json`) | `vz-cognition.evaluation` benchmark inputs |
| **Pre-trained bootstraps** (`bootstraps/*.snap`, `*.bs`) | `vz-temporal.MetacontrollerParameterSnapshot` + `vz-cognition.regime.RegimeBootstrap` |

## Public API

```python
from lifeform_domain_emogpt import (
    build_companion_package,           # DomainExperiencePackage data
    scenarios_dir,                     # path to scripted scenarios
    bootstraps_dir,                    # path to pre-trained artifacts
    load_companion_temporal_bootstrap, # MetacontrollerParameterSnapshot
    load_companion_regime_bootstrap,   # RegimeBootstrap
    build_companion_lifeform,          # ready-to-run Lifeform with everything wired
)
```

## "Vertical-shipped calibration" — what it is and why

The kernel ships flat / uniform defaults so it stays vertical-agnostic. Every concrete vertical **encodes its product priors** by:

1. Defining its scripted scenarios (`scenarios/*.json`).
2. Running `lifeform-super-loop` over those scenarios, which jointly trains the metacontroller and the regime classifier.
3. Saving the best-round artifacts into `bootstraps/`.
4. Shipping all of the above as wheel package data.

A product that wants the relationship-companion archetype just calls:

```python
from lifeform_domain_emogpt import build_companion_lifeform

life = build_companion_lifeform()
session = life.create_session(session_id="my-product-session")
result = session.run_turn("I have been feeling really stuck lately.")
```

…and gets the calibrated lifeform without ever running training itself. The kernel layer is untouched; everything domain-specific lives in this wheel.

Adding a new vertical (coding assistant, customer-service bot, teacher) is a new `lifeform-domain-*` package with its own `scenarios/` and `bootstraps/`. The kernel never knows which vertical is loaded. This is what proves trigger ② of `SPLIT.md` ("second consumer of the brain kernel").

## Updating the bootstraps

When you change scenarios, prior data, or the calibration loop, regenerate the artifacts:

```bash
lifeform-super-loop \
  --scenarios packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/scenarios \
  --rounds 3 \
  --save-temporal packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/bootstraps/companion-temporal.snap \
  --save-regime   packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/bootstraps/companion-regime.bs
```

Both files use magic-byte-prefixed pickle envelopes (`VZ-METASNAP\0` and `VZ-REGIMEBS\0`); reading them via the typed loaders fail-fast on schema-version drift, so we know when to retrain rather than load a stale artifact.

## Ablation

`build_companion_lifeform(use_temporal_bootstrap=False, use_regime_bootstrap=False)` returns the lifeform with neither bootstrap applied. Useful for evaluation harnesses comparing baseline vs. each axis vs. both.

## Sharing one Qwen across many sessions

When `lifeform-service` is deployed on a single GPU, every session must share **one** in-memory open-weight runtime. Pass the pre-built runtime through:

```python
from volvence_zero.substrate import build_transformers_runtime_with_fallback
from lifeform_domain_emogpt import build_companion_lifeform

shared = build_transformers_runtime_with_fallback(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    allow_live_substrate_mutation=False,  # required when sharing
)
life_a = build_companion_lifeform(substrate_runtime=shared)
life_b = build_companion_lifeform(substrate_runtime=shared)
```

`build_companion_lifeform` forces the underlying `BrainConfig` into `substrate_mode="injected"` whenever `substrate_runtime` is supplied so the brain consumes the shared instance instead of building a fresh one per session. The runtime must be frozen (`allow_live_substrate_mutation=False`, the default) — sharing a mutation-capable runtime would let one session's adapter-delta updates leak into every other session. `lifeform-service.create_app` enforces this fail-loud at construction time.
