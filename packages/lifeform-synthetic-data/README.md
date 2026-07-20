# lifeform-synthetic-data

Proprietary offline owner of the Unified Synthetic Experience Corpus v1.

The wheel freezes one master trajectory contract and keeps three evidence layers separate:

- deterministic generator/world truth;
- LLM-rendered text slots;
- public immutable runtime snapshot observations.

It does not add a runtime owner or slot. It does not load Companion Bench held-out scenarios. Model and judge outputs cannot become hard training labels.

## Validate contracts and scenarios

```bash
lifeform-synthetic-data conformance
lifeform-synthetic-data validate-scenarios
```

## Structural golden run

```bash
lifeform-synthetic-data generate \
  --stage golden96 \
  --run-id unified-v1-golden96 \
  --max-cost-usd 0
```

## Rendered run

Create an untracked endpoint config:

```json
{
  "rate_card": {
    "input_usd_per_million": 0.0,
    "output_usd_per_million": 0.0
  },
  "endpoints": [
    {
      "base_url": "https://provider.example/v1",
      "api_key_env": "SYNTHETIC_DATA_API_KEY",
      "model_id": "provider/model"
    }
  ]
}
```

Always estimate first, then run with an explicit hard budget:

```bash
lifeform-synthetic-data estimate \
  --stage master10240 \
  --run-id unified-v1-master10240 \
  --endpoint-config .local/renderer-endpoints.json \
  --max-cost-usd 100

lifeform-synthetic-data generate \
  --stage master10240 \
  --run-id unified-v1-master10240 \
  --endpoint-config .local/renderer-endpoints.json \
  --max-cost-usd 100 \
  --concurrency 8
```

Authentication and quota denials fail loudly. Retry exhaustion and malformed samples enter the append-only quarantine ledger. `--resume` reuses content-addressed objects and verifies their hashes.

## Project and audit

```bash
lifeform-synthetic-data project \
  --run-root data/synthetic/unified_v1/unified-v1-master10240

lifeform-synthetic-data audit \
  --run-root data/synthetic/unified_v1/unified-v1-master10240 \
  --expected-count 10240
```

The 1,024-trajectory live-through run is a deterministic scenario-stratified
sample of the rendered master. It never rebuilds placeholder text:

```bash
lifeform-synthetic-data generate \
  --stage live1024 \
  --run-id unified-v1-live1024 \
  --source-run-root data/synthetic/unified_v1/unified-v1-master10240 \
  --concurrency 8
```

Generated data stays under the gitignored `data/synthetic/unified_v1/` tree. Only code, schemas, prompts, the 96 blueprints, and small fixtures belong in git.
