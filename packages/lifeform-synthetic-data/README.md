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

### Cursor-authored, zero API

The checked-in render assets are written directly by Cursor agents. Each turn
has four semantically equivalent, substantially different variants. For every
user turn, variant 1 is first compiled into generator truth as the canonical
observable event; only then do the seed, stable turn ID, and complete asset
hash choose an expression variant deterministically.

```bash
lifeform-synthetic-data validate-render-assets

lifeform-synthetic-data generate \
  --stage pilot96 \
  --renderer cursor \
  --run-id unified-v1-cursor-pilot96 \
  --max-cost-usd 0

lifeform-synthetic-data generate \
  --stage scale768 \
  --renderer cursor \
  --run-id unified-v1-cursor-scale768 \
  --max-cost-usd 0

lifeform-synthetic-data generate \
  --stage master10240 \
  --renderer cursor \
  --run-id unified-v1-cursor-master10240 \
  --max-cost-usd 0 \
  --concurrency 8
```

No API token or API cost is claimed for this path. Provenance records the
actual Cursor model family and a content hash of all authored dialogue assets.

Versioned post-v1 expansion bundles live under
`render_asset_bundles/<bundle_id>/` and never overwrite the frozen
`render_assets/` baseline. Validation also requires zero normalized variant
overlap with the baseline:

```bash
lifeform-synthetic-data validate-render-assets \
  --cursor-asset-bundle expansion_20260720

lifeform-synthetic-data generate \
  --stage scale768 \
  --renderer cursor \
  --cursor-asset-bundle expansion_20260720 \
  --run-id unified-v1-cursor-expansion-20260720-scale768 \
  --max-cost-usd 0 \
  --concurrency 8

lifeform-synthetic-data generate \
  --stage master50000 \
  --renderer cursor \
  --cursor-asset-bundle expansion_20260720 \
  --run-id unified-v1-cursor-expansion-20260720-master50000 \
  --max-cost-usd 0 \
  --concurrency 8
```

`master50000` expands the fixed 96-blueprint split without moving scenarios
between splits: 40,000 train, 5,008 validation, and 4,992 test trajectories.

### OpenAI-compatible endpoint

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
