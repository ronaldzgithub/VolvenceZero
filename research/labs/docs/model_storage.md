# Model Storage Strategy

## Principle

Model weights are **large and mutable** (HF revisions change). They do NOT belong in CAS.

CAS stores only a **reference** (model_id + revision → sha). The actual weights live in `.labs/models/` or `.labs/hf_cache/` (both gitignored).




## Layout

```
.labs/
├── hf_cache/          # huggingface_hub cache (snapshot_download target)
│   └── models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/
│       └── snapshots/<commit_hash>/...
└── models/            # (reserved for future non-HF weights)
```

## API

```python
from volvence_labs.framework.snapshot.weights import (
    model_sha_for_revision,
    ensure_model_downloaded,
    register_model_in_cas,
)

# 1. Download if needed
local_path = ensure_model_downloaded("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# 2. Register reference in CAS
m_sha = register_model_in_cas(store, "TinyLlama/TinyLlama-1.1B-Chat-v1.0", local_path=local_path)

# 3. Use m_sha in probe tags for traceability
tags = {"model_sha": m_sha, ...}
```

## model_sha computation

`model_sha = sha256(canonical_dumps({"model_id": ..., "revision": ...}))`

This is NOT the sha of the weight bytes (too expensive for multi-GB files). It's a stable key for the (repo, commit) pair.

## F5 v1 implications

When F5 v1 checks rollback correctness on real models:
- `model_sha` must be bit-exact across reruns (same model_id + revision → same sha).
- Numeric readouts (float metrics) use ε-tolerance (`abs(a-b) < 1e-5`).
- Token-level outputs (ids) remain bit-exact.
