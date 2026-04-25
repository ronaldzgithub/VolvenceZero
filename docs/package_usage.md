# Volvence Zero Package Usage

> Scope: local package use for other projects on the same machine.
> Status: draft
> Last updated: 2026-04-25

This guide explains how to use the Volvence Zero brain kernel as a local Python package. It does not publish the package, upload code, download model weights, or start a public service.

## Install Locally

From this repository:

```bash
cd /Users/mengfu/Documents/GitHub/VolvenceZero
python -m pip install -e . --no-deps
```

From another project on the same machine:

```bash
python -m pip install -e /Users/mengfu/Documents/GitHub/VolvenceZero --no-deps
```

You can also add this to another project's `requirements.txt`:

```text
-e /Users/mengfu/Documents/GitHub/VolvenceZero
```

## Stable API

Use `volvence_zero.brain` as the package-facing API:

```python
from volvence_zero.brain import Brain, BrainConfig

brain = Brain(BrainConfig())
session = brain.create_session(session_id="local-session")
result = session.run_turn("I feel stuck and need help deciding.")

print(result.response.text)
```

The default config uses `substrate_mode="synthetic"`, so it does not require Qwen, Hugging Face model weights, `torch`, or `transformers`.

## Async Use

If your app already runs inside an event loop, use `run_turn_async()`:

```python
from volvence_zero.brain import Brain, BrainConfig

brain = Brain(BrainConfig())
session = brain.create_session(session_id="async-session")

result = await session.run_turn_async("Help me slow down and think this through.")
```

## Use Hugging Face / Qwen Explicitly

Install the optional runtime dependencies:

```bash
python -m pip install -e "/Users/mengfu/Documents/GitHub/VolvenceZero[hf]"
```

Then opt in explicitly:

```python
from volvence_zero.brain import Brain, BrainConfig

brain = Brain(
    BrainConfig(
        substrate_mode="hf",
        substrate_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        substrate_local_files_only=True,
    )
)
```

Model weights are not part of the package. They must exist in the runtime environment or Hugging Face cache. Keep `substrate_local_files_only=True` when you want to prevent network model fetches.

## Inject Your Own Runtime

Products and services can provide their own substrate runtime:

```python
from volvence_zero.brain import Brain, BrainConfig

brain = Brain(
    BrainConfig(substrate_mode="injected"),
    substrate_runtime=my_runtime,
)
```

This is the preferred path for a service that uses a remote model endpoint or a separately managed inference process.

## Load Domain Experience Packages

Vertical experience should be passed as `DomainExperiencePackage` data, not hardcoded into the kernel:

```python
from volvence_zero.brain import Brain, BrainConfig

brain = Brain(
    BrainConfig(
        domain_experience_packages=(my_domain_package,),
    )
)
session = brain.create_session(session_id="domain-session")
```

Packages are compiled into existing application owners: domain knowledge, case memory, strategy playbook, boundary hints, and rare-heavy application state.

## Inject Semantic State Proposals

The core package includes the semantic-state owner contracts and the prompt/schema resources needed by an external structured runtime:

```python
from volvence_zero.brain import Brain, BrainConfig
from volvence_zero.semantic_state import (
    SemanticProposalRuntime,
    load_semantic_json_schema,
    load_semantic_prompt_template,
)

prompt = load_semantic_prompt_template()
schema = load_semantic_json_schema()

brain = Brain(
    BrainConfig(),
    semantic_proposal_runtime=my_runtime,  # implements SemanticProposalRuntime
)
```

The default synthetic path uses `NoOpSemanticProposalRuntime`, so local installs still run without a model-backed semantic extractor. Product runtimes can inject a structured proposal runtime without becoming a second owner of the semantic state; each owner still publishes its own immutable snapshot.

## Submit External Semantic Events

External information does not need to arrive through dialogue. Submit structured events before the next turn; the session will drain them into semantic owner proposals exactly once:

```python
from volvence_zero.brain import Brain, BrainConfig

session = Brain(BrainConfig()).create_session(session_id="product-session")

session.submit_tool_result(
    event_id="tool:deploy:1",
    tool_name="deploy",
    action_id="deploy:1",
    status="succeeded",
    summary="Deploy finished",
    detail="The deploy tool produced artifact deploy-log-1.",
    artifact_refs=("deploy-log-1",),
    plan_ref="launch-plan",
)

result = session.run_turn("Continue from the deployment result.")
```

The package also exposes `submit_profile_event()`, `submit_task_event()`, and `submit_reviewed_knowledge_event()`. These helpers enqueue typed external events only; semantic owners still merge and publish their own immutable snapshots.

## Persistence

Persistence paths are outside the package:

```python
from volvence_zero.brain import Brain, BrainConfig

brain = Brain(
    BrainConfig(
        application_persistence_dir="/path/to/product/storage",
    )
)
```

Do not store product user data, secrets, or deployment state inside the package directory.

## Public vs Internal Surfaces

Stable package-facing API:

- `volvence_zero.brain.Brain`
- `volvence_zero.brain.BrainConfig`
- `volvence_zero.brain.BrainSession`
- `volvence_zero.application.domain_experience` package schema and compiler
- `volvence_zero.semantic_state` proposal runtime contracts and bundled schema/prompt resources

Internal or research-heavy surfaces:

- `volvence_zero.agent.dialogue_benchmark`
- `volvence_zero.agent.eta_proof_benchmark`
- broad re-exports from `volvence_zero.agent`
- internal integration helpers with leading underscores

These may remain useful for development and evaluation, but product projects should prefer `volvence_zero.brain`.

## What This Does Not Do

Using the package locally does not:

- upload to PyPI
- push to GitHub
- start a public HTTP service
- package Qwen or other model weights
- include product user data
- download model weights unless you explicitly configure a runtime path that does so

## Quick Smoke Test

```bash
python - <<'PY'
from volvence_zero.brain import Brain, BrainConfig

session = Brain(BrainConfig()).create_session(session_id="smoke")
result = session.run_turn("I need help making a careful decision.")
print(result.response.text)
PY
```
