# Integrating an External MCP Bundle

How to attach an external repo of tools / knowledge / eval scenarios
to a Volvence Zero lifeform via the MCP bundle bridge. See
[`docs/specs/mcp-bridge.md`](../specs/mcp-bridge.md) for the full
contract.

## TL;DR

The default bundle [`external/vz-bundle/`](../../external/vz-bundle/)
is already attached as a git submodule. After cloning the main repo:

```bash
git submodule update --init external/vz-bundle
pip install -e external/vz-bundle
```

To attach an additional bundle (or replace the default), fork the
[`external/vz-bundle`](../../external/vz-bundle/) repo into your own
git remote, then:

1. Implement your tools under `src/<bundle_name>/tools/` and add
   reviewed manifest entries to `.vzbridge.yaml`.
2. Add it as another submodule alongside `vz-bundle`.
3. From your VolvenceZero deployment:

```python
from lifeform_core.lifeform import Lifeform, LifeformConfig
from lifeform_mcp_bridge import MCPServerSpec
from volvence_zero.brain import BrainConfig

bundle_root = "external/my-bundle"   # path inside the deployment
spec = MCPServerSpec(
    name="my-bundle",
    transport="stdio",
    command=("python", "-m", "my_bundle.server"),
    safety_manifest_path=f"{bundle_root}/.vzbridge.yaml",
)
lifeform = Lifeform(
    LifeformConfig(
        brain_config=BrainConfig(),
        mcp_server_specs=(spec,),
    )
)

await lifeform.start()
session = lifeform.create_session()
await session.flush_mcp_resources()    # ingest the bundle's knowledge once
# ... use session.mcp_invoker.invoke(...) for tool calls ...
await lifeform.shutdown()              # at deployment teardown
```

## Wiring options

The bridge has three rollout states via
`LifeformConfig.mcp_bridge_wiring`:

| Wiring | Behaviour |
|---|---|
| `ACTIVE` (default) | Spawns servers, registers affordances, ingests resources / prompts. Production state. |
| `SHADOW` | Spawns servers (verifies they come up cleanly), but does NOT register affordances or ingest. Used to gate a rollout. |
| `DISABLED` | Skips bridge construction entirely. Rollback escape hatch. |

Multiple specs are fine; bridge prefixes affordance names with
`<spec.name>.<tool_name>` so they cannot clash.

## Safety manifest is mandatory

Every tool the MCP server exposes MUST have a reviewed entry in the
spec's `safety_manifest_path` YAML. Missing entries fail-loud at
`Lifeform.start()` with `MCPMissingSafetyManifestError`. There is
no "default to safe" — see R10 in
[`docs/next_gen_emogpt.md`](../next_gen_emogpt.md). The bundle
template ships a complete reference manifest you can copy into your
fork as a starting point.

## Submodule layout

The recommended deployment layout treats the external bundle as a
git submodule of the VolvenceZero repo:

```text
VolvenceZero/
├── external/
│   └── my-bundle/                    # git submodule
│       ├── .vzbridge.yaml
│       ├── src/my_bundle/server.py
│       ├── knowledge/
│       └── eval-scenarios/
└── packages/
    └── ...                           # main project, unchanged
```

```bash
git submodule add git@github.com:my-org/my-bundle.git external/my-bundle
git submodule update --init --remote external/my-bundle
pip install -e external/my-bundle
```

Submodule pins the bundle to a specific commit; you only update when
you intend to. The main project does NOT depend on the bundle in
`pyproject.toml`; bundle install is a deployment-time step.

## Lifecycle hygiene

- `Lifeform.start()` is idempotent. Call it once per `Lifeform`
  instance after construction.
- `Lifeform.shutdown()` closes every spawned server. Call it once
  at deployment teardown (CLI exit, service shutdown, fixture
  teardown).
- `LifeformSession.flush_mcp_resources()` ingests the bundle's
  resources via the canonical `IngestionPipeline`. Idempotent on
  the second call. Designed to be called once per session, before
  the first user turn.
- `LifeformSession.flush_mcp_prompts()` is similar but only fires
  when both `MCPServerSpec.enable_prompts=True` AND the manifest's
  `prompts.enabled=True`.

## Failure handling

- Server fails to spawn → `MCPServerSpawnError` at `Lifeform.start()`.
  Operator must remove or fix the spec; bridge does NOT silently
  down-grade.
- Server returns a tool whose name is not in the manifest →
  `MCPMissingSafetyManifestError` at `Lifeform.start()`. Add the
  manifest entry.
- Server crashes mid-call → `AffordanceInvocationStatus.BACKEND_FAILED`
  with `error_class` indicating the cause. Main process keeps
  running. Per `MCPServerSpec.restart_policy` the pool may attempt
  a restart on the next call.
- Manifest YAML is malformed → `MCPSafetyManifestSchemaError`. Error
  message includes the offending key path; fix the YAML.

## Eval scenarios

`MCPEvalScenario` records discovered from
`<bundle_root>/eval-scenarios/*.json` are returned by
`load_scenarios(repo_root=bundle_root)` for downstream benchmark
consumers (e.g. `lifeform-evolution`). The minimum schema is
documented in
[`external/vz-bundle/docs/adding-eval-scenario.md`](../../external/vz-bundle/docs/adding-eval-scenario.md).
The bridge does NOT auto-run scenarios; that decision is made by
the benchmark harness.

## Where to look in the code

| Concern | Path |
|---|---|
| Spec | [`docs/specs/mcp-bridge.md`](../specs/mcp-bridge.md) |
| Bridge wheel | [`packages/lifeform-mcp-bridge/`](../../packages/lifeform-mcp-bridge/) |
| Reference external repo (default submodule) | [`external/vz-bundle/`](../../external/vz-bundle/) |
| LifeformConfig wiring | [`packages/lifeform-core/src/lifeform_core/lifeform.py`](../../packages/lifeform-core/src/lifeform_core/lifeform.py) (`Lifeform.start` / `LifeformSession.flush_mcp_resources`) |
| Tests | `tests/contracts/test_mcp_*.py` + `tests/lifeform_e2e/test_mcp_*.py` + `tests/longitudinal/test_mcp_resource_becomes_durable_knowledge.py` |
