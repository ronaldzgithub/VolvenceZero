# lifeform-domain-coding

Vertical for the **pair-programmer engineering partner** archetype.

A vertical is **data + light glue** that compiles into the kernel's owner snapshots:

| Asset | Compiles into kernel surface |
|---|---|
| `DomainExperiencePackage` (engineering heuristics, problem-pattern cases, playbook rules, boundary hints for security-critical paths) | `vz-application.domain_knowledge / case_memory / strategy_playbook / boundary_policy` |
| Drive set (built into the package) | `lifeform-core.vitals` slow-scale PE |
| Scenario packs (`scenarios/*.json`) | `lifeform-evolution` benchmark inputs |

The contrast with `lifeform-domain-emogpt` is the proof of trigger ② in [`SPLIT.md`](../../SPLIT.md): the kernel ships zero awareness of which vertical is loaded. Where the companion vertical's drives are about **bond and engagement**, this vertical's drives are about **intent clarity and direction**. Both compile through the same kernel surfaces.

## Public API

```python
from lifeform_domain_coding import (
    build_coding_package,               # DomainExperiencePackage data
    build_coding_vitals_bootstrap,      # VitalsBootstrap (3 drives)
    scenarios_dir,                      # path to scripted scenarios
    build_coding_lifeform,              # ready-to-run Lifeform with everything wired
    # Gap 1 slice 2b+2c: affordances
    CODING_AFFORDANCE_DESCRIPTORS,      # 5 tuple[AffordanceDescriptor, ...]
    CONSENT_FILESYSTEM_READ,            # read_file / list_dir / grep gate
    CONSENT_FILESYSTEM_WRITE,           # write_file gate
    CONSENT_RUN_SHELL_COMMANDS,         # run_test gate
    build_coding_affordance_registry,   # AffordanceRegistry factory (optionally sealed)
    build_coding_affordance_invoker,    # AffordanceInvoker wired to a sandbox
    build_coding_affordance_backends,   # name -> async backend dict (low-level)
    resolve_sandbox_path,               # path safety helper (exported for reuse)
    SandboxPathError,                   # raised on sandbox escape / missing / non-file
)
```

## Affordances

This vertical ships five affordances scoped to a sandbox root:

| Name         | Kind | Latency | Consent grant required | Extra safety | Blocked in regimes |
|--------------|------|---------|------------------------|--------------|--------------------|
| `read_file`  | tool | FAST    | `filesystem_read`      | —            | `casual_social` / `emotional_support` / `repair_and_deescalation` |
| `list_dir`   | tool | INSTANT | `filesystem_read`      | —            | same               |
| `grep`       | tool | FAST    | `filesystem_read`      | —            | same               |
| `write_file` | tool | FAST    | `filesystem_write`     | requires user confirmation + irreversible + audit | same |
| `run_test`   | tool | SLOW    | `run_shell_commands`   | rate-limited 6/min + audit                 | same |

Every backend is guarded by `resolve_sandbox_path` which rejects `..` escape, absolute paths outside the sandbox, and symlinks pointing outside the sandbox. `write_file` enforces `mode ∈ {create, overwrite, append}` with a 10 MB per-call hard cap and rejects auto-creating missing parent directories. `run_test` shells out via `python -m pytest -q` under `asyncio.create_subprocess_exec` with a configurable timeout (default 30 s, max 300 s); on timeout it kills the process and still returns structured `timed_out=True` output bounded to 64 KB per stream.

Typical host wiring:

```python
from lifeform_domain_coding import (
    build_coding_affordance_invoker,
    build_coding_lifeform,
    CONSENT_FILESYSTEM_READ,
)

lifeform = build_coding_lifeform()
session = lifeform.create_session(session_id="coding-1")
invoker = build_coding_affordance_invoker(sandbox_root="/path/to/workspace")

result = await invoker.invoke(
    "read_file",
    {"path": "src/main.py"},
    session=session.brain_session,
    event_id="turn-001-read-1",
    granted_consents=frozenset({CONSENT_FILESYSTEM_READ}),
    active_regime_id="problem_solving",
)
# result.payload -> {"content": "...", "truncated": False, ...}
# A successful call also publishes an execution_result record to
# the kernel's tool-result bus; denied / invalid calls are silent
# to the kernel (see docs/specs/affordance.md §6).
```

See [`docs/specs/affordance.md`](../../docs/specs/affordance.md) and Gap 1 in [`docs/implementation/13_emogpt_prd_alignment_upgrade.md`](../../docs/implementation/13_emogpt_prd_alignment_upgrade.md).

## Drive set

| Drive                  | Decay/tick | Per-turn | Regime bonuses                                                        |
|------------------------|-----------:|---------:|-----------------------------------------------------------------------|
| `solution_clarity`     |      0.004 |     0.05 | problem_solving +0.20 / guided_exploration +0.08                       |
| `code_freshness`       |      0.018 |     0.25 | (none — every user turn recharges)                                     |
| `direction_certainty`  |      0.010 |     0.05 | problem_solving +0.18 / guided_exploration **−0.05** (drains during exploration) |

`direction_certainty` deliberately uses a negative regime recharge: exploration is supposed to *reduce* certainty, and the drive layer now supports that explicitly (clamped to `[0, 1]`).

## Scenarios

Four scripted scenarios in `scenarios/`:

* `bug-no-repro` — vague bug report; expected behaviour is `clarify-first` and an ask for a minimal repro.
* `concrete-debug` — concrete failing test plus shared code; expected behaviour is `structure-first` problem solving.
* `vague-feature-request` — feature wish without user / workflow / outcome; expected behaviour is guided exploration that narrows scope.
* `security-refer-out` — security-critical guidance ask; expected behaviour is to surface refer-out / disclaimer rather than ship implementation details.

Use them via:

```bash
lifeform-bench --vertical coding \
  --scenarios packages/lifeform-domain-coding/src/lifeform_domain_coding/scenarios \
  --family-report
```

## Pre-trained calibration

This vertical ships pre-trained bootstraps under `bootstraps/`:

* `coding-temporal.snap` — `MetacontrollerParameterSnapshot` (β_t / z_t calibration)
* `coding-regime.bs` — `RegimeBootstrap` (regime selection_weights)

`build_coding_lifeform()` loads them automatically; pass `use_temporal_bootstrap=False` and/or `use_regime_bootstrap=False` for ablation runs. Same envelope shape as `lifeform-domain-emogpt` (magic-byte-prefixed pickle); the loaders fail loudly on schema-version drift, so a hand-edited or stale artifact never silently changes behaviour.

To regenerate the artifacts (e.g. after editing scenarios or expanding the pack):

```bash
lifeform-super-loop \
  --vertical coding \
  --rounds 3 \
  --save-temporal packages/lifeform-domain-coding/src/lifeform_domain_coding/bootstraps/coding-temporal.snap \
  --save-regime   packages/lifeform-domain-coding/src/lifeform_domain_coding/bootstraps/coding-regime.bs
```

`--vertical coding` resolves the scenario set AND the `DomainExperiencePackage` simultaneously, so you get a coding-specific training run rather than a companion run pointed at coding scenarios. The same training pipeline produces companion bootstraps via `--vertical companion`.

## Benchmark

```bash
lifeform-bench --vertical coding --family-report
```

Runs all four coding scenarios on the trained coding lifeform and prints the R12 six-family grouped report. Pass `--require-family-pass` to make this a CI gate.

## Calibration history

A previous (2026-04-29-am) version of the regime calibrator collapsed to "always pick `guided_exploration`" on this scenario pack — every coding scenario lists `guided_exploration` in `expected_regime_in`, so the unconstrained calibrator found the cheap monoculture solution at 100% regime_match. The shipped bootstraps come from the **diversity-aware** calibrator (2026-04-29-pm onward) which adds an anti-monoculture penalty: any regime predicted on more than `--diversity-threshold` of turns has its weight pulled back proportional to overuse. The bootstraps now ship with two regimes active across scenarios (`problem_solving=9`, `guided_exploration=4` over 13 turns) instead of one.

To rerun:

```bash
lifeform-super-loop --vertical coding --rounds 4 \
  --diversity-threshold 0.50 --diversity-lr 0.30 \
  --save-temporal packages/lifeform-domain-coding/src/lifeform_domain_coding/bootstraps/coding-temporal.snap \
  --save-regime   packages/lifeform-domain-coding/src/lifeform_domain_coding/bootstraps/coding-regime.bs
```

The CLI saves two artifacts from potentially **different** rounds: temporal from the sparsest-β_t round (`best_temporal_round()`), regime from the most-diverse-predicted-distribution round (`best_regime_round()`). This avoids the trap where a single round is healthy on one axis but not the other.

To recover the pre-penalty behaviour for ablation: pass `--diversity-lr 0.0`.
