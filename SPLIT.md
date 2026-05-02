# Repository Split Charter

> Status: active charter
> Last updated: 2026-04-29

## Recent changelog

- 2026-04-29: **Trigger \u2461 met + same training pipeline confirmed across verticals.**
  A second product consumer (`lifeform-domain-coding`, pair-programmer
  engineering partner) lives alongside `lifeform-domain-emogpt` in the
  workspace, and now also ships its own pre-trained
  `coding-temporal.snap` + `coding-regime.bs` artefacts produced by
  `lifeform-super-loop --vertical coding`. The same evolution loops
  (super-loop, multi-round-loop, learning-loop, regime-calibrator) drive
  both verticals via a `domain_experience_packages` parameter; the CLIs
  expose `--vertical {companion,coding}` resolving to the right
  scenarios + DomainExperiencePackage. Two verticals coexist in one
  Python process with disjoint drive sets, distinct DomainExperiencePackages,
  independent scenario packs, AND parallel bootstrap pipelines. The
  "kernel is vertical-agnostic" claim is no longer aspirational. See
  `tests/lifeform_e2e/test_coding_vertical.py` for the proof artefacts.
- 2026-04-28: M0 wheel-split debt resolved. ``vz-application`` is now its own
  wheel under ``packages/vz-application``. The cycle (``vz-cognition.evaluation``
  importing application owner snapshot types) was broken by extracting the
  type definitions into ``volvence_zero.application_types`` (still in
  vz-cognition). Owner code lives in ``vz-application`` and depends on
  vz-cognition for those type definitions plus DualTrackSnapshot etc.

This monorepo houses two architecturally distinct concerns that share a single
git history **on purpose**, with a written exit plan:

| Concern | Wheel prefix | Owns |
|---|---|---|
| **Brain kernel** | `vz-*` | NL+ETA contracts, owners, learning loops; zero product knowledge |
| **Digital lifeform** | `lifeform-*` | Tick engine, expression layer, vertical packs, services, evolution |

Repository boundary `≠` module boundary. The wheels are already split; the
git boundary is on standby.

## Phase 1 — Now (single monorepo, multi-wheel)

Active for the entire M0 → M5 window of `docs/next_gen_emogpt.md`. Rationale
is in `archetecture.md` §Repository decision.

**Discipline that keeps Phase 1 split-ready:**

1. Each wheel under `packages/` has its own `pyproject.toml`, version, and
   stable wheel name.
2. `tests/contracts/test_import_boundaries.py` is non-negotiable CI: kernel
   wheels (`vz-*`) cannot import from lifeform wheels (`lifeform-*`).
3. Cross-wheel imports must be declared in BOTH the consumer's
   `pyproject.toml` AND `tests/contracts/test_import_boundaries.py:ALLOWED_VZ_UPSTREAM`.
4. No `shared/` directory. Common primitives live in `vz-contracts`.
5. The top-level `pyproject.toml` is a workspace meta only — no source code
   lives at repo root.

## Phase 2 — Trigger conditions

Split into two repositories when **at least two** of the following are true:

| ID | Trigger | Status | Verifiable how |
|---|---|---|---|
| ① | Contract surface stable | pending | `docs/DATA_CONTRACT.md` has no breaking change for ≥4 consecutive weeks |
| ② | Second product consumer | **MET (2026-04-29)** | `lifeform-domain-coding` joined `lifeform-domain-emogpt`; both auto-discovered by the service registry; both pass the boundary linter |
| ③ | Governance pressure | pending | Independent versioning / release cadence / contributor policy is required for the kernel |
| ④ | External distribution | pending | Kernel needs to ship to a private artifactory or PyPI as a stand-alone product |
| ⑤ | Compliance | pending | Lifeform repo must enforce stricter access control around tenant data |
| ⑥ | Scale | pending | Repo > ~200k LoC or PR throughput is bottlenecked on CI |

Phase 2 fires when **two** triggers are met. With ② met, the next trigger to
watch is ① (contract stability). M3 evidence runs and metric reports under
`docs/specs/evaluation.md` are the source of truth for whether ① has
stabilised.

Predicted earliest fire date: **after M4** (`vz-temporal` lands and Internal
RL evidence stabilises). That is roughly the 6–9 month window from 2026-04.

## Phase 2 — Mechanical split procedure

Performed in the lifeform-target repo.

```bash
# 1) Carve lifeform out, preserving history.
git clone <volvence-zero-repo> lifeform-emogpt
cd lifeform-emogpt
git filter-repo \
  --path packages/lifeform-core \
  --path packages/lifeform-expression \
  --path packages/lifeform-domain-emogpt \
  --path packages/lifeform-domain-coding \
  --path packages/lifeform-service \
  --path packages/lifeform-evolution \
  --path docs/specs/lifeform-vitals.md \
  --path docs/specs/domain-experience-layer.md \
  --path docs/specs/affordance.md \
  --path docs/scenarios \
  --path tests/lifeform_e2e

# 2) Drop lifeform from the kernel repo.
cd <volvence-zero-repo>
git rm -r packages/lifeform-* docs/specs/lifeform-vitals.md docs/specs/domain-experience-layer.md docs/specs/affordance.md docs/scenarios tests/lifeform_e2e
git commit -m "chore: split lifeform layer into its own repository"

# 3) Pin wheel versions in lifeform-emogpt.
#    packages/lifeform-core/pyproject.toml:
#        dependencies = [
#            "vz-runtime==0.5.*",
#            "vz-contracts==0.5.*",
#        ]

# 4) Establish dev workflow.
#    Either uv workspace + private index, or:
pip install -e ../VolvenceZero/packages/vz-runtime
pip install -e ../VolvenceZero/packages/vz-contracts
```

After the split, contract changes flow as:

```
volvence-zero kernel repo
  ↓ release (semver)
private PyPI / artifactory
  ↓ pin bump
lifeform-emogpt repo
```

## Cost of getting this wrong

- Splitting too early (before contracts settle): doubles every cross-cutting
  refactor PR; introduces version-coordination noise; slows M0–M4 by months.
- Never splitting (monorepo forever): the kernel never gets its own version
  number; second-product reuse becomes implausible because every change is
  product-coupled by gravity; contributor policy stays single-track.

The Phase 1 → Phase 2 transition exists so neither cost is paid.

## What this charter does NOT decide

- Which CI provider, secrets manager, or artifact registry to use.
- Public open-source licensing.
- Tenant-data residency.

These are deployment concerns, decided when Phase 2 fires.
