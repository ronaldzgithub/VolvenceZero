# Bootstrapping the Companion Bench held-out repo

> Status: one-time organiser action
> Audience: Companion Bench working-group chair (or the org running the leaderboard)

The held-out scenario set lives in a **separate private GitHub
repository** at `VolvenceZero/companion-bench-heldout` (per RFC §3 P3 and §8.6).
The public monorepo references it as a git submodule at
`external/companion-bench-heldout/`, which is gitignored so the YAML body never
enters the public history.

## One-time bootstrap (organiser action)

The 96 seed held-out scenarios are deterministic variants of the 24
public scenarios. To produce them run:

```bash
python scripts/companion_bench/generate_heldout_seeds.py \
  --output external/companion-bench-heldout/scenarios
```

This emits 96 YAML files (4 variants per public scenario × 6 families
× 4 public per family). Each variant changes persona / occupation /
contextual detail surface form while preserving the family-level
probe — so a system overfit on the public set shows a measurable gap.

Then move that tree into the private repo:

```bash
cd external/lscb-heldout
git init -b main
git add .
git commit -m "lscb-heldout v1.0 seed (96 scenarios)"
git remote add origin git@github.com:VolvenceZero/companion-bench-heldout.git
git push -u origin main
```

## Wiring in CI

The `.gitmodules` entry in the monorepo points at the private repo
already. CI nightly + release tiers must check it out via deploy key:

```yaml
# Excerpt from .github/workflows/lscb-paper-suite-full.yml
- uses: actions/checkout@v4
  with:
    submodules: true
    ssh-key: ${{ secrets.COMPANION_BENCH_HELDOUT_DEPLOY_KEY }}
```

The deploy key is a read-only SSH key registered on the private
`lscb-heldout` repo. The corresponding private key lives in the
public monorepo's repository secrets as `COMPANION_BENCH_HELDOUT_DEPLOY_KEY`.

## Public PRs

Public PRs run against `lscb-ci-smoke` only, which does NOT check
out the held-out submodule. The harness emits a single warning and
proceeds with public scenarios. This keeps the held-out body out of
PR diffs while letting external contributors iterate freely.

## Quarterly rotation

Per RFC §8.2 the held-out paraphrase seeds rotate quarterly. The
rotation is performed by re-running `generate_heldout_seeds.py` with
a new `--variant-salt` (parameter to be added in v1.1) and pushing
the new tree to the private repo on a quarterly cadence. The hash
manifest lives at `external/companion-bench-heldout/HASHES.txt` (private) and
its rotation log on the public side at
`docs/external/companion-bench-heldout-rotation-log.md` (just the dates and
that a rotation occurred — never the hashes themselves).

## What never enters the public history

- `external/companion-bench-heldout/scenarios/*.yaml` (the 96 YAML files)
- `external/companion-bench-heldout/HASHES.txt`
- The rotated paraphrase seeds

## What is publicly visible

- `.gitmodules` reference (just the URL)
- The number of held-out scenarios (96)
- The variant scheme (4 surface variants per public)
- Family-level distribution (16 per family)
- The fact that a given submission was scored on the private set
  (logged via per-scenario hash, never YAML body)
