# LSCB Leaderboard — Static Site

> Status: v1.0 GitHub Pages target
> Files: `index.html`, `style.css`, `leaderboard.js`, `data/aggregate_results.json`

This directory is the GitHub Pages root for the LSCB public
leaderboard. It is a pure static site (no build step, no framework)
so any contributor can audit the rendering code in a single read.

## Layout

```
site/leaderboard/
├── index.html           # Page structure + section anchors
├── style.css            # Theme (dark, minimal)
├── leaderboard.js       # Reads data/aggregate_results.json, renders table
└── data/
    └── aggregate_results.json   # Output of score_reference_systems.py
```

## Deploy

The intent is to publish from `site/leaderboard/` to GitHub Pages
under a custom subdomain `lscb-bench.volvencezero.org` (or fall back
to `*.github.io`). The publishing workflow lives at
[`.github/workflows/lscb-leaderboard-publish.yml`](../../.github/workflows/lscb-leaderboard-publish.yml).

## Updating data

Real reference-system runs:

```bash
python scripts/lscb/score_reference_systems.py \
  --output-dir artifacts/lscb/reference \
  --user-sim-model anthropic/claude-3.7-sonnet \
  --user-sim-key-env ANTHROPIC_API_KEY \
  --perturn-model anthropic/claude-3.7-sonnet \
  --perturn-key-env ANTHROPIC_API_KEY \
  --arc-model openai/gpt-5 \
  --arc-key-env OPENAI_API_KEY
cp artifacts/lscb/reference/aggregate_results.json site/leaderboard/data/aggregate_results.json
```

Demo-only (no API spend) regeneration of the placeholder data file:

```bash
python scripts/lscb/generate_demo_aggregate.py \
  --output site/leaderboard/data/aggregate_results.json
```

The site renders a `Demo data` banner whenever the JSON has
`"demo": true`, so viewers know the numbers are placeholders.

## Manual preview

```bash
python -m http.server --directory site/leaderboard 8089
# open http://localhost:8089
```
