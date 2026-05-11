# site/ — Companion Bench public site

> Status: GitHub Pages target for <https://companion-bench.org/>
> Files: pure static HTML / CSS / vanilla JS — no build step, no framework
> Auto-deploys on push to `main` via [`.github/workflows/companion-bench-publish.yml`](../.github/workflows/companion-bench-publish.yml)

## Page map

```
site/
├── index.html              Landing + top-N leaderboard preview
├── leaderboard.html        Full leaderboard with filter/sort/search
├── methodology.html        Six axes + scoring formula + EQ-Bench crosswalk
├── scenarios.html          24 public scenario browser (FSM viewer)
├── submit.html             Submission protocol + manifest schema
├── governance.html         Charter + working group + rotation policy
├── judges.html             Quarterly rotation + agreement + calibration
├── compare.html            Side-by-side pairwise comparison viewer
├── about.html              Background, FAQ, citation, contact
├── results/
│   └── index.html          Single-template per-submission detail page
│                           (URL: /results/?s=<submission_id>)
├── assets/
│   ├── style.css           Cozy light + dark theme; toggle persists in localStorage
│   ├── partials.js         Shared header/footer injector + theme toggle
│   ├── leaderboard.js      Reads data/aggregate_results.json
│   ├── detail-page.js      Reads data/submissions/<id>.json
│   ├── compare.js          Reads data/aggregate_results.json + data/pairwise.json
│   ├── scenarios.js        Reads data/scenarios.json
│   ├── judges.js           Reads data/judge_calibration.json
│   └── charts.js           Inline-SVG bar / forest / heatmap / scatter helpers
├── data/
│   ├── aggregate_results.json     One row per submission for leaderboard table
│   ├── submissions/<id>.json      Per-submission detail data
│   ├── scenarios.json             Compiled from companion_bench scenarios
│   ├── pairwise.json              Per-arc winner + axis margin between systems
│   └── judge_calibration.json     Quarterly rotation + agreement rates
├── CNAME, robots.txt, sitemap.xml
```

## Local preview

```bash
python -m http.server --directory site 8089
# open http://localhost:8089
```

## Updating data

### Real reference-system run (release tier)

Requires real API keys for the user simulator + per-turn judge + arc judge,
and budgets per RFC §6.7 ($40–115 per submission, single-seed).

```bash
python scripts/companion_bench/score_reference_systems.py \
  --output-dir artifacts/companion-bench/reference \
  --user-sim-model anthropic/claude-3.7-sonnet \
  --user-sim-key-env ANTHROPIC_API_KEY \
  --perturn-model anthropic/claude-3.7-sonnet \
  --perturn-key-env ANTHROPIC_API_KEY \
  --arc-model openai/gpt-5 \
  --arc-key-env OPENAI_API_KEY

python scripts/companion_bench/build_site.py \
  --artifact-dir artifacts/companion-bench/reference \
  --site-dir site
```

The build_site step writes:
- `site/data/aggregate_results.json` (one row per submission for the leaderboard)
- `site/data/submissions/<submission_id>.json` (per-submission detail page)
- `site/data/pairwise.json` (TrueSkill + BT + per-arc winners)
- `site/data/scenarios.json` (compiled from `companion_bench` scenarios)

When the artifacts come from real scoring, `aggregate_results.json` has
`"demo": false` and the public site drops the "Demo data" banner.

### Demo-only fully-populated site (no API spend)

When real API keys / budget are not yet available, `populate_demo_site.py`
runs the deterministic-fake pipeline for 8 mock submissions across all 24
public scenarios so every page (leaderboard, per-submission detail,
pairwise compare, scenarios browser) has fully-populated content. The
resulting JSON files are marked `"demo": true` so the site shows the
banner.

```bash
python scripts/companion_bench/populate_demo_site.py
```

### Scenarios-only regeneration

When you only need to refresh `site/data/scenarios.json` (e.g., after
adding a new public scenario):

```bash
python scripts/companion_bench/build_site.py --scenarios-only --site-dir site
```

The site renders a "Demo data" banner whenever the leaderboard JSON has
`"demo": true`, so viewers know the numbers are placeholders.

## Theme

Cozy light by default; click the moon icon in the header to toggle dark.
Choice persists in `localStorage` under `cb-theme`. Fall-back honours
`prefers-color-scheme: dark`.

## Accessibility

* Landmark roles via semantic HTML (`<header>`, `<nav>`, `<main>`, `<footer>`).
* Tabular data has `<th scope>` and numeric columns are right-aligned with `tabular-nums`.
* Focus styles preserved (no `outline: 0`).
* SVG charts include `role="img"` + `aria-label`; cells include `<title>` for hover tips.

## Style guide

* Theme variables only — never hard-code colours in components.
* No external chart libraries; all charts are inline SVG via `assets/charts.js`.
* Single template for results pages — all 100 submissions share one `results/index.html`
  and load their data from `data/submissions/<id>.json`.
