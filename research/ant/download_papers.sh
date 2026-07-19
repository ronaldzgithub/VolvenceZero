#!/usr/bin/env bash
# Download open-access papers for research/ant/. Best-effort: paywalled items are
# recorded in _download_summary.md with link-only fallback, not force-downloaded.
set -uo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPERS="$BASE/papers"

UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

declare -a ENTRIES=(
  # category|filename|url
  "connectome|reference-brain-clonal-raider-ant-2025.pdf|https://www.biorxiv.org/content/10.1101/2025.10.13.679875v1.full.pdf"
  "mushroom-body|ardin-mb-route-memory-2016.pdf|https://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1004683&type=printable"
  "mushroom-body|snn-mb-visual-navigation-2024.pdf|https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2024.1379977/pdf"
  "mushroom-body|latent-learning-no-map-ants-2024.pdf|https://www.biorxiv.org/content/10.1101/2024.08.29.610243v1.full.pdf"
  "central-complex|head-direction-circuit-two-insect-species-2020.pdf|https://pmc.ncbi.nlm.nih.gov/articles/PMC7419142/pdf/nihms-1616423.pdf"
  "central-complex|emergent-spatial-goals-cx-2024.pdf|https://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1011480&type=printable"
  "central-complex|multimodal-navigation-cx-coordination.pdf|https://eprints.whiterose.ac.uk/id/eprint/181601/7/elife-73077-v2.pdf"
  "central-complex|locust-heading-angular-velocity-2024.pdf|https://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1012155&type=printable"
  "central-complex|theoretical-principles-head-direction-2024.pdf|https://elifesciences.org/articles/91533.pdf"
  "collective-intelligence|bayesian-superorganism-swarm-2019.pdf|https://www.biorxiv.org/content/10.1101/468942v1.full.pdf"
  "collective-intelligence|bayesian-superorganism-externalized-memories-2020.pdf|https://pmc.ncbi.nlm.nih.gov/articles/PMC7328406/pdf/rsif20190848.pdf"
  "collective-intelligence|active-inferants-ant-colony-2021.pdf|https://www.frontiersin.org/journals/behavioral-neuroscience/articles/10.3389/fnbeh.2021.647732/pdf"
  "caste-plasticity|mosaic-brain-differentiation-atta-2023.pdf|https://pmc.ncbi.nlm.nih.gov/articles/PMC10039480/pdf/main.pdf"
  "caste-plasticity|neuropeptides-division-of-labor-atta-2024.pdf|https://www.biorxiv.org/content/10.1101/2024.11.07.622473v1.full.pdf"
  "caste-plasticity|epigenetic-reprogramming-camponotus-2016.pdf|https://pmc.ncbi.nlm.nih.gov/articles/PMC5057185/pdf/pnas.201610800.pdf"
  "caste-plasticity|tramtrack-ant-caste-identity-2021.pdf|https://journals.plos.org/plosgenetics/article/file?id=10.1371/journal.pgen.1009801&type=printable"
  "neuromodulation|dopamine-octopamine-opponency-olfaction-2025.pdf|https://www.biorxiv.org/content/10.64898/2025.12.23.696261v1.full.pdf"
  "connectome-critique|connectome-weight-optimization-celegans-2026.pdf|https://www.nature.com/articles/s41598-026-54384-5.pdf"
  "miniaturization|constant-neuropilar-ratio-insect-brain-2020.pdf|https://www.nature.com/articles/s41598-020-78599-2.pdf"
  "miniaturization|anucleate-neurons-microwasps-2023.pdf|https://www.nature.com/articles/s41598-023-31529-4.pdf"
  "robotics|antbot-celestial-compass-2018.pdf|https://amu.hal.science/hal-02075674/file/Dupeyroux_et_al_LM2018.pdf"
  "connectome|ant-alarm-glomerulus-sparse-coding-2023.pdf|https://pmc.ncbi.nlm.nih.gov/articles/PMC10334690/pdf/main.pdf"
  "connectome|pheromone-antennal-lobe-age-2024.pdf|https://pmc.ncbi.nlm.nih.gov/articles/PMC10888935/pdf/main.pdf"
)

mkdir -p "$PAPERS"
SUCCESS=()
FAILED=()

for entry in "${ENTRIES[@]}"; do
  IFS='|' read -r category filename url <<< "$entry"
  outdir="$PAPERS/$category"
  mkdir -p "$outdir"
  out="$outdir/$filename"
  echo "==> $category/$filename"
  if curl -sL -A "$UA" --max-time 60 -o "$out" "$url"; then
    ftype=$(file -b "$out" 2>/dev/null || echo "unknown")
    if echo "$ftype" | grep -qi "PDF"; then
      size=$(du -h "$out" | cut -f1)
      echo "    OK ($size, $ftype)"
      SUCCESS+=("$category/$filename")
    else
      echo "    NOT-PDF ($ftype) -- removing"
      rm -f "$out"
      FAILED+=("$category/$filename :: $url :: got non-PDF ($ftype)")
    fi
  else
    echo "    CURL-FAILED"
    FAILED+=("$category/$filename :: $url :: curl error")
  fi
done

{
  echo "# Ant Neuroscience Papers — Download Summary"
  echo
  echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')"
  echo
  echo "## Success (${#SUCCESS[@]})"
  echo
  for s in "${SUCCESS[@]:-}"; do
    [ -n "$s" ] && echo "- $s"
  done
  echo
  echo "## Failed / paywalled (${#FAILED[@]})"
  echo
  for f in "${FAILED[@]:-}"; do
    [ -n "$f" ] && echo "- $f"
  done
} > "$BASE/_download_summary.md"

echo
echo "Summary written to $BASE/_download_summary.md"
