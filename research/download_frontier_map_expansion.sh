#!/usr/bin/env bash
# Download the broad 2024-2026 frontier-map expansion. Entries are strictly
# deduplicated against research/papers and the 28-paper July sweep.
set -uo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPERS="$BASE/papers/frontier-map"
SUMMARY="$BASE/frontier-map-2024-2026-download-summary.md"
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) research-survey/1.0"

declare -a ENTRIES=(
  # axis|filename|url
  "architecture|multi-timescale-abstractions-planning-2605.17058.pdf|https://arxiv.org/pdf/2605.17058"
  "architecture|claw-continuous-latent-action-world-model-2606.04130.pdf|https://arxiv.org/pdf/2606.04130"
  "architecture|mind-the-gap-hierarchical-planning-2607.12547.pdf|https://arxiv.org/pdf/2607.12547"
  "architecture|latent-spatial-memory-world-models-2606.09828.pdf|https://arxiv.org/pdf/2606.09828"
  "architecture|active-epistemic-control-2602.03974.pdf|https://arxiv.org/pdf/2602.03974"
  "architecture|renew-world-model-repair-2607.14180.pdf|https://arxiv.org/pdf/2607.14180"
  "architecture|budget-curated-agent-memory-2606.25115.pdf|https://arxiv.org/pdf/2606.25115"
  "architecture|test-time-learning-evolving-library-2605.14477.pdf|https://arxiv.org/pdf/2605.14477"
  "architecture|deltamem-residual-trees-2606.03083.pdf|https://arxiv.org/pdf/2606.03083"
  "architecture|adamem-adaptive-memory-agents-2606.05684.pdf|https://arxiv.org/pdf/2606.05684"
  "architecture|mindjourney-world-model-spatial-reasoning-2507.12508.pdf|https://arxiv.org/pdf/2507.12508"
  "architecture|just-in-time-rl-agent-memory-2601.18510.pdf|https://arxiv.org/pdf/2601.18510"
  "architecture|evotest-evolutionary-test-time-learning-2510.13220.pdf|https://arxiv.org/pdf/2510.13220"
  "architecture|hierarchy-groups-policy-optimization-2602.22817.pdf|https://arxiv.org/pdf/2602.22817"
  "architecture|agent-authored-world-modeling-2606.25421.pdf|https://arxiv.org/pdf/2606.25421"

  "safety|auditing-hidden-objectives-2503.10965.pdf|https://arxiv.org/pdf/2503.10965"
  "safety|auditbench-2602.22755.pdf|https://arxiv.org/pdf/2602.22755"
  "safety|strategic-deception-alignment-audits-2602.08877.pdf|https://arxiv.org/pdf/2602.08877"
  "safety|shade-arena-sabotage-monitoring-2506.15740.pdf|https://arxiv.org/pdf/2506.15740"
  "safety|monitor-information-access-sabotage-2601.21112.pdf|https://arxiv.org/pdf/2601.21112"
  "safety|sabotage-evaluations-frontier-models-2410.21514.pdf|https://arxiv.org/pdf/2410.21514"
  "safety|sabotage-ai-safety-research-2604.24618.pdf|https://arxiv.org/pdf/2604.24618"
  "safety|models-know-being-evaluated-2505.23836.pdf|https://arxiv.org/pdf/2505.23836"
  "safety|lure-live-usage-replay-evals-2605.26438.pdf|https://arxiv.org/pdf/2605.26438"
  "safety|agentspec-runtime-enforcement-2503.18666.pdf|https://arxiv.org/pdf/2503.18666"
  "safety|muse-machine-unlearning-evaluation-2407.06460.pdf|https://arxiv.org/pdf/2407.06460"
  "safety|openunlearning-2506.12618.pdf|https://arxiv.org/pdf/2506.12618"
  "safety|agentdojo-prompt-injection-2406.13352.pdf|https://arxiv.org/pdf/2406.13352"
  "safety|agent-security-bench-2410.02644.pdf|https://arxiv.org/pdf/2410.02644"
  "safety|basharena-control-setting-2512.15688.pdf|https://arxiv.org/pdf/2512.15688"
  "safety|camel-prompt-injection-by-design-2503.18813.pdf|https://arxiv.org/pdf/2503.18813"
  "safety|position-aware-circuit-discovery-2502.04577.pdf|https://arxiv.org/pdf/2502.04577"

  "social|collabllm-active-collaborators-2502.00640.pdf|https://arxiv.org/pdf/2502.00640"
  "social|hypothetical-minds-multi-agent-tom-2407.07086.pdf|https://arxiv.org/pdf/2407.07086"
  "social|tom-benchmarks-are-broken-2412.19726.pdf|https://arxiv.org/pdf/2412.19726"
  "social|dynamic-theory-of-mind-2505.17663.pdf|https://arxiv.org/pdf/2505.17663"
  "social|lifelong-sotopia-2506.12666.pdf|https://arxiv.org/pdf/2506.12666"
  "social|extended-chatbot-use-rct-2503.17473.pdf|https://arxiv.org/pdf/2503.17473"
  "social|trust-development-repair-ai-facct2024.pdf|https://www.jorgegoncalves.com/docs/facct24.pdf"
  "social|interpretability-feedback-trust-chi2024.pdf|https://arxiv.org/pdf/2111.08222"
  "social|negotiation-tom-2404.13627.pdf|https://arxiv.org/pdf/2404.13627"
  "social|agentsense-social-intelligence-naacl2025.pdf|https://aclanthology.org/2025.naacl-long.257.pdf"
  "social|language-grounded-marl-2409.17348.pdf|https://arxiv.org/pdf/2409.17348"
  "social|cooperate-or-collapse-2404.16698.pdf|https://arxiv.org/pdf/2404.16698"
  "social|evolving-multi-agent-orchestration-2505.19591.pdf|https://arxiv.org/pdf/2505.19591"
  "social|shapley-coop-credit-2506.07388.pdf|https://arxiv.org/pdf/2506.07388"
  "social|collaborative-preference-learning-2503.01658.pdf|https://arxiv.org/pdf/2503.01658"
  "social|tombench-2402.15052.pdf|https://arxiv.org/pdf/2402.15052"

  "embodiment|dino-world-model-2411.04983.pdf|https://arxiv.org/pdf/2411.04983"
  "embodiment|navigation-world-models-2412.03572.pdf|https://arxiv.org/pdf/2412.03572"
  "embodiment|lapa-latent-action-pretraining-2410.11758.pdf|https://arxiv.org/pdf/2410.11758"
  "embodiment|latent-action-distractors-laom-2502.00379.pdf|https://arxiv.org/pdf/2502.00379"
  "embodiment|heterogeneous-masked-autoregression-2502.04296.pdf|https://arxiv.org/pdf/2502.04296"
  "embodiment|crossformer-cross-embodied-policy-2408.11812.pdf|https://arxiv.org/pdf/2408.11812"
  "embodiment|fast-action-tokenization-2501.09747.pdf|https://arxiv.org/pdf/2501.09747"
  "embodiment|heterogeneous-pretrained-transformers-2409.20537.pdf|https://arxiv.org/pdf/2409.20537"
  "embodiment|octo-generalist-robot-policy-2405.12213.pdf|https://arxiv.org/pdf/2405.12213"
  "embodiment|groot-n1-humanoid-foundation-model-2503.14734.pdf|https://arxiv.org/pdf/2503.14734"
  "embodiment|gemini-robotics-2503.20020.pdf|https://arxiv.org/pdf/2503.20020"
  "embodiment|hi-robot-hierarchical-vla-2502.19417.pdf|https://arxiv.org/pdf/2502.19417"
  "embodiment|navila-legged-navigation-2412.04453.pdf|https://arxiv.org/pdf/2412.04453"
  "embodiment|dreamgen-world-model-data-2505.12705.pdf|https://arxiv.org/pdf/2505.12705"
  "embodiment|spatial-vla-2501.15830.pdf|https://arxiv.org/pdf/2501.15830"
  "embodiment|robospatial-spatial-reasoning-2411.16537.pdf|https://arxiv.org/pdf/2411.16537"

  "neuroscience|thalamocortical-sensory-pe-nature2024.pdf|https://www.nature.com/articles/s41586-024-07851-w.pdf"
  "neuroscience|sleep-synchrony-memory-consolidation-nn2023.pdf|https://www.nature.com/articles/s41593-023-01324-5.pdf"
  "neuroscience|hippocampal-reactivation-balance-science2024.pdf|https://www.science.org/doi/pdf/10.1126/science.ado5708"
  "neuroscience|hippocampal-temporal-structure-nature2024.pdf|https://www.nature.com/articles/s41586-024-07973-1.pdf"
  "neuroscience|hippocampal-abstract-representations-nature2024.pdf|https://www.nature.com/articles/s41586-024-07799-x.pdf"
  "neuroscience|recurrent-planning-replay-nn2024.pdf|https://www.biorxiv.org/content/10.1101/2023.01.16.523429v3.full.pdf"
  "neuroscience|humans-temporally-abstract-world-models.pdf|https://www.biorxiv.org/content/10.1101/2023.11.28.569070v3.full.pdf"
  "neuroscience|cognitive-model-discovery-rnn-neurips2023.pdf|https://papers.nips.cc/paper_files/paper/2023/file/c194ced51c857ec2c1928b02250e0ac8-Paper-Conference.pdf"
  "neuroscience|free-energy-in-vitro-neural-networks-nc2023.pdf|https://www.nature.com/articles/s41467-023-40141-z.pdf"
  "neuroscience|hybrid-memory-reward-learning-nhb2025.pdf|https://www.nature.com/articles/s41562-025-02324-0.pdf"
  "neuroscience|human-dopamine-reward-punishment-pe-2023.pdf|https://fbri.vtc.vt.edu/content/dam/fbri_vtc_vt_edu/publications/montague-publications/4.%20Subsecond%20Fluctuations.pdf"
  "neuroscience|dopamine-errors-informational-domains-2025.pdf|https://www.biorxiv.org/content/10.1101/2023.08.19.553959v3.full.pdf"
  "neuroscience|bumblebee-social-learning-nature2024.pdf|https://www.nature.com/articles/s41586-024-07126-4.pdf"
  "neuroscience|ant-cooperative-clearing-frontiers2025.pdf|https://www.frontiersin.org/journals/behavioral-neuroscience/articles/10.3389/fnbeh.2025.1533372/pdf"
)

mkdir -p "$PAPERS"
SUCCESS=()
FAILED=()

validate_pdf() {
  local path="$1"
  local size
  size=$(stat -f%z "$path" 2>/dev/null || printf '0')
  [[ "$size" -ge 10000 ]] && file -b "$path" | rg -qi 'PDF'
}

for entry in "${ENTRIES[@]}"; do
  IFS='|' read -r axis filename url <<< "$entry"
  out="$PAPERS/$filename"
  printf '==> %s %s\n' "$axis" "$filename"

  if [[ -f "$out" ]] && validate_pdf "$out"; then
    size=$(stat -f%z "$out")
    SUCCESS+=("$axis|$filename|$size")
    printf '    SKIP (%s bytes)\n' "$size"
    continue
  fi

  rm -f "$out"
  ok=false
  for attempt in 1 2 3; do
    if curl --fail --location --silent --show-error \
      --retry 2 --retry-all-errors --connect-timeout 20 --max-time 240 \
      --user-agent "$UA" --output "$out" "$url" && validate_pdf "$out"; then
      size=$(stat -f%z "$out")
      SUCCESS+=("$axis|$filename|$size")
      printf '    OK (%s bytes, attempt %s)\n' "$size" "$attempt"
      ok=true
      break
    fi
    rm -f "$out"
    sleep "$attempt"
  done
  if [[ "$ok" != true ]]; then
    FAILED+=("$axis|$filename|$url")
    printf '    FAILED\n'
  fi
done

{
  printf '# Frontier Map 2024–2026 — Download Summary\n\n'
  printf 'Generated: %s\n\n' "$(date '+%Y-%m-%d %H:%M:%S %z')"
  printf '## Success (%s/%s)\n\n' "${#SUCCESS[@]}" "${#ENTRIES[@]}"
  for item in "${SUCCESS[@]:-}"; do
    [[ -z "$item" ]] && continue
    IFS='|' read -r axis filename size <<< "$item"
    printf -- '- `%s` — %s; %s bytes\n' "$filename" "$axis" "$size"
  done
  printf '\n## Failed / link-only (%s/%s)\n\n' "${#FAILED[@]}" "${#ENTRIES[@]}"
  for item in "${FAILED[@]:-}"; do
    [[ -z "$item" ]] && continue
    IFS='|' read -r axis filename url <<< "$item"
    printf -- '- `%s` — %s; %s\n' "$filename" "$axis" "$url"
  done
} > "$SUMMARY"

printf '\nSummary: %s\n' "$SUMMARY"
[[ "${#FAILED[@]}" -eq 0 ]]
