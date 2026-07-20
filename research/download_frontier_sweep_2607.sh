#!/usr/bin/env bash
# Download the July 2026 frontier-labs sweep. Every artifact is validated as
# a non-trivial PDF; incomplete or HTML responses are removed automatically.
set -uo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPERS="$BASE/papers/sweep-2607"
SUMMARY="$BASE/frontier-sweep-2026-07-20-download-summary.md"
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 research-survey/1.0"

declare -a ENTRIES=(
  # lab|primitive|filename|url
  "Meta FAIR / MSL|P6|hyperagents-metacognitive-self-modification-2603.19461.pdf|https://arxiv.org/pdf/2603.19461"
  "Meta FAIR|P4|s-ember-streaming-egocentric-memory-retrieval-2607.02689.pdf|https://arxiv.org/pdf/2607.02689"
  "Meta MSL|P4/R11|personalized-agents-human-feedback-2602.16173.pdf|https://arxiv.org/pdf/2602.16173"
  "Meta MSL|P4/R9|reason-by-analogy-ra-rft-2606.13680.pdf|https://arxiv.org/pdf/2606.13680"
  "Anthropic|P7/R7|emotion-concepts-function-llm-2604.07729.pdf|https://arxiv.org/pdf/2604.07729"
  "OpenAI|P7/R12/R15|deployment-simulation-llm-safety-2607.07184.pdf|https://arxiv.org/pdf/2607.07184"
  "Google DeepMind|P7/R12|gram-agent-sabotage-auditing-2605.30322.pdf|https://arxiv.org/pdf/2605.30322"
  "Google DeepMind|P7/R12|realistic-honeypot-scheming-evaluation-2605.29729.pdf|https://arxiv.org/pdf/2605.29729"
  "Google DeepMind|P7/R12|proeval-proactive-failure-discovery-2604.23099.pdf|https://arxiv.org/pdf/2604.23099"
  "Sakana AI / MIT|P6|digital-red-queen-adversarial-evolution-2601.03335.pdf|https://arxiv.org/pdf/2601.03335"
  "Sakana AI|P3/P6|sakana-fugu-agent-orchestration-2606.21228.pdf|https://arxiv.org/pdf/2606.21228"
  "UNC / UC Berkeley / UCSC|P4/P6/R15|evolvemem-guarded-memory-autoresearch-2605.13941.pdf|https://arxiv.org/pdf/2605.13941"
  "Google Cloud AI Research / UIUC / MIT|P4/R9|skillos-self-evolving-skill-curation-2605.06614.pdf|https://arxiv.org/pdf/2605.06614"
  "T-Tech|P1/P2/P5|ne-dreamer-next-embedding-world-model-2603.02765.pdf|https://arxiv.org/pdf/2603.02765"
  "UALR (IROS 2026)|P3/P6/R15|formal-verification-learned-marl-policies-2606.19632.pdf|https://arxiv.org/pdf/2606.19632"
  "Meta / Google Research|P4/R1/R5|online-neural-space-time-memory-2607.15271.pdf|https://arxiv.org/pdf/2607.15271"
  "Google Research / Mila|P5/R-PE|icl-intrinsic-curiosity-2606.19476.pdf|https://arxiv.org/pdf/2606.19476"
  "Google Research|P7/R12|geometric-signatures-reasoning-2607.01571.pdf|https://arxiv.org/pdf/2607.01571"
  "CMU|P4/R12/R15|behavioral-eval-deployment-memory-2607.00368.pdf|https://arxiv.org/pdf/2607.00368"
  "Google Cloud AI Research|P4/R9/R15|mars-reflective-search-ai-research-2602.02660.pdf|https://arxiv.org/pdf/2602.02660"
  "Anthropic Fellows|P7/R12/R15|cross-architecture-model-diffing-2602.11729.pdf|https://arxiv.org/pdf/2602.11729"
  "Meta Reality Labs|P4/R9/R11|salimory-cognitive-memory-orchestration-2606.04120.pdf|https://arxiv.org/pdf/2606.04120"
  "CMU / Princeton / Cartesia lineage|P1/P4|mamba-3-state-space-principles-2603.15569.pdf|https://arxiv.org/pdf/2603.15569"
  "MIT / Liquid AI|P2/P5/P6|real-time-recurrent-rl-adaptive-control-2602.02236.pdf|https://arxiv.org/pdf/2602.02236"
  "Sakana AI|P3/R8/R11|sheaf-admm-multi-agent-coordination-2605.31005.pdf|https://arxiv.org/pdf/2605.31005"
  "Physical Intelligence|P2/P4|multi-scale-embodied-memory-2603.03596.pdf|https://arxiv.org/pdf/2603.03596"
  "Friston / UCL / Monash|P5/R-PE|active-inference-test-time-scaling-2606.22813.pdf|https://arxiv.org/pdf/2606.22813"
  "Independent memory research|P4/P6/R15|forgetful-attention-exact-unlearning-2607.12204.pdf|https://arxiv.org/pdf/2607.12204"
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
  IFS='|' read -r lab primitive filename url <<< "$entry"
  out="$PAPERS/$filename"
  printf '==> %s [%s] %s\n' "$lab" "$primitive" "$filename"

  if [[ -f "$out" ]] && validate_pdf "$out"; then
    size=$(stat -f%z "$out")
    printf '    SKIP valid existing PDF (%s bytes)\n' "$size"
    SUCCESS+=("$lab|$primitive|$filename|$size")
    continue
  fi

  rm -f "$out"
  ok=false
  for attempt in 1 2 3; do
    if curl --fail --location --silent --show-error \
      --retry 2 --retry-all-errors --connect-timeout 20 --max-time 240 \
      --user-agent "$UA" --output "$out" "$url" && validate_pdf "$out"; then
      size=$(stat -f%z "$out")
      printf '    OK (%s bytes, attempt %s)\n' "$size" "$attempt"
      SUCCESS+=("$lab|$primitive|$filename|$size")
      ok=true
      break
    fi
    rm -f "$out"
    printf '    retrying after failed validation (attempt %s)\n' "$attempt"
    sleep "$attempt"
  done

  if [[ "$ok" != true ]]; then
    FAILED+=("$lab|$primitive|$filename|$url")
    printf '    FAILED\n'
  fi
done

{
  printf '# Frontier Labs Sweep — Download Summary\n\n'
  printf 'Generated: %s\n\n' "$(date '+%Y-%m-%d %H:%M:%S %z')"
  printf '## Success (%s/%s)\n\n' "${#SUCCESS[@]}" "${#ENTRIES[@]}"
  for item in "${SUCCESS[@]:-}"; do
    [[ -z "$item" ]] && continue
    IFS='|' read -r lab primitive filename size <<< "$item"
    printf -- '- `%s` — %s; %s; %s bytes\n' "$filename" "$lab" "$primitive" "$size"
  done
  printf '\n## Failed (%s/%s)\n\n' "${#FAILED[@]}" "${#ENTRIES[@]}"
  for item in "${FAILED[@]:-}"; do
    [[ -z "$item" ]] && continue
    IFS='|' read -r lab primitive filename url <<< "$item"
    printf -- '- `%s` — %s; %s; %s\n' "$filename" "$lab" "$primitive" "$url"
  done
} > "$SUMMARY"

printf '\nSummary: %s\n' "$SUMMARY"
[[ "${#FAILED[@]}" -eq 0 ]]
