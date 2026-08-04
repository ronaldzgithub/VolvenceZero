#!/usr/bin/env bash
# Download the activation-steering / representation-intervention literature that
# frames Volvence 的 ETA→LLM steering 路线（S1 识别 / S2 因果 / S3 学何时出手）。
# 触发：Stage-3 kill-eta(operationalization) 后，转向 "读残差 + 有界 steering + Internal RL 学干预策略"，
# 需要把学界已定罪的失败模式与最优路线沉淀为可引用证据。
#
# 已对 research/** 去重（无重叠 ID）；与既有 KV-cache steering (2507.08799) 不同来源。
set -uo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPERS="$BASE/steering-2026-08/papers"
SUMMARY="$BASE/steering-2026-08/download-summary.md"
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 research-survey/1.0"

mkdir -p "$PAPERS"

declare -a ENTRIES=(
  # strand|vz-anchor|filename|url

  # A 失败模式已知：naive static steering 的可靠性负结果（对应我们 S2 复现的负结果）
  "A reliability|S2/R4|understanding-unreliability-of-steering-vectors-2505.22637.pdf|https://arxiv.org/pdf/2505.22637"
  "A reliability|S2/R15|steering-off-course-reliability-challenges-acl2025.pdf|https://aclanthology.org/2025.acl-long.974.pdf"
  "A reliability|S2/R12/R15|faithsteer-bench-deployment-stress-test-2603.18329.pdf|https://arxiv.org/pdf/2603.18329"

  # B 方向来源：probe 权重错，均值差/学习式对（对应 S2 死因1：方向来源）
  "B direction|S2/R4/R-PE|contrastive-activation-addition-caa-2312.06681.pdf|https://arxiv.org/pdf/2312.06681"
  "B direction|S1/R4|generative-causal-mediation-where-to-steer-2602.16080.pdf|https://arxiv.org/pdf/2602.16080"

  # C 学习式干预：当前最强路线（对应我们 B screen = ReFT 家族）
  "C learned|S2/S3/R3/R4|reft-representation-finetuning-loreft-2404.03592.pdf|https://arxiv.org/pdf/2404.03592"
  "C learned|S3/R3/R4/R9|reps-reference-free-preference-steering-neurips2025.pdf|https://proceedings.neurips.cc/paper_files/paper/2025/file/eb1ef82926376d252dde00d5dd909f4b-Paper-Conference.pdf"

  # D 综述 + 条件门控：排序证据与 "何时出手"（对应 S3 = CAST + Internal RL）
  "D survey/gating|S1/S2/S3|from-weights-to-activations-repe-survey-acl2026-long1377.pdf|https://aclanthology.org/2026.acl-long.1377.pdf"
  "D survey/gating|S3/R9/R10|conditional-activation-steering-cast-2409.05907.pdf|https://arxiv.org/pdf/2409.05907"
)

echo "# Steering / Representation-Intervention 下载清单（2026-08）" > "$SUMMARY"
echo "" >> "$SUMMARY"
echo "| strand | vz-anchor | file | status | sha256 | url |" >> "$SUMMARY"
echo "|---|---|---|---|---|---|" >> "$SUMMARY"

for entry in "${ENTRIES[@]}"; do
  IFS='|' read -r strand anchor fname url <<< "$entry"
  dest="$PAPERS/$fname"
  status="ok"
  if [[ -f "$dest" ]]; then
    status="skip-exists"
  else
    if ! curl -fsSL -A "$UA" --retry 3 --retry-delay 2 -o "$dest" "$url"; then
      status="FAIL-download"
    elif ! head -c 5 "$dest" | grep -q "%PDF"; then
      status="FAIL-not-pdf"
      rm -f "$dest"
    fi
  fi
  sha="-"
  if [[ -f "$dest" ]]; then
    sha="$(shasum -a 256 "$dest" | cut -d' ' -f1 | cut -c1-16)"
  fi
  echo "| $strand | $anchor | \`$fname\` | $status | $sha | $url |" >> "$SUMMARY"
  echo "[$status] $fname  ($sha)"
done

echo ""
echo "汇总写入: $SUMMARY"
