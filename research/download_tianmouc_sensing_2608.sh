#!/usr/bin/env bash
# Download companion materials for the Nature Sensors 2026 cover article
# "Brain-inspired visual sensing–perception for open-world environments"
# (Lin & Chen et al., doi:10.1038/s44460-026-00095-3, Tsinghua CBICR / Tianmouc team).
#
# The main paper and its Nature 2024 predecessor are PAYWALLED (no public PDF,
# no arXiv preprint, no SharedIt link found as of 2026-08-12) -> link-only,
# see tianmouc-sensing-perception-2026-08/download-summary.md.
# Primary open evidence source is the official code release (cloned separately).
set -uo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPERS="$BASE/papers/tianmouc-sensing-2608"
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 research-survey/1.0"
mkdir -p "$PAPERS"

declare -a ENTRIES=(
  # strand|vz-anchor|filename|url
  # 同团队对 Tianmouc 数据模态（COP RGB / AOP TD+SD）的定量评估，是理解本篇输入表示的最好公开材料
  "modality evidence|R-PE/R8|tianmouc-quantitative-evaluation-bvs-2504.19253.pdf|https://arxiv.org/pdf/2504.19253"
  # 同团队 ICCV 2025：同一互补传感器上的扩散式重建（bottom-up internal model 的邻接方法）
  "adjacent method|R2/R13|diffusion-extreme-highspeed-reconstruction-iccv2025.pdf|https://openaccess.thecvf.com/content/ICCV2025/papers/Meng_Diffusion-Based_Extreme_High-speed_Scenes_Reconstruction_with_the_Complementary_Vision_Sensor_ICCV_2025_paper.pdf"
)

for entry in "${ENTRIES[@]}"; do
  IFS='|' read -r strand anchor fname url <<<"$entry"
  dest="$PAPERS/$fname"
  if [[ -s "$dest" ]] && file "$dest" | grep -q "PDF document"; then
    echo "SKIP (exists): $fname"
    continue
  fi
  echo "GET  [$strand / $anchor] $fname"
  curl -sL -A "$UA" -o "$dest" "$url"
  if ! file "$dest" | grep -q "PDF document"; then
    echo "FAIL (not a PDF, removed): $fname" >&2
    rm -f "$dest"
  fi
done

# Official code release (primary open evidence for the paper's method):
#   git clone --depth 1 https://github.com/Tianmouc/TMC-SSL-Representation.git
# Dataset (129 GB, NOT downloaded):
#   https://huggingface.co/datasets/ordinarabbit/Tianmouc-R
# Link-only (paywalled or anti-bot):
#   main paper   https://doi.org/10.1038/s44460-026-00095-3
#   predecessor  https://doi.org/10.1038/s41586-024-07358-4  (Tianmouc chip, Nature 2024)
#   denoising    https://doi.org/10.1088/2634-4386/ae0a76    (open access but IOP blocks curl)

( cd "$PAPERS" && shasum -a 256 *.pdf > CHECKSUMS.sha256 && cat CHECKSUMS.sha256 )
