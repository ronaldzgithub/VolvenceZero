#!/usr/bin/env bash
set -u
set -o pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
paper_dir="${script_dir}/papers"
summary_tmp="${paper_dir}/.download-summary.tsv.part"
summary_file="${paper_dir}/download-summary.tsv"
checksums_tmp="${paper_dir}/.SHA256SUMS.part"
checksums_file="${paper_dir}/SHA256SUMS"

mkdir -p "${paper_dir}"

entries=(
  "P05_image_parsing_iccv2003.pdf|https://www.robots.ox.ac.uk/~cvrg/michaelmas2003/chen_iccv03.pdf"
  "P06_stochastic_grammar_of_images_2006.pdf|https://dash.harvard.edu/server/api/core/bitstreams/7312037c-4e5a-6bd4-e053-0100007fdf3b/content"
  "P09_understanding_tools_cvpr2015.pdf|https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Zhu_Understanding_Tools_Task-Oriented_2015_CVPR_paper.pdf"
  "P10_inferring_forces_utilities_cvpr2016.pdf|https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhu_Inferring_Forces_and_CVPR_2016_paper.pdf"
  "P12_social_affordance_ijcai2016.pdf|https://www.ijcai.org/Proceedings/16/Papers/488.pdf"
  "P15_interpretable_cnn_cvpr2018.pdf|https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Interpretable_Convolutional_Neural_CVPR_2018_paper.pdf"
  "P16_closed_loop_neural_symbolic_icml2020.pdf|https://proceedings.mlr.press/v119/li20f/li20f.pdf"
  "P17_dark_beyond_deep_2020.pdf|https://arxiv.org/pdf/2004.09044"
  "P18_acre_cvpr2021.pdf|https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_ACRE_Abstract_Causal_REasoning_Beyond_Covariation_CVPR_2021_paper.pdf"
  "P19_in_situ_value_alignment_2022.pdf|https://sites.lifesci.ucla.edu/psych-cvl/wp-content/uploads/sites/162/2022/12/A122.Yuan-et-al.-scirobotics.pdf"
  "P21_communicative_learning_2023.pdf|https://www.engineering.org.cn/engi/EN/PDF/10.1016/j.eng.2022.10.017"
  "P22_tong_test_2024.pdf|https://www.engineering.org.cn/engi/EN/PDF/10.1016/j.eng.2023.07.006"
  "P25_minimax_concept_induction_2024.pdf|https://buzz-beater.github.io/assets/publications/2024_minimax_sciadv/paper.pdf"
  "P26_neural_symbolic_recursive_machine_iclr2024.pdf|https://arxiv.org/pdf/2210.01603"
  "P27_leo_icml2024.pdf|https://arxiv.org/pdf/2311.12871"
  "P28_civrealm_iclr2024.pdf|https://arxiv.org/pdf/2401.10568"
  "P29_adasociety_neurips2024.pdf|https://papers.nips.cc/paper_files/paper/2024/file/3e4d8407cb468850f2f8f4a949e64bf0-Paper-Datasets_and_Benchmarks_Track.pdf"
  "P30_proagent_aaai2024.pdf|https://www.bigai.ai/wp-content/uploads/2024/03/AAAI24_ProAgent.pdf"
  "P31_clova_cvpr2024.pdf|https://openaccess.thecvf.com/content/CVPR2024/papers/Gao_CLOVA_A_Closed-LOop_Visual_Assistant_with_Tool_Usage_and_Update_CVPR_2024_paper.pdf"
  "P32_social_world_model_neurips2025.pdf|https://arxiv.org/pdf/2510.19270"
  "P33_physical_social_world_models_position_2025.pdf|https://arxiv.org/pdf/2510.21219"
  "P34_absolute_zero_neurips2025.pdf|https://arxiv.org/pdf/2505.03335"
  "P35_unifp_corl2025.pdf|https://arxiv.org/pdf/2505.20829"
  "P36_tonggeometry_nmi2026.pdf|https://arxiv.org/pdf/2412.10673"
  "P38_omnixtreme_preprint2026.pdf|https://arxiv.org/pdf/2602.23843"
)

# SciOpen exposes P20 in a browser, but its automated endpoints returned 404/405
# during this archive run. Keep it auditable rather than bypassing access controls.
link_only_entries=(
  "P20_artificial_social_intelligence_2023.pdf|https://www.sciopen.com/article/pdf/10.26599/AIR.2022.9150010.pdf"
)

is_pdf() {
  local candidate="$1"

  if [[ ! -s "${candidate}" ]]; then
    return 1
  fi
  if ! head -c 1024 "${candidate}" | LC_ALL=C grep -aq '%PDF-'; then
    return 1
  fi
  if ! tail -c 4096 "${candidate}" | LC_ALL=C grep -aq '%%EOF'; then
    return 1
  fi
  if command -v pdfinfo >/dev/null 2>&1 && ! pdfinfo "${candidate}" >/dev/null 2>&1; then
    return 1
  fi
  return 0
}

printf 'status\tfile\turl\n' > "${summary_tmp}"
failure_count=0

for entry in "${entries[@]}"; do
  filename="${entry%%|*}"
  url="${entry#*|}"
  destination="${paper_dir}/${filename}"
  partial="${paper_dir}/.${filename}.part"

  if is_pdf "${destination}"; then
    printf 'PRESENT\t%s\t%s\n' "${filename}" "${url}" >> "${summary_tmp}"
    continue
  fi

  if [[ -e "${destination}" ]]; then
    printf 'INVALID_EXISTING\t%s\t%s\n' "${filename}" "${url}" >> "${summary_tmp}"
    failure_count=$((failure_count + 1))
    continue
  fi

  rm -f -- "${partial}"
  if curl --fail --location --silent --show-error --retry 3 --retry-delay 2 --connect-timeout 20 \
      --user-agent 'VolvenceResearch/1.0 (+open academic archive)' \
      --output "${partial}" "${url}" && is_pdf "${partial}"; then
    mv -- "${partial}" "${destination}"
    printf 'DOWNLOADED\t%s\t%s\n' "${filename}" "${url}" >> "${summary_tmp}"
  else
    rm -f -- "${partial}"
    printf 'FAILED\t%s\t%s\n' "${filename}" "${url}" >> "${summary_tmp}"
    failure_count=$((failure_count + 1))
  fi
done

for entry in "${link_only_entries[@]}"; do
  filename="${entry%%|*}"
  url="${entry#*|}"
  printf 'LINK_ONLY_ACCESS\t%s\t%s\n' "${filename}" "${url}" >> "${summary_tmp}"
done

mv -- "${summary_tmp}" "${summary_file}"

find "${paper_dir}" -maxdepth 1 -type f -name '*.pdf' -print0 \
  | sort -z \
  | xargs -0 shasum -a 256 \
  | sed "s#  ${paper_dir}/#  #" > "${checksums_tmp}"
mv -- "${checksums_tmp}" "${checksums_file}"

downloaded_count="$(find "${paper_dir}" -maxdepth 1 -type f -name '*.pdf' | wc -l | tr -d ' ')"
printf 'Validated PDFs: %s; failed targets: %s\n' "${downloaded_count}" "${failure_count}"

if (( failure_count > 0 )); then
  exit 1
fi
