#!/usr/bin/env bash
# 下载杨柳（Liu Yang，CMU PhD 2013，Hanneke 长期合作者）论文集 40 篇中可公开获取的 37 篇 + 6 份补充材料。
# 论文编号对应 EmoGPT/emogpt_docs/yangliu/副本杨柳论文集20230630V2.4.txt 的"按重要程度"序号。
# 不可下载（仅登记引用）：#02 Annals of Statistics 投稿中；#37 IJCAI-WS 2007 无公开版；#40 2004 华中科技大学学报。
# 用法：bash download_yangliu_papers.sh   （幂等：已存在且非空的文件跳过）
set -uo pipefail
cd "$(dirname "$0")/papers"
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) research-download"

dl() {
  local name="$1" url="$2"
  if [ -s "$name" ]; then echo "SKIP  $name"; return 0; fi
  if curl -sL --retry 2 --max-time 120 -A "$UA" -o "$name" "$url"; then
    if head -c 5 "$name" | grep -q "%PDF"; then
      echo "OK    $name  ($(wc -c < "$name" | tr -d ' ') bytes)"
    else
      echo "FAIL  $name  (非 PDF 响应) $url"; mv "$name" "$name.bad"
    fi
  else
    echo "FAIL  $name  (下载失败) $url"
  fi
}

# ============ 核心 40 篇（37 篇可下载） ============
dl 01-bandit-learnability-undecidable-colt2023.pdf        "http://web.ics.purdue.edu/~hanneke/docs/2023/Bandits-Hanneke-Yang-2023.pdf"
# 02 Active Learning with Identifiable Mixture Models —— 投稿中，无公开 PDF（引用登记）
dl 03-reliable-active-apprenticeship-learning-alt2025.pdf "https://raw.githubusercontent.com/mlresearch/v272/main/assets/hanneke25a/hanneke25a.pdf"
dl 04-online-selective-sampling-aistats2021.pdf           "http://web.ics.purdue.edu/~hanneke/docs/2021/selective-sampling.pdf"
dl 05-small-connectivity-local-cut-soda2020.pdf           "https://arxiv.org/pdf/1910.14344"
dl 06-nonstationary-mixing-processes-aistats2019.pdf      "http://web.ics.purdue.edu/~hanneke/docs/2018/mixing-drift.pdf"
dl 07-surrogate-losses-passive-active-ejs2019.pdf         "http://web.ics.purdue.edu/~hanneke/docs/2012/surrogate-losses.pdf"
dl 08-theory-transfer-learning-active-ml2013.pdf          "http://web.ics.purdue.edu/~hanneke/docs/2010/atl.pdf"
dl 09-minimax-analysis-active-learning-jmlr2015.pdf       "http://web.ics.purdue.edu/~hanneke/docs/2014/hanneke14a.pdf"
dl 10-identifiability-priors-transfer-colt2011.pdf        "http://web.ics.purdue.edu/~hanneke/docs/2011/transfer.pdf"
dl 11-active-learning-drifting-distribution-nips2011.pdf  "https://www.cs.cmu.edu/~liuy/active_distribution_nips.pdf"
dl 12-learning-drifting-target-concept-alt2015.pdf        "http://web.ics.purdue.edu/~hanneke/docs/2015/concept-drift.pdf"
dl 13-buy-in-bulk-active-learning-nips2013.pdf            "https://papers.nips.cc/paper_files/paper/2013/file/43baa6762fa81bb43b39c62553b2970d-Paper.pdf"
dl 14-active-property-testing-focs2012.pdf                "https://arxiv.org/pdf/1111.0897"
dl 15-prior-estimation-vc-minimax-alt2015.pdf             "http://web.ics.purdue.edu/~hanneke/docs/2015/prior-estimation.pdf"
dl 16-bayesian-al-binary-queries-alt2010.pdf              "https://www.cs.cmu.edu/~liuy/alt_bayesian_active_liuy.pdf"
dl 17-activized-learning-uniform-noise-icml2013.pdf       "http://web.ics.purdue.edu/~hanneke/docs/2013/unif_activize.pdf"
dl 18-online-learning-ellipsoid-icml2009.pdf              "https://icml.cc/Conferences/2009/papers/472.pdf"
dl 19-online-allocation-economies-scale-wine2015.pdf      "https://www.cs.cmu.edu/~liuy/DecMarCost.pdf"
dl 20-risk-averse-matchings-ecml2018.pdf                  "https://arxiv.org/pdf/1801.03190"
dl 21-prior-estimation-vc-minimax-tcs2018.pdf             "http://web.ics.purdue.edu/~hanneke/docs/2016/full-prior-estimation.pdf"
dl 22-dnf-representation-specific-queries-itcs2013.pdf    "https://www.cs.cmu.edu/~liuy/dnf_queries_ITCS.pdf"
dl 23-testing-piecewise-functions-tcs2018.pdf             "http://web.ics.purdue.edu/~hanneke/docs/2017/piecewise-functions.pdf"
dl 24-self-verifying-bayesian-al-aistats2011.pdf          "http://web.ics.purdue.edu/~hanneke/docs/2011/self-verifying.pdf"
dl 25-negative-results-convex-losses-aistats2010.pdf      "http://web.ics.purdue.edu/~hanneke/docs/2010/convex-active-minimax.pdf"
dl 26-dynamic-matrix-factorization-social-mlsp2016.pdf    "https://arxiv.org/pdf/1604.06194"
# 27 独立 PDF 已失效（donsheehy.net 404）；改用 EuroCG 2016 官方论文集（目标论文在 pp.159-162）
dl 27-eurocg2016-proceedings-booklet.pdf                  "https://computational-geometry.org/abstracts/eurocg/2016.pdf"
dl 28-boosting-visuality-preserving-dml-tpami2010.pdf     "http://www.cs.cmu.edu/~satya/docdir/yang-pami-2010.pdf"
dl 29-distortion-one-bad-point-fwcg2015.pdf               "https://www.cse.buffalo.edu/fwcg2015/assets/pdf/FWCG_2015_paper_26.pdf"
dl 30-proactive-learning-cost-complexity-cmu-ml-09-113.pdf "http://reports-archive.adm.cs.cmu.edu/anon/ml2009/CMU-ML-09-113.pdf"
dl 31-adaptive-proactive-learning-cmu-ml-09-114.pdf       "http://reports-archive.adm.cs.cmu.edu/anon/ml2009/CMU-ML-09-114.pdf"
dl 32-ssl-weakly-related-unlabeled-nips2008.pdf           "http://papers.neurips.cc/paper/3488-semi-supervised-learning-with-weakly-related-unlabeled-data-towards-better-text-categorization.pdf"
dl 33-unifying-codebook-classifier-cvpr2008.pdf           "https://www.cs.cmu.edu/~rahuls/pub/cvpr2008-unified-rahuls.pdf"
dl 34-bayesian-active-dml-uai2007.pdf                     "https://www.cs.cmu.edu/~liuy/uai2007_bayesian.pdf"
dl 35-discriminative-cluster-refinement-cvpr2007.pdf      "https://www.cs.cmu.edu/~liuy/cvpr2007-dcr.pdf"
dl 36-distance-metrics-mammograms-spie2007.pdf            "https://www.cs.cmu.edu/~liuy/liuy-mi2007.pdf"
# 37 Resource-constrained Supervised Dimensionality Reduction (IJCAI-WS 2007) —— 无公开 PDF（引用登记）
dl 38-local-distance-metric-learning-aaai2006.pdf         "https://www.cs.cmu.edu/~liuy/aaai2006-distance-v7.pdf"
dl 39-ssl-multilabel-cnmf-aaai2006.pdf                    "https://cdn.aaai.org/AAAI/2006/AAAI06-067.pdf"
# 40 基于边缘匹配与多尺度小波变换的图像配准算法（华中科技大学学报 2004）—— 无公开 PDF（引用登记）

# ============ 补充材料 ============
dl supp-phd-thesis-oracles-cmu2013.pdf                    "https://ndownloader.figshare.com/files/12254909"
dl supp-cv-liu-yang-2014.pdf                              "https://www.cs.cmu.edu/~liuy/cv_liu_yang_2014.pdf"
dl supp-buy-in-bulk-techreport-cmu-ml-12-110.pdf          "http://reports-archive.adm.cs.cmu.edu/anon/ml2011/CMU-ML-12-110.pdf"
dl supp-lossy-coding-journal-manuscript.pdf               "https://www.cs.cmu.edu/~liuy/ent-journal.pdf"
dl supp-dml-comprehensive-survey-msu2006.pdf              "https://www.cs.cmu.edu/~liuy/dist_overview.pdf"
dl supp-activized-icml2013-supplemental.pdf               "http://web.ics.purdue.edu/~hanneke/docs/2013/supplemental.pdf"

echo; echo "== 完成。文件统计：$(ls -1 *.pdf 2>/dev/null | wc -l | tr -d ' ') 个 PDF =="
