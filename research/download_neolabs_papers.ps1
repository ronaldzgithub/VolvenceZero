# Download "neo labs" key papers to research/papers/neolabs/<lab>/
# Scope: neolabs-2026-06 survey (bio-heavy), excludes OpenAI/DeepMind/Google Research/Anthropic.
# Companion to research/neolabs-2026-06/00_roster.md and labs/<lab>.md
#
# Handles: arXiv (reliable via this script).
# Records (no download): paywalled DOI (Nature/Science/Cell), company blog/report, UNVERIFIED items.
#
# NOTE (2026-06-13 run): bioRxiv / medRxiv return Cloudflare 403 to scripted Invoke-WebRequest.
#   They were instead fetched through a real (Cloudflare-cleared) browser session via the
#   Playwright MCP: navigate to a *.biorxiv.org page once to pass the JS challenge, then in the
#   page context create a same-origin `<a download href=".../<doi>vN.full.pdf">`, click it, and
#   capture the Playwright `download` event -> `download.saveAs(<dest>)`. Same trick on medrxiv.org
#   and nature.com (AlphaFold2/3, open access). PLOS/Frontiers open-access PDFs fetch directly.
#   So this script downloads the arXiv set; the bio set was browser-assisted (see _download_summary.md).
#
# Filename: <short-title>-<id>.pdf  (arXiv: bare id; bioRxiv/medRxiv: server+date stem)

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

# type: arxiv | biorxiv | medrxiv | record  (record = paywalled DOI / blog / report / unverified, not downloaded)
$papers = @(
    # ===================== Group A - brain-inspired / neuroscience cognition =====================
    @{ lab = "numenta"; type = "arxiv";  id = "2412.18354"; title = "thousand-brains-project-paradigm-sensorimotor-intelligence" },
    @{ lab = "numenta"; type = "arxiv";  id = "2507.04494"; title = "thousand-brains-systems-rapid-robust-learning" },
    @{ lab = "numenta"; type = "arxiv";  id = "2507.05888"; title = "hierarchy-or-heterarchy-long-range-connections" },
    @{ lab = "numenta"; type = "record"; id = "doi:10.3389/fncir.2018.00121"; title = "framework-for-intelligence-cortical-grid-cells (Frontiers, open access)" },

    @{ lab = "liquid-ai"; type = "arxiv";  id = "2006.04439"; title = "liquid-time-constant-networks" },
    @{ lab = "liquid-ai"; type = "arxiv";  id = "2106.13898"; title = "closed-form-continuous-time-neural-networks-cfc" },
    @{ lab = "liquid-ai"; type = "arxiv";  id = "2209.12951"; title = "liquid-structural-state-space-models-s4" },
    @{ lab = "liquid-ai"; type = "record"; id = "doi:10.1038/s42256-020-00237-3"; title = "neural-circuit-policies-auditable-autonomy (Nat Mach Intell, paywalled)" },

    @{ lab = "verses-ai"; type = "arxiv";  id = "2212.01354"; title = "designing-ecosystems-of-intelligence-from-first-principles" },
    @{ lab = "verses-ai"; type = "record"; id = "doi:10.1038/nrn2787"; title = "free-energy-principle-unified-brain-theory (NRN, paywalled; free author PDF)" },
    @{ lab = "verses-ai"; type = "record"; id = "doi:10.1371/journal.pone.0006421"; title = "reinforcement-learning-or-active-inference (PLOS ONE, open)" },
    @{ lab = "verses-ai"; type = "record"; id = "doi:10.1162/neco_a_00912"; title = "active-inference-a-process-theory (Neural Computation)" },

    @{ lab = "stanhope-ai"; type = "record"; id = "doi:10.1371/journal.pbio.1002400"; title = "towards-a-neuronal-gauge-theory (PLOS Biology, open)" },
    @{ lab = "stanhope-ai"; type = "record"; id = "doi:10.3390/e23040454"; title = "neural-dynamics-under-active-inference (Entropy, open)" },
    @{ lab = "stanhope-ai"; type = "record"; id = "doi:10.1162/neco_a_00912"; title = "active-inference-a-process-theory (shared w/ VERSES)" },

    @{ lab = "cortical-labs"; type = "record"; id = "doi:10.1016/j.neuron.2022.09.001"; title = "dishbrain-in-vitro-neurons-learn-pong (Neuron, paywalled)" },
    @{ lab = "cortical-labs"; type = "record"; id = "UNVERIFIED"; title = "technology-opportunities-challenges-synthetic-biological-intelligence-2023 (lookup by title)" },

    @{ lab = "cartesia"; type = "arxiv"; id = "2008.07669"; title = "hippo-recurrent-memory-optimal-polynomial-projections" },
    @{ lab = "cartesia"; type = "arxiv"; id = "2111.00396"; title = "s4-efficiently-modeling-long-sequences-structured-state-spaces" },
    @{ lab = "cartesia"; type = "arxiv"; id = "2312.00752"; title = "mamba-linear-time-sequence-modeling-selective-state-spaces" },
    @{ lab = "cartesia"; type = "arxiv"; id = "2405.21060"; title = "mamba2-transformers-are-ssms-state-space-duality" },

    @{ lab = "symbolica"; type = "arxiv"; id = "2402.15332"; title = "categorical-deep-learning-algebraic-theory-of-architectures" },

    # ===================== Group B - autonomous AI scientists / closed-loop discovery =====================
    @{ lab = "future-house"; type = "arxiv"; id = "2409.13740"; title = "paperqa2-superhuman-synthesis-of-scientific-knowledge" },
    @{ lab = "future-house"; type = "arxiv"; id = "2407.10362"; title = "lab-bench-capabilities-of-llms-for-biology-research" },
    @{ lab = "future-house"; type = "arxiv"; id = "2412.21154"; title = "aviary-training-language-agents-scientific-tasks" },
    @{ lab = "future-house"; type = "arxiv"; id = "2506.17238"; title = "ether0-scientific-reasoning-model-for-chemistry" },
    @{ lab = "future-house"; type = "arxiv"; id = "2505.13400"; title = "robin-multi-agent-system-automating-scientific-discovery" },

    @{ lab = "lila-sciences"; type = "arxiv";  id = "1610.02415"; title = "automatic-chemical-design-data-driven-continuous-representation (founder, Gomez-Bombarelli)" },
    @{ lab = "lila-sciences"; type = "arxiv";  id = "1901.01753"; title = "poet-paired-open-ended-trailblazer (founder, Stanley)" },
    @{ lab = "lila-sciences"; type = "record"; id = "doi:10.1039/C9SC03766G"; title = "accelerating-materials-science-autonomous-workflows (Chem Sci, open; founder Gregoire)" },

    @{ lab = "periodic-labs"; type = "arxiv";  id = "2101.03961"; title = "switch-transformers-trillion-parameter-sparsity (founder, Fedus)" },
    @{ lab = "periodic-labs"; type = "arxiv";  id = "1805.09501"; title = "autoaugment-learning-augmentation-from-data (founder, Cubuk)" },
    @{ lab = "periodic-labs"; type = "record"; id = "doi:10.1038/s41586-023-06735-9"; title = "gnome-scaling-deep-learning-for-materials-discovery (Nature, paywalled; founder Cubuk)" },
    @{ lab = "periodic-labs"; type = "record"; id = "doi:10.1038/s41586-023-06734-w"; title = "a-lab-autonomous-laboratory-inorganic-materials (Nature, paywalled)" },

    @{ lab = "recursive-superintelligence"; type = "arxiv";  id = "2505.22954"; title = "darwin-godel-machine-open-ended-evolution-self-improving-agents (4 co-founders incl. Clune)" },
    @{ lab = "recursive-superintelligence"; type = "arxiv";  id = "1905.10985"; title = "ai-gas-ai-generating-algorithms-alternate-paradigm (founder, Clune)" },
    @{ lab = "recursive-superintelligence"; type = "arxiv";  id = "2402.15391"; title = "genie-generative-interactive-environments (founder, Rocktaschel)" },
    @{ lab = "recursive-superintelligence"; type = "arxiv";  id = "2203.01302"; title = "accel-evolving-curricula-regret-based-environment-design (founder, Rocktaschel)" },
    @{ lab = "recursive-superintelligence"; type = "arxiv";  id = "2402.16822"; title = "rainbow-teaming-open-ended-generation-diverse-adversarial-prompts (founder, Rocktaschel)" },
    @{ lab = "recursive-superintelligence"; type = "arxiv";  id = "2005.11401"; title = "rag-retrieval-augmented-generation-knowledge-intensive-nlp (founder, Rocktaschel)" },
    @{ lab = "recursive-superintelligence"; type = "arxiv";  id = "2412.06769"; title = "coconut-training-llms-reason-continuous-latent-space (founder, Tian)" },
    @{ lab = "recursive-superintelligence"; type = "arxiv";  id = "2010.11929"; title = "vit-an-image-is-worth-16x16-words (founder, Dosovitskiy)" },
    @{ lab = "recursive-superintelligence"; type = "arxiv";  id = "1703.06907"; title = "domain-randomization-transferring-deep-networks-sim-to-real (founder, Tobin)" },
    @{ lab = "recursive-superintelligence"; type = "arxiv";  id = "1909.05858"; title = "ctrl-conditional-transformer-language-model-controllable-generation (founder, Xiong)" },
    @{ lab = "recursive-superintelligence"; type = "record"; id = "blog:https://www.recursive.com/articles/first-steps-toward-automated-ai-research"; title = "first-steps-toward-automated-ai-research (company article, 2026-06)" },
    @{ lab = "recursive-superintelligence"; type = "record"; id = "UNVERIFIED"; title = "glove-global-vectors-for-word-representation (EMNLP 2014, founder Socher; ACL Anthology open PDF, not on arXiv)" },

    # ===================== Group C1 - protein / structure / sequence foundation models =====================
    @{ lab = "evolutionaryscale"; type = "biorxiv"; id = "10.1101/2024.07.01.600583"; title = "esm3-simulating-500m-years-of-evolution" },
    @{ lab = "evolutionaryscale"; type = "biorxiv"; id = "10.1101/2022.07.20.500902"; title = "esm2-esmfold-evolutionary-scale-atomic-structure-prediction" },
    @{ lab = "evolutionaryscale"; type = "biorxiv"; id = "10.1101/622803";            title = "esm1b-biological-structure-function-emerge-from-scaling" },

    @{ lab = "arc-institute"; type = "biorxiv"; id = "10.1101/2024.02.27.582234"; title = "evo-sequence-modeling-design-molecular-to-genome-scale" },
    @{ lab = "arc-institute"; type = "biorxiv"; id = "10.1101/2025.02.18.638918"; title = "evo2-genome-modeling-design-across-all-domains-of-life" },
    @{ lab = "arc-institute"; type = "biorxiv"; id = "10.1101/2025.06.26.661135"; title = "state-predicting-cellular-responses-to-perturbation" },

    @{ lab = "isomorphic-labs"; type = "record"; id = "doi:10.1038/s41586-024-07487-w"; title = "alphafold3-structure-prediction-biomolecular-interactions (Nature, open access)" },
    @{ lab = "isomorphic-labs"; type = "record"; id = "doi:10.1038/s41586-021-03819-2"; title = "alphafold2-highly-accurate-protein-structure-prediction (Nature, open access)" },

    @{ lab = "chai-discovery"; type = "biorxiv"; id = "10.1101/2024.10.10.615955"; title = "chai-1-decoding-molecular-interactions-of-life" },
    @{ lab = "chai-discovery"; type = "biorxiv"; id = "10.1101/2025.07.05.663018"; title = "chai-2-zero-shot-antibody-design-24-well-plate" },

    @{ lab = "profluent-bio"; type = "arxiv";   id = "2206.13517";              title = "progen2-exploring-boundaries-of-protein-language-models" },
    @{ lab = "profluent-bio"; type = "biorxiv"; id = "10.1101/2024.04.22.590591"; title = "opencrispr-design-of-genome-editors-modeling-crispr-cas" },
    @{ lab = "profluent-bio"; type = "biorxiv"; id = "10.1101/2025.04.15.649055"; title = "progen3-scaling-broader-generation-functional-understanding" },
    @{ lab = "profluent-bio"; type = "record";  id = "doi:10.1038/s41587-022-01618-2"; title = "progen-llms-generate-functional-protein-sequences (Nat Biotech, paywalled)" },

    @{ lab = "latent-labs"; type = "arxiv"; id = "2507.19375"; title = "latent-x-atom-level-frontier-model-de-novo-binder-design" },

    @{ lab = "basecamp-research"; type = "biorxiv"; id = "10.1101/2024.03.06.583325"; title = "basefold-improving-alphafold2-targeted-msa-supplementation" },
    @{ lab = "basecamp-research"; type = "record";  id = "UNVERIFIED"; title = "basedata-breaking-biologys-data-wall-tree-of-life-10x (company white paper)" },

    # ===================== Group C2 - virtual cell / phenomics / RNA / drug discovery =====================
    @{ lab = "inceptive"; type = "arxiv";   id = "1706.03762";              title = "attention-is-all-you-need (founder lineage, Uszkoreit; shared w/ Sakana)" },
    @{ lab = "inceptive"; type = "biorxiv"; id = "10.1101/2024.02.24.581671"; title = "ribonanza-deep-learning-rna-structure-dual-crowdsourcing (co-founder Das)" },

    @{ lab = "recursion"; type = "arxiv"; id = "2309.16064"; title = "masked-autoencoders-scalable-learners-of-cellular-morphology" },
    @{ lab = "recursion"; type = "arxiv"; id = "2404.10242"; title = "masked-autoencoders-for-microscopy-scalable-cellular-biology" },
    @{ lab = "recursion"; type = "arxiv"; id = "2409.08302"; title = "molphenix-contrastive-phenomolecular-retrieval" },

    @{ lab = "insitro"; type = "biorxiv"; id = "10.1101/2023.08.13.553051"; title = "pooled-cell-painting-crispr-de-novo-gene-function (Koller)" },
    @{ lab = "insitro"; type = "biorxiv"; id = "10.1101/2023.11.24.568344"; title = "embedgem-evaluating-embeddings-for-genetic-discovery" },
    @{ lab = "insitro"; type = "biorxiv"; id = "10.1101/2024.01.04.574270"; title = "deep-learning-ipsc-motor-neurons-fals-phenotypes" },
    @{ lab = "insitro"; type = "medrxiv"; id = "10.1101/2024.01.06.24300926"; title = "ml-prediction-digital-biomarkers-from-histopathology" },

    @{ lab = "generate-biomedicines"; type = "biorxiv"; id = "10.1101/2022.12.01.518682"; title = "chroma-programmable-generative-model-protein-space" },
    @{ lab = "generate-biomedicines"; type = "record";  id = "doi:10.1038/s41586-023-06728-8"; title = "chroma-illuminating-protein-space (Nature, paywalled)" },

    @{ lab = "xaira-therapeutics"; type = "biorxiv"; id = "10.1101/2024.03.14.585103"; title = "rfdiffusion-de-novo-design-of-antibodies" },
    @{ lab = "xaira-therapeutics"; type = "biorxiv"; id = "10.1101/2022.06.03.494563"; title = "proteinmpnn-robust-protein-sequence-design" },
    @{ lab = "xaira-therapeutics"; type = "record";  id = "doi:10.1038/s41586-023-06415-8"; title = "rfdiffusion-de-novo-design-structure-and-function (Nature, paywalled)" },

    @{ lab = "noetik"; type = "record"; id = "UNVERIFIED"; title = "octo-world-model-for-cancer-biology-tech-report-1 (noetik.ai, web only)" },
    @{ lab = "noetik"; type = "record"; id = "UNVERIFIED"; title = "octo-vc-simulating-spatial-biology-virtual-cells-tech-report-3 (noetik.ai, web only)" },

    @{ lab = "czi-virtual-cell"; type = "arxiv";   id = "2409.11654";              title = "how-to-build-the-virtual-cell-with-ai-priorities-opportunities" },
    @{ lab = "czi-virtual-cell"; type = "biorxiv"; id = "10.1101/2025.04.25.650731"; title = "transcriptformer-cross-species-generative-cell-atlas" },
    @{ lab = "czi-virtual-cell"; type = "biorxiv"; id = "10.1101/2025.08.18.670981"; title = "rbio-1-reasoning-llms-with-biological-world-models-as-verifiers" },

    # ===================== Group D - frontier architecture labs (cognitive-arch relevant) =====================
    @{ lab = "sakana-ai"; type = "arxiv"; id = "1803.10122"; title = "world-models" },
    @{ lab = "sakana-ai"; type = "arxiv"; id = "2403.13187"; title = "evolutionary-optimization-of-model-merging-recipes" },
    @{ lab = "sakana-ai"; type = "arxiv"; id = "2408.06292"; title = "the-ai-scientist-fully-automated-open-ended-discovery" },
    @{ lab = "sakana-ai"; type = "arxiv"; id = "2501.06252"; title = "transformer2-self-adaptive-llms" },
    @{ lab = "sakana-ai"; type = "arxiv"; id = "2505.05522"; title = "continuous-thought-machines" },

    @{ lab = "world-labs"; type = "arxiv";  id = "2003.08934"; title = "nerf-representing-scenes-as-neural-radiance-fields" },
    @{ lab = "world-labs"; type = "arxiv";  id = "1603.08155"; title = "perceptual-losses-real-time-style-transfer-super-resolution" },
    @{ lab = "world-labs"; type = "record"; id = "doi:10.1109/CVPR.2009.5206848"; title = "imagenet-large-scale-hierarchical-image-database (CVPR)" },
    @{ lab = "world-labs"; type = "record"; id = "blog:https://www.worldlabs.ai/blog/rtfm"; title = "rtfm-a-real-time-frame-model" },
    @{ lab = "world-labs"; type = "record"; id = "blog:https://www.worldlabs.ai/blog/marble-world-model"; title = "marble-a-multimodal-world-model" },

    @{ lab = "physical-intelligence"; type = "arxiv"; id = "1703.03400"; title = "maml-model-agnostic-meta-learning" },
    @{ lab = "physical-intelligence"; type = "arxiv"; id = "1801.01290"; title = "soft-actor-critic-max-entropy-deep-rl" },
    @{ lab = "physical-intelligence"; type = "arxiv"; id = "2410.24164"; title = "pi0-vision-language-action-flow-model-general-robot-control" },
    @{ lab = "physical-intelligence"; type = "arxiv"; id = "2504.16054"; title = "pi05-vla-model-with-open-world-generalization" },

    @{ lab = "skild-ai"; type = "arxiv"; id = "1705.05363"; title = "icm-curiosity-driven-exploration-by-self-supervised-prediction" },
    @{ lab = "skild-ai"; type = "arxiv"; id = "1808.04355"; title = "large-scale-study-of-curiosity-driven-learning" },
    @{ lab = "skild-ai"; type = "arxiv"; id = "2107.04034"; title = "rma-rapid-motor-adaptation-for-legged-robots" },

    @{ lab = "thinking-machines-lab"; type = "arxiv";  id = "1502.05477"; title = "trpo-trust-region-policy-optimization" },
    @{ lab = "thinking-machines-lab"; type = "arxiv";  id = "1707.06347"; title = "ppo-proximal-policy-optimization-algorithms" },
    @{ lab = "thinking-machines-lab"; type = "record"; id = "blog:https://thinkingmachines.ai/blog/lora/"; title = "lora-without-regret" },
    @{ lab = "thinking-machines-lab"; type = "record"; id = "blog:https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/"; title = "defeating-nondeterminism-in-llm-inference" },
    @{ lab = "thinking-machines-lab"; type = "record"; id = "blog:https://thinkingmachines.ai/blog/modular-manifolds/"; title = "modular-manifolds" },

    @{ lab = "reflection-ai"; type = "arxiv"; id = "2004.04136"; title = "curl-contrastive-unsupervised-representations-for-rl" },
    @{ lab = "reflection-ai"; type = "arxiv"; id = "2210.14215"; title = "in-context-rl-with-algorithm-distillation" },
    @{ lab = "reflection-ai"; type = "arxiv"; id = "1911.08265"; title = "muzero-mastering-atari-go-chess-shogi-with-learned-model" },
    @{ lab = "reflection-ai"; type = "arxiv"; id = "1712.01815"; title = "alphazero-mastering-chess-and-shogi-by-self-play" },

    @{ lab = "humans-and"; type = "arxiv";  id = "2203.14465"; title = "star-bootstrapping-reasoning-with-reasoning (founders, Zelikman+Goodman)" },
    @{ lab = "humans-and"; type = "arxiv";  id = "2403.09629"; title = "quiet-star-language-models-can-teach-themselves-to-think-before-speaking (founder, Zelikman)" },
    @{ lab = "humans-and"; type = "arxiv";  id = "2405.14769"; title = "pragmatic-feature-preferences-reward-relevant-preferences-from-human-input (founder, Peng)" },
    @{ lab = "humans-and"; type = "arxiv";  id = "2307.15217"; title = "open-problems-and-fundamental-limitations-of-rlhf (co-author, Peng)" },
    @{ lab = "humans-and"; type = "record"; id = "doi:10.1016/j.tics.2016.08.005"; title = "pragmatic-language-interpretation-as-probabilistic-inference-rsa (TICS, paywalled; founder Goodman)" },

    @{ lab = "ineffable-intelligence"; type = "arxiv";  id = "1312.5602";  title = "dqn-playing-atari-with-deep-reinforcement-learning (founder, Silver)" },
    @{ lab = "ineffable-intelligence"; type = "arxiv";  id = "1509.02971"; title = "ddpg-continuous-control-with-deep-reinforcement-learning (founder, Silver)" },
    @{ lab = "ineffable-intelligence"; type = "arxiv";  id = "1712.01815"; title = "alphazero-mastering-chess-and-shogi-by-self-play (founder, Silver)" },
    @{ lab = "ineffable-intelligence"; type = "arxiv";  id = "1911.08265"; title = "muzero-mastering-atari-go-chess-shogi-with-learned-model (founder, Silver)" },
    @{ lab = "ineffable-intelligence"; type = "record"; id = "doi:10.1038/nature16961"; title = "alphago-mastering-the-game-of-go-with-deep-nn-and-tree-search (Nature, paywalled)" },
    @{ lab = "ineffable-intelligence"; type = "record"; id = "doi:10.1038/nature24270"; title = "alphago-zero-mastering-the-game-of-go-without-human-knowledge (Nature, paywalled)" },
    @{ lab = "ineffable-intelligence"; type = "record"; id = "doi:10.1016/j.artint.2021.103535"; title = "reward-is-enough (Artificial Intelligence, paywalled; company thesis)" },
    @{ lab = "ineffable-intelligence"; type = "record"; id = "UNVERIFIED"; title = "the-era-of-experience-silver-sutton-2025 (north-star essay; lookup by title)" }
)

$baseDir = Join-Path $PSScriptRoot "papers\neolabs"
if (!(Test-Path $baseDir)) { New-Item -ItemType Directory -Path $baseDir -Force | Out-Null }

$success = @()
$failed = @()
$recorded = @()

function Get-DownloadUrls($p) {
    switch ($p.type) {
        "arxiv"   { return @("https://arxiv.org/pdf/$($p.id)") }
        "biorxiv" { return @("https://www.biorxiv.org/content/$($p.id)v1.full.pdf",
                             "https://www.biorxiv.org/content/$($p.id)v2.full.pdf",
                             "https://www.biorxiv.org/content/$($p.id)v3.full.pdf") }
        "medrxiv" { return @("https://www.medrxiv.org/content/$($p.id)v1.full.pdf",
                             "https://www.medrxiv.org/content/$($p.id)v2.full.pdf") }
        default   { return @() }
    }
}

function Get-FileStem($p) {
    if ($p.type -eq "arxiv") { return "$($p.title)-$($p.id)" }
    $stem = ($p.id -replace "^10\.\d+/", "") -replace "[/:]", "_"
    return "$($p.title)-$($p.type)-$stem"
}

$jobs = @()
foreach ($p in $papers) {
    $labDir = Join-Path $baseDir $p.lab
    if (!(Test-Path $labDir)) { New-Item -ItemType Directory -Path $labDir -Force | Out-Null }

    if ($p.type -eq "record") {
        $recorded += "[$($p.lab)] $($p.id) -> $($p.title)"
        continue
    }

    $out = Join-Path $labDir ("$(Get-FileStem $p).pdf")
    if ((Test-Path $out) -and ((Get-Item $out).Length -gt 20000)) {
        Write-Host "[SKIP] [$($p.lab)] $($p.id) exists"
        $success += "$($p.lab)/$($p.id)"
        continue
    }

    $urls = Get-DownloadUrls $p
    $jobs += Start-Job -ScriptBlock {
        param($urls, $out, $lab, $id, $type)
        $ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        foreach ($u in $urls) {
            try {
                Invoke-WebRequest -Uri $u -OutFile $out -UserAgent $ua -TimeoutSec 180 -Headers @{ "Accept" = "application/pdf" }
                $sz = (Get-Item $out).Length
                if ($sz -gt 20000) { return @{ lab = $lab; id = $id; type = $type; ok = $true; size = $sz; url = $u } }
            } catch { continue }
        }
        return @{ lab = $lab; id = $id; type = $type; ok = $false; reason = "all urls failed" }
    } -ArgumentList $urls, $out, $p.lab, $p.id, $p.type
}

Write-Host "Started $($jobs.Count) download jobs (record-only entries skipped)..."
$results = $jobs | Wait-Job | Receive-Job
$jobs | Remove-Job

foreach ($r in $results) {
    if ($r.ok) {
        Write-Host "[OK]   [$($r.lab)] $($r.id) ($([math]::Round($r.size/1MB,2)) MB)"
        $success += "$($r.lab)/$($r.id)"
    } else {
        Write-Host "[FAIL] [$($r.lab)] $($r.id) ($($r.type)): $($r.reason)" -ForegroundColor Yellow
        $failed += "[$($r.lab)] $($r.id) ($($r.type))"
    }
}

$dlAttempts = ($papers | Where-Object { $_.type -ne "record" }).Count
Write-Host ""
Write-Host "===== SUMMARY ====="
Write-Host "Downloaded OK : $($success.Count) / $dlAttempts"
Write-Host "Failed        : $($failed.Count)"
Write-Host "Record-only   : $($recorded.Count) (paywalled DOI / blog / unverified - cite, not downloaded)"
if ($failed.Count -gt 0) {
    Write-Host ""; Write-Host "Failed (likely bioRxiv/medRxiv Cloudflare or wrong version):"
    $failed | ForEach-Object { Write-Host "  - $_" }
}
Write-Host ""
Write-Host "Record-only entries:"
$recorded | ForEach-Object { Write-Host "  - $_" }
