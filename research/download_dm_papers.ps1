# Download DeepMind + key alumni cognitive AGI papers to research/papers/dm/
# Format: <short-title>-<arxiv_id>.pdf
# Scope: complement of arxiv-survey-2026-05.md and core-author-paper-assessment-2026-05.md
# Focus axes: R3/R4 (temporal abstraction + internal control), R-PE (prediction error), R10 (credit + self-modification)

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

$papers = @(
    # ===== Tier A: high alignment with R3/R4/R-PE/R10 (10 papers, must read) =====
    @{ id = "2509.24527"; title = "dreamer4-training-agents-inside-scalable-world-models"; tier = "A" },
    @{ id = "2506.13131"; title = "alphaevolve-coding-agent-scientific-engineering-discovery"; tier = "A" },
    @{ id = "2505.00787"; title = "option-keyboard-controllable-world-models-small"; tier = "A" },
    @{ id = "2304.06729"; title = "meta-learned-models-of-cognition"; tier = "A" },
    @{ id = "2507.16598"; title = "depression-as-disorder-of-distributional-coding"; tier = "A" },
    @{ id = "2311.02462"; title = "levels-of-agi-operationalizing-progress"; tier = "A" },
    @{ id = "2505.17895"; title = "datarater-meta-learned-dataset-curation"; tier = "A" },
    @{ id = "2506.14045"; title = "discovering-temporal-structure-hrl-overview"; tier = "A" },
    @{ id = "2201.02628"; title = "attention-option-critic"; tier = "A" },
    @{ id = "2502.18864"; title = "towards-an-ai-co-scientist"; tier = "A" },

    # ===== Tier B: medium alignment, brief evaluation (8 arxiv + 2 Nature-only) =====
    # NOTE: AlphaDev (Mankowitz 2023 Nature) and AlphaTensor (Fawzi 2022 Nature) have no arxiv preprint.
    #       They are referenced by DOI only in the assessment report.
    @{ id = "2402.15391"; title = "genie-generative-interactive-environments"; tier = "B" },
    @{ id = "2512.04797"; title = "sima2-generalist-embodied-agent-virtual-worlds"; tier = "B" },
    @{ id = "2107.12808"; title = "open-ended-learning-generally-capable-agents"; tier = "B" },
    @{ id = "2507.06261"; title = "gemini25-pushing-the-frontier-tech-report"; tier = "B" },
    @{ id = "2411.19744"; title = "amplifying-human-combinatorial-competitive-programming"; tier = "B" },
    @{ id = "2404.07839"; title = "recurrentgemma-moving-past-transformers-efficient"; tier = "B" },
    @{ id = "2001.00271"; title = "options-of-interest-temporal-abstraction-interest-functions"; tier = "B" },
    @{ id = "2109.00157"; title = "survey-of-exploration-methods-rl"; tier = "B" },

    # ===== Tier C: background / application reference, register only (3 arxiv papers) =====
    # Med-PaLM (subagent gave 2312.13120 -> actually NMR chemistry paper, real Med-PaLM ID
    #          is split across 2212.13138/2305.09617; not re-downloaded, see report).
    # Botvinick 2018 prefrontal meta-RL (Nature Neurosci) and Lake et al. 2017 BBS are
    # referenced by DOI only; no arXiv preprint.
    @{ id = "2306.11706"; title = "robocat-self-improving-generalist-agent-robotic-manipulation"; tier = "C" },
    @{ id = "2212.12794"; title = "graphcast-medium-range-global-weather-forecasting"; tier = "C" },
    @{ id = "2401.05654"; title = "amie-towards-conversational-diagnostic-ai"; tier = "C" }
)

$outDir = Join-Path $PSScriptRoot "papers\dm"
if (!(Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir -Force | Out-Null }

$success = @()
$failed = @()

# Parallelize downloads via Start-Job (PowerShell 5.1+ compatible)
$jobs = @()
foreach ($p in $papers) {
    $url = "https://arxiv.org/pdf/$($p.id)"
    $out = Join-Path $outDir "$($p.title)-$($p.id).pdf"

    if (Test-Path $out) {
        $sz = (Get-Item $out).Length
        if ($sz -gt 10000) {
            Write-Host "[SKIP] [$($p.tier)] $($p.id) already exists ($sz bytes)"
            $success += $p.id
            continue
        }
    }

    $jobs += Start-Job -ScriptBlock {
        param($url, $out, $id, $tier)
        try {
            Invoke-WebRequest -Uri $url -OutFile $out -UserAgent "Mozilla/5.0 (research-survey)" -TimeoutSec 180
            $sz = (Get-Item $out).Length
            if ($sz -lt 10000) {
                return @{ id = $id; tier = $tier; ok = $false; reason = "too small ($sz bytes)" }
            }
            return @{ id = $id; tier = $tier; ok = $true; size = $sz }
        } catch {
            return @{ id = $id; tier = $tier; ok = $false; reason = $_.Exception.Message }
        }
    } -ArgumentList $url, $out, $p.id, $p.tier
}

Write-Host "Started $($jobs.Count) download jobs..."
$results = $jobs | Wait-Job | Receive-Job
$jobs | Remove-Job

foreach ($r in $results) {
    if ($r.ok) {
        Write-Host "[OK]   [$($r.tier)] $($r.id) ($($r.size) bytes)"
        $success += $r.id
    } else {
        Write-Host "[FAIL] [$($r.tier)] $($r.id): $($r.reason)" -ForegroundColor Yellow
        $failed += "[$($r.tier)] $($r.id): $($r.reason)"
    }
}

Write-Host ""
Write-Host "===== SUMMARY ====="
Write-Host "Success: $($success.Count) / $($papers.Count)"
Write-Host "Failed: $($failed.Count)"
if ($failed.Count -gt 0) {
    Write-Host ""
    Write-Host "Failed papers:"
    $failed | ForEach-Object { Write-Host "  - $_" }
}
