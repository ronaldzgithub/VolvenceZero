# Download cognitive AGI Probe 100 papers to research/probe/papers/<axis>/
# Format: <axis>/<short-title>-<arxiv_id>.pdf
# Companion: research/probe/_candidates.md (final 100 master list)
# Reuses research/papers/ for already-downloaded PDFs (skips download)
# Spec: research/probe/01_method_and_scoring.md
#
# Strategy: Use System.Net.Http.HttpClient with concurrency limit (faster than Start-Job).
# Each download has 60s timeout. Failed downloads recorded for retry.

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

# ===== 47 N_new + 1 H_html papers (52 Y_existing skipped — see _candidates.md §3) =====
$papers = @(
    # A1 — Reasoning & Test-Time Compute (10)
    @{ id = "2502.05171"; title = "recurrent-depth-test-time-latent-reasoning"; axis = "A1" },
    @{ id = "2412.06769"; title = "coconut-continuous-latent-space-reasoning"; axis = "A1" },
    @{ id = "2403.09629"; title = "quiet-star-think-before-speaking"; axis = "A1" },
    @{ id = "2305.20050"; title = "lets-verify-step-by-step"; axis = "A1" },
    @{ id = "2409.12917"; title = "score-self-correct-via-rl"; axis = "A1" },
    @{ id = "2501.12948"; title = "deepseek-r1-incentivizing-reasoning-rl"; axis = "A1" },
    @{ id = "2312.08935"; title = "math-shepherd-step-verifier-no-human"; axis = "A1" },
    @{ id = "2408.03314"; title = "snell-test-time-compute-optimal"; axis = "A1" },
    @{ id = "2501.19393"; title = "s1-simple-test-time-scaling"; axis = "A1" },
    @{ id = "2203.14465"; title = "star-self-taught-reasoner"; axis = "A1" },

    # A2 (7)
    @{ id = "2506.09985"; title = "v-jepa-2-self-supervised-video-models"; axis = "A2" },
    @{ id = "1911.08265"; title = "muzero-mastering-with-learned-model"; axis = "A2" },
    @{ id = "2405.12399"; title = "diamond-diffusion-world-modeling-atari"; axis = "A2" },
    @{ id = "2403.00564"; title = "efficientzero-v2-discrete-continuous-control"; axis = "A2" },
    @{ id = "2503.18938"; title = "adaworld-adaptable-world-models-latent-actions"; axis = "A2" },
    @{ id = "2410.24164"; title = "pi-zero-vla-flow-model-robot-control"; axis = "A2" },
    @{ id = "2301.08243"; title = "i-jepa-joint-embedding-predictive-architecture"; axis = "A2" },

    # A3 (4)
    @{ id = "2601.09913"; title = "cma-continuum-memory-architectures"; axis = "A3" },
    @{ id = "2312.00752"; title = "mamba-selective-state-space"; axis = "A3" },
    @{ id = "1612.00796"; title = "ewc-overcoming-catastrophic-forgetting"; axis = "A3" },
    @{ id = "1410.3916"; title = "memory-networks-weston-chopra-bordes"; axis = "A3" },

    # A5 (2)
    @{ id = "2407.04620"; title = "ttt-rnns-expressive-hidden-states"; axis = "A5" },
    @{ id = "2210.14215"; title = "algorithm-distillation-in-context-rl"; axis = "A5" },

    # B1 (4)
    @{ id = "2107.12979"; title = "predictive-coding-theoretical-experimental-review"; axis = "B1" },
    @{ id = "2006.04182"; title = "pcn-approximates-backprop-arbitrary-graphs"; axis = "B1" },
    @{ id = "1705.05363"; title = "icm-curiosity-self-supervised-prediction"; axis = "B1" },
    @{ id = "1810.12894"; title = "rnd-random-network-distillation"; axis = "B1" },

    # B2 (2)
    @{ id = "2210.05492"; title = "no-press-diplomacy-human-regularized-rl"; axis = "B2" },
    @{ id = "2306.15448"; title = "bigtom-social-reasoning-language-models"; axis = "B2" },

    # B3 (4)
    @{ id = "2502.03544"; title = "alphageometry-2-gold-medalist-olympiad"; axis = "B3" },
    @{ id = "2505.03335"; title = "absolute-zero-reasoner-self-play"; axis = "B3" },
    @{ id = "1901.01753"; title = "poet-paired-open-ended-trailblazer"; axis = "B3" },
    @{ id = "2012.02096"; title = "paired-emergent-complexity-ued"; axis = "B3" },

    # C1 (6)
    @{ id = "2401.10020"; title = "self-rewarding-language-models"; axis = "C1" },
    @{ id = "2212.08073"; title = "constitutional-ai-harmlessness-from-ai-feedback"; axis = "C1" },
    @{ id = "2511.18397"; title = "natural-emergent-misalignment-reward-hacking"; axis = "C1" },
    @{ id = "2401.05566"; title = "sleeper-agents-deceptive-llms-persist"; axis = "C1" },
    @{ id = "2412.14093"; title = "alignment-faking-in-large-language-models"; axis = "C1" },
    @{ id = "2312.09390"; title = "weak-to-strong-generalization"; axis = "C1" },

    # C2 (8)
    @{ id = "2403.19647"; title = "sparse-feature-circuits-marks-bau"; axis = "C2" },
    @{ id = "2408.05147"; title = "gemma-scope-jumprelu-saes"; axis = "C2" },
    @{ id = "2306.03341"; title = "iti-inference-time-intervention-truthful"; axis = "C2" },
    @{ id = "2310.01405"; title = "representation-engineering-top-down-transparency"; axis = "C2" },
    @{ id = "2310.15213"; title = "function-vectors-in-llms"; axis = "C2" },
    @{ id = "2406.11717"; title = "refusal-mediated-by-single-direction"; axis = "C2" },
    @{ id = "2507.21509"; title = "persona-vectors-monitoring-character-traits"; axis = "C2" },
    @{ id = "2211.00593"; title = "ioi-circuit-gpt2-small-interpretability-wild"; axis = "C2" }
)

# Create per-axis directories
$baseDir = Join-Path $PSScriptRoot "papers"
$axisDirs = @("A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "C1", "C2")
foreach ($axis in $axisDirs) {
    $d = Join-Path $baseDir $axis
    if (!(Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
}

Add-Type -AssemblyName System.Net.Http

$success = New-Object System.Collections.ArrayList
$failed = New-Object System.Collections.ArrayList

# Sequential download (avoids arxiv rate-limiting + low PowerShell job overhead)
# arxiv typically tolerates ~1 req/sec
$totalCount = $papers.Count
$idx = 0
foreach ($p in $papers) {
    $idx += 1
    $url = "https://arxiv.org/pdf/$($p.id)"
    $axisDir = Join-Path $baseDir $p.axis
    $out = Join-Path $axisDir "$($p.title)-$($p.id).pdf"

    if (Test-Path $out) {
        $sz = (Get-Item $out).Length
        if ($sz -gt 10000) {
            Write-Host "[SKIP $idx/$totalCount] [$($p.axis)] $($p.id) already exists ($sz bytes)"
            [void]$success.Add($p.id)
            continue
        }
    }

    $attempt = 0
    $maxAttempts = 2
    $ok = $false
    while ($attempt -lt $maxAttempts -and -not $ok) {
        $attempt += 1
        try {
            $client = New-Object System.Net.Http.HttpClient
            $client.Timeout = [TimeSpan]::FromSeconds(60)
            $client.DefaultRequestHeaders.Add("User-Agent", "Mozilla/5.0 (research-survey)")
            $bytes = $client.GetByteArrayAsync($url).GetAwaiter().GetResult()
            [System.IO.File]::WriteAllBytes($out, $bytes)
            $client.Dispose()
            $sz = (Get-Item $out).Length
            if ($sz -lt 10000) {
                throw "too small ($sz bytes)"
            }
            Write-Host "[OK   $idx/$totalCount] [$($p.axis)] $($p.id) ($sz bytes)"
            [void]$success.Add($p.id)
            $ok = $true
        } catch {
            $msg = $_.Exception.Message
            if ($attempt -lt $maxAttempts) {
                Write-Host "[RETRY $idx/$totalCount] [$($p.axis)] $($p.id): $msg" -ForegroundColor DarkYellow
                Start-Sleep -Seconds 3
            } else {
                Write-Host "[FAIL $idx/$totalCount] [$($p.axis)] $($p.id): $msg" -ForegroundColor Yellow
                [void]$failed.Add("[$($p.axis)] $($p.id): $msg")
            }
        }
    }
    Start-Sleep -Milliseconds 500  # arxiv courtesy delay
}

# H_html: Scaling Monosemanticity
$htmlOut = Join-Path $baseDir "C2\scaling-monosemanticity-anthropic-2024.html"
$htmlUrl = "https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html"
if (!(Test-Path $htmlOut)) {
    try {
        $client = New-Object System.Net.Http.HttpClient
        $client.Timeout = [TimeSpan]::FromSeconds(60)
        $client.DefaultRequestHeaders.Add("User-Agent", "Mozilla/5.0 (research-survey)")
        $bytes = $client.GetByteArrayAsync($htmlUrl).GetAwaiter().GetResult()
        [System.IO.File]::WriteAllBytes($htmlOut, $bytes)
        $client.Dispose()
        $sz = (Get-Item $htmlOut).Length
        if ($sz -gt 10000) {
            Write-Host "[OK]   [C2] H_html scaling-monosemanticity ($sz bytes HTML)"
            [void]$success.Add("scaling-monosemanticity-html")
        } else {
            Write-Host "[FAIL] [C2] H_html scaling-monosemanticity: too small ($sz bytes)" -ForegroundColor Yellow
            [void]$failed.Add("[C2] scaling-monosemanticity-html: too small")
        }
    } catch {
        Write-Host "[FAIL] [C2] H_html scaling-monosemanticity: $($_.Exception.Message)" -ForegroundColor Yellow
        [void]$failed.Add("[C2] scaling-monosemanticity-html: $($_.Exception.Message)")
    }
}

Write-Host ""
Write-Host "===== SUMMARY ====="
Write-Host "Total papers in script: $($papers.Count) (+ 1 H_html)"
Write-Host "Success: $($success.Count)"
Write-Host "Failed: $($failed.Count)"
if ($failed.Count -gt 0) {
    Write-Host ""
    Write-Host "Failed papers (record into research/probe/_candidates.md §6):"
    $failed | ForEach-Object { Write-Host "  - $_" }
}

# Write summary file
$summaryPath = Join-Path $PSScriptRoot "_download_summary.md"
$lines = @(
    "# Probe Download Summary",
    "",
    "Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
    "",
    "## Success ($($success.Count))",
    ""
)
foreach ($s in $success) { $lines += "- $s" }
$lines += ""
$lines += "## Failed ($($failed.Count))"
$lines += ""
foreach ($f in $failed) { $lines += "- $f" }
$lines | Out-File -FilePath $summaryPath -Encoding UTF8
Write-Host ""
Write-Host "Wrote summary to $summaryPath"
