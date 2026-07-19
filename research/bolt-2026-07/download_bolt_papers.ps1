# Download BOLT-related papers to research/bolt-2026-07/papers/<cluster>/
# Scope: verified core papers for BOLT author lineage and adjacent Bayesian online-learning transformer work.

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

$papers = @(
    # Ross M. Clarke
    @{ cluster = "clarke"; title = "scalable-one-pass-weight-update-hyperparams"; url = "https://arxiv.org/pdf/2110.10461" },
    @{ cluster = "clarke"; title = "adam-through-second-order-lens-icml2024"; url = "https://raw.githubusercontent.com/mlresearch/v235/main/assets/clarke24a/clarke24a.pdf" },
    @{ cluster = "clarke"; title = "studying-kfac-heuristics-adam-second-order-lens"; url = "https://arxiv.org/pdf/2310.14963" },
    @{ cluster = "clarke"; title = "series-hessian-vector-products-saddle-free-newton"; url = "https://arxiv.org/pdf/2310.14901" },

    # Jose Miguel Hernandez-Lobato core subset
    @{ cluster = "jmhl"; title = "probabilistic-backpropagation-bayesian-neural-networks"; url = "https://arxiv.org/pdf/1502.05336" },
    @{ cluster = "jmhl"; title = "black-box-alpha-divergence-minimization"; url = "https://arxiv.org/pdf/1511.03243" },
    @{ cluster = "jmhl"; title = "deep-gaussian-processes-approximate-ep"; url = "https://arxiv.org/pdf/1602.04133" },
    @{ cluster = "jmhl"; title = "sequence-tutor-kl-control"; url = "https://arxiv.org/pdf/1611.02796" },
    @{ cluster = "jmhl"; title = "decomposition-uncertainty-bayesian-deep-learning"; url = "https://arxiv.org/pdf/1710.07283" },
    @{ cluster = "jmhl"; title = "meta-learning-stochastic-gradient-mcmc"; url = "https://arxiv.org/pdf/1806.04522" },
    @{ cluster = "jmhl"; title = "variational-implicit-processes"; url = "https://arxiv.org/pdf/1806.02390" },
    @{ cluster = "jmhl"; title = "bayesian-deep-learning-subnetwork-inference"; url = "https://arxiv.org/pdf/2010.14689" },
    @{ cluster = "jmhl"; title = "depth-uncertainty-neural-networks"; url = "https://arxiv.org/pdf/2006.08437" },
    @{ cluster = "jmhl"; title = "functional-variational-inference-spg"; url = "https://proceedings.neurips.cc/paper_files/paper/2021/file/b613e70fd9f59310cf0a8d33de3f2800-Paper.pdf" },
    @{ cluster = "jmhl"; title = "improving-continual-learning-gradient-reconstructions"; url = "https://openreview.net/pdf?id=b1fpfCjja1" },
    @{ cluster = "jmhl"; title = "sampling-based-inference-large-linear-models-linearised-laplace"; url = "https://arxiv.org/pdf/2210.04994" },
    @{ cluster = "jmhl"; title = "sampling-variational-posterior-local-refinement"; url = "https://arxiv.org/pdf/2110.11268" },
    @{ cluster = "jmhl"; title = "successor-uncertainties-temporal-difference-learning"; url = "https://arxiv.org/pdf/1810.06530" },
    @{ cluster = "jmhl"; title = "edddi-efficient-dynamic-discovery-partial-vae"; url = "https://arxiv.org/pdf/1903.09226" },
    @{ cluster = "jmhl"; title = "icebreaker-efficient-information-acquisition"; url = "https://arxiv.org/pdf/1902.04534" },

    # Boltzbit / Yichuan Zhang / Jinli Hu verified relevant papers
    @{ cluster = "boltzbit"; title = "quasi-newton-methods-mcmc"; url = "https://proceedings.neurips.cc/paper_files/paper/2011/file/e702e51da2c0f5be4dd354bb3e295d37-Paper.pdf" },
    @{ cluster = "boltzbit"; title = "continuous-relaxations-discrete-hmc"; url = "https://proceedings.neurips.cc/paper_files/paper/2012/file/c913303f392ffc643f7240b180602652-Paper.pdf" },
    @{ cluster = "boltzbit"; title = "semi-separable-hmc-bayesian-hierarchical-models"; url = "https://arxiv.org/pdf/1406.3843" },
    @{ cluster = "boltzbit"; title = "ergodic-inference-accelerate-convergence"; url = "https://arxiv.org/pdf/1805.10377" },
    @{ cluster = "boltzbit"; title = "ergodic-measure-preserving-flows"; url = "https://openreview.net/pdf?id=HJx4KjRqYQ" },
    @{ cluster = "boltzbit"; title = "theory-algorithm-ergodic-inference"; url = "https://arxiv.org/pdf/1811.07192" },
    @{ cluster = "boltzbit"; title = "hmc-hyperparameter-optimization-gradient-strategy"; url = "https://proceedings.mlr.press/v139/campbell21a/campbell21a.pdf" },
    @{ cluster = "boltzbit"; title = "multi-period-trading-prediction-markets"; url = "https://arxiv.org/pdf/1403.0648" },
    @{ cluster = "boltzbit"; title = "combinatorial-modelling-learning-prediction-markets"; url = "https://arxiv.org/pdf/1201.3851" },

    # Adjacent public literature
    @{ cluster = "adjacent"; title = "distribution-transformers-prior-adaptation"; url = "https://arxiv.org/pdf/2502.02463" },
    @{ cluster = "adjacent"; title = "transformers-can-do-bayesian-inference-pfn"; url = "https://arxiv.org/pdf/2112.10510" },
    @{ cluster = "adjacent"; title = "full-bayesian-inference-in-context"; url = "https://arxiv.org/pdf/2501.16825" },
    @{ cluster = "adjacent"; title = "memory-based-meta-learning-nonstationary"; url = "https://arxiv.org/pdf/2302.03067" },
    @{ cluster = "adjacent"; title = "continuous-latent-contexts-online-learning-transformers"; url = "https://arxiv.org/pdf/2605.09867" },
    @{ cluster = "adjacent"; title = "palimpsa-remember-learn-forget-attention"; url = "https://arxiv.org/pdf/2602.09075" }
)

$baseDir = Join-Path $PSScriptRoot "papers"
$success = New-Object System.Collections.ArrayList
$failed = New-Object System.Collections.ArrayList

Add-Type -AssemblyName System.Net.Http

foreach ($cluster in @("clarke", "jmhl", "boltzbit", "adjacent")) {
    $dir = Join-Path $baseDir $cluster
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
}

$total = $papers.Count
$index = 0

foreach ($paper in $papers) {
    $index += 1
    $ext = ".pdf"
    $out = Join-Path (Join-Path $baseDir $paper.cluster) "$($paper.title)$ext"

    if (Test-Path $out) {
        $size = (Get-Item $out).Length
        if ($size -gt 10000) {
            Write-Host "[SKIP $index/$total] [$($paper.cluster)] $($paper.title) ($size bytes)"
            [void]$success.Add("$($paper.cluster)/$($paper.title)")
            continue
        }
    }

    $ok = $false
    for ($attempt = 1; $attempt -le 2 -and -not $ok; $attempt += 1) {
        try {
            $client = New-Object System.Net.Http.HttpClient
            $client.Timeout = [TimeSpan]::FromSeconds(90)
            $client.DefaultRequestHeaders.UserAgent.ParseAdd("Mozilla/5.0 (VolvenceZero BOLT research)")
            $bytes = $client.GetByteArrayAsync($paper.url).GetAwaiter().GetResult()
            [System.IO.File]::WriteAllBytes($out, $bytes)
            $client.Dispose()

            $size = (Get-Item $out).Length
            if ($size -lt 10000) {
                throw "download too small ($size bytes)"
            }

            Write-Host "[OK   $index/$total] [$($paper.cluster)] $($paper.title) ($size bytes)"
            [void]$success.Add("$($paper.cluster)/$($paper.title)")
            $ok = $true
        } catch {
            if ($client) {
                $client.Dispose()
            }
            $message = $_.Exception.Message
            if ($attempt -lt 2) {
                Write-Host "[RETRY $index/$total] [$($paper.cluster)] $($paper.title): $message" -ForegroundColor DarkYellow
                Start-Sleep -Seconds 3
            } else {
                Write-Host "[FAIL $index/$total] [$($paper.cluster)] $($paper.title): $message" -ForegroundColor Yellow
                [void]$failed.Add("$($paper.cluster)/$($paper.title): $($paper.url) -- $message")
            }
        }
    }

    Start-Sleep -Milliseconds 600
}

$summaryPath = Join-Path $PSScriptRoot "_download_summary.md"
$lines = @(
    "# BOLT Paper Download Summary",
    "",
    "Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
    "",
    "## Success ($($success.Count))",
    ""
)
foreach ($item in $success) {
    $lines += "- $item"
}
$lines += ""
$lines += "## Failed ($($failed.Count))"
$lines += ""
foreach ($item in $failed) {
    $lines += "- $item"
}
$lines | Out-File -FilePath $summaryPath -Encoding UTF8

Write-Host ""
Write-Host "===== SUMMARY ====="
Write-Host "Total: $total"
Write-Host "Success: $($success.Count)"
Write-Host "Failed: $($failed.Count)"
Write-Host "Wrote summary to $summaryPath"
