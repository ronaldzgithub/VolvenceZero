# Download all surveyed arxiv papers to research/papers/
# Format: <title>-<arxiv_id>.pdf

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

$papers = @(
    # Tier 1 — Core
    @{ id = "2512.24695"; title = "nested-learning-illusion-of-deep-architectures" },
    @{ id = "2512.20605"; title = "emergent-temporal-abstractions-hierarchical-rl" },
    @{ id = "2510.04399"; title = "two-gate-guardrail-self-modifying-agents" },
    @{ id = "2604.18701"; title = "curiosity-critic-cumulative-prediction-error" },
    @{ id = "2512.18202"; title = "sophia-system3-persistent-agent" },

    # R2/R4 — Frozen substrate + controller
    @{ id = "2503.21383"; title = "cola-controlling-llms-with-latent-actions" },
    @{ id = "2509.24238"; title = "fr-ponder-adaptive-reasoning-latent-space" },
    @{ id = "2512.11816"; title = "rl-for-latent-space-thinking-llms" },
    @{ id = "2507.08799"; title = "kv-cache-steering-frozen-llms" },

    # R-PE — Prediction error
    @{ id = "2506.06725"; title = "worldllm-curiosity-driven-theory-making" },
    @{ id = "2505.13934"; title = "rlvr-world-training-world-models-rl" },
    @{ id = "2506.00138"; title = "3m-progress-zebrafish-intrinsic-motivation" },

    # R3/R10 — Temporal abstraction / options
    @{ id = "2508.17751"; title = "mango-multi-layer-abstraction-nested-options" },
    @{ id = "2505.12737"; title = "ota-option-aware-temporally-abstracted-value" },
    @{ id = "2507.16473"; title = "variational-homomorphisms-option-induced-mdps" },
    @{ id = "2510.24988"; title = "change-point-detection-option-critic" },
    @{ id = "2503.19007"; title = "ldsc-llm-guided-semantic-hrl" },

    # R5/R6 — Memory continuum
    @{ id = "2502.14802"; title = "hipporag2-from-rag-to-memory" },
    @{ id = "2502.12110"; title = "a-mem-agentic-memory-llm-agents" },
    @{ id = "2508.19828"; title = "memory-r1-rl-memory-management" },
    @{ id = "2601.02744"; title = "synapse-spreading-activation-memory" },
    @{ id = "2509.16189"; title = "latent-learning-episodic-memory-complements" },
    @{ id = "2512.22716"; title = "memento2-stateful-reflective-memory" },

    # R7/R14 — Dual track + persistent regime
    @{ id = "2502.07591"; title = "dmwm-dual-mind-world-model" },
    @{ id = "2509.25299"; title = "id-rag-identity-retrieval-augmented-generation" },
    @{ id = "2512.07092"; title = "geometry-of-persona-soul-engine" },
    @{ id = "2502.05907"; title = "evoagent-self-evolving-continual-world-model" },

    # R8/R9/R10/R15 — Self-modification gates
    @{ id = "2510.10232"; title = "statistical-godel-machine-self-modification" },
    @{ id = "2505.22954"; title = "darwin-godel-machine-self-improving-agents" },

    # R11 — Named internal states / ToM
    @{ id = "2505.14685"; title = "lookback-mechanism-belief-tracking-llm" },
    @{ id = "2502.11881"; title = "thoughttracing-hypothesis-driven-tom" },
    @{ id = "2502.14171"; title = "tom-aligned-conversational-agents-bdi" },
    @{ id = "2512.12716"; title = "coda-context-decoupled-hierarchical-agent" },

    # R12 — Evaluation beyond task
    @{ id = "2503.16416"; title = "survey-evaluation-llm-based-agents" },
    @{ id = "2509.17158"; title = "gaia2-are-platform-async-eval" },
    @{ id = "2601.11044"; title = "agencybench-core-agentic-capabilities" },
    @{ id = "2503.03056"; title = "a2perf-ood-efficiency-data-cost" },

    # EQ / relationship
    @{ id = "2506.03543"; title = "cognipair-gnwt-multi-agent-digital-twins" },
    @{ id = "2506.16756"; title = "socialsim-emotional-support-conversation" },
    @{ id = "2508.09521"; title = "compeer-empathetic-rl-emotional-support" },
    @{ id = "2508.12935"; title = "rlff-esc-future-oriented-rewards" },
    @{ id = "2510.07925"; title = "personalized-long-term-llm-interactions" },
    @{ id = "2512.06688"; title = "personamem-v2-implicit-personas-agentic-memory" },

    # Other infrastructure
    @{ id = "2401.08623"; title = "wake-sleep-consolidated-learning" },
    @{ id = "2504.14727"; title = "semi-parametric-memory-consolidation-brain-like" },
    @{ id = "2507.02901"; title = "seslr-sleep-enhanced-latent-replay" },
    @{ id = "2602.23681"; title = "odar-active-inference-routing" },
    @{ id = "2412.10425"; title = "active-inference-self-organizing-multi-llm" },
    @{ id = "2410.09918"; title = "dualformer-controllable-fast-slow-thinking" },
    @{ id = "2510.04618"; title = "ace-agentic-context-engineering" }
)

$outDir = Join-Path $PSScriptRoot "papers"
if (!(Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

$success = @()
$failed = @()

# Use jobs for parallelism (PowerShell 5.1+ compatible via Start-Job)
$jobs = @()
foreach ($p in $papers) {
    $url = "https://arxiv.org/pdf/$($p.id)"
    $out = Join-Path $outDir "$($p.title)-$($p.id).pdf"

    if (Test-Path $out) {
        $sz = (Get-Item $out).Length
        if ($sz -gt 10000) {
            Write-Host "[SKIP] $($p.id) already exists ($sz bytes)"
            $success += $p.id
            continue
        }
    }

    $jobs += Start-Job -ScriptBlock {
        param($url, $out, $id)
        try {
            Invoke-WebRequest -Uri $url -OutFile $out -UserAgent "Mozilla/5.0 (research-survey)" -TimeoutSec 120
            $sz = (Get-Item $out).Length
            if ($sz -lt 10000) {
                return @{ id = $id; ok = $false; reason = "too small ($sz bytes)" }
            }
            return @{ id = $id; ok = $true; size = $sz }
        } catch {
            return @{ id = $id; ok = $false; reason = $_.Exception.Message }
        }
    } -ArgumentList $url, $out, $p.id
}

Write-Host "Started $($jobs.Count) download jobs..."
$results = $jobs | Wait-Job | Receive-Job
$jobs | Remove-Job

foreach ($r in $results) {
    if ($r.ok) {
        Write-Host "[OK]   $($r.id) ($($r.size) bytes)"
        $success += $r.id
    } else {
        Write-Host "[FAIL] $($r.id): $($r.reason)" -ForegroundColor Yellow
        $failed += "$($r.id): $($r.reason)"
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
