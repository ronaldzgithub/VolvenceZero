[CmdletBinding()]
param(
    [string] $PythonBin = "python",
    [string] $Extras = ""
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RootDir

Write-Host "Installing Volvence Zero workspace into the current Python environment..."
Write-Host "Python: $(& $PythonBin -c 'import sys; print(sys.executable)')"

# Order matters: dependencies must be installed before dependents.
$Packages = @(
    "packages\vz-contracts",
    "packages\vz-substrate",
    "packages\vz-memory",
    "packages\vz-cognition",
    "packages\vz-application",
    "packages\vz-temporal",
    "packages\vz-runtime",
    "packages\lifeform-core",
    "packages\lifeform-thinking",
    "packages\lifeform-ingestion",
    "packages\lifeform-affordance",
    "packages\lifeform-expression",
    "packages\lifeform-domain-character",
    "packages\lifeform-domain-emogpt",
    "packages\lifeform-domain-coding",
    "packages\lifeform-domain-figure",
    "packages\lifeform-domain-growth-advisor",
    "packages\lifeform-service",
    "packages\lifeform-evolution",
    "packages\lifeform-openai-compat",
    "packages\lifeform-protocol-runtime",
    "packages\lifeform-mcp-bridge",
    "packages\dlaas-platform-contracts",
    "packages\dlaas-platform-registry",
    "packages\dlaas-platform-launcher",
    "packages\dlaas-platform-ops",
    "packages\dlaas-platform-eval",
    "packages\dlaas-platform-api"
)

foreach ($pkg in $Packages) {
    if (Test-Path $pkg) {
        Write-Host "==> pip install -e $pkg"
        & $PythonBin -m pip install -e $pkg --no-deps
        if ($LASTEXITCODE -ne 0) { throw "pip install failed for $pkg" }
    }
}

if ($Extras) {
    Write-Host "==> pip install vz-runtime[$Extras] (extras only)"
    & $PythonBin -m pip install "vz-runtime[$Extras]"
    if ($LASTEXITCODE -ne 0) { throw "pip install extras failed" }
}

Write-Host "Volvence Zero workspace installed successfully."
