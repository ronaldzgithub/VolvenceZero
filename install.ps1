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
    "packages\lifeform-domain-emogpt",
    "packages\lifeform-domain-coding",
    "packages\lifeform-service",
    "packages\lifeform-evolution"
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
