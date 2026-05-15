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

# Pass 1: register every workspace sibling editably with --no-deps so the
# circular workspace dependency cluster (lifeform-openai-compat <->
# lifeform-service <-> lifeform-protocol-runtime) does not cause pip's
# resolver to try to fetch unpublished `==0.1.*` siblings from PyPI.
foreach ($pkg in $Packages) {
    if (Test-Path $pkg) {
        Write-Host "==> [pass 1] pip install -e $pkg --no-deps"
        & $PythonBin -m pip install -e $pkg --no-deps
        if ($LASTEXITCODE -ne 0) { throw "pip install (pass 1) failed for $pkg" }
    }
}

# Pass 2: re-run with full dep resolution. By now every workspace sibling
# is editable and satisfies its ==0.1.* constraint, so pip only fetches
# the *external* PyPI deps declared in each wheel's pyproject.toml
# (aiohttp / pypdf / beautifulsoup4 / mwparserfromhell / lxml / requests
# / PyYAML / ...). This keeps pyproject.toml as the single source of
# truth for runtime deps -- install.ps1 never has to mirror the list.
foreach ($pkg in $Packages) {
    if (Test-Path $pkg) {
        Write-Host "==> [pass 2] pip install -e $pkg"
        & $PythonBin -m pip install -e $pkg
        if ($LASTEXITCODE -ne 0) { throw "pip install (pass 2) failed for $pkg" }
    }
}

if ($Extras) {
    Write-Host "==> pip install vz-runtime[$Extras] (extras only)"
    & $PythonBin -m pip install "vz-runtime[$Extras]"
    if ($LASTEXITCODE -ne 0) { throw "pip install extras failed" }
}

Write-Host "Volvence Zero workspace installed successfully."
