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

function Invoke-PipInstall {
    param([string[]] $PipArgs)
    # pip commonly writes harmless warnings to stderr (e.g. "Ignoring
    # invalid distribution ..."). With $ErrorActionPreference=Stop those
    # warnings become terminating ErrorRecords in Windows PowerShell. For
    # native commands, LASTEXITCODE is the only verdict we care about.
    $previousEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $PythonBin -m pip @PipArgs 2>&1 | ForEach-Object { Write-Host $_ }
    } finally {
        $ErrorActionPreference = $previousEap
    }
    return $LASTEXITCODE
}

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
    "packages\lifeform-cultivation",
    "packages\companion-bench",
    "packages\companion-ref-harness",
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
        $exit = Invoke-PipInstall @("install", "-e", $pkg, "--no-deps")
        if ($exit -ne 0) { throw "pip install (pass 1) failed for $pkg" }
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
        $exit = Invoke-PipInstall @("install", "-e", $pkg)
        if ($exit -ne 0) { throw "pip install (pass 2) failed for $pkg" }
    }
}

if ($Extras) {
    Write-Host "==> pip install vz-runtime[$Extras] (extras only)"
    $exit = Invoke-PipInstall @("install", "vz-runtime[$Extras]")
    if ($exit -ne 0) { throw "pip install extras failed" }
    if ($Extras -match "dev") {
        Write-Host "==> pip install -e .[dev] (workspace dev tooling)"
        $exit = Invoke-PipInstall @("install", "-e", ".[dev]")
        if ($exit -ne 0) { throw "pip install dev extras failed" }
    }
}

function Get-PytorchWheelIndex {
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $gpuList = & nvidia-smi -L 2>$null
        if ($LASTEXITCODE -eq 0 -and $gpuList) {
            return @{
                Url = "https://download.pytorch.org/whl/cu126"
                Label = "CUDA 12.6 (GPU detected)"
            }
        }
    }
    return @{
        Url = "https://download.pytorch.org/whl/cpu"
        Label = "CPU (no NVIDIA GPU detected)"
    }
}

# PyPI's default Windows torch wheel can fail to load c10.dll (WinError 1114),
# especially in venvs created from conda Python. Reinstall from the official
# PyTorch index: cu126 when an NVIDIA GPU is present, otherwise CPU.
if ($Extras -and $Extras -match "hf") {
    $torchIndex = Get-PytorchWheelIndex
    Write-Host "==> Reinstall torch from $($torchIndex.Url) ($($torchIndex.Label))"
    & $PythonBin -m pip uninstall -y torch 2>$null | Out-Null
    $exit = Invoke-PipInstall @("install", "torch", "--index-url", $torchIndex.Url)
    if ($exit -ne 0) { throw "pip install torch ($($torchIndex.Label)) failed" }
    & $PythonBin -c @"
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
"@
    if ($LASTEXITCODE -ne 0) {
        throw @"
torch import failed after official-wheel reinstall. Recreate the venv with a
standalone python.org interpreter (not conda), then rerun install.ps1 -Extras hf:

  Remove-Item -Recurse -Force .venv
  `$py = `"`$env:LOCALAPPDATA\Programs\Python\Python311\python.exe`"
  & `$py -m venv .venv
  .\install.ps1 -PythonBin .\.venv\Scripts\python.exe -Extras hf
"@
    }
}

Write-Host "Volvence Zero workspace installed successfully."
