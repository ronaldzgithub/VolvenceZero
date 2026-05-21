[CmdletBinding()]
param(
    [ValidateSet("synthetic", "hf", "full", "dev")]
    [string] $Profile = "synthetic",
    [string] $InstallDir = "",
    [string] $VenvDir = "",
    [string] $RepoUrl = "",
    [string] $RepoRef = "main",
    [string] $PythonBin = "",
    [switch] $SkipSystemDeps,
    [switch] $SkipSmokeTest,
    [switch] $DryRun
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$DefaultRepoRoot = Split-Path -Parent $ScriptDir

if (-not $InstallDir) { $InstallDir = $DefaultRepoRoot }
if (-not $VenvDir) { $VenvDir = Join-Path $InstallDir ".venv" }

function Write-Log([string]$Message) { Write-Host "[bootstrap] $Message" }
function Write-Warn([string]$Message) { Write-Warning "[bootstrap] $Message" }
function Stop-Bootstrap([string]$Message) { throw "[bootstrap] ERROR: $Message" }

function Test-PythonVersion([string]$Bin) {
    & $Bin -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" 2>$null
    return $LASTEXITCODE -eq 0
}

function Find-Python {
    param([string]$Preferred)
    if ($Preferred) {
        if (-not (Test-PythonVersion $Preferred)) {
            Stop-Bootstrap "$Preferred is not Python >= 3.11"
        }
        return $Preferred
    }
    # Prefer python.org installs over conda: conda-created venvs often break
    # PyTorch DLL loading on Windows (WinError 1114 on c10.dll).
    $standalone = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"),
        (Join-Path $env:ProgramFiles "Python311\python.exe"),
        (Join-Path $env:ProgramFiles "Python312\python.exe")
    )
    foreach ($path in $standalone) {
        if ((Test-Path $path) -and (Test-PythonVersion $path)) {
            return $path
        }
    }
    foreach ($candidate in @("py -3.11", "py -3.12", "python", "python3")) {
        try {
            $parts = $candidate -split " ", 2
            if ($parts.Count -eq 2) {
                $resolved = & $parts[0] $parts[1] -c "import sys; print(sys.executable)" 2>$null
            } else {
                $resolved = & $candidate -c "import sys; print(sys.executable)" 2>$null
            }
            if ($LASTEXITCODE -ne 0 -or -not $resolved) { continue }
            $resolved = $resolved.Trim()
            if (Test-PythonVersion $resolved) {
                return $resolved
            }
        } catch {
            continue
        }
    }
    return $null
}

function Install-WindowsPackages {
    Write-Log "Ensuring winget packages (git, python)..."
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Warn "winget not found; install Python 3.11+ and Git manually."
        return
    }
    if (-not $DryRun) {
        winget install --id Git.Git -e --accept-source-agreements --accept-package-agreements 2>$null | Out-Null
        winget install --id Python.Python.3.12 -e --accept-source-agreements --accept-package-agreements 2>$null | Out-Null
    }
}

function Ensure-Repo {
    $installScript = Join-Path $InstallDir "install.ps1"
    if (Test-Path $installScript) {
        Write-Log "Using existing checkout: $InstallDir"
        return
    }
    if (-not $RepoUrl) {
        Stop-Bootstrap "$InstallDir is not a Volvence Zero checkout and -RepoUrl was not set."
    }
    Write-Log "Cloning $RepoUrl -> $InstallDir"
    if (-not $DryRun) {
        if (-not (Test-Path $InstallDir)) {
            New-Item -ItemType Directory -Force -Path (Split-Path $InstallDir) | Out-Null
            git clone $RepoUrl $InstallDir
        } else {
            git -C $InstallDir fetch --all --tags
        }
        git -C $InstallDir checkout $RepoRef
    }
}

function New-Venv {
    param([string]$BasePython)
    $venvPython = Join-Path $VenvDir "Scripts\python.exe"
    if (Test-Path $venvPython) {
        Write-Log "Reusing venv: $VenvDir"
        return $venvPython
    }
    Write-Log "Creating venv: $VenvDir"
    if (-not $DryRun) {
        & $BasePython -m venv $VenvDir
        & $venvPython -m pip install --upgrade pip setuptools wheel
    }
    return $venvPython
}

function Resolve-Extras {
    switch ($Profile) {
        "synthetic" { return "" }
        "hf"        { return "hf" }
        "full"      { return "hf,torch" }
        "dev"       { return "" }
    }
}

function Install-Workspace {
    param([string]$PythonExe, [string]$Extras)
    Write-Log "Installing workspace wheels (profile=$Profile)"
    Push-Location $InstallDir
    try {
        if ($DryRun) {
            Write-Log "+ .\install.ps1 -Extras $Extras"
            return
        }
        if ($Extras) {
            & (Join-Path $InstallDir "install.ps1") -PythonBin $PythonExe -Extras $Extras
        } else {
            & (Join-Path $InstallDir "install.ps1") -PythonBin $PythonExe
        }
        if ($LASTEXITCODE -ne 0) { Stop-Bootstrap "install.ps1 failed" }
        if ($Profile -eq "dev") {
            & $PythonExe -m pip install -e ".[dev]"
        }
    } finally {
        Pop-Location
    }
}

function Invoke-SmokeTest {
    param([string]$PythonExe)
    Write-Log "Running brain kernel smoke test"
    if ($DryRun) { return }
    & $PythonExe -c @"
from volvence_zero.brain import Brain, BrainConfig
session = Brain(BrainConfig()).create_session(session_id='bootstrap-smoke')
result = session.run_turn('I need help making a careful decision.')
text = (result.response.text or '').strip()
if not text:
    raise SystemExit('smoke test returned empty response')
print(text[:200])
"@
    if ($LASTEXITCODE -ne 0) { Stop-Bootstrap "Smoke test failed" }
}

Write-Log "Volvence Zero bare-metal bootstrap (Windows)"
Write-Log "install_dir=$InstallDir profile=$Profile venv=$VenvDir"

if (-not $SkipSystemDeps) {
    Install-WindowsPackages
} else {
    Write-Log "Skipping OS package install (-SkipSystemDeps)"
}

$basePython = Find-Python -Preferred $PythonBin
if (-not $basePython) {
    Stop-Bootstrap "Python >= 3.11 not found. Install it or pass -PythonBin."
}
Write-Log "Using Python: $(& $basePython -c 'import sys; print(sys.executable, sys.version.split()[0])')"

Ensure-Repo
$pythonExe = New-Venv -BasePython $basePython
$extras = Resolve-Extras
Install-Workspace -PythonExe $pythonExe -Extras $extras

if (-not $SkipSmokeTest) {
    Invoke-SmokeTest -PythonExe $pythonExe
} else {
    Write-Log "Skipping smoke test"
}

Write-Host @"

======================================================================
Volvence Zero bootstrap complete
======================================================================
Install dir : $InstallDir
Python      : $pythonExe
Profile     : $Profile

Activate:
  $($VenvDir)\Scripts\Activate.ps1

Start HTTP service:
  $($VenvDir)\Scripts\lifeform-serve.exe --vertical companion --substrate-mode synthetic --host 127.0.0.1 --port 8765

Production (shared Qwen on GPU):
  $($VenvDir)\Scripts\lifeform-serve.exe `
    --vertical companion `
    --substrate-mode hf-shared `
    --substrate-model-id Qwen/Qwen2.5-0.5B-Instruct `
    --substrate-device auto `
    --host 0.0.0.0 `
    --port 8765

"@
