# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.
#
# PowerShell publisher for the public slice of this monorepo to
# companionbench/bench. Native to Windows — no bash / rsync needed.
#
# Mirrors scripts/companion_bench/publish_public_bench.sh; the bash
# version is for Linux CI, this one is for Windows dev boxes.
#
# Usage:
#   .\scripts\companion_bench\publish_public_bench.ps1 [-Mode dry-run|push]
#                                                       [-Remote git@github.com:companionbench/bench.git]
#                                                       [-Branch main]

[CmdletBinding()]
param(
    [ValidateSet("dry-run", "push")]
    [string]$Mode = "dry-run",

    [string]$Remote = "git@github.com:companionbench/bench.git",

    [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

# Allow-list of paths to copy (files OR directories), relative to RepoRoot.
$AllowList = @(
    "packages/companion-bench",
    "site",
    "scripts/companion_bench",
    "docs/external/companion-bench-rfc-v0.md",
    "docs/external/companion-bench-submission-protocol.md",
    "docs/external/companion-bench-governance-charter-draft.md",
    "docs/external/companion-bench-heldout-bootstrap.md",
    "docs/external/companion-bench-public-scenario-hashes.txt",
    "docs/external/eqbench3-submission-protocol.md",
    "docs/external/eqbench3-public-submission-checklist.md",
    "docs/external/eqbench3-results-internal.md",
    ".github/ISSUE_TEMPLATE",
    ".github/workflows/companion-bench-ci-smoke.yml",
    ".github/workflows/companion-bench-paper-suite-small.yml",
    ".github/workflows/companion-bench-paper-suite-full.yml",
    ".github/workflows/companion-bench-publish.yml",
    "tests/contracts/test_companion_bench_no_internal_imports.py",
    "tests/contracts/test_companion_bench_g2_site_cleanup.py",
    "tests/contracts/test_companion_bench_judge_family_rotation.py",
    "tests/contracts/test_no_lscb_strings.py"
)

# Directories pruned from any copy regardless of allow-list.
$ExcludeDirs = @(
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    ".venv",
    "venv"
)
$ExcludeDirSuffixes = @(".egg-info")
$ExcludeFiles = @(".DS_Store", ".coverage")

# Files that legitimately reference the legacy "lscb" token (the guard
# test names it; the G2 cleanup test asserts legacy stubs are absent;
# this publish script logs the legacy token in its diagnostics).
$AllowedLegacyNames = @(
    "test_no_lscb_strings.py",
    "test_companion_bench_g2_site_cleanup.py",
    "publish_public_bench.sh",
    "publish_public_bench.ps1"
)

function Test-Excluded {
    param([string]$Path)
    foreach ($part in ($Path -split "[\\/]")) {
        if ($ExcludeDirs -contains $part) { return $true }
        foreach ($suf in $ExcludeDirSuffixes) {
            if ($part.EndsWith($suf)) { return $true }
        }
        if ($ExcludeFiles -contains $part) { return $true }
    }
    return $false
}

# Stage into a fresh temp dir.
$StageDir = Join-Path ([System.IO.Path]::GetTempPath()) ("companionbench-bench-" + [System.Guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Path $StageDir | Out-Null

Write-Host "[publish] staging into $StageDir"

foreach ($rel in $AllowList) {
    $src = Join-Path $RepoRoot $rel
    if (-not (Test-Path $src)) {
        Write-Warning "allow-list entry missing: $rel"
        continue
    }
    $dst = Join-Path $StageDir $rel
    $dstParent = Split-Path $dst -Parent
    if (-not (Test-Path $dstParent)) {
        New-Item -ItemType Directory -Path $dstParent -Force | Out-Null
    }
    if ((Get-Item $src).PSIsContainer) {
        # Walk the dir and copy non-excluded files.
        Get-ChildItem -Path $src -Recurse -File | ForEach-Object {
            $relInside = $_.FullName.Substring($src.Length).TrimStart("\", "/")
            if (Test-Excluded $relInside) { return }
            $target = Join-Path $dst $relInside
            $targetParent = Split-Path $target -Parent
            if (-not (Test-Path $targetParent)) {
                New-Item -ItemType Directory -Path $targetParent -Force | Out-Null
            }
            Copy-Item $_.FullName $target
        }
    } else {
        Copy-Item $src $dst
    }
}

# Top-level repo polish.
$readmeSrc = Join-Path $RepoRoot "packages/companion-bench/README.md"
if (Test-Path $readmeSrc) {
    Copy-Item $readmeSrc (Join-Path $StageDir "README.md")
}

@'
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   Copyright 2026 Companion Bench Contributors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
   implied.  See the License for the specific language governing
   permissions and limitations under the License.
'@ | Set-Content -Path (Join-Path $StageDir "LICENSE") -Encoding utf8

@'
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.ruff_cache/
*.egg-info/
.coverage
htmlcov/
.DS_Store
artifacts/
'@ | Set-Content -Path (Join-Path $StageDir ".gitignore") -Encoding utf8

# Brand-consistency self-check.
Write-Host "[publish] brand consistency check (no 'lscb' tokens in staged tree)..."
$offenders = @()
Get-ChildItem -Path $StageDir -Recurse -File | ForEach-Object {
    if ($AllowedLegacyNames -contains $_.Name) { return }
    try {
        $text = Get-Content $_.FullName -Raw -Encoding utf8 -ErrorAction Stop
    } catch {
        return
    }
    if ($text -match "(?i)lscb") {
        $offenders += $_.FullName
    }
}
if ($offenders.Count -gt 0) {
    Write-Host "[publish] FAILED: staged tree contains lscb token in:" -ForegroundColor Red
    $offenders | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    Remove-Item $StageDir -Recurse -Force
    exit 1
}
Write-Host "[publish] OK: 0 lscb tokens (legacy-allowed files excepted)"

# Summarise top-level tree.
Write-Host "[publish] staged top-level tree:"
Get-ChildItem $StageDir | Sort-Object Name | ForEach-Object {
    Write-Host ("  " + $_.Name + ($(if ($_.PSIsContainer) { "/" } else { "" })))
}

switch ($Mode) {
    "dry-run" {
        Write-Host ""
        Write-Host "[publish] dry-run complete; staged at: $StageDir"
        Write-Host "[publish] to publish for real: .\scripts\companion_bench\publish_public_bench.ps1 -Mode push"
        Write-Host "[publish] (the staged dir is NOT auto-deleted in dry-run so you can inspect it)"
    }
    "push" {
        Write-Host "[publish] pushing to $Remote (branch $Branch)..."
        Push-Location $StageDir
        try {
            git init -q -b $Branch
            git add .
            $monoHead = (git -C $RepoRoot rev-parse --short HEAD).Trim()
            $stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
            $msg = "publish $stamp from VolvenceZero/VolvenceZero@$monoHead"
            git `
                -c user.email="bot@companionbench.com" `
                -c user.name="Companion Bench Publisher" `
                commit -q -m $msg
            git remote add origin $Remote
            git push --force origin $Branch
        } finally {
            Pop-Location
        }
        Write-Host "[publish] done."
        Write-Host "[publish] staged dir kept at $StageDir for inspection; delete manually when done."
    }
}
