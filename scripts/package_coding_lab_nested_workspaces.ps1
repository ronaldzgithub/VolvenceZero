param(
    [string]$OutputRoot = "artifacts/coding_lab/nested_workspace_bundles_20260830",
    [string[]]$SourceRoots = @(
        "artifacts/coding_lab/packet35_directed_smoke_20260828a",
        "artifacts/coding_lab/packet36_advice_smoke_20260828a",
        "artifacts/coding_lab/packet36_v21_formal_qwen3codernext_20260828",
        "artifacts/coding_lab/t0_intrinsic_noconv_qwen3codernext_20260828"
    )
)

$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$archiveRoot = Join-Path $repositoryRoot $OutputRoot

if (Test-Path -LiteralPath $archiveRoot) {
    throw "Refusing to overwrite an existing archive root: $archiveRoot"
}

$repositoryDirectories = foreach ($sourceRoot in $SourceRoots) {
    $absoluteSourceRoot = Join-Path $repositoryRoot $sourceRoot
    if (-not (Test-Path -LiteralPath $absoluteSourceRoot -PathType Container)) {
        throw "Source root does not exist: $absoluteSourceRoot"
    }
    Get-ChildItem -LiteralPath $absoluteSourceRoot -Recurse -Directory -Filter repo |
        Sort-Object FullName |
        Select-Object -ExpandProperty FullName
}

if ($repositoryDirectories.Count -eq 0) {
    throw "No nested repo directories were found."
}

New-Item -ItemType Directory -Path (Join-Path $archiveRoot "bundles") | Out-Null
$records = foreach ($repositoryDirectory in $repositoryDirectories) {
    $relativePath = [System.IO.Path]::GetRelativePath($repositoryRoot, $repositoryDirectory).Replace("\", "/")
    $dirtyState = git -C $repositoryDirectory status --porcelain
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to inspect nested repository: $relativePath"
    }
    if ($dirtyState) {
        throw "Nested repository is dirty and cannot be losslessly bundled: $relativePath"
    }

    $head = (git -C $repositoryDirectory rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to resolve HEAD for nested repository: $relativePath"
    }
    $archiveName = ($relativePath -replace "[^A-Za-z0-9._-]", "_") + ".bundle"
    $bundlePath = Join-Path (Join-Path $archiveRoot "bundles") $archiveName
    git -C $repositoryDirectory bundle create $bundlePath --all
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to create bundle for nested repository: $relativePath"
    }
    git bundle verify $bundlePath | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Bundle verification failed for nested repository: $relativePath"
    }

    [ordered]@{
        source_relative_path = $relativePath
        head_commit = $head
        bundle_relative_path = ("bundles/" + $archiveName)
        bundle_sha256 = (Get-FileHash -LiteralPath $bundlePath -Algorithm SHA256).Hash.ToLowerInvariant()
        bundle_bytes = (Get-Item -LiteralPath $bundlePath).Length
        source_status = "clean"
    }
}

$manifest = [ordered]@{
    schema_version = "coding-lab-nested-workspace-bundle.v1"
    package_kind = "git_bundle"
    generated_at_utc = [DateTime]::UtcNow.ToString("o")
    repository_count = @($records).Count
    source_roots = $SourceRoots
    restoration_contract = "Verify SHA-256, git clone --no-checkout the bundle into source_relative_path, then git checkout --detach head_commit."
    records = @($records)
}
$manifestPath = Join-Path $archiveRoot "manifest.json"
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding utf8NoBOM

$readme = @"
# Coding Lab nested-workspace bundle

This package preserves the nested Git workspaces that are intentionally not stored as parent-repository gitlinks.

Each record in `manifest.json` includes the original relative path, exact HEAD commit, bundle SHA-256, and byte size.

To restore one record from the repository root:

```powershell
Get-FileHash <bundle> -Algorithm SHA256
git clone --no-checkout <bundle> <source_relative_path>
git -C <source_relative_path> checkout --detach <head_commit>
```

The packager accepts only clean, non-shallow nested repositories and verifies every generated bundle before writing the manifest.
"@
Set-Content -LiteralPath (Join-Path $archiveRoot "README.md") -Value $readme -Encoding utf8NoBOM

Write-Output "Packaged $(@($records).Count) nested repositories at $archiveRoot"
