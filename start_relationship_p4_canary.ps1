$ErrorActionPreference = "Stop"

$VolvenceRepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VolvenceSourcePaths = Get-ChildItem -Path (Join-Path $VolvenceRepoRoot "packages") -Directory |
    ForEach-Object { Join-Path $_.FullName "src" } |
    Where-Object { Test-Path -Path $_ -PathType Container }
if ($VolvenceSourcePaths.Count -eq 0) {
    throw "No workspace package sources found under $VolvenceRepoRoot/packages."
}
$VolvenceWorkspacePythonPath = $VolvenceSourcePaths -join [IO.Path]::PathSeparator
if ($env:PYTHONPATH) {
    $VolvenceWorkspacePythonPath = "$VolvenceWorkspacePythonPath$([IO.Path]::PathSeparator)$env:PYTHONPATH"
}
$env:PYTHONPATH = $VolvenceWorkspacePythonPath

$VolvenceArgs = @($args)
if ($VolvenceArgs.Count -eq 0) {
    $VolvenceArgs = @("--prepare")
}
$VolvencePythonBin = if ($env:VOLVENCE_PYTHON_BIN) { $env:VOLVENCE_PYTHON_BIN } else { "python" }
& $VolvencePythonBin -m lifeform_domain_emogpt.lab.p4_canary_cli @VolvenceArgs
exit $LASTEXITCODE
