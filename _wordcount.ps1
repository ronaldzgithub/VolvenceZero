$content = Get-Content 'docs/business/xfund-pitch-deck-v10-zhaojiangbo.md'
$inScript = $false
$scriptLines = @()
foreach ($line in $content) {
    if ($line -match '^\*\*Speaker script') { $inScript = $true; continue }
    if ($inScript -and ($line -match '^\*\*Design note\*\*' -or $line -match '^---$' -or $line -match '^## ' -or $line -match '^### ' -or $line -match '^\*\*Speaker notes')) { $inScript = $false }
    if ($inScript) { $scriptLines += $line }
}
$scriptText = $scriptLines -join ' '
$scriptWords = ($scriptText -split '\s+' | Where-Object { $_ -ne '' }).Count
$digest = (Get-Content 'docs/business/xfund-pitch-deck-v10-digest.md' -Raw) -split '\s+' | Where-Object { $_ -ne '' }
Write-Host "Original Speaker script words: $scriptWords"
Write-Host "Digest file words            : $($digest.Count)"
Write-Host "Ratio (digest / scripts only): $([math]::Round($digest.Count / $scriptWords * 100, 1))%"
