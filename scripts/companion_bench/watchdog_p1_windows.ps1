param(
    [Parameter(Mandatory = $true)]
    [string]$PidFile,
    [Parameter(Mandatory = $true)]
    [string]$LogPath,
    [int]$IntervalSeconds = 15,
    [double]$MinAvailableMemoryGB = 4.0,
    [double]$MaxGpuMemoryUsedPct = 94.0,
    [int]$SustainedBreaches = 3
)

$ErrorActionPreference = "Stop"

function Write-WatchdogLog {
    param([string]$Message)
    $stamp = (Get-Date).ToString("o")
    Add-Content -Path $LogPath -Value "[$stamp] $Message"
}

function Get-LiveServiceProcesses {
    if (-not (Test-Path $PidFile)) {
        return @()
    }
    $pids = @(Get-Content $PidFile | Where-Object { $_ })
    $live = @()
    foreach ($pidText in $pids) {
        $proc = Get-Process -Id ([int]$pidText) -ErrorAction SilentlyContinue
        if ($proc) {
            $live += $proc
        }
    }
    return $live
}

function Get-MemorySnapshot {
    $os = Get-CimInstance Win32_OperatingSystem
    $availableGb = ([double]$os.FreePhysicalMemory) / 1024.0 / 1024.0
    $totalGb = ([double]$os.TotalVisibleMemorySize) / 1024.0 / 1024.0
    return [pscustomobject]@{
        AvailableGB = $availableGb
        TotalGB = $totalGb
    }
}

function Get-MaxGpuMemoryUsedPct {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        return $null
    }
    $rows = & nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $rows) {
        return $null
    }
    $maxPct = 0.0
    foreach ($row in $rows) {
        $parts = $row.Split(",") | ForEach-Object { $_.Trim() }
        if ($parts.Count -lt 2) {
            continue
        }
        $used = 0.0
        $total = 0.0
        if (-not [double]::TryParse($parts[0], [ref]$used)) {
            continue
        }
        if (-not [double]::TryParse($parts[1], [ref]$total)) {
            continue
        }
        if ($total -le 0) {
            continue
        }
        $pct = ($used / $total) * 100.0
        if ($pct -gt $maxPct) {
            $maxPct = $pct
        }
    }
    return $maxPct
}

function Stop-WatchedServices {
    param([string]$Reason)
    Write-WatchdogLog "stopping watched services: $Reason"
    foreach ($proc in (Get-LiveServiceProcesses)) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
}

New-Item -ItemType Directory -Force -Path (Split-Path $LogPath -Parent) | Out-Null
Write-WatchdogLog "started pid_file=$PidFile interval=${IntervalSeconds}s min_available_memory_gb=$MinAvailableMemoryGB max_gpu_memory_used_pct=$MaxGpuMemoryUsedPct sustained_breaches=$SustainedBreaches"

$breaches = 0
while ($true) {
    $live = @(Get-LiveServiceProcesses)
    if ($live.Count -eq 0) {
        Write-WatchdogLog "exiting: no live watched service processes"
        exit 0
    }

    $memory = Get-MemorySnapshot
    $gpuPct = Get-MaxGpuMemoryUsedPct
    $reasons = @()
    if ($memory.AvailableGB -lt $MinAvailableMemoryGB) {
        $reasons += ("available_memory_gb={0:N2} below {1:N2}" -f $memory.AvailableGB, $MinAvailableMemoryGB)
    }
    if ($null -ne $gpuPct -and $gpuPct -ge $MaxGpuMemoryUsedPct) {
        $reasons += ("gpu_memory_used_pct={0:N1} above {1:N1}" -f $gpuPct, $MaxGpuMemoryUsedPct)
    }

    if ($reasons.Count -gt 0) {
        $breaches += 1
        Write-WatchdogLog "breach ${breaches}/${SustainedBreaches}: $($reasons -join '; ')"
        if ($breaches -ge $SustainedBreaches) {
            Stop-WatchedServices -Reason ($reasons -join "; ")
            exit 42
        }
    } else {
        if ($breaches -gt 0) {
            Write-WatchdogLog ("recovered: available_memory_gb={0:N2} gpu_memory_used_pct={1}" -f $memory.AvailableGB, $(if ($null -eq $gpuPct) { "n/a" } else { "{0:N1}" -f $gpuPct }))
        }
        $breaches = 0
    }

    Start-Sleep -Seconds $IntervalSeconds
}
