[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Baseline", "Prelaunch", "Delta", "FindAnchors", "WriteAnchor", "VerifyAnchor")]
    [string]$Mode,

    [string]$ScopeId,
    [string]$AnchorPayloadBase64,
    [long]$RecordId,
    [string]$BaselinePath,
    [int]$MaxNewRecordsPerChannel = 4096
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)

if (
    [string]$PSVersionTable.PSEdition -ne "Desktop" -or
    $PSVersionTable.PSVersion.Major -ne 5 -or
    $PSVersionTable.PSVersion.Minor -ne 1
) {
    throw "collector requires Windows PowerShell Desktop 5.1"
}

$AnchorSource = "VolvenceEvidence"
$AnchorSourceRegistryPath = "HKLM:\SYSTEM\CurrentControlSet\Services\EventLog\Application\VolvenceEvidence"
$AnchorEventIds = @{
    preregistration = 8201
    launch = 8202
    terminal = 8203
}

function ConvertTo-UtcText {
    param([Parameter(Mandatory = $true)][datetime]$Value)
    return $Value.ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fffZ", [Globalization.CultureInfo]::InvariantCulture)
}

function Get-Sha256Utf8 {
    param([Parameter(Mandatory = $true)][string]$Text)
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [Text.Encoding]::UTF8.GetBytes($Text)
        return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $sha.Dispose()
    }
}

function Write-JsonOutput {
    param([Parameter(Mandatory = $true)]$Value)
    $json = $Value | ConvertTo-Json -Depth 16 -Compress
    [Console]::Out.WriteLine($json)
}

function Get-MicrocodeRegistryRawLittleEndianHex {
    $properties = Get-ItemProperty -LiteralPath "HKLM:\HARDWARE\DESCRIPTION\System\CentralProcessor\0"
    $value = $properties.'Update Revision'
    if ($null -eq $value) {
        throw "CPU microcode registry value is unavailable"
    }
    if ($value -is [byte[]]) {
        return (($value | ForEach-Object { $_.ToString("x2") }) -join "")
    }
    return ([Convert]::ToString([long]$value, 16)).ToLowerInvariant()
}

function Get-HostObservation {
    $machineGuid = [string](Get-ItemPropertyValue -LiteralPath "HKLM:\SOFTWARE\Microsoft\Cryptography" -Name "MachineGuid")
    if ([string]::IsNullOrWhiteSpace($machineGuid)) {
        throw "Windows MachineGuid is unavailable"
    }
    $machineIdentity = Get-Sha256Utf8 -Text ("volvence.windows-machine.v1" + [char]0 + $machineGuid.Trim().ToLowerInvariant())
    $operatingSystem = Get-CimInstance -ClassName Win32_OperatingSystem
    $bootText = ConvertTo-UtcText -Value $operatingSystem.LastBootUpTime
    $bootIdentity = Get-Sha256Utf8 -Text ("volvence.windows-boot.v1" + [char]0 + $machineIdentity + [char]0 + $bootText)
    $processor = Get-CimInstance -ClassName Win32_Processor | Select-Object -First 1
    $bios = Get-CimInstance -ClassName Win32_BIOS
    $baseboard = Get-CimInstance -ClassName Win32_BaseBoard | Select-Object -First 1
    $gpuAdapters = @(
        Get-CimInstance -ClassName Win32_VideoController | Sort-Object -Property Name | ForEach-Object {
            [ordered]@{
                name = [string]$_.Name
                driver_version = [string]$_.DriverVersion
            }
        }
    )
    return [ordered]@{
        platform_system = "Windows"
        machine_identity_sha256 = $machineIdentity
        boot_identity_sha256 = $bootIdentity
        last_boot_up_time_utc = $bootText
        powershell_version = $PSVersionTable.PSVersion.ToString()
        os = [ordered]@{
            caption = [string]$operatingSystem.Caption
            version = [string]$operatingSystem.Version
            build_number = [string]$operatingSystem.BuildNumber
        }
        cpu = [ordered]@{
            name = [string]$processor.Name
            physical_core_count = [int]$processor.NumberOfCores
            logical_processor_count = [int]$processor.NumberOfLogicalProcessors
        }
        bios = [ordered]@{
            manufacturer = [string]$bios.Manufacturer
            smbios_version = [string]$bios.SMBIOSBIOSVersion
            release_date_utc = ConvertTo-UtcText -Value $bios.ReleaseDate
        }
        baseboard = [ordered]@{
            manufacturer = [string]$baseboard.Manufacturer
            product = [string]$baseboard.Product
            version = [string]$baseboard.Version
        }
        microcode_registry_raw_le_hex = Get-MicrocodeRegistryRawLittleEndianHex
        gpu_adapters = $gpuAdapters
    }
}

function Get-EventXmlSha256 {
    param([Parameter(Mandatory = $true)]$Event)
    return Get-Sha256Utf8 -Text $Event.ToXml()
}

function Get-ChannelCursor {
    param([Parameter(Mandatory = $true)][string]$LogName)
    $metadata = Get-WinEvent -ListLog $LogName
    if (-not $metadata.IsEnabled) {
        throw "Windows Event Log channel is disabled: $LogName"
    }
    if ([string]$metadata.LogMode -ne "Circular") {
        throw "Windows Event Log channel must use Circular mode: $LogName"
    }
    $newest = Get-WinEvent -LogName $LogName -MaxEvents 1
    $oldest = Get-WinEvent -LogName $LogName -Oldest -MaxEvents 1
    if ($null -eq $newest -or $null -eq $oldest) {
        throw "Windows Event Log channel has no readable records: $LogName"
    }
    return [ordered]@{
        log_name = $LogName
        enabled = [bool]$metadata.IsEnabled
        record_count = [long]$metadata.RecordCount
        oldest_record_id = [long]$oldest.RecordId
        newest_record_id = [long]$newest.RecordId
        newest_record_xml_sha256 = Get-EventXmlSha256 -Event $newest
        maximum_size_bytes = [long]$metadata.MaximumSizeInBytes
        log_mode = [string]$metadata.LogMode
    }
}

function Get-EventData {
    param([Parameter(Mandatory = $true)]$Event)
    [xml]$xml = $Event.ToXml()
    $result = @()
    $nodes = @($xml.SelectNodes("/*[local-name()='Event']/*[local-name()='EventData']/*[local-name()='Data']"))
    foreach ($node in $nodes) {
        if ($node -isnot [System.Xml.XmlElement]) {
            throw "Windows Event Log Data node is not an XML element"
        }
        $result += [ordered]@{
            name = [string]$node.GetAttribute("Name")
            value = [string]$node.InnerText
        }
    }
    return @($result)
}

function Get-EventPayloadKind {
    param([Parameter(Mandatory = $true)]$Event)
    [xml]$xml = $Event.ToXml()
    if ($xml.SelectNodes("/*[local-name()='Event']/*[local-name()='EventData']").Count -gt 0) {
        return "event_data"
    }
    if ($xml.SelectNodes("/*[local-name()='Event']/*[local-name()='UserData']").Count -gt 0) {
        return "user_data"
    }
    if ($xml.SelectNodes("/*[local-name()='Event']/*[local-name()='BinaryEventData']").Count -gt 0) {
        return "binary_event_data"
    }
    return "none"
}

function Convert-EventRecord {
    param(
        [Parameter(Mandatory = $true)]$Event,
        [Parameter(Mandatory = $true)][string]$LogName
    )
    $level = $null
    if ($null -ne $Event.Level) {
        $level = [int]$Event.Level
    }
    return [ordered]@{
        log_name = $LogName
        provider_name = [string]$Event.ProviderName
        event_id = [int]$Event.Id
        record_id = [long]$Event.RecordId
        level = $level
        time_created_utc = ConvertTo-UtcText -Value $Event.TimeCreated
        xml_sha256 = Get-EventXmlSha256 -Event $Event
        payload_kind = Get-EventPayloadKind -Event $Event
        event_data = @(Get-EventData -Event $Event)
    }
}

function Get-AnchorPayloadText {
    param([Parameter(Mandatory = $true)]$Event)
    if ($Event.Properties.Count -ne 1 -or $null -eq $Event.Properties[0].Value) {
        throw "Volvence anchor must contain exactly one direct-string property"
    }
    [xml]$xml = $Event.ToXml()
    $nodes = @($xml.SelectNodes("/*[local-name()='Event']/*[local-name()='EventData']/*[local-name()='Data']"))
    if ($nodes.Count -ne 1 -or $nodes[0] -isnot [System.Xml.XmlElement]) {
        throw "Volvence anchor XML must contain exactly one EventData/Data element"
    }
    if (-not [string]::IsNullOrEmpty([string]$nodes[0].GetAttribute("Name"))) {
        throw "Volvence anchor EventData/Data element must be unnamed"
    }
    $propertyText = [string]$Event.Properties[0].Value
    $xmlText = [string]$nodes[0].InnerText
    if ($propertyText -ne $xmlText) {
        throw "Volvence anchor direct-string property and XML payload drift"
    }
    return $propertyText
}

function Assert-AnchorSourceRegistered {
    if (-not (Test-Path -LiteralPath $AnchorSourceRegistryPath -PathType Container)) {
        throw "VolvenceEvidence Application Event Log source must be provisioned before qualification"
    }
}

function Convert-AnchorObservation {
    param([Parameter(Mandatory = $true)]$Event)
    $payloadText = Get-AnchorPayloadText -Event $Event
    return [ordered]@{
        log_name = "Application"
        provider_name = [string]$Event.ProviderName
        event_id = [int]$Event.Id
        record_id = [long]$Event.RecordId
        time_created_utc = ConvertTo-UtcText -Value $Event.TimeCreated
        xml_sha256 = Get-EventXmlSha256 -Event $Event
        payload_base64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($payloadText))
    }
}

function Get-AnchorEvents {
    Assert-AnchorSourceRegistered
    try {
        return @(
            Get-WinEvent -FilterHashtable @{
                LogName = "Application"
                ProviderName = $AnchorSource
                Id = @($AnchorEventIds.preregistration, $AnchorEventIds.launch, $AnchorEventIds.terminal)
            } | Sort-Object -Property RecordId
        )
    }
    catch [System.Exception] {
        if ($_.FullyQualifiedErrorId -eq "NoMatchingEventsFound,Microsoft.PowerShell.Commands.GetWinEventCommand") {
            return @()
        }
        throw
    }
}

if ($Mode -eq "Prelaunch") {
    if ([string]::IsNullOrWhiteSpace($BaselinePath) -or -not (Test-Path -LiteralPath $BaselinePath -PathType Leaf)) {
        throw "Prelaunch requires an existing BaselinePath"
    }
    if ($MaxNewRecordsPerChannel -lt 1 -or $MaxNewRecordsPerChannel -gt 100000) {
        throw "MaxNewRecordsPerChannel must be within 1..100000"
    }
    $startedAt = ConvertTo-UtcText -Value ([datetime]::UtcNow)
    $baseline = Get-Content -LiteralPath $BaselinePath -Raw -Encoding UTF8 | ConvertFrom-Json -ErrorAction Stop
    if ([string]$baseline.schema_version -ne "windows-cuda-strict-32k-host-campaign-event-log-baseline.v1") {
        throw "Prelaunch baseline receipt schema drift"
    }
    $hostObservation = Get-HostObservation
    $channels = @()
    foreach ($baselineChannel in @($baseline.channels)) {
        $logName = [string]$baselineChannel.log_name
        if ($logName -notin @("Application", "System")) {
            throw "Prelaunch baseline channel name drift"
        }
        $baselineNewest = [long]$baselineChannel.newest_record_id
        $cursor = Get-ChannelCursor -LogName $logName
        $boundaryXPath = "*[System[EventRecordID=$baselineNewest]]"
        try {
            $boundary = @(Get-WinEvent -LogName $logName -FilterXPath $boundaryXPath)
        }
        catch [System.Exception] {
            if ($_.FullyQualifiedErrorId -eq "NoMatchingEventsFound,Microsoft.PowerShell.Commands.GetWinEventCommand") {
                $boundary = @()
            }
            else {
                throw
            }
        }
        $boundaryHash = $null
        if ($boundary.Count -eq 1) {
            $boundaryHash = Get-EventXmlSha256 -Event $boundary[0]
        }
        $difference = [long]$cursor.newest_record_id - $baselineNewest
        $channels += [ordered]@{
            log_name = $logName
            baseline_newest_record_id = $baselineNewest
            baseline_boundary_present = ($boundary.Count -eq 1)
            baseline_boundary_xml_sha256 = $boundaryHash
            end_cursor = $cursor
            new_record_count = $difference
            within_record_budget = ($difference -ge 0 -and $difference -le $MaxNewRecordsPerChannel)
        }
    }
    $completedAt = ConvertTo-UtcText -Value ([datetime]::UtcNow)
    Write-JsonOutput -Value ([ordered]@{
        schema_version = "windows-host-event-log-prelaunch-collector.v1"
        collection_started_at_utc = $startedAt
        collection_completed_at_utc = $completedAt
        host = $hostObservation
        channels = $channels
    })
    exit 0
}

if ($Mode -eq "Baseline") {
    $startedAt = ConvertTo-UtcText -Value ([datetime]::UtcNow)
    $hostObservation = Get-HostObservation
    $channels = @(
        Get-ChannelCursor -LogName "Application"
        Get-ChannelCursor -LogName "System"
    )
    $completedAt = ConvertTo-UtcText -Value ([datetime]::UtcNow)
    Write-JsonOutput -Value ([ordered]@{
        schema_version = "windows-host-event-log-baseline-collector.v1"
        collection_started_at_utc = $startedAt
        collection_completed_at_utc = $completedAt
        host = $hostObservation
        channels = $channels
    })
    exit 0
}

if ($Mode -eq "FindAnchors") {
    if ($ScopeId -notmatch '^[0-9a-f]{64}$') {
        throw "FindAnchors requires one lowercase SHA-256 ScopeId"
    }
    $anchors = @()
    foreach ($event in @(Get-AnchorEvents)) {
        $payloadText = Get-AnchorPayloadText -Event $event
        try {
            $payload = $payloadText | ConvertFrom-Json -ErrorAction Stop
        }
        catch [System.ArgumentException] {
            throw "Volvence Event Log anchor payload is invalid JSON at record $($event.RecordId)"
        }
        catch [System.Management.Automation.RuntimeException] {
            throw "Volvence Event Log anchor payload is invalid JSON at record $($event.RecordId)"
        }
        if ([string]$payload.schema_version -ne "volvence-local-event-anchor.v1") {
            throw "Volvence Event Log anchor schema drift at record $($event.RecordId)"
        }
        if ([string]$payload.scope_id -eq $ScopeId) {
            $anchors += Convert-AnchorObservation -Event $event
        }
    }
    Write-JsonOutput -Value ([ordered]@{
        schema_version = "volvence-local-event-anchor-inventory.v1"
        scope_id = $ScopeId
        anchors = @($anchors)
    })
    exit 0
}

if ($Mode -eq "WriteAnchor") {
    if ([string]::IsNullOrWhiteSpace($AnchorPayloadBase64)) {
        throw "WriteAnchor requires AnchorPayloadBase64"
    }
    $payloadText = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($AnchorPayloadBase64))
    $payload = $payloadText | ConvertFrom-Json -ErrorAction Stop
    $kind = [string]$payload.anchor_kind
    if (-not $AnchorEventIds.ContainsKey($kind)) {
        throw "unknown Volvence anchor kind: $kind"
    }
    Assert-AnchorSourceRegistered
    $beforeCursor = Get-ChannelCursor -LogName "Application"
    Write-EventLog -LogName "Application" -Source $AnchorSource -EventId $AnchorEventIds[$kind] -EntryType Information -Message $payloadText
    $afterCursor = Get-ChannelCursor -LogName "Application"
    if ([long]$afterCursor.newest_record_id -le [long]$beforeCursor.newest_record_id) {
        throw "Application Event Log cursor did not advance after Volvence anchor write"
    }
    $anchorXPath = "*[System[(EventRecordID > $([long]$beforeCursor.newest_record_id)) and (EventRecordID <= $([long]$afterCursor.newest_record_id))]]"
    $matches = @(
        Get-WinEvent -LogName "Application" -FilterXPath $anchorXPath -Oldest |
            Where-Object {
                $_.ProviderName -eq $AnchorSource -and
                $_.Id -eq $AnchorEventIds[$kind] -and
                (Get-AnchorPayloadText -Event $_) -eq $payloadText
            }
    )
    if ($matches.Count -ne 1) {
        throw "new Volvence Event Log anchor was not observed exactly once"
    }
    $observation = Convert-AnchorObservation -Event $matches[0]
    Write-JsonOutput -Value ([ordered]@{
        schema_version = "volvence-local-event-anchor-observation.v1"
        log_name = $observation.log_name
        provider_name = $observation.provider_name
        event_id = $observation.event_id
        record_id = $observation.record_id
        time_created_utc = $observation.time_created_utc
        xml_sha256 = $observation.xml_sha256
        payload_base64 = $observation.payload_base64
    })
    exit 0
}

if ($Mode -eq "VerifyAnchor") {
    if ($RecordId -le 0 -or [string]::IsNullOrWhiteSpace($AnchorPayloadBase64)) {
        throw "VerifyAnchor requires RecordId and AnchorPayloadBase64"
    }
    $payloadText = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($AnchorPayloadBase64))
    Assert-AnchorSourceRegistered
    $verifyXPath = "*[System[EventRecordID=$RecordId]]"
    $matches = @(
        Get-WinEvent -LogName "Application" -FilterXPath $verifyXPath |
            Where-Object {
                $_.ProviderName -eq $AnchorSource -and
                $_.Id -in @($AnchorEventIds.preregistration, $AnchorEventIds.launch, $AnchorEventIds.terminal) -and
                (Get-AnchorPayloadText -Event $_) -eq $payloadText
            }
    )
    if ($matches.Count -ne 1) {
        throw "Volvence Event Log anchor is missing, duplicated, or drifted"
    }
    $observation = Convert-AnchorObservation -Event $matches[0]
    Write-JsonOutput -Value ([ordered]@{
        schema_version = "volvence-local-event-anchor-observation.v1"
        log_name = $observation.log_name
        provider_name = $observation.provider_name
        event_id = $observation.event_id
        record_id = $observation.record_id
        time_created_utc = $observation.time_created_utc
        xml_sha256 = $observation.xml_sha256
        payload_base64 = $observation.payload_base64
    })
    exit 0
}

if ($Mode -eq "Delta") {
    if ([string]::IsNullOrWhiteSpace($BaselinePath) -or -not (Test-Path -LiteralPath $BaselinePath -PathType Leaf)) {
        throw "Delta requires an existing BaselinePath"
    }
    if ($MaxNewRecordsPerChannel -lt 1 -or $MaxNewRecordsPerChannel -gt 100000) {
        throw "MaxNewRecordsPerChannel must be within 1..100000"
    }
    $startedAt = ConvertTo-UtcText -Value ([datetime]::UtcNow)
    $baseline = Get-Content -LiteralPath $BaselinePath -Raw -Encoding UTF8 | ConvertFrom-Json -ErrorAction Stop
    if ([string]$baseline.schema_version -ne "windows-cuda-strict-32k-host-campaign-event-log-baseline.v1") {
        throw "Delta baseline receipt schema drift"
    }
    $hostObservation = Get-HostObservation
    $channelDeltas = @()
    foreach ($baselineChannel in @($baseline.channels)) {
        $logName = [string]$baselineChannel.log_name
        if ($logName -notin @("Application", "System")) {
            throw "Delta baseline channel name drift"
        }
        $baselineNewest = [long]$baselineChannel.newest_record_id
        $endCursor = Get-ChannelCursor -LogName $logName
        $difference = [long]$endCursor.newest_record_id - $baselineNewest
        $truncated = $difference -gt $MaxNewRecordsPerChannel
        $configurationStable = (
            [string]$baselineChannel.log_mode -eq "Circular" -and
            [string]$endCursor.log_mode -eq [string]$baselineChannel.log_mode -and
            [long]$endCursor.maximum_size_bytes -eq [long]$baselineChannel.maximum_size_bytes
        )
        if ($truncated -or $difference -lt 0) {
            $windowXPath = "*[System[EventRecordID=$baselineNewest]]"
        }
        else {
            $windowXPath = "*[System[(EventRecordID >= $baselineNewest) and (EventRecordID <= $([long]$endCursor.newest_record_id))]]"
        }
        try {
            $window = @(
                Get-WinEvent -LogName $logName -FilterXPath $windowXPath -Oldest |
                    Sort-Object -Property RecordId
            )
        }
        catch [System.Exception] {
            if ($_.FullyQualifiedErrorId -eq "NoMatchingEventsFound,Microsoft.PowerShell.Commands.GetWinEventCommand") {
                $window = @()
            }
            else {
                throw
            }
        }
        $boundary = @($window | Where-Object { $_.RecordId -eq $baselineNewest })
        $newEvents = @(
            $window | Where-Object {
                $_.RecordId -gt $baselineNewest -and
                $_.RecordId -le [long]$endCursor.newest_record_id
            }
        )
        $boundaryHash = $null
        if ($boundary.Count -eq 1) {
            $boundaryHash = Get-EventXmlSha256 -Event $boundary[0]
        }
        $idsContinuous = $true
        $expectedRecordId = $baselineNewest + 1L
        foreach ($newEvent in $newEvents) {
            if ([long]$newEvent.RecordId -ne $expectedRecordId) {
                $idsContinuous = $false
            }
            $expectedRecordId += 1L
        }
        $endHashExact = $false
        if ($difference -eq 0 -and $boundary.Count -eq 1) {
            $endHashExact = $boundaryHash -eq [string]$endCursor.newest_record_xml_sha256
        }
        elseif ($difference -gt 0 -and $newEvents.Count -gt 0) {
            $endHashExact = (Get-EventXmlSha256 -Event $newEvents[-1]) -eq [string]$endCursor.newest_record_xml_sha256
        }
        $rangeComplete = (
            -not $truncated -and
            $difference -ge 0 -and
            $configurationStable -and
            $boundary.Count -eq 1 -and
            $newEvents.Count -eq $difference -and
            $idsContinuous -and
            $endHashExact
        )
        $events = @(
            $newEvents | ForEach-Object { Convert-EventRecord -Event $_ -LogName $logName }
        )
        $channelDeltas += [ordered]@{
            log_name = $logName
            baseline_newest_record_id = $baselineNewest
            baseline_boundary_present = ($boundary.Count -eq 1)
            baseline_boundary_xml_sha256 = $boundaryHash
            end_cursor = $endCursor
            channel_configuration_stable = [bool]$configurationStable
            end_cursor_hash_exact = [bool]$endHashExact
            scanned_record_count = [int]$events.Count
            record_id_range_complete = [bool]$rangeComplete
            truncated = [bool]$truncated
            events = $events
        }
    }
    $completedAt = ConvertTo-UtcText -Value ([datetime]::UtcNow)
    Write-JsonOutput -Value ([ordered]@{
        schema_version = "windows-host-event-log-delta-collector.v1"
        collection_started_at_utc = $startedAt
        collection_completed_at_utc = $completedAt
        host = $hostObservation
        channels = $channelDeltas
    })
    exit 0
}

throw "unreachable collector mode: $Mode"
