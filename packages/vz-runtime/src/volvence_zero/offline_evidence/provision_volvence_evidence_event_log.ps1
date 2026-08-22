[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Provision", "Audit")]
    [string]$Mode,

    [switch]$AllowSourceCreation
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$ConfigSchemaVersion = "volvence-evidence-event-log-source-config.v1"
$AuditSchemaVersion = "volvence-evidence-event-log-provisioning-audit.v2"
$FailureSchemaVersion = "volvence-evidence-event-log-provisioning-failure.v1"
$LogName = "Application"
$SourceName = "VolvenceEvidence"
$ApplicationRegistrySubKey = "SYSTEM\CurrentControlSet\Services\EventLog\Application"
$SourceRegistrySubKey = "$ApplicationRegistrySubKey\$SourceName"
$ExpectedEventMessageFile = "%SystemRoot%\Microsoft.NET\Framework64\v4.0.30319\EventLogMessages.dll"
$ExpectedSourceAclSddl = "O:BAG:SYD:P(A;CI;KA;;;SY)(A;CI;KA;;;BA)(A;CI;KR;;;BU)(A;CI;KR;;;LS)"
$ScriptRelativePath = "packages/vz-runtime/src/volvence_zero/offline_evidence/provision_volvence_evidence_event_log.ps1"

$ExpectedSourceValues = @(
    [ordered]@{
        name = "EventMessageFile"
        kind = "ExpandString"
        data = $ExpectedEventMessageFile
    }
    [ordered]@{
        name = "TypesSupported"
        kind = "DWord"
        data = [uint32]7
    }
)

function Assert-ExecutionEnvironment {
    if (
        [string]$PSVersionTable.PSEdition -ne "Desktop" -or
        $PSVersionTable.PSVersion.Major -ne 5 -or
        $PSVersionTable.PSVersion.Minor -ne 1
    ) {
        throw "provisioner requires Windows PowerShell Desktop 5.1"
    }
    if (
        [Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT -or
        -not [Environment]::Is64BitOperatingSystem -or
        -not [Environment]::Is64BitProcess
    ) {
        throw "provisioner requires a 64-bit Windows operating system and 64-bit process"
    }
}

function Get-Sha256HexFromBytes {
    param([Parameter(Mandatory = $true)][byte[]]$Bytes)

    $sha256 = [Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($sha256.ComputeHash($Bytes))).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $sha256.Dispose()
    }
}

function Get-Sha256HexFromUtf8 {
    param([Parameter(Mandatory = $true)][string]$Text)

    return Get-Sha256HexFromBytes -Bytes ([Text.Encoding]::UTF8.GetBytes($Text))
}

function Get-UtcText {
    return [datetime]::UtcNow.ToString(
        "yyyy-MM-ddTHH:mm:ss.fffffffZ",
        [Globalization.CultureInfo]::InvariantCulture
    )
}

function Get-ScriptIntegrityObservation {
    if ([string]::IsNullOrWhiteSpace($PSCommandPath)) {
        throw "provisioner script path is unavailable"
    }
    $expectedLeaf = "provision_volvence_evidence_event_log.ps1"
    if ([IO.Path]::GetFileName($PSCommandPath) -cne $expectedLeaf) {
        throw "provisioner script filename drift"
    }
    $strictUtf8 = [Text.UTF8Encoding]::new($false, $true)
    $rawBytes = [IO.File]::ReadAllBytes($PSCommandPath)
    $scriptText = $strictUtf8.GetString($rawBytes)
    $lfText = $scriptText.Replace("`r`n", "`n").Replace("`r", "`n")
    $lfBytes = [Text.Encoding]::UTF8.GetBytes($lfText)
    return [ordered]@{
        repository_relative_path = $ScriptRelativePath
        source_hash_mode = "utf8_lf_canonical_v1"
        lf_canonical_byte_count = [long]$lfBytes.Length
        observed_lf_canonical_sha256 = Get-Sha256HexFromBytes -Bytes $lfBytes
        self_signature_authoritative = $false
        node_protocol_pin_required = $true
        trust_boundary = "observed digest only; a separate Node protocol must pin and independently rehash this script"
    }
}

function Get-RegistryValueData {
    param(
        [Parameter(Mandatory = $true)]$RegistryKey,
        [Parameter(Mandatory = $true)][string]$ValueName,
        [Parameter(Mandatory = $true)][Microsoft.Win32.RegistryValueKind]$Kind
    )

    $value = $RegistryKey.GetValue(
        $ValueName,
        $null,
        [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames
    )
    if ($null -eq $value) {
        throw "registry value unexpectedly resolved to null: $ValueName"
    }
    switch ($Kind) {
        "String" { return [string]$value }
        "ExpandString" { return [string]$value }
        "MultiString" { return ,([string[]]$value) }
        "Binary" { return ([BitConverter]::ToString([byte[]]$value)).Replace("-", "").ToLowerInvariant() }
        "DWord" { return [uint32]$value }
        "QWord" {
            return ([uint64]$value).ToString([Globalization.CultureInfo]::InvariantCulture)
        }
        "None" { return ([BitConverter]::ToString([byte[]]$value)).Replace("-", "").ToLowerInvariant() }
        default { throw "unsupported registry value kind for $ValueName`: $Kind" }
    }
}

function Get-RegistryKeyObservation {
    param(
        [Parameter(Mandatory = $true)][string]$SubKeyPath,
        [Parameter(Mandatory = $true)][bool]$Required
    )

    $baseKey = [Microsoft.Win32.RegistryKey]::OpenBaseKey(
        [Microsoft.Win32.RegistryHive]::LocalMachine,
        [Microsoft.Win32.RegistryView]::Registry64
    )
    try {
        $key = $baseKey.OpenSubKey($SubKeyPath, $false)
        if ($null -eq $key) {
            if ($Required) {
                throw "required registry key is missing: HKLM:\$SubKeyPath"
            }
            return [ordered]@{
                hive = "HKEY_LOCAL_MACHINE"
                registry_view = "Registry64"
                subkey = $SubKeyPath
                present = $false
                values = @()
                security_descriptor_sddl = $null
                security_descriptor_sha256 = $null
                owner_sid = $null
            }
        }
        try {
            $valueNames = [string[]]$key.GetValueNames()
            [Array]::Sort($valueNames, [StringComparer]::Ordinal)
            $values = @()
            foreach ($valueName in $valueNames) {
                $kind = $key.GetValueKind($valueName)
                $values += [ordered]@{
                    name = [string]$valueName
                    kind = $kind.ToString()
                    data = Get-RegistryValueData -RegistryKey $key -ValueName $valueName -Kind $kind
                }
            }
            $security = $key.GetAccessControl()
            $sddl = $security.GetSecurityDescriptorSddlForm(
                [Security.AccessControl.AccessControlSections]::All
            )
            $owner = $security.GetOwner([Security.Principal.SecurityIdentifier])
            if ($null -eq $owner) {
                throw "registry key owner SID is unavailable: HKLM:\$SubKeyPath"
            }
            return [ordered]@{
                hive = "HKEY_LOCAL_MACHINE"
                registry_view = "Registry64"
                subkey = $SubKeyPath
                present = $true
                values = $values
                security_descriptor_sddl = $sddl
                security_descriptor_sha256 = Get-Sha256HexFromUtf8 -Text $sddl
                owner_sid = $owner.Value
            }
        }
        finally {
            $key.Dispose()
        }
    }
    finally {
        $baseKey.Dispose()
    }
}

function Get-NormalizedExpectedSourceAcl {
    $security = [Security.AccessControl.RegistrySecurity]::new()
    $security.SetSecurityDescriptorSddlForm(
        $ExpectedSourceAclSddl,
        [Security.AccessControl.AccessControlSections]::All
    )
    $sddl = $security.GetSecurityDescriptorSddlForm(
        [Security.AccessControl.AccessControlSections]::All
    )
    $owner = $security.GetOwner([Security.Principal.SecurityIdentifier])
    if ($null -eq $owner) {
        throw "expected source registry ACL has no owner SID"
    }
    return [ordered]@{
        security = $security
        sddl = $sddl
        sha256 = Get-Sha256HexFromUtf8 -Text $sddl
        owner_sid = $owner.Value
    }
}

function Test-RegistryValueDataExact {
    param(
        [Parameter(Mandatory = $true)]$Observed,
        [Parameter(Mandatory = $true)]$Expected
    )

    $observedIsArray = $Observed -is [Array]
    $expectedIsArray = $Expected -is [Array]
    if ($observedIsArray -ne $expectedIsArray) {
        return $false
    }
    if ($observedIsArray) {
        if ($Observed.Count -ne $Expected.Count) {
            return $false
        }
        for ($index = 0; $index -lt $Observed.Count; $index += 1) {
            if ([string]$Observed[$index] -cne [string]$Expected[$index]) {
                return $false
            }
        }
        return $true
    }
    if ($Observed.GetType().FullName -cne $Expected.GetType().FullName) {
        return $false
    }
    return [object]::Equals($Observed, $Expected)
}

function Test-SourceValuesExact {
    param([Parameter(Mandatory = $true)]$ObservedValues)

    if ($ObservedValues.Count -ne $ExpectedSourceValues.Count) {
        return $false
    }
    for ($index = 0; $index -lt $ExpectedSourceValues.Count; $index += 1) {
        $observed = $ObservedValues[$index]
        $expected = $ExpectedSourceValues[$index]
        if (
            [string]$observed.name -cne [string]$expected.name -or
            [string]$observed.kind -cne [string]$expected.kind -or
            -not (Test-RegistryValueDataExact -Observed $observed.data -Expected $expected.data)
        ) {
            return $false
        }
    }
    return $true
}

function Get-SourceConformance {
    param(
        [Parameter(Mandatory = $true)]$SourceObservation,
        [Parameter(Mandatory = $true)]$ExpectedAcl
    )

    $valuesExact = $false
    $sddlExact = $false
    $ownerExact = $false
    if ([bool]$SourceObservation.present) {
        $valuesExact = Test-SourceValuesExact -ObservedValues @($SourceObservation.values)
        $sddlExact = [string]$SourceObservation.security_descriptor_sddl -ceq [string]$ExpectedAcl.sddl
        $ownerExact = [string]$SourceObservation.owner_sid -ceq [string]$ExpectedAcl.owner_sid
    }
    return [ordered]@{
        source_present = [bool]$SourceObservation.present
        source_values_exact = [bool]$valuesExact
        source_acl_sddl_exact = [bool]$sddlExact
        source_owner_sid_exact = [bool]$ownerExact
        source_configuration_exact = (
            [bool]$SourceObservation.present -and
            $valuesExact -and
            $sddlExact -and
            $ownerExact
        )
    }
}

function Get-ApplicationChannelObservation {
    $metadataItems = @(
        Microsoft.PowerShell.Diagnostics\Get-WinEvent -ListLog $LogName -ErrorAction Stop
    )
    if ($metadataItems.Count -ne 1) {
        throw "Application channel metadata must resolve exactly once"
    }
    $metadata = $metadataItems[0]
    $channelSddl = [string]$metadata.SecurityDescriptor
    if ([string]::IsNullOrWhiteSpace($channelSddl)) {
        throw "Application channel security descriptor is unavailable"
    }
    $rawSecurity = [Security.AccessControl.RawSecurityDescriptor]::new($channelSddl)
    if ($null -eq $rawSecurity.Owner) {
        throw "Application channel security descriptor has no owner SID"
    }
    $providerNames = [string[]]@($metadata.ProviderNames)
    [Array]::Sort($providerNames, [StringComparer]::Ordinal)
    return [ordered]@{
        log_name = [string]$metadata.LogName
        log_type = [string]$metadata.LogType
        isolation = [string]$metadata.Isolation
        is_enabled = [bool]$metadata.IsEnabled
        is_classic_log = [bool]$metadata.IsClassicLog
        log_mode = [string]$metadata.LogMode
        maximum_size_in_bytes = [long]$metadata.MaximumSizeInBytes
        log_file_path = [string]$metadata.LogFilePath
        owning_provider_name = [string]$metadata.OwningProviderName
        provider_names = @($providerNames)
        security_descriptor_sddl = $channelSddl
        security_descriptor_sha256 = Get-Sha256HexFromUtf8 -Text $channelSddl
        owner_sid = $rawSecurity.Owner.Value
    }
}

function Get-ApplicationChannelConformance {
    param([Parameter(Mandatory = $true)]$Channel)

    return [ordered]@{
        log_name_exact = [string]$Channel.log_name -ceq $LogName
        enabled = [bool]$Channel.is_enabled
        classic_log = [bool]$Channel.is_classic_log
        circular_log_mode = [string]$Channel.log_mode -ceq "Circular"
        positive_maximum_size = [long]$Channel.maximum_size_in_bytes -gt 0
        source_provider_membership_present = (
            [string[]]@($Channel.provider_names) -ccontains $SourceName
        )
    }
}

function Get-PrincipalObservation {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    if ($null -eq $identity.User) {
        throw "current Windows principal SID is unavailable"
    }
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return [ordered]@{
        name = $identity.Name
        sid = $identity.User.Value
        is_administrator = $principal.IsInRole(
            [Security.Principal.WindowsBuiltInRole]::Administrator
        )
    }
}

function Get-MachineObservation {
    $machineGuid = [string](
        Microsoft.PowerShell.Management\Get-ItemPropertyValue `
            -LiteralPath "HKLM:\SOFTWARE\Microsoft\Cryptography" `
            -Name "MachineGuid"
    )
    if ([string]::IsNullOrWhiteSpace($machineGuid)) {
        throw "Windows MachineGuid is unavailable"
    }
    $machineIdentity = Get-Sha256HexFromUtf8 -Text (
        "volvence.windows-machine.v1" + [char]0 + $machineGuid.Trim().ToLowerInvariant()
    )
    return [ordered]@{
        platform_system = "Windows"
        computer_name = [Environment]::MachineName
        machine_identity_sha256 = $machineIdentity
        registry_view = "Registry64"
        process_architecture = "x64"
        powershell_edition = [string]$PSVersionTable.PSEdition
        powershell_version = $PSVersionTable.PSVersion.ToString()
    }
}

function Assert-ApplicationChannelCanHostSource {
    param([Parameter(Mandatory = $true)]$Conformance)

    if (-not [bool]$Conformance.log_name_exact) {
        throw "Application channel log-name drift"
    }
    if (-not [bool]$Conformance.enabled) {
        throw "Application channel is disabled; provisioner does not modify channel configuration"
    }
    if (-not [bool]$Conformance.classic_log) {
        throw "Application channel is not a classic log; provisioner does not modify channel configuration"
    }
    if (-not [bool]$Conformance.circular_log_mode) {
        throw "Application channel must use Circular mode; provisioner does not modify channel configuration"
    }
    if (-not [bool]$Conformance.positive_maximum_size) {
        throw "Application channel maximum size is invalid"
    }
}

function New-VolvenceEvidenceSource {
    param(
        [Parameter(Mandatory = $true)]$ExpectedAcl,
        [Parameter(Mandatory = $true)][System.Collections.IDictionary]$ProvisioningState
    )

    $ProvisioningState.attempted = $true
    $ProvisioningState.failure_stage = "validate_message_resource"
    $expandedMessageFile = [Environment]::ExpandEnvironmentVariables($ExpectedEventMessageFile)
    if (-not (
        Microsoft.PowerShell.Management\Test-Path `
            -LiteralPath $expandedMessageFile `
            -PathType Leaf
    )) {
        throw "fixed Event Log message resource is missing: $expandedMessageFile"
    }
    $newEventLogArguments = @{
        LogName = $LogName
        Source = $SourceName
        ErrorAction = "Stop"
    }
    $ProvisioningState.failure_stage = "register_event_source"
    $ProvisioningState.source_registration_started_at_utc = Get-UtcText
    $null = Microsoft.PowerShell.Management\New-EventLog @newEventLogArguments
    $ProvisioningState.source_registration_completed = $true
    $ProvisioningState.source_registration_completed_at_utc = Get-UtcText
    $ProvisioningState.failure_stage = "open_source_registry"
    $baseKey = [Microsoft.Win32.RegistryKey]::OpenBaseKey(
        [Microsoft.Win32.RegistryHive]::LocalMachine,
        [Microsoft.Win32.RegistryView]::Registry64
    )
    try {
        $sourceRegistryRights = (
            [Security.AccessControl.RegistryRights]::ReadKey -bor
            [Security.AccessControl.RegistryRights]::WriteKey -bor
            [Security.AccessControl.RegistryRights]::ChangePermissions -bor
            [Security.AccessControl.RegistryRights]::TakeOwnership
        )
        $sourceKey = $baseKey.OpenSubKey(
            $SourceRegistrySubKey,
            [Microsoft.Win32.RegistryKeyPermissionCheck]::ReadWriteSubTree,
            $sourceRegistryRights
        )
        if ($null -eq $sourceKey) {
            throw "New-EventLog did not create the VolvenceEvidence source registry key"
        }
        try {
            $ProvisioningState.failure_stage = "write_source_registry_values"
            $sourceKey.SetValue(
                "EventMessageFile",
                $ExpectedEventMessageFile,
                [Microsoft.Win32.RegistryValueKind]::ExpandString
            )
            $sourceKey.SetValue(
                "TypesSupported",
                [uint32]7,
                [Microsoft.Win32.RegistryValueKind]::DWord
            )
            $ProvisioningState.registry_values_completed = $true
            $ProvisioningState.failure_stage = "write_source_registry_acl"
            $sourceKey.SetAccessControl($ExpectedAcl.security)
            $ProvisioningState.source_acl_completed = $true
            $ProvisioningState.failure_stage = "flush_source_registry"
            $sourceKey.Flush()
            $ProvisioningState.registry_flush_completed = $true
        }
        finally {
            $sourceKey.Dispose()
        }
    }
    finally {
        $baseKey.Dispose()
    }
    $ProvisioningState.mutation_completed = $true
    $ProvisioningState.mutation_completed_at_utc = Get-UtcText
    $ProvisioningState.failure_stage = $null
}

function ConvertTo-CompactJsonBytes {
    param([Parameter(Mandatory = $true)]$Value)

    $json = Microsoft.PowerShell.Utility\ConvertTo-Json `
        -InputObject $Value `
        -Depth 32 `
        -Compress
    return ,([Text.Encoding]::UTF8.GetBytes($json))
}

function Write-Utf8Bytes {
    param([Parameter(Mandatory = $true)][byte[]]$Bytes)

    $outputBytes = [byte[]]::new($Bytes.Length + 1)
    [Array]::Copy($Bytes, $outputBytes, $Bytes.Length)
    $outputBytes[$Bytes.Length] = [byte]10
    $stdout = [Console]::OpenStandardOutput()
    $stdout.Write($outputBytes, 0, $outputBytes.Length)
    $stdout.Flush()
}

function Get-RequiredCmdletProvenance {
    $requirements = @(
        [ordered]@{ name = "Get-WinEvent"; module = "Microsoft.PowerShell.Diagnostics" }
        [ordered]@{ name = "New-EventLog"; module = "Microsoft.PowerShell.Management" }
        [ordered]@{ name = "Get-ItemPropertyValue"; module = "Microsoft.PowerShell.Management" }
        [ordered]@{ name = "Test-Path"; module = "Microsoft.PowerShell.Management" }
        [ordered]@{ name = "ConvertTo-Json"; module = "Microsoft.PowerShell.Utility" }
    )
    $observations = @()
    foreach ($requirement in $requirements) {
        $commands = @(
            Microsoft.PowerShell.Core\Get-Command `
                -Name $requirement.name `
                -Module $requirement.module `
                -CommandType Cmdlet `
                -ErrorAction Stop
        )
        if ($commands.Count -ne 1) {
            throw "required cmdlet must resolve exactly once: $($requirement.module)\$($requirement.name)"
        }
        $command = $commands[0]
        if (
            $command.CommandType.ToString() -cne "Cmdlet" -or
            [string]$command.ModuleName -cne [string]$requirement.module
        ) {
            throw "required cmdlet provenance drift: $($requirement.module)\$($requirement.name)"
        }
        $implementingType = $command.ImplementingType
        if ($null -eq $implementingType) {
            throw "required cmdlet implementing type is unavailable: $($requirement.name)"
        }
        $assemblyLocation = [string]$implementingType.Assembly.Location
        if (
            [string]::IsNullOrWhiteSpace($assemblyLocation) -or
            -not [IO.File]::Exists($assemblyLocation)
        ) {
            throw "required cmdlet assembly is unavailable: $($requirement.name)"
        }
        $modulePath = [string]$command.Module.Path
        if (
            [string]::IsNullOrWhiteSpace($modulePath) -or
            -not [IO.File]::Exists($modulePath)
        ) {
            throw "required cmdlet module path is unavailable: $($requirement.name)"
        }
        $assemblyName = $implementingType.Assembly.GetName()
        $publicKeyToken = [byte[]]$assemblyName.GetPublicKeyToken()
        if ($null -eq $publicKeyToken -or $publicKeyToken.Length -eq 0) {
            throw "required cmdlet assembly public-key token is unavailable: $($requirement.name)"
        }
        $observations += [ordered]@{
            command_name = [string]$command.Name
            command_type = $command.CommandType.ToString()
            module_name = [string]$command.ModuleName
            module_version = $command.Module.Version.ToString()
            module_path = $modulePath
            module_path_sha256 = Get-Sha256HexFromBytes -Bytes ([IO.File]::ReadAllBytes($modulePath))
            implementing_type = $implementingType.FullName
            assembly_location = $assemblyLocation
            assembly_sha256 = Get-Sha256HexFromBytes -Bytes ([IO.File]::ReadAllBytes($assemblyLocation))
            assembly_version = $assemblyName.Version.ToString()
            assembly_public_key_token = (
                [BitConverter]::ToString($publicKeyToken)
            ).Replace("-", "").ToLowerInvariant()
            module_qualified_invocation = "$($requirement.module)\$($requirement.name)"
            provenance_authoritative = $false
        }
    }
    return @($observations)
}

function Get-ApplicationChannelStableProjection {
    param([Parameter(Mandatory = $true)]$Channel)

    return [ordered]@{
        log_name = $Channel.log_name
        log_type = $Channel.log_type
        isolation = $Channel.isolation
        is_enabled = $Channel.is_enabled
        is_classic_log = $Channel.is_classic_log
        log_mode = $Channel.log_mode
        maximum_size_in_bytes = $Channel.maximum_size_in_bytes
        log_file_path = $Channel.log_file_path
        owning_provider_name = $Channel.owning_provider_name
        security_descriptor_sddl = $Channel.security_descriptor_sddl
        security_descriptor_sha256 = $Channel.security_descriptor_sha256
        owner_sid = $Channel.owner_sid
    }
}

function Test-CompactJsonValueExact {
    param(
        [Parameter(Mandatory = $true)]$Before,
        [Parameter(Mandatory = $true)]$After
    )

    [byte[]]$beforeBytes = ConvertTo-CompactJsonBytes -Value $Before
    [byte[]]$afterBytes = ConvertTo-CompactJsonBytes -Value $After
    $beforeBase64 = [Convert]::ToBase64String($beforeBytes)
    $afterBase64 = [Convert]::ToBase64String($afterBytes)
    return $beforeBase64 -ceq $afterBase64
}

function Get-ApplicationChannelProviderTransition {
    param(
        [Parameter(Mandatory = $true)]$Before,
        [Parameter(Mandatory = $true)]$After
    )

    $beforeProviders = [string[]]@($Before.provider_names)
    $afterProviders = [string[]]@($After.provider_names)
    if (Test-CompactJsonValueExact -Before $beforeProviders -After $afterProviders) {
        return [ordered]@{
            disposition = "unchanged"
            allowed_for_source_creation = $true
            before_count = [int]$beforeProviders.Count
            after_count = [int]$afterProviders.Count
        }
    }
    $expectedAfterProviders = [string[]]@($beforeProviders + $SourceName)
    [Array]::Sort($expectedAfterProviders, [StringComparer]::Ordinal)
    if (Test-CompactJsonValueExact -Before $expectedAfterProviders -After $afterProviders) {
        return [ordered]@{
            disposition = "exact_source_name_addition"
            allowed_for_source_creation = $true
            before_count = [int]$beforeProviders.Count
            after_count = [int]$afterProviders.Count
        }
    }
    return [ordered]@{
        disposition = "unexpected_provider_membership_transition"
        allowed_for_source_creation = $false
        before_count = [int]$beforeProviders.Count
        after_count = [int]$afterProviders.Count
    }
}

function Test-ApplicationChannelBaseConformance {
    param([Parameter(Mandatory = $true)]$Conformance)

    return (
        [bool]$Conformance.log_name_exact -and
        [bool]$Conformance.enabled -and
        [bool]$Conformance.classic_log -and
        [bool]$Conformance.circular_log_mode -and
        [bool]$Conformance.positive_maximum_size
    )
}

$provisioningState = [ordered]@{
    attempted = $false
    completed = $false
    mutation_completed = $false
    source_registration_completed = $false
    registry_values_completed = $false
    source_acl_completed = $false
    registry_flush_completed = $false
    source_registration_started_at_utc = $null
    source_registration_completed_at_utc = $null
    mutation_completed_at_utc = $null
    failure_stage = $null
    failure = $null
    transactional = $false
    partial_failure_may_leave_source_registered = $true
    automatic_rollback_performed = $false
}
$scriptIntegrity = $null
$principal = $null
$machine = $null
$cmdletProvenance = $null
$sourceBefore = $null
$sourceAfter = $null
$applicationRegistryBefore = $null
$applicationRegistryAfter = $null
$applicationChannelBefore = $null
$applicationChannelAfter = $null
$sourceCreated = $false
$overallConformant = $false
$processExitCode = 3
$result = $null
$resultBytes = $null

try {
    $provisioningState.failure_stage = "validate_execution_environment"
    Assert-ExecutionEnvironment
    if ($Mode -eq "Audit" -and [bool]$AllowSourceCreation) {
        throw "AllowSourceCreation is valid only with explicit Provision mode"
    }

    $provisioningState.failure_stage = "observe_control_plane"
    $scriptIntegrity = Get-ScriptIntegrityObservation
    $cmdletProvenance = [ordered]@{
        observations = @(Get-RequiredCmdletProvenance)
        module_qualified_invocation_required = $true
        powershell_executable_identity_attested = $false
        provenance_authoritative = $false
    }
    $principal = Get-PrincipalObservation
    $machine = Get-MachineObservation
    $expectedAcl = Get-NormalizedExpectedSourceAcl
    $applicationRegistryBefore = Get-RegistryKeyObservation `
        -SubKeyPath $ApplicationRegistrySubKey `
        -Required $true
    $applicationChannelBefore = Get-ApplicationChannelObservation
    $channelConformanceBefore = Get-ApplicationChannelConformance -Channel $applicationChannelBefore
    $sourceBefore = Get-RegistryKeyObservation -SubKeyPath $SourceRegistrySubKey -Required $false
    $sourceBeforeConformance = Get-SourceConformance `
        -SourceObservation $sourceBefore `
        -ExpectedAcl $expectedAcl

    if ($Mode -eq "Provision") {
        if (-not [bool]$principal.is_administrator) {
            throw "Provision mode requires an elevated local administrator token"
        }
        Assert-ApplicationChannelCanHostSource -Conformance $channelConformanceBefore
        if ([bool]$sourceBefore.present) {
            if ([bool]$AllowSourceCreation) {
                throw "AllowSourceCreation is invalid because the source is already present"
            }
            if (-not [bool]$sourceBeforeConformance.source_configuration_exact) {
                throw (
                    "existing VolvenceEvidence source drift; automatic repair is forbidden: " +
                    "values_exact=$([bool]$sourceBeforeConformance.source_values_exact), " +
                    "acl_exact=$([bool]$sourceBeforeConformance.source_acl_sddl_exact), " +
                    "owner_exact=$([bool]$sourceBeforeConformance.source_owner_sid_exact)"
                )
            }
        }
        else {
            if (-not [bool]$AllowSourceCreation) {
                throw (
                    "source is absent and history is indeterminate; explicit " +
                    "-AllowSourceCreation operator intent is required"
                )
            }
            if ([string[]]@($applicationChannelBefore.provider_names) -ccontains $SourceName) {
                throw (
                    "Application channel already reports VolvenceEvidence provider membership " +
                    "while the source registry key is absent; automatic repair is forbidden"
                )
            }
            New-VolvenceEvidenceSource `
                -ExpectedAcl $expectedAcl `
                -ProvisioningState $provisioningState
            $sourceCreated = $true
        }
    }

    $provisioningState.failure_stage = "observe_post_state"
    $sourceAfter = Get-RegistryKeyObservation -SubKeyPath $SourceRegistrySubKey -Required $false
    $sourceAfterConformance = Get-SourceConformance `
        -SourceObservation $sourceAfter `
        -ExpectedAcl $expectedAcl
    $applicationRegistryAfter = Get-RegistryKeyObservation `
        -SubKeyPath $ApplicationRegistrySubKey `
        -Required $true
    $applicationChannelAfter = Get-ApplicationChannelObservation
    $channelConformanceAfter = Get-ApplicationChannelConformance -Channel $applicationChannelAfter
    $postObservationCompletedAtUtc = Get-UtcText

    if ($Mode -eq "Provision" -and -not [bool]$sourceAfterConformance.source_configuration_exact) {
        throw "VolvenceEvidence source creation did not produce the exact versioned configuration"
    }
    if ($Mode -eq "Provision") {
        Assert-ApplicationChannelCanHostSource -Conformance $channelConformanceAfter
        if ([bool]$provisioningState.attempted) {
            $provisioningState.completed = $true
        }
    }

    $applicationChannelBeforeProjection = Get-ApplicationChannelStableProjection `
        -Channel $applicationChannelBefore
    $applicationChannelAfterProjection = Get-ApplicationChannelStableProjection `
        -Channel $applicationChannelAfter
    $applicationChannelStableProjectionEndpointEqual = Test-CompactJsonValueExact `
        -Before $applicationChannelBeforeProjection `
        -After $applicationChannelAfterProjection
    $applicationChannelFullEndpointEqual = Test-CompactJsonValueExact `
        -Before $applicationChannelBefore `
        -After $applicationChannelAfter
    $applicationChannelStableProjectionEndpointChanged = -not (
        $applicationChannelStableProjectionEndpointEqual
    )
    $applicationChannelFullEndpointChanged = -not $applicationChannelFullEndpointEqual
    $applicationChannelProviderTransition = Get-ApplicationChannelProviderTransition `
        -Before $applicationChannelBefore `
        -After $applicationChannelAfter
    $applicationRegistryEndpointChanged = -not (
        Test-CompactJsonValueExact `
            -Before $applicationRegistryBefore `
            -After $applicationRegistryAfter
    )
    $sourceRegistryEndpointChanged = -not (
        Test-CompactJsonValueExact -Before $sourceBefore -After $sourceAfter
    )
    $channelBeforeConformant = Test-ApplicationChannelBaseConformance `
        -Conformance $channelConformanceBefore
    $channelAfterConformant = Test-ApplicationChannelBaseConformance `
        -Conformance $channelConformanceAfter

    if ($Mode -eq "Audit") {
        $overallConformant = (
            [bool]$sourceBeforeConformance.source_configuration_exact -and
            [bool]$sourceAfterConformance.source_configuration_exact -and
            [bool]$channelConformanceBefore.log_name_exact -and
            [bool]$channelConformanceBefore.enabled -and
            [bool]$channelConformanceBefore.classic_log -and
            [bool]$channelConformanceBefore.circular_log_mode -and
            [bool]$channelConformanceBefore.positive_maximum_size -and
            [bool]$channelConformanceBefore.source_provider_membership_present -and
            [bool]$channelConformanceAfter.log_name_exact -and
            [bool]$channelConformanceAfter.enabled -and
            [bool]$channelConformanceAfter.classic_log -and
            [bool]$channelConformanceAfter.circular_log_mode -and
            [bool]$channelConformanceAfter.positive_maximum_size -and
            [bool]$channelConformanceAfter.source_provider_membership_present -and
            [bool]$channelBeforeConformant -and
            [bool]$channelAfterConformant -and
            -not [bool]$applicationChannelFullEndpointChanged -and
            -not [bool]$applicationRegistryEndpointChanged -and
            -not [bool]$sourceRegistryEndpointChanged
        )
    }
    else {
        $overallConformant = (
            [bool]$sourceAfterConformance.source_configuration_exact -and
            [bool]$channelBeforeConformant -and
            [bool]$channelAfterConformant -and
            (
                (
                    [bool]$sourceCreated -and
                    -not [bool]$applicationChannelStableProjectionEndpointChanged -and
                    [bool]$applicationChannelProviderTransition.allowed_for_source_creation
                ) -or
                (
                    -not [bool]$sourceCreated -and
                    -not [bool]$applicationChannelFullEndpointChanged -and
                    [bool]$channelConformanceBefore.source_provider_membership_present -and
                    [bool]$channelConformanceAfter.source_provider_membership_present
                )
            ) -and
            -not [bool]$applicationRegistryEndpointChanged -and
            (
                (
                    [bool]$sourceCreated -and
                    [bool]$sourceRegistryEndpointChanged
                ) -or
                (
                    -not [bool]$sourceCreated -and
                    -not [bool]$sourceRegistryEndpointChanged
                )
            ) -and
            (
                -not [bool]$provisioningState.attempted -or
                [bool]$provisioningState.completed
            )
        )
    }

    $sourceHistoryTransition = "present_continuity_within_invocation"
    if (-not [bool]$sourceBefore.present -and [bool]$sourceAfter.present) {
        $sourceHistoryTransition = "created_during_this_invocation"
    }
    elseif ([bool]$sourceBefore.present -and -not [bool]$sourceAfter.present) {
        $sourceHistoryTransition = "deletion_observed_during_this_invocation"
    }
    elseif (-not [bool]$sourceBefore.present -and -not [bool]$sourceAfter.present) {
        $sourceHistoryTransition = "absent_prior_history_indeterminate"
    }

    $fixedContract = [ordered]@{
        config_schema_version = $ConfigSchemaVersion
        provisioning_owner = "manual_local_administrator_control_plane"
        downstream_authorization = "none; this audit cannot qualify a host or authorize a campaign"
        powershell = "Windows PowerShell Desktop 5.1 x64"
        registry_view = "Registry64"
        log_name = $LogName
        source_name = $SourceName
        write_semantics = "classic direct-string entries; no resource-identifier contract"
        application_registry_subkey = $ApplicationRegistrySubKey
        source_registry_subkey = $SourceRegistrySubKey
        source_values = $ExpectedSourceValues
        source_acl_sddl = $expectedAcl.sddl
        source_acl_sha256 = $expectedAcl.sha256
        source_owner_sid = $expectedAcl.owner_sid
        audit_nonconformance_exit_code = 2
        process_failure_exit_code = 3
        allow_source_creation_is_operator_intent_not_history_proof = $true
        module_qualified_cmdlets_required = $true
    }
    $contentCore = [ordered]@{
        schema_version = "volvence-evidence-event-log-machine-config-core.v2"
        machine = $machine
        script_integrity = $scriptIntegrity
        cmdlet_provenance = $cmdletProvenance
        fixed_contract = $fixedContract
        observed_source_registry = $sourceAfter
        observed_application_registry = $applicationRegistryAfter
        observed_application_channel = $applicationChannelAfter
    }
    $provisioningState.failure_stage = "serialize_machine_config_core"
    [byte[]]$contentBytes = ConvertTo-CompactJsonBytes -Value $contentCore
    $provisioningState.failure_stage = "derive_machine_config_content_id"
    $machineConfigContentId = Get-Sha256HexFromBytes -Bytes $contentBytes
    $requiresColdOrServiceRefresh = $null
    $refreshDisposition = "not_observed_or_proven_by_this_invocation"
    if ($sourceCreated) {
        $requiresColdOrServiceRefresh = $true
        $refreshDisposition = "required_due_to_source_creation_this_invocation"
    }
    $processExitCode = 0
    if (-not [bool]$overallConformant) {
        $processExitCode = 2
    }
    $provisioningState.failure_stage = "construct_success_receipt"
    $result = [ordered]@{
        schema_version = $AuditSchemaVersion
        config_schema_version = $ConfigSchemaVersion
        mode = $Mode
        observed_at_utc = Get-UtcText
        process_exit_code = [int]$processExitCode
        overall_conformant = [bool]$overallConformant
        result_disposition = $(
            if ($overallConformant) {
                if ($Mode -eq "Audit") {
                    "audit_conformant"
                }
                elseif ($sourceCreated) {
                    "provisioned_conformant_refresh_required"
                }
                else {
                    "already_conformant_no_mutation"
                }
            }
            elseif ($Mode -eq "Audit") {
                "audit_nonconformant"
            }
            else {
                "provision_nonconformant"
            }
        )
        source_created_this_invocation = [bool]$sourceCreated
        source_history_transition = $sourceHistoryTransition
        source_absence_history_resolved = $false
        allow_source_creation_operator_intent = [bool]$AllowSourceCreation
        machine = $machine
        invoking_principal = $principal
        script_integrity = $scriptIntegrity
        cmdlet_provenance = $cmdletProvenance
        fixed_contract = $fixedContract
        observed = [ordered]@{
            source_registry_before = $sourceBefore
            source_registry_after = $sourceAfter
            application_registry_before = $applicationRegistryBefore
            application_registry_after = $applicationRegistryAfter
            application_channel_before = $applicationChannelBefore
            application_channel_after = $applicationChannelAfter
        }
        conformance = [ordered]@{
            source_before = $sourceBeforeConformance
            source_after = $sourceAfterConformance
            application_channel_before = $channelConformanceBefore
            application_channel_after = $channelConformanceAfter
            application_channel_full_endpoint_equal = (
                -not [bool]$applicationChannelFullEndpointChanged
            )
            application_channel_stable_projection_endpoint_equal = (
                -not [bool]$applicationChannelStableProjectionEndpointChanged
            )
            application_channel_provider_membership_transition = (
                $applicationChannelProviderTransition
            )
            application_registry_endpoint_equal = (
                -not [bool]$applicationRegistryEndpointChanged
            )
            source_registry_endpoint_equal = (-not [bool]$sourceRegistryEndpointChanged)
            continuous_stability_proven = $false
        }
        provisioning = $provisioningState
        machine_config_content_id_method = "sha256_of_utf8_no_bom_compact_ordered_json_core_v2"
        machine_config_content_id = $machineConfigContentId
        machine_config_content_id_basis_base64 = [Convert]::ToBase64String($contentBytes)
        requires_cold_or_service_refresh = $requiresColdOrServiceRefresh
        refresh_disposition = $refreshDisposition
        refresh_chronology = [ordered]@{
            authoritative = $false
            prior_refresh_state = "not_assessed"
            required_due_to_source_creation_this_invocation = [bool]$sourceCreated
            cold_boot_observed = $false
            eventlog_service_restart_observed = $false
            refresh_verified = $false
            source_registration_started_at_utc = (
                $provisioningState.source_registration_started_at_utc
            )
            source_registration_completed_at_utc = (
                $provisioningState.source_registration_completed_at_utc
            )
            source_configuration_completed_at_utc = $provisioningState.mutation_completed_at_utc
            post_observation_completed_at_utc = $postObservationCompletedAtUtc
        }
        qualification_not_authorized = $true
        safety_boundary = [ordered]@{
            event_log_records_read = $false
            event_log_records_written = $false
            source_registration_performed = [bool]$sourceCreated
            application_channel_full_endpoint_changed = (
                [bool]$applicationChannelFullEndpointChanged
            )
            application_channel_stable_projection_endpoint_changed = (
                [bool]$applicationChannelStableProjectionEndpointChanged
            )
            application_channel_provider_membership_transition = (
                $applicationChannelProviderTransition
            )
            application_registry_endpoint_changed = [bool]$applicationRegistryEndpointChanged
            source_registry_endpoint_changed = [bool]$sourceRegistryEndpointChanged
            continuous_stability_proven = $false
            application_channel_change_attributed_to_script = $false
            automatic_repair_performed = $false
            qualification_or_campaign_invoked = $false
            automatic_invocation_by_qualification_or_campaign_forbidden = $true
            qualification_handoff_emitted = $false
            production_evidence_authorized = $false
            unprivileged_local_event_forgery_excluded = $false
            script_self_signature_authoritative = $false
            cmdlet_provenance_authoritative = $false
            external_node_protocol_pin_required = $true
        }
    }
    $provisioningState.failure_stage = $null
    try {
        $resultBytes = [byte[]](ConvertTo-CompactJsonBytes -Value $result)
    }
    catch [System.Exception] {
        $provisioningState.failure_stage = "serialize_success_receipt"
        throw
    }
}
catch [System.Exception] {
    $failureRecord = $_
    $provisioningState.failure = [ordered]@{
        failure_stage = $provisioningState.failure_stage
        exception_type = $failureRecord.Exception.GetType().FullName
        exception_message = $failureRecord.Exception.Message
        fully_qualified_error_id = [string]$failureRecord.FullyQualifiedErrorId
        script_stack_trace = [string]$failureRecord.ScriptStackTrace
    }

    $failurePostObservation = [ordered]@{
        attempted = $true
        completed = $false
        completed_at_utc = $null
        source_registry_observation_completed = $false
        application_registry_observation_completed = $false
        application_channel_observation_completed = $false
        source_registry_after = $null
        application_registry_after = $null
        application_channel_after = $null
        failure = $null
    }
    try {
        $failurePostObservation.source_registry_after = Get-RegistryKeyObservation `
            -SubKeyPath $SourceRegistrySubKey `
            -Required $false
        $failurePostObservation.source_registry_observation_completed = $true
        $failurePostObservation.application_registry_after = Get-RegistryKeyObservation `
            -SubKeyPath $ApplicationRegistrySubKey `
            -Required $true
        $failurePostObservation.application_registry_observation_completed = $true
        $failurePostObservation.application_channel_after = Get-ApplicationChannelObservation
        $failurePostObservation.application_channel_observation_completed = $true
        $failurePostObservation.completed_at_utc = Get-UtcText
        $failurePostObservation.completed = $true
    }
    catch [System.Exception] {
        $postFailureRecord = $_
        $failurePostObservation.failure = [ordered]@{
            exception_type = $postFailureRecord.Exception.GetType().FullName
            exception_message = $postFailureRecord.Exception.Message
            fully_qualified_error_id = [string]$postFailureRecord.FullyQualifiedErrorId
            script_stack_trace = [string]$postFailureRecord.ScriptStackTrace
        }
    }

    $failureSourceHistoryTransition = "indeterminate_due_to_process_failure"
    if (
        $null -ne $sourceBefore -and
        [bool]$failurePostObservation.source_registry_observation_completed -and
        $null -ne $failurePostObservation.source_registry_after
    ) {
        $failureSourcePresentAfter = [bool]$failurePostObservation.source_registry_after.present
        if (-not [bool]$sourceBefore.present -and $failureSourcePresentAfter) {
            $failureSourceHistoryTransition = "created_or_partially_created_during_this_invocation"
        }
        elseif ([bool]$sourceBefore.present -and -not $failureSourcePresentAfter) {
            $failureSourceHistoryTransition = "deletion_observed_during_this_invocation"
        }
        elseif (-not [bool]$sourceBefore.present -and -not $failureSourcePresentAfter) {
            $failureSourceHistoryTransition = "absent_prior_history_indeterminate"
        }
        else {
            $failureSourceHistoryTransition = "present_continuity_within_invocation"
        }
    }
    $failureSourceAppearedBetweenEndpoints = (
        $null -ne $sourceBefore -and
        -not [bool]$sourceBefore.present -and
        [bool]$failurePostObservation.source_registry_observation_completed -and
        $null -ne $failurePostObservation.source_registry_after -and
        [bool]$failurePostObservation.source_registry_after.present
    )
    $failureRefreshRequired = (
        [bool]$provisioningState.source_registration_completed -or
        [bool]$failureSourceAppearedBetweenEndpoints
    )
    $failureRequiresColdOrServiceRefresh = $null
    $failureRefreshDisposition = "not_observed_or_proven_by_failed_invocation"
    if ($failureRefreshRequired) {
        $failureRequiresColdOrServiceRefresh = $true
        $failureRefreshDisposition = (
            "required_due_to_source_registration_or_absent_to_present_endpoint_transition_" +
            "before_failure"
        )
    }

    $processExitCode = 3
    $overallConformant = $false
    $result = [ordered]@{
        schema_version = $FailureSchemaVersion
        config_schema_version = $ConfigSchemaVersion
        mode = $Mode
        observed_at_utc = Get-UtcText
        process_exit_code = 3
        overall_conformant = $false
        completed = $false
        source_created_this_invocation = [bool]$provisioningState.source_registration_completed
        source_history_transition = $failureSourceHistoryTransition
        source_absence_history_resolved = $false
        allow_source_creation_operator_intent = [bool]$AllowSourceCreation
        invoking_principal = $principal
        machine = $machine
        script_integrity = $scriptIntegrity
        cmdlet_provenance = $cmdletProvenance
        provisioning = $provisioningState
        post_failure_observation = $failurePostObservation
        requires_cold_or_service_refresh = $failureRequiresColdOrServiceRefresh
        refresh_disposition = $failureRefreshDisposition
        refresh_chronology = [ordered]@{
            authoritative = $false
            prior_refresh_state = "not_assessed"
            required_due_to_source_registration_before_failure = (
                [bool]$provisioningState.source_registration_completed
            )
            cold_boot_observed = $false
            eventlog_service_restart_observed = $false
            refresh_verified = $false
            source_registration_started_at_utc = (
                $provisioningState.source_registration_started_at_utc
            )
            source_registration_completed_at_utc = (
                $provisioningState.source_registration_completed_at_utc
            )
            source_configuration_completed_at_utc = $provisioningState.mutation_completed_at_utc
            post_failure_observation_completed_at_utc = $failurePostObservation.completed_at_utc
        }
        qualification_not_authorized = $true
        safety_boundary = [ordered]@{
            event_log_records_read = $false
            event_log_records_written = $false
            automatic_repair_performed = $false
            automatic_rollback_performed = $false
            partial_failure_may_leave_source_registered = $true
            source_registration_completed_before_failure = (
                [bool]$provisioningState.source_registration_completed
            )
            source_appeared_between_observed_endpoints = (
                [bool]$failureSourceAppearedBetweenEndpoints
            )
            qualification_or_campaign_invoked = $false
            qualification_handoff_emitted = $false
            production_evidence_authorized = $false
        }
    }
    $resultBytes = [byte[]](ConvertTo-CompactJsonBytes -Value $result)
}

Write-Utf8Bytes -Bytes $resultBytes
if ($processExitCode -eq 3) {
    exit 3
}
if (-not [bool]$overallConformant) {
    exit 2
}
exit 0
