import assert from "node:assert/strict";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";

const REPOSITORY_ROOT = path.resolve(import.meta.dirname, "../../../");
const PROVISIONER_RELATIVE_PATH =
  "packages/vz-runtime/src/volvence_zero/offline_evidence/provision_volvence_evidence_event_log.ps1";
const PROTOCOL_RELATIVE_PATH =
  "packages/vz-runtime/src/volvence_zero/offline_evidence/protocols/windows_cuda_host_stability_qualification_v1.json";
const PROVISIONER_PATH = path.join(REPOSITORY_ROOT, PROVISIONER_RELATIVE_PATH);
const PROTOCOL_PATH = path.join(REPOSITORY_ROOT, PROTOCOL_RELATIVE_PATH);

// This explicit sentinel forces the qualification protocol and its independent
// validator pin to move with every reviewed provisioner source change.
const EXPECTED_PROVISIONER_LF_SHA256 =
  "be0c02f136761f83412f31cdbf1f3249ad7ed15de1aff28e27fe1a8597888406";

const SOURCE = fs.readFileSync(PROVISIONER_PATH, "utf8").replace(/\r\n?/gu, "\n");

function sha256(raw) {
  return crypto.createHash("sha256").update(raw).digest("hex");
}

// This intentionally performs lexical masking only. It lets the tests inspect
// command tokens without executing or dot-sourcing the privileged script.
function maskPowerShellStringsAndComments(source) {
  let output = "";
  let state = "code";
  for (let index = 0; index < source.length; index += 1) {
    const character = source[index];
    const next = source[index + 1] ?? "";
    if (state === "line-comment") {
      if (character === "\n") {
        state = "code";
        output += "\n";
      } else {
        output += " ";
      }
      continue;
    }
    if (state === "block-comment") {
      if (character === "#" && next === ">") {
        output += "  ";
        index += 1;
        state = "code";
      } else {
        output += character === "\n" ? "\n" : " ";
      }
      continue;
    }
    if (state === "single-quoted") {
      if (character === "'" && next === "'") {
        output += "  ";
        index += 1;
      } else if (character === "'") {
        output += " ";
        state = "code";
      } else {
        output += character === "\n" ? "\n" : " ";
      }
      continue;
    }
    if (state === "double-quoted") {
      if (character === "`") {
        output += " ";
        if (next !== "") {
          output += next === "\n" ? "\n" : " ";
          index += 1;
        }
      } else if (character === '"') {
        output += " ";
        state = "code";
      } else {
        output += character === "\n" ? "\n" : " ";
      }
      continue;
    }

    if (character === "#") {
      output += " ";
      state = "line-comment";
    } else if (character === "<" && next === "#") {
      output += "  ";
      index += 1;
      state = "block-comment";
    } else if (character === "'") {
      output += " ";
      state = "single-quoted";
    } else if (character === '"') {
      output += " ";
      state = "double-quoted";
    } else {
      output += character;
    }
  }
  return output;
}

const CODE = maskPowerShellStringsAndComments(SOURCE);

function commandLines(commandName) {
  const commandPattern = new RegExp(`\\b${commandName}\\b`, "u");
  return CODE.split("\n").filter((line) => commandPattern.test(line));
}

function assertTrustedCmdlet(commandName, moduleName) {
  const escapedCommand = commandName.replace(/[.*+?^${}()|[\]\\]/gu, "\\$&");
  const escapedModule = moduleName.replace(/[.*+?^${}()|[\]\\]/gu, "\\$&");
  const moduleQualified = new RegExp(
    `\\b${escapedModule}\\\\${escapedCommand}\\b`,
    "u",
  ).test(CODE);
  const provenanceResolution =
    /\bGet-Command\b/u.test(CODE) &&
    new RegExp(`["']${escapedCommand}["']`, "u").test(SOURCE) &&
    /\.CommandType\s*-c?eq\s*["']Cmdlet["']/u.test(SOURCE) &&
    new RegExp(`\\.ModuleName\\s*-c?eq\\s*["']${escapedModule}["']`, "u").test(
      SOURCE,
    );
  assert.equal(
    moduleQualified || provenanceResolution,
    true,
    `${commandName} must be module-qualified or resolved with exact Cmdlet/module checks`,
  );
}

test("the privileged script exposes exactly Provision and Audit modes", () => {
  const modeParameter = SOURCE.match(
    /\[ValidateSet\(([^)]*)\)\]\s*\[string\]\$Mode/u,
  );
  assert.ok(modeParameter, "Mode must remain an explicit validated string parameter");
  const modes = [...modeParameter[1].matchAll(/["']([^"']+)["']/gu)].map(
    (match) => match[1],
  );
  assert.deepEqual(modes, ["Provision", "Audit"]);
  assert.match(
    SOURCE,
    /\[Parameter\(Mandatory\s*=\s*\$true\)\]\s*\[ValidateSet/u,
  );
  assert.match(SOURCE, /\[switch\]\$AllowSourceCreation/u);
  assert.match(
    SOURCE,
    /source is absent and history is indeterminate[\s\S]{0,180}-AllowSourceCreation/u,
  );
  assert.match(
    SOURCE,
    /allow_source_creation_is_operator_intent_not_history_proof\s*=\s*\$true/u,
  );
});

test("Event Log access is metadata-only and the provisioner never writes records", () => {
  assert.doesNotMatch(CODE, /\bWrite-EventLog\b/u);
  assert.doesNotMatch(CODE, /\bWriteEntry\s*\(/u);

  const reads = commandLines("Get-WinEvent");
  assert.ok(reads.length > 0, "Application channel metadata must be observed");
  for (const line of reads) {
    assert.match(line, /\s-ListLog(?:\s|$)/u);
    assert.doesNotMatch(
      line,
      /\s-(?:FilterHashtable|FilterXml|FilterXPath|LogName|Path|MaxEvents|Oldest)(?:\s|$)/u,
    );
  }
  assert.match(SOURCE, /event_log_records_read\s*=\s*\$false/u);
  assert.match(SOURCE, /event_log_records_written\s*=\s*\$false/u);
});

test("the receipt can never authorize qualification or production evidence", () => {
  const authorizationAssignments = [
    ...SOURCE.matchAll(/^\s*qualification_not_authorized\s*=\s*([^\n]+)$/gmu),
  ];
  assert.ok(
    authorizationAssignments.length >= 2,
    "success and process-failure receipts must both publish the firewall",
  );
  for (const assignment of authorizationAssignments) {
    assert.equal(assignment[1].trim(), "$true");
  }
  assert.doesNotMatch(
    SOURCE,
    /^\s*qualification_not_authorized\s*=\s*\$false\s*$/gmu,
  );
  assert.match(SOURCE, /qualification_or_campaign_invoked\s*=\s*\$false/u);
  assert.match(SOURCE, /qualification_handoff_emitted\s*=\s*\$false/u);
  assert.match(SOURCE, /production_evidence_authorized\s*=\s*\$false/u);
});

test("overall conformance is derived and a nonconformant Audit exits nonzero", () => {
  const derivationStart = SOURCE.indexOf(
    'if ($Mode -eq "Audit") {\n        $overallConformant = (',
  );
  assert.notEqual(derivationStart, -1, "an explicit overall conformance value is required");
  const derivation = SOURCE.slice(derivationStart, derivationStart + 3000);
  for (const requiredTerm of [
    "$sourceAfterConformance.source_configuration_exact",
    "$channelConformanceAfter.log_name_exact",
    "$channelConformanceAfter.enabled",
    "$channelConformanceAfter.classic_log",
    "$channelConformanceAfter.circular_log_mode",
    "$channelConformanceAfter.positive_maximum_size",
    "$channelConformanceBefore.source_provider_membership_present",
    "$channelConformanceAfter.source_provider_membership_present",
  ]) {
    assert.ok(
      derivation.includes(requiredTerm),
      `overall conformance must derive ${requiredTerm}`,
    );
  }
  assert.match(
    SOURCE,
    /overall_conformant\s*=\s*\[bool\]\$overallConformant/u,
  );

  const emitIndex = SOURCE.lastIndexOf("Write-Utf8Bytes -Bytes $resultBytes");
  const failureExitIndex = SOURCE.search(
    /if\s*\(\s*-not\s+\[bool\]\$overallConformant\s*\)\s*\{\s*exit\s+2\s*\}/su,
  );
  assert.ok(emitIndex >= 0, "the audit receipt must be emitted");
  assert.ok(
    failureExitIndex > emitIndex,
    "the complete receipt must be emitted before the nonconformance exit decision",
  );
  assert.match(SOURCE, /\bexit\s+0\b/u);
});

test("refresh chronology never turns a false requirement into proof of refresh", () => {
  assert.match(SOURCE, /refresh_chronology\s*=\s*\[ordered\]@\{/u);
  assert.match(SOURCE, /authoritative\s*=\s*\$false/u);
  assert.match(SOURCE, /cold_boot_observed\s*=\s*\$false/u);
  assert.match(SOURCE, /eventlog_service_restart_observed\s*=\s*\$false/u);
  assert.match(SOURCE, /refresh_verified\s*=\s*\$false/u);
  assert.match(SOURCE, /source_registration_started_at_utc\s*=/u);
  assert.match(SOURCE, /source_registration_completed_at_utc\s*=/u);
  assert.match(SOURCE, /source_configuration_completed_at_utc\s*=/u);
  assert.match(SOURCE, /post_observation_completed_at_utc\s*=/u);

  assert.doesNotMatch(
    SOURCE,
    /requires_cold_or_service_refresh\s*=\s*\[bool\]\$sourceCreated/u,
  );
  assert.doesNotMatch(
    SOURCE,
    /requires_cold_or_service_refresh\s*=\s*\$false/u,
  );
  assert.match(SOURCE, /\$requiresColdOrServiceRefresh\s*=\s*\$null/u);
  assert.match(
    SOURCE,
    /requires_cold_or_service_refresh\s*=\s*\$requiresColdOrServiceRefresh/u,
  );
  assert.match(
    SOURCE,
    /if\s*\(\s*\$sourceCreated\s*\)[\s\S]{0,300}\$requiresColdOrServiceRefresh\s*=\s*\$true/u,
  );
});

test("an existing source is audited exactly and is never repaired automatically", () => {
  assert.match(SOURCE, /automatic repair is forbidden/u);
  assert.match(SOURCE, /automatic_repair_performed\s*=\s*\$false/u);
  assert.doesNotMatch(SOURCE, /automatic_repair_performed\s*=\s*\$true/u);

  const presentBranchStart = SOURCE.indexOf(
    "if ([bool]$sourceBefore.present) {",
  );
  const absentBranchStart = SOURCE.indexOf("\n        else {", presentBranchStart);
  const branchEnd = SOURCE.indexOf("\n    $provisioningState.failure_stage", absentBranchStart);
  assert.ok(
    presentBranchStart >= 0 && absentBranchStart > presentBranchStart && branchEnd > absentBranchStart,
    "source creation must be isolated to the absent-source branch",
  );
  const presentBranch = SOURCE.slice(presentBranchStart, absentBranchStart);
  const absentBranch = SOURCE.slice(absentBranchStart, branchEnd);
  assert.match(
    presentBranch,
    /if\s*\(\s*-not\s+\[bool\]\$sourceBeforeConformance\.source_configuration_exact\s*\)[\s\S]*?throw/u,
  );
  assert.doesNotMatch(presentBranch, /\bNew-VolvenceEvidenceSource\b/u);
  assert.match(absentBranch, /\bNew-VolvenceEvidenceSource\b/u);
});

test("registry MultiString values preserve array shape, order, and element identity", () => {
  assert.match(
    SOURCE,
    /["']MultiString["']\s*\{\s*return\s+,\(\[string\[\]\]\$value\)\s*\}/u,
  );
  assert.match(SOURCE, /\$observedIsArray\s*=\s*\$Observed\s+-is\s+\[Array\]/u);
  assert.match(SOURCE, /\$expectedIsArray\s*=\s*\$Expected\s+-is\s+\[Array\]/u);
  assert.match(SOURCE, /\$Observed\.Count\s+-ne\s+\$Expected\.Count/u);
  assert.match(
    SOURCE,
    /\[string\]\$Observed\[\$index\]\s+-cne\s+\[string\]\$Expected\[\$index\]/u,
  );
  assert.doesNotMatch(
    SOURCE,
    /["']MultiString["'][\s\S]{0,160}(?:-join|\.Join\s*\()/u,
  );
});

test("Application endpoint equality is computed without claiming continuous stability", () => {
  assert.match(SOURCE, /function\s+Test-CompactJsonValueExact\b/u);
  assert.match(
    SOURCE,
    /\$applicationChannelBeforeProjection\s*=\s*Get-ApplicationChannelStableProjection/u,
  );
  assert.match(
    SOURCE,
    /\$applicationChannelAfterProjection\s*=\s*Get-ApplicationChannelStableProjection/u,
  );
  assert.match(
    SOURCE,
    /Test-CompactJsonValueExact\s*`\s*\n\s*-Before\s+\$applicationChannelBeforeProjection\s*`\s*\n\s*-After\s+\$applicationChannelAfterProjection/u,
  );
  assert.match(
    SOURCE,
    /\$applicationChannelFullEndpointEqual\s*=\s*Test-CompactJsonValueExact/u,
  );
  assert.match(
    SOURCE,
    /\$applicationChannelFullEndpointChanged\s*=\s*-not\s+\$applicationChannelFullEndpointEqual/u,
  );
  assert.match(
    SOURCE,
    /application_channel_full_endpoint_changed\s*=\s*\([\s\S]{0,100}\[bool\]\$applicationChannelFullEndpointChanged/u,
  );
  assert.doesNotMatch(
    SOURCE,
    /application_channel_full_endpoint_changed\s*=\s*\$false/u,
  );
  assert.match(SOURCE, /continuous_stability_proven\s*=\s*\$false/u);
  assert.match(
    SOURCE,
    /if \(\$Mode -eq "Audit"\)[\s\S]{0,1800}-not \[bool\]\$applicationChannelFullEndpointChanged/u,
  );
  assert.match(
    SOURCE,
    /\[bool\]\$sourceCreated\s+-and\s+-not \[bool\]\$applicationChannelStableProjectionEndpointChanged/u,
  );
  assert.match(
    SOURCE,
    /\[bool\]\$applicationChannelProviderTransition\.allowed_for_source_creation/u,
  );
  assert.match(
    SOURCE,
    /-not \[bool\]\$sourceCreated\s+-and\s+-not \[bool\]\$applicationChannelFullEndpointChanged/u,
  );
  assert.match(SOURCE, /function\s+Get-ApplicationChannelProviderTransition\b/u);
  assert.match(SOURCE, /disposition\s*=\s*"unchanged"/u);
  assert.match(SOURCE, /disposition\s*=\s*"exact_source_name_addition"/u);
  assert.match(SOURCE, /disposition\s*=\s*"unexpected_provider_membership_transition"/u);
  assert.match(
    SOURCE,
    /source_provider_membership_present\s*=\s*\([\s\S]{0,120}-ccontains\s+\$SourceName/u,
  );
  assert.match(
    SOURCE,
    /provider membership[\s\S]{0,160}source registry key is absent[\s\S]{0,160}automatic repair is forbidden/u,
  );
});

test("external cmdlets have fixed provenance and that observation is published", () => {
  assertTrustedCmdlet("Get-WinEvent", "Microsoft.PowerShell.Diagnostics");
  assertTrustedCmdlet("New-EventLog", "Microsoft.PowerShell.Management");
  assertTrustedCmdlet("Get-ItemPropertyValue", "Microsoft.PowerShell.Management");
  assertTrustedCmdlet("Test-Path", "Microsoft.PowerShell.Management");

  assert.match(SOURCE, /\$cmdletProvenance\s*=\s*\[ordered\]@\{/u);
  assert.match(SOURCE, /cmdlet_provenance\s*=\s*\$cmdletProvenance/u);
  assert.match(SOURCE, /observations\s*=\s*@\(Get-RequiredCmdletProvenance\)/u);
  assert.match(SOURCE, /command_type\s*=/u);
  assert.match(SOURCE, /module_name\s*=/u);
  assert.match(SOURCE, /module_version\s*=/u);
  assert.match(SOURCE, /module_path\s*=/u);
  assert.match(SOURCE, /module_path_sha256\s*=/u);
  assert.match(SOURCE, /implementing_type\s*=/u);
  assert.match(SOURCE, /assembly_location\s*=/u);
  assert.match(SOURCE, /assembly_sha256\s*=/u);
  assert.match(SOURCE, /assembly_version\s*=/u);
  assert.match(SOURCE, /assembly_public_key_token\s*=/u);
  assert.match(SOURCE, /provenance_authoritative\s*=\s*\$false/u);
});

test("Provision exposes its nontransactional partial-failure boundary", () => {
  assert.match(SOURCE, /\$provisioningState\s*=\s*\[ordered\]@\{/u);
  assert.match(SOURCE, /provisioning\s*=\s*\$provisioningState/u);
  assert.match(SOURCE, /attempted\s*=\s*/u);
  assert.match(SOURCE, /completed\s*=\s*/u);
  assert.match(SOURCE, /source_registration_completed\s*=\s*/u);
  assert.match(SOURCE, /registry_values_completed\s*=\s*/u);
  assert.match(SOURCE, /source_acl_completed\s*=\s*/u);
  assert.match(SOURCE, /failure\s*=\s*/u);
  assert.match(SOURCE, /transactional\s*=\s*\$false/u);
  assert.match(SOURCE, /partial_failure_may_leave_source_registered\s*=\s*\$true/u);
  assert.match(SOURCE, /catch\s+\[System\.Exception\]\s*\{/u);
  assert.match(SOURCE, /failure_stage\s*=/u);
  assert.match(SOURCE, /exception_type\s*=/u);
  assert.match(SOURCE, /exception_message\s*=/u);
  assert.match(SOURCE, /fully_qualified_error_id\s*=/u);
  assert.match(SOURCE, /script_stack_trace\s*=/u);
  assert.match(SOURCE, /\bexit\s+3\b/u);
  assert.match(
    SOURCE,
    /required_due_to_source_registration_or_absent_to_present_endpoint_transition_/u,
  );
  assert.match(
    SOURCE,
    /source_registration_completed_before_failure\s*=\s*\(/u,
  );
  assert.match(
    SOURCE,
    /source_registry_observation_completed\s*=\s*\$true/u,
  );
  assert.match(
    SOURCE,
    /\$failureSourceAppearedBetweenEndpoints\s*=\s*\([\s\S]{0,360}\$failurePostObservation\.source_registry_observation_completed/u,
  );
  assert.match(SOURCE, /failure_stage\s*=\s*"serialize_machine_config_core"/u);
  assert.match(SOURCE, /failure_stage\s*=\s*"serialize_success_receipt"/u);
  const processBoundary = SOURCE.indexOf("try {", SOURCE.indexOf("$result = $null"));
  const executionGuard = SOURCE.indexOf(
    "\n    Assert-ExecutionEnvironment",
    processBoundary,
  );
  assert.ok(
    executionGuard > processBoundary,
    "execution-environment failures must be captured by the process receipt boundary",
  );
});

test("the v2 Audit schema publishes explicit conformance and process outcomes", () => {
  assert.match(
    SOURCE,
    /\$AuditSchemaVersion\s*=\s*["']volvence-evidence-event-log-provisioning-audit\.v2["']/u,
  );
  assert.match(SOURCE, /process_exit_code\s*=\s*\[int\]\$processExitCode/u);
  assert.match(SOURCE, /overall_conformant\s*=\s*\[bool\]\$overallConformant/u);
  assert.match(SOURCE, /audit_nonconformance_exit_code\s*=\s*2/u);
  assert.match(SOURCE, /process_failure_exit_code\s*=\s*3/u);
  assert.match(
    SOURCE,
    /return\s+,\(\[Text\.Encoding\]::UTF8\.GetBytes\(\$json\)\)/u,
  );
  const stdoutWriter = SOURCE.match(
    /function\s+Write-Utf8Bytes\s*\{([\s\S]*?)\n\}/u,
  );
  assert.ok(stdoutWriter, "a single byte-oriented stdout writer is required");
  assert.equal(
    [...stdoutWriter[1].matchAll(/\$stdout\.Write\s*\(/gu)].length,
    1,
    "the JSON line must be handed to stdout in one write call",
  );
  assert.doesNotMatch(SOURCE, /\[Console\]::OutputEncoding/u);
});

test("the qualification protocol pins the final LF-canonical provisioner bytes", () => {
  assert.match(
    EXPECTED_PROVISIONER_LF_SHA256,
    /^[0-9a-f]{64}$/u,
    "replace the explicit sentinel after the hardened PS1 is final",
  );
  const protocol = JSON.parse(fs.readFileSync(PROTOCOL_PATH, "utf8"));
  assert.equal(protocol.source_hash_mode, "utf8_lf_canonical_v1");
  assert.equal(
    protocol.source_sha256[PROVISIONER_RELATIVE_PATH],
    EXPECTED_PROVISIONER_LF_SHA256,
  );
  assert.equal(
    sha256(Buffer.from(SOURCE, "utf8")),
    EXPECTED_PROVISIONER_LF_SHA256,
  );
});
