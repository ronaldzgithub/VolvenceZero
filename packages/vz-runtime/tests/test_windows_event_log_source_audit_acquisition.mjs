import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  __testing,
  acquireEventLogSourceAudit,
  EVENT_LOG_SOURCE_AUDIT_ACQUISITION_CLAIM_SCHEMA_VERSION,
  EVENT_LOG_SOURCE_AUDIT_ACQUISITION_PROTOCOL_SCHEMA_VERSION,
  EVENT_LOG_SOURCE_AUDIT_ACQUISITION_TERMINAL_SCHEMA_VERSION,
  validateEventLogSourceAuditAcquisitionArtifact,
} from "../src/volvence_zero/offline_evidence/windows_event_log_source_audit_acquisition.mjs";
import { adaptProvisionerAuditV2Artifact } from "../src/volvence_zero/offline_evidence/windows_cuda_host_stability_qualification.mjs";

const REPOSITORY_ROOT = path.resolve(import.meta.dirname, "../../../");
const MODULE_RELATIVE_PATH =
  "packages/vz-runtime/src/volvence_zero/offline_evidence/windows_event_log_source_audit_acquisition.mjs";
const PROVISIONER_RELATIVE_PATH =
  "packages/vz-runtime/src/volvence_zero/offline_evidence/provision_volvence_evidence_event_log.ps1";
const CLI_RELATIVE_PATH = "scripts/run_windows_event_log_source_audit_acquisition.mjs";
const GIT_ATTRIBUTES_RELATIVE_PATH = ".gitattributes";
const PROTOCOL_PATH = path.join(
  REPOSITORY_ROOT,
  "packages/vz-runtime/src/volvence_zero/offline_evidence/protocols/windows_event_log_source_audit_acquisition_v2.json",
);
const QUALIFICATION_PROTOCOL_PATH = path.join(
  REPOSITORY_ROOT,
  "packages/vz-runtime/src/volvence_zero/offline_evidence/protocols/windows_cuda_host_stability_qualification_v1.json",
);
const EXPECTED_PROTOCOL_ID =
  "4b1b5f5dbf28cf3a51a2c6604cf8f3f061aa241941eecbf6296b890df62e6287";
const EXPECTED_PROTOCOL_RAW_SHA256 =
  "02c8d605766eca3064ad0038432b20354be780d7b8f065fad2325faa166582fb";
const EXPECTED_SOURCE_PINS = Object.freeze({
  [GIT_ATTRIBUTES_RELATIVE_PATH]:
    "4659362bf4b8804f37fb96f7987f6643bc9eed78769f6c6bab61530fabc3ec61",
  [MODULE_RELATIVE_PATH]:
    "b5dfaa8245a8193cf506ce6cf111c95e89b46ceb214ac47eeeff7cd90b22f965",
  [PROVISIONER_RELATIVE_PATH]:
    "be0c02f136761f83412f31cdbf1f3249ad7ed15de1aff28e27fe1a8597888406",
  [CLI_RELATIVE_PATH]:
    "9cacd8e178d6528b7b7bfa319ba18037a9e32e19b2aa3921434016299740491d",
});
const EXPECTED_FILES = Object.freeze([
  "000_scope_claim.json",
  "001_terminal.json",
  "streams/audit.stderr.bin",
  "streams/audit.stdout.bin",
]);
const MACHINE_ID = "a".repeat(64);
const BOOT_ID = "b".repeat(64);
const OPERATOR_SCOPE_ID = "c".repeat(64);
const AUDIT_SCHEMA = "volvence-evidence-event-log-provisioning-audit.v2";
const FAILURE_SCHEMA = "volvence-evidence-event-log-provisioning-failure.v1";
const CAPTURE_ENVELOPE_KEYS = Object.freeze([
  "schema_version",
  "capture_role",
  "stdout_raw_sha256",
  "stdout_byte_count",
  "stderr_raw_sha256",
  "stderr_byte_count",
  "process_exit_code",
  "process_started_at_utc",
  "process_exited_at_utc",
  "stdout_captured_at_utc",
  "machine_identity_sha256",
  "boot_identity_sha256",
  "capture_authoritative",
]);

const PROTOCOL_RAW = fs.readFileSync(PROTOCOL_PATH);
const PROTOCOL = JSON.parse(PROTOCOL_RAW);
const MODULE_SOURCE = fs.readFileSync(path.join(REPOSITORY_ROOT, MODULE_RELATIVE_PATH), "utf8");
const CLI_SOURCE = fs.readFileSync(path.join(REPOSITORY_ROOT, CLI_RELATIVE_PATH), "utf8");

function sha256(raw) {
  return crypto.createHash("sha256").update(raw).digest("hex");
}

function canonicalJson(value) {
  if (value === null) return "null";
  if (typeof value === "boolean" || typeof value === "number") return JSON.stringify(value);
  if (typeof value === "string") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  return `{${Object.keys(value)
    .sort()
    .map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`)
    .join(",")}}`;
}

function contentId(value) {
  return sha256(Buffer.from(canonicalJson(value), "utf8"));
}

function lfCanonicalSha256(filePath) {
  const source = fs.readFileSync(filePath, "utf8").replace(/\r\n?/gu, "\n");
  return sha256(Buffer.from(source, "utf8"));
}

function domainSeparatedSha256(domain, components) {
  const hash = crypto.createHash("sha256");
  hash.update(Buffer.from(domain, "utf8"));
  for (const component of components) {
    const bytes = Buffer.from(component, "utf8");
    const length = Buffer.alloc(8);
    length.writeBigUInt64BE(BigInt(bytes.length));
    hash.update(length);
    hash.update(bytes);
  }
  return hash.digest("hex");
}

function cloneJson(value) {
  return JSON.parse(JSON.stringify(value));
}

function temporaryArtifactRoot(t, leaf = "acquisition") {
  const parent = fs.mkdtempSync(path.join(os.tmpdir(), "volvence-audit-acquisition-test-"));
  t.after(() => fs.rmSync(parent, { recursive: true, force: true }));
  return path.join(parent, leaf);
}

function auditDiscriminatorRaw({
  exitCode = 0,
  overallConformant = exitCode === 0,
  mode = "Audit",
  schemaVersion = AUDIT_SCHEMA,
  completed,
} = {}) {
  const value = {
    schema_version: schemaVersion,
    mode,
    process_exit_code: exitCode,
    overall_conformant: overallConformant,
  };
  if (completed !== undefined) value.completed = completed;
  return Buffer.from(`${JSON.stringify(value)}\n`, "utf8");
}

function failureRaw() {
  return auditDiscriminatorRaw({
    exitCode: 3,
    overallConformant: false,
    schemaVersion: FAILURE_SCHEMA,
    completed: false,
  });
}

function processOutcome({
  stdout = auditDiscriminatorRaw(),
  stderr = Buffer.alloc(0),
  exitCode = 0,
  signal = null,
  timedOut = false,
  overflowStream = null,
  killAttempted = false,
  killAttemptCount = 0,
  spawnErrorName = null,
  spawnErrorMessage = null,
} = {}) {
  const stdoutBytes = Buffer.isBuffer(stdout) ? stdout : Buffer.from(stdout, "utf8");
  const stderrBytes = Buffer.isBuffer(stderr) ? stderr : Buffer.from(stderr, "utf8");
  return {
    processStartedAtUtc: "2026-08-22T00:00:00.0000000Z",
    processExitedAtUtc: "2026-08-22T00:00:03.0000000Z",
    streamsClosedAtUtc: "2026-08-22T00:00:04.0000000Z",
    exitCode,
    signal,
    timedOut,
    overflowStream,
    killAttempted,
    killAttemptCount,
    spawnErrorName,
    spawnErrorMessage,
    stdoutBase64: stdoutBytes.toString("base64"),
    stderrBase64: stderrBytes.toString("base64"),
  };
}

function createSynthetic(t, {
  leaf,
  captureRole = "qualification_source_audit_before",
  machineIdentitySha256 = MACHINE_ID,
  bootIdentitySha256 = BOOT_ID,
  outcome = processOutcome(),
} = {}) {
  const artifactRoot = temporaryArtifactRoot(t, leaf);
  const validation = __testing.createSyntheticEventLogSourceAuditAcquisitionArtifact({
    artifactRoot,
    captureRole,
    operatorScopeBindingId: OPERATOR_SCOPE_ID,
    machineIdentitySha256,
    bootIdentitySha256,
    processOutcome: outcome,
  });
  return { artifactRoot, validation };
}

function readArtifact(artifactRoot) {
  return {
    claimRaw: fs.readFileSync(path.join(artifactRoot, "000_scope_claim.json")),
    claim: JSON.parse(fs.readFileSync(path.join(artifactRoot, "000_scope_claim.json"), "utf8")),
    terminalRaw: fs.readFileSync(path.join(artifactRoot, "001_terminal.json")),
    terminal: JSON.parse(fs.readFileSync(path.join(artifactRoot, "001_terminal.json"), "utf8")),
    stdout: fs.readFileSync(path.join(artifactRoot, "streams/audit.stdout.bin")),
    stderr: fs.readFileSync(path.join(artifactRoot, "streams/audit.stderr.bin")),
  };
}

function rewriteTerminal(artifactRoot, mutate) {
  const terminalPath = path.join(artifactRoot, "001_terminal.json");
  const terminal = JSON.parse(fs.readFileSync(terminalPath, "utf8"));
  mutate(terminal);
  delete terminal.terminal_id;
  terminal.terminal_id = contentId(terminal);
  fs.writeFileSync(terminalPath, Buffer.from(`${canonicalJson(terminal)}\n`, "utf8"));
  return terminal;
}

function listArtifactFiles(root) {
  return fs
    .readdirSync(root, { recursive: true, withFileTypes: true })
    .filter((entry) => entry.isFile())
    .map((entry) => path.relative(root, path.join(entry.parentPath, entry.name)).replaceAll("\\", "/"))
    .sort();
}

function assertDeepFrozen(value, seen = new Set()) {
  if (value === null || typeof value !== "object" || seen.has(value)) return;
  seen.add(value);
  assert.equal(Object.isFrozen(value), true);
  for (const item of Object.values(value)) assertDeepFrozen(item, seen);
}

function registryObservation(subkey, values, sddl, ownerSid) {
  return {
    hive: "HKEY_LOCAL_MACHINE",
    registry_view: "Registry64",
    subkey,
    present: true,
    values: cloneJson(values),
    security_descriptor_sddl: sddl,
    security_descriptor_sha256: sha256(Buffer.from(sddl, "utf8")),
    owner_sid: ownerSid,
  };
}

function buildFullAuditV2({ conformant }) {
  const qualificationProtocol = JSON.parse(fs.readFileSync(QUALIFICATION_PROTOCOL_PATH, "utf8"));
  const fixedContract = cloneJson(qualificationProtocol.event_log_source.required_fixed_contract);
  const provisionerSource = fs
    .readFileSync(path.join(REPOSITORY_ROOT, PROVISIONER_RELATIVE_PATH), "utf8")
    .replace(/\r\n?/gu, "\n");
  const scriptIntegrity = {
    repository_relative_path: PROVISIONER_RELATIVE_PATH,
    source_hash_mode: "utf8_lf_canonical_v1",
    lf_canonical_byte_count: Buffer.byteLength(provisionerSource, "utf8"),
    observed_lf_canonical_sha256: sha256(Buffer.from(provisionerSource, "utf8")),
    self_signature_authoritative: false,
    node_protocol_pin_required: true,
    trust_boundary:
      "observed digest only; a separate Node protocol must pin and independently rehash this script",
  };
  const cmdletPairs = [
    ["Get-WinEvent", "Microsoft.PowerShell.Diagnostics"],
    ["New-EventLog", "Microsoft.PowerShell.Management"],
    ["Get-ItemPropertyValue", "Microsoft.PowerShell.Management"],
    ["Test-Path", "Microsoft.PowerShell.Management"],
    ["ConvertTo-Json", "Microsoft.PowerShell.Utility"],
  ];
  const cmdletProvenance = {
    observations: cmdletPairs.map(([commandName, moduleName], index) => ({
      command_name: commandName,
      command_type: "Cmdlet",
      module_name: moduleName,
      module_version: "3.1.0.0",
      module_path: `C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\Modules\\${moduleName}.psd1`,
      module_path_sha256: String(index + 1).repeat(64),
      implementing_type: `Synthetic.${commandName.replaceAll("-", "")}`,
      assembly_location: `C:\\Windows\\Microsoft.NET\\assembly\\${moduleName}.dll`,
      assembly_sha256: ["6", "7", "8", "9", "a"][index].repeat(64),
      assembly_version: "10.0.0.0",
      assembly_public_key_token: "31bf3856ad364e35",
      module_qualified_invocation: `${moduleName}\\${commandName}`,
      provenance_authoritative: false,
    })),
    module_qualified_invocation_required: true,
    powershell_executable_identity_attested: false,
    provenance_authoritative: false,
  };
  const machine = {
    platform_system: "Windows",
    computer_name: "SYNTHETIC-AUDIT-HOST",
    machine_identity_sha256: MACHINE_ID,
    registry_view: "Registry64",
    process_architecture: "x64",
    powershell_edition: "Desktop",
    powershell_version: "5.1.26100.1",
  };
  const sourceRegistry = registryObservation(
    fixedContract.source_registry_subkey,
    fixedContract.source_values,
    fixedContract.source_acl_sddl,
    fixedContract.source_owner_sid,
  );
  const applicationRegistrySddl = "O:BAG:SYD:P(A;;KA;;;SY)";
  const applicationRegistry = registryObservation(
    fixedContract.application_registry_subkey,
    [],
    applicationRegistrySddl,
    "S-1-5-32-544",
  );
  const channelSddl = "O:BAG:SYD:(A;;0x1;;;SY)";
  const applicationChannel = {
    log_name: "Application",
    log_type: "Administrative",
    isolation: "Application",
    is_enabled: conformant,
    is_classic_log: true,
    log_mode: "Circular",
    maximum_size_in_bytes: 20_971_520,
    log_file_path: "%SystemRoot%\\System32\\Winevt\\Logs\\Application.evtx",
    owning_provider_name: "Microsoft-Windows-Eventlog",
    provider_names: ["Application Error", "VolvenceEvidence"],
    security_descriptor_sddl: channelSddl,
    security_descriptor_sha256: sha256(Buffer.from(channelSddl, "utf8")),
    owner_sid: "S-1-5-32-544",
  };
  const observed = {
    source_registry_before: cloneJson(sourceRegistry),
    source_registry_after: cloneJson(sourceRegistry),
    application_registry_before: cloneJson(applicationRegistry),
    application_registry_after: cloneJson(applicationRegistry),
    application_channel_before: cloneJson(applicationChannel),
    application_channel_after: cloneJson(applicationChannel),
  };
  const sourceConformance = {
    source_present: true,
    source_values_exact: true,
    source_acl_sddl_exact: true,
    source_owner_sid_exact: true,
    source_configuration_exact: true,
  };
  const channelConformance = {
    log_name_exact: true,
    enabled: conformant,
    classic_log: true,
    circular_log_mode: true,
    positive_maximum_size: true,
    source_provider_membership_present: true,
  };
  const providerTransition = {
    disposition: "unchanged",
    allowed_for_source_creation: true,
    before_count: 2,
    after_count: 2,
  };
  const conformance = {
    source_before: cloneJson(sourceConformance),
    source_after: cloneJson(sourceConformance),
    application_channel_before: cloneJson(channelConformance),
    application_channel_after: cloneJson(channelConformance),
    application_channel_full_endpoint_equal: true,
    application_channel_stable_projection_endpoint_equal: true,
    application_channel_provider_membership_transition: cloneJson(providerTransition),
    application_registry_endpoint_equal: true,
    source_registry_endpoint_equal: true,
    continuous_stability_proven: false,
  };
  const contentCore = {
    schema_version: "volvence-evidence-event-log-machine-config-core.v2",
    machine,
    script_integrity: scriptIntegrity,
    cmdlet_provenance: cmdletProvenance,
    fixed_contract: fixedContract,
    observed_source_registry: observed.source_registry_after,
    observed_application_registry: observed.application_registry_after,
    observed_application_channel: observed.application_channel_after,
  };
  const basisBytes = Buffer.from(JSON.stringify(contentCore), "utf8");
  const exitCode = conformant ? 0 : 2;
  const receipt = {
    schema_version: AUDIT_SCHEMA,
    config_schema_version: "volvence-evidence-event-log-source-config.v1",
    mode: "Audit",
    observed_at_utc: "2026-08-22T00:00:02.0000000Z",
    process_exit_code: exitCode,
    overall_conformant: conformant,
    result_disposition: conformant ? "audit_conformant" : "audit_nonconformant",
    source_created_this_invocation: false,
    source_history_transition: "present_continuity_within_invocation",
    source_absence_history_resolved: false,
    allow_source_creation_operator_intent: false,
    machine,
    invoking_principal: {
      name: "SYNTHETIC-AUDIT-HOST\\operator",
      sid: "S-1-5-21-1-2-3-1001",
      is_administrator: false,
    },
    script_integrity: scriptIntegrity,
    cmdlet_provenance: cmdletProvenance,
    fixed_contract: fixedContract,
    observed,
    conformance,
    provisioning: {
      attempted: false,
      completed: false,
      mutation_completed: false,
      source_registration_completed: false,
      registry_values_completed: false,
      source_acl_completed: false,
      registry_flush_completed: false,
      source_registration_started_at_utc: null,
      source_registration_completed_at_utc: null,
      mutation_completed_at_utc: null,
      failure_stage: null,
      failure: null,
      transactional: false,
      partial_failure_may_leave_source_registered: true,
      automatic_rollback_performed: false,
    },
    machine_config_content_id_method:
      "sha256_of_utf8_no_bom_compact_ordered_json_core_v2",
    machine_config_content_id: sha256(basisBytes),
    machine_config_content_id_basis_base64: basisBytes.toString("base64"),
    requires_cold_or_service_refresh: null,
    refresh_disposition: "not_observed_or_proven_by_this_invocation",
    refresh_chronology: {
      authoritative: false,
      prior_refresh_state: "not_assessed",
      required_due_to_source_creation_this_invocation: false,
      cold_boot_observed: false,
      eventlog_service_restart_observed: false,
      refresh_verified: false,
      source_registration_started_at_utc: null,
      source_registration_completed_at_utc: null,
      source_configuration_completed_at_utc: null,
      post_observation_completed_at_utc: "2026-08-22T00:00:01.0000000Z",
    },
    qualification_not_authorized: true,
    safety_boundary: {
      event_log_records_read: false,
      event_log_records_written: false,
      source_registration_performed: false,
      application_channel_full_endpoint_changed: false,
      application_channel_stable_projection_endpoint_changed: false,
      application_channel_provider_membership_transition: providerTransition,
      application_registry_endpoint_changed: false,
      source_registry_endpoint_changed: false,
      continuous_stability_proven: false,
      application_channel_change_attributed_to_script: false,
      automatic_repair_performed: false,
      qualification_or_campaign_invoked: false,
      automatic_invocation_by_qualification_or_campaign_forbidden: true,
      qualification_handoff_emitted: false,
      production_evidence_authorized: false,
      unprivileged_local_event_forgery_excluded: false,
      script_self_signature_authoritative: false,
      cmdlet_provenance_authoritative: false,
      external_node_protocol_pin_required: true,
    },
  };
  return { receipt, raw: Buffer.from(`${JSON.stringify(receipt)}\n`, "utf8") };
}

test("acquisition protocol identity, critical source pins, and fixed Audit-only invocation are frozen", () => {
  assert.equal(PROTOCOL.schema_version, EVENT_LOG_SOURCE_AUDIT_ACQUISITION_PROTOCOL_SCHEMA_VERSION);
  assert.equal(sha256(PROTOCOL_RAW), EXPECTED_PROTOCOL_RAW_SHA256);
  assert.equal(contentId(PROTOCOL), EXPECTED_PROTOCOL_ID);
  assert.deepEqual(PROTOCOL.source_sha256, EXPECTED_SOURCE_PINS);
  for (const [relativePath, expectedHash] of Object.entries(EXPECTED_SOURCE_PINS)) {
    assert.equal(lfCanonicalSha256(path.join(REPOSITORY_ROOT, relativePath)), expectedHash);
  }
  assert.equal(PROTOCOL.execution.production_entrypoint_enabled, false);
  assert.equal(PROTOCOL.execution.platform_system, "Windows");
  assert.equal(
    PROTOCOL.execution.executable_template,
    "{SystemRoot}\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",
  );
  assert.deepEqual(PROTOCOL.execution.argv_template, [
    "-NoLogo",
    "-NoProfile",
    "-NonInteractive",
    "-EncodedCommand",
    "{frozen_same_buffer_launcher_utf16le_base64}",
  ]);
  const launcher = __testing.frozenSourceBindingLauncherObservationForTesting();
  assert.equal(
    launcher.sourceUtf8Sha256,
    PROTOCOL.execution.source_execution_binding.launcher_source_utf8_sha256,
  );
  assert.equal(
    launcher.utf16leSha256,
    PROTOCOL.execution.source_execution_binding.launcher_utf16le_sha256,
  );
  assert.equal(
    PROTOCOL.execution.source_execution_binding.provisioner_relative_path,
    PROVISIONER_RELATIVE_PATH,
  );
  assert.equal(PROTOCOL.execution.source_execution_binding.same_handle_read_hash_execute, true);
  assert.equal(
    PROTOCOL.execution.source_execution_binding
      .handle_held_through_script_execution_and_exit_unwind,
    true,
  );
  assert.equal(
    PROTOCOL.execution.source_execution_binding
      .handle_held_until_os_process_exit_attested,
    false,
  );
  assert.equal(PROTOCOL.execution.source_execution_binding.path_reopened_for_execution, false);
  assert.equal(PROTOCOL.execution.source_execution_binding.realized_source_execution_attested, false);
  assert.equal(PROTOCOL.execution.source_execution_binding.executable_image_identity_attested, false);
  assert.equal(PROTOCOL.execution.shell, false);
  assert.equal(PROTOCOL.execution.windows_hide, true);
  assert.equal(PROTOCOL.execution.source_and_executable_pre_post_endpoint_equality_required, true);
  assert.equal(PROTOCOL.execution.endpoint_equality_proves_continuous_stability, false);
  assert.equal(PROTOCOL.execution.post_kill_hard_cutoff_required, true);
  assert.equal(PROTOCOL.execution.overall_supervision_deadline_required, true);
  assert.equal(
    PROTOCOL.budgets.overall_supervision_deadline_milliseconds,
    PROTOCOL.budgets.timeout_milliseconds +
      PROTOCOL.budgets.post_kill_pipe_drain_grace_milliseconds,
  );
  assert.equal(PROTOCOL.scope.attempt_budget_applies_to, "single_artifact_root_only");
  assert.equal(PROTOCOL.scope.same_root_overwrite_or_retry_permitted, false);
  assert.equal(PROTOCOL.scope.cross_root_duplicate_scope_excluded, false);
  assert.equal(PROTOCOL.scope.scope_global_no_retry_proven, false);
  assert.equal(PROTOCOL.output_contract.artifact_parent_must_preexist_as_non_symlink_directory, true);
  assert.equal(PROTOCOL.output_contract.artifact_parent_chain_reparse_points_excluded, false);
  assert.equal(
    PROTOCOL.output_contract.claim_file_descriptor_content_fsync_before_process_creation,
    true,
  );
  assert.equal(PROTOCOL.output_contract.directory_entry_durability_guaranteed, false);
  assert.equal(PROTOCOL.output_contract.normal_candidate_requires_child_exit_and_close, true);
  assert.equal(PROTOCOL.output_contract.hard_cutoff_persists_bounded_prefix_for_quarantine, true);
  assert.equal(
    PROTOCOL.output_contract.async_process_supervision_deadline_excludes_synchronous_persistence,
    true,
  );
  for (const prohibited of PROTOCOL.execution.prohibited_tokens) {
    assert.equal(PROTOCOL.execution.argv_template.includes(prohibited), false);
  }
  assert.deepEqual(PROTOCOL.output_contract.exact_files, [
    "000_scope_claim.json",
    "001_terminal.json",
    "streams/audit.stdout.bin",
    "streams/audit.stderr.bin",
  ]);
});

test("critical source LF hashing preserves a leading UTF-8 BOM", () => {
  const observation =
    __testing.frozenUtf8BomCanonicalizationObservationForTesting();
  assert.equal(
    observation.withBomLfCanonicalSha256,
    observation.expectedPreservedBomLfCanonicalSha256,
  );
  assert.notEqual(
    observation.withBomLfCanonicalSha256,
    observation.withoutBomLfCanonicalSha256,
  );
});

test("repository attributes materialize the reviewed provisioner as LF", () => {
  const safeRoot = REPOSITORY_ROOT.replace(/\\/gu, "/");
  const attributes = execFileSync(
    "git",
    [
      "-c",
      `safe.directory=${safeRoot}`,
      "check-attr",
      "text",
      "eol",
      "--",
      PROVISIONER_RELATIVE_PATH,
    ],
    { cwd: REPOSITORY_ROOT, encoding: "utf8" },
  );
  assert.match(attributes, /: text: set\r?\n/u);
  assert.match(attributes, /: eol: lf\r?\n/u);
  assert.ok(EXPECTED_SOURCE_PINS[GIT_ATTRIBUTES_RELATIVE_PATH]);
});

test("public acquisition gate throws before a poison Proxy can be observed", () => {
  let observations = 0;
  const poison = new Proxy(
    {},
    {
      get() {
        observations += 1;
        throw new Error("poison get");
      },
      ownKeys() {
        observations += 1;
        throw new Error("poison ownKeys");
      },
      getOwnPropertyDescriptor() {
        observations += 1;
        throw new Error("poison descriptor");
      },
    },
  );
  assert.throws(
    () => acquireEventLogSourceAudit(poison),
    /production-disabled in protocol v2/i,
  );
  assert.equal(observations, 0);
  assert.deepEqual(Object.keys(__testing), [
    "createSyntheticEventLogSourceAuditAcquisitionArtifact",
    "exerciseFixedAuditLifecycleScenarioForTesting",
    "exerciseSourceBindingLauncherFixtureForTesting",
    "frozenSourceBindingLauncherObservationForTesting",
    "frozenUtf8BomCanonicalizationObservationForTesting",
  ]);
  assert.match(
    acquireEventLogSourceAudit.toString(),
    /^function acquireEventLogSourceAudit\(\) \{\s*throw new Error\(PRODUCTION_DISABLED_MESSAGE\);\s*\}$/u,
  );
});

test("public CLI is a fixed no-override route to the static gate", () => {
  assert.match(
    CLI_SOURCE,
    /import \{ acquireEventLogSourceAudit \} from "\.\.\/packages\/vz-runtime\/src\/volvence_zero\/offline_evidence\/windows_event_log_source_audit_acquisition\.mjs";/u,
  );
  assert.equal((CLI_SOURCE.match(/acquireEventLogSourceAudit\(\);/gu) ?? []).length, 1);
  assert.doesNotMatch(CLI_SOURCE, /process\.argv|process\.env|spawn|execFile|child_process/iu);
  assert.doesNotMatch(CLI_SOURCE, /artifactRoot|captureRole|operatorScope|machineIdentity|bootIdentity/iu);
  assert.doesNotMatch(CLI_SOURCE, /-Command|-EncodedCommand|-ExecutionPolicy|AllowSourceCreation|Provision/iu);
  const syntheticStart = MODULE_SOURCE.indexOf(
    "function createSyntheticEventLogSourceAuditAcquisitionArtifact",
  );
  const privateBackendStart = MODULE_SOURCE.indexOf("function fixedPowerShellExecutablePath");
  assert.ok(syntheticStart > 0 && privateBackendStart > syntheticStart);
  assert.doesNotMatch(MODULE_SOURCE.slice(syntheticStart, privateBackendStart), /\bspawn\s*\(/u);
  assert.doesNotMatch(
    MODULE_SOURCE,
    /from\s+["']\.\/windows_cuda_host_stability_qualification\.mjs["']/u,
  );
  assert.doesNotMatch(MODULE_SOURCE, /adaptProvisionerAuditV2Artifact\s*\(/u);
});

test("fixed lifecycle supervision terminalizes kill, pipe, and persistence failures", async (t) => {
  const cutoffScenarios = [
    "kill_false_never_close",
    "kill_throw_never_close",
    "kill_true_never_close",
    "exit_without_close",
    "stdout_pipe_error",
  ];
  for (const scenario of cutoffScenarios) {
    await t.test(scenario, async () => {
      const result =
        await __testing.exerciseFixedAuditLifecycleScenarioForTesting(scenario);
      const observation = result.processObservation;
      assert.ok(result.elapsedMilliseconds < 2000);
      assert.equal(observation.capture_complete, false);
      assert.equal(observation.kill_attempt_count, 1);
      assert.equal(result.killCallCount, 1);
      assert.equal(observation.streams_close_observation, "not_observed");
      assert.equal(observation.streams_closed_at_utc, null);
      assert.equal(observation.capture_detached_at_hard_cutoff, true);
      assert.ok(
        ["post_kill_grace_cutoff", "overall_hard_cutoff"].includes(
          observation.finalization_reason,
        ),
      );
      assert.deepEqual(result.fakeChildListenerCounts, {
        error: 1,
        exit: 0,
        close: 1,
      });
      assert.deepEqual(result.fakeStreamListenerCounts, {
        stdoutError: 1,
        stderrError: 1,
      });
    });
  }

  const exitedWithoutClose =
    await __testing.exerciseFixedAuditLifecycleScenarioForTesting(
      "exit_without_close",
    );
  assert.equal(
    exitedWithoutClose.processObservation.process_exit_observation,
    "child_exit_event",
  );
  assert.equal(exitedWithoutClose.processObservation.termination_confirmed, true);
  assert.equal(exitedWithoutClose.processObservation.process_may_remain_running, false);

  const pipeError =
    await __testing.exerciseFixedAuditLifecycleScenarioForTesting(
      "stdout_pipe_error",
    );
  assert.equal(
    pipeError.processObservation.stream_outcomes.stdout.capture_error_stage,
    "pipe",
  );

  const closeWithoutEnd =
    await __testing.exerciseFixedAuditLifecycleScenarioForTesting(
      "close_without_end",
    );
  assert.equal(closeWithoutEnd.processObservation.finalization_reason, "child_close");
  assert.equal(closeWithoutEnd.processObservation.capture_complete, false);
  for (const outcome of Object.values(
    closeWithoutEnd.processObservation.stream_outcomes,
  )) {
    assert.equal(outcome.pipe_end_observed, false);
    assert.equal(outcome.pipe_close_observed, true);
  }

  const lateError =
    await __testing.exerciseFixedAuditLifecycleScenarioForTesting(
      "late_error_after_cutoff",
    );
  assert.equal(lateError.processObservation.capture_complete, false);
  assert.equal(lateError.processObservation.capture_detached_at_hard_cutoff, true);
  assert.deepEqual(
    lateError.lateErrors.map(({ origin }) => origin),
    ["child process", "stdout pipe", "stderr pipe"],
  );
  for (const observed of lateError.lateErrors) {
    assert.match(observed.messageSha256, /^[0-9a-f]{64}$/u);
  }
  assert.deepEqual(lateError.lateErrorGuardianCountsBeforeClose, {
    child: 1,
    stdout: 1,
    stderr: 1,
  });
  assert.deepEqual(lateError.lateErrorGuardianCountsAfterClose, {
    child: 0,
    stdout: 0,
    stderr: 0,
  });

  for (const [scenario, stage] of [
    ["write_failure", "write"],
    ["fsync_failure", "fsync"],
  ]) {
    const result =
      await __testing.exerciseFixedAuditLifecycleScenarioForTesting(scenario);
    assert.equal(result.processObservation.finalization_reason, "child_close");
    assert.equal(result.processObservation.capture_complete, false);
    assert.equal(
      result.processObservation.stream_outcomes.stdout.persistence_error_stage,
      stage,
    );
    assert.equal(result.processObservation.kill_attempt_count, 0);
  }

  const readbackFailure =
    await __testing.exerciseFixedAuditLifecycleScenarioForTesting(
      "readback_failure",
    );
  assert.ok(readbackFailure.elapsedMilliseconds < 2000);
  assert.equal(readbackFailure.killCallCount, 0);
  assert.equal(readbackFailure.boundedFailure.name, "Error");
  assert.match(readbackFailure.boundedFailure.messageSha256, /^[0-9a-f]{64}$/u);
  assert.equal("processObservation" in readbackFailure, false);

  const clean =
    await __testing.exerciseFixedAuditLifecycleScenarioForTesting("clean_close");
  assert.equal(clean.processObservation.finalization_reason, "child_close");
  assert.equal(clean.processObservation.process_exit_observation, "child_exit_event");
  assert.equal(clean.processObservation.streams_close_observation, "child_close_event");
  assert.equal(clean.processObservation.capture_complete, true);
  assert.equal(Buffer.from(clean.stdoutBase64, "base64").toString("utf8"), "ok");
});

test(
  "Windows descendant-held pipes reach the hard cutoff and the fixture is cleaned",
  { skip: process.platform !== "win32" },
  async () => {
    const result =
      await __testing.exerciseFixedAuditLifecycleScenarioForTesting(
        "descendant_holds_pipe",
      );
    const observation = result.processObservation;
    assert.ok(result.elapsedMilliseconds < 15_000);
    assert.equal(observation.finalization_reason, "overall_hard_cutoff");
    assert.equal(observation.process_exit_observation, "child_exit_event");
    assert.equal(observation.streams_close_observation, "not_observed");
    assert.equal(observation.streams_closed_at_utc, null);
    assert.equal(observation.capture_detached_at_hard_cutoff, true);
    assert.equal(observation.capture_complete, false);
    assert.equal(observation.kill_attempt_count, 1);
    assert.equal(observation.descendants_contained, false);
    assert.equal(result.descendantCleanup.killAccepted, true);
    assert.equal(result.descendantCleanup.terminationConfirmed, true);
  },
);

test(
  "Windows same-buffer launcher preserves Audit semantics and rejects source drift",
  { skip: process.platform !== "win32" },
  async (t) => {
    const valid =
      await __testing.exerciseSourceBindingLauncherFixtureForTesting(
        "valid_exit_2_and_lock",
      );
    assert.equal(valid.processObservation.exit_code, 2);
    assert.equal(valid.processObservation.capture_complete, true);
    assert.equal(valid.stderrUtf8, "");
    const receipt = JSON.parse(valid.stdoutUtf8.trim());
    assert.equal(receipt.Mode, "Audit");
    assert.equal(receipt.Root.replace(/\\/gu, "/"), REPOSITORY_ROOT.replace(/\\/gu, "/"));
    assert.match(receipt.Path, /volvence-source-binding-.+\.ps1$/u);
    assert.equal(valid.renameBlockedWhileHandleHeld, true);
    assert.match(valid.renameBlockErrorCode, /EBUSY|EPERM|EACCES/u);
    assert.equal(valid.handleReleasedAfterExit, true);

    for (const scenario of [
      "normal_return_with_stale_last_exit_code",
      "raw_mismatch_lf_equal",
      "lf_mismatch_raw_equal",
      "utf8_bom",
      "invalid_utf8",
    ]) {
      await t.test(scenario, async () => {
        const rejected =
          await __testing.exerciseSourceBindingLauncherFixtureForTesting(scenario);
        assert.equal(rejected.processObservation.exit_code, 3);
        assert.equal(rejected.processObservation.capture_complete, true);
        assert.equal(rejected.stdoutUtf8, "");
        assert.match(rejected.stderrUtf8, /fixed source-binding launcher failed/u);
        assert.equal(rejected.handleReleasedAfterExit, true);
        if (scenario === "raw_mismatch_lf_equal") {
          assert.notEqual(rejected.sourceRawSha256, rejected.expectedRawSha256);
          assert.equal(
            rejected.sourceLfCanonicalSha256,
            rejected.expectedLfCanonicalSha256,
          );
        } else if (scenario === "lf_mismatch_raw_equal") {
          assert.equal(rejected.sourceRawSha256, rejected.expectedRawSha256);
          assert.notEqual(
            rejected.sourceLfCanonicalSha256,
            rejected.expectedLfCanonicalSha256,
          );
        }
      });
    }
  },
);

test("exit 0 and exit 2 Audit v2 declarations produce non-authoritative adapter envelopes", async (t) => {
  const cases = [
    {
      name: "exit 0 conformant candidate",
      conformant: true,
      role: "qualification_source_audit_before",
      disposition: "audit_v2_conformant_capture_candidate",
    },
    {
      name: "exit 2 nonconformant candidate",
      conformant: false,
      role: "qualification_source_audit_after",
      disposition: "audit_v2_nonconformant_capture_candidate",
    },
  ];
  for (const scenario of cases) {
    await t.test(scenario.name, () => {
      const full = buildFullAuditV2({ conformant: scenario.conformant });
      const exitCode = scenario.conformant ? 0 : 2;
      const created = createSynthetic(t, {
        captureRole: scenario.role,
        outcome: processOutcome({ stdout: full.raw, exitCode }),
      });
      const artifact = readArtifact(created.artifactRoot);
      const outcome = created.validation;

      assert.equal(outcome.disposition, scenario.disposition);
      assert.equal(outcome.captureCandidate, true);
      assert.equal(outcome.quarantined, false);
      assert.deepEqual(Object.keys(outcome.captureEnvelope), CAPTURE_ENVELOPE_KEYS);
      assert.equal(outcome.captureEnvelope.capture_authoritative, false);
      assert.equal(outcome.captureEnvelope.capture_role, scenario.role);
      assert.equal(outcome.captureEnvelope.process_exit_code, exitCode);
      assert.equal(outcome.captureEnvelope.machine_identity_sha256, MACHINE_ID);
      assert.equal(outcome.captureEnvelope.boot_identity_sha256, BOOT_ID);
      assert.equal(outcome.captureEnvelope.stdout_raw_sha256, sha256(full.raw));
      assert.equal(outcome.captureEnvelope.stderr_raw_sha256, sha256(Buffer.alloc(0)));
      assert.deepEqual(outcome.captureEnvelope, artifact.terminal.capture_envelope);
      assert.equal(artifact.terminal.process_observation.spawn_attempted, false);
      assert.equal(artifact.terminal.process_observation.process_id, null);
      assert.equal(
        artifact.terminal.process_observation.process_started_at_semantics,
        "synthetic_declaration",
      );
      assert.equal(artifact.terminal.process_observation.os_process_creation_time_attested, false);
      assert.equal(artifact.terminal.process_observation.descendants_contained, false);
      assertDeepFrozen(outcome);
    });
  }
});

test("exit 3 failure-v1 is quarantined without adapter handoff or projection", (t) => {
  const created = createSynthetic(t, {
    outcome: processOutcome({
      stdout: failureRaw(),
      stderr: Buffer.from("synthetic diagnostic", "utf8"),
      exitCode: 3,
    }),
  });
  const artifact = readArtifact(created.artifactRoot);
  assert.equal(created.validation.disposition, "failure_v1_quarantined");
  assert.equal(created.validation.captureCandidate, false);
  assert.equal(created.validation.quarantined, true);
  assert.equal(created.validation.captureEnvelope, null);
  assert.equal(artifact.terminal.capture_envelope, null);
  assert.equal(created.validation.sourceConfigProjectionEmitted, false);
  assert.equal(created.validation.validatedEligibleAsHostQualificationInput, false);
  assert.equal(created.validation.eligibleAsHostQualificationInput, false);
  assert.equal(artifact.terminal.audit_discriminator.schema_version, FAILURE_SCHEMA);
  assert.equal(artifact.terminal.audit_discriminator.completed, false);
});

test("exact root, 000 claim, raw streams, and 001 terminal lineage fully revalidate", (t) => {
  const raw = auditDiscriminatorRaw();
  const created = createSynthetic(t, { outcome: processOutcome({ stdout: raw }) });
  const artifact = readArtifact(created.artifactRoot);
  const terminalCore = { ...artifact.terminal };
  delete terminalCore.terminal_id;

  assert.deepEqual(listArtifactFiles(created.artifactRoot), EXPECTED_FILES);
  assert.equal(artifact.claim.schema_version, EVENT_LOG_SOURCE_AUDIT_ACQUISITION_CLAIM_SCHEMA_VERSION);
  assert.equal(artifact.claim.sequence, 0);
  assert.equal(artifact.claim.previous_receipt_sha256, null);
  assert.equal(artifact.claim.acquisition_protocol_id, EXPECTED_PROTOCOL_ID);
  assert.equal(artifact.claim.acquisition_protocol_raw_sha256, EXPECTED_PROTOCOL_RAW_SHA256);
  assert.equal(artifact.claim.attempt_number, 1);
  assert.equal(artifact.claim.attempt_budget, 1);
  assert.equal(artifact.claim.retry_budget, 0);
  assert.equal(artifact.claim.attempt_budget_applies_to, "single_artifact_root_only");
  assert.equal(artifact.claim.same_root_overwrite_or_retry_permitted, false);
  assert.equal(artifact.claim.cross_root_duplicate_scope_excluded, false);
  assert.equal(artifact.claim.scope_global_no_retry_proven, false);
  assert.equal(
    artifact.claim.claim_file_descriptor_content_fsync_required_before_process_creation,
    true,
  );
  assert.equal(artifact.claim.directory_entry_durability_guaranteed, false);
  assert.equal(artifact.claim.identity_binding_authoritative, false);
  assert.equal(artifact.claim.invocation.backend_id, PROTOCOL.execution.synthetic_backend_id);
  assert.equal(
    artifact.claim.invocation.executable_template,
    PROTOCOL.execution.executable_template,
  );
  assert.equal(
    artifact.claim.invocation.requested_executable,
    "synthetic://windows-powershell-5.1",
  );
  assert.deepEqual(artifact.claim.invocation.argv_template, PROTOCOL.execution.argv_template);
  const launcher = __testing.frozenSourceBindingLauncherObservationForTesting();
  assert.deepEqual(artifact.claim.invocation.requested_argv, [
    "-NoLogo",
    "-NoProfile",
    "-NonInteractive",
    "-EncodedCommand",
    launcher.encodedCommand,
  ]);
  assert.deepEqual(
    artifact.claim.invocation.source_execution_binding,
    PROTOCOL.execution.source_execution_binding,
  );
  assert.equal(artifact.claim.invocation.cwd, REPOSITORY_ROOT);
  assert.equal(artifact.claim.invocation.invocation_realization, "synthetic_not_realized");
  assert.equal(artifact.claim.invocation.environment_inherited, false);
  assert.equal(
    artifact.claim.scope_id,
    domainSeparatedSha256(PROTOCOL.scope.domain_separator, [
      EXPECTED_PROTOCOL_ID,
      PROTOCOL.qualification_compatibility.qualification_protocol_id,
      OPERATOR_SCOPE_ID,
      MACHINE_ID,
      BOOT_ID,
      "qualification_source_audit_before",
      PROTOCOL.execution.synthetic_backend_id,
    ]),
  );
  assert.equal(artifact.terminal.schema_version, EVENT_LOG_SOURCE_AUDIT_ACQUISITION_TERMINAL_SCHEMA_VERSION);
  assert.equal(artifact.terminal.sequence, 1);
  assert.equal(artifact.terminal.previous_receipt_sha256, sha256(artifact.claimRaw));
  assert.equal(artifact.terminal.terminal_id, contentId(terminalCore));
  assert.equal(artifact.terminal.stream_capture.stdout_raw_sha256, sha256(artifact.stdout));
  assert.equal(artifact.terminal.stream_capture.stdout_byte_count, artifact.stdout.length);
  assert.equal(artifact.terminal.stream_capture.stderr_raw_sha256, sha256(artifact.stderr));
  assert.equal(artifact.terminal.stream_capture.stderr_byte_count, artifact.stderr.length);
  assert.equal(
    artifact.terminal.process_observation.schema_version,
    PROTOCOL.output_contract.process_observation_schema_version,
  );
  assert.equal(
    artifact.terminal.process_observation.process_exit_observation,
    "synthetic_declaration",
  );
  assert.equal(
    artifact.terminal.process_observation.streams_close_observation,
    "synthetic_declaration",
  );
  assert.equal(artifact.terminal.process_observation.capture_complete, true);
  assert.equal(created.validation.terminalId, artifact.terminal.terminal_id);
  assert.equal(created.validation.terminalRawSha256, sha256(artifact.terminalRaw));
  assert.equal(created.validation.fullRootIdentity, sha256(artifact.terminalRaw));
  assert.equal(
    created.validation.claimFileDescriptorContentFsyncRequiredBeforeProcessCreation,
    true,
  );
  assert.equal(created.validation.directoryEntryDurabilityGuaranteed, false);
  assert.equal(created.validation.pathParentReparseTrustProven, false);
  assert.deepEqual(
    validateEventLogSourceAuditAcquisitionArtifact({ artifactRoot: created.artifactRoot }),
    created.validation,
  );
});

test("synthetic success never upgrades real, qualification, CUDA, ACTIVE, or four-axis claims", (t) => {
  const created = createSynthetic(t);
  const { claim, terminal } = readArtifact(created.artifactRoot);
  const outcome = created.validation;
  const falseOutcomeKeys = [
    "captureEnvelopeAuthoritative",
    "realProcessObservation",
    "realProvisionerObservation",
    "sourceConfigProjectionEmitted",
    "validatedEligibleAsHostQualificationInput",
    "eligibleAsHostQualificationInput",
    "qualificationAuthorized",
    "cudaExecutionAuthorized",
    "formalEvidenceAuthorized",
    "productionActiveAuthorized",
    "appendableProven",
    "readableProven",
    "learnableProven",
    "steerableProven",
    "continuousEndpointStabilityProven",
    "fourCapabilityClaimAuthorized",
    "tamperResistanceProven",
    "sameRootOverwriteOrRetryPermitted",
    "crossRootDuplicateScopeExcluded",
    "scopeGlobalNoRetryProven",
    "directoryEntryDurabilityGuaranteed",
    "pathParentReparseTrustProven",
  ];
  for (const key of falseOutcomeKeys) assert.equal(outcome[key], false, key);
  const falseFirewallKeys = [
    "capture_envelope_authoritative",
    "real_process_observation",
    "real_provisioner_observation",
    "source_config_projection_emitted",
    "eligible_as_host_qualification_input",
    "qualification_authorized",
    "cuda_execution_authorized",
    "formal_evidence_authorized",
    "production_active_authorized",
    "appendable_proven",
    "readable_proven",
    "learnable_proven",
    "steerable_proven",
    "four_capability_claim_authorized",
    "tamper_resistance_proven",
    "continuous_endpoint_stability_proven",
  ];
  for (const firewall of [claim.evidence_firewall, terminal.evidence_firewall]) {
    for (const key of falseFirewallKeys) assert.equal(firewall[key], false, key);
  }
  assert.equal(terminal.capture_envelope.capture_authoritative, false);
  assert.equal(terminal.continuous_endpoint_stability_proven, false);
  assertDeepFrozen(outcome);
});

test("schema, exit, stderr, and hostile stdout combinations are quarantined", async (t) => {
  const audit0 = auditDiscriminatorRaw();
  const audit2 = auditDiscriminatorRaw({ exitCode: 2, overallConformant: false });
  const scenarios = [
    {
      name: "exit 0 carrying an exit 2 Audit receipt",
      outcome: processOutcome({ stdout: audit2, exitCode: 0 }),
    },
    {
      name: "exit 2 carrying an exit 0 Audit receipt",
      outcome: processOutcome({ stdout: audit0, exitCode: 2 }),
    },
    {
      name: "exit 3 carrying Audit v2 instead of failure-v1",
      outcome: processOutcome({ stdout: audit0, exitCode: 3 }),
    },
    {
      name: "exit 0 carrying failure-v1",
      outcome: processOutcome({ stdout: failureRaw(), exitCode: 0 }),
    },
    {
      name: "candidate with nonempty stderr",
      outcome: processOutcome({ stdout: audit0, stderr: "diagnostic" }),
    },
    {
      name: "empty stdout",
      outcome: processOutcome({ stdout: Buffer.alloc(0) }),
    },
    {
      name: "invalid UTF-8 stdout",
      outcome: processOutcome({ stdout: Buffer.from([0xc3, 0x28]) }),
    },
    {
      name: "UTF-8 BOM stdout",
      outcome: processOutcome({ stdout: Buffer.concat([Buffer.from([0xef, 0xbb, 0xbf]), audit0]) }),
    },
    {
      name: "duplicate JSON key stdout",
      outcome: processOutcome({
        stdout: `{"schema_version":"${AUDIT_SCHEMA}","schema_version":"${AUDIT_SCHEMA}","mode":"Audit","process_exit_code":0,"overall_conformant":true}\n`,
      }),
    },
    {
      name: "float lexical form stdout",
      outcome: processOutcome({
        stdout: `{"schema_version":"${AUDIT_SCHEMA}","mode":"Audit","process_exit_code":0.0,"overall_conformant":true}\n`,
      }),
    },
    {
      name: "unsafe integer stdout",
      outcome: processOutcome({
        stdout: `{"schema_version":"${AUDIT_SCHEMA}","mode":"Audit","process_exit_code":9007199254740992,"overall_conformant":true}\n`,
      }),
    },
    {
      name: "trailing JSON content stdout",
      outcome: processOutcome({ stdout: `${audit0.toString("utf8").trimEnd()} trailing\n` }),
    },
    {
      name: "multiple JSON lines stdout",
      outcome: processOutcome({ stdout: Buffer.concat([audit0, audit0]) }),
    },
    {
      name: "wrong mode stdout",
      outcome: processOutcome({ stdout: auditDiscriminatorRaw({ mode: "Provision" }) }),
    },
    {
      name: "exit 0 with false conformance",
      outcome: processOutcome({
        stdout: auditDiscriminatorRaw({ exitCode: 0, overallConformant: false }),
      }),
    },
    {
      name: "exit 2 with true conformance",
      outcome: processOutcome({
        stdout: auditDiscriminatorRaw({ exitCode: 2, overallConformant: true }),
        exitCode: 2,
      }),
    },
  ];

  for (const scenario of scenarios) {
    await t.test(scenario.name, () => {
      const created = createSynthetic(t, { outcome: scenario.outcome });
      const artifact = readArtifact(created.artifactRoot);
      assert.equal(
        created.validation.disposition,
        "unclassified_process_or_output_quarantined",
      );
      assert.equal(created.validation.captureCandidate, false);
      assert.equal(created.validation.quarantined, true);
      assert.equal(created.validation.captureEnvelope, null);
      assert.equal(artifact.terminal.capture_envelope, null);
      assert.equal(created.validation.sourceConfigProjectionEmitted, false);
      assert.equal(created.validation.eligibleAsHostQualificationInput, false);
      assertDeepFrozen(created.validation);
    });
  }
});

test("bounded overflow is quarantined and an actually oversized declaration is never written", (t) => {
  const maximum = PROTOCOL.budgets.stdout_max_bytes;
  const bounded = createSynthetic(t, {
    leaf: "bounded-overflow",
    outcome: processOutcome({
      stdout: Buffer.alloc(maximum, 0x78),
      exitCode: null,
      signal: "SIGTERM",
      overflowStream: "stdout",
      killAttempted: true,
      killAttemptCount: 1,
    }),
  });
  const boundedArtifact = readArtifact(bounded.artifactRoot);
  assert.equal(bounded.validation.quarantined, true);
  assert.equal(bounded.validation.captureEnvelope, null);
  assert.equal(bounded.validation.streamCapture.stdout_byte_count, maximum);
  assert.equal(boundedArtifact.terminal.process_observation.overflow_stream, "stdout");
  assert.equal(boundedArtifact.terminal.process_observation.kill_attempt_count, 1);
  assert.equal(boundedArtifact.terminal.process_observation.kill_request_accepted, false);

  const oversizedRoot = temporaryArtifactRoot(t, "oversized");
  assert.throws(
    () =>
      __testing.createSyntheticEventLogSourceAuditAcquisitionArtifact({
        artifactRoot: oversizedRoot,
        captureRole: "qualification_source_audit_before",
        operatorScopeBindingId: OPERATOR_SCOPE_ID,
        machineIdentitySha256: MACHINE_ID,
        bootIdentitySha256: BOOT_ID,
        processOutcome: processOutcome({ stdout: Buffer.alloc(maximum + 1, 0x78) }),
      }),
    /exceeds the frozen bounded capture size/i,
  );
  assert.equal(fs.existsSync(oversizedRoot), false);
});

test("same root is create-only, partial roots are not retried, and no synthetic route launches a backend", (t) => {
  const directOptions = (artifactRoot) => ({
    artifactRoot,
    captureRole: "qualification_source_audit_before",
    operatorScopeBindingId: OPERATOR_SCOPE_ID,
    machineIdentitySha256: MACHINE_ID,
    bootIdentitySha256: BOOT_ID,
    processOutcome: processOutcome(),
  });
  const completeRoot = temporaryArtifactRoot(t, "complete");
  __testing.createSyntheticEventLogSourceAuditAcquisitionArtifact(directOptions(completeRoot));
  const before = Object.fromEntries(
    EXPECTED_FILES.map((relative) => [
      relative,
      sha256(fs.readFileSync(path.join(completeRoot, ...relative.split("/")))),
    ]),
  );
  assert.throws(
    () => __testing.createSyntheticEventLogSourceAuditAcquisitionArtifact(directOptions(completeRoot)),
    /root already exists/i,
  );
  const after = Object.fromEntries(
    EXPECTED_FILES.map((relative) => [
      relative,
      sha256(fs.readFileSync(path.join(completeRoot, ...relative.split("/")))),
    ]),
  );
  assert.deepEqual(after, before);

  const partialRoot = temporaryArtifactRoot(t, "partial");
  fs.mkdirSync(partialRoot);
  fs.writeFileSync(path.join(partialRoot, "operator-marker.txt"), "untouched", { flag: "wx" });
  assert.throws(
    () => __testing.createSyntheticEventLogSourceAuditAcquisitionArtifact(directOptions(partialRoot)),
    /root already exists/i,
  );
  assert.deepEqual(fs.readdirSync(partialRoot), ["operator-marker.txt"]);
  assert.equal(fs.readFileSync(path.join(partialRoot, "operator-marker.txt"), "utf8"), "untouched");

  const absentParent = temporaryArtifactRoot(t, "absent-direct-parent");
  const childRoot = path.join(absentParent, "child");
  assert.throws(
    () => __testing.createSyntheticEventLogSourceAuditAcquisitionArtifact(directOptions(childRoot)),
    /ENOENT|no such file|parent.*preexist|parent.*directory/i,
  );
  assert.equal(fs.existsSync(absentParent), false);

  const completeTerminal = readArtifact(completeRoot).terminal;
  assert.equal(completeTerminal.process_observation.spawn_attempted, false);
  assert.equal(completeTerminal.process_observation.process_id, null);
  assert.equal(completeTerminal.execution_backend_id, PROTOCOL.execution.synthetic_backend_id);
});

test("same scope can currently exist in distinct roots and is not misreported as globally excluded", (t) => {
  const first = createSynthetic(t, { leaf: "first-root" });
  const second = createSynthetic(t, { leaf: "second-root" });
  assert.notEqual(first.artifactRoot, second.artifactRoot);
  assert.equal(first.validation.scopeId, second.validation.scopeId);
  assert.equal(PROTOCOL.scope.cross_root_duplicate_scope_excluded, false);
  assert.equal(first.validation.crossRootDuplicateScopeExcluded, false);
  assert.equal(second.validation.crossRootDuplicateScopeExcluded, false);
  assert.equal(first.validation.scopeGlobalNoRetryProven, false);
  assert.equal(second.validation.scopeGlobalNoRetryProven, false);
  assert.equal(first.validation.sameRootOverwriteOrRetryPermitted, false);
  assert.equal(first.validation.directoryEntryDurabilityGuaranteed, false);
  assert.equal(first.validation.pathParentReparseTrustProven, false);
});

test("full validator rejects extra members, stream mutation, claim drift, and terminal ID drift", async (t) => {
  await t.test("extra root member", () => {
    const created = createSynthetic(t);
    fs.writeFileSync(path.join(created.artifactRoot, "extra.txt"), "extra", { flag: "wx" });
    assert.throws(
      () => validateEventLogSourceAuditAcquisitionArtifact({ artifactRoot: created.artifactRoot }),
      /top-level entries.*drift/i,
    );
  });
  await t.test("captured stdout mutation", () => {
    const created = createSynthetic(t);
    fs.appendFileSync(path.join(created.artifactRoot, "streams/audit.stdout.bin"), "x");
    assert.throws(
      () => validateEventLogSourceAuditAcquisitionArtifact({ artifactRoot: created.artifactRoot }),
      /stream_capture|stream capture|drift/i,
    );
  });
  await t.test("claim acquisition protocol drift", () => {
    const created = createSynthetic(t);
    const claimPath = path.join(created.artifactRoot, "000_scope_claim.json");
    const claim = JSON.parse(fs.readFileSync(claimPath, "utf8"));
    claim.acquisition_protocol_id = "f".repeat(64);
    fs.writeFileSync(claimPath, `${canonicalJson(claim)}\n`);
    assert.throws(
      () => validateEventLogSourceAuditAcquisitionArtifact({ artifactRoot: created.artifactRoot }),
      /claim fixed contract drift/i,
    );
  });
  await t.test("terminal content ID drift", () => {
    const created = createSynthetic(t);
    const terminalPath = path.join(created.artifactRoot, "001_terminal.json");
    const terminal = JSON.parse(fs.readFileSync(terminalPath, "utf8"));
    terminal.terminal_id = "f".repeat(64);
    fs.writeFileSync(terminalPath, `${canonicalJson(terminal)}\n`);
    assert.throws(
      () => validateEventLogSourceAuditAcquisitionArtifact({ artifactRoot: created.artifactRoot }),
      /terminal ID drift/i,
    );
  });
});

test("source or executable endpoint drift forces quarantine and removes the adapter envelope", async (t) => {
  for (const endpoint of ["source", "executable"]) {
    await t.test(`${endpoint} endpoint drift`, () => {
      const created = createSynthetic(t);
      rewriteTerminal(created.artifactRoot, (terminal) => {
        terminal[`${endpoint}_endpoint_after`].raw_sha256 = "d".repeat(64);
        terminal[`${endpoint}_endpoint_equal`] = false;
        terminal.disposition = "unclassified_process_or_output_quarantined";
        terminal.capture_candidate = false;
        terminal.quarantined = true;
        terminal.capture_envelope = null;
      });
      const validation = validateEventLogSourceAuditAcquisitionArtifact({
        artifactRoot: created.artifactRoot,
      });
      assert.equal(validation.disposition, "unclassified_process_or_output_quarantined");
      assert.equal(validation.captureCandidate, false);
      assert.equal(validation.quarantined, true);
      assert.equal(validation.captureEnvelope, null);
      assert.equal(validation.continuousEndpointStabilityProven, false);
    });
  }
});

test("timeout, overflow, spawn failure, and kill-once declarations remain quarantined", async (t) => {
  const cases = [
    {
      name: "timeout",
      outcome: processOutcome({
        exitCode: null,
        signal: "SIGTERM",
        timedOut: true,
        killAttempted: true,
        killAttemptCount: 1,
      }),
    },
    {
      name: "stderr overflow",
      outcome: processOutcome({
        exitCode: null,
        signal: "SIGTERM",
        overflowStream: "stderr",
        killAttempted: true,
        killAttemptCount: 1,
      }),
    },
    {
      name: "spawn failure",
      outcome: processOutcome({
        exitCode: null,
        spawnErrorName: "SyntheticSpawnError",
        spawnErrorMessage: "declarative only",
      }),
    },
  ];
  for (const scenario of cases) {
    await t.test(scenario.name, () => {
      const created = createSynthetic(t, { outcome: scenario.outcome });
      const artifact = readArtifact(created.artifactRoot);
      assert.equal(created.validation.quarantined, true);
      assert.equal(created.validation.captureEnvelope, null);
      assert.equal(artifact.terminal.capture_envelope, null);
      assert.equal(artifact.terminal.process_observation.spawn_attempted, false);
      assert.equal(artifact.terminal.process_observation.descendants_contained, false);
    });
  }

  const missingKillRoot = temporaryArtifactRoot(t, "missing-kill");
  assert.throws(
    () =>
      __testing.createSyntheticEventLogSourceAuditAcquisitionArtifact({
        artifactRoot: missingKillRoot,
        captureRole: "qualification_source_audit_before",
        operatorScopeBindingId: OPERATOR_SCOPE_ID,
        machineIdentitySha256: MACHINE_ID,
        bootIdentitySha256: BOOT_ID,
        processOutcome: processOutcome({ timedOut: true }),
      }),
    /must attempt exactly one kill/i,
  );
  assert.equal(fs.existsSync(missingKillRoot), false);

  const doubleKillRoot = temporaryArtifactRoot(t, "double-kill");
  assert.throws(
    () =>
      __testing.createSyntheticEventLogSourceAuditAcquisitionArtifact({
        artifactRoot: doubleKillRoot,
        captureRole: "qualification_source_audit_before",
        operatorScopeBindingId: OPERATOR_SCOPE_ID,
        machineIdentitySha256: MACHINE_ID,
        bootIdentitySha256: BOOT_ID,
        processOutcome: processOutcome({
          timedOut: true,
          killAttempted: true,
          killAttemptCount: 2,
        }),
      }),
    /kill-once contract drift/i,
  );
  assert.equal(fs.existsSync(doubleKillRoot), false);
});

test("exit 0 and exit 2 acquisition envelopes hand the same stdout bytes to the pure adapter", async (t) => {
  const cases = [
    {
      name: "before exit 0",
      conformant: true,
      captureRole: "qualification_source_audit_before",
    },
    {
      name: "after exit 2",
      conformant: false,
      captureRole: "qualification_source_audit_after",
    },
  ];
  for (const scenario of cases) {
    await t.test(scenario.name, () => {
      const full = buildFullAuditV2({ conformant: scenario.conformant });
      const exitCode = scenario.conformant ? 0 : 2;
      const created = createSynthetic(t, {
        captureRole: scenario.captureRole,
        outcome: processOutcome({ stdout: full.raw, exitCode }),
      });
      assert.notEqual(created.validation.captureEnvelope, null);
      const adapterSnapshot = adaptProvisionerAuditV2Artifact({
        artifactRoot: created.artifactRoot,
        auditRelativePath: created.validation.streamCapture.stdout_relative_path,
        captureEnvelope: created.validation.captureEnvelope,
        expectedProtocolId: PROTOCOL.qualification_compatibility.qualification_protocol_id,
        expectedProtocolRawSha256:
          PROTOCOL.qualification_compatibility.qualification_protocol_raw_sha256,
        expectedMachineConfigContentId:
          scenario.captureRole === "qualification_source_audit_after"
            ? full.receipt.machine_config_content_id
            : null,
      });

      assert.equal(adapterSnapshot.raw_audit.raw_sha256, sha256(full.raw));
      assert.equal(adapterSnapshot.raw_audit.process_exit_code, exitCode);
      assert.equal(adapterSnapshot.raw_audit.overall_conformant, scenario.conformant);
      assert.equal(adapterSnapshot.verification.full_raw_audit_bound, true);
      assert.equal(adapterSnapshot.verification.raw_audit_content_id_basis_revalidated, true);
      assert.equal(adapterSnapshot.boundary.projection_emitted, false);
      assert.equal(adapterSnapshot.boundary.real_provisioner_observation, false);
      assert.equal(adapterSnapshot.boundary.eligible_as_host_qualification_input, false);
      assert.equal(adapterSnapshot.boundary.cuda_execution_authorized, false);
      assert.equal(adapterSnapshot.boundary.formal_evidence_authorized, false);
      assert.equal(adapterSnapshot.boundary.production_active_authorized, false);
      assert.equal(adapterSnapshot.boundary.four_capability_claim_authorized, false);
      assertDeepFrozen(adapterSnapshot);
    });
  }
});
