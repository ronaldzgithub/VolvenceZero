import assert from "node:assert/strict";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  __testing,
  adaptProvisionerAuditV2Artifact,
  EVENT_LOG_SOURCE_AUDIT_ADAPTER_SNAPSHOT_SCHEMA_VERSION,
  preregisterHostQualification,
  runHostQualification,
  validateHostQualification,
  validateSyntheticQualificationArtifact,
} from "../src/volvence_zero/offline_evidence/windows_cuda_host_stability_qualification.mjs";

const REPOSITORY_ROOT = path.resolve(import.meta.dirname, "../../../");
const MODULE_RELATIVE_PATH =
  "packages/vz-runtime/src/volvence_zero/offline_evidence/windows_cuda_host_stability_qualification.mjs";
const PROVISIONER_RELATIVE_PATH =
  "packages/vz-runtime/src/volvence_zero/offline_evidence/provision_volvence_evidence_event_log.ps1";
const HOST_BLOCK_RELATIVE_PATH =
  "artifacts/relationship_lab/p4_windows_cuda_physical_residual_actuation_host_block_windows_20260822.json";
const PROTOCOL_PATH = path.join(
  REPOSITORY_ROOT,
  "packages/vz-runtime/src/volvence_zero/offline_evidence/protocols/windows_cuda_host_stability_qualification_v1.json",
);
const EXPECTED_PROTOCOL_ID =
  "32f35e4f7027e9519522e099efb696fb352a48faf3ba69be861929304fae1d5f";
const EXPECTED_PROTOCOL_RAW_SHA256 =
  "30a881838b41fa5b7e6de5aba6bc94131245796126be5b49c4ebab539f8c4132";
const EXPECTED_MODULE_LF_SHA256 =
  "7efff6c353d147f994a1e431903bb1ccb8772e89b7a99753fede5fd3434172e7";
const EXPECTED_PROVISIONER_LF_SHA256 =
  "be0c02f136761f83412f31cdbf1f3249ad7ed15de1aff28e27fe1a8597888406";
const EXPECTED_HOST_BLOCK_RAW_SHA256 =
  "5e02aec731db429fa699176edd8cd6cf44c52e68193c6a0d22c32112c8c4a34f";
const EXPECTED_CURRENT_OUTER_PROTOCOL_ID =
  "cf62484fccdcaf71e6db1f2f0b6d9034443ded15891a8f41fa73b638e7bd3194";

function sha256(raw) {
  return crypto.createHash("sha256").update(raw).digest("hex");
}

function canonicalJson(value) {
  if (value === null) return "null";
  if (typeof value === "boolean" || typeof value === "number") {
    return JSON.stringify(value);
  }
  if (typeof value === "string") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  return `{${Object.keys(value)
    .sort()
    .map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`)
    .join(",")}}`;
}

function lfCanonicalSha256(filePath) {
  const source = fs.readFileSync(filePath, "utf8").replace(/\r\n?/gu, "\n");
  return sha256(Buffer.from(source, "utf8"));
}

function temporaryOutput(t, leaf = "qualification") {
  const temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), "volvence-host-qualification-"));
  t.after(() => fs.rmSync(temporaryRoot, { recursive: true, force: true }));
  return path.join(temporaryRoot, leaf);
}

function createSynthetic(t, options = {}) {
  return __testing.createSyntheticQualificationArtifact({
    outputRoot: temporaryOutput(t),
    protocolPath: PROTOCOL_PATH,
    repositoryRoot: REPOSITORY_ROOT,
    verifySources: true,
    ...options,
  });
}

function validateSynthetic(created) {
  return validateSyntheticQualificationArtifact({
    qualificationRoot: created.qualificationRoot,
    protocolPath: PROTOCOL_PATH,
    repositoryRoot: REPOSITORY_ROOT,
    verifySources: true,
  });
}

function cloneJson(value) {
  return JSON.parse(JSON.stringify(value));
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

function createRawAuditFixture(t, { conformant = true } = {}) {
  const protocol = JSON.parse(fs.readFileSync(PROTOCOL_PATH, "utf8"));
  const fixedContract = cloneJson(protocol.event_log_source.required_fixed_contract);
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
    machine_identity_sha256: "a".repeat(64),
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
    schema_version: "volvence-evidence-event-log-provisioning-audit.v2",
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
  const raw = Buffer.from(`${JSON.stringify(receipt)}\n`, "utf8");
  const artifactRoot = temporaryOutput(t, "raw-audit");
  fs.mkdirSync(artifactRoot);
  const auditRelativePath = "audit.json";
  fs.writeFileSync(path.join(artifactRoot, auditRelativePath), raw, { flag: "wx" });
  const captureEnvelope = {
    schema_version: "windows-event-log-source-audit-capture-envelope.v1",
    capture_role: "qualification_source_audit_before",
    stdout_raw_sha256: sha256(raw),
    stdout_byte_count: raw.length,
    stderr_raw_sha256: sha256(Buffer.alloc(0)),
    stderr_byte_count: 0,
    process_exit_code: exitCode,
    process_started_at_utc: "2026-08-22T00:00:00.0000000Z",
    process_exited_at_utc: "2026-08-22T00:00:03.0000000Z",
    stdout_captured_at_utc: "2026-08-22T00:00:04.0000000Z",
    machine_identity_sha256: machine.machine_identity_sha256,
    boot_identity_sha256: "b".repeat(64),
    capture_authoritative: false,
  };
  return {
    receipt,
    raw,
    options: {
      artifactRoot,
      auditRelativePath,
      captureEnvelope,
      expectedProtocolId: EXPECTED_PROTOCOL_ID,
      expectedProtocolRawSha256: EXPECTED_PROTOCOL_RAW_SHA256,
      expectedMachineConfigContentId: null,
    },
  };
}

function replaceRawAuditFixture(t, fixture, raw, capturePatch = {}) {
  const artifactRoot = temporaryOutput(t, "raw-audit-rewrite");
  fs.mkdirSync(artifactRoot);
  const auditRelativePath = "audit.json";
  fs.writeFileSync(path.join(artifactRoot, auditRelativePath), raw, { flag: "wx" });
  return {
    ...fixture.options,
    artifactRoot,
    auditRelativePath,
    captureEnvelope: {
      ...fixture.options.captureEnvelope,
      stdout_raw_sha256: sha256(raw),
      stdout_byte_count: raw.length,
      ...capturePatch,
    },
  };
}

function assertDeepFrozen(value) {
  if (value === null || typeof value !== "object") return;
  assert.equal(Object.isFrozen(value), true);
  for (const item of Object.values(value)) assertDeepFrozen(item);
}

function assertStaticProductionGate(entrypoint) {
  let accessCount = 0;
  const poison = new Proxy(
    {},
    {
      get() {
        accessCount += 1;
        throw new Error("options getter was evaluated");
      },
      getOwnPropertyDescriptor() {
        accessCount += 1;
        throw new Error("options descriptor was evaluated");
      },
      has() {
        accessCount += 1;
        throw new Error("options membership was evaluated");
      },
      ownKeys() {
        accessCount += 1;
        throw new Error("options keys were evaluated");
      },
    },
  );
  assert.throws(() => entrypoint(poison), /production.*disabled|disabled.*production/i);
  assert.equal(accessCount, 0, "the disabled gate must precede every option/path read");
}

test("protocol freezes source pins and remains fail-closed", () => {
  const protocolRaw = fs.readFileSync(PROTOCOL_PATH);
  const protocol = JSON.parse(protocolRaw.toString("utf8"));
  assert.equal(sha256(protocolRaw), EXPECTED_PROTOCOL_RAW_SHA256);
  assert.equal(sha256(Buffer.from(canonicalJson(protocol), "utf8")), EXPECTED_PROTOCOL_ID);
  assert.equal(
    protocol.schema_version,
    "windows-cuda-host-stability-qualification-protocol.v1",
  );
  assert.equal(protocol.probe.production_entrypoints_enabled, false);
  assert.equal(protocol.probe.production_probe_implemented, false);
  assert.equal(
    protocol.probe.synthetic_test_backend_id,
    "synthetic-node-host-qualification-test-double.v1",
  );
  assert.equal(protocol.probe.synthetic_artifacts_are_real_host_observations, false);
  assert.equal(protocol.probe.synthetic_artifacts_can_be_eligible, false);
  assert.equal(
    protocol.consumer_compatibility.current_outer_protocol_id,
    EXPECTED_CURRENT_OUTER_PROTOCOL_ID,
  );
  assert.equal(
    protocol.consumer_compatibility.published_terminal_schema_version,
    "windows-cuda-host-stability-qualification-terminal.v2",
  );
  assert.equal(
    protocol.consumer_compatibility.current_outer_accepts_published_terminal,
    false,
  );
  assert.equal(
    protocol.consumer_compatibility.terminal_eligibility_self_report_authoritative,
    false,
  );
  assert.equal(protocol.consumer_compatibility.only_full_validator_return_can_be_consumed, true);
  assert.equal(protocol.firmware_gate.minimum_microcode_revision_integer, 303);
  assert.equal(
    protocol.event_log_source.provisioning_audit_schema_version,
    "volvence-evidence-event-log-provisioning-audit.v2",
  );
  assert.equal(
    protocol.event_log_source.qualification_projection_schema_version,
    "windows-cuda-host-stability-source-audit-projection.v2",
  );
  assert.equal(protocol.event_log_source.audit_overall_conformant_required, true);
  assert.equal(protocol.event_log_source.audit_nonconformance_exit_code, 2);
  assert.equal(protocol.event_log_source.audit_exit_code_alone_proves_conformance, false);
  assert.equal(
    protocol.event_log_source
      .refresh_requirement_field_alone_proves_service_refresh_or_cold_boot,
    false,
  );
  assert.equal(protocol.event_log_source.machine_config_content_id_alone_proves_conformance, false);
  assert.equal(protocol.event_log_source.continuous_stability_proven_required, false);
  assert.equal(
    protocol.event_log_source.required_provider_membership_transition_disposition,
    "unchanged",
  );
  assert.equal(protocol.event_log_source.module_qualification_proves_trusted_execution, false);
  assert.equal(protocol.event_log_source.synthetic_projection_matches_provisioner_raw_schema, false);
  assert.equal(protocol.event_log_source.raw_audit_artifact_adapter_core_implemented, true);
  assert.equal(
    protocol.event_log_source
      .production_qualification_receipt_raw_audit_binding_implemented,
    false,
  );
  assert.equal(protocol.event_log_source.production_raw_audit_acquisition_implemented, false);
  assert.equal(
    protocol.event_log_source.independent_control_plane_reobservation_implemented,
    false,
  );
  assert.equal(
    protocol.event_log_source.production_requires_full_raw_audit_binding_and_independent_revalidation,
    true,
  );
  assert.equal(protocol.evidence_firewall.synthetic_test_backend_non_evidence, true);
  assert.equal(protocol.evidence_firewall.synthetic_validated_eligibility_always_false, true);
  assert.equal(protocol.evidence_firewall.cuda_execution_authorized, false);
  assert.equal(protocol.evidence_firewall.raw_audit_artifact_adapter_core_present, true);
  assert.equal(protocol.evidence_firewall.production_raw_audit_adapter_present, false);
  assert.equal(protocol.evidence_firewall.raw_event_xml_independently_parsed, false);
  assert.equal(protocol.evidence_firewall.four_capability_claim_authorized, false);
  assert.deepEqual(protocol.output_contract.complete_files, [
    "000_scope_claim.json",
    "001_preregistration.json",
    "002_source_audit_before.json",
    "003_event_log_baseline.json",
    "004_probe_launch.json",
    "005_probe_exit.json",
    "006_process_window_delta.json",
    "007_cooldown_delta.json",
    "008_handoff_and_source_audit_after.json",
    "009_qualification_report.json",
    "010_manifest.json",
    "011_terminal.json",
    "streams/probe.stdout.log",
    "streams/probe.stderr.log",
  ]);

  assert.equal(protocol.source_sha256[MODULE_RELATIVE_PATH], EXPECTED_MODULE_LF_SHA256);
  assert.equal(
    protocol.source_sha256[MODULE_RELATIVE_PATH],
    lfCanonicalSha256(path.join(REPOSITORY_ROOT, MODULE_RELATIVE_PATH)),
  );
  assert.equal(protocol.source_sha256[PROVISIONER_RELATIVE_PATH], EXPECTED_PROVISIONER_LF_SHA256);
  assert.equal(
    lfCanonicalSha256(path.join(REPOSITORY_ROOT, PROVISIONER_RELATIVE_PATH)),
    EXPECTED_PROVISIONER_LF_SHA256,
  );
  assert.equal(protocol.host_block.receipt_raw_sha256, EXPECTED_HOST_BLOCK_RAW_SHA256);
  assert.equal(
    sha256(fs.readFileSync(path.join(REPOSITORY_ROOT, HOST_BLOCK_RELATIVE_PATH))),
    EXPECTED_HOST_BLOCK_RAW_SHA256,
  );
});

test("all public production entrypoints throw before reading options or paths", () => {
  assertStaticProductionGate(preregisterHostQualification);
  assertStaticProductionGate(runHostQualification);
  assertStaticProductionGate(validateHostQualification);

  for (const entrypoint of [
    preregisterHostQualification,
    runHostQualification,
    validateHostQualification,
  ]) {
    assert.throws(
      () => entrypoint({ qualificationRoot: "Z:\\must-not-be-read\\missing" }),
      /production.*disabled|disabled.*production/i,
    );
  }
});

test("a valid synthetic root has full integrity but can never qualify the host", (t) => {
  const created = createSynthetic(t);
  const validated = validateSynthetic(created);
  assert.equal(validated.integrityValid, true);
  assert.equal(validated.sourceLineageVerified, true);
  assert.equal(validated.artifactId, created.artifactId);
  assert.equal(validated.terminalId, created.terminalId);
  assert.equal(validated.terminalRawSha256, created.terminalRawSha256);
  assert.equal(validated.criteriaPassed, false);
  assert.equal(validated.realHostObservation, false);
  assert.equal(validated.validatedEligibleAsHostQualificationInput, false);
  assert.deepEqual(validated.failureCodes, ["synthetic_test_backend_not_evidence"]);

  const terminal = JSON.parse(
    fs.readFileSync(path.join(created.qualificationRoot, "011_terminal.json"), "utf8"),
  );
  assert.equal(
    terminal.schema_version,
    "windows-cuda-host-stability-qualification-terminal.v2",
  );
  assert.equal(Object.hasOwn(terminal, "passed"), false);
  assert.equal(Object.hasOwn(terminal, "real_cuda_evidence_authorized"), false);
  assert.equal(terminal.criteria_passed, false);
  assert.equal(terminal.real_host_observation, false);
  assert.equal(terminal.eligible_as_host_qualification_input, false);
});

test("tampering with a sealed receipt fails full-root validation", (t) => {
  const created = createSynthetic(t);
  fs.appendFileSync(path.join(created.qualificationRoot, "009_qualification_report.json"), " ");
  assert.throws(() => validateSynthetic(created), /canonical|drift|hash|sha256/i);
});

test("missing and extra root entries both fail exact-entry validation", (t) => {
  const missing = createSynthetic(t, { outputRoot: temporaryOutput(t, "missing") });
  fs.rmSync(path.join(missing.qualificationRoot, "007_cooldown_delta.json"));
  assert.throws(() => validateSynthetic(missing), /entry set|missing|required/i);

  const extra = createSynthetic(t, { outputRoot: temporaryOutput(t, "extra") });
  fs.writeFileSync(path.join(extra.qualificationRoot, "unregistered.txt"), "not registered");
  assert.throws(() => validateSynthetic(extra), /entry set|extra|unexpected/i);
});

test("an extra empty directory fails exact-directory validation", (t) => {
  const created = createSynthetic(t);
  fs.mkdirSync(path.join(created.qualificationRoot, "unregistered-empty-directory"));
  assert.throws(
    () => validateSynthetic(created),
    /directory.*entry set|entry set.*directory|extra|unexpected/i,
  );
});

test("source verification cannot be disabled for full-root validation", (t) => {
  const created = createSynthetic(t);
  assert.throws(
    () =>
      validateSyntheticQualificationArtifact({
        qualificationRoot: created.qualificationRoot,
        protocolPath: PROTOCOL_PATH,
        repositoryRoot: REPOSITORY_ROOT,
        verifySources: false,
      }),
    /verifySources|source verification|source.*required|required.*source/i,
  );
});

test("more than 4096 events in one channel and window exceeds the sealed budget", (t) => {
  const faultEvents = Array.from({ length: 4097 }, () => ({
    logName: "Application",
    providerName: "Synthetic-Benign-Provider",
    eventId: 1000,
  }));
  const created = createSynthetic(t, { faultEvents });
  assert.throws(
    () => validateSynthetic(created),
    /4096|budget|event.*limit|record.*limit|too many/i,
  );
});

test("source-audit nonconformance exit 2 cannot remain conformant", (t) => {
  const created = createSynthetic(t, { sourceAuditExitCode: 2 });
  let validated;
  try {
    validated = validateSynthetic(created);
  } catch (error) {
    assert.match(
      String(error?.message ?? error),
      /source.*audit|audit.*exit|exit.*code|nonconformant/i,
    );
    return;
  }
  assert.equal(validated.integrityValid, true);
  assert.equal(validated.criteriaPassed, false);
  assert.equal(validated.validatedEligibleAsHostQualificationInput, false);
  assert.ok(validated.failureCodes.includes("event_log_source_before_nonconformant"));

  assert.throws(
    () => createSynthetic(t, { sourceAuditExitCode: 3 }),
    /Audit v2 exit 0 or nonconformance exit 2/,
  );
});

test("an operator declaration with the wrong schema version is rejected", (t) => {
  const created = createSynthetic(t, {
    operatorDeclarationSchemaVersion: "wrong-operator-declaration-schema.v999",
  });
  assert.throws(
    () => validateSynthetic(created),
    /operator.*declaration.*schema|declaration.*schema|preregistration.*drift/i,
  );
});

test("the process delta cannot substitute a different window identity", (t) => {
  const created = createSynthetic(t, { processWindowName: "not_probe_process" });
  assert.throws(
    () => validateSynthetic(created),
    /process.*window|window.*name|probe_process/i,
  );
});

test("little-endian microcode is decoded numerically and fails below 0x12f", (t) => {
  const created = createSynthetic(t, { microcodeRawLeHex: "20010000" });
  const validated = validateSynthetic(created);
  assert.equal(validated.integrityValid, true);
  assert.equal(validated.criteriaPassed, false);
  assert.equal(validated.validatedEligibleAsHostQualificationInput, false);
  assert.ok(validated.failureCodes.includes("microcode_revision_below_minimum"));
});

test("a classified WHEA observation is recomputed as a qualification failure", (t) => {
  const created = createSynthetic(t, {
    faultEvents: [
      {
        logName: "System",
        providerName: "Microsoft-Windows-WHEA-Logger",
        eventId: 19,
      },
    ],
  });
  const validated = validateSynthetic(created);
  assert.equal(validated.integrityValid, true);
  assert.equal(validated.criteriaPassed, false);
  assert.equal(validated.validatedEligibleAsHostQualificationInput, false);
  assert.ok(validated.failureCodes.includes("new_whea_event"));
});

test("post-audit source configuration drift is preserved as a qualification failure", (t) => {
  const created = createSynthetic(t, {
    postSourceConfigurationId: "e".repeat(64),
  });
  const validated = validateSynthetic(created);
  assert.equal(validated.integrityValid, true);
  assert.equal(validated.criteriaPassed, false);
  assert.equal(validated.validatedEligibleAsHostQualificationInput, false);
  assert.ok(validated.failureCodes.includes("event_log_source_configuration_drift"));
});

test("strict JSON rejects duplicate keys and noncanonical numbers", (t) => {
  const duplicate = createSynthetic(t, { outputRoot: temporaryOutput(t, "duplicate") });
  const duplicatePath = path.join(duplicate.qualificationRoot, "000_scope_claim.json");
  const duplicateRaw = fs.readFileSync(duplicatePath, "utf8");
  fs.writeFileSync(duplicatePath, `{"schema_version":"duplicate",${duplicateRaw.slice(1)}`);
  assert.throws(() => validateSynthetic(duplicate), /duplicate|JSON/i);

  const number = createSynthetic(t, { outputRoot: temporaryOutput(t, "number") });
  const numberPath = path.join(number.qualificationRoot, "001_preregistration.json");
  const numberRaw = fs.readFileSync(numberPath, "utf8");
  const noncanonicalRaw = numberRaw.replace(/:1(?=[,}])/u, ":1.00");
  assert.notEqual(noncanonicalRaw, numberRaw, "fixture must replace one canonical integer");
  fs.writeFileSync(numberPath, noncanonicalRaw);
  assert.throws(() => validateSynthetic(number), /canonical|number|JSON/i);
});

test("a conformant raw Audit v2 is fully rebound but remains non-authorizing", (t) => {
  const fixture = createRawAuditFixture(t);
  const snapshot = adaptProvisionerAuditV2Artifact(fixture.options);

  assert.equal(
    snapshot.schema_version,
    EVENT_LOG_SOURCE_AUDIT_ADAPTER_SNAPSHOT_SCHEMA_VERSION,
  );
  assert.equal(snapshot.raw_audit.raw_sha256, sha256(fixture.raw));
  assert.equal(snapshot.raw_audit.overall_conformant, true);
  assert.equal(snapshot.verification.full_raw_audit_bound, true);
  assert.equal(snapshot.verification.raw_audit_content_id_basis_revalidated, true);
  assert.equal(snapshot.source_lineage.source_pin_revalidated, true);
  assert.equal(snapshot.boundary.projection_emitted, false);
  assert.equal(snapshot.boundary.real_provisioner_observation, false);
  assert.equal(snapshot.boundary.eligible_as_host_qualification_input, false);
  assert.equal(snapshot.boundary.cuda_execution_authorized, false);
  assert.equal(snapshot.boundary.four_capability_claim_authorized, false);
  assert.deepEqual(snapshot.diagnostic_failure_codes, [
    "production_capture_untrusted",
    "independent_control_plane_reobservation_missing",
  ]);
  const snapshotCore = { ...snapshot };
  delete snapshotCore.snapshot_id;
  assert.equal(snapshot.snapshot_id, sha256(Buffer.from(canonicalJson(snapshotCore), "utf8")));
  assertDeepFrozen(snapshot);
});

test("an internally consistent exit-2 raw Audit remains a diagnostic snapshot", (t) => {
  const fixture = createRawAuditFixture(t, { conformant: false });
  const snapshot = adaptProvisionerAuditV2Artifact(fixture.options);

  assert.equal(snapshot.raw_audit.process_exit_code, 2);
  assert.equal(snapshot.raw_audit.overall_conformant, false);
  assert.equal(snapshot.boundary.eligible_as_host_qualification_input, false);
  assert.ok(
    snapshot.diagnostic_failure_codes.includes("raw_audit_recomputed_nonconformant"),
  );
});

test("raw Audit capture, schema, and process outcomes are strictly separated", async (t) => {
  const fixture = createRawAuditFixture(t);
  await t.test("capture identity mismatch", () => {
    assert.throws(
      () =>
        adaptProvisionerAuditV2Artifact({
          ...fixture.options,
          captureEnvelope: {
            ...fixture.options.captureEnvelope,
            stdout_raw_sha256: "f".repeat(64),
          },
        }),
      /capture envelope identity/i,
    );
  });
  await t.test("capture byte count is checked before the bounded read", () => {
    assert.throws(
      () =>
        adaptProvisionerAuditV2Artifact({
          ...fixture.options,
          captureEnvelope: {
            ...fixture.options.captureEnvelope,
            stdout_byte_count: fixture.raw.length - 1,
          },
        }),
      /size does not match the bounded capture claim/i,
    );
    assert.throws(
      () =>
        adaptProvisionerAuditV2Artifact({
          ...fixture.options,
          captureEnvelope: {
            ...fixture.options.captureEnvelope,
            stdout_byte_count: 1_048_577,
          },
        }),
      /outside the frozen adapter budget/i,
    );
  });
  await t.test("failure exit 3 is not Audit v2", () => {
    assert.throws(
      () =>
        adaptProvisionerAuditV2Artifact({
          ...fixture.options,
          captureEnvelope: {
            ...fixture.options.captureEnvelope,
            process_exit_code: 3,
          },
        }),
      /must be 0 or 2/i,
    );
  });
  await t.test("exit 0 cannot be paired with capture claim 2", () => {
    assert.throws(
      () =>
        adaptProvisionerAuditV2Artifact({
          ...fixture.options,
          captureEnvelope: {
            ...fixture.options.captureEnvelope,
            process_exit_code: 2,
          },
        }),
      /exit\/disposition/i,
    );
  });
  await t.test("capture machine identity is only cross-checked, never trusted", () => {
    assert.throws(
      () =>
        adaptProvisionerAuditV2Artifact({
          ...fixture.options,
          captureEnvelope: {
            ...fixture.options.captureEnvelope,
            machine_identity_sha256: "c".repeat(64),
          },
        }),
      /machine identity.*capture envelope/i,
    );
  });
  await t.test("Provision output is rejected", () => {
    const receipt = cloneJson(fixture.receipt);
    receipt.mode = "Provision";
    const raw = Buffer.from(`${JSON.stringify(receipt)}\n`, "utf8");
    assert.throws(
      () => adaptProvisionerAuditV2Artifact(replaceRawAuditFixture(t, fixture, raw)),
      /identity or mode drift/i,
    );
  });
  await t.test("failure v1 cannot masquerade as Audit v2", () => {
    const receipt = cloneJson(fixture.receipt);
    receipt.schema_version = "volvence-evidence-event-log-provisioning-failure.v1";
    const raw = Buffer.from(`${JSON.stringify(receipt)}\n`, "utf8");
    assert.throws(
      () => adaptProvisionerAuditV2Artifact(replaceRawAuditFixture(t, fixture, raw)),
      /failure v1 cannot be adapted/i,
    );
  });
  await t.test("duplicate key, float, BOM, and trailing JSON are rejected", () => {
    const text = fixture.raw.toString("utf8");
    const duplicate = Buffer.from(
      `{"schema_version":"duplicate",${text.slice(1)}`,
      "utf8",
    );
    assert.throws(
      () => adaptProvisionerAuditV2Artifact(replaceRawAuditFixture(t, fixture, duplicate)),
      /duplicate|JSON/i,
    );
    const floating = Buffer.from(
      text.replace('"process_exit_code":0', '"process_exit_code":0.0'),
      "utf8",
    );
    assert.throws(
      () => adaptProvisionerAuditV2Artifact(replaceRawAuditFixture(t, fixture, floating)),
      /canonical|integer|number/i,
    );
    const bom = Buffer.concat([Buffer.from([0xef, 0xbb, 0xbf]), fixture.raw]);
    assert.throws(
      () => adaptProvisionerAuditV2Artifact(replaceRawAuditFixture(t, fixture, bom)),
      /BOM/i,
    );
    const invalidUtf8 = Buffer.from(fixture.raw);
    invalidUtf8[10] = 0xff;
    assert.throws(
      () =>
        adaptProvisionerAuditV2Artifact(
          replaceRawAuditFixture(t, fixture, invalidUtf8),
        ),
      /encoded data|UTF-8|encoding/i,
    );
    const trailing = Buffer.concat([fixture.raw, Buffer.from("{}\n", "utf8")]);
    assert.throws(
      () => adaptProvisionerAuditV2Artifact(replaceRawAuditFixture(t, fixture, trailing)),
      /one compact JSON value/i,
    );
    const spaced = Buffer.from(
      text.replace(',"config_schema_version"', ', "config_schema_version"'),
      "utf8",
    );
    assert.throws(
      () => adaptProvisionerAuditV2Artifact(replaceRawAuditFixture(t, fixture, spaced)),
      /compact ordered JSON bytes/i,
    );
  });
});

test("raw Audit basis, derivations, ordering, and scope cross-bind fail closed", async (t) => {
  const fixture = createRawAuditFixture(t);
  await t.test("forged conformance boolean", () => {
    const receipt = cloneJson(fixture.receipt);
    receipt.conformance.application_channel_before.enabled = false;
    const raw = Buffer.from(`${JSON.stringify(receipt)}\n`, "utf8");
    assert.throws(
      () => adaptProvisionerAuditV2Artifact(replaceRawAuditFixture(t, fixture, raw)),
      /conformance recomputation|value\/type drift/i,
    );
  });
  await t.test("noncanonical machine-config basis", () => {
    const receipt = cloneJson(fixture.receipt);
    receipt.machine_config_content_id_basis_base64 += "=";
    const raw = Buffer.from(`${JSON.stringify(receipt)}\n`, "utf8");
    assert.throws(
      () => adaptProvisionerAuditV2Artifact(replaceRawAuditFixture(t, fixture, raw)),
      /base64/i,
    );
  });
  await t.test("basis whitespace and ordered-key drift remain invalid after re-signing", () => {
    const whitespaceReceipt = cloneJson(fixture.receipt);
    const basisText = Buffer.from(
      whitespaceReceipt.machine_config_content_id_basis_base64,
      "base64",
    ).toString("utf8");
    const whitespaceBasis = Buffer.from(`{ ${basisText.slice(1)}`, "utf8");
    whitespaceReceipt.machine_config_content_id_basis_base64 =
      whitespaceBasis.toString("base64");
    whitespaceReceipt.machine_config_content_id = sha256(whitespaceBasis);
    const whitespaceRaw = Buffer.from(`${JSON.stringify(whitespaceReceipt)}\n`, "utf8");
    assert.throws(
      () =>
        adaptProvisionerAuditV2Artifact(
          replaceRawAuditFixture(t, fixture, whitespaceRaw),
        ),
      /compact ordered JSON bytes/i,
    );

    const orderReceipt = cloneJson(fixture.receipt);
    const basis = JSON.parse(
      Buffer.from(orderReceipt.machine_config_content_id_basis_base64, "base64").toString(
        "utf8",
      ),
    );
    const reorderedBasis = {
      machine: basis.machine,
      schema_version: basis.schema_version,
      script_integrity: basis.script_integrity,
      cmdlet_provenance: basis.cmdlet_provenance,
      fixed_contract: basis.fixed_contract,
      observed_source_registry: basis.observed_source_registry,
      observed_application_registry: basis.observed_application_registry,
      observed_application_channel: basis.observed_application_channel,
    };
    const reorderedBasisBytes = Buffer.from(JSON.stringify(reorderedBasis), "utf8");
    orderReceipt.machine_config_content_id_basis_base64 =
      reorderedBasisBytes.toString("base64");
    orderReceipt.machine_config_content_id = sha256(reorderedBasisBytes);
    const orderRaw = Buffer.from(`${JSON.stringify(orderReceipt)}\n`, "utf8");
    assert.throws(
      () =>
        adaptProvisionerAuditV2Artifact(replaceRawAuditFixture(t, fixture, orderRaw)),
      /ordered keys drifted/i,
    );
  });
  await t.test("provider order drift", () => {
    const receipt = cloneJson(fixture.receipt);
    receipt.observed.application_channel_before.provider_names.reverse();
    receipt.observed.application_channel_after.provider_names.reverse();
    const raw = Buffer.from(`${JSON.stringify(receipt)}\n`, "utf8");
    assert.throws(
      () => adaptProvisionerAuditV2Artifact(replaceRawAuditFixture(t, fixture, raw)),
      /ordinal sort order/i,
    );
  });
  await t.test("provider duplicates cannot satisfy exact membership", () => {
    const receipt = cloneJson(fixture.receipt);
    receipt.observed.application_channel_before.provider_names.push("VolvenceEvidence");
    receipt.observed.application_channel_after.provider_names.push("VolvenceEvidence");
    const raw = Buffer.from(`${JSON.stringify(receipt)}\n`, "utf8");
    assert.throws(
      () => adaptProvisionerAuditV2Artifact(replaceRawAuditFixture(t, fixture, raw)),
      /must not contain duplicates/i,
    );
  });
  await t.test("single-endpoint object-key reorder is rejected", () => {
    const receipt = cloneJson(fixture.receipt);
    const after = receipt.observed.application_channel_after;
    const entries = Object.entries(after);
    receipt.observed.application_channel_after = Object.fromEntries([
      entries[1],
      entries[0],
      ...entries.slice(2),
    ]);
    const raw = Buffer.from(`${JSON.stringify(receipt)}\n`, "utf8");
    assert.throws(
      () => adaptProvisionerAuditV2Artifact(replaceRawAuditFixture(t, fixture, raw)),
      /ordered keys drifted/i,
    );
  });
  await t.test("100ns chronology cannot hide inside one millisecond", () => {
    const receipt = cloneJson(fixture.receipt);
    receipt.refresh_chronology.post_observation_completed_at_utc =
      "2026-08-22T00:00:00.0008000Z";
    receipt.observed_at_utc = "2026-08-22T00:00:00.0001000Z";
    const raw = Buffer.from(`${JSON.stringify(receipt)}\n`, "utf8");
    const options = replaceRawAuditFixture(t, fixture, raw, {
      process_started_at_utc: "2026-08-22T00:00:00.0009000Z",
      process_exited_at_utc: "2026-08-22T00:00:00.0010000Z",
      stdout_captured_at_utc: "2026-08-22T00:00:00.0011000Z",
    });
    assert.throws(
      () => adaptProvisionerAuditV2Artifact(options),
      /observation falls outside|chronology/i,
    );
  });
  await t.test("source lineage and observed source values are independently checked", () => {
    const lineageReceipt = cloneJson(fixture.receipt);
    lineageReceipt.script_integrity.observed_lf_canonical_sha256 = "f".repeat(64);
    const lineageRaw = Buffer.from(`${JSON.stringify(lineageReceipt)}\n`, "utf8");
    assert.throws(
      () =>
        adaptProvisionerAuditV2Artifact(
          replaceRawAuditFixture(t, fixture, lineageRaw),
        ),
      /source-lineage drift/i,
    );

    const valueReceipt = cloneJson(fixture.receipt);
    valueReceipt.observed.source_registry_after.values[1].data = 6;
    const valueRaw = Buffer.from(`${JSON.stringify(valueReceipt)}\n`, "utf8");
    assert.throws(
      () => adaptProvisionerAuditV2Artifact(replaceRawAuditFixture(t, fixture, valueRaw)),
      /conformance recomputation|value\/type drift/i,
    );
  });
  await t.test("registry endpoint and safety claims are recomputed", () => {
    const registryReceipt = cloneJson(fixture.receipt);
    registryReceipt.observed.application_registry_after.values.push({
      name: "SyntheticDrift",
      kind: "DWord",
      data: 1,
    });
    const registryRaw = Buffer.from(`${JSON.stringify(registryReceipt)}\n`, "utf8");
    assert.throws(
      () =>
        adaptProvisionerAuditV2Artifact(
          replaceRawAuditFixture(t, fixture, registryRaw),
        ),
      /conformance recomputation|value\/type drift/i,
    );

    const safetyReceipt = cloneJson(fixture.receipt);
    safetyReceipt.safety_boundary.production_evidence_authorized = true;
    const safetyRaw = Buffer.from(`${JSON.stringify(safetyReceipt)}\n`, "utf8");
    assert.throws(
      () => adaptProvisionerAuditV2Artifact(replaceRawAuditFixture(t, fixture, safetyRaw)),
      /safety_boundary.*value\/type drift/i,
    );
  });
  await t.test("refresh null is never allowed to become refresh proof", () => {
    const receipt = cloneJson(fixture.receipt);
    receipt.requires_cold_or_service_refresh = false;
    receipt.refresh_chronology.refresh_verified = true;
    const raw = Buffer.from(`${JSON.stringify(receipt)}\n`, "utf8");
    assert.throws(
      () => adaptProvisionerAuditV2Artifact(replaceRawAuditFixture(t, fixture, raw)),
      /refresh or authorization boundary drift/i,
    );
  });
  await t.test("post-Audit scope configuration mismatch", () => {
    assert.throws(
      () =>
        adaptProvisionerAuditV2Artifact({
          ...fixture.options,
          captureEnvelope: {
            ...fixture.options.captureEnvelope,
            capture_role: "qualification_source_audit_after",
          },
          expectedMachineConfigContentId: "f".repeat(64),
        }),
      /frozen scope value/i,
    );
  });
  await t.test("after-role caller content-ID match is not called a scope proof", () => {
    const snapshot = adaptProvisionerAuditV2Artifact({
      ...fixture.options,
      captureEnvelope: {
        ...fixture.options.captureEnvelope,
        capture_role: "qualification_source_audit_after",
      },
      expectedMachineConfigContentId: fixture.receipt.machine_config_content_id,
    });
    assert.equal(snapshot.machine_config.caller_expected_content_id_matched, true);
    assert.equal(snapshot.boundary.capture_envelope_authoritative, false);
    assert.equal(snapshot.boundary.eligible_as_host_qualification_input, false);
  });
  await t.test("external protocol pin mismatch", () => {
    assert.throws(
      () =>
        adaptProvisionerAuditV2Artifact({
          ...fixture.options,
          expectedProtocolId: "f".repeat(64),
        }),
      /external pin/i,
    );
  });
});
