import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  __testing,
  HOST_QUALIFICATION_TERMINAL_SCHEMA_VERSION,
  loadHostCampaignProtocol,
  preregisterHostCampaign as productionPreregisterHostCampaign,
  validateHostCampaign as productionValidateHostCampaign,
} from "../src/volvence_zero/offline_evidence/windows_cuda_strict_32k_host_campaign.mjs";

const preregisterHostCampaign = __testing.preregisterSyntheticHostCampaign;
const runHostCampaign = __testing.runSyntheticHostCampaign;
const validateHostCampaign = __testing.validateSyntheticHostCampaign;

const REPOSITORY_ROOT = path.resolve(import.meta.dirname, "../../../");
const PROTOCOL_PATH = path.join(
  REPOSITORY_ROOT,
  "packages/vz-runtime/src/volvence_zero/offline_evidence/protocols/windows_cuda_strict_32k_host_campaign_v1.json",
);
const CHILD_PROTOCOL_PATH = path.join(
  REPOSITORY_ROOT,
  "packages/vz-runtime/src/volvence_zero/offline_evidence/protocols/windows_cuda_strict_32k_smoke_v1.json",
);
const HOST_ID = "1".repeat(64);
const BOOT_ID = "2".repeat(64);
const BASE_APPLICATION_XML = "3".repeat(64);
const BASE_SYSTEM_XML = "4".repeat(64);

function canonicalWrite(filePath, payload) {
  fs.writeFileSync(filePath, __testing.canonicalBytes(payload), { flag: "wx" });
}

function buildQualificationTerminal(directory, suffix = "a") {
  const core = {
    schema_version: HOST_QUALIFICATION_TERMINAL_SCHEMA_VERSION,
    qualification_protocol_id: "5".repeat(64),
    artifact_id: suffix.repeat(64),
    manifest_sha256: "6".repeat(64),
    host_identity_sha256: HOST_ID,
    boot_identity_sha256: BOOT_ID,
    passed: true,
    real_cuda_evidence_authorized: true,
    completed_at_utc: new Date(Date.now() - 120_000).toISOString(),
  };
  const payload = {
    ...core,
    terminal_id: __testing.sha256Bytes(__testing.canonicalBytes(core)),
  };
  const output = path.join(directory, `qualification-${suffix}.json`);
  canonicalWrite(output, payload);
  return output;
}

function channelCursor(logName, newestRecordId, xmlSha256) {
  return {
    log_name: logName,
    enabled: true,
    record_count: newestRecordId,
    oldest_record_id: 1,
    newest_record_id: newestRecordId,
    newest_record_xml_sha256: xmlSha256,
    maximum_size_bytes: 20_971_520,
    log_mode: "Circular",
  };
}

class FakeEventLogCollector {
  constructor() {
    this.anchors = [];
    this.applicationEvents = [];
    this.systemEvents = [];
    this.applicationBase = 1000;
    this.systemBase = 2000;
  }

  timestamp() {
    return new Date().toISOString();
  }

  host() {
    return {
      platform_system: "Windows",
      machine_identity_sha256: HOST_ID,
      boot_identity_sha256: BOOT_ID,
      last_boot_up_time_utc: new Date(Date.now() - 3_600_000).toISOString(),
      powershell_version: "5.1.22621.4391",
      os: { caption: "Windows 11", version: "10.0.22631", build_number: "22631" },
      cpu: { name: "test CPU", physical_core_count: 24, logical_processor_count: 32 },
      bios: {
        manufacturer: "test",
        smbios_version: "test-0x12f",
        release_date_utc: new Date(Date.now() - 86_400_000).toISOString(),
      },
      baseboard: { manufacturer: "test", product: "test", version: "1" },
      microcode_registry_raw_le_hex: "2f010000",
      gpu_adapters: [{ name: "NVIDIA GeForce RTX 4090", driver_version: "560.94" }],
    };
  }

  findAnchors({ scopeId }) {
    return {
      schema_version: "volvence-local-event-anchor-inventory.v1",
      scope_id: scopeId,
      anchors: this.anchors
        .filter((anchor) => anchor.scopeId === scopeId)
        .map((anchor) => {
          const { schema_version: _schemaVersion, ...inventoryObservation } = anchor.observation;
          return inventoryObservation;
        }),
    };
  }

  captureBaseline() {
    return {
      schema_version: "windows-host-event-log-baseline-collector.v1",
      collection_started_at_utc: this.timestamp(),
      collection_completed_at_utc: this.timestamp(),
      host: this.host(),
      channels: [
        channelCursor("Application", this.applicationBase, BASE_APPLICATION_XML),
        channelCursor("System", this.systemBase, BASE_SYSTEM_XML),
      ],
    };
  }

  capturePrelaunch() {
    const applicationNewest = this.applicationBase + this.applicationEvents.length;
    const systemNewest =
      this.systemEvents.length === 0
        ? this.systemBase
        : this.systemEvents[this.systemEvents.length - 1].record_id;
    return {
      schema_version: "windows-host-event-log-prelaunch-collector.v1",
      collection_started_at_utc: this.timestamp(),
      collection_completed_at_utc: this.timestamp(),
      host: this.host(),
      channels: [
        {
          log_name: "Application",
          baseline_newest_record_id: this.applicationBase,
          baseline_boundary_present: true,
          baseline_boundary_xml_sha256: BASE_APPLICATION_XML,
          end_cursor: channelCursor(
            "Application",
            applicationNewest,
            this.applicationEvents.at(-1)?.xml_sha256 ?? BASE_APPLICATION_XML,
          ),
          new_record_count: this.applicationEvents.length,
          within_record_budget: true,
        },
        {
          log_name: "System",
          baseline_newest_record_id: this.systemBase,
          baseline_boundary_present: true,
          baseline_boundary_xml_sha256: BASE_SYSTEM_XML,
          end_cursor: channelCursor(
            "System",
            systemNewest,
            this.systemEvents.at(-1)?.xml_sha256 ?? BASE_SYSTEM_XML,
          ),
          new_record_count: this.systemEvents.length,
          within_record_budget: true,
        },
      ],
    };
  }

  writeAnchor({ protocol, message }) {
    const eventId = Number(protocol.payload.event_log.anchor_event_ids[message.anchor_kind].value);
    const recordId = this.applicationBase + this.applicationEvents.length + 1;
    const payload = __testing.canonicalBytes(message, false);
    const xmlSha256 = __testing.sha256Bytes(Buffer.from(`anchor-${recordId}`));
    const observation = {
      schema_version: "volvence-local-event-anchor-observation.v1",
      log_name: "Application",
      provider_name: "VolvenceEvidence",
      event_id: eventId,
      record_id: recordId,
      time_created_utc: this.timestamp(),
      xml_sha256: xmlSha256,
      payload_base64: payload.toString("base64"),
    };
    this.applicationEvents.push({
      log_name: "Application",
      provider_name: "VolvenceEvidence",
      event_id: eventId,
      record_id: recordId,
      level: 4,
      time_created_utc: observation.time_created_utc,
      xml_sha256: xmlSha256,
      payload_kind: "event_data",
      event_data: [{ name: "", value: payload.toString("utf8") }],
    });
    this.anchors.push({ scopeId: message.scope_id, message, observation });
    return observation;
  }

  verifyAnchor({ message, recordId }) {
    const expectedRecordId = Number(recordId?.value ?? recordId);
    const match = this.anchors.find(
      (anchor) =>
        anchor.observation.record_id === expectedRecordId &&
        __testing.canonicalJson(anchor.message) === __testing.canonicalJson(message),
    );
    if (!match) throw new Error("fake anchor missing");
    return match.observation;
  }

  captureDelta() {
    const applicationNewest = this.applicationBase + this.applicationEvents.length;
    const systemNewest =
      this.systemEvents.length === 0
        ? this.systemBase
        : this.systemEvents[this.systemEvents.length - 1].record_id;
    return {
      schema_version: "windows-host-event-log-delta-collector.v1",
      collection_started_at_utc: this.timestamp(),
      collection_completed_at_utc: this.timestamp(),
      host: this.host(),
      channels: [
        {
          log_name: "Application",
          baseline_newest_record_id: this.applicationBase,
          baseline_boundary_present: true,
          baseline_boundary_xml_sha256: BASE_APPLICATION_XML,
          end_cursor: channelCursor(
            "Application",
            applicationNewest,
            this.applicationEvents.at(-1)?.xml_sha256 ?? BASE_APPLICATION_XML,
          ),
          channel_configuration_stable: true,
          end_cursor_hash_exact: true,
          scanned_record_count: this.applicationEvents.length,
          record_id_range_complete: true,
          truncated: false,
          events: this.applicationEvents,
        },
        {
          log_name: "System",
          baseline_newest_record_id: this.systemBase,
          baseline_boundary_present: true,
          baseline_boundary_xml_sha256: BASE_SYSTEM_XML,
          end_cursor: channelCursor(
            "System",
            systemNewest,
            this.systemEvents.at(-1)?.xml_sha256 ?? BASE_SYSTEM_XML,
          ),
          channel_configuration_stable: true,
          end_cursor_hash_exact: true,
          scanned_record_count: this.systemEvents.length,
          record_id_range_complete: true,
          truncated: false,
          events: this.systemEvents,
        },
      ],
    };
  }

  addWhea() {
    const recordId = this.systemBase + this.systemEvents.length + 1;
    this.systemEvents.push({
      log_name: "System",
      provider_name: "Microsoft-Windows-WHEA-Logger",
      event_id: 19,
      record_id: recordId,
      level: 3,
      time_created_utc: this.timestamp(),
      xml_sha256: __testing.sha256Bytes(Buffer.from(`whea-${recordId}`)),
      payload_kind: "event_data",
      event_data: [{ name: "ApicId", value: "32" }],
    });
  }
}

function childProtocolPayload() {
  return __testing.parseJsonStrict(
    fs.readFileSync(CHILD_PROTOCOL_PATH, "utf8"),
    "test child protocol",
  );
}

function buildChildAttestation() {
  const core = {
    attention_implementation: "sdpa",
    capture_failure_mode: "raise",
    context_window_tokens: 32768,
    cuda_version: "12.6",
    cudnn_version: 91002,
    device: "cuda",
    device_compute_capability: [8, 9],
    device_name: "NVIDIA GeForce RTX 4090",
    execution_assets_sha256: "bbb5446f8d802b437c2fc7e2cefcdabb996bbd4bc657fe155ea015d30a841bb0",
    fail_on_truncation: true,
    fallback_mode: "deny",
    generation_capture_strategy: "first-full-prompt-set-once",
    generation_use_cache: true,
    hidden_size: 1536,
    hook_layer_indices: [20],
    local_files_only: true,
    model_dtype: "bfloat16",
    model_id: "Qwen/Qwen2.5-1.5B-Instruct",
    model_max_position_embeddings: 32768,
    model_revision: "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    model_weights_sha256: "fb8c44c48b8359fdd306cdc5f473d7c04d88955013f0dd8549f266e248194da4",
    platform_release: "10",
    platform_system: "Windows",
    preset_name: "windows-cuda-cudnn-sdpa-cached-strict.v1",
    profile_id: "3be84d866afbda07cf80dee277d89cdc0e366ce545bf7e97f015cf8afcbfe21a",
    python_version: "3.11.15",
    require_generation_chat_template: true,
    runtime_origin: "hf-local",
    schema_version: "transformers-execution-attestation.v1",
    sdpa_backend: "cudnn",
    sdpa_backend_exclusive: true,
    sdpa_backend_policy: "exclusive-cudnn",
    torch_version: "2.12.0+cu126",
    transformers_version: "5.9.0",
  };
  const attestationId = __testing.sha256Bytes(__testing.canonicalBytes(core, false));
  assert.equal(
    attestationId,
    "9a33a698b95d923d6a4e82b64471213d529b0cbbf6a30ca24644860211e6dde1",
  );
  return { ...core, attestation_id: attestationId };
}

function buildValidChildArtifact({
  outputDir,
  leaseId,
  processId,
  passed = true,
  integerFeatureSubstitution = false,
}) {
  const protocol = childProtocolPayload();
  const protocolRaw = fs.readFileSync(CHILD_PROTOCOL_PATH);
  const protocolId = __testing.sha256Bytes(__testing.canonicalBytes(protocol));
  const protocolRawSha256 = __testing.sha256Bytes(protocolRaw);
  fs.mkdirSync(outputDir, { recursive: true });
  const launchCore = {
    schema_version: "windows-cuda-strict-32k-smoke-launch.v1",
    protocol_id: protocolId,
    protocol_raw_sha256: protocolRawSha256,
    source_hash_mode: protocol.source_hash_mode,
    source_sha256: protocol.source_sha256,
    attempt_budget: 1,
    retry_budget: 0,
    attempt_budget_scope: "per_frozen_output_root",
    retry_enforcement_owner: "outer_host_campaign",
    outer_attempt_lease_id: leaseId,
    process_id: processId,
    started_at_utc: new Date().toISOString(),
  };
  const attemptId = __testing.sha256Bytes(__testing.canonicalBytes(launchCore));
  const launch = { ...launchCore, attempt_id: attemptId };
  const attestation = buildChildAttestation();
  const expectedCapture = protocol.diagnostic.expected_capture;
  const residualValueCount =
    Number(expectedCapture.residual_sequence_length.value) *
    Number(expectedCapture.activation_width.value);
  const featureValue = (value) =>
    integerFeatureSubstitution ? Number(value.value) : value;
  const capture = {
    schema_version: expectedCapture.audit_summary_schema_version,
    residual_sequence_length: expectedCapture.residual_sequence_length,
    residual_step_continuity_exact: true,
    capture_layer_exact: true,
    capture_width_exact: true,
    residual_activation_value_count: residualValueCount,
    finite_residual_activation_value_count: residualValueCount,
    capture_values_all_finite: true,
    residual_sequence_sha256: "7".repeat(64),
    latest_activation_width: expectedCapture.activation_width,
    latest_activation_sha256: "8".repeat(64),
    latest_matches_sequence_exact: true,
    top_logit_count: passed ? 2 : 0,
    top_logits_finite_nonempty: passed,
    top_logits_sha256: "9".repeat(64),
    selected_feature_values: {
      hook_layer_coverage: featureValue(expectedCapture.hook_layer_coverage),
      hook_fire_rate: featureValue(expectedCapture.hook_fire_rate),
      token_step_coverage: featureValue(expectedCapture.token_step_coverage),
      residual_sequence_present: featureValue(expectedCapture.residual_sequence_present),
      fallback_active: featureValue(expectedCapture.fallback_active),
    },
    description_sha256: "a".repeat(64),
  };
  const checks = Object.fromEntries(
    [
      "execution_attestation_exact",
      "generation_execution_lineage_exact",
      "rendered_prompt_hash_exact",
      "input_token_count_exact",
      "context_budget_exact",
      "generated_token_count_exact",
      "capture_present",
      "residual_sequence_length_exact",
      "residual_step_continuity_exact",
      "capture_layer_exact",
      "capture_width_exact",
      "latest_capture_matches_sequence_exact",
      "capture_values_all_finite",
      "top_logits_finite_nonempty",
      "hook_layer_coverage_exact",
      "hook_fire_rate_exact",
      "token_step_coverage_exact",
      "residual_sequence_present_exact",
      "fallback_inactive",
      "no_conditioning_or_steering_applied",
    ].map((key) => [key, true]),
  );
  checks.top_logits_finite_nonempty = passed;
  const verdict = passed
    ? "passed_exact_strict_32767_plus_1_engineering_diagnostic"
    : "failed_diagnostic_stop_no_retry";
  const report = {
    schema_version: "windows-cuda-strict-32k-smoke-report.v1",
    attempt_id: attemptId,
    outer_attempt_lease_id: leaseId,
    protocol_id: protocolId,
    protocol_raw_sha256: protocolRawSha256,
    source_hash_mode: protocol.source_hash_mode,
    source_sha256: protocol.source_sha256,
    execution_attestation_id: attestation.attestation_id,
    generation_call: protocol.diagnostic.generation_call,
    observation: {
      generated_text_sha256: "b".repeat(64),
      generated_text_byte_count: 1,
      generated_token_count: 1,
      input_token_count: protocol.diagnostic.expected_context_budget.input_token_count,
      rendered_prompt_sha256: protocol.diagnostic.prompt_recipe.expected_rendered_prompt_sha256,
      context_budget: {
        ...protocol.diagnostic.expected_context_budget,
        execution_attestation_id: attestation.attestation_id,
      },
      capture,
      application_flags: {
        personal_conditioning_applied: false,
        conditioning_bank_carrier_count: 0,
        character_prefix_applied: false,
        character_residual_applied: false,
        steering_intervention_applied: false,
      },
    },
    checks,
    passed,
    verdict,
    evidence_firewall: protocol.evidence_firewall,
    claim_boundary: protocol.claim_boundary,
  };
  const payloads = {
    "launch_receipt.json": launch,
    "execution_attestation.json": attestation,
    "strict_32k_smoke_report.json": report,
  };
  const payloadBytes = Object.fromEntries(
    Object.entries(payloads).map(([name, payload]) => [name, __testing.canonicalBytes(payload)]),
  );
  for (const [name, raw] of Object.entries(payloadBytes)) {
    fs.writeFileSync(path.join(outputDir, name), raw, { flag: "wx" });
  }
  const manifestCore = {
    schema_version: "windows-cuda-strict-32k-smoke-manifest.v1",
    attempt_id: attemptId,
    outer_attempt_lease_id: leaseId,
    protocol_id: protocolId,
    protocol_raw_sha256: protocolRawSha256,
    source_hash_mode: protocol.source_hash_mode,
    source_sha256: protocol.source_sha256,
    execution_attestation_id: attestation.attestation_id,
    passed,
    verdict: report.verdict,
    files: Object.entries(payloadBytes).map(([name, raw]) => ({
      path: name,
      byte_count: raw.length,
      sha256: __testing.sha256Bytes(raw),
    })),
    evidence_firewall: protocol.evidence_firewall,
    claim_boundary: protocol.claim_boundary,
  };
  const artifactId = __testing.sha256Bytes(__testing.canonicalBytes(manifestCore));
  canonicalWrite(path.join(outputDir, "manifest.json"), {
    ...manifestCore,
    artifact_id: artifactId,
  });
  const completionCore = {
    schema_version: "windows-cuda-strict-32k-smoke-completion.v1",
    attempt_id: attemptId,
    outer_attempt_lease_id: leaseId,
    artifact_id: artifactId,
    protocol_id: protocolId,
    execution_attestation_id: attestation.attestation_id,
    passed,
    verdict: report.verdict,
    completed_at_utc: new Date().toISOString(),
  };
  canonicalWrite(path.join(outputDir, "completion_receipt.json"), {
    ...completionCore,
    completion_id: __testing.sha256Bytes(__testing.canonicalBytes(completionCore)),
  });
  return {
    artifact_id: artifactId,
    attempt_id: attemptId,
    outer_attempt_lease_id: leaseId,
    protocol_id: protocolId,
    execution_attestation_id: attestation.attestation_id,
    passed,
    verdict: report.verdict,
  };
}

function successfulExecutor({ stdoutFd, childOutputDir, leaseId }) {
  const processId = 4242;
  const terminal = buildValidChildArtifact({
    outputDir: childOutputDir,
    leaseId,
    processId,
  });
  fs.writeSync(stdoutFd, Buffer.from(`[strict-32k-smoke] test\n${JSON.stringify(terminal)}\n`));
  return {
    process_started: true,
    process_id: processId,
    exit_code: 0,
    signal: null,
    error_code: null,
    timed_out: false,
    duration_milliseconds: 123,
  };
}

function failedDiagnosticExecutor({ stdoutFd, childOutputDir, leaseId }) {
  const processId = 4343;
  const terminal = buildValidChildArtifact({
    outputDir: childOutputDir,
    leaseId,
    processId,
    passed: false,
  });
  fs.writeSync(stdoutFd, Buffer.from(`${JSON.stringify(terminal)}\n`));
  return {
    process_started: true,
    process_id: processId,
    exit_code: 2,
    signal: null,
    error_code: null,
    timed_out: false,
    duration_milliseconds: 123,
  };
}

function integerFeatureExecutor({ stdoutFd, childOutputDir, leaseId }) {
  const processId = 4444;
  const terminal = buildValidChildArtifact({
    outputDir: childOutputDir,
    leaseId,
    processId,
    integerFeatureSubstitution: true,
  });
  fs.writeSync(stdoutFd, Buffer.from(`${JSON.stringify(terminal)}\n`));
  return {
    process_started: true,
    process_id: processId,
    exit_code: 0,
    signal: null,
    error_code: null,
    timed_out: false,
    duration_milliseconds: 123,
  };
}

function makeFixture(t) {
  const temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), "volvence-host-campaign-"));
  t.after(() => fs.rmSync(temporaryRoot, { recursive: true, force: true }));
  return {
    temporaryRoot,
    campaignBaseDir: path.join(temporaryRoot, "campaigns"),
    qualificationPath: buildQualificationTerminal(temporaryRoot),
    eventLogCollector: new FakeEventLogCollector(),
  };
}

test("protocol freezes the existing child and does not claim an external anchor", () => {
  const protocol = loadHostCampaignProtocol({
    protocolPath: PROTOCOL_PATH,
    repositoryRoot: REPOSITORY_ROOT,
    verifySources: true,
  });
  assert.equal(
    protocol.payload.child.protocol_id,
    "4934a344550aab5c98f33892dd6d1ec2e5fe51c00694d2cc5b0a45fbc31e2c1a",
  );
  assert.equal(protocol.payload.evidence_firewall.external_append_only_anchor_present, false);
  assert.equal(protocol.payload.evidence_firewall.child_transitive_local_source_closure_pinned, false);
  assert.equal(protocol.payload.evidence_firewall.producer_full_artifact_revalidated_before_pass_return, false);
  assert.equal(protocol.payload.evidence_firewall.terminal_anchor_and_delayed_faults_covered_by_delta, false);
  assert.equal(protocol.payload.evidence_firewall.four_capability_claim_authorized, false);
  assert.equal(protocol.payload.host_qualification.production_preregistration_enabled, false);
});

test("synthetic chain exercises the receipt algorithm but is never evidence or PASS", (t) => {
  const fixture = makeFixture(t);
  const preregistered = preregisterHostCampaign({
    hostQualificationTerminalPath: fixture.qualificationPath,
    pythonExecutable: process.execPath,
    campaignBaseDir: fixture.campaignBaseDir,
    protocolPath: PROTOCOL_PATH,
    repositoryRoot: REPOSITORY_ROOT,
    eventLogCollector: fixture.eventLogCollector,
    verifySources: false,
  });
  assert.equal(preregistered.status, "preregistered");
  assert.equal(preregistered.leaseId.length, 64);
  const completed = runHostCampaign({
    campaignRoot: preregistered.campaignRoot,
    protocolPath: PROTOCOL_PATH,
    repositoryRoot: REPOSITORY_ROOT,
    eventLogCollector: fixture.eventLogCollector,
    childExecutor: successfulExecutor,
    verifySources: false,
  });
  assert.equal(completed.passed, false);
  assert.equal(completed.realExecutionObservationAuthorized, false);
  assert.deepEqual(completed.failureCodes, ["synthetic_test_backend_not_evidence"]);
  const validated = validateHostCampaign({
    campaignRoot: completed.campaignRoot,
    protocolPath: PROTOCOL_PATH,
    repositoryRoot: REPOSITORY_ROOT,
    verifySources: false,
  });
  assert.equal(validated.passed, false);
  assert.equal(validated.realExecutionObservationAuthorized, false);
  assert.deepEqual(validated.failureCodes, ["synthetic_test_backend_not_evidence"]);
  assert.equal(validated.campaignArtifactId, completed.campaignArtifactId);
  assert.throws(
    () => productionValidateHostCampaign({ campaignRoot: completed.campaignRoot }),
    /execution backend is not authorized/,
  );
  assert.throws(
    () =>
      runHostCampaign({
        campaignRoot: completed.campaignRoot,
        protocolPath: PROTOCOL_PATH,
        repositoryRoot: REPOSITORY_ROOT,
        eventLogCollector: fixture.eventLogCollector,
        childExecutor: successfulExecutor,
        verifySources: false,
      }),
    /entry set drift/,
  );
});

test("public production entry rejects dependency injection", () => {
  assert.throws(
    () =>
      productionPreregisterHostCampaign({
        hostQualificationTerminalPath: "unused",
        pythonExecutable: "unused",
        eventLogCollector: new FakeEventLogCollector(),
      }),
    /does not accept option eventLogCollector/,
  );
  assert.throws(
    () =>
      productionPreregisterHostCampaign({
        hostQualificationTerminalPath: "not-read-while-disabled",
        pythonExecutable: process.execPath,
      }),
    /production preregistration is disabled/,
  );
});

test("strict JSON rejects duplicate keys and non-Python-canonical numbers", () => {
  assert.throws(() => __testing.parseJsonStrict('{"a":1,"a":2}', "duplicate"), /duplicate/);
  assert.throws(() => __testing.parseJsonStrict('{"value":1.00}', "float"), /not Python-canonical/);
  assert.throws(() => __testing.parseJsonStrict('{"value":-0}', "integer"), /not Python-canonical/);
  assert.throws(() => __testing.parseJsonStrict('{"value":1e+02}', "float"), /not Python-canonical/);
  assert.throws(
    () => __testing.parseJsonStrict('{"value":1.0000000000000001}', "float"),
    /not Python-canonical/,
  );
  assert.throws(() => __testing.parseJsonStrict('{"value":1e-999}', "float"), /not Python-canonical/);
  assert.equal(__testing.canonicalJson(__testing.parseJsonStrict('{"value":1.0}', "valid")), '{"value":1.0}');
});

test("UTC timestamps reject calendar normalization and preserve 100 ns ordering", () => {
  assert.throws(
    () => __testing.requireUtcTimestamp("2026-02-30T00:00:00Z", "invalid day"),
    /not a valid UTC calendar timestamp/,
  );
  assert.throws(
    () => __testing.requireUtcTimestamp("2026-01-01T24:00:00Z", "invalid hour"),
    /not a valid UTC calendar timestamp/,
  );
  assert.ok(
    __testing.requireUtcTimestamp("2026-01-01T00:00:00.0000001Z", "later") >
      __testing.requireUtcTimestamp("2026-01-01T00:00:00.0000000Z", "earlier"),
  );
});

test("a second root for the same deterministic scope is rejected by the anchor inventory", (t) => {
  const fixture = makeFixture(t);
  preregisterHostCampaign({
    hostQualificationTerminalPath: fixture.qualificationPath,
    pythonExecutable: process.execPath,
    campaignBaseDir: fixture.campaignBaseDir,
    protocolPath: PROTOCOL_PATH,
    repositoryRoot: REPOSITORY_ROOT,
    eventLogCollector: fixture.eventLogCollector,
    verifySources: false,
  });
  assert.throws(
    () =>
      preregisterHostCampaign({
        hostQualificationTerminalPath: fixture.qualificationPath,
        pythonExecutable: process.execPath,
        campaignBaseDir: path.join(fixture.temporaryRoot, "other-campaigns"),
        protocolPath: PROTOCOL_PATH,
        repositoryRoot: REPOSITORY_ROOT,
        eventLogCollector: fixture.eventLogCollector,
        verifySources: false,
      }),
    /already has a local Event Log anchor/,
  );
});

test("a WHEA delta seals a terminal failure and never permits retry", (t) => {
  const fixture = makeFixture(t);
  const preregistered = preregisterHostCampaign({
    hostQualificationTerminalPath: fixture.qualificationPath,
    pythonExecutable: process.execPath,
    campaignBaseDir: fixture.campaignBaseDir,
    protocolPath: PROTOCOL_PATH,
    repositoryRoot: REPOSITORY_ROOT,
    eventLogCollector: fixture.eventLogCollector,
    verifySources: false,
  });
  const executor = (inputs) => {
    const result = successfulExecutor(inputs);
    fixture.eventLogCollector.addWhea();
    return result;
  };
  const completed = runHostCampaign({
    campaignRoot: preregistered.campaignRoot,
    protocolPath: PROTOCOL_PATH,
    repositoryRoot: REPOSITORY_ROOT,
    eventLogCollector: fixture.eventLogCollector,
    childExecutor: executor,
    verifySources: false,
  });
  assert.equal(completed.passed, false);
  assert.ok(completed.failureCodes.includes("new_whea_event"));
  const validated = validateHostCampaign({
    campaignRoot: completed.campaignRoot,
    protocolPath: PROTOCOL_PATH,
    repositoryRoot: REPOSITORY_ROOT,
    verifySources: false,
  });
  assert.deepEqual(validated.failureCodes, completed.failureCodes);
});

test("a structurally valid failed child uses exit 2 without a lineage false positive", (t) => {
  const fixture = makeFixture(t);
  const preregistered = preregisterHostCampaign({
    hostQualificationTerminalPath: fixture.qualificationPath,
    pythonExecutable: process.execPath,
    campaignBaseDir: fixture.campaignBaseDir,
    protocolPath: PROTOCOL_PATH,
    repositoryRoot: REPOSITORY_ROOT,
    eventLogCollector: fixture.eventLogCollector,
    verifySources: false,
  });
  const completed = runHostCampaign({
    campaignRoot: preregistered.campaignRoot,
    protocolPath: PROTOCOL_PATH,
    repositoryRoot: REPOSITORY_ROOT,
    eventLogCollector: fixture.eventLogCollector,
    childExecutor: failedDiagnosticExecutor,
    verifySources: false,
  });
  assert.ok(completed.failureCodes.includes("child_diagnostic_failed_exit_2"));
  assert.equal(completed.failureCodes.includes("child_lineage_mismatch"), false);
  const validated = validateHostCampaign({
    campaignRoot: completed.campaignRoot,
    protocolPath: PROTOCOL_PATH,
    repositoryRoot: REPOSITORY_ROOT,
    verifySources: false,
  });
  assert.deepEqual(validated.failureCodes, completed.failureCodes);
});

test("integer substitution cannot satisfy protocol float feature checks", (t) => {
  const fixture = makeFixture(t);
  const preregistered = preregisterHostCampaign({
    hostQualificationTerminalPath: fixture.qualificationPath,
    pythonExecutable: process.execPath,
    campaignBaseDir: fixture.campaignBaseDir,
    protocolPath: PROTOCOL_PATH,
    repositoryRoot: REPOSITORY_ROOT,
    eventLogCollector: fixture.eventLogCollector,
    verifySources: false,
  });
  const completed = runHostCampaign({
    campaignRoot: preregistered.campaignRoot,
    protocolPath: PROTOCOL_PATH,
    repositoryRoot: REPOSITORY_ROOT,
    eventLogCollector: fixture.eventLogCollector,
    childExecutor: integerFeatureExecutor,
    verifySources: false,
  });
  assert.ok(completed.failureCodes.includes("child_lineage_mismatch"));
});

test("prelaunch boot drift refuses lease consumption and child creation", (t) => {
  const fixture = makeFixture(t);
  const preregistered = preregisterHostCampaign({
    hostQualificationTerminalPath: fixture.qualificationPath,
    pythonExecutable: process.execPath,
    campaignBaseDir: fixture.campaignBaseDir,
    protocolPath: PROTOCOL_PATH,
    repositoryRoot: REPOSITORY_ROOT,
    eventLogCollector: fixture.eventLogCollector,
    verifySources: false,
  });
  const capturePrelaunch = fixture.eventLogCollector.capturePrelaunch.bind(
    fixture.eventLogCollector,
  );
  fixture.eventLogCollector.capturePrelaunch = (...args) => {
    const payload = capturePrelaunch(...args);
    return { ...payload, host: { ...payload.host, boot_identity_sha256: "e".repeat(64) } };
  };
  assert.throws(
    () =>
      runHostCampaign({
        campaignRoot: preregistered.campaignRoot,
        protocolPath: PROTOCOL_PATH,
        repositoryRoot: REPOSITORY_ROOT,
        eventLogCollector: fixture.eventLogCollector,
        childExecutor: successfulExecutor,
        verifySources: false,
      }),
    /not the qualified baseline machine and boot/,
  );
  assert.equal(
    fs.existsSync(path.join(preregistered.campaignRoot, "004_launch.json")),
    false,
  );
});

test("Event Log collector failure seals a reproducible non-PASS result", (t) => {
  const fixture = makeFixture(t);
  const preregistered = preregisterHostCampaign({
    hostQualificationTerminalPath: fixture.qualificationPath,
    pythonExecutable: process.execPath,
    campaignBaseDir: fixture.campaignBaseDir,
    protocolPath: PROTOCOL_PATH,
    repositoryRoot: REPOSITORY_ROOT,
    eventLogCollector: fixture.eventLogCollector,
    verifySources: false,
  });
  fixture.eventLogCollector.captureDelta = () => {
    throw new Error("synthetic collector failure");
  };
  const completed = runHostCampaign({
    campaignRoot: preregistered.campaignRoot,
    protocolPath: PROTOCOL_PATH,
    repositoryRoot: REPOSITORY_ROOT,
    eventLogCollector: fixture.eventLogCollector,
    childExecutor: successfulExecutor,
    verifySources: false,
  });
  assert.ok(completed.failureCodes.includes("event_log_collection_failed"));
  assert.ok(completed.failureCodes.includes("local_anchor_mismatch"));
  const validated = validateHostCampaign({
    campaignRoot: completed.campaignRoot,
    protocolPath: PROTOCOL_PATH,
    repositoryRoot: REPOSITORY_ROOT,
    verifySources: false,
  });
  assert.deepEqual(validated.failureCodes, completed.failureCodes);
});

test("a wrapper interruption after launch is permanently incomplete and not rerun", (t) => {
  const fixture = makeFixture(t);
  const preregistered = preregisterHostCampaign({
    hostQualificationTerminalPath: fixture.qualificationPath,
    pythonExecutable: process.execPath,
    campaignBaseDir: fixture.campaignBaseDir,
    protocolPath: PROTOCOL_PATH,
    repositoryRoot: REPOSITORY_ROOT,
    eventLogCollector: fixture.eventLogCollector,
    verifySources: false,
  });
  assert.throws(
    () =>
      runHostCampaign({
        campaignRoot: preregistered.campaignRoot,
        protocolPath: PROTOCOL_PATH,
        repositoryRoot: REPOSITORY_ROOT,
        eventLogCollector: fixture.eventLogCollector,
        childExecutor: () => {
          throw new Error("synthetic wrapper interruption");
        },
        verifySources: false,
      }),
    /synthetic wrapper interruption/,
  );
  const incomplete = validateHostCampaign({
    campaignRoot: preregistered.campaignRoot,
    protocolPath: PROTOCOL_PATH,
    repositoryRoot: REPOSITORY_ROOT,
    verifySources: false,
  });
  assert.equal(incomplete.status, "incomplete_consumed");
  assert.equal(incomplete.passed, false);
  assert.throws(
    () =>
      runHostCampaign({
        campaignRoot: preregistered.campaignRoot,
        protocolPath: PROTOCOL_PATH,
        repositoryRoot: REPOSITORY_ROOT,
        eventLogCollector: fixture.eventLogCollector,
        childExecutor: successfulExecutor,
        verifySources: false,
      }),
    /entry set drift/,
  );
});

test("tampering with a sealed report is rejected", (t) => {
  const fixture = makeFixture(t);
  const preregistered = preregisterHostCampaign({
    hostQualificationTerminalPath: fixture.qualificationPath,
    pythonExecutable: process.execPath,
    campaignBaseDir: fixture.campaignBaseDir,
    protocolPath: PROTOCOL_PATH,
    repositoryRoot: REPOSITORY_ROOT,
    eventLogCollector: fixture.eventLogCollector,
    verifySources: false,
  });
  const completed = runHostCampaign({
    campaignRoot: preregistered.campaignRoot,
    protocolPath: PROTOCOL_PATH,
    repositoryRoot: REPOSITORY_ROOT,
    eventLogCollector: fixture.eventLogCollector,
    childExecutor: successfulExecutor,
    verifySources: false,
  });
  fs.appendFileSync(path.join(completed.campaignRoot, "008_campaign_report.json"), " ");
  assert.throws(
    () =>
      validateHostCampaign({
        campaignRoot: completed.campaignRoot,
        protocolPath: PROTOCOL_PATH,
        repositoryRoot: REPOSITORY_ROOT,
        verifySources: false,
      }),
    /not canonical|drift/,
  );
});
