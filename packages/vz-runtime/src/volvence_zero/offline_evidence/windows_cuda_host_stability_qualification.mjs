/**
 * Production-disabled Windows host-stability qualification publisher scaffold.
 *
 * The public production surface is intentionally a static fail-closed gate. The
 * only qualification-chain writer in this revision is an explicitly synthetic
 * test-double used to exercise the create-only receipt chain and the independent
 * full-root validator. A separate pure adapter can bind and recompute a captured
 * raw Audit v2 artifact, but its snapshot is always non-authorizing. No function
 * in this module reads Windows Event Log records, launches PowerShell/Python/CUDA/
 * a model, or authorizes a later campaign.
 */

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const HOST_QUALIFICATION_PROTOCOL_SCHEMA_VERSION =
  "windows-cuda-host-stability-qualification-protocol.v1";
export const HOST_QUALIFICATION_TERMINAL_SCHEMA_VERSION =
  "windows-cuda-host-stability-qualification-terminal.v2";
export const EVENT_LOG_SOURCE_AUDIT_CAPTURE_SCHEMA_VERSION =
  "windows-event-log-source-audit-capture-envelope.v1";
export const EVENT_LOG_SOURCE_AUDIT_ADAPTER_SNAPSHOT_SCHEMA_VERSION =
  "windows-event-log-source-audit-artifact-adapter-snapshot.v1";

const RECEIPT_SCHEMA_VERSION =
  "windows-cuda-host-stability-qualification-receipt.v1";
const MANIFEST_SCHEMA_VERSION =
  "windows-cuda-host-stability-qualification-manifest.v1";
const SOURCE_AUDIT_PROJECTION_SCHEMA_VERSION =
  "windows-cuda-host-stability-source-audit-projection.v2";
const PROVISIONING_AUDIT_SCHEMA_VERSION =
  "volvence-evidence-event-log-provisioning-audit.v2";
const PROVISIONING_FAILURE_SCHEMA_VERSION =
  "volvence-evidence-event-log-provisioning-failure.v1";
const EVENT_LOG_SOURCE_CONFIG_SCHEMA_VERSION =
  "volvence-evidence-event-log-source-config.v1";
const EVENT_LOG_MACHINE_CONFIG_CORE_SCHEMA_VERSION =
  "volvence-evidence-event-log-machine-config-core.v2";
const SYNTHETIC_BACKEND_ID =
  "synthetic-node-host-qualification-test-double.v1";
const PRODUCTION_DISABLED_MESSAGE =
  "Windows host qualification production entrypoints are disabled in protocol v1";

const MODULE_PATH = fileURLToPath(import.meta.url);
const MODULE_DIR = path.dirname(MODULE_PATH);
const DEFAULT_REPOSITORY_ROOT = path.resolve(MODULE_DIR, "../../../../../");
const DEFAULT_PROTOCOL_PATH = path.join(
  MODULE_DIR,
  "protocols",
  "windows_cuda_host_stability_qualification_v1.json",
);

const RECEIPT_FILES = Object.freeze([
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
]);
const STREAM_FILES = Object.freeze([
  "streams/probe.stdout.log",
  "streams/probe.stderr.log",
]);
const MANIFEST_FILE = "010_manifest.json";
const TERMINAL_FILE = "011_terminal.json";
const COMPLETE_FILES = Object.freeze([
  ...RECEIPT_FILES,
  MANIFEST_FILE,
  TERMINAL_FILE,
  ...STREAM_FILES,
]);
const MANIFEST_INVENTORY_FILES = Object.freeze([
  ...RECEIPT_FILES,
  ...STREAM_FILES,
]);
const CRITICAL_SOURCE_PATHS = Object.freeze([
  "packages/vz-runtime/src/volvence_zero/offline_evidence/windows_cuda_host_stability_qualification.mjs",
  "packages/vz-runtime/src/volvence_zero/offline_evidence/provision_volvence_evidence_event_log.ps1",
]);
const CHANNEL_NAMES = Object.freeze(["Application", "System"]);
const OPERATOR_DECLARATION_FIELDS = Object.freeze([
  "bios_updated_with_microcode_at_least_0x12f",
  "intel_defaults_loaded",
  "cold_boot_completed_after_firmware_change",
  "xmp_disabled",
  "cpu_overclock_disabled",
  "undervolt_disabled",
  "memory_tuning_disabled",
  "same_physical_chassis_as_host_block_receipt",
]);
const FAILURE_CODE_ORDER = Object.freeze([
  "synthetic_test_backend_not_evidence",
  "operator_declaration_incomplete",
  "microcode_revision_encoding_invalid",
  "microcode_revision_decode_mismatch",
  "microcode_revision_below_minimum",
  "event_log_source_before_nonconformant",
  "event_log_source_after_nonconformant",
  "event_log_source_configuration_drift",
  "machine_identity_drift",
  "boot_identity_drift",
  "event_log_continuity_lost",
  "probe_execution_failed",
  "probe_descendants_not_quiesced",
  "cooldown_window_too_short",
  "terminal_tail_window_too_short",
  "event_log_cleared_or_rolled_over",
  "new_whea_event",
  "new_bugcheck_or_unexpected_shutdown",
  "new_gpu_driver_fault",
  "probe_process_crash_event",
]);

class JsonInteger {
  constructor(raw) {
    if (raw === "-0") throw new SyntaxError("JSON integer -0 is not canonical");
    this.raw = raw;
    this.value = Number(raw);
    if (!Number.isSafeInteger(this.value)) {
      throw new RangeError("JSON integer exceeds the exact safe range");
    }
    Object.freeze(this);
  }
}

class StrictJsonParser {
  constructor(text, label) {
    this.text = text;
    this.label = label;
    this.index = 0;
  }

  parse() {
    this.#skipWhitespace();
    const value = this.#parseValue();
    this.#skipWhitespace();
    if (this.index !== this.text.length) this.#fail("trailing content");
    return value;
  }

  #parseValue() {
    const character = this.text[this.index];
    if (character === "{") return this.#parseObject();
    if (character === "[") return this.#parseArray();
    if (character === '"') return this.#parseString();
    if (character === "t") return this.#parseLiteral("true", true);
    if (character === "f") return this.#parseLiteral("false", false);
    if (character === "n") return this.#parseLiteral("null", null);
    if (character === "-" || (character >= "0" && character <= "9")) {
      return this.#parseInteger();
    }
    this.#fail("invalid JSON value");
  }

  #parseObject() {
    this.index += 1;
    const result = Object.create(null);
    const keys = new Set();
    this.#skipWhitespace();
    if (this.text[this.index] === "}") {
      this.index += 1;
      return result;
    }
    while (true) {
      this.#skipWhitespace();
      if (this.text[this.index] !== '"') this.#fail("object key must be a string");
      const key = this.#parseString();
      if (keys.has(key)) this.#fail(`duplicate object key ${JSON.stringify(key)}`);
      keys.add(key);
      this.#skipWhitespace();
      if (this.text[this.index] !== ":") this.#fail("missing object colon");
      this.index += 1;
      this.#skipWhitespace();
      result[key] = this.#parseValue();
      this.#skipWhitespace();
      if (this.text[this.index] === "}") {
        this.index += 1;
        return result;
      }
      if (this.text[this.index] !== ",") this.#fail("missing object comma");
      this.index += 1;
    }
  }

  #parseArray() {
    this.index += 1;
    const result = [];
    this.#skipWhitespace();
    if (this.text[this.index] === "]") {
      this.index += 1;
      return result;
    }
    while (true) {
      this.#skipWhitespace();
      result.push(this.#parseValue());
      this.#skipWhitespace();
      if (this.text[this.index] === "]") {
        this.index += 1;
        return result;
      }
      if (this.text[this.index] !== ",") this.#fail("missing array comma");
      this.index += 1;
    }
  }

  #parseString() {
    const start = this.index;
    this.index += 1;
    let escaped = false;
    while (this.index < this.text.length) {
      const character = this.text[this.index];
      if (!escaped && character === '"') {
        this.index += 1;
        try {
          return JSON.parse(this.text.slice(start, this.index));
        } catch (error) {
          throw new SyntaxError(`${this.label}: invalid JSON string at byte ${start}`, {
            cause: error,
          });
        }
      }
      if (!escaped && character.charCodeAt(0) < 0x20) {
        this.#fail("unescaped control character");
      }
      if (!escaped && character === "\\") escaped = true;
      else escaped = false;
      this.index += 1;
    }
    this.#fail("unterminated string");
  }

  #parseInteger() {
    const match = /^-?(?:0|[1-9]\d*)/u.exec(this.text.slice(this.index));
    if (match === null) this.#fail("invalid JSON number");
    this.index += match[0].length;
    const next = this.text[this.index];
    if (next === "." || next === "e" || next === "E") {
      this.#fail("noncanonical/non-integer JSON number");
    }
    return new JsonInteger(match[0]);
  }

  #parseLiteral(raw, value) {
    if (!this.text.startsWith(raw, this.index)) this.#fail(`invalid ${raw} literal`);
    this.index += raw.length;
    return value;
  }

  #skipWhitespace() {
    while (
      this.index < this.text.length &&
      [" ", "\n", "\r", "\t"].includes(this.text[this.index])
    ) {
      this.index += 1;
    }
  }

  #fail(message) {
    throw new SyntaxError(`${this.label}: ${message} at byte ${this.index}`);
  }
}

function parseJsonStrict(text, label) {
  if (typeof text !== "string") throw new TypeError(`${label} must be text`);
  return new StrictJsonParser(text, label).parse();
}

function canonicalJson(value) {
  if (value instanceof JsonInteger) return value.raw;
  if (value === null) return "null";
  if (typeof value === "string") return JSON.stringify(value);
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") {
    if (!Number.isSafeInteger(value)) throw new TypeError("JSON numbers must be safe integers");
    return String(value);
  }
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (typeof value === "object") {
    return `{${Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`)
      .join(",")}}`;
  }
  throw new TypeError(`unsupported JSON value type: ${typeof value}`);
}

function canonicalBytes(value, newline = true) {
  return Buffer.from(canonicalJson(value) + (newline ? "\n" : ""), "utf8");
}

function materializeJsonIntegers(value) {
  if (value instanceof JsonInteger) return value.value;
  if (Array.isArray(value)) return value.map(materializeJsonIntegers);
  if (value !== null && typeof value === "object") {
    const materialized = Object.create(null);
    for (const [key, item] of Object.entries(value)) {
      materialized[key] = materializeJsonIntegers(item);
    }
    return materialized;
  }
  return value;
}

function sha256Bytes(payload) {
  return crypto.createHash("sha256").update(payload).digest("hex");
}

function contentId(value) {
  return sha256Bytes(canonicalBytes(value, false));
}

function domainSeparatedSha256(domain, components) {
  const hash = crypto.createHash("sha256");
  hash.update(Buffer.from(domain, "utf8"));
  for (const component of components) {
    const payload = Buffer.from(component, "utf8");
    const length = Buffer.alloc(8);
    length.writeBigUInt64BE(BigInt(payload.length));
    hash.update(length);
    hash.update(payload);
  }
  return hash.digest("hex");
}

function requireObject(value, label) {
  if (
    value === null ||
    Array.isArray(value) ||
    typeof value !== "object" ||
    value instanceof JsonInteger
  ) {
    throw new TypeError(`${label} must be an object`);
  }
  return value;
}

function requireArray(value, label) {
  if (!Array.isArray(value)) throw new TypeError(`${label} must be an array`);
  return value;
}

function requireText(value, label) {
  if (typeof value !== "string" || value.trim() === "") {
    throw new TypeError(`${label} must be nonempty text`);
  }
  return value;
}

function requireString(value, label) {
  if (typeof value !== "string") throw new TypeError(`${label} must be a string`);
  return value;
}

function requireBoolean(value, label) {
  if (typeof value !== "boolean") throw new TypeError(`${label} must be a boolean`);
  return value;
}

function requireInteger(value, label) {
  if (value instanceof JsonInteger) return value.value;
  if (!Number.isSafeInteger(value)) throw new TypeError(`${label} must be a safe integer`);
  return value;
}

function requireSha256(value, label) {
  const text = requireText(value, label);
  if (!/^[0-9a-f]{64}$/u.test(text)) {
    throw new Error(`${label} must be a lowercase SHA-256`);
  }
  return text;
}

function requireUtcTimestamp(value, label) {
  const text = requireText(value, label);
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/u.test(text)) {
    throw new Error(`${label} must be canonical whole-second UTC text`);
  }
  const milliseconds = Date.parse(text);
  if (!Number.isSafeInteger(milliseconds) || new Date(milliseconds).toISOString() !== text.replace("Z", ".000Z")) {
    throw new Error(`${label} is not a valid UTC timestamp`);
  }
  return milliseconds;
}

function requireProvisionerUtcTimestamp(value, label) {
  const text = requireText(value, label);
  const match = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})\.(\d{7})Z$/u.exec(
    text,
  );
  if (match === null) {
    throw new Error(`${label} must be Windows PowerShell seven-digit UTC text`);
  }
  const [, year, month, day, hour, minute, second, fraction] = match;
  const milliseconds = Date.UTC(
    Number(year),
    Number(month) - 1,
    Number(day),
    Number(hour),
    Number(minute),
    Number(second),
    Number(fraction.slice(0, 3)),
  );
  if (!Number.isSafeInteger(milliseconds)) throw new Error(`${label} is invalid`);
  const expected = `${year}-${month}-${day}T${hour}:${minute}:${second}.${fraction.slice(0, 3)}Z`;
  if (new Date(milliseconds).toISOString() !== expected) {
    throw new Error(`${label} is not a valid calendar timestamp`);
  }
  return BigInt(milliseconds) * 10_000n + BigInt(fraction.slice(3));
}

function deepFreeze(value) {
  if (value !== null && typeof value === "object" && !Object.isFrozen(value)) {
    for (const item of Object.values(value)) deepFreeze(item);
    Object.freeze(value);
  }
  return value;
}

function exactKeys(value, expected, label) {
  const actual = Object.keys(requireObject(value, label)).sort();
  const wanted = [...expected].sort();
  if (
    actual.length !== wanted.length ||
    actual.some((key, index) => key !== wanted[index])
  ) {
    throw new Error(`${label} keys drifted`);
  }
}

function orderedExactKeys(value, expected, label) {
  const actual = Object.keys(requireObject(value, label));
  if (
    actual.length !== expected.length ||
    actual.some((key, index) => key !== expected[index])
  ) {
    throw new Error(`${label} ordered keys drifted`);
  }
}

function deepExact(actual, expected, label) {
  if (actual instanceof JsonInteger && typeof expected === "number") {
    if (actual.value !== expected) throw new Error(`${label} value drift`);
    return;
  }
  if (Array.isArray(expected)) {
    const array = requireArray(actual, label);
    if (array.length !== expected.length) throw new Error(`${label} length drift`);
    expected.forEach((item, index) => deepExact(array[index], item, `${label}[${index}]`));
    return;
  }
  if (expected !== null && typeof expected === "object") {
    const object = requireObject(actual, label);
    exactKeys(object, Object.keys(expected), label);
    for (const key of Object.keys(expected)) {
      deepExact(object[key], expected[key], `${label}.${key}`);
    }
    return;
  }
  if (typeof actual !== typeof expected || actual !== expected) {
    throw new Error(`${label} value/type drift`);
  }
}

function orderedDeepExact(actual, expected, label) {
  if (Array.isArray(expected)) {
    const array = requireArray(actual, label);
    if (array.length !== expected.length) throw new Error(`${label} length drift`);
    expected.forEach((item, index) =>
      orderedDeepExact(array[index], item, `${label}[${index}]`),
    );
    return;
  }
  if (expected !== null && typeof expected === "object") {
    const object = requireObject(actual, label);
    orderedExactKeys(object, Object.keys(expected), label);
    for (const key of Object.keys(expected)) {
      orderedDeepExact(object[key], expected[key], `${label}.${key}`);
    }
    return;
  }
  if (typeof actual !== typeof expected || actual !== expected) {
    throw new Error(`${label} value/type drift`);
  }
}

function loadStrictJsonFile(filePath, label, canonicalRequired = false) {
  const stat = fs.lstatSync(filePath);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1) {
    throw new Error(`${label} must be one regular, non-linked file`);
  }
  const raw = fs.readFileSync(filePath);
  if (raw.subarray(0, 3).equals(Buffer.from([0xef, 0xbb, 0xbf]))) {
    throw new Error(`${label} must not contain a UTF-8 BOM`);
  }
  const text = new TextDecoder("utf-8", { fatal: true }).decode(raw);
  const parsed = parseJsonStrict(text, label);
  if (canonicalRequired && !raw.equals(canonicalBytes(parsed))) {
    throw new Error(`${label} is not canonical UTF-8/LF JSON`);
  }
  return {
    value: materializeJsonIntegers(parsed),
    raw,
    rawSha256: sha256Bytes(raw),
  };
}

function writeCreateFile(filePath, payload) {
  const descriptor = fs.openSync(filePath, "wx", 0o600);
  try {
    let offset = 0;
    while (offset < payload.length) offset += fs.writeSync(descriptor, payload, offset);
    fs.fsyncSync(descriptor);
  } finally {
    fs.closeSync(descriptor);
  }
}

function writeCreateJson(filePath, value) {
  const raw = canonicalBytes(value);
  writeCreateFile(filePath, raw);
  return { raw, rawSha256: sha256Bytes(raw) };
}

function sourceTextIdentity(filePath) {
  const raw = fs.readFileSync(filePath);
  if (raw.subarray(0, 3).equals(Buffer.from([0xef, 0xbb, 0xbf]))) {
    throw new Error(`critical source carries a UTF-8 BOM: ${filePath}`);
  }
  const text = new TextDecoder("utf-8", { fatal: true }).decode(raw);
  const lfCanonicalBytes = Buffer.from(text.replace(/\r\n?/gu, "\n"), "utf8");
  return {
    lfCanonicalByteCount: lfCanonicalBytes.length,
    lfCanonicalSha256: sha256Bytes(lfCanonicalBytes),
  };
}

function sourceTextSha256(filePath) {
  return sourceTextIdentity(filePath).lfCanonicalSha256;
}

function requireRelativePosixPath(value, label) {
  const text = requireText(value, label);
  if (
    text.includes("\\") ||
    text.startsWith("/") ||
    text.split("/").some((part) => part === "" || part === "." || part === "..")
  ) {
    throw new Error(`${label} must be a canonical relative POSIX path`);
  }
  return text;
}

function resolveRepositoryPath(repositoryRoot, relativePosixPath) {
  const relative = requireRelativePosixPath(relativePosixPath, "repository relative path");
  const root = fs.realpathSync(repositoryRoot);
  const candidate = path.resolve(root, ...relative.split("/"));
  const nativeRelative = path.relative(root, candidate);
  if (nativeRelative.startsWith("..") || path.isAbsolute(nativeRelative)) {
    throw new Error(`repository path escapes root: ${relative}`);
  }
  return candidate;
}

function validateProtocol(protocol) {
  exactKeys(
    protocol,
    [
      "schema_version",
      "owner",
      "host_block",
      "consumer_compatibility",
      "scope",
      "firmware_gate",
      "event_log_source",
      "event_log",
      "probe",
      "timing",
      "source_hash_mode",
      "source_sha256",
      "output_contract",
      "evidence_firewall",
      "claim_boundary",
    ],
    "qualification protocol",
  );
  if (protocol.schema_version !== HOST_QUALIFICATION_PROTOCOL_SCHEMA_VERSION) {
    throw new Error("qualification protocol schema drift");
  }
  if (
    protocol.owner.qualification_owner_wheel !== "vz-runtime" ||
    protocol.owner.qualification_owner !==
      "volvence_zero.offline_evidence.windows_cuda_host_stability_qualification" ||
    protocol.owner.mode !== "offline_host_stability_qualification_publisher" ||
    protocol.owner.distribution_scope !== "repository-source-checkout-only"
  ) {
    throw new Error("qualification owner contract drift");
  }
  requireSha256(protocol.host_block.receipt_raw_sha256, "host-block receipt hash");
  requireRelativePosixPath(
    protocol.host_block.receipt_relative_path,
    "host-block receipt path",
  );
  if (
    protocol.host_block.same_physical_chassis_machine_verifiable_from_receipt !== false ||
    protocol.host_block.same_physical_chassis_requires_operator_declaration !== true ||
    protocol.host_block.qualification_replaces_host_block_automatically !== false
  ) {
    throw new Error("host-block boundary drift");
  }
  if (
    protocol.consumer_compatibility.published_terminal_schema_version !==
      HOST_QUALIFICATION_TERMINAL_SCHEMA_VERSION ||
    protocol.consumer_compatibility.current_outer_accepts_published_terminal !== false ||
    protocol.consumer_compatibility.current_outer_full_artifact_consumer_present !== false ||
    protocol.consumer_compatibility.terminal_eligibility_self_report_authoritative !== false ||
    protocol.consumer_compatibility.only_full_validator_return_can_be_consumed !== true
  ) {
    throw new Error("qualification consumer compatibility drift");
  }
  if (
    protocol.scope.scope_id_method !== "sha256_domain_separated_length_framed_v1" ||
    protocol.scope.attempt_number !== 1 ||
    protocol.scope.attempt_budget !== 1 ||
    protocol.scope.retry_budget !== 0 ||
    protocol.scope.failed_terminal_retry_permitted !== false ||
    protocol.scope.incomplete_consumed_retry_permitted !== false ||
    protocol.scope.scope_excludes_refreshable_audit_timestamp_and_raw_sha256 !== true
  ) {
    throw new Error("qualification scope contract drift");
  }
  deepExact(
    protocol.scope.scope_id_components,
    [
      "qualification_protocol_id",
      "host_block_receipt_raw_sha256",
      "machine_identity_sha256",
      "boot_identity_sha256",
      "source_machine_config_content_id",
      "operator_declaration_id",
      "execution_backend_id",
    ],
    "protocol.scope.scope_id_components",
  );
  if (
    protocol.firmware_gate.microcode_registry_encoding !==
      "little_endian_unsigned_32_bit_hex" ||
    protocol.firmware_gate.minimum_microcode_revision_integer !== 303 ||
    protocol.firmware_gate.comparison !== "decoded_integer_greater_than_or_equal" ||
    protocol.firmware_gate.operator_declaration_is_cryptographic_signature !== false
  ) {
    throw new Error("qualification firmware gate drift");
  }
  deepExact(
    protocol.firmware_gate.required_operator_declarations,
    OPERATOR_DECLARATION_FIELDS,
    "protocol.firmware_gate.required_operator_declarations",
  );
  exactKeys(
    protocol.event_log_source,
    [
      "provisioner_relative_path",
      "provisioning_audit_schema_version",
      "qualification_projection_schema_version",
      "source_config_schema_version",
      "raw_audit_capture_schema_version",
      "raw_audit_adapter_snapshot_schema_version",
      "raw_audit_capture_roles",
      "maximum_raw_audit_bytes",
      "log_name",
      "source_name",
      "required_mode",
      "source_created_this_invocation_required",
      "requires_cold_or_service_refresh_required",
      "refresh_disposition_required",
      "audit_overall_conformant_required",
      "audit_nonconformance_exit_code",
      "process_failure_exit_code",
      "qualification_not_authorized_required",
      "fresh_post_refresh_audit_required",
      "required_conformance_booleans",
      "required_provider_membership_transition_disposition",
      "required_fixed_contract",
      "continuous_stability_proven_required",
      "audit_exit_code_alone_proves_conformance",
      "refresh_requirement_field_alone_proves_service_refresh_or_cold_boot",
      "machine_config_content_id_alone_proves_conformance",
      "allow_source_creation_required_for_absent_provision",
      "allow_source_creation_proves_first_bootstrap",
      "provisioning_mutation_transactional",
      "partial_failure_may_leave_source_registered",
      "cmdlet_provenance_authoritative",
      "module_qualification_proves_trusted_execution",
      "synthetic_projection_matches_provisioner_raw_schema",
      "raw_audit_artifact_adapter_core_implemented",
      "raw_audit_capture_metadata_authoritative",
      "raw_audit_artifact_self_consistency_proves_real_observation",
      "production_qualification_receipt_raw_audit_binding_implemented",
      "production_raw_audit_acquisition_implemented",
      "independent_control_plane_reobservation_implemented",
      "production_requires_full_raw_audit_binding_and_independent_revalidation",
      "pre_and_post_machine_config_content_id_equal_required",
      "qualification_may_invoke_provision_mode",
      "provision_output_can_authorize_qualification",
    ],
    "protocol.event_log_source",
  );
  if (
    protocol.event_log_source.provisioner_relative_path !== CRITICAL_SOURCE_PATHS[1] ||
    protocol.event_log_source.provisioning_audit_schema_version !==
      PROVISIONING_AUDIT_SCHEMA_VERSION ||
    protocol.event_log_source.source_config_schema_version !==
      EVENT_LOG_SOURCE_CONFIG_SCHEMA_VERSION ||
    protocol.event_log_source.raw_audit_capture_schema_version !==
      EVENT_LOG_SOURCE_AUDIT_CAPTURE_SCHEMA_VERSION ||
    protocol.event_log_source.raw_audit_adapter_snapshot_schema_version !==
      EVENT_LOG_SOURCE_AUDIT_ADAPTER_SNAPSHOT_SCHEMA_VERSION ||
    requireInteger(
      protocol.event_log_source.maximum_raw_audit_bytes,
      "maximum raw Audit bytes",
    ) !== 1_048_576 ||
    protocol.event_log_source.log_name !== "Application" ||
    protocol.event_log_source.source_name !== "VolvenceEvidence" ||
    protocol.event_log_source.required_mode !== "Audit" ||
    protocol.event_log_source.qualification_projection_schema_version !==
      SOURCE_AUDIT_PROJECTION_SCHEMA_VERSION ||
    protocol.event_log_source.source_created_this_invocation_required !== false ||
    protocol.event_log_source.requires_cold_or_service_refresh_required !== null ||
    protocol.event_log_source.refresh_disposition_required !==
      "not_observed_or_proven_by_this_invocation" ||
    protocol.event_log_source.audit_overall_conformant_required !== true ||
    protocol.event_log_source.audit_nonconformance_exit_code !== 2 ||
    protocol.event_log_source.process_failure_exit_code !== 3 ||
    protocol.event_log_source.qualification_not_authorized_required !== true ||
    protocol.event_log_source.fresh_post_refresh_audit_required !== true ||
    protocol.event_log_source.audit_exit_code_alone_proves_conformance !== false ||
    protocol.event_log_source
      .refresh_requirement_field_alone_proves_service_refresh_or_cold_boot !== false ||
    protocol.event_log_source.machine_config_content_id_alone_proves_conformance !== false ||
    protocol.event_log_source.allow_source_creation_required_for_absent_provision !== true ||
    protocol.event_log_source.allow_source_creation_proves_first_bootstrap !== false ||
    protocol.event_log_source.provisioning_mutation_transactional !== false ||
    protocol.event_log_source.partial_failure_may_leave_source_registered !== true ||
    protocol.event_log_source.cmdlet_provenance_authoritative !== false ||
    protocol.event_log_source.module_qualification_proves_trusted_execution !== false ||
    protocol.event_log_source.continuous_stability_proven_required !== false ||
    protocol.event_log_source.required_provider_membership_transition_disposition !==
      "unchanged" ||
    protocol.event_log_source.synthetic_projection_matches_provisioner_raw_schema !== false ||
    protocol.event_log_source.raw_audit_artifact_adapter_core_implemented !== true ||
    protocol.event_log_source.raw_audit_capture_metadata_authoritative !== false ||
    protocol.event_log_source
      .raw_audit_artifact_self_consistency_proves_real_observation !== false ||
    protocol.event_log_source
      .production_qualification_receipt_raw_audit_binding_implemented !== false ||
    protocol.event_log_source.production_raw_audit_acquisition_implemented !== false ||
    protocol.event_log_source.independent_control_plane_reobservation_implemented !== false ||
    protocol.event_log_source
      .production_requires_full_raw_audit_binding_and_independent_revalidation !== true ||
    protocol.event_log_source.pre_and_post_machine_config_content_id_equal_required !== true ||
    protocol.event_log_source.qualification_may_invoke_provision_mode !== false ||
    protocol.event_log_source.provision_output_can_authorize_qualification !== false
  ) {
    throw new Error("qualification Event Log source gate drift");
  }
  deepExact(
    protocol.event_log_source.raw_audit_capture_roles,
    ["qualification_source_audit_before", "qualification_source_audit_after"],
    "protocol.event_log_source.raw_audit_capture_roles",
  );
  const requiredFixedContract = requireObject(
    protocol.event_log_source.required_fixed_contract,
    "protocol.event_log_source.required_fixed_contract",
  );
  exactKeys(
    requiredFixedContract,
    [
      "config_schema_version",
      "provisioning_owner",
      "downstream_authorization",
      "powershell",
      "registry_view",
      "log_name",
      "source_name",
      "write_semantics",
      "application_registry_subkey",
      "source_registry_subkey",
      "source_values",
      "source_acl_sddl",
      "source_acl_sha256",
      "source_owner_sid",
      "audit_nonconformance_exit_code",
      "process_failure_exit_code",
      "allow_source_creation_is_operator_intent_not_history_proof",
      "module_qualified_cmdlets_required",
    ],
    "protocol.event_log_source.required_fixed_contract",
  );
  if (
    requiredFixedContract.config_schema_version !== EVENT_LOG_SOURCE_CONFIG_SCHEMA_VERSION ||
    requiredFixedContract.log_name !== protocol.event_log_source.log_name ||
    requiredFixedContract.source_name !== protocol.event_log_source.source_name ||
    requiredFixedContract.registry_view !== "Registry64" ||
    requiredFixedContract.audit_nonconformance_exit_code !== 2 ||
    requiredFixedContract.process_failure_exit_code !== 3 ||
    requiredFixedContract.allow_source_creation_is_operator_intent_not_history_proof !== true ||
    requiredFixedContract.module_qualified_cmdlets_required !== true
  ) {
    throw new Error("qualification Event Log fixed contract drift");
  }
  requireText(requiredFixedContract.source_acl_sddl, "required source ACL SDDL");
  requireSha256(requiredFixedContract.source_acl_sha256, "required source ACL SHA-256");
  if (
    sha256Bytes(Buffer.from(requiredFixedContract.source_acl_sddl, "utf8")) !==
    requiredFixedContract.source_acl_sha256
  ) {
    throw new Error("qualification Event Log source ACL digest drift");
  }
  requireText(requiredFixedContract.source_owner_sid, "required source owner SID");
  requireArray(requiredFixedContract.source_values, "required source values").forEach(
    (item, index) => {
      exactKeys(item, ["name", "kind", "data"], `required source values[${index}]`);
      validateRegistryValueData(
        requireText(item.kind, `required source values[${index}].kind`),
        item.data,
        `required source values[${index}].data`,
      );
      requireString(item.name, `required source values[${index}].name`);
    },
  );
  deepExact(
    protocol.event_log_source.required_conformance_booleans,
    [
      "source_before.source_configuration_exact",
      "source_after.source_configuration_exact",
      "application_channel_before.log_name_exact",
      "application_channel_before.enabled",
      "application_channel_before.classic_log",
      "application_channel_before.circular_log_mode",
      "application_channel_before.positive_maximum_size",
      "application_channel_before.source_provider_membership_present",
      "application_channel_after.log_name_exact",
      "application_channel_after.enabled",
      "application_channel_after.classic_log",
      "application_channel_after.circular_log_mode",
      "application_channel_after.positive_maximum_size",
      "application_channel_after.source_provider_membership_present",
      "application_channel_full_endpoint_equal",
      "application_channel_stable_projection_endpoint_equal",
      "application_channel_provider_membership_transition.allowed_for_source_creation",
      "application_registry_endpoint_equal",
      "source_registry_endpoint_equal",
    ],
    "protocol.event_log_source.required_conformance_booleans",
  );
  deepExact(protocol.event_log.channels, CHANNEL_NAMES, "protocol.event_log.channels");
  if (
    protocol.event_log.required_log_mode !== "Circular" ||
    protocol.event_log.max_new_records_per_channel_per_window !== 4096 ||
    protocol.event_log.baseline_boundary_xml_hash_required !== true ||
    protocol.event_log.contiguous_record_ids_required !== true ||
    protocol.event_log.same_machine_required !== true ||
    protocol.event_log.same_boot_required !== true
  ) {
    throw new Error("qualification Event Log contract drift");
  }
  if (
    protocol.probe.synthetic_test_backend_id !== SYNTHETIC_BACKEND_ID ||
    protocol.probe.production_entrypoints_enabled !== false ||
    protocol.probe.production_probe_implemented !== false ||
    protocol.probe.synthetic_artifacts_are_real_host_observations !== false ||
    protocol.probe.synthetic_artifacts_can_be_eligible !== false ||
    protocol.probe.launch_receipt_fsync_before_process_creation_required !== true
  ) {
    throw new Error("qualification probe authorization drift");
  }
  if (
    requireInteger(protocol.timing.minimum_cooldown_seconds, "minimum cooldown") !== 300 ||
    requireInteger(protocol.timing.minimum_terminal_tail_seconds, "minimum terminal tail") !== 120
  ) {
    throw new Error("qualification timing contract drift");
  }
  if (protocol.source_hash_mode !== "utf8_lf_canonical_v1") {
    throw new Error("qualification source hash mode drift");
  }
  const sourceHashes = requireObject(protocol.source_sha256, "protocol.source_sha256");
  deepExact(Object.keys(sourceHashes), CRITICAL_SOURCE_PATHS, "critical source path set/order");
  for (const sourcePath of CRITICAL_SOURCE_PATHS) {
    requireSha256(sourceHashes[sourcePath], `protocol.source_sha256.${sourcePath}`);
  }
  deepExact(
    protocol.output_contract.receipt_chain_sequences,
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "protocol.output_contract.receipt_chain_sequences",
  );
  deepExact(
    protocol.output_contract.manifest_inventory_files,
    MANIFEST_INVENTORY_FILES,
    "protocol.output_contract.manifest_inventory_files",
  );
  deepExact(
    protocol.output_contract.complete_files,
    COMPLETE_FILES,
    "protocol.output_contract.complete_files",
  );
  if (
    protocol.output_contract.create_only !== true ||
    protocol.output_contract.single_file_fsync !== true ||
    protocol.output_contract.manifest_sequence !== 10 ||
    protocol.output_contract.terminal_sequence !== 11 ||
    protocol.output_contract.exact_entry_set_required !== true ||
    protocol.output_contract.regular_files_only !== true ||
    protocol.output_contract.symlinks_and_reparse_points_forbidden !== true ||
    protocol.output_contract.full_validator_recomputes_all_derived_criteria !== true
  ) {
    throw new Error("qualification output contract drift");
  }
  if (
    protocol.evidence_firewall.production_publisher_enabled !== false ||
    protocol.evidence_firewall.current_outer_consumer_enabled !== false ||
    protocol.evidence_firewall.synthetic_test_backend_non_evidence !== true ||
    protocol.evidence_firewall.synthetic_validated_eligibility_always_false !== true ||
    protocol.evidence_firewall.production_raw_audit_adapter_present !== false ||
    protocol.evidence_firewall.provisioning_endpoint_equality_proves_continuous_stability !==
      false ||
    protocol.evidence_firewall
      .provisioner_module_qualification_proves_trusted_execution !== false ||
    protocol.evidence_firewall.raw_audit_artifact_adapter_core_present !== true ||
    protocol.evidence_firewall
      .raw_audit_artifact_self_consistency_proves_real_observation !== false ||
    protocol.evidence_firewall.raw_event_xml_independently_parsed !== false ||
    protocol.evidence_firewall.normalized_event_fields_independently_authenticated !== false ||
    protocol.evidence_firewall.cuda_execution_authorized !== false ||
    protocol.evidence_firewall.four_capability_claim_authorized !== false ||
    protocol.evidence_firewall.formal_evidence_authorized !== false
  ) {
    throw new Error("qualification evidence firewall drift");
  }
  requireText(protocol.claim_boundary, "protocol.claim_boundary");
}

function loadProtocol({ protocolPath, repositoryRoot, verifySources }) {
  if (fs.realpathSync(protocolPath) !== fs.realpathSync(DEFAULT_PROTOCOL_PATH)) {
    throw new Error("qualification protocol path must be the bundled reviewed protocol");
  }
  const loaded = loadStrictJsonFile(protocolPath, "qualification protocol", false);
  const protocol = loaded.value;
  validateProtocol(protocol);
  const protocolId = contentId(protocol);
  if (verifySources) {
    for (const sourcePath of CRITICAL_SOURCE_PATHS) {
      const absolutePath = resolveRepositoryPath(repositoryRoot, sourcePath);
      const sourceStat = fs.lstatSync(absolutePath);
      if (!sourceStat.isFile() || sourceStat.isSymbolicLink() || sourceStat.nlink !== 1) {
        throw new Error(`critical source must be a regular single-link file: ${sourcePath}`);
      }
      const repositoryReal = fs.realpathSync(repositoryRoot);
      const sourceReal = fs.realpathSync(absolutePath);
      const sourceRelative = path.relative(repositoryReal, sourceReal);
      if (sourceRelative.startsWith("..") || path.isAbsolute(sourceRelative)) {
        throw new Error(`critical source escapes repository root: ${sourcePath}`);
      }
      const actual = sourceTextSha256(absolutePath);
      if (actual !== protocol.source_sha256[sourcePath]) {
        throw new Error(`qualification critical source hash drift: ${sourcePath}`);
      }
    }
    const blockPath = resolveRepositoryPath(
      repositoryRoot,
      protocol.host_block.receipt_relative_path,
    );
    const blockStat = fs.lstatSync(blockPath);
    if (!blockStat.isFile() || blockStat.isSymbolicLink() || blockStat.nlink !== 1) {
      throw new Error("host-block receipt must be a regular single-link file");
    }
    const repositoryReal = fs.realpathSync(repositoryRoot);
    const blockReal = fs.realpathSync(blockPath);
    const blockRelative = path.relative(repositoryReal, blockReal);
    if (blockRelative.startsWith("..") || path.isAbsolute(blockRelative)) {
      throw new Error("host-block receipt escapes repository root");
    }
    const blockHash = sha256Bytes(fs.readFileSync(blockPath));
    if (blockHash !== protocol.host_block.receipt_raw_sha256) {
      throw new Error("host-block receipt raw SHA-256 drift");
    }
  }
  return {
    protocol,
    protocolId,
    protocolRawSha256: loaded.rawSha256,
  };
}

function decodeMicrocodeLittleEndian(rawHex) {
  if (!/^[0-9a-f]{8}$/u.test(rawHex)) {
    throw new Error("microcode raw value must be exactly four lowercase little-endian bytes");
  }
  return Buffer.from(rawHex, "hex").readUInt32LE(0);
}

function makeCursor(logName, newestRecordId, capturedAtUtc, machineId, bootId) {
  return {
    log_name: logName,
    enabled: true,
    record_count: newestRecordId,
    oldest_record_id: 1,
    newest_record_id: newestRecordId,
    newest_record_xml_sha256: sha256Bytes(
      Buffer.from(`${logName}:${newestRecordId}:synthetic-boundary`, "utf8"),
    ),
    maximum_size_bytes: 20_971_520,
    log_mode: "Circular",
    configuration_sha256: sha256Bytes(
      Buffer.from(`${logName}:Circular:20971520:synthetic-config`, "utf8"),
    ),
    machine_identity_sha256: machineId,
    boot_identity_sha256: bootId,
    captured_at_utc: capturedAtUtc,
  };
}

function makeSourceAuditProjection({
  observedAtUtc,
  machineId,
  bootId,
  sourceConfigurationId,
  auditExitCode,
}) {
  return {
    projection_schema_version: SOURCE_AUDIT_PROJECTION_SCHEMA_VERSION,
    claimed_input_audit_schema_version:
      "volvence-evidence-event-log-provisioning-audit.v2",
    config_schema_version: "volvence-evidence-event-log-source-config.v1",
    mode: "Audit",
    observed_at_utc: observedAtUtc,
    audit_exit_code: auditExitCode,
    overall_conformant: auditExitCode === 0,
    source_created_this_invocation: false,
    requires_cold_or_service_refresh: null,
    refresh_disposition: "not_observed_or_proven_by_this_invocation",
    qualification_not_authorized: true,
    machine_identity_sha256: machineId,
    boot_identity_sha256: bootId,
    machine_config_content_id: sourceConfigurationId,
    source_name: "VolvenceEvidence",
    log_name: "Application",
    conformance: {
      source_before: { source_configuration_exact: true },
      source_after: { source_configuration_exact: true },
      application_channel_before: {
        log_name_exact: true,
        enabled: true,
        classic_log: true,
        circular_log_mode: true,
        positive_maximum_size: true,
        source_provider_membership_present: true,
      },
      application_channel_after: {
        log_name_exact: true,
        enabled: true,
        classic_log: true,
        circular_log_mode: true,
        positive_maximum_size: true,
        source_provider_membership_present: true,
      },
      application_channel_full_endpoint_equal: true,
      application_channel_stable_projection_endpoint_equal: true,
      application_channel_provider_membership_transition: {
        disposition: "unchanged",
        allowed_for_source_creation: true,
      },
      application_registry_endpoint_equal: true,
      source_registry_endpoint_equal: true,
      continuous_stability_proven: false,
    },
    audit_exit_code_alone_proves_conformance: false,
    refresh_requirement_field_alone_proves_service_refresh_or_cold_boot: false,
    machine_config_content_id_alone_proves_conformance: false,
    full_raw_audit_bound: false,
    raw_audit_sha256: null,
    raw_audit_content_id_basis_revalidated: false,
    real_provisioner_observation: false,
  };
}

function loadStrictProvisionerJsonBytes(rawAuditBytes, maximumBytes) {
  if (!(rawAuditBytes instanceof Uint8Array)) {
    throw new TypeError("rawAuditBytes must be a Uint8Array");
  }
  const raw = Buffer.from(rawAuditBytes);
  if (raw.length === 0 || raw.length > maximumBytes) {
    throw new Error("raw Audit byte count is outside the frozen adapter budget");
  }
  if (raw.subarray(0, 3).equals(Buffer.from([0xef, 0xbb, 0xbf]))) {
    throw new Error("raw Audit must not contain a UTF-8 BOM");
  }
  const text = new TextDecoder("utf-8", { fatal: true }).decode(raw);
  if (
    !text.endsWith("\n") ||
    text.includes("\r") ||
    text.indexOf("\n") !== text.length - 1
  ) {
    throw new Error("raw Audit must be one compact JSON value followed by one LF");
  }
  const parsed = materializeJsonIntegers(
    parseJsonStrict(text.slice(0, -1), "raw Event Log Audit v2"),
  );
  const reconstructedRaw = Buffer.from(`${JSON.stringify(parsed)}\n`, "utf8");
  if (!raw.equals(reconstructedRaw)) {
    throw new Error("raw Audit must preserve compact ordered JSON bytes");
  }
  const schemaVersion = requireText(parsed.schema_version, "raw Audit.schema_version");
  if (schemaVersion === PROVISIONING_FAILURE_SCHEMA_VERSION) {
    throw new Error(
      "process failure v1 cannot be adapted as a complete Event Log Audit v2",
    );
  }
  if (schemaVersion !== PROVISIONING_AUDIT_SCHEMA_VERSION) {
    throw new Error("raw Event Log Audit schema drift");
  }
  return { value: parsed, raw, rawSha256: sha256Bytes(raw) };
}

function jsonValueEqual(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function requireCanonicalStringArray(value, label, { nonempty = false } = {}) {
  const values = requireArray(value, label).map((item, index) => {
    if (nonempty) return requireText(item, `${label}[${index}]`);
    return requireString(item, `${label}[${index}]`);
  });
  const sorted = [...values].sort();
  if (values.some((item, index) => item !== sorted[index])) {
    throw new Error(`${label} must use ordinal sort order`);
  }
  if (new Set(values).size !== values.length) {
    throw new Error(`${label} must not contain duplicates`);
  }
  return values;
}

function validateRegistryValueData(kind, data, label) {
  if (kind === "String" || kind === "ExpandString") {
    requireString(data, label);
    return;
  }
  if (kind === "MultiString") {
    requireArray(data, label).forEach((item, index) =>
      requireString(item, `${label}[${index}]`),
    );
    return;
  }
  if (kind === "Binary" || kind === "None") {
    const encoded = requireString(data, label);
    if (!/^(?:[0-9a-f]{2})*$/u.test(encoded)) {
      throw new Error(`${label} must be lowercase whole-byte hex`);
    }
    return;
  }
  if (kind === "DWord") {
    const integer = requireInteger(data, label);
    if (integer < 0 || integer > 0xffff_ffff) {
      throw new Error(`${label} is outside unsigned 32-bit range`);
    }
    return;
  }
  if (kind === "QWord") {
    const decimal = requireString(data, label);
    if (!/^(?:0|[1-9]\d*)$/u.test(decimal) || BigInt(decimal) > 0xffff_ffff_ffff_ffffn) {
      throw new Error(`${label} is not canonical unsigned 64-bit decimal text`);
    }
    return;
  }
  throw new Error(`${label} carries unsupported registry value kind ${kind}`);
}

function validateRegistryObservation(observation, label, expectedSubkey) {
  orderedExactKeys(
    observation,
    [
      "hive",
      "registry_view",
      "subkey",
      "present",
      "values",
      "security_descriptor_sddl",
      "security_descriptor_sha256",
      "owner_sid",
    ],
    label,
  );
  if (
    observation.hive !== "HKEY_LOCAL_MACHINE" ||
    observation.registry_view !== "Registry64" ||
    observation.subkey !== expectedSubkey
  ) {
    throw new Error(`${label} registry identity drift`);
  }
  requireBoolean(observation.present, `${label}.present`);
  const values = requireArray(observation.values, `${label}.values`);
  const valueNames = [];
  values.forEach((item, index) => {
    const valueLabel = `${label}.values[${index}]`;
    orderedExactKeys(item, ["name", "kind", "data"], valueLabel);
    const name = requireString(item.name, `${valueLabel}.name`);
    const kind = requireText(item.kind, `${valueLabel}.kind`);
    valueNames.push(name);
    validateRegistryValueData(kind, item.data, `${valueLabel}.data`);
  });
  const sortedNames = [...valueNames].sort();
  if (valueNames.some((name, index) => name !== sortedNames[index])) {
    throw new Error(`${label}.values must use ordinal name order`);
  }
  if (new Set(valueNames).size !== valueNames.length) {
    throw new Error(`${label}.values must not contain duplicate names`);
  }
  if (!observation.present) {
    if (
      values.length !== 0 ||
      observation.security_descriptor_sddl !== null ||
      observation.security_descriptor_sha256 !== null ||
      observation.owner_sid !== null
    ) {
      throw new Error(`${label} absent-key shape drift`);
    }
    return;
  }
  const sddl = requireText(
    observation.security_descriptor_sddl,
    `${label}.security_descriptor_sddl`,
  );
  requireSha256(
    observation.security_descriptor_sha256,
    `${label}.security_descriptor_sha256`,
  );
  if (sha256Bytes(Buffer.from(sddl, "utf8")) !== observation.security_descriptor_sha256) {
    throw new Error(`${label} security descriptor digest mismatch`);
  }
  requireText(observation.owner_sid, `${label}.owner_sid`);
}

function validateApplicationChannelObservation(channel, label) {
  orderedExactKeys(
    channel,
    [
      "log_name",
      "log_type",
      "isolation",
      "is_enabled",
      "is_classic_log",
      "log_mode",
      "maximum_size_in_bytes",
      "log_file_path",
      "owning_provider_name",
      "provider_names",
      "security_descriptor_sddl",
      "security_descriptor_sha256",
      "owner_sid",
    ],
    label,
  );
  requireText(channel.log_name, `${label}.log_name`);
  requireString(channel.log_type, `${label}.log_type`);
  requireString(channel.isolation, `${label}.isolation`);
  requireBoolean(channel.is_enabled, `${label}.is_enabled`);
  requireBoolean(channel.is_classic_log, `${label}.is_classic_log`);
  requireText(channel.log_mode, `${label}.log_mode`);
  requireInteger(channel.maximum_size_in_bytes, `${label}.maximum_size_in_bytes`);
  requireString(channel.log_file_path, `${label}.log_file_path`);
  requireString(channel.owning_provider_name, `${label}.owning_provider_name`);
  requireCanonicalStringArray(channel.provider_names, `${label}.provider_names`, {
    nonempty: true,
  });
  const sddl = requireText(
    channel.security_descriptor_sddl,
    `${label}.security_descriptor_sddl`,
  );
  requireSha256(
    channel.security_descriptor_sha256,
    `${label}.security_descriptor_sha256`,
  );
  if (sha256Bytes(Buffer.from(sddl, "utf8")) !== channel.security_descriptor_sha256) {
    throw new Error(`${label} security descriptor digest mismatch`);
  }
  requireText(channel.owner_sid, `${label}.owner_sid`);
}

function deriveSourceConformance(observation, fixedContract) {
  const sourcePresent = observation.present === true;
  const sourceValuesExact =
    sourcePresent && jsonValueEqual(observation.values, fixedContract.source_values);
  const sourceAclSddlExact =
    sourcePresent && observation.security_descriptor_sddl === fixedContract.source_acl_sddl;
  const sourceOwnerSidExact =
    sourcePresent && observation.owner_sid === fixedContract.source_owner_sid;
  return {
    source_present: sourcePresent,
    source_values_exact: sourceValuesExact,
    source_acl_sddl_exact: sourceAclSddlExact,
    source_owner_sid_exact: sourceOwnerSidExact,
    source_configuration_exact:
      sourcePresent &&
      sourceValuesExact &&
      sourceAclSddlExact &&
      sourceOwnerSidExact,
  };
}

function deriveApplicationChannelConformance(channel, fixedContract) {
  return {
    log_name_exact: channel.log_name === fixedContract.log_name,
    enabled: channel.is_enabled === true,
    classic_log: channel.is_classic_log === true,
    circular_log_mode: channel.log_mode === "Circular",
    positive_maximum_size: channel.maximum_size_in_bytes > 0,
    source_provider_membership_present: channel.provider_names.includes(
      fixedContract.source_name,
    ),
  };
}

function applicationChannelStableProjection(channel) {
  return {
    log_name: channel.log_name,
    log_type: channel.log_type,
    isolation: channel.isolation,
    is_enabled: channel.is_enabled,
    is_classic_log: channel.is_classic_log,
    log_mode: channel.log_mode,
    maximum_size_in_bytes: channel.maximum_size_in_bytes,
    log_file_path: channel.log_file_path,
    owning_provider_name: channel.owning_provider_name,
    security_descriptor_sddl: channel.security_descriptor_sddl,
    security_descriptor_sha256: channel.security_descriptor_sha256,
    owner_sid: channel.owner_sid,
  };
}

function deriveProviderMembershipTransition(before, after, sourceName) {
  const beforeProviders = before.provider_names;
  const afterProviders = after.provider_names;
  if (jsonValueEqual(beforeProviders, afterProviders)) {
    return {
      disposition: "unchanged",
      allowed_for_source_creation: true,
      before_count: beforeProviders.length,
      after_count: afterProviders.length,
    };
  }
  const expectedAfter = [...beforeProviders, sourceName].sort();
  if (jsonValueEqual(expectedAfter, afterProviders)) {
    return {
      disposition: "exact_source_name_addition",
      allowed_for_source_creation: true,
      before_count: beforeProviders.length,
      after_count: afterProviders.length,
    };
  }
  return {
    disposition: "unexpected_provider_membership_transition",
    allowed_for_source_creation: false,
    before_count: beforeProviders.length,
    after_count: afterProviders.length,
  };
}

function validateProvisionerMachine(machine, label) {
  orderedExactKeys(
    machine,
    [
      "platform_system",
      "computer_name",
      "machine_identity_sha256",
      "registry_view",
      "process_architecture",
      "powershell_edition",
      "powershell_version",
    ],
    label,
  );
  if (
    machine.platform_system !== "Windows" ||
    machine.registry_view !== "Registry64" ||
    machine.process_architecture !== "x64" ||
    machine.powershell_edition !== "Desktop" ||
    !/^5\.1(?:\.|$)/u.test(requireText(machine.powershell_version, `${label}.powershell_version`))
  ) {
    throw new Error(`${label} execution identity drift`);
  }
  requireText(machine.computer_name, `${label}.computer_name`);
  requireSha256(machine.machine_identity_sha256, `${label}.machine_identity_sha256`);
}

function validateInvokingPrincipal(principal, label) {
  orderedExactKeys(principal, ["name", "sid", "is_administrator"], label);
  requireText(principal.name, `${label}.name`);
  requireText(principal.sid, `${label}.sid`);
  requireBoolean(principal.is_administrator, `${label}.is_administrator`);
}

function validateScriptIntegrity(scriptIntegrity, protocol, label) {
  orderedExactKeys(
    scriptIntegrity,
    [
      "repository_relative_path",
      "source_hash_mode",
      "lf_canonical_byte_count",
      "observed_lf_canonical_sha256",
      "self_signature_authoritative",
      "node_protocol_pin_required",
      "trust_boundary",
    ],
    label,
  );
  const provisionerPath = protocol.event_log_source.provisioner_relative_path;
  if (
    scriptIntegrity.repository_relative_path !== provisionerPath ||
    scriptIntegrity.source_hash_mode !== protocol.source_hash_mode ||
    scriptIntegrity.observed_lf_canonical_sha256 !==
      protocol.source_sha256[provisionerPath] ||
    scriptIntegrity.self_signature_authoritative !== false ||
    scriptIntegrity.node_protocol_pin_required !== true
  ) {
    throw new Error(`${label} source-lineage drift`);
  }
  if (requireInteger(scriptIntegrity.lf_canonical_byte_count, `${label}.lf_canonical_byte_count`) <= 0) {
    throw new Error(`${label}.lf_canonical_byte_count must be positive`);
  }
  requireText(scriptIntegrity.trust_boundary, `${label}.trust_boundary`);
}

function validateCmdletProvenance(provenance, label) {
  orderedExactKeys(
    provenance,
    [
      "observations",
      "module_qualified_invocation_required",
      "powershell_executable_identity_attested",
      "provenance_authoritative",
    ],
    label,
  );
  if (
    provenance.module_qualified_invocation_required !== true ||
    provenance.powershell_executable_identity_attested !== false ||
    provenance.provenance_authoritative !== false
  ) {
    throw new Error(`${label} authority boundary drift`);
  }
  const expected = [
    ["Get-WinEvent", "Microsoft.PowerShell.Diagnostics"],
    ["New-EventLog", "Microsoft.PowerShell.Management"],
    ["Get-ItemPropertyValue", "Microsoft.PowerShell.Management"],
    ["Test-Path", "Microsoft.PowerShell.Management"],
    ["ConvertTo-Json", "Microsoft.PowerShell.Utility"],
  ];
  const observations = requireArray(provenance.observations, `${label}.observations`);
  if (observations.length !== expected.length) {
    throw new Error(`${label}.observations length drift`);
  }
  observations.forEach((observation, index) => {
    const itemLabel = `${label}.observations[${index}]`;
    orderedExactKeys(
      observation,
      [
        "command_name",
        "command_type",
        "module_name",
        "module_version",
        "module_path",
        "module_path_sha256",
        "implementing_type",
        "assembly_location",
        "assembly_sha256",
        "assembly_version",
        "assembly_public_key_token",
        "module_qualified_invocation",
        "provenance_authoritative",
      ],
      itemLabel,
    );
    const [commandName, moduleName] = expected[index];
    if (
      observation.command_name !== commandName ||
      observation.command_type !== "Cmdlet" ||
      observation.module_name !== moduleName ||
      observation.module_qualified_invocation !== `${moduleName}\\${commandName}` ||
      observation.provenance_authoritative !== false
    ) {
      throw new Error(`${itemLabel} cmdlet identity drift`);
    }
    for (const field of [
      "module_version",
      "module_path",
      "implementing_type",
      "assembly_location",
      "assembly_version",
    ]) {
      requireText(observation[field], `${itemLabel}.${field}`);
    }
    requireSha256(observation.module_path_sha256, `${itemLabel}.module_path_sha256`);
    requireSha256(observation.assembly_sha256, `${itemLabel}.assembly_sha256`);
    if (!/^(?:[0-9a-f]{2})+$/u.test(observation.assembly_public_key_token)) {
      throw new Error(`${itemLabel}.assembly_public_key_token must be lowercase hex`);
    }
  });
}

function validateFixedContract(fixedContract, protocol, label) {
  orderedDeepExact(
    fixedContract,
    protocol.event_log_source.required_fixed_contract,
    label,
  );
}

function deriveAuditConformance(observed, fixedContract) {
  const sourceBefore = deriveSourceConformance(
    observed.source_registry_before,
    fixedContract,
  );
  const sourceAfter = deriveSourceConformance(
    observed.source_registry_after,
    fixedContract,
  );
  const channelBefore = deriveApplicationChannelConformance(
    observed.application_channel_before,
    fixedContract,
  );
  const channelAfter = deriveApplicationChannelConformance(
    observed.application_channel_after,
    fixedContract,
  );
  const channelFullEndpointEqual = jsonValueEqual(
    observed.application_channel_before,
    observed.application_channel_after,
  );
  const channelStableProjectionEndpointEqual = jsonValueEqual(
    applicationChannelStableProjection(observed.application_channel_before),
    applicationChannelStableProjection(observed.application_channel_after),
  );
  const providerTransition = deriveProviderMembershipTransition(
    observed.application_channel_before,
    observed.application_channel_after,
    fixedContract.source_name,
  );
  const applicationRegistryEndpointEqual = jsonValueEqual(
    observed.application_registry_before,
    observed.application_registry_after,
  );
  const sourceRegistryEndpointEqual = jsonValueEqual(
    observed.source_registry_before,
    observed.source_registry_after,
  );
  const conformance = {
    source_before: sourceBefore,
    source_after: sourceAfter,
    application_channel_before: channelBefore,
    application_channel_after: channelAfter,
    application_channel_full_endpoint_equal: channelFullEndpointEqual,
    application_channel_stable_projection_endpoint_equal:
      channelStableProjectionEndpointEqual,
    application_channel_provider_membership_transition: providerTransition,
    application_registry_endpoint_equal: applicationRegistryEndpointEqual,
    source_registry_endpoint_equal: sourceRegistryEndpointEqual,
    continuous_stability_proven: false,
  };
  const overallConformant =
    sourceBefore.source_configuration_exact &&
    sourceAfter.source_configuration_exact &&
    channelBefore.log_name_exact &&
    channelBefore.enabled &&
    channelBefore.classic_log &&
    channelBefore.circular_log_mode &&
    channelBefore.positive_maximum_size &&
    channelBefore.source_provider_membership_present &&
    channelAfter.log_name_exact &&
    channelAfter.enabled &&
    channelAfter.classic_log &&
    channelAfter.circular_log_mode &&
    channelAfter.positive_maximum_size &&
    channelAfter.source_provider_membership_present &&
    channelFullEndpointEqual &&
    applicationRegistryEndpointEqual &&
    sourceRegistryEndpointEqual;
  return { conformance, overallConformant };
}

function deriveSourceHistoryTransition(before, after) {
  if (!before.present && after.present) return "created_during_this_invocation";
  if (before.present && !after.present) return "deletion_observed_during_this_invocation";
  if (!before.present && !after.present) return "absent_prior_history_indeterminate";
  return "present_continuity_within_invocation";
}

function decodeStrictBase64(value, label) {
  const encoded = requireText(value, label);
  if (!/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/u.test(encoded)) {
    throw new Error(`${label} must be canonical base64`);
  }
  const decoded = Buffer.from(encoded, "base64");
  if (decoded.toString("base64") !== encoded) {
    throw new Error(`${label} base64 round-trip drift`);
  }
  return decoded;
}

function parseMachineConfigBasis(audit, protocol) {
  if (
    audit.machine_config_content_id_method !==
    "sha256_of_utf8_no_bom_compact_ordered_json_core_v2"
  ) {
    throw new Error("raw Audit machine-config content-ID method drift");
  }
  const basisBytes = decodeStrictBase64(
    audit.machine_config_content_id_basis_base64,
    "raw Audit.machine_config_content_id_basis_base64",
  );
  if (
    basisBytes.length === 0 ||
    basisBytes.length > protocol.event_log_source.maximum_raw_audit_bytes
  ) {
    throw new Error("raw Audit machine-config basis is outside the frozen byte budget");
  }
  if (basisBytes.subarray(0, 3).equals(Buffer.from([0xef, 0xbb, 0xbf]))) {
    throw new Error("raw Audit machine-config basis must not contain a UTF-8 BOM");
  }
  const basisText = new TextDecoder("utf-8", { fatal: true }).decode(basisBytes);
  if (basisText.includes("\r") || basisText.includes("\n")) {
    throw new Error("raw Audit machine-config basis must be compact one-line JSON");
  }
  const basis = materializeJsonIntegers(
    parseJsonStrict(basisText, "raw Audit machine-config basis"),
  );
  if (!basisBytes.equals(Buffer.from(JSON.stringify(basis), "utf8"))) {
    throw new Error("raw Audit machine-config basis must preserve compact ordered JSON bytes");
  }
  requireSha256(audit.machine_config_content_id, "raw Audit.machine_config_content_id");
  if (sha256Bytes(basisBytes) !== audit.machine_config_content_id) {
    throw new Error("raw Audit machine-config content-ID digest mismatch");
  }
  orderedExactKeys(
    basis,
    [
      "schema_version",
      "machine",
      "script_integrity",
      "cmdlet_provenance",
      "fixed_contract",
      "observed_source_registry",
      "observed_application_registry",
      "observed_application_channel",
    ],
    "raw Audit machine-config basis",
  );
  const expectedBasis = {
    schema_version: EVENT_LOG_MACHINE_CONFIG_CORE_SCHEMA_VERSION,
    machine: audit.machine,
    script_integrity: audit.script_integrity,
    cmdlet_provenance: audit.cmdlet_provenance,
    fixed_contract: audit.fixed_contract,
    observed_source_registry: audit.observed.source_registry_after,
    observed_application_registry: audit.observed.application_registry_after,
    observed_application_channel: audit.observed.application_channel_after,
  };
  orderedDeepExact(basis, expectedBasis, "raw Audit machine-config basis reconstruction");
  return { basisBytes, basisSha256: sha256Bytes(basisBytes) };
}

function validateAuditProvisioningState(provisioning) {
  orderedDeepExact(
    provisioning,
    {
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
    "raw Audit.provisioning",
  );
}

function validateCaptureEnvelope(envelope, protocol) {
  orderedExactKeys(
    envelope,
    [
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
    ],
    "raw Audit capture envelope",
  );
  if (
    envelope.schema_version !== EVENT_LOG_SOURCE_AUDIT_CAPTURE_SCHEMA_VERSION ||
    !protocol.event_log_source.raw_audit_capture_roles.includes(envelope.capture_role) ||
    envelope.capture_authoritative !== false
  ) {
    throw new Error("raw Audit capture envelope contract drift");
  }
  requireSha256(envelope.stdout_raw_sha256, "capture.stdout_raw_sha256");
  const stdoutByteCount = requireInteger(
    envelope.stdout_byte_count,
    "capture.stdout_byte_count",
  );
  if (
    stdoutByteCount <= 0 ||
    stdoutByteCount > protocol.event_log_source.maximum_raw_audit_bytes
  ) {
    throw new Error("capture.stdout_byte_count is outside the frozen adapter budget");
  }
  requireSha256(envelope.stderr_raw_sha256, "capture.stderr_raw_sha256");
  if (
    requireInteger(envelope.stderr_byte_count, "capture.stderr_byte_count") !== 0 ||
    envelope.stderr_raw_sha256 !== sha256Bytes(Buffer.alloc(0))
  ) {
    throw new Error("complete Audit v2 capture must have empty stderr");
  }
  const exitCode = requireInteger(envelope.process_exit_code, "capture.process_exit_code");
  if (![0, 2].includes(exitCode)) {
    throw new Error("Audit v2 capture process exit code must be 0 or 2");
  }
  const started = requireProvisionerUtcTimestamp(
    envelope.process_started_at_utc,
    "capture.process_started_at_utc",
  );
  const exited = requireProvisionerUtcTimestamp(
    envelope.process_exited_at_utc,
    "capture.process_exited_at_utc",
  );
  const captured = requireProvisionerUtcTimestamp(
    envelope.stdout_captured_at_utc,
    "capture.stdout_captured_at_utc",
  );
  if (started > exited || exited > captured) {
    throw new Error("raw Audit capture chronology drift");
  }
  requireSha256(envelope.machine_identity_sha256, "capture.machine_identity_sha256");
  requireSha256(envelope.boot_identity_sha256, "capture.boot_identity_sha256");
  return { started, exited, captured };
}

function readSingleAuditArtifact(
  artifactRoot,
  auditRelativePath,
  expectedByteCount,
  maximumBytes,
) {
  const rootPath = requireText(artifactRoot, "artifactRoot");
  const rootStat = fs.lstatSync(rootPath);
  if (!rootStat.isDirectory() || rootStat.isSymbolicLink()) {
    throw new Error("artifactRoot must be one real directory");
  }
  const rootReal = fs.realpathSync(rootPath);
  const relative = requireRelativePosixPath(auditRelativePath, "auditRelativePath");
  const candidate = path.resolve(rootReal, ...relative.split("/"));
  const nativeRelative = path.relative(rootReal, candidate);
  if (nativeRelative.startsWith("..") || path.isAbsolute(nativeRelative)) {
    throw new Error("raw Audit artifact escapes artifactRoot");
  }
  const beforePathStat = fs.lstatSync(candidate);
  if (
    !beforePathStat.isFile() ||
    beforePathStat.isSymbolicLink() ||
    beforePathStat.nlink !== 1
  ) {
    throw new Error("raw Audit artifact must be a regular single-link file");
  }
  const candidateReal = fs.realpathSync(candidate);
  const realRelative = path.relative(rootReal, candidateReal);
  if (realRelative.startsWith("..") || path.isAbsolute(realRelative)) {
    throw new Error("raw Audit artifact resolves outside artifactRoot");
  }
  const descriptor = fs.openSync(candidate, "r");
  try {
    const beforeDescriptorStat = fs.fstatSync(descriptor);
    if (!beforeDescriptorStat.isFile() || beforeDescriptorStat.nlink !== 1) {
      throw new Error("opened raw Audit artifact identity drift");
    }
    if (
      beforeDescriptorStat.size !== expectedByteCount ||
      beforeDescriptorStat.size <= 0 ||
      beforeDescriptorStat.size > maximumBytes
    ) {
      throw new Error("raw Audit artifact size does not match the bounded capture claim");
    }
    if (
      beforeDescriptorStat.dev !== beforePathStat.dev ||
      beforeDescriptorStat.ino !== beforePathStat.ino
    ) {
      throw new Error("raw Audit artifact changed between path check and open");
    }
    const raw = fs.readFileSync(descriptor);
    const afterDescriptorStat = fs.fstatSync(descriptor);
    if (
      afterDescriptorStat.dev !== beforeDescriptorStat.dev ||
      afterDescriptorStat.ino !== beforeDescriptorStat.ino ||
      afterDescriptorStat.size !== beforeDescriptorStat.size ||
      afterDescriptorStat.mtimeMs !== beforeDescriptorStat.mtimeMs
    ) {
      throw new Error("raw Audit artifact changed during descriptor read");
    }
    return raw;
  } finally {
    fs.closeSync(descriptor);
  }
}

function validateAndDeriveProvisionerAudit(
  audit,
  protocol,
  captureEnvelope,
  checkoutProvisionerIdentity,
  expectedMachineConfigContentId,
) {
  orderedExactKeys(
    audit,
    [
      "schema_version",
      "config_schema_version",
      "mode",
      "observed_at_utc",
      "process_exit_code",
      "overall_conformant",
      "result_disposition",
      "source_created_this_invocation",
      "source_history_transition",
      "source_absence_history_resolved",
      "allow_source_creation_operator_intent",
      "machine",
      "invoking_principal",
      "script_integrity",
      "cmdlet_provenance",
      "fixed_contract",
      "observed",
      "conformance",
      "provisioning",
      "machine_config_content_id_method",
      "machine_config_content_id",
      "machine_config_content_id_basis_base64",
      "requires_cold_or_service_refresh",
      "refresh_disposition",
      "refresh_chronology",
      "qualification_not_authorized",
      "safety_boundary",
    ],
    "raw Event Log Audit v2",
  );
  if (
    audit.schema_version !== PROVISIONING_AUDIT_SCHEMA_VERSION ||
    audit.config_schema_version !== EVENT_LOG_SOURCE_CONFIG_SCHEMA_VERSION ||
    audit.mode !== "Audit"
  ) {
    throw new Error("raw Event Log Audit identity or mode drift");
  }
  const observedAt = requireProvisionerUtcTimestamp(
    audit.observed_at_utc,
    "raw Audit.observed_at_utc",
  );
  const captureTimes = validateCaptureEnvelope(captureEnvelope, protocol);
  if (observedAt < captureTimes.started || observedAt > captureTimes.exited) {
    throw new Error("raw Audit observation falls outside the captured process interval");
  }
  const claimedExitCode = requireInteger(
    audit.process_exit_code,
    "raw Audit.process_exit_code",
  );
  requireBoolean(audit.overall_conformant, "raw Audit.overall_conformant");
  requireText(audit.result_disposition, "raw Audit.result_disposition");
  if (
    audit.source_created_this_invocation !== false ||
    audit.source_absence_history_resolved !== false ||
    audit.allow_source_creation_operator_intent !== false
  ) {
    throw new Error("raw Audit attempted to carry Provision semantics");
  }

  validateProvisionerMachine(audit.machine, "raw Audit.machine");
  if (audit.machine.machine_identity_sha256 !== captureEnvelope.machine_identity_sha256) {
    throw new Error("raw Audit machine identity does not match the capture envelope");
  }
  validateInvokingPrincipal(audit.invoking_principal, "raw Audit.invoking_principal");
  validateScriptIntegrity(audit.script_integrity, protocol, "raw Audit.script_integrity");
  if (
    audit.script_integrity.lf_canonical_byte_count !==
      checkoutProvisionerIdentity.lfCanonicalByteCount ||
    audit.script_integrity.observed_lf_canonical_sha256 !==
      checkoutProvisionerIdentity.lfCanonicalSha256
  ) {
    throw new Error("raw Audit source identity does not match the reviewed checkout bytes");
  }
  validateCmdletProvenance(audit.cmdlet_provenance, "raw Audit.cmdlet_provenance");
  validateFixedContract(audit.fixed_contract, protocol, "raw Audit.fixed_contract");

  orderedExactKeys(
    audit.observed,
    [
      "source_registry_before",
      "source_registry_after",
      "application_registry_before",
      "application_registry_after",
      "application_channel_before",
      "application_channel_after",
    ],
    "raw Audit.observed",
  );
  const fixedContract = audit.fixed_contract;
  validateRegistryObservation(
    audit.observed.source_registry_before,
    "raw Audit.observed.source_registry_before",
    fixedContract.source_registry_subkey,
  );
  validateRegistryObservation(
    audit.observed.source_registry_after,
    "raw Audit.observed.source_registry_after",
    fixedContract.source_registry_subkey,
  );
  validateRegistryObservation(
    audit.observed.application_registry_before,
    "raw Audit.observed.application_registry_before",
    fixedContract.application_registry_subkey,
  );
  validateRegistryObservation(
    audit.observed.application_registry_after,
    "raw Audit.observed.application_registry_after",
    fixedContract.application_registry_subkey,
  );
  if (
    !audit.observed.application_registry_before.present ||
    !audit.observed.application_registry_after.present
  ) {
    throw new Error("raw Audit Application registry observations must both be present");
  }
  validateApplicationChannelObservation(
    audit.observed.application_channel_before,
    "raw Audit.observed.application_channel_before",
  );
  validateApplicationChannelObservation(
    audit.observed.application_channel_after,
    "raw Audit.observed.application_channel_after",
  );

  const derived = deriveAuditConformance(audit.observed, fixedContract);
  orderedDeepExact(
    audit.conformance,
    derived.conformance,
    "raw Audit.conformance recomputation",
  );
  if (audit.overall_conformant !== derived.overallConformant) {
    throw new Error("raw Audit overall conformance claim disagrees with observations");
  }
  const derivedExitCode = derived.overallConformant ? 0 : 2;
  const derivedDisposition = derived.overallConformant
    ? "audit_conformant"
    : "audit_nonconformant";
  if (
    claimedExitCode !== derivedExitCode ||
    captureEnvelope.process_exit_code !== derivedExitCode ||
    audit.result_disposition !== derivedDisposition
  ) {
    throw new Error("raw Audit exit/disposition does not match recomputed conformance");
  }
  const derivedHistory = deriveSourceHistoryTransition(
    audit.observed.source_registry_before,
    audit.observed.source_registry_after,
  );
  if (audit.source_history_transition !== derivedHistory) {
    throw new Error("raw Audit source-history transition drift");
  }

  validateAuditProvisioningState(audit.provisioning);
  if (
    audit.requires_cold_or_service_refresh !== null ||
    audit.refresh_disposition !== "not_observed_or_proven_by_this_invocation" ||
    audit.qualification_not_authorized !== true
  ) {
    throw new Error("raw Audit refresh or authorization boundary drift");
  }
  orderedExactKeys(
    audit.refresh_chronology,
    [
      "authoritative",
      "prior_refresh_state",
      "required_due_to_source_creation_this_invocation",
      "cold_boot_observed",
      "eventlog_service_restart_observed",
      "refresh_verified",
      "source_registration_started_at_utc",
      "source_registration_completed_at_utc",
      "source_configuration_completed_at_utc",
      "post_observation_completed_at_utc",
    ],
    "raw Audit.refresh_chronology",
  );
  const postObservationAt = requireProvisionerUtcTimestamp(
    audit.refresh_chronology.post_observation_completed_at_utc,
    "raw Audit.refresh_chronology.post_observation_completed_at_utc",
  );
  if (postObservationAt < captureTimes.started || postObservationAt > observedAt) {
    throw new Error("raw Audit post-observation chronology falls outside the process interval");
  }
  orderedDeepExact(
    audit.refresh_chronology,
    {
      authoritative: false,
      prior_refresh_state: "not_assessed",
      required_due_to_source_creation_this_invocation: false,
      cold_boot_observed: false,
      eventlog_service_restart_observed: false,
      refresh_verified: false,
      source_registration_started_at_utc: null,
      source_registration_completed_at_utc: null,
      source_configuration_completed_at_utc: null,
      post_observation_completed_at_utc:
        audit.refresh_chronology.post_observation_completed_at_utc,
    },
    "raw Audit.refresh_chronology recomputation",
  );

  const channelTransition =
    derived.conformance.application_channel_provider_membership_transition;
  orderedDeepExact(
    audit.safety_boundary,
    {
      event_log_records_read: false,
      event_log_records_written: false,
      source_registration_performed: false,
      application_channel_full_endpoint_changed:
        !derived.conformance.application_channel_full_endpoint_equal,
      application_channel_stable_projection_endpoint_changed:
        !derived.conformance.application_channel_stable_projection_endpoint_equal,
      application_channel_provider_membership_transition: channelTransition,
      application_registry_endpoint_changed:
        !derived.conformance.application_registry_endpoint_equal,
      source_registry_endpoint_changed: !derived.conformance.source_registry_endpoint_equal,
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
    "raw Audit.safety_boundary recomputation",
  );

  const machineConfig = parseMachineConfigBasis(audit, protocol);
  if (
    expectedMachineConfigContentId !== null &&
    audit.machine_config_content_id !== expectedMachineConfigContentId
  ) {
    throw new Error("raw Audit machine-config content ID does not match the frozen scope value");
  }
  return {
    ...derived,
    observedAt,
    postObservationAt,
    machineConfig,
    derivedExitCode,
  };
}

function normalizeSyntheticFaultEvents(events, baselineCursors, timestamp) {
  const source = requireArray(events, "faultEvents");
  const nextRecordIds = Object.fromEntries(
    baselineCursors.map((cursor) => [cursor.log_name, cursor.newest_record_id]),
  );
  return source.map((event, index) => {
    const item = requireObject(event, `faultEvents[${index}]`);
    const logName = requireText(item.logName, `faultEvents[${index}].logName`);
    if (!CHANNEL_NAMES.includes(logName)) throw new Error("fault event logName is unsupported");
    const providerName = requireText(
      item.providerName,
      `faultEvents[${index}].providerName`,
    );
    const eventId = requireInteger(item.eventId, `faultEvents[${index}].eventId`);
    nextRecordIds[logName] += 1;
    const recordId =
      item.recordId === undefined
        ? nextRecordIds[logName]
        : requireInteger(item.recordId, `faultEvents[${index}].recordId`);
    nextRecordIds[logName] = recordId;
    return {
      log_name: logName,
      provider_name: providerName,
      event_id: eventId,
      record_id: recordId,
      created_at_utc: timestamp,
      xml_sha256: sha256Bytes(
        Buffer.from(`${logName}:${providerName}:${eventId}:${recordId}`, "utf8"),
      ),
    };
  });
}

function advanceCursors(startCursors, events, capturedAtUtc, machineId, bootId) {
  return startCursors.map((cursor) => {
    const channelEvents = events.filter((event) => event.log_name === cursor.log_name);
    const newest =
      channelEvents.length === 0
        ? cursor.newest_record_id
        : channelEvents[channelEvents.length - 1].record_id;
    return {
      ...cursor,
      record_count: cursor.record_count + channelEvents.length,
      newest_record_id: newest,
      newest_record_xml_sha256:
        channelEvents.length === 0
          ? cursor.newest_record_xml_sha256
          : channelEvents[channelEvents.length - 1].xml_sha256,
      machine_identity_sha256: machineId,
      boot_identity_sha256: bootId,
      captured_at_utc: capturedAtUtc,
    };
  });
}

function makeDelta({
  windowName,
  startedAtUtc,
  endedAtUtc,
  startCursors,
  events,
  machineId,
  bootId,
}) {
  return {
    window_name: windowName,
    interval_start_exclusive_utc: startedAtUtc,
    interval_end_inclusive_utc: endedAtUtc,
    start_cursors: startCursors,
    end_cursors: advanceCursors(startCursors, events, endedAtUtc, machineId, bootId),
    events,
  };
}

function classifyFaults(events, protocol) {
  const classified = [];
  for (const event of events) {
    for (const rule of protocol.event_log.fault_classification) {
      const eventIds = rule.event_ids === null ? null : rule.event_ids.map((value) => requireInteger(value, "fault event ID"));
      if (
        event.log_name === rule.log_name &&
        event.provider_name === rule.provider_name &&
        (eventIds === null || eventIds.includes(event.event_id))
      ) {
        classified.push({
          failure_code: rule.failure_code,
          log_name: event.log_name,
          provider_name: event.provider_name,
          event_id: event.event_id,
          record_id: event.record_id,
          xml_sha256: event.xml_sha256,
        });
        break;
      }
    }
  }
  return classified;
}

function sourceAuditConformant(audit, protocol) {
  try {
    return (
      audit.projection_schema_version ===
        protocol.event_log_source.qualification_projection_schema_version &&
      audit.claimed_input_audit_schema_version ===
        protocol.event_log_source.provisioning_audit_schema_version &&
      audit.config_schema_version === protocol.event_log_source.source_config_schema_version &&
      audit.mode === "Audit" &&
      audit.audit_exit_code === 0 &&
      audit.overall_conformant === true &&
      audit.source_created_this_invocation === false &&
      audit.requires_cold_or_service_refresh === null &&
      audit.refresh_disposition ===
        protocol.event_log_source.refresh_disposition_required &&
      audit.qualification_not_authorized === true &&
      audit.source_name === protocol.event_log_source.source_name &&
      audit.log_name === protocol.event_log_source.log_name &&
      audit.conformance.source_before.source_configuration_exact === true &&
      audit.conformance.source_after.source_configuration_exact === true &&
      audit.conformance.application_channel_before.log_name_exact === true &&
      audit.conformance.application_channel_before.enabled === true &&
      audit.conformance.application_channel_before.classic_log === true &&
      audit.conformance.application_channel_before.circular_log_mode === true &&
      audit.conformance.application_channel_before.positive_maximum_size === true &&
      audit.conformance.application_channel_before.source_provider_membership_present === true &&
      audit.conformance.application_channel_after.log_name_exact === true &&
      audit.conformance.application_channel_after.enabled === true &&
      audit.conformance.application_channel_after.classic_log === true &&
      audit.conformance.application_channel_after.circular_log_mode === true &&
      audit.conformance.application_channel_after.positive_maximum_size === true &&
      audit.conformance.application_channel_after.source_provider_membership_present === true &&
      audit.conformance.application_channel_full_endpoint_equal === true &&
      audit.conformance.application_channel_stable_projection_endpoint_equal === true &&
      audit.conformance.application_channel_provider_membership_transition.disposition ===
        "unchanged" &&
      audit.conformance.application_channel_provider_membership_transition
        .allowed_for_source_creation === true &&
      audit.conformance.application_registry_endpoint_equal === true &&
      audit.conformance.source_registry_endpoint_equal === true &&
      audit.conformance.continuous_stability_proven === false &&
      audit.audit_exit_code_alone_proves_conformance === false &&
      audit.refresh_requirement_field_alone_proves_service_refresh_or_cold_boot === false &&
      audit.machine_config_content_id_alone_proves_conformance === false &&
      audit.full_raw_audit_bound === false &&
      audit.raw_audit_sha256 === null &&
      audit.raw_audit_content_id_basis_revalidated === false
    );
  } catch (error) {
    if (error instanceof TypeError) return false;
    throw error;
  }
}

function validateCursor(cursor, label) {
  exactKeys(
    cursor,
    [
      "log_name",
      "enabled",
      "record_count",
      "oldest_record_id",
      "newest_record_id",
      "newest_record_xml_sha256",
      "maximum_size_bytes",
      "log_mode",
      "configuration_sha256",
      "machine_identity_sha256",
      "boot_identity_sha256",
      "captured_at_utc",
    ],
    label,
  );
  if (!CHANNEL_NAMES.includes(cursor.log_name)) throw new Error(`${label} channel drift`);
  if (requireBoolean(cursor.enabled, `${label}.enabled`) !== true) {
    throw new Error(`${label} is disabled`);
  }
  const recordCount = requireInteger(cursor.record_count, `${label}.record_count`);
  const oldestRecordId = requireInteger(
    cursor.oldest_record_id,
    `${label}.oldest_record_id`,
  );
  const newestRecordId = requireInteger(
    cursor.newest_record_id,
    `${label}.newest_record_id`,
  );
  if (
    recordCount <= 0 ||
    oldestRecordId <= 0 ||
    newestRecordId < oldestRecordId ||
    recordCount !== newestRecordId - oldestRecordId + 1
  ) {
    throw new Error(`${label} record-count/range relationship drift`);
  }
  requireSha256(cursor.newest_record_xml_sha256, `${label}.newest_record_xml_sha256`);
  if (requireInteger(cursor.maximum_size_bytes, `${label}.maximum_size_bytes`) <= 0) {
    throw new Error(`${label} maximum size must be positive`);
  }
  if (cursor.log_mode !== "Circular") throw new Error(`${label} log mode drift`);
  requireSha256(cursor.configuration_sha256, `${label}.configuration_sha256`);
  requireSha256(cursor.machine_identity_sha256, `${label}.machine_identity_sha256`);
  requireSha256(cursor.boot_identity_sha256, `${label}.boot_identity_sha256`);
  requireUtcTimestamp(cursor.captured_at_utc, `${label}.captured_at_utc`);
}

function validateDelta(delta, label, protocol, expectedWindowName) {
  exactKeys(
    delta,
    [
      "window_name",
      "interval_start_exclusive_utc",
      "interval_end_inclusive_utc",
      "start_cursors",
      "end_cursors",
      "events",
    ],
    label,
  );
  if (delta.window_name !== expectedWindowName) {
    throw new Error(`${label}.window_name drift`);
  }
  const startTime = requireUtcTimestamp(
    delta.interval_start_exclusive_utc,
    `${label}.interval_start_exclusive_utc`,
  );
  const endTime = requireUtcTimestamp(
    delta.interval_end_inclusive_utc,
    `${label}.interval_end_inclusive_utc`,
  );
  if (endTime < startTime) throw new Error(`${label} time order drift`);
  const starts = requireArray(delta.start_cursors, `${label}.start_cursors`);
  const ends = requireArray(delta.end_cursors, `${label}.end_cursors`);
  if (starts.length !== 2 || ends.length !== 2) throw new Error(`${label} cursor count drift`);
  starts.forEach((cursor, index) => validateCursor(cursor, `${label}.start_cursors[${index}]`));
  ends.forEach((cursor, index) => validateCursor(cursor, `${label}.end_cursors[${index}]`));
  deepExact(starts.map((cursor) => cursor.log_name), CHANNEL_NAMES, `${label} start channels`);
  deepExact(ends.map((cursor) => cursor.log_name), CHANNEL_NAMES, `${label} end channels`);
  if (
    starts.some(
      (cursor) =>
        requireUtcTimestamp(cursor.captured_at_utc, `${label} start cursor time`) >
        startTime,
    ) ||
    ends.some(
      (cursor) =>
        requireUtcTimestamp(cursor.captured_at_utc, `${label} end cursor time`) !==
        endTime,
    )
  ) {
    throw new Error(`${label} cursor/interval timestamp drift`);
  }
  const events = requireArray(delta.events, `${label}.events`);
  for (const [index, event] of events.entries()) {
    exactKeys(
      event,
      [
        "log_name",
        "provider_name",
        "event_id",
        "record_id",
        "created_at_utc",
        "xml_sha256",
      ],
      `${label}.events[${index}]`,
    );
    if (!CHANNEL_NAMES.includes(event.log_name)) throw new Error(`${label} event channel drift`);
    requireText(event.provider_name, `${label}.events[${index}].provider_name`);
    requireInteger(event.event_id, `${label}.events[${index}].event_id`);
    requireInteger(event.record_id, `${label}.events[${index}].record_id`);
    const eventTime = requireUtcTimestamp(
      event.created_at_utc,
      `${label}.events[${index}].created_at_utc`,
    );
    if (eventTime <= startTime || eventTime > endTime) {
      throw new Error(`${label}.events[${index}] falls outside the frozen interval`);
    }
    requireSha256(event.xml_sha256, `${label}.events[${index}].xml_sha256`);
  }
  for (const channel of CHANNEL_NAMES) {
    const count = events.filter((event) => event.log_name === channel).length;
    if (count > protocol.event_log.max_new_records_per_channel_per_window) {
      throw new Error(`${label} exceeds the per-channel Event Log record budget`);
    }
  }
  return { startTime, endTime };
}

function deltaContinuityValid(delta) {
  for (const channel of CHANNEL_NAMES) {
    const start = delta.start_cursors.find((cursor) => cursor.log_name === channel);
    const end = delta.end_cursors.find((cursor) => cursor.log_name === channel);
    if (start === undefined || end === undefined) return false;
    if (
      start.machine_identity_sha256 !== end.machine_identity_sha256 ||
      start.boot_identity_sha256 !== end.boot_identity_sha256 ||
      start.configuration_sha256 !== end.configuration_sha256 ||
      start.log_mode !== end.log_mode ||
      start.oldest_record_id !== end.oldest_record_id
    ) {
      return false;
    }
    const channelEvents = delta.events.filter((event) => event.log_name === channel);
    let expectedRecordId = start.newest_record_id;
    for (const event of channelEvents) {
      expectedRecordId += 1;
      if (event.record_id !== expectedRecordId) return false;
    }
    if (end.newest_record_id !== expectedRecordId) return false;
    if (end.record_count !== start.record_count + channelEvents.length) return false;
    const expectedBoundarySha =
      channelEvents.length === 0
        ? start.newest_record_xml_sha256
        : channelEvents[channelEvents.length - 1].xml_sha256;
    if (end.newest_record_xml_sha256 !== expectedBoundarySha) return false;
  }
  return true;
}

function cursorsJoin(left, right) {
  try {
    for (const channel of CHANNEL_NAMES) {
      const first = left.find((cursor) => cursor.log_name === channel);
      const second = right.find((cursor) => cursor.log_name === channel);
      if (first === undefined || second === undefined) return false;
      if (canonicalJson(first) !== canonicalJson(second)) return false;
    }
    return true;
  } catch (error) {
    if (error instanceof TypeError) return false;
    throw error;
  }
}

function deriveQualification(receipts, protocol) {
  const scope = receipts[0].body;
  const preregistration = receipts[1].body;
  const sourceBefore = receipts[2].body.source_audit_projection;
  const baseline = receipts[3].body;
  const launch = receipts[4].body;
  const exit = receipts[5].body;
  const processDelta = receipts[6].body.delta;
  const cooldownDelta = receipts[7].body.delta;
  const handoff = receipts[8].body;
  const tailDelta = handoff.terminal_tail_delta;

  const failures = new Set();
  if (scope.execution_backend_id === SYNTHETIC_BACKEND_ID) {
    failures.add("synthetic_test_backend_not_evidence");
  }
  const declaration = preregistration.operator_declaration;
  if (
    declaration.schema_version !==
      protocol.firmware_gate.operator_declaration_schema_version ||
    OPERATOR_DECLARATION_FIELDS.some((field) => declaration[field] !== true) ||
    contentId(declaration) !== preregistration.operator_declaration_id ||
    preregistration.operator_declaration_content_address_is_signature !== false
  ) {
    failures.add("operator_declaration_incomplete");
  }

  let decodedMicrocode = null;
  try {
    decodedMicrocode = decodeMicrocodeLittleEndian(
      baseline.firmware_observation.microcode_raw_little_endian_hex,
    );
  } catch {
    failures.add("microcode_revision_encoding_invalid");
  }
  if (
    decodedMicrocode !== null &&
    decodedMicrocode !== baseline.firmware_observation.microcode_revision_integer
  ) {
    failures.add("microcode_revision_decode_mismatch");
  }
  if (
    decodedMicrocode === null ||
    decodedMicrocode < protocol.firmware_gate.minimum_microcode_revision_integer
  ) {
    failures.add("microcode_revision_below_minimum");
  }

  if (!sourceAuditConformant(sourceBefore, protocol)) {
    failures.add("event_log_source_before_nonconformant");
  }
  if (!sourceAuditConformant(handoff.source_audit_projection_after, protocol)) {
    failures.add("event_log_source_after_nonconformant");
  }
  if (
    sourceBefore.machine_config_content_id !==
    handoff.source_audit_projection_after.machine_config_content_id
  ) {
    failures.add("event_log_source_configuration_drift");
  }

  const machineIds = [
    scope.machine_identity_sha256,
    sourceBefore.machine_identity_sha256,
    baseline.machine_observation.machine_identity_sha256,
    ...baseline.baseline_cursors.map((cursor) => cursor.machine_identity_sha256),
    ...processDelta.start_cursors.map((cursor) => cursor.machine_identity_sha256),
    ...processDelta.end_cursors.map((cursor) => cursor.machine_identity_sha256),
    ...cooldownDelta.start_cursors.map((cursor) => cursor.machine_identity_sha256),
    ...cooldownDelta.end_cursors.map((cursor) => cursor.machine_identity_sha256),
    ...tailDelta.start_cursors.map((cursor) => cursor.machine_identity_sha256),
    ...tailDelta.end_cursors.map((cursor) => cursor.machine_identity_sha256),
    handoff.source_audit_projection_after.machine_identity_sha256,
  ];
  if (new Set(machineIds).size !== 1) failures.add("machine_identity_drift");
  const bootIds = [
    scope.boot_identity_sha256,
    sourceBefore.boot_identity_sha256,
    baseline.machine_observation.boot_identity_sha256,
    ...baseline.baseline_cursors.map((cursor) => cursor.boot_identity_sha256),
    ...processDelta.start_cursors.map((cursor) => cursor.boot_identity_sha256),
    ...processDelta.end_cursors.map((cursor) => cursor.boot_identity_sha256),
    ...cooldownDelta.start_cursors.map((cursor) => cursor.boot_identity_sha256),
    ...cooldownDelta.end_cursors.map((cursor) => cursor.boot_identity_sha256),
    ...tailDelta.start_cursors.map((cursor) => cursor.boot_identity_sha256),
    ...tailDelta.end_cursors.map((cursor) => cursor.boot_identity_sha256),
    handoff.source_audit_projection_after.boot_identity_sha256,
  ];
  if (new Set(bootIds).size !== 1) failures.add("boot_identity_drift");

  const continuityValid =
    deltaContinuityValid(processDelta) &&
    deltaContinuityValid(cooldownDelta) &&
    deltaContinuityValid(tailDelta) &&
    cursorsJoin(baseline.baseline_cursors, processDelta.start_cursors) &&
    cursorsJoin(processDelta.end_cursors, cooldownDelta.start_cursors) &&
    cursorsJoin(cooldownDelta.end_cursors, tailDelta.start_cursors) &&
    cursorsJoin(tailDelta.end_cursors, handoff.handoff_cursors);
  if (!continuityValid) failures.add("event_log_continuity_lost");

  if (
    launch.execution_backend_id !== scope.execution_backend_id ||
    launch.launch_receipt_fsynced_before_process_creation !== true ||
    exit.exit_code !== protocol.probe.success_exit_code ||
    exit.timed_out !== false
  ) {
    failures.add("probe_execution_failed");
  }
  if (exit.descendants_quiesced_before_cooldown !== true) {
    failures.add("probe_descendants_not_quiesced");
  }
  const cooldownDuration =
    requireUtcTimestamp(cooldownDelta.interval_end_inclusive_utc, "cooldown end") -
    requireUtcTimestamp(cooldownDelta.interval_start_exclusive_utc, "cooldown start");
  if (cooldownDuration < protocol.timing.minimum_cooldown_seconds * 1000) {
    failures.add("cooldown_window_too_short");
  }
  const tailDuration =
    requireUtcTimestamp(tailDelta.interval_end_inclusive_utc, "tail end") -
    requireUtcTimestamp(tailDelta.interval_start_exclusive_utc, "tail start");
  if (tailDuration < protocol.timing.minimum_terminal_tail_seconds * 1000) {
    failures.add("terminal_tail_window_too_short");
  }

  const allEvents = [
    ...processDelta.events,
    ...cooldownDelta.events,
    ...tailDelta.events,
  ];
  const classifiedFaults = classifyFaults(allEvents, protocol);
  for (const fault of classifiedFaults) failures.add(fault.failure_code);
  const failureCodes = FAILURE_CODE_ORDER.filter((code) => failures.has(code));
  const criteriaPassed = failureCodes.length === 0;
  const realHostObservation =
    scope.execution_backend_id === protocol.probe.production_backend_id &&
    launch.real_process_observation === true &&
    exit.real_process_observation === true &&
    sourceBefore.real_provisioner_observation === true &&
    handoff.source_audit_projection_after.real_provisioner_observation === true;
  const eligible = criteriaPassed && realHostObservation;

  return {
    criteria: {
      operator_declaration_complete: !failures.has("operator_declaration_incomplete"),
      microcode_minimum_met:
        !failures.has("microcode_revision_encoding_invalid") &&
        !failures.has("microcode_revision_decode_mismatch") &&
        !failures.has("microcode_revision_below_minimum"),
      source_audit_before_conformant: !failures.has(
        "event_log_source_before_nonconformant",
      ),
      source_audit_after_conformant: !failures.has(
        "event_log_source_after_nonconformant",
      ),
      source_configuration_stable: !failures.has(
        "event_log_source_configuration_drift",
      ),
      same_machine: !failures.has("machine_identity_drift"),
      same_boot: !failures.has("boot_identity_drift"),
      event_log_continuity: !failures.has("event_log_continuity_lost"),
      probe_completed: !failures.has("probe_execution_failed"),
      probe_descendants_quiesced: !failures.has("probe_descendants_not_quiesced"),
      cooldown_observed: !failures.has("cooldown_window_too_short"),
      terminal_tail_observed: !failures.has("terminal_tail_window_too_short"),
      classified_fault_free: classifiedFaults.length === 0,
      real_backend_observed: realHostObservation,
    },
    failure_codes: failureCodes,
    classified_faults: classifiedFaults,
    criteria_passed: criteriaPassed,
    real_host_observation: realHostObservation,
    eligible_as_host_qualification_input: eligible,
    cuda_execution_authorized: false,
    formal_evidence_authorized: false,
    production_active_authorized: false,
    appendable_proven: false,
    readable_proven: false,
    learnable_proven: false,
    steerable_proven: false,
    four_capability_claim_authorized: false,
    claim_boundary: protocol.claim_boundary,
  };
}

function makeReceipt(sequence, scopeId, previousReceiptRawSha256, observedAtUtc, body) {
  return {
    schema_version: RECEIPT_SCHEMA_VERSION,
    sequence,
    scope_id: scopeId,
    previous_receipt_raw_sha256: previousReceiptRawSha256,
    observed_at_utc: observedAtUtc,
    body,
  };
}

function receiptBody(receipt, sequence, scopeId, previousRawSha, label) {
  exactKeys(
    receipt,
    [
      "schema_version",
      "sequence",
      "scope_id",
      "previous_receipt_raw_sha256",
      "observed_at_utc",
      "body",
    ],
    label,
  );
  if (receipt.schema_version !== RECEIPT_SCHEMA_VERSION) {
    throw new Error(`${label} schema drift`);
  }
  if (requireInteger(receipt.sequence, `${label}.sequence`) !== sequence) {
    throw new Error(`${label} sequence drift`);
  }
  if (receipt.scope_id !== scopeId) throw new Error(`${label} scope drift`);
  if (receipt.previous_receipt_raw_sha256 !== previousRawSha) {
    throw new Error(`${label} previous raw SHA-256 chain drift`);
  }
  requireUtcTimestamp(receipt.observed_at_utc, `${label}.observed_at_utc`);
  return requireObject(receipt.body, `${label}.body`);
}

function inventoryEntry(root, relativePath) {
  const absolutePath = path.join(root, ...relativePath.split("/"));
  const raw = fs.readFileSync(absolutePath);
  return {
    path: relativePath,
    byte_count: raw.length,
    raw_sha256: sha256Bytes(raw),
  };
}

function listRootFilesStrict(root) {
  const rootStat = fs.lstatSync(root);
  if (!rootStat.isDirectory() || rootStat.isSymbolicLink()) {
    throw new Error("qualification root must be a non-linked directory");
  }
  const rootReal = fs.realpathSync(root);
  const files = [];
  const directories = [];
  const visit = (directory, relativeDirectory) => {
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      const relative = relativeDirectory === "" ? entry.name : `${relativeDirectory}/${entry.name}`;
      const absolute = path.join(directory, entry.name);
      const stat = fs.lstatSync(absolute);
      if (stat.isSymbolicLink()) throw new Error(`qualification symlink/reparse entry forbidden: ${relative}`);
      if (stat.isDirectory()) {
        directories.push(relative.replace(/\\/gu, "/"));
        visit(absolute, relative);
        continue;
      }
      if (!stat.isFile() || stat.nlink !== 1) {
        throw new Error(`qualification entry must be a regular single-link file: ${relative}`);
      }
      const real = fs.realpathSync(absolute);
      const nativeRelative = path.relative(rootReal, real);
      if (nativeRelative.startsWith("..") || path.isAbsolute(nativeRelative)) {
        throw new Error(`qualification entry escapes root: ${relative}`);
      }
      files.push(relative.replace(/\\/gu, "/"));
    }
  };
  visit(root, "");
  if (directories.length !== 1 || directories[0] !== "streams") {
    throw new Error("qualification exact directory entry set drift");
  }
  return files.sort();
}

function validateSourceAuditShape(audit, label) {
  exactKeys(
    audit,
    [
      "projection_schema_version",
      "claimed_input_audit_schema_version",
      "config_schema_version",
      "mode",
      "observed_at_utc",
      "audit_exit_code",
      "overall_conformant",
      "source_created_this_invocation",
      "requires_cold_or_service_refresh",
      "refresh_disposition",
      "qualification_not_authorized",
      "machine_identity_sha256",
      "boot_identity_sha256",
      "machine_config_content_id",
      "source_name",
      "log_name",
      "conformance",
      "audit_exit_code_alone_proves_conformance",
      "refresh_requirement_field_alone_proves_service_refresh_or_cold_boot",
      "machine_config_content_id_alone_proves_conformance",
      "full_raw_audit_bound",
      "raw_audit_sha256",
      "raw_audit_content_id_basis_revalidated",
      "real_provisioner_observation",
    ],
    label,
  );
  requireUtcTimestamp(audit.observed_at_utc, `${label}.observed_at_utc`);
  requireInteger(audit.audit_exit_code, `${label}.audit_exit_code`);
  requireBoolean(audit.overall_conformant, `${label}.overall_conformant`);
  requireBoolean(
    audit.source_created_this_invocation,
    `${label}.source_created_this_invocation`,
  );
  if (audit.requires_cold_or_service_refresh !== null) {
    requireBoolean(
      audit.requires_cold_or_service_refresh,
      `${label}.requires_cold_or_service_refresh`,
    );
  }
  requireBoolean(audit.qualification_not_authorized, `${label}.qualification_not_authorized`);
  requireSha256(audit.machine_identity_sha256, `${label}.machine_identity_sha256`);
  requireSha256(audit.boot_identity_sha256, `${label}.boot_identity_sha256`);
  requireSha256(audit.machine_config_content_id, `${label}.machine_config_content_id`);
  requireBoolean(audit.real_provisioner_observation, `${label}.real_provisioner_observation`);
  requireBoolean(audit.full_raw_audit_bound, `${label}.full_raw_audit_bound`);
  if (audit.raw_audit_sha256 !== null) {
    requireSha256(audit.raw_audit_sha256, `${label}.raw_audit_sha256`);
  }
  requireBoolean(
    audit.raw_audit_content_id_basis_revalidated,
    `${label}.raw_audit_content_id_basis_revalidated`,
  );
  exactKeys(
    audit.conformance,
    [
      "source_before",
      "source_after",
      "application_channel_before",
      "application_channel_after",
      "application_channel_full_endpoint_equal",
      "application_channel_stable_projection_endpoint_equal",
      "application_channel_provider_membership_transition",
      "application_registry_endpoint_equal",
      "source_registry_endpoint_equal",
      "continuous_stability_proven",
    ],
    `${label}.conformance`,
  );
  for (const side of ["source_before", "source_after"]) {
    exactKeys(
      audit.conformance[side],
      ["source_configuration_exact"],
      `${label}.conformance.${side}`,
    );
    requireBoolean(
      audit.conformance[side].source_configuration_exact,
      `${label}.conformance.${side}.source_configuration_exact`,
    );
  }
  for (const side of ["application_channel_before", "application_channel_after"]) {
    exactKeys(
      audit.conformance[side],
      [
        "log_name_exact",
        "enabled",
        "classic_log",
        "circular_log_mode",
        "positive_maximum_size",
        "source_provider_membership_present",
      ],
      `${label}.conformance.${side}`,
    );
    for (const field of [
      "log_name_exact",
      "enabled",
      "classic_log",
      "circular_log_mode",
      "positive_maximum_size",
      "source_provider_membership_present",
    ]) {
      requireBoolean(
        audit.conformance[side][field],
        `${label}.conformance.${side}.${field}`,
      );
    }
  }
  for (const field of [
    "application_channel_full_endpoint_equal",
    "application_channel_stable_projection_endpoint_equal",
    "application_registry_endpoint_equal",
    "source_registry_endpoint_equal",
    "continuous_stability_proven",
  ]) {
    requireBoolean(audit.conformance[field], `${label}.conformance.${field}`);
  }
  exactKeys(
    audit.conformance.application_channel_provider_membership_transition,
    ["disposition", "allowed_for_source_creation"],
    `${label}.conformance.application_channel_provider_membership_transition`,
  );
  requireText(
    audit.conformance.application_channel_provider_membership_transition.disposition,
    `${label}.conformance.application_channel_provider_membership_transition.disposition`,
  );
  requireBoolean(
    audit.conformance.application_channel_provider_membership_transition
      .allowed_for_source_creation,
    `${label}.conformance.application_channel_provider_membership_transition.allowed_for_source_creation`,
  );
}

function validateReceiptBodies(receipts, protocol, protocolId, protocolRawSha256) {
  const receiptTimes = receipts.map((receipt, index) =>
    requireUtcTimestamp(receipt.observed_at_utc, `receipt ${index}.observed_at_utc`),
  );
  for (let index = 1; index < receiptTimes.length; index += 1) {
    if (receiptTimes[index] < receiptTimes[index - 1]) {
      throw new Error(`receipt ${index} timestamp precedes its predecessor`);
    }
  }
  const scope = receipts[0].body;
  exactKeys(
    scope,
    [
      "qualification_protocol_id",
      "qualification_protocol_raw_sha256",
      "host_block_receipt_relative_path",
      "host_block_receipt_raw_sha256",
      "machine_identity_sha256",
      "boot_identity_sha256",
      "source_machine_config_content_id",
      "operator_declaration_id",
      "execution_backend_id",
      "scope_components",
      "attempt_number",
      "retry_number",
      "real_host_observation",
    ],
    "000 scope body",
  );
  if (
    scope.qualification_protocol_id !== protocolId ||
    scope.qualification_protocol_raw_sha256 !== protocolRawSha256 ||
    scope.host_block_receipt_relative_path !== protocol.host_block.receipt_relative_path ||
    scope.host_block_receipt_raw_sha256 !== protocol.host_block.receipt_raw_sha256 ||
    scope.execution_backend_id !== SYNTHETIC_BACKEND_ID ||
    scope.attempt_number !== 1 ||
    scope.retry_number !== 0 ||
    scope.real_host_observation !== false
  ) {
    throw new Error("000 scope contract drift");
  }
  [
    "machine_identity_sha256",
    "boot_identity_sha256",
    "source_machine_config_content_id",
    "operator_declaration_id",
  ].forEach((key) => requireSha256(scope[key], `000 scope body.${key}`));
  deepExact(
    scope.scope_components,
    [
      protocolId,
      protocol.host_block.receipt_raw_sha256,
      scope.machine_identity_sha256,
      scope.boot_identity_sha256,
      scope.source_machine_config_content_id,
      scope.operator_declaration_id,
      SYNTHETIC_BACKEND_ID,
    ],
    "000 scope components",
  );
  const expectedScopeId = domainSeparatedSha256(
    protocol.scope.scope_id_domain_separator,
    scope.scope_components,
  );
  if (receipts[0].scope_id !== expectedScopeId) throw new Error("qualification scope ID drift");

  const preregistration = receipts[1].body;
  exactKeys(
    preregistration,
    [
      "operator_declaration_schema_version",
      "operator_declaration",
      "operator_declaration_id",
      "operator_declaration_content_address_is_signature",
      "declared_at_utc",
      "attempt_number",
      "retry_budget",
    ],
    "001 preregistration body",
  );
  exactKeys(
    preregistration.operator_declaration,
    ["schema_version", ...OPERATOR_DECLARATION_FIELDS],
    "001 operator declaration",
  );
  if (
    preregistration.operator_declaration.schema_version !==
    protocol.firmware_gate.operator_declaration_schema_version
  ) {
    throw new Error("001 operator declaration schema drift");
  }
  for (const field of OPERATOR_DECLARATION_FIELDS) {
    requireBoolean(
      preregistration.operator_declaration[field],
      `001 operator declaration.${field}`,
    );
  }
  requireUtcTimestamp(preregistration.declared_at_utc, "001 declared_at_utc");
  if (
    preregistration.operator_declaration_schema_version !==
      protocol.firmware_gate.operator_declaration_schema_version ||
    preregistration.operator_declaration_id !== scope.operator_declaration_id ||
    preregistration.attempt_number !== 1 ||
    preregistration.retry_budget !== 0
  ) {
    throw new Error("001 preregistration contract drift");
  }
  if (preregistration.declared_at_utc !== receipts[1].observed_at_utc) {
    throw new Error("001 declaration/receipt timestamp drift");
  }

  exactKeys(
    receipts[2].body,
    ["source_audit_projection"],
    "002 source audit projection body",
  );
  validateSourceAuditShape(
    receipts[2].body.source_audit_projection,
    "002 source audit projection",
  );
  if (
    receipts[2].body.source_audit_projection.machine_config_content_id !==
    scope.source_machine_config_content_id
  ) {
    throw new Error("002 source configuration does not match scope");
  }
  if (
    receipts[2].body.source_audit_projection.observed_at_utc !==
      receipts[2].observed_at_utc ||
    receipts[2].body.source_audit_projection.real_provisioner_observation !== false ||
    receipts[2].body.source_audit_projection.full_raw_audit_bound !== false
  ) {
    throw new Error("002 synthetic source observation contract drift");
  }

  const baseline = receipts[3].body;
  exactKeys(
    baseline,
    ["machine_observation", "firmware_observation", "baseline_cursors"],
    "003 baseline body",
  );
  exactKeys(
    baseline.machine_observation,
    [
      "machine_identity_sha256",
      "boot_identity_sha256",
      "computer_name",
      "machine_guid_sha256",
      "board_manufacturer",
      "board_product",
      "cpu_name",
      "gpu_name",
    ],
    "003 machine observation",
  );
  requireSha256(
    baseline.machine_observation.machine_identity_sha256,
    "003 machine identity",
  );
  requireSha256(baseline.machine_observation.boot_identity_sha256, "003 boot identity");
  for (const field of [
    "computer_name",
    "machine_guid_sha256",
    "board_manufacturer",
    "board_product",
    "cpu_name",
    "gpu_name",
  ]) {
    if (field === "machine_guid_sha256") {
      requireSha256(baseline.machine_observation[field], `003 machine.${field}`);
    } else {
      requireText(baseline.machine_observation[field], `003 machine.${field}`);
    }
  }
  const observedMachineCore = {
    computer_name: baseline.machine_observation.computer_name,
    machine_guid_sha256: baseline.machine_observation.machine_guid_sha256,
    board_manufacturer: baseline.machine_observation.board_manufacturer,
    board_product: baseline.machine_observation.board_product,
    cpu_name: baseline.machine_observation.cpu_name,
    gpu_name: baseline.machine_observation.gpu_name,
  };
  if (
    contentId(observedMachineCore) !==
    baseline.machine_observation.machine_identity_sha256
  ) {
    throw new Error("003 machine identity content ID drift");
  }
  exactKeys(
    baseline.firmware_observation,
    [
      "bios_vendor",
      "bios_version",
      "bios_release_date",
      "microcode_raw_little_endian_hex",
      "microcode_revision_integer",
      "minimum_microcode_revision_integer",
      "microcode_minimum_machine_verified",
      "intel_defaults_machine_verified",
      "cold_boot_machine_verified",
      "absence_of_xmp_overclock_undervolt_or_memory_tuning_machine_verified",
      "same_physical_chassis_machine_verified",
    ],
    "003 firmware observation",
  );
  requireInteger(
    baseline.firmware_observation.microcode_revision_integer,
    "003 microcode revision",
  );
  requireText(baseline.firmware_observation.bios_vendor, "003 BIOS vendor");
  requireText(baseline.firmware_observation.bios_version, "003 BIOS version");
  requireText(
    baseline.firmware_observation.bios_release_date,
    "003 BIOS release date",
  );
  if (
    !/^[0-9a-f]{8}$/u.test(
      baseline.firmware_observation.microcode_raw_little_endian_hex,
    )
  ) {
    throw new Error("003 microcode raw little-endian encoding drift");
  }
  if (
    baseline.firmware_observation.minimum_microcode_revision_integer !==
      protocol.firmware_gate.minimum_microcode_revision_integer ||
    baseline.firmware_observation.microcode_minimum_machine_verified !==
      (baseline.firmware_observation.microcode_revision_integer >=
        protocol.firmware_gate.minimum_microcode_revision_integer) ||
    baseline.firmware_observation.intel_defaults_machine_verified !== false ||
    baseline.firmware_observation.cold_boot_machine_verified !== false ||
    baseline.firmware_observation
      .absence_of_xmp_overclock_undervolt_or_memory_tuning_machine_verified !== false ||
    baseline.firmware_observation.same_physical_chassis_machine_verified !== false
  ) {
    throw new Error("003 firmware observation contract drift");
  }
  const baselineCursors = requireArray(baseline.baseline_cursors, "003 baseline cursors");
  if (baselineCursors.length !== 2) throw new Error("003 baseline cursor count drift");
  baselineCursors.forEach((cursor, index) =>
    validateCursor(cursor, `003 baseline cursors[${index}]`),
  );
  deepExact(
    baselineCursors.map((cursor) => cursor.log_name),
    CHANNEL_NAMES,
    "003 baseline channel order",
  );
  if (
    baselineCursors.some(
      (cursor) => cursor.captured_at_utc !== receipts[3].observed_at_utc,
    )
  ) {
    throw new Error("003 baseline cursor timestamp drift");
  }

  const launch = receipts[4].body;
  exactKeys(
    launch,
    [
      "execution_backend_id",
      "probe_definition_id",
      "fixed_executable",
      "fixed_argv",
      "stdout_relative_path",
      "stderr_relative_path",
      "launch_receipt_fsynced_before_process_creation",
      "process_creation_performed",
      "real_process_observation",
    ],
    "004 launch body",
  );
  requireSha256(launch.probe_definition_id, "004 probe definition ID");
  requireArray(launch.fixed_argv, "004 fixed argv");
  const expectedProbeDefinition = {
    execution_backend_id: SYNTHETIC_BACKEND_ID,
    fixed_executable: "node:test-double-no-process",
    fixed_argv: ["--synthetic-host-qualification"],
    stdout_relative_path: protocol.probe.stdout_relative_path,
    stderr_relative_path: protocol.probe.stderr_relative_path,
  };
  if (
    launch.execution_backend_id !== SYNTHETIC_BACKEND_ID ||
    launch.probe_definition_id !== contentId(expectedProbeDefinition) ||
    launch.fixed_executable !== expectedProbeDefinition.fixed_executable ||
    launch.stdout_relative_path !== expectedProbeDefinition.stdout_relative_path ||
    launch.stderr_relative_path !== expectedProbeDefinition.stderr_relative_path ||
    launch.launch_receipt_fsynced_before_process_creation !== true ||
    launch.process_creation_performed !== false ||
    launch.real_process_observation !== false
  ) {
    throw new Error("004 synthetic launch definition drift");
  }
  deepExact(launch.fixed_argv, expectedProbeDefinition.fixed_argv, "004 fixed argv");

  const exit = receipts[5].body;
  exactKeys(
    exit,
    [
      "exit_code",
      "timed_out",
      "descendants_quiesced_before_cooldown",
      "process_id",
      "real_process_observation",
      "stdout",
      "stderr",
    ],
    "005 exit body",
  );
  requireInteger(exit.exit_code, "005 exit code");
  requireBoolean(exit.timed_out, "005 timed_out");
  requireBoolean(
    exit.descendants_quiesced_before_cooldown,
    "005 descendants_quiesced_before_cooldown",
  );
  if (exit.process_id !== null) requireInteger(exit.process_id, "005 process_id");
  if (exit.process_id !== null || exit.real_process_observation !== false) {
    throw new Error("005 synthetic process observation drift");
  }
  for (const [name, stream] of [
    ["stdout", exit.stdout],
    ["stderr", exit.stderr],
  ]) {
    exactKeys(stream, ["path", "byte_count", "raw_sha256"], `005 ${name}`);
    requireRelativePosixPath(stream.path, `005 ${name}.path`);
    requireInteger(stream.byte_count, `005 ${name}.byte_count`);
    requireSha256(stream.raw_sha256, `005 ${name}.raw_sha256`);
  }
  if (
    exit.stdout.path !== protocol.probe.stdout_relative_path ||
    exit.stderr.path !== protocol.probe.stderr_relative_path
  ) {
    throw new Error("005 probe stream path drift");
  }

  exactKeys(receipts[6].body, ["delta"], "006 process delta body");
  validateDelta(
    receipts[6].body.delta,
    "006 process delta",
    protocol,
    "probe_process",
  );
  exactKeys(receipts[7].body, ["delta"], "007 cooldown delta body");
  validateDelta(receipts[7].body.delta, "007 cooldown delta", protocol, "cooldown");

  const handoff = receipts[8].body;
  exactKeys(
    handoff,
    ["terminal_tail_delta", "source_audit_projection_after", "handoff_cursors"],
    "008 handoff body",
  );
  validateDelta(
    handoff.terminal_tail_delta,
    "008 terminal tail delta",
    protocol,
    "terminal_tail",
  );
  validateSourceAuditShape(
    handoff.source_audit_projection_after,
    "008 source audit projection after",
  );
  const handoffCursors = requireArray(handoff.handoff_cursors, "008 handoff cursors");
  if (handoffCursors.length !== 2) throw new Error("008 handoff cursor count drift");
  handoffCursors.forEach((cursor, index) =>
    validateCursor(cursor, `008 handoff cursors[${index}]`),
  );
  if (
    receipts[6].body.delta.interval_start_exclusive_utc !==
      receipts[3].observed_at_utc ||
    receipts[6].body.delta.interval_end_inclusive_utc !==
      receipts[6].observed_at_utc ||
    receipts[7].body.delta.interval_start_exclusive_utc !==
      receipts[6].body.delta.interval_end_inclusive_utc ||
    receipts[7].body.delta.interval_end_inclusive_utc !==
      receipts[7].observed_at_utc ||
    handoff.terminal_tail_delta.interval_start_exclusive_utc !==
      receipts[7].body.delta.interval_end_inclusive_utc ||
    handoff.terminal_tail_delta.interval_end_inclusive_utc !==
      receipts[8].observed_at_utc ||
    handoff.source_audit_projection_after.observed_at_utc !==
      receipts[8].observed_at_utc ||
    handoff.source_audit_projection_after.real_provisioner_observation !== false ||
    handoff.source_audit_projection_after.full_raw_audit_bound !== false ||
    handoffCursors.some(
      (cursor) => cursor.captured_at_utc !== receipts[8].observed_at_utc,
    )
  ) {
    throw new Error("qualification observation time-chain drift");
  }
  if (
    receiptTimes[4] < receiptTimes[3] ||
    receiptTimes[5] < receiptTimes[4] ||
    receiptTimes[6] < receiptTimes[5] ||
    receiptTimes[9] < receiptTimes[8]
  ) {
    throw new Error("qualification launch/exit/report time order drift");
  }
}

function createSyntheticQualificationArtifact(options) {
  const {
    outputRoot,
    protocolPath = DEFAULT_PROTOCOL_PATH,
    repositoryRoot = DEFAULT_REPOSITORY_ROOT,
    verifySources = true,
    microcodeRawLeHex = "2f010000",
    faultEvents = [],
    sourceAuditExitCode = 0,
    operatorDeclarationSchemaVersion,
    processWindowName = "probe_process",
    postSourceConfigurationId,
    postHostId,
    postBootId,
  } = requireObject(options, "synthetic qualification options");
  const root = path.resolve(requireText(outputRoot, "outputRoot"));
  const loadedProtocol = loadProtocol({
    protocolPath: path.resolve(protocolPath),
    repositoryRoot: path.resolve(repositoryRoot),
    verifySources,
  });
  const { protocol, protocolId, protocolRawSha256 } = loadedProtocol;
  const decodedMicrocode = decodeMicrocodeLittleEndian(microcodeRawLeHex);
  requireInteger(sourceAuditExitCode, "sourceAuditExitCode");
  if (![0, 2].includes(sourceAuditExitCode)) {
    throw new Error(
      "sourceAuditExitCode must be Audit v2 exit 0 or nonconformance exit 2",
    );
  }

  const machineCore = {
    computer_name: "SYNTHETIC-HOST",
    machine_guid_sha256: "a".repeat(64),
    board_manufacturer: "SYNTHETIC",
    board_product: "SYNTHETIC-BOARD",
    cpu_name: "SYNTHETIC-CPU",
    gpu_name: "SYNTHETIC-GPU",
  };
  const machineId = contentId(machineCore);
  const bootId = sha256Bytes(Buffer.from("synthetic-boot-identity-v1", "utf8"));
  const sourceConfigurationId = sha256Bytes(
    Buffer.from("synthetic-event-log-source-config-v1", "utf8"),
  );
  const finalMachineId = postHostId ?? machineId;
  const finalBootId = postBootId ?? bootId;
  requireSha256(finalMachineId, "postHostId");
  requireSha256(finalBootId, "postBootId");
  const finalSourceConfigurationId =
    postSourceConfigurationId ?? sourceConfigurationId;
  requireSha256(finalSourceConfigurationId, "postSourceConfigurationId");

  const operatorDeclaration = {
    schema_version:
      operatorDeclarationSchemaVersion ??
      protocol.firmware_gate.operator_declaration_schema_version,
    bios_updated_with_microcode_at_least_0x12f: true,
    intel_defaults_loaded: true,
    cold_boot_completed_after_firmware_change: true,
    xmp_disabled: true,
    cpu_overclock_disabled: true,
    undervolt_disabled: true,
    memory_tuning_disabled: true,
    same_physical_chassis_as_host_block_receipt: true,
  };
  const operatorDeclarationId = contentId(operatorDeclaration);
  const scopeComponents = [
    protocolId,
    protocol.host_block.receipt_raw_sha256,
    machineId,
    bootId,
    sourceConfigurationId,
    operatorDeclarationId,
    SYNTHETIC_BACKEND_ID,
  ];
  const scopeId = domainSeparatedSha256(
    protocol.scope.scope_id_domain_separator,
    scopeComponents,
  );

  if (fs.existsSync(root)) throw new Error("qualification output root already exists");
  fs.mkdirSync(root, { recursive: false, mode: 0o700 });
  fs.mkdirSync(path.join(root, "streams"), { recursive: false, mode: 0o700 });

  let previousRawSha256 = null;
  const receipts = [];
  const writeReceipt = (sequence, timestamp, body) => {
    const receipt = makeReceipt(sequence, scopeId, previousRawSha256, timestamp, body);
    const written = writeCreateJson(path.join(root, RECEIPT_FILES[sequence]), receipt);
    previousRawSha256 = written.rawSha256;
    receipts.push(receipt);
  };

  writeReceipt(0, "2026-08-22T00:00:00Z", {
    qualification_protocol_id: protocolId,
    qualification_protocol_raw_sha256: protocolRawSha256,
    host_block_receipt_relative_path: protocol.host_block.receipt_relative_path,
    host_block_receipt_raw_sha256: protocol.host_block.receipt_raw_sha256,
    machine_identity_sha256: machineId,
    boot_identity_sha256: bootId,
    source_machine_config_content_id: sourceConfigurationId,
    operator_declaration_id: operatorDeclarationId,
    execution_backend_id: SYNTHETIC_BACKEND_ID,
    scope_components: scopeComponents,
    attempt_number: 1,
    retry_number: 0,
    real_host_observation: false,
  });
  writeReceipt(1, "2026-08-22T00:00:01Z", {
    operator_declaration_schema_version:
      protocol.firmware_gate.operator_declaration_schema_version,
    operator_declaration: operatorDeclaration,
    operator_declaration_id: operatorDeclarationId,
    operator_declaration_content_address_is_signature: false,
    declared_at_utc: "2026-08-22T00:00:01Z",
    attempt_number: 1,
    retry_budget: 0,
  });
  const sourceAuditBefore = makeSourceAuditProjection({
    observedAtUtc: "2026-08-22T00:00:02Z",
    machineId,
    bootId,
    sourceConfigurationId,
    auditExitCode: sourceAuditExitCode,
  });
  writeReceipt(2, "2026-08-22T00:00:02Z", {
    source_audit_projection: sourceAuditBefore,
  });

  const machineObservation = {
    machine_identity_sha256: machineId,
    boot_identity_sha256: bootId,
    ...machineCore,
  };
  const baselineCursors = [
    makeCursor("Application", 1000, "2026-08-22T00:00:03Z", machineId, bootId),
    makeCursor("System", 2000, "2026-08-22T00:00:03Z", machineId, bootId),
  ];
  writeReceipt(3, "2026-08-22T00:00:03Z", {
    machine_observation: machineObservation,
    firmware_observation: {
      bios_vendor: "SYNTHETIC",
      bios_version: "SYNTHETIC-0x12F",
      bios_release_date: "2026-08-22",
      microcode_raw_little_endian_hex: microcodeRawLeHex,
      microcode_revision_integer: decodedMicrocode,
      minimum_microcode_revision_integer:
        protocol.firmware_gate.minimum_microcode_revision_integer,
      microcode_minimum_machine_verified:
        decodedMicrocode >= protocol.firmware_gate.minimum_microcode_revision_integer,
      intel_defaults_machine_verified: false,
      cold_boot_machine_verified: false,
      absence_of_xmp_overclock_undervolt_or_memory_tuning_machine_verified: false,
      same_physical_chassis_machine_verified: false,
    },
    baseline_cursors: baselineCursors,
  });

  const probeDefinition = {
    execution_backend_id: SYNTHETIC_BACKEND_ID,
    fixed_executable: "node:test-double-no-process",
    fixed_argv: ["--synthetic-host-qualification"],
    stdout_relative_path: protocol.probe.stdout_relative_path,
    stderr_relative_path: protocol.probe.stderr_relative_path,
  };
  writeReceipt(4, "2026-08-22T00:00:04Z", {
    execution_backend_id: SYNTHETIC_BACKEND_ID,
    probe_definition_id: contentId(probeDefinition),
    fixed_executable: probeDefinition.fixed_executable,
    fixed_argv: probeDefinition.fixed_argv,
    stdout_relative_path: probeDefinition.stdout_relative_path,
    stderr_relative_path: probeDefinition.stderr_relative_path,
    launch_receipt_fsynced_before_process_creation: true,
    process_creation_performed: false,
    real_process_observation: false,
  });

  const stdoutRaw = Buffer.from("synthetic host qualification probe: no process executed\n", "utf8");
  const stderrRaw = Buffer.alloc(0);
  writeCreateFile(path.join(root, ...STREAM_FILES[0].split("/")), stdoutRaw);
  writeCreateFile(path.join(root, ...STREAM_FILES[1].split("/")), stderrRaw);
  writeReceipt(5, "2026-08-22T00:00:05Z", {
    exit_code: 0,
    timed_out: false,
    descendants_quiesced_before_cooldown: true,
    process_id: null,
    real_process_observation: false,
    stdout: {
      path: STREAM_FILES[0],
      byte_count: stdoutRaw.length,
      raw_sha256: sha256Bytes(stdoutRaw),
    },
    stderr: {
      path: STREAM_FILES[1],
      byte_count: stderrRaw.length,
      raw_sha256: sha256Bytes(stderrRaw),
    },
  });

  const normalizedEvents = normalizeSyntheticFaultEvents(
    faultEvents,
    baselineCursors,
    "2026-08-22T00:00:06Z",
  );
  const processDelta = makeDelta({
    windowName: processWindowName,
    startedAtUtc: "2026-08-22T00:00:03Z",
    endedAtUtc: "2026-08-22T00:00:06Z",
    startCursors: baselineCursors,
    events: normalizedEvents,
    machineId,
    bootId,
  });
  writeReceipt(6, "2026-08-22T00:00:06Z", { delta: processDelta });

  const cooldownDelta = makeDelta({
    windowName: "cooldown",
    startedAtUtc: "2026-08-22T00:00:06Z",
    endedAtUtc: "2026-08-22T00:05:06Z",
    startCursors: processDelta.end_cursors,
    events: [],
    machineId,
    bootId,
  });
  writeReceipt(7, "2026-08-22T00:05:06Z", { delta: cooldownDelta });

  const tailDelta = makeDelta({
    windowName: "terminal_tail",
    startedAtUtc: "2026-08-22T00:05:06Z",
    endedAtUtc: "2026-08-22T00:07:06Z",
    startCursors: cooldownDelta.end_cursors,
    events: [],
    machineId: finalMachineId,
    bootId: finalBootId,
  });
  const sourceAuditAfter = makeSourceAuditProjection({
    observedAtUtc: "2026-08-22T00:07:06Z",
    machineId: finalMachineId,
    bootId: finalBootId,
    sourceConfigurationId: finalSourceConfigurationId,
    auditExitCode: sourceAuditExitCode,
  });
  writeReceipt(8, "2026-08-22T00:07:06Z", {
    terminal_tail_delta: tailDelta,
    source_audit_projection_after: sourceAuditAfter,
    handoff_cursors: tailDelta.end_cursors,
  });

  const derived = deriveQualification(receipts, protocol);
  writeReceipt(9, "2026-08-22T00:07:07Z", derived);

  const inventory = MANIFEST_INVENTORY_FILES.map((relativePath) =>
    inventoryEntry(root, relativePath),
  );
  const manifestCore = {
    schema_version: MANIFEST_SCHEMA_VERSION,
    sequence: 10,
    scope_id: scopeId,
    previous_receipt_raw_sha256: previousRawSha256,
    qualification_protocol_id: protocolId,
    qualification_protocol_raw_sha256: protocolRawSha256,
    inventory,
  };
  const artifactId = contentId(manifestCore);
  const manifest = { ...manifestCore, artifact_id: artifactId };
  const manifestWritten = writeCreateJson(path.join(root, MANIFEST_FILE), manifest);

  const terminalCore = {
    schema_version: HOST_QUALIFICATION_TERMINAL_SCHEMA_VERSION,
    sequence: 11,
    scope_id: scopeId,
    previous_receipt_raw_sha256: manifestWritten.rawSha256,
    artifact_id: artifactId,
    qualification_protocol_id: protocolId,
    qualification_protocol_raw_sha256: protocolRawSha256,
    criteria_passed: derived.criteria_passed,
    real_host_observation: derived.real_host_observation,
    eligible_as_host_qualification_input:
      derived.eligible_as_host_qualification_input,
    cuda_execution_authorized: false,
    formal_evidence_authorized: false,
    production_active_authorized: false,
    appendable_proven: false,
    readable_proven: false,
    learnable_proven: false,
    steerable_proven: false,
    four_capability_claim_authorized: false,
    failure_codes: derived.failure_codes,
    handoff_cursors: tailDelta.end_cursors,
    completed_at_utc: "2026-08-22T00:07:07Z",
    claim_boundary: protocol.claim_boundary,
  };
  const terminalId = contentId(terminalCore);
  const terminalWritten = writeCreateJson(path.join(root, TERMINAL_FILE), {
    ...terminalCore,
    terminal_id: terminalId,
  });
  return Object.freeze({
    qualificationRoot: root,
    artifactId,
    terminalId,
    terminalRawSha256: terminalWritten.rawSha256,
  });
}

export function validateSyntheticQualificationArtifact(options) {
  const {
    qualificationRoot,
    protocolPath = DEFAULT_PROTOCOL_PATH,
    repositoryRoot = DEFAULT_REPOSITORY_ROOT,
    verifySources = true,
  } = requireObject(options, "synthetic qualification validation options");
  if (verifySources !== true) {
    throw new Error("full synthetic qualification validation requires source verification");
  }
  const root = path.resolve(requireText(qualificationRoot, "qualificationRoot"));
  const loadedProtocol = loadProtocol({
    protocolPath: path.resolve(protocolPath),
    repositoryRoot: path.resolve(repositoryRoot),
    verifySources,
  });
  const { protocol, protocolId, protocolRawSha256 } = loadedProtocol;

  const actualFiles = listRootFilesStrict(root);
  const expectedFiles = [...COMPLETE_FILES].sort();
  if (
    actualFiles.length !== expectedFiles.length ||
    actualFiles.some((entry, index) => entry !== expectedFiles[index])
  ) {
    throw new Error("qualification exact entry set drift (missing, extra, or unexpected entry)");
  }

  const loadedReceipts = RECEIPT_FILES.map((relativePath, index) =>
    loadStrictJsonFile(path.join(root, relativePath), `receipt ${index}`, true),
  );
  const receipts = loadedReceipts.map((loaded) => loaded.value);
  const scopeId = requireText(receipts[0].scope_id, "receipt 0 scope_id");
  let previousRawSha = null;
  for (let index = 0; index < receipts.length; index += 1) {
    receiptBody(receipts[index], index, scopeId, previousRawSha, `receipt ${index}`);
    previousRawSha = loadedReceipts[index].rawSha256;
  }
  validateReceiptBodies(receipts, protocol, protocolId, protocolRawSha256);

  const exit = receipts[5].body;
  for (const stream of [exit.stdout, exit.stderr]) {
    const raw = fs.readFileSync(path.join(root, ...stream.path.split("/")));
    if (raw.length !== stream.byte_count || sha256Bytes(raw) !== stream.raw_sha256) {
      throw new Error(`probe stream hash/byte-count drift: ${stream.path}`);
    }
  }

  const derived = deriveQualification(receipts, protocol);
  deepExact(receipts[9].body, derived, "009 qualification report derived body");

  const loadedManifest = loadStrictJsonFile(
    path.join(root, MANIFEST_FILE),
    "qualification manifest",
    true,
  );
  const manifest = loadedManifest.value;
  exactKeys(
    manifest,
    [
      "schema_version",
      "sequence",
      "scope_id",
      "previous_receipt_raw_sha256",
      "qualification_protocol_id",
      "qualification_protocol_raw_sha256",
      "inventory",
      "artifact_id",
    ],
    "qualification manifest",
  );
  if (
    manifest.schema_version !== MANIFEST_SCHEMA_VERSION ||
    manifest.sequence !== 10 ||
    manifest.scope_id !== scopeId ||
    manifest.previous_receipt_raw_sha256 !== previousRawSha ||
    manifest.qualification_protocol_id !== protocolId ||
    manifest.qualification_protocol_raw_sha256 !== protocolRawSha256
  ) {
    throw new Error("qualification manifest lineage drift");
  }
  const expectedInventory = MANIFEST_INVENTORY_FILES.map((relativePath) =>
    inventoryEntry(root, relativePath),
  );
  deepExact(manifest.inventory, expectedInventory, "qualification manifest inventory");
  const manifestCore = {
    schema_version: manifest.schema_version,
    sequence: 10,
    scope_id: scopeId,
    previous_receipt_raw_sha256: previousRawSha,
    qualification_protocol_id: protocolId,
    qualification_protocol_raw_sha256: protocolRawSha256,
    inventory: expectedInventory,
  };
  const artifactId = contentId(manifestCore);
  if (manifest.artifact_id !== artifactId) throw new Error("qualification artifact ID drift");

  const loadedTerminal = loadStrictJsonFile(
    path.join(root, TERMINAL_FILE),
    "qualification terminal",
    true,
  );
  const terminal = loadedTerminal.value;
  exactKeys(
    terminal,
    [
      "schema_version",
      "sequence",
      "scope_id",
      "previous_receipt_raw_sha256",
      "artifact_id",
      "qualification_protocol_id",
      "qualification_protocol_raw_sha256",
      "criteria_passed",
      "real_host_observation",
      "eligible_as_host_qualification_input",
      "cuda_execution_authorized",
      "formal_evidence_authorized",
      "production_active_authorized",
      "appendable_proven",
      "readable_proven",
      "learnable_proven",
      "steerable_proven",
      "four_capability_claim_authorized",
      "failure_codes",
      "handoff_cursors",
      "completed_at_utc",
      "claim_boundary",
      "terminal_id",
    ],
    "qualification terminal",
  );
  const terminalCore = {
    schema_version: HOST_QUALIFICATION_TERMINAL_SCHEMA_VERSION,
    sequence: 11,
    scope_id: scopeId,
    previous_receipt_raw_sha256: loadedManifest.rawSha256,
    artifact_id: artifactId,
    qualification_protocol_id: protocolId,
    qualification_protocol_raw_sha256: protocolRawSha256,
    criteria_passed: derived.criteria_passed,
    real_host_observation: derived.real_host_observation,
    eligible_as_host_qualification_input:
      derived.eligible_as_host_qualification_input,
    cuda_execution_authorized: false,
    formal_evidence_authorized: false,
    production_active_authorized: false,
    appendable_proven: false,
    readable_proven: false,
    learnable_proven: false,
    steerable_proven: false,
    four_capability_claim_authorized: false,
    failure_codes: derived.failure_codes,
    handoff_cursors: receipts[8].body.handoff_cursors,
    completed_at_utc: receipts[9].observed_at_utc,
    claim_boundary: protocol.claim_boundary,
  };
  const terminalId = contentId(terminalCore);
  deepExact(
    terminal,
    { ...terminalCore, terminal_id: terminalId },
    "qualification terminal derived payload",
  );

  if (
    terminal.criteria_passed !== false ||
    terminal.real_host_observation !== false ||
    terminal.eligible_as_host_qualification_input !== false
  ) {
    throw new Error("synthetic qualification terminal attempted to claim eligibility");
  }

  return Object.freeze({
    integrityValid: true,
    artifactId,
    terminalId,
    terminalRawSha256: loadedTerminal.rawSha256,
    sourceLineageVerified: true,
    criteriaPassed: derived.criteria_passed,
    realHostObservation: derived.real_host_observation,
    validatedEligibleAsHostQualificationInput: false,
    failureCodes: Object.freeze([...derived.failure_codes]),
  });
}

export function adaptProvisionerAuditV2Artifact(options) {
  const input = requireObject(options, "raw Audit adapter options");
  exactKeys(
    input,
    [
      "artifactRoot",
      "auditRelativePath",
      "captureEnvelope",
      "expectedProtocolId",
      "expectedProtocolRawSha256",
      "expectedMachineConfigContentId",
    ],
    "raw Audit adapter options",
  );
  const expectedProtocolId = requireSha256(
    input.expectedProtocolId,
    "expectedProtocolId",
  );
  const expectedProtocolRawSha256 = requireSha256(
    input.expectedProtocolRawSha256,
    "expectedProtocolRawSha256",
  );
  const loadedProtocol = loadProtocol({
    protocolPath: DEFAULT_PROTOCOL_PATH,
    repositoryRoot: DEFAULT_REPOSITORY_ROOT,
    verifySources: true,
  });
  if (
    loadedProtocol.protocolId !== expectedProtocolId ||
    loadedProtocol.protocolRawSha256 !== expectedProtocolRawSha256
  ) {
    throw new Error("bundled qualification protocol identity does not match the external pin");
  }
  const captureEnvelope = requireObject(input.captureEnvelope, "captureEnvelope");
  validateCaptureEnvelope(captureEnvelope, loadedProtocol.protocol);
  let expectedMachineConfigContentId = input.expectedMachineConfigContentId;
  if (captureEnvelope.capture_role === "qualification_source_audit_before") {
    if (expectedMachineConfigContentId !== null) {
      throw new Error("before-source Audit must discover, not preclaim, machine-config content ID");
    }
  } else {
    expectedMachineConfigContentId = requireSha256(
      expectedMachineConfigContentId,
      "expectedMachineConfigContentId",
    );
  }

  const raw = readSingleAuditArtifact(
    input.artifactRoot,
    input.auditRelativePath,
    captureEnvelope.stdout_byte_count,
    loadedProtocol.protocol.event_log_source.maximum_raw_audit_bytes,
  );
  if (
    raw.length !== captureEnvelope.stdout_byte_count ||
    sha256Bytes(raw) !== captureEnvelope.stdout_raw_sha256
  ) {
    throw new Error("raw Audit artifact does not match the capture envelope identity");
  }
  const loadedAudit = loadStrictProvisionerJsonBytes(
    raw,
    loadedProtocol.protocol.event_log_source.maximum_raw_audit_bytes,
  );
  const provisionerPath = loadedProtocol.protocol.event_log_source.provisioner_relative_path;
  const checkoutProvisionerIdentity = sourceTextIdentity(
    resolveRepositoryPath(DEFAULT_REPOSITORY_ROOT, provisionerPath),
  );
  const derived = validateAndDeriveProvisionerAudit(
    loadedAudit.value,
    loadedProtocol.protocol,
    captureEnvelope,
    checkoutProvisionerIdentity,
    expectedMachineConfigContentId,
  );

  const captureCore = {
    schema_version: captureEnvelope.schema_version,
    capture_role: captureEnvelope.capture_role,
    stdout_raw_sha256: captureEnvelope.stdout_raw_sha256,
    stdout_byte_count: captureEnvelope.stdout_byte_count,
    stderr_raw_sha256: captureEnvelope.stderr_raw_sha256,
    stderr_byte_count: captureEnvelope.stderr_byte_count,
    process_exit_code: captureEnvelope.process_exit_code,
    process_started_at_utc: captureEnvelope.process_started_at_utc,
    process_exited_at_utc: captureEnvelope.process_exited_at_utc,
    stdout_captured_at_utc: captureEnvelope.stdout_captured_at_utc,
    machine_identity_sha256: captureEnvelope.machine_identity_sha256,
    boot_identity_sha256: captureEnvelope.boot_identity_sha256,
    capture_authoritative: false,
  };
  const diagnosticFailureCodes = [];
  if (!derived.overallConformant) {
    diagnosticFailureCodes.push("raw_audit_recomputed_nonconformant");
  }
  diagnosticFailureCodes.push(
    "production_capture_untrusted",
    "independent_control_plane_reobservation_missing",
  );
  const snapshotCore = {
    schema_version: EVENT_LOG_SOURCE_AUDIT_ADAPTER_SNAPSHOT_SCHEMA_VERSION,
    snapshot_id_method: "sha256_of_canonical_snapshot_core_without_snapshot_id",
    qualification_protocol: {
      protocol_id: loadedProtocol.protocolId,
      protocol_raw_sha256: loadedProtocol.protocolRawSha256,
    },
    capture: {
      ...captureCore,
      capture_envelope_id: contentId(captureCore),
    },
    raw_audit: {
      schema_version: loadedAudit.value.schema_version,
      raw_sha256: loadedAudit.rawSha256,
      byte_count: loadedAudit.raw.length,
      process_exit_code: derived.derivedExitCode,
      observed_at_utc: loadedAudit.value.observed_at_utc,
      overall_conformant: derived.overallConformant,
      result_disposition: loadedAudit.value.result_disposition,
    },
    source_lineage: {
      provisioner_relative_path: provisionerPath,
      source_hash_mode: loadedProtocol.protocol.source_hash_mode,
      protocol_lf_sha256: loadedProtocol.protocol.source_sha256[provisionerPath],
      checkout_lf_sha256: checkoutProvisionerIdentity.lfCanonicalSha256,
      audit_observed_lf_sha256:
        loadedAudit.value.script_integrity.observed_lf_canonical_sha256,
      lf_canonical_byte_count: checkoutProvisionerIdentity.lfCanonicalByteCount,
      source_pin_revalidated: true,
      provenance_authoritative: false,
    },
    machine_identity_sha256: loadedAudit.value.machine.machine_identity_sha256,
    boot_identity_sha256: captureEnvelope.boot_identity_sha256,
    machine_config: {
      config_schema_version: loadedAudit.value.config_schema_version,
      content_id: loadedAudit.value.machine_config_content_id,
      basis_sha256: derived.machineConfig.basisSha256,
      basis_byte_count: derived.machineConfig.basisBytes.length,
      basis_strict_json_valid: true,
      basis_matches_reconstructed_core: true,
      caller_expected_content_id_matched: expectedMachineConfigContentId !== null,
    },
    recomputed_conformance: derived.conformance,
    verification: {
      exact_schema_revalidated: true,
      stdout_artifact_matches_capture_claim: true,
      receipt_exit_matches_untrusted_capture_claim: true,
      raw_audit_content_id_basis_revalidated: true,
      claimed_conformance_revalidated: true,
      invariants_recomputed: true,
      full_raw_audit_bound: true,
    },
    boundary: {
      capture_envelope_authoritative: false,
      content_hashes_are_identity_signatures: false,
      caller_claimed_boot_identity_content_bound: true,
      raw_audit_boot_identity_observed: false,
      trusted_provisioner_execution_proven: false,
      independent_control_plane_reobservation_performed: false,
      refresh_chronology_proven: false,
      continuous_stability_proven: false,
      cross_role_replay_excluded: false,
      projection_emitted: false,
      real_provisioner_observation: false,
      eligible_as_host_qualification_input: false,
      cuda_execution_authorized: false,
      formal_evidence_authorized: false,
      production_active_authorized: false,
      four_capability_claim_authorized: false,
      tamper_resistance_proven: false,
    },
    diagnostic_failure_codes: diagnosticFailureCodes,
  };
  return deepFreeze({
    ...snapshotCore,
    snapshot_id: contentId(snapshotCore),
  });
}

export function preregisterHostQualification() {
  throw new Error(PRODUCTION_DISABLED_MESSAGE);
}

export function runHostQualification() {
  throw new Error(PRODUCTION_DISABLED_MESSAGE);
}

export function validateHostQualification() {
  throw new Error(PRODUCTION_DISABLED_MESSAGE);
}

export const __testing = Object.freeze({
  createSyntheticQualificationArtifact,
  decodeMicrocodeLittleEndian,
});
