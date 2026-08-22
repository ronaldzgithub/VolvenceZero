/**
 * Independent Windows host control plane for the strict 32767+1 diagnostic.
 *
 * This module deliberately uses only Node built-ins.  The outer process owns
 * preregistration, the one-use lease, child-process observation, local Windows
 * Event Log cross-anchors, and the append-only receipt chain.  It never imports
 * Python, torch, CUDA, transformers, or the Volvence runtime.
 */

import childProcess from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const HOST_CAMPAIGN_PROTOCOL_SCHEMA_VERSION =
  "windows-cuda-strict-32k-host-campaign.v1";
export const HOST_QUALIFICATION_TERMINAL_SCHEMA_VERSION =
  "windows-cuda-host-stability-qualification-terminal.v1";

const MODULE_PATH = fileURLToPath(import.meta.url);
const MODULE_DIR = path.dirname(MODULE_PATH);
const REPOSITORY_ROOT = path.resolve(MODULE_DIR, "../../../../../");
const DEFAULT_PROTOCOL_PATH = path.join(
  MODULE_DIR,
  "protocols",
  "windows_cuda_strict_32k_host_campaign_v1.json",
);
const DEFAULT_CAMPAIGN_BASE_DIR = path.join(
  REPOSITORY_ROOT,
  "artifacts",
  "relationship_lab",
  "windows_cuda_strict_32k_host_campaigns",
);

const RECEIPT_FILES = Object.freeze({
  scopeClaim: "000_scope_claim.json",
  eventLogBaseline: "001_event_log_baseline.json",
  preregistration: "002_preregistration.json",
  preregistrationAnchor: "003_preregistration_event_anchor.json",
  launch: "004_launch.json",
  launchAnchor: "005_launch_event_anchor.json",
  processExit: "006_process_exit.json",
  eventLogDelta: "007_event_log_delta.json",
  campaignReport: "008_campaign_report.json",
  manifest: "009_manifest.json",
  terminal: "010_terminal_receipt.json",
  terminalAnchor: "011_terminal_event_anchor.json",
  seal: "012_campaign_seal.json",
});
const PREREGISTERED_FILES = Object.freeze([
  RECEIPT_FILES.scopeClaim,
  RECEIPT_FILES.eventLogBaseline,
  RECEIPT_FILES.preregistration,
  RECEIPT_FILES.preregistrationAnchor,
]);
const PRE_MANIFEST_OUTER_FILES = Object.freeze([
  ...PREREGISTERED_FILES,
  RECEIPT_FILES.launch,
  "streams/child.stdout.log",
  "streams/child.stderr.log",
  RECEIPT_FILES.launchAnchor,
  RECEIPT_FILES.processExit,
  RECEIPT_FILES.eventLogDelta,
  RECEIPT_FILES.campaignReport,
]);
const POST_MANIFEST_FILES = Object.freeze([
  RECEIPT_FILES.manifest,
  RECEIPT_FILES.terminal,
  RECEIPT_FILES.terminalAnchor,
  RECEIPT_FILES.seal,
]);
const CHILD_REQUIRED_FILES = Object.freeze([
  "launch_receipt.json",
  "execution_attestation.json",
  "strict_32k_smoke_report.json",
  "manifest.json",
  "completion_receipt.json",
]);
const CHILD_PAYLOAD_FILES = Object.freeze([
  "launch_receipt.json",
  "execution_attestation.json",
  "strict_32k_smoke_report.json",
]);
const CHILD_ATTESTATION_KEYS = Object.freeze([
  "schema_version",
  "profile_id",
  "preset_name",
  "model_id",
  "model_revision",
  "model_weights_sha256",
  "execution_assets_sha256",
  "runtime_origin",
  "platform_system",
  "platform_release",
  "device",
  "device_name",
  "python_version",
  "torch_version",
  "transformers_version",
  "cuda_version",
  "cudnn_version",
  "device_compute_capability",
  "attention_implementation",
  "sdpa_backend",
  "sdpa_backend_policy",
  "sdpa_backend_exclusive",
  "generation_use_cache",
  "require_generation_chat_template",
  "generation_capture_strategy",
  "capture_failure_mode",
  "context_window_tokens",
  "local_files_only",
  "fallback_mode",
  "fail_on_truncation",
  "model_dtype",
  "hidden_size",
  "model_max_position_embeddings",
  "hook_layer_indices",
  "attestation_id",
]);
const CAMPAIGN_CRITICAL_SOURCE_PATHS = Object.freeze([
  "packages/vz-runtime/src/volvence_zero/offline_evidence/windows_cuda_strict_32k_host_campaign.mjs",
  "packages/vz-runtime/src/volvence_zero/offline_evidence/collect_windows_host_event_log.ps1",
  "scripts/run_windows_cuda_strict_32k_host_campaign.mjs",
]);
const PINNED_CHILD_PROTOCOL_ID =
  "4934a344550aab5c98f33892dd6d1ec2e5fe51c00694d2cc5b0a45fbc31e2c1a";
const PINNED_CHILD_PROTOCOL_RAW_SHA256 =
  "ec7b7bcec82668ac89b549b8769997897d07720a507ded6a316d4cee3b785eb4";
const PINNED_CHILD_PROTOCOL_RELATIVE_PATH =
  "packages/vz-runtime/src/volvence_zero/offline_evidence/protocols/windows_cuda_strict_32k_smoke_v1.json";
const PINNED_CHILD_RUNNER_RELATIVE_PATH = "scripts/run_windows_cuda_strict_32k_smoke.py";
const PINNED_CHILD_OUTPUT_RELATIVE_PATH = "child/strict_32k_smoke";
const PINNED_FAULT_CLASSIFICATION = Object.freeze([
  Object.freeze({
    failure_code: "new_whea_event",
    log_name: "System",
    provider_name: "Microsoft-Windows-WHEA-Logger",
    event_ids: null,
  }),
  Object.freeze({
    failure_code: "new_bugcheck_or_unexpected_shutdown",
    log_name: "System",
    provider_name: "Microsoft-Windows-Kernel-Power",
    event_ids: Object.freeze([41]),
  }),
  Object.freeze({
    failure_code: "new_bugcheck_or_unexpected_shutdown",
    log_name: "System",
    provider_name: "EventLog",
    event_ids: Object.freeze([6008]),
  }),
  Object.freeze({
    failure_code: "new_bugcheck_or_unexpected_shutdown",
    log_name: "System",
    provider_name: "Microsoft-Windows-WER-SystemErrorReporting",
    event_ids: Object.freeze([1001]),
  }),
  Object.freeze({
    failure_code: "event_log_cleared_or_rolled_over",
    log_name: "System",
    provider_name: "Microsoft-Windows-Eventlog",
    event_ids: Object.freeze([104]),
  }),
  Object.freeze({
    failure_code: "new_gpu_driver_fault",
    log_name: "System",
    provider_name: "Display",
    event_ids: Object.freeze([4101]),
  }),
  Object.freeze({
    failure_code: "new_gpu_driver_fault",
    log_name: "System",
    provider_name: "nvlddmkm",
    event_ids: null,
  }),
  Object.freeze({
    failure_code: "child_process_crash_event",
    log_name: "Application",
    provider_name: "Application Error",
    event_ids: Object.freeze([1000]),
  }),
  Object.freeze({
    failure_code: "child_process_crash_event",
    log_name: "Application",
    provider_name: "Windows Error Reporting",
    event_ids: Object.freeze([1001]),
  }),
]);

const PRODUCTION_BACKEND_ID = "windows-powershell-eventlog-and-node-spawn.v1";
const SYNTHETIC_TEST_BACKEND_ID = "synthetic-node-test-double.v1";
const CHILD_ENVIRONMENT_REMOVED = Object.freeze([
  "PYTHONINSPECT",
  "PYTHONPATH",
  "PYTHONSTARTUP",
  "PYTHONUSERBASE",
  "PYTHONHOME",
]);

const FAILURE_CODE_ORDER = Object.freeze([
  "synthetic_test_backend_not_evidence",
  "process_start_failed",
  "child_timeout",
  "child_diagnostic_failed_exit_2",
  "child_runtime_failed_nonzero",
  "child_root_missing_or_incomplete",
  "child_lineage_mismatch",
  "event_log_collection_failed",
  "event_log_continuity_lost",
  "event_log_cleared_or_rolled_over",
  "new_whea_event",
  "new_bugcheck_or_unexpected_shutdown",
  "new_gpu_driver_fault",
  "child_process_crash_event",
  "local_anchor_mismatch",
]);

const OUTER_EVIDENCE_FIREWALL = Object.freeze({
  outer_local_append_only_policy_present: true,
  local_event_log_cross_anchor_required: true,
  external_append_only_anchor_present: false,
  third_party_or_remote_anchor_present: false,
  local_admin_tamper_resistance_proven: false,
  unprivileged_local_event_forgery_excluded: false,
  hidden_unregistered_execution_excluded: false,
  powershell_executable_trust_proven: false,
  node_control_plane_runtime_attested: false,
  realized_child_environment_fully_frozen: false,
  child_process_tree_containment_proven: false,
  child_transitive_local_source_closure_pinned: false,
  producer_full_artifact_revalidated_before_pass_return: false,
  terminal_anchor_and_delayed_faults_covered_by_delta: false,
  child_standalone_artifact_proves_physical_execution: false,
  general_host_stability_proven: false,
  long_context_information_utilization_proven: false,
  appendable_proven: false,
  readable_proven: false,
  learnable_proven: false,
  steerable_proven: false,
  four_capability_claim_authorized: false,
  independent_subject_evidence: false,
  long_companion_multi_session_evidence: false,
  production_active_authorized: false,
  formal_evidence_authorized: false,
});

const OUTER_CLAIM_BOUNDARY =
  "Production preregistration is disabled in this v1 protocol. Known activation " +
  "blockers include a pinned host-qualification protocol with full artifact and " +
  "qualification-to-baseline continuity validation, provisioned Event Log " +
  "infrastructure, attested PowerShell/Python/Node runtime identity, a frozen " +
  "realized child environment, a complete transitive local source/import " +
  "closure, producer-side full-artifact revalidation before any PASS return, " +
  "post-terminal/delayed-fault Event Log coverage, Event Log cross-binding, " +
  "and child process-tree containment. " +
  "Enabling production requires a separately " +
  "audited protocol revision; closing only the qualification gate is insufficient. " +
  "Synthetic dependency tests are explicitly marked, can never PASS, and are " +
  "rejected by the public validator. A future authorized complete PASS campaign " +
  "would record one deterministic, preregistered local attempt scope, one " +
  "lease consumption before child creation, one fixed-argv child process " +
  "observation, the child artifact, and a bounded Windows Event " +
  "Log interval under a create-only receipt chain. It supports only a " +
  "cooperative local-machine observation that the reviewed-entrypoint strict 32767+1 " +
  "engineering diagnostic completed without a newly observed classified host " +
  "fault in that interval. Windows Event Log is a second local channel, not an " +
  "independent or WORM authority; an administrator or an unprivileged local " +
  "principal with Application-log write access can delete, clear, copy, or forge " +
  "local state, and unregistered hidden executions cannot be excluded. " +
  "The campaign does not prove general host stability, long-context information " +
  "utilization, independent subjects, long companionship, multiple sessions, " +
  "production ACTIVE, Appendable, Readable, Learnable, Steerable, or the " +
  "four-capability system claim. A consumed chain without the final seal is " +
  "permanently incomplete and never authorizes retry or PASS.";

function pythonCanonicalFloat(value) {
  if (!Number.isFinite(value)) throw new TypeError("JSON float must be finite");
  if (Object.is(value, -0)) return "-0.0";
  if (value === 0) return "0.0";
  const sign = value < 0 ? "-" : "";
  const shortest = Math.abs(value).toString().toLowerCase();
  let digits;
  let exponent;
  if (shortest.includes("e")) {
    const [coefficient, exponentText] = shortest.split("e");
    digits = coefficient.replace(".", "");
    exponent = Number(exponentText) + coefficient.split(".")[0].length - 1;
  } else {
    const [integerPart, fractionalPart = ""] = shortest.split(".");
    if (integerPart !== "0") {
      digits = `${integerPart}${fractionalPart}`;
      exponent = integerPart.length - 1;
    } else {
      const firstNonzero = fractionalPart.search(/[1-9]/);
      if (firstNonzero < 0) throw new Error("nonzero float lost all significant digits");
      digits = fractionalPart.slice(firstNonzero);
      exponent = -firstNonzero - 1;
    }
  }
  if (exponent < -4 || exponent >= 16) {
    const coefficient = digits.length === 1 ? digits : `${digits[0]}.${digits.slice(1)}`;
    const exponentSign = exponent < 0 ? "-" : "+";
    return `${sign}${coefficient}e${exponentSign}${String(Math.abs(exponent)).padStart(2, "0")}`;
  }
  const decimalPosition = exponent + 1;
  let body;
  if (decimalPosition <= 0) {
    body = `0.${"0".repeat(-decimalPosition)}${digits}`;
  } else if (decimalPosition >= digits.length) {
    body = `${digits}${"0".repeat(decimalPosition - digits.length)}.0`;
  } else {
    body = `${digits.slice(0, decimalPosition)}.${digits.slice(decimalPosition)}`;
  }
  return `${sign}${body}`;
}

class JsonNumber {
  constructor(raw) {
    this.raw = raw;
    this.isFloat = raw.includes(".") || raw.includes("e") || raw.includes("E");
    if (raw === "-0") {
      throw new SyntaxError("JSON integer -0 is not Python-canonical");
    }
    this.value = Number(raw);
    if (!Number.isFinite(this.value)) {
      throw new TypeError("JSON number must be finite");
    }
    if (!this.isFloat && !Number.isSafeInteger(this.value)) {
      throw new RangeError("JSON integer exceeds the exact safe range");
    }
    if (this.isFloat && raw !== pythonCanonicalFloat(this.value)) {
      throw new SyntaxError(`JSON float is not Python-canonical: ${raw}`);
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
    if (this.index !== this.text.length) {
      this.#fail("trailing content");
    }
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
      return this.#parseNumber();
    }
    this.#fail("invalid value");
  }

  #parseObject() {
    this.index += 1;
    const result = Object.create(null);
    const seen = new Set();
    this.#skipWhitespace();
    if (this.text[this.index] === "}") {
      this.index += 1;
      return result;
    }
    while (true) {
      this.#skipWhitespace();
      if (this.text[this.index] !== '"') this.#fail("object key must be text");
      const key = this.#parseString();
      if (seen.has(key)) this.#fail(`duplicate object key ${JSON.stringify(key)}`);
      seen.add(key);
      this.#skipWhitespace();
      if (this.text[this.index] !== ":") this.#fail("missing object colon");
      this.index += 1;
      this.#skipWhitespace();
      result[key] = this.#parseValue();
      this.#skipWhitespace();
      const character = this.text[this.index];
      if (character === "}") {
        this.index += 1;
        return result;
      }
      if (character !== ",") this.#fail("missing object comma");
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
      const character = this.text[this.index];
      if (character === "]") {
        this.index += 1;
        return result;
      }
      if (character !== ",") this.#fail("missing array comma");
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
        const raw = this.text.slice(start, this.index);
        try {
          return JSON.parse(raw);
        } catch (error) {
          throw new SyntaxError(`${this.label}: invalid JSON string at ${start}`, {
            cause: error,
          });
        }
      }
      if (!escaped && character.charCodeAt(0) < 0x20) {
        this.#fail("unescaped control character in string");
      }
      if (!escaped && character === "\\") {
        escaped = true;
      } else {
        escaped = false;
      }
      this.index += 1;
    }
    this.#fail("unterminated string");
  }

  #parseNumber() {
    const match = /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/.exec(
      this.text.slice(this.index),
    );
    if (match === null) this.#fail("invalid number");
    this.index += match[0].length;
    return new JsonNumber(match[0]);
  }

  #parseLiteral(raw, value) {
    if (!this.text.startsWith(raw, this.index)) this.#fail(`invalid ${raw} literal`);
    this.index += raw.length;
    return value;
  }

  #skipWhitespace() {
    while (
      this.index < this.text.length &&
      (this.text[this.index] === " " ||
        this.text[this.index] === "\n" ||
        this.text[this.index] === "\r" ||
        this.text[this.index] === "\t")
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
  if (value instanceof JsonNumber) return value.raw;
  if (value === null) return "null";
  if (typeof value === "string") return JSON.stringify(value);
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") {
    if (!Number.isSafeInteger(value)) {
      throw new TypeError("outer campaign JSON numbers must be safe integers");
    }
    return String(value);
  }
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (typeof value === "object") {
    const keys = Object.keys(value).sort();
    return `{${keys
      .map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`)
      .join(",")}}`;
  }
  throw new TypeError(`unsupported JSON value type: ${typeof value}`);
}

function canonicalBytes(value, newline = true) {
  return Buffer.from(canonicalJson(value) + (newline ? "\n" : ""), "utf8");
}

function sha256Bytes(payload) {
  return crypto.createHash("sha256").update(payload).digest("hex");
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

function exactKeys(value, expected, label) {
  const actual = Object.keys(requireObject(value, label)).sort();
  const wanted = [...expected].sort();
  if (actual.length !== wanted.length || actual.some((key, index) => key !== wanted[index])) {
    throw new Error(`${label} keys drifted`);
  }
}

function requireObject(value, label) {
  if (value === null || Array.isArray(value) || typeof value !== "object" || value instanceof JsonNumber) {
    throw new TypeError(`${label} must be an exact object`);
  }
  return value;
}

function requireArray(value, label) {
  if (!Array.isArray(value)) throw new TypeError(`${label} must be an exact array`);
  return value;
}

function requireText(value, label) {
  if (typeof value !== "string" || value.trim() === "") {
    throw new TypeError(`${label} must be nonempty text`);
  }
  return value;
}

function requireBoolean(value, label) {
  if (typeof value !== "boolean") throw new TypeError(`${label} must be an exact bool`);
  return value;
}

function requireInteger(value, label) {
  if (value instanceof JsonNumber) {
    if (value.isFloat || !Number.isSafeInteger(value.value)) {
      throw new TypeError(`${label} must be an exact safe integer`);
    }
    return value.value;
  }
  if (!Number.isSafeInteger(value)) throw new TypeError(`${label} must be an exact safe integer`);
  return value;
}

function requireNumber(value, label) {
  if (value instanceof JsonNumber) return value.value;
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be a finite number`);
  }
  return value;
}

function requireSha256(value, label) {
  const text = requireText(value, label);
  if (!/^[0-9a-f]{64}$/.test(text)) throw new Error(`${label} must be a lowercase SHA-256`);
  return text;
}

function requireUtcTimestamp(value, label) {
  const text = requireText(value, label);
  const match =
    /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,7}))?Z$/.exec(text);
  if (match === null) {
    throw new Error(`${label} must be UTC ISO-8601 text`);
  }
  const [, yearText, monthText, dayText, hourText, minuteText, secondText, fraction = ""] =
    match;
  const [year, month, day, hour, minute, second] = [
    yearText,
    monthText,
    dayText,
    hourText,
    minuteText,
    secondText,
  ].map(Number);
  const leap = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
  const daysInMonth = [31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  if (
    year < 2000 ||
    month < 1 ||
    month > 12 ||
    day < 1 ||
    day > daysInMonth[month - 1] ||
    hour > 23 ||
    minute > 59 ||
    second > 59
  ) {
    throw new Error(`${label} is not a valid UTC calendar timestamp`);
  }
  const epochMilliseconds = Date.UTC(year, month - 1, day, hour, minute, second, 0);
  if (!Number.isSafeInteger(epochMilliseconds)) {
    throw new Error(`${label} is outside the exact supported timestamp range`);
  }
  return BigInt(epochMilliseconds) * 10_000n + BigInt(fraction.padEnd(7, "0") || "0");
}

function requireRelativePosixPath(value, label) {
  const text = requireText(value, label);
  if (
    text.includes("\\") ||
    text.startsWith("/") ||
    text === "." ||
    text === ".." ||
    text.split("/").some((part) => part === "" || part === "." || part === "..")
  ) {
    throw new Error(`${label} must be a canonical relative POSIX path`);
  }
  return text;
}

function deepExact(actual, expected, label) {
  if (actual instanceof JsonNumber || expected instanceof JsonNumber) {
    if (actual instanceof JsonNumber && typeof expected === "number") {
      if (!Number.isSafeInteger(expected) || actual.isFloat || actual.value !== expected) {
        throw new TypeError(`${label} numeric type/value drift`);
      }
      return;
    }
    if (expected instanceof JsonNumber && typeof actual === "number") {
      if (!Number.isSafeInteger(actual) || expected.isFloat || expected.value !== actual) {
        throw new TypeError(`${label} numeric type/value drift`);
      }
      return;
    }
    if (!(actual instanceof JsonNumber) || !(expected instanceof JsonNumber)) {
      throw new TypeError(`${label} numeric type drift`);
    }
    if (actual.raw !== expected.raw) {
      throw new Error(`${label} numeric lexical/value drift`);
    }
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
    for (const key of Object.keys(expected)) deepExact(object[key], expected[key], `${label}.${key}`);
    return;
  }
  if (typeof actual !== typeof expected) throw new TypeError(`${label} type drift`);
  if (actual !== expected) throw new Error(`${label} value drift`);
}

function loadStrictJsonFile(filePath, label, canonicalRequired = false) {
  const stat = fs.lstatSync(filePath);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1) {
    throw new Error(`${label} must be one regular, non-linked file`);
  }
  const raw = fs.readFileSync(filePath);
  if (raw.subarray(0, 3).equals(Buffer.from([0xef, 0xbb, 0xbf]))) {
    throw new Error(`${label} must not carry a UTF-8 BOM`);
  }
  const text = new TextDecoder("utf-8", { fatal: true }).decode(raw);
  const value = parseJsonStrict(text, label);
  if (canonicalRequired && !raw.equals(canonicalBytes(value))) {
    throw new Error(`${label} is not canonical UTF-8/LF JSON`);
  }
  return { value, raw, rawSha256: sha256Bytes(raw) };
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

function sourceTextSha256(filePath) {
  const raw = fs.readFileSync(filePath);
  if (raw.subarray(0, 3).equals(Buffer.from([0xef, 0xbb, 0xbf]))) {
    throw new Error(`critical source carries UTF-8 BOM: ${filePath}`);
  }
  const text = new TextDecoder("utf-8", { fatal: true }).decode(raw);
  return sha256Bytes(Buffer.from(text.replace(/\r\n/g, "\n").replace(/\r/g, "\n"), "utf8"));
}

function resolveRepositoryPath(repositoryRoot, relativePosixPath) {
  const relative = requireRelativePosixPath(relativePosixPath, "repository relative path");
  const root = fs.realpathSync(repositoryRoot);
  const candidate = path.resolve(root, ...relative.split("/"));
  const relativeNative = path.relative(root, candidate);
  if (relativeNative.startsWith("..") || path.isAbsolute(relativeNative)) {
    throw new Error(`repository path escapes root: ${relative}`);
  }
  return candidate;
}

function validateProtocolPayload(protocol) {
  exactKeys(
    protocol,
    [
      "schema_version",
      "owner",
      "child",
      "scope",
      "host_qualification",
      "execution_authorization",
      "event_log",
      "child_process",
      "source_hash_mode",
      "source_sha256",
      "output_contract",
      "evidence_firewall",
      "claim_boundary",
    ],
    "host campaign protocol",
  );
  if (protocol.schema_version !== HOST_CAMPAIGN_PROTOCOL_SCHEMA_VERSION) {
    throw new Error("host campaign protocol schema drift");
  }
  const owner = requireObject(protocol.owner, "protocol.owner");
  exactKeys(
    owner,
    [
      "campaign_owner_wheel",
      "campaign_owner",
      "child_execution_owner_wheel",
      "mode",
      "distribution_scope",
    ],
    "protocol.owner",
  );
  if (
    owner.campaign_owner_wheel !== "vz-runtime" ||
    owner.campaign_owner !==
      "volvence_zero.offline_evidence.windows_cuda_strict_32k_host_campaign" ||
    owner.child_execution_owner_wheel !== "vz-substrate" ||
    owner.mode !== "offline_host_campaign_control" ||
    owner.distribution_scope !== "repository-source-checkout-only"
  ) {
    throw new Error("host campaign owner contract drift");
  }
  const child = requireObject(protocol.child, "protocol.child");
  exactKeys(
    child,
    [
      "protocol_id",
      "protocol_raw_sha256",
      "protocol_relative_path",
      "runner_relative_path",
      "output_relative_path",
      "expected_complete_files",
      "success_exit_code",
      "diagnostic_failure_exit_code",
    ],
    "protocol.child",
  );
  requireSha256(child.protocol_id, "protocol.child.protocol_id");
  requireSha256(child.protocol_raw_sha256, "protocol.child.protocol_raw_sha256");
  requireRelativePosixPath(child.protocol_relative_path, "protocol.child.protocol_relative_path");
  requireRelativePosixPath(child.runner_relative_path, "protocol.child.runner_relative_path");
  requireRelativePosixPath(child.output_relative_path, "protocol.child.output_relative_path");
  if (
    child.protocol_id !== PINNED_CHILD_PROTOCOL_ID ||
    child.protocol_raw_sha256 !== PINNED_CHILD_PROTOCOL_RAW_SHA256 ||
    child.protocol_relative_path !== PINNED_CHILD_PROTOCOL_RELATIVE_PATH ||
    child.runner_relative_path !== PINNED_CHILD_RUNNER_RELATIVE_PATH ||
    child.output_relative_path !== PINNED_CHILD_OUTPUT_RELATIVE_PATH
  ) {
    throw new Error("host campaign pinned child lineage drift");
  }
  deepExact(child.expected_complete_files, CHILD_REQUIRED_FILES, "protocol.child.expected_complete_files");
  if (requireInteger(child.success_exit_code, "protocol.child.success_exit_code") !== 0) {
    throw new Error("child success exit code drift");
  }
  if (requireInteger(child.diagnostic_failure_exit_code, "protocol.child.diagnostic_failure_exit_code") !== 2) {
    throw new Error("child diagnostic failure exit code drift");
  }
  const scope = requireObject(protocol.scope, "protocol.scope");
  exactKeys(
    scope,
    [
      "scope_id_method",
      "scope_id_domain_separator",
      "scope_id_components",
      "campaign_root_relative_template",
      "attempt_budget",
      "retry_budget",
      "retry_scope",
      "lease_id_method",
      "preregistration_requires_host_qualification_pass",
    ],
    "protocol.scope",
  );
  if (
    scope.scope_id_method !== "sha256_domain_separated_length_framed_v1" ||
    scope.scope_id_domain_separator !== "volvence.windows-cuda-strict-32k-host-scope.v1" ||
    scope.campaign_root_relative_template !==
      "artifacts/relationship_lab/windows_cuda_strict_32k_host_campaigns/{scope_id}" ||
    requireInteger(scope.attempt_budget, "protocol.scope.attempt_budget") !== 1 ||
    requireInteger(scope.retry_budget, "protocol.scope.retry_budget") !== 0 ||
    scope.retry_scope !==
      "outer_protocol+child_protocol+host_qualification_artifact+host_identity+execution_backend" ||
    scope.lease_id_method !== "raw_sha256_of_002_preregistration_json" ||
    scope.preregistration_requires_host_qualification_pass !== true
  ) {
    throw new Error("host campaign scope contract drift");
  }
  deepExact(
    scope.scope_id_components,
    [
      "outer_protocol_id",
      "child_protocol_id",
      "host_qualification_artifact_id",
      "host_identity_sha256",
      "execution_backend_id",
    ],
    "protocol.scope.scope_id_components",
  );
  const qualification = requireObject(protocol.host_qualification, "protocol.host_qualification");
  exactKeys(
    qualification,
    [
      "terminal_schema_version",
      "passed_required",
      "real_cuda_evidence_authorized_required",
      "same_machine_required",
      "same_boot_required",
      "full_artifact_revalidated_by_campaign",
      "production_preregistration_enabled",
      "pinned_qualification_protocol_id",
      "qualification_validator_relative_path",
    ],
    "protocol.host_qualification",
  );
  if (
    qualification.terminal_schema_version !== HOST_QUALIFICATION_TERMINAL_SCHEMA_VERSION ||
    qualification.passed_required !== true ||
    qualification.real_cuda_evidence_authorized_required !== true ||
    qualification.same_machine_required !== true ||
    qualification.same_boot_required !== true ||
    qualification.full_artifact_revalidated_by_campaign !== false ||
    qualification.production_preregistration_enabled !== false ||
    qualification.pinned_qualification_protocol_id !== null ||
    qualification.qualification_validator_relative_path !== null
  ) {
    throw new Error("host qualification input contract drift");
  }
  const executionAuthorization = requireObject(
    protocol.execution_authorization,
    "protocol.execution_authorization",
  );
  deepExact(
    executionAuthorization,
    {
      production_backend_id: PRODUCTION_BACKEND_ID,
      synthetic_test_backend_id: SYNTHETIC_TEST_BACKEND_ID,
      production_requires_bundled_event_log_collector: true,
      production_requires_bundled_child_executor: true,
      production_requires_bundled_child_validator: true,
      production_requires_source_verification: true,
      synthetic_artifacts_real_execution_authorized: false,
      synthetic_artifacts_can_pass: false,
      public_validator_accepts_synthetic_artifacts: false,
    },
    "protocol.execution_authorization",
  );
  const eventLog = requireObject(protocol.event_log, "protocol.event_log");
  exactKeys(
    eventLog,
    [
      "collector_relative_path",
      "platform_system",
      "powershell_mode",
      "collector_timeout_seconds",
      "application_anchor_source",
      "anchor_event_ids",
      "channels",
      "max_new_records_per_channel",
      "baseline_boundary_xml_hash_required",
      "prelaunch_same_machine_boot_and_boundary_required",
      "same_boot_required",
      "fault_classification",
    ],
    "protocol.event_log",
  );
  if (
    eventLog.collector_relative_path !==
      "packages/vz-runtime/src/volvence_zero/offline_evidence/collect_windows_host_event_log.ps1" ||
    eventLog.platform_system !== "Windows" ||
    eventLog.powershell_mode !== "WindowsPowerShell-5.1-no-profile-noninteractive" ||
    requireInteger(eventLog.collector_timeout_seconds, "Event Log collector timeout") !== 120 ||
    eventLog.application_anchor_source !== "VolvenceEvidence" ||
    requireInteger(eventLog.max_new_records_per_channel, "max_new_records_per_channel") !== 4096 ||
    eventLog.baseline_boundary_xml_hash_required !== true ||
    eventLog.prelaunch_same_machine_boot_and_boundary_required !== true ||
    eventLog.same_boot_required !== true
  ) {
    throw new Error("host campaign Event Log contract drift");
  }
  deepExact(eventLog.anchor_event_ids, { preregistration: 8201, launch: 8202, terminal: 8203 }, "protocol.event_log.anchor_event_ids");
  deepExact(eventLog.channels, ["Application", "System"], "protocol.event_log.channels");
  const rules = requireArray(eventLog.fault_classification, "protocol.event_log.fault_classification");
  deepExact(rules, PINNED_FAULT_CLASSIFICATION, "protocol.event_log.fault_classification");
  if (rules.length !== 9) throw new Error("host campaign fault-classification rule count drift");
  for (const [index, rule] of rules.entries()) {
    exactKeys(
      rule,
      ["failure_code", "log_name", "provider_name", "event_ids"],
      `protocol.event_log.fault_classification[${index}]`,
    );
    if (!FAILURE_CODE_ORDER.includes(rule.failure_code)) {
      throw new Error("host campaign fault classification uses an unknown failure code");
    }
    if (!new Set(["Application", "System"]).has(rule.log_name)) {
      throw new Error("host campaign fault classification log drift");
    }
    requireText(rule.provider_name, "host campaign fault classification provider");
    if (rule.event_ids !== null) {
      const ids = requireArray(rule.event_ids, "host campaign fault classification event_ids");
      if (ids.length === 0) throw new Error("fault classification event ID list must be nonempty");
      ids.forEach((value) => requireInteger(value, "fault classification event ID"));
    }
  }
  const childProcessContract = requireObject(protocol.child_process, "protocol.child_process");
  exactKeys(
    childProcessContract,
    [
      "timeout_seconds",
      "shell",
      "argv_template",
      "environment_overrides",
      "environment_removed",
      "stdout_relative_path",
      "stderr_relative_path",
    ],
    "protocol.child_process",
  );
  if (
    requireInteger(childProcessContract.timeout_seconds, "child process timeout") !== 3600 ||
    childProcessContract.shell !== false ||
    childProcessContract.stdout_relative_path !== "streams/child.stdout.log" ||
    childProcessContract.stderr_relative_path !== "streams/child.stderr.log"
  ) {
    throw new Error("host campaign child-process contract drift");
  }
  deepExact(
    childProcessContract.argv_template,
    [
      "{python_executable}",
      "-I",
      "scripts/run_windows_cuda_strict_32k_smoke.py",
      "run",
      "--output-dir",
      "{child_output}",
      "--outer-attempt-lease-id",
      "{lease_id}",
      "--protocol",
      "packages/vz-runtime/src/volvence_zero/offline_evidence/protocols/windows_cuda_strict_32k_smoke_v1.json",
    ],
    "protocol.child_process.argv_template",
  );
  deepExact(
    childProcessContract.environment_overrides,
    {
      CUDA_VISIBLE_DEVICES: "0",
      HF_HUB_OFFLINE: "1",
      PYTHONNOUSERSITE: "1",
      TOKENIZERS_PARALLELISM: "false",
      TRANSFORMERS_OFFLINE: "1",
    },
    "protocol.child_process.environment_overrides",
  );
  deepExact(
    childProcessContract.environment_removed,
    CHILD_ENVIRONMENT_REMOVED,
    "protocol.child_process.environment_removed",
  );
  const sourceHashes = requireObject(protocol.source_sha256, "protocol.source_sha256");
  if (
    Object.keys(sourceHashes).length !== CAMPAIGN_CRITICAL_SOURCE_PATHS.length ||
    Object.keys(sourceHashes).some((sourcePath, index) => sourcePath !== CAMPAIGN_CRITICAL_SOURCE_PATHS[index])
  ) {
    throw new Error("campaign critical source set/order drift");
  }
  for (const [relative, digest] of Object.entries(sourceHashes)) {
    requireRelativePosixPath(relative, "protocol critical source path");
    requireSha256(digest, `protocol.source_sha256.${relative}`);
  }
  if (protocol.source_hash_mode !== "utf8_lf_canonical_v1") {
    throw new Error("campaign source hash mode drift");
  }
  const outputContract = requireObject(protocol.output_contract, "protocol.output_contract");
  deepExact(
    outputContract,
    {
      create_only: true,
      immutable_by_wrapper_policy: true,
      single_file_fsync: true,
      directory_entry_durability_guaranteed: false,
      launch_fsync_before_process_creation: true,
      receipt_chain_sequences: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
      preregistered_files: PREREGISTERED_FILES,
      complete_outer_files: [
        RECEIPT_FILES.scopeClaim,
        RECEIPT_FILES.eventLogBaseline,
        RECEIPT_FILES.preregistration,
        RECEIPT_FILES.preregistrationAnchor,
        RECEIPT_FILES.launch,
        RECEIPT_FILES.launchAnchor,
        RECEIPT_FILES.processExit,
        RECEIPT_FILES.eventLogDelta,
        RECEIPT_FILES.campaignReport,
        RECEIPT_FILES.manifest,
        RECEIPT_FILES.terminal,
        RECEIPT_FILES.terminalAnchor,
        RECEIPT_FILES.seal,
        "streams/child.stdout.log",
        "streams/child.stderr.log",
      ],
      child_output_relative_path: "child/strict_32k_smoke",
      validate_existing_starts_python: false,
      validate_existing_imports_torch_or_cuda: false,
      validate_existing_queries_live_event_log: false,
      incomplete_consumed_is_pass: false,
      failed_terminal_retry_permitted: false,
      privileged_local_tamper_resistance: false,
    },
    "protocol.output_contract",
  );
  deepExact(protocol.evidence_firewall, OUTER_EVIDENCE_FIREWALL, "protocol.evidence_firewall");
  if (protocol.claim_boundary !== OUTER_CLAIM_BOUNDARY) {
    throw new Error("campaign claim boundary drift");
  }
}

export function loadHostCampaignProtocol({
  protocolPath = DEFAULT_PROTOCOL_PATH,
  repositoryRoot = REPOSITORY_ROOT,
  verifySources = true,
} = {}) {
  const loaded = loadStrictJsonFile(path.resolve(protocolPath), "host campaign protocol");
  validateProtocolPayload(loaded.value);
  if (verifySources) {
    for (const [relative, expected] of Object.entries(loaded.value.source_sha256)) {
      const sourcePath = resolveRepositoryPath(repositoryRoot, relative);
      const stat = fs.lstatSync(sourcePath);
      if (!stat.isFile() || stat.isSymbolicLink()) {
        throw new Error(`critical campaign source is missing or linked: ${relative}`);
      }
      if (sourceTextSha256(sourcePath) !== expected) {
        throw new Error(`critical campaign source SHA-256 drift: ${relative}`);
      }
    }
  }
  return Object.freeze({
    payload: loaded.value,
    protocolId: sha256Bytes(canonicalBytes(loaded.value)),
    protocolRawSha256: loaded.rawSha256,
    protocolPath: path.resolve(protocolPath),
    repositoryRoot: path.resolve(repositoryRoot),
  });
}

export function loadHostQualificationTerminal(terminalPath) {
  const loaded = loadStrictJsonFile(
    path.resolve(terminalPath),
    "host qualification terminal",
    true,
  );
  const terminal = loaded.value;
  exactKeys(
    terminal,
    [
      "schema_version",
      "qualification_protocol_id",
      "artifact_id",
      "manifest_sha256",
      "host_identity_sha256",
      "boot_identity_sha256",
      "passed",
      "real_cuda_evidence_authorized",
      "completed_at_utc",
      "terminal_id",
    ],
    "host qualification terminal",
  );
  if (terminal.schema_version !== HOST_QUALIFICATION_TERMINAL_SCHEMA_VERSION) {
    throw new Error("host qualification terminal schema drift");
  }
  for (const key of [
    "qualification_protocol_id",
    "artifact_id",
    "manifest_sha256",
    "host_identity_sha256",
    "boot_identity_sha256",
    "terminal_id",
  ]) {
    requireSha256(terminal[key], `host qualification terminal.${key}`);
  }
  if (terminal.passed !== true || terminal.real_cuda_evidence_authorized !== true) {
    throw new Error("host qualification terminal does not authorize a CUDA evidence attempt");
  }
  requireUtcTimestamp(terminal.completed_at_utc, "host qualification terminal.completed_at_utc");
  const core = { ...terminal };
  delete core.terminal_id;
  if (terminal.terminal_id !== sha256Bytes(canonicalBytes(core))) {
    throw new Error("host qualification terminal ID drift");
  }
  return Object.freeze({
    payload: terminal,
    rawSha256: loaded.rawSha256,
    path: path.resolve(terminalPath),
  });
}

function loadChildProtocol(outerProtocol) {
  const childFacts = outerProtocol.payload.child;
  const childPath = resolveRepositoryPath(
    outerProtocol.repositoryRoot,
    childFacts.protocol_relative_path,
  );
  const loaded = loadStrictJsonFile(childPath, "strict 32K child protocol");
  const protocolId = sha256Bytes(canonicalBytes(loaded.value));
  if (
    protocolId !== childFacts.protocol_id ||
    loaded.rawSha256 !== childFacts.protocol_raw_sha256
  ) {
    throw new Error("strict 32K child protocol lineage drift");
  }
  return Object.freeze({
    payload: loaded.value,
    protocolId,
    protocolRawSha256: loaded.rawSha256,
    protocolPath: childPath,
  });
}

function normalizedAbsolutePath(inputPath) {
  return path.resolve(inputPath).replace(/\\/g, "/");
}

function computeScopeId(outerProtocol, childProtocol, qualification, executionBackendId) {
  const scope = outerProtocol.payload.scope;
  if (![PRODUCTION_BACKEND_ID, SYNTHETIC_TEST_BACKEND_ID].includes(executionBackendId)) {
    throw new Error("host campaign scope requires one frozen execution backend ID");
  }
  return domainSeparatedSha256(scope.scope_id_domain_separator, [
    outerProtocol.protocolId,
    childProtocol.protocolId,
    qualification.payload.artifact_id,
    qualification.payload.host_identity_sha256,
    executionBackendId,
  ]);
}

function nowUtc() {
  return new Date().toISOString();
}

function chainBase({ schemaVersion, sequence, scopeId, previousReceiptSha256 }) {
  return {
    schema_version: schemaVersion,
    sequence,
    scope_id: scopeId,
    previous_receipt_sha256: previousReceiptSha256,
  };
}

function validateReceiptChainBase(receipt, {
  schemaVersion,
  sequence,
  scopeId,
  previousReceiptSha256,
  label,
}) {
  if (receipt.schema_version !== schemaVersion) throw new Error(`${label} schema drift`);
  if (requireInteger(receipt.sequence, `${label}.sequence`) !== sequence) {
    throw new Error(`${label} sequence drift`);
  }
  if (receipt.scope_id !== scopeId) throw new Error(`${label} scope drift`);
  if (previousReceiptSha256 === null) {
    if (receipt.previous_receipt_sha256 !== null) {
      throw new Error(`${label} must start the receipt chain`);
    }
  } else if (
    requireSha256(receipt.previous_receipt_sha256, `${label}.previous_receipt_sha256`) !==
    previousReceiptSha256
  ) {
    throw new Error(`${label} previous receipt drift`);
  }
}

function validateHostObservation(observation, label) {
  const value = requireObject(observation, label);
  exactKeys(
    value,
    [
      "platform_system",
      "machine_identity_sha256",
      "boot_identity_sha256",
      "last_boot_up_time_utc",
      "powershell_version",
      "os",
      "cpu",
      "bios",
      "baseboard",
      "microcode_registry_raw_le_hex",
      "gpu_adapters",
    ],
    label,
  );
  if (value.platform_system !== "Windows") throw new Error(`${label} is not Windows`);
  requireSha256(value.machine_identity_sha256, `${label}.machine_identity_sha256`);
  requireSha256(value.boot_identity_sha256, `${label}.boot_identity_sha256`);
  requireUtcTimestamp(value.last_boot_up_time_utc, `${label}.last_boot_up_time_utc`);
  const powershellVersion = requireText(
    value.powershell_version,
    `${label}.powershell_version`,
  );
  if (!/^5\.1(?:\.|$)/.test(powershellVersion)) {
    throw new Error(`${label}.powershell_version must be Windows PowerShell 5.1`);
  }
  for (const key of ["os", "cpu", "bios", "baseboard"]) {
    requireObject(value[key], `${label}.${key}`);
  }
  requireText(
    value.microcode_registry_raw_le_hex,
    `${label}.microcode_registry_raw_le_hex`,
  );
  for (const [index, adapter] of requireArray(value.gpu_adapters, `${label}.gpu_adapters`).entries()) {
    exactKeys(adapter, ["name", "driver_version"], `${label}.gpu_adapters[${index}]`);
    requireText(adapter.name, `${label}.gpu_adapters[${index}].name`);
    requireText(adapter.driver_version, `${label}.gpu_adapters[${index}].driver_version`);
  }
  return value;
}

function validateEventChannelCursor(cursor, label) {
  const value = requireObject(cursor, label);
  exactKeys(
    value,
    [
      "log_name",
      "enabled",
      "record_count",
      "oldest_record_id",
      "newest_record_id",
      "newest_record_xml_sha256",
      "maximum_size_bytes",
      "log_mode",
    ],
    label,
  );
  if (!new Set(["Application", "System"]).has(value.log_name)) {
    throw new Error(`${label}.log_name drift`);
  }
  if (requireBoolean(value.enabled, `${label}.enabled`) !== true) {
    throw new Error(`${label} must be enabled`);
  }
  const oldest = requireInteger(value.oldest_record_id, `${label}.oldest_record_id`);
  const newest = requireInteger(value.newest_record_id, `${label}.newest_record_id`);
  if (oldest <= 0 || newest < oldest) throw new Error(`${label} record cursor drift`);
  const recordCount = requireInteger(value.record_count, `${label}.record_count`);
  if (recordCount <= 0) {
    throw new Error(`${label}.record_count must be positive`);
  }
  const newestXmlSha256 = requireSha256(
    value.newest_record_xml_sha256,
    `${label}.newest_record_xml_sha256`,
  );
  const maximumSizeBytes = requireInteger(
    value.maximum_size_bytes,
    `${label}.maximum_size_bytes`,
  );
  if (maximumSizeBytes <= 0) {
    throw new Error(`${label}.maximum_size_bytes must be positive`);
  }
  const logMode = requireText(value.log_mode, `${label}.log_mode`);
  if (logMode !== "Circular") throw new Error(`${label}.log_mode must be Circular`);
  return {
    log_name: value.log_name,
    enabled: true,
    record_count: recordCount,
    oldest_record_id: oldest,
    newest_record_id: newest,
    newest_record_xml_sha256: newestXmlSha256,
    maximum_size_bytes: maximumSizeBytes,
    log_mode: logMode,
  };
}

function normalizeBaselineCollectorPayload(raw, protocol) {
  const payload = requireObject(raw, "Event Log baseline collector output");
  exactKeys(
    payload,
    ["schema_version", "collection_started_at_utc", "collection_completed_at_utc", "host", "channels"],
    "Event Log baseline collector output",
  );
  if (payload.schema_version !== "windows-host-event-log-baseline-collector.v1") {
    throw new Error("Event Log baseline collector schema drift");
  }
  const started = requireUtcTimestamp(
    payload.collection_started_at_utc,
    "Event Log baseline.collection_started_at_utc",
  );
  const completed = requireUtcTimestamp(
    payload.collection_completed_at_utc,
    "Event Log baseline.collection_completed_at_utc",
  );
  if (completed < started) throw new Error("Event Log baseline completion predates start");
  const host = validateHostObservation(payload.host, "Event Log baseline.host");
  const channels = requireArray(payload.channels, "Event Log baseline.channels");
  if (channels.length !== 2) throw new Error("Event Log baseline channel count drift");
  channels.forEach((channel, index) =>
    validateEventChannelCursor(channel, `Event Log baseline.channels[${index}]`),
  );
  if (channels[0].log_name !== "Application" || channels[1].log_name !== "System") {
    throw new Error("Event Log baseline channel order drift");
  }
  deepExact(
    protocol.payload.event_log.channels,
    ["Application", "System"],
    "protocol.event_log.channels",
  );
  return {
    collection_started_at_utc: payload.collection_started_at_utc,
    collection_completed_at_utc: payload.collection_completed_at_utc,
    host,
    channels,
  };
}

function normalizePrelaunchCollectorPayload(raw, { protocol, baseline, qualification }) {
  const payload = requireObject(raw, "Event Log prelaunch collector output");
  exactKeys(
    payload,
    ["schema_version", "collection_started_at_utc", "collection_completed_at_utc", "host", "channels"],
    "Event Log prelaunch collector output",
  );
  if (payload.schema_version !== "windows-host-event-log-prelaunch-collector.v1") {
    throw new Error("Event Log prelaunch collector schema drift");
  }
  const started = requireUtcTimestamp(
    payload.collection_started_at_utc,
    "Event Log prelaunch.collection_started_at_utc",
  );
  const completed = requireUtcTimestamp(
    payload.collection_completed_at_utc,
    "Event Log prelaunch.collection_completed_at_utc",
  );
  if (completed < started) throw new Error("Event Log prelaunch completion predates start");
  const host = validateHostObservation(payload.host, "Event Log prelaunch.host");
  if (
    host.machine_identity_sha256 !== qualification.payload.host_identity_sha256 ||
    host.boot_identity_sha256 !== qualification.payload.boot_identity_sha256 ||
    host.machine_identity_sha256 !== baseline.host.machine_identity_sha256 ||
    host.boot_identity_sha256 !== baseline.host.boot_identity_sha256
  ) {
    throw new Error("prelaunch host is not the qualified baseline machine and boot");
  }
  const channels = requireArray(payload.channels, "Event Log prelaunch.channels");
  if (channels.length !== baseline.channels.length) {
    throw new Error("Event Log prelaunch channel count drift");
  }
  const normalizedChannels = channels.map((rawChannel, index) => {
    const channel = requireObject(rawChannel, `Event Log prelaunch.channels[${index}]`);
    exactKeys(
      channel,
      [
        "log_name",
        "baseline_newest_record_id",
        "baseline_boundary_present",
        "baseline_boundary_xml_sha256",
        "end_cursor",
        "new_record_count",
        "within_record_budget",
      ],
      `Event Log prelaunch.channels[${index}]`,
    );
    const baselineChannel = validateEventChannelCursor(
      baseline.channels[index],
      `Event Log prelaunch baseline.channels[${index}]`,
    );
    if (channel.log_name !== baselineChannel.log_name) {
      throw new Error("Event Log prelaunch channel order/name drift");
    }
    const baselineNewest = requireInteger(
      channel.baseline_newest_record_id,
      `Event Log prelaunch.channels[${index}].baseline_newest_record_id`,
    );
    if (baselineNewest !== baselineChannel.newest_record_id) {
      throw new Error("Event Log prelaunch baseline cursor drift");
    }
    if (requireBoolean(channel.baseline_boundary_present, "prelaunch boundary present") !== true) {
      throw new Error("Event Log prelaunch baseline boundary is missing");
    }
    if (
      requireSha256(channel.baseline_boundary_xml_sha256, "prelaunch boundary hash") !==
      baselineChannel.newest_record_xml_sha256
    ) {
      throw new Error("Event Log prelaunch baseline boundary hash drift");
    }
    const endCursor = validateEventChannelCursor(
      channel.end_cursor,
      `Event Log prelaunch.channels[${index}].end_cursor`,
    );
    if (
      endCursor.log_name !== baselineChannel.log_name ||
      endCursor.log_mode !== baselineChannel.log_mode ||
      endCursor.maximum_size_bytes !== baselineChannel.maximum_size_bytes
    ) {
      throw new Error("Event Log prelaunch channel configuration drift");
    }
    const difference = endCursor.newest_record_id - baselineNewest;
    if (
      difference < 0 ||
      requireInteger(channel.new_record_count, "prelaunch new_record_count") !== difference ||
      difference >
        requireInteger(
          protocol.payload.event_log.max_new_records_per_channel,
          "protocol max new Event Log records",
        ) ||
      requireBoolean(channel.within_record_budget, "prelaunch within_record_budget") !== true
    ) {
      throw new Error("Event Log prelaunch cursor budget is not safe for one attempt");
    }
    return {
      log_name: channel.log_name,
      baseline_newest_record_id: baselineNewest,
      baseline_boundary_present: true,
      baseline_boundary_xml_sha256: channel.baseline_boundary_xml_sha256,
      end_cursor: endCursor,
      new_record_count: difference,
      within_record_budget: true,
    };
  });
  return {
    schema_version: payload.schema_version,
    collection_started_at_utc: payload.collection_started_at_utc,
    collection_completed_at_utc: payload.collection_completed_at_utc,
    host,
    channels: normalizedChannels,
  };
}

function anchorMessage({ protocol, scopeId, kind, sequence, receiptSha256, leaseId }) {
  return {
    schema_version: "volvence-local-event-anchor.v1",
    outer_protocol_id: protocol.protocolId,
    scope_id: scopeId,
    anchor_kind: kind,
    sequence,
    receipt_sha256: receiptSha256,
    lease_id: leaseId,
  };
}

function normalizeAnchorObservation(raw, expectedMessage, protocol) {
  const observation = requireObject(raw, "local Event Log anchor observation");
  exactKeys(
    observation,
    [
      "schema_version",
      "log_name",
      "provider_name",
      "event_id",
      "record_id",
      "time_created_utc",
      "xml_sha256",
      "payload_base64",
    ],
    "local Event Log anchor observation",
  );
  if (observation.schema_version !== "volvence-local-event-anchor-observation.v1") {
    throw new Error("local Event Log anchor observation schema drift");
  }
  if (
    observation.log_name !== "Application" ||
    observation.provider_name !== protocol.payload.event_log.application_anchor_source
  ) {
    throw new Error("local Event Log anchor provider drift");
  }
  const expectedEventId = requireInteger(
    protocol.payload.event_log.anchor_event_ids[expectedMessage.anchor_kind],
    "expected anchor event ID",
  );
  if (requireInteger(observation.event_id, "anchor.event_id") !== expectedEventId) {
    throw new Error("local Event Log anchor event ID drift");
  }
  if (requireInteger(observation.record_id, "anchor.record_id") <= 0) {
    throw new Error("local Event Log anchor record ID must be positive");
  }
  requireUtcTimestamp(observation.time_created_utc, "anchor.time_created_utc");
  const xmlSha256 = requireSha256(observation.xml_sha256, "anchor.xml_sha256");
  const decoded = Buffer.from(requireText(observation.payload_base64, "anchor.payload_base64"), "base64");
  const message = parseJsonStrict(new TextDecoder("utf-8", { fatal: true }).decode(decoded), "anchor payload");
  deepExact(message, expectedMessage, "local Event Log anchor payload");
  if (!decoded.equals(canonicalBytes(expectedMessage, false))) {
    throw new Error("local Event Log anchor payload is not canonical");
  }
  return {
    schema_version: observation.schema_version,
    log_name: observation.log_name,
    provider_name: observation.provider_name,
    event_id: requireInteger(observation.event_id, "anchor.event_id"),
    record_id: requireInteger(observation.record_id, "anchor.record_id"),
    time_created_utc: observation.time_created_utc,
    xml_sha256: xmlSha256,
    payload_base64: observation.payload_base64,
    payload_sha256: sha256Bytes(decoded),
  };
}

function validateStoredAnchorObservation(observation, expectedMessage, protocol) {
  exactKeys(
    observation,
    [
      "schema_version",
      "log_name",
      "provider_name",
      "event_id",
      "record_id",
      "time_created_utc",
      "xml_sha256",
      "payload_base64",
      "payload_sha256",
    ],
    "stored local Event Log anchor observation",
  );
  const normalized = normalizeAnchorObservation(
    {
      schema_version: observation.schema_version,
      log_name: observation.log_name,
      provider_name: observation.provider_name,
      event_id: observation.event_id,
      record_id: observation.record_id,
      time_created_utc: observation.time_created_utc,
      xml_sha256: observation.xml_sha256,
      payload_base64: observation.payload_base64,
    },
    expectedMessage,
    protocol,
  );
  deepExact(observation, normalized, "stored local Event Log anchor observation");
  return normalized;
}

function defaultPowerShellPath() {
  const systemRoot = process.env.SystemRoot || process.env.WINDIR;
  if (!systemRoot) throw new Error("SystemRoot/WINDIR is required for the frozen PowerShell collector");
  return path.join(systemRoot, "System32", "WindowsPowerShell", "v1.0", "powershell.exe");
}

function invokePowerShellCollector(protocol, mode, argumentsList = []) {
  const scriptPath = resolveRepositoryPath(
    protocol.repositoryRoot,
    protocol.payload.event_log.collector_relative_path,
  );
  const executable = defaultPowerShellPath();
  const result = childProcess.spawnSync(
    executable,
    [
      "-NoLogo",
      "-NoProfile",
      "-NonInteractive",
      "-ExecutionPolicy",
      "Bypass",
      "-File",
      scriptPath,
      "-Mode",
      mode,
      ...argumentsList,
    ],
    {
      encoding: "utf8",
      windowsHide: true,
      shell: false,
      maxBuffer: 64 * 1024 * 1024,
      timeout:
        requireInteger(
          protocol.payload.event_log.collector_timeout_seconds,
          "Event Log collector timeout",
        ) * 1000,
    },
  );
  if (result.error) throw new Error(`PowerShell Event Log collector failed to start: ${result.error.message}`);
  if (result.status !== 0) {
    throw new Error(
      `PowerShell Event Log collector ${mode} failed with ${result.status}: ${String(result.stderr).trim()}`,
    );
  }
  return parseJsonStrict(String(result.stdout).trim(), `PowerShell Event Log collector ${mode}`);
}

export const defaultWindowsEventLogCollector = Object.freeze({
  findAnchors({ protocol, scopeId }) {
    return invokePowerShellCollector(protocol, "FindAnchors", ["-ScopeId", scopeId]);
  },
  captureBaseline({ protocol }) {
    return invokePowerShellCollector(protocol, "Baseline");
  },
  capturePrelaunch({ protocol, baselinePath }) {
    return invokePowerShellCollector(protocol, "Prelaunch", [
      "-BaselinePath",
      path.resolve(baselinePath),
      "-MaxNewRecordsPerChannel",
      String(requireInteger(protocol.payload.event_log.max_new_records_per_channel, "max new records")),
    ]);
  },
  writeAnchor({ protocol, message }) {
    const payloadBase64 = canonicalBytes(message, false).toString("base64");
    return invokePowerShellCollector(protocol, "WriteAnchor", [
      "-AnchorPayloadBase64",
      payloadBase64,
    ]);
  },
  verifyAnchor({ protocol, message, recordId }) {
    const payloadBase64 = canonicalBytes(message, false).toString("base64");
    return invokePowerShellCollector(protocol, "VerifyAnchor", [
      "-RecordId",
      String(recordId),
      "-AnchorPayloadBase64",
      payloadBase64,
    ]);
  },
  captureDelta({ protocol, baselinePath }) {
    return invokePowerShellCollector(protocol, "Delta", [
      "-BaselinePath",
      path.resolve(baselinePath),
      "-MaxNewRecordsPerChannel",
      String(requireInteger(protocol.payload.event_log.max_new_records_per_channel, "max new records")),
    ]);
  },
});

function normalizeAnchorInventory(raw, { protocol, scopeId }) {
  const payload = requireObject(raw, "local Event Log anchor inventory");
  exactKeys(payload, ["schema_version", "scope_id", "anchors"], "local Event Log anchor inventory");
  if (payload.schema_version !== "volvence-local-event-anchor-inventory.v1") {
    throw new Error("local Event Log anchor inventory schema drift");
  }
  if (payload.scope_id !== scopeId) throw new Error("local Event Log anchor inventory scope drift");
  const anchors = requireArray(payload.anchors, "local Event Log anchor inventory.anchors");
  return anchors.map((rawAnchor, index) => {
    const anchor = requireObject(rawAnchor, `anchor inventory[${index}]`);
    exactKeys(
      anchor,
      [
        "log_name",
        "provider_name",
        "event_id",
        "record_id",
        "time_created_utc",
        "xml_sha256",
        "payload_base64",
      ],
      `anchor inventory[${index}]`,
    );
    if (
      anchor.log_name !== "Application" ||
      anchor.provider_name !== protocol.payload.event_log.application_anchor_source
    ) {
      throw new Error("anchor inventory provider drift");
    }
    requireInteger(anchor.event_id, `anchor inventory[${index}].event_id`);
    requireInteger(anchor.record_id, `anchor inventory[${index}].record_id`);
    requireUtcTimestamp(anchor.time_created_utc, `anchor inventory[${index}].time_created_utc`);
    requireSha256(anchor.xml_sha256, `anchor inventory[${index}].xml_sha256`);
    const decoded = Buffer.from(requireText(anchor.payload_base64, "anchor payload base64"), "base64");
    const message = parseJsonStrict(
      new TextDecoder("utf-8", { fatal: true }).decode(decoded),
      `anchor inventory[${index}] payload`,
    );
    exactKeys(
      message,
      [
        "schema_version",
        "outer_protocol_id",
        "scope_id",
        "anchor_kind",
        "sequence",
        "receipt_sha256",
        "lease_id",
      ],
      `anchor inventory[${index}] payload`,
    );
    if (
      message.schema_version !== "volvence-local-event-anchor.v1" ||
      message.outer_protocol_id !== protocol.protocolId ||
      message.scope_id !== scopeId
    ) {
      throw new Error("anchor inventory payload lineage drift");
    }
    if (!new Set(["preregistration", "launch", "terminal"]).has(message.anchor_kind)) {
      throw new Error("anchor inventory kind drift");
    }
    requireInteger(message.sequence, "anchor inventory payload.sequence");
    requireSha256(message.receipt_sha256, "anchor inventory payload.receipt_sha256");
    requireSha256(message.lease_id, "anchor inventory payload.lease_id");
    const expectedEventId = requireInteger(
      protocol.payload.event_log.anchor_event_ids[message.anchor_kind],
      "protocol anchor event ID",
    );
    if (requireInteger(anchor.event_id, "anchor inventory.event_id") !== expectedEventId) {
      throw new Error("anchor inventory event ID/kind drift");
    }
    return { ...anchor, message };
  });
}

function buildEventAnchorReceipt({
  protocol,
  scopeId,
  sequence,
  kind,
  previousReceiptSha256,
  leaseId,
  observation,
}) {
  const message = anchorMessage({
    protocol,
    scopeId,
    kind,
    sequence,
    receiptSha256: previousReceiptSha256,
    leaseId,
  });
  const normalized = normalizeAnchorObservation(observation, message, protocol);
  return {
    ...chainBase({
      schemaVersion: "windows-cuda-strict-32k-host-campaign-event-anchor.v1",
      sequence,
      scopeId,
      previousReceiptSha256,
    }),
    anchor_kind: kind,
    anchored_receipt_sha256: previousReceiptSha256,
    lease_id: leaseId,
    observation: normalized,
    local_event_log_is_independent_authority: false,
  };
}

function validateEventAnchorReceipt(receipt, {
  protocol,
  scopeId,
  sequence,
  kind,
  previousReceiptSha256,
  leaseId,
  label,
}) {
  exactKeys(
    receipt,
    [
      "schema_version",
      "sequence",
      "scope_id",
      "previous_receipt_sha256",
      "anchor_kind",
      "anchored_receipt_sha256",
      "lease_id",
      "observation",
      "local_event_log_is_independent_authority",
    ],
    label,
  );
  validateReceiptChainBase(receipt, {
    schemaVersion: "windows-cuda-strict-32k-host-campaign-event-anchor.v1",
    sequence,
    scopeId,
    previousReceiptSha256,
    label,
  });
  if (
    receipt.anchor_kind !== kind ||
    receipt.anchored_receipt_sha256 !== previousReceiptSha256 ||
    receipt.lease_id !== leaseId ||
    receipt.local_event_log_is_independent_authority !== false
  ) {
    throw new Error(`${label} lineage drift`);
  }
  const message = anchorMessage({
    protocol,
    scopeId,
    kind,
    sequence,
    receiptSha256: previousReceiptSha256,
    leaseId,
  });
  validateStoredAnchorObservation(receipt.observation, message, protocol);
  return message;
}

function normalizedEnvironmentOverrides(protocol) {
  const overrides = requireObject(
    protocol.payload.child_process.environment_overrides,
    "protocol.child_process.environment_overrides",
  );
  const result = Object.create(null);
  for (const key of Object.keys(overrides).sort()) {
    result[key] = requireText(overrides[key], `environment override ${key}`);
  }
  return result;
}

function isolatedChildEnvironment(protocol) {
  const result = { ...process.env };
  const overrides = normalizedEnvironmentOverrides(protocol);
  const removed = new Set(
    [
      ...protocol.payload.child_process.environment_removed,
      ...Object.keys(overrides),
    ].map((name) => name.toUpperCase()),
  );
  for (const key of Object.keys(result)) {
    if (removed.has(key.toUpperCase())) delete result[key];
  }
  return { ...result, ...overrides };
}

function inspectExecutable(executablePath) {
  const absolute = path.resolve(executablePath);
  const stat = fs.lstatSync(absolute);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1) {
    throw new Error("child Python executable must be one regular, non-linked file");
  }
  const raw = fs.readFileSync(absolute);
  return {
    absolute_path: normalizedAbsolutePath(absolute),
    byte_count: raw.length,
    sha256: sha256Bytes(raw),
  };
}

function buildRealizedChildArgv({ protocol, preregistration, leaseId, campaignRoot }) {
  const executable = preregistration.python_executable.absolute_path;
  const runnerPath = normalizedAbsolutePath(
    resolveRepositoryPath(protocol.repositoryRoot, protocol.payload.child.runner_relative_path),
  );
  const childProtocolPath = normalizedAbsolutePath(
    resolveRepositoryPath(protocol.repositoryRoot, protocol.payload.child.protocol_relative_path),
  );
  const childOutput = normalizedAbsolutePath(
    path.join(campaignRoot, ...protocol.payload.child.output_relative_path.split("/")),
  );
  return [
    executable,
    "-I",
    runnerPath,
    "run",
    "--output-dir",
    childOutput,
    "--outer-attempt-lease-id",
    leaseId,
    "--protocol",
    childProtocolPath,
  ];
}

function validateFixedChildArgv(argv, { protocol, preregistration, leaseId, campaignRoot, label }) {
  const expected = buildRealizedChildArgv({ protocol, preregistration, leaseId, campaignRoot });
  deepExact(argv, expected, label);
  return expected;
}

function campaignRootForScope(campaignBaseDir, scopeId) {
  return path.resolve(campaignBaseDir, scopeId);
}

function ensureEmptyCreateOnlyRoot(campaignRoot) {
  if (fs.existsSync(campaignRoot)) {
    throw new Error(`host campaign root is create-only and already exists: ${campaignRoot}`);
  }
  fs.mkdirSync(path.dirname(campaignRoot), { recursive: true });
  fs.mkdirSync(campaignRoot, { recursive: false });
}

function ensureExactTopLevelEntries(campaignRoot, expected, label) {
  const stat = fs.lstatSync(campaignRoot);
  if (!stat.isDirectory() || stat.isSymbolicLink()) {
    throw new Error(`${label} root must be a regular directory`);
  }
  const actual = fs.readdirSync(campaignRoot).sort();
  const wanted = [...expected].sort();
  if (actual.length !== wanted.length || actual.some((name, index) => name !== wanted[index])) {
    throw new Error(`${label} root entry set drift`);
  }
}

function validateScopeClaim(
  receipt,
  {
    protocol,
    childProtocol,
    qualification,
    campaignRoot,
    scopeId,
    allowSyntheticTestBackend = false,
  },
) {
  exactKeys(
    receipt,
    [
      "schema_version",
      "sequence",
      "scope_id",
      "previous_receipt_sha256",
      "scope_id_method",
      "outer_protocol_id",
      "outer_protocol_raw_sha256",
      "child_protocol_id",
      "child_protocol_raw_sha256",
      "host_qualification",
      "host_qualification_terminal_raw_sha256",
      "host_identity_sha256",
      "campaign_root",
      "child_output_relative_path",
      "attempt_budget",
      "retry_budget",
      "execution_backend_id",
      "real_execution_observation_authorized",
      "claimed_at_utc",
      "evidence_firewall",
      "claim_boundary",
    ],
    "scope claim",
  );
  validateReceiptChainBase(receipt, {
    schemaVersion: "windows-cuda-strict-32k-host-campaign-scope-claim.v1",
    sequence: 0,
    scopeId,
    previousReceiptSha256: null,
    label: "scope claim",
  });
  if (
    receipt.scope_id_method !== protocol.payload.scope.scope_id_method ||
    receipt.outer_protocol_id !== protocol.protocolId ||
    receipt.outer_protocol_raw_sha256 !== protocol.protocolRawSha256 ||
    receipt.child_protocol_id !== childProtocol.protocolId ||
    receipt.child_protocol_raw_sha256 !== childProtocol.protocolRawSha256 ||
    receipt.host_qualification_terminal_raw_sha256 !== qualification.rawSha256 ||
    receipt.host_identity_sha256 !== qualification.payload.host_identity_sha256 ||
    receipt.campaign_root !== normalizedAbsolutePath(campaignRoot) ||
    receipt.child_output_relative_path !== protocol.payload.child.output_relative_path ||
    requireInteger(receipt.attempt_budget, "scope claim.attempt_budget") !== 1 ||
    requireInteger(receipt.retry_budget, "scope claim.retry_budget") !== 0
  ) {
    throw new Error("scope claim lineage drift");
  }
  const productionBackend =
    receipt.execution_backend_id === PRODUCTION_BACKEND_ID &&
    receipt.real_execution_observation_authorized === true;
  const syntheticTestBackend =
    receipt.execution_backend_id === SYNTHETIC_TEST_BACKEND_ID &&
    receipt.real_execution_observation_authorized === false;
  if (!productionBackend && !(allowSyntheticTestBackend && syntheticTestBackend)) {
    throw new Error("scope claim execution backend is not authorized by this validator");
  }
  deepExact(receipt.host_qualification, qualification.payload, "scope claim.host_qualification");
  requireUtcTimestamp(receipt.claimed_at_utc, "scope claim.claimed_at_utc");
  deepExact(receipt.evidence_firewall, OUTER_EVIDENCE_FIREWALL, "scope claim.evidence_firewall");
  if (receipt.claim_boundary !== OUTER_CLAIM_BOUNDARY) throw new Error("scope claim boundary drift");
}

function validateBaselineReceipt(receipt, {
  protocol,
  qualification,
  scopeId,
  previousReceiptSha256,
}) {
  exactKeys(
    receipt,
    [
      "schema_version",
      "sequence",
      "scope_id",
      "previous_receipt_sha256",
      "collection_started_at_utc",
      "collection_completed_at_utc",
      "host",
      "channels",
      "host_qualification_artifact_id",
      "same_qualified_machine",
      "same_qualified_boot",
    ],
    "Event Log baseline receipt",
  );
  validateReceiptChainBase(receipt, {
    schemaVersion: "windows-cuda-strict-32k-host-campaign-event-log-baseline.v1",
    sequence: 1,
    scopeId,
    previousReceiptSha256,
    label: "Event Log baseline receipt",
  });
  const baseline = normalizeBaselineCollectorPayload(
    {
      schema_version: "windows-host-event-log-baseline-collector.v1",
      collection_started_at_utc: receipt.collection_started_at_utc,
      collection_completed_at_utc: receipt.collection_completed_at_utc,
      host: receipt.host,
      channels: receipt.channels,
    },
    protocol,
  );
  if (
    receipt.host_qualification_artifact_id !== qualification.payload.artifact_id ||
    receipt.same_qualified_machine !== true ||
    receipt.same_qualified_boot !== true ||
    baseline.host.machine_identity_sha256 !== qualification.payload.host_identity_sha256 ||
    baseline.host.boot_identity_sha256 !== qualification.payload.boot_identity_sha256
  ) {
    throw new Error("Event Log baseline/qualification lineage drift");
  }
  if (
    requireUtcTimestamp(
      qualification.payload.completed_at_utc,
      "host qualification completed_at_utc",
    ) >
    requireUtcTimestamp(receipt.collection_started_at_utc, "baseline collection_started_at_utc")
  ) {
    throw new Error("Event Log baseline predates host qualification completion");
  }
}

function validatePreregistrationReceipt(receipt, {
  protocol,
  childProtocol,
  qualification,
  scopeId,
  campaignRoot,
  previousReceiptSha256,
}) {
  exactKeys(
    receipt,
    [
      "schema_version",
      "sequence",
      "scope_id",
      "previous_receipt_sha256",
      "outer_protocol_id",
      "outer_protocol_raw_sha256",
      "child_protocol_id",
      "child_protocol_raw_sha256",
      "host_qualification_artifact_id",
      "host_identity_sha256",
      "boot_identity_sha256",
      "campaign_root",
      "child_output_relative_path",
      "python_executable",
      "argv_template",
      "environment_overrides",
      "environment_removed",
      "attempt_budget",
      "retry_budget",
      "retry_scope",
      "timeout_seconds",
      "expected_child_complete_files",
      "success_exit_code",
      "diagnostic_failure_exit_code",
      "preregistered_at_utc",
      "evidence_firewall",
      "claim_boundary",
    ],
    "campaign preregistration",
  );
  validateReceiptChainBase(receipt, {
    schemaVersion: "windows-cuda-strict-32k-host-campaign-preregistration.v1",
    sequence: 2,
    scopeId,
    previousReceiptSha256,
    label: "campaign preregistration",
  });
  if (
    receipt.outer_protocol_id !== protocol.protocolId ||
    receipt.outer_protocol_raw_sha256 !== protocol.protocolRawSha256 ||
    receipt.child_protocol_id !== childProtocol.protocolId ||
    receipt.child_protocol_raw_sha256 !== childProtocol.protocolRawSha256 ||
    receipt.host_qualification_artifact_id !== qualification.payload.artifact_id ||
    receipt.host_identity_sha256 !== qualification.payload.host_identity_sha256 ||
    receipt.boot_identity_sha256 !== qualification.payload.boot_identity_sha256 ||
    receipt.campaign_root !== normalizedAbsolutePath(campaignRoot) ||
    receipt.child_output_relative_path !== protocol.payload.child.output_relative_path ||
    requireInteger(receipt.attempt_budget, "campaign preregistration.attempt_budget") !== 1 ||
    requireInteger(receipt.retry_budget, "campaign preregistration.retry_budget") !== 0 ||
    receipt.retry_scope !== protocol.payload.scope.retry_scope ||
    requireInteger(receipt.timeout_seconds, "campaign preregistration.timeout_seconds") !==
      requireInteger(protocol.payload.child_process.timeout_seconds, "protocol timeout") ||
    requireInteger(receipt.success_exit_code, "campaign preregistration.success_exit_code") !== 0 ||
    requireInteger(
      receipt.diagnostic_failure_exit_code,
      "campaign preregistration.diagnostic_failure_exit_code",
    ) !== 2
  ) {
    throw new Error("campaign preregistration lineage drift");
  }
  const executable = requireObject(receipt.python_executable, "campaign preregistration.python_executable");
  exactKeys(executable, ["absolute_path", "byte_count", "sha256"], "campaign preregistration.python_executable");
  requireText(executable.absolute_path, "campaign preregistration.python_executable.absolute_path");
  if (requireInteger(executable.byte_count, "python executable byte_count") <= 0) {
    throw new Error("python executable byte count must be positive");
  }
  requireSha256(executable.sha256, "campaign preregistration.python_executable.sha256");
  deepExact(
    receipt.argv_template,
    protocol.payload.child_process.argv_template,
    "campaign preregistration.argv_template",
  );
  deepExact(
    receipt.environment_overrides,
    normalizedEnvironmentOverrides(protocol),
    "campaign preregistration.environment_overrides",
  );
  deepExact(
    receipt.environment_removed,
    protocol.payload.child_process.environment_removed,
    "campaign preregistration.environment_removed",
  );
  deepExact(
    receipt.expected_child_complete_files,
    CHILD_REQUIRED_FILES,
    "campaign preregistration.expected_child_complete_files",
  );
  requireUtcTimestamp(receipt.preregistered_at_utc, "campaign preregistration.preregistered_at_utc");
  deepExact(
    receipt.evidence_firewall,
    OUTER_EVIDENCE_FIREWALL,
    "campaign preregistration.evidence_firewall",
  );
  if (receipt.claim_boundary !== OUTER_CLAIM_BOUNDARY) {
    throw new Error("campaign preregistration claim boundary drift");
  }
}

function loadPreregisteredCampaign({
  campaignRoot,
  protocolPath = DEFAULT_PROTOCOL_PATH,
  repositoryRoot = REPOSITORY_ROOT,
  verifySources = true,
  requireExactTopLevel = true,
  allowSyntheticTestBackend = false,
}) {
  const root = path.resolve(campaignRoot);
  if (requireExactTopLevel) {
    ensureExactTopLevelEntries(root, PREREGISTERED_FILES, "preregistered campaign");
  } else {
    const stat = fs.lstatSync(root);
    if (!stat.isDirectory() || stat.isSymbolicLink()) {
      throw new Error("campaign root must be a regular directory");
    }
    for (const required of PREREGISTERED_FILES) {
      if (!fs.existsSync(path.join(root, required))) {
        throw new Error(`campaign preregistration prefix is incomplete: ${required}`);
      }
    }
  }
  const protocol = loadHostCampaignProtocol({ protocolPath, repositoryRoot, verifySources });
  const childProtocol = loadChildProtocol(protocol);
  const scopeLoaded = loadStrictJsonFile(
    path.join(root, RECEIPT_FILES.scopeClaim),
    "scope claim",
    true,
  );
  const scopeId = requireSha256(scopeLoaded.value.scope_id, "scope claim.scope_id");
  if (path.basename(root).toLowerCase() !== scopeId) {
    throw new Error("campaign root basename must equal its deterministic scope ID");
  }
  const qualificationPayload = requireObject(
    scopeLoaded.value.host_qualification,
    "scope claim.host_qualification",
  );
  exactKeys(
    qualificationPayload,
    [
      "schema_version",
      "qualification_protocol_id",
      "artifact_id",
      "manifest_sha256",
      "host_identity_sha256",
      "boot_identity_sha256",
      "passed",
      "real_cuda_evidence_authorized",
      "completed_at_utc",
      "terminal_id",
    ],
    "scope claim.host_qualification",
  );
  const qualification = Object.freeze({
    payload: qualificationPayload,
    rawSha256: requireSha256(
      scopeLoaded.value.host_qualification_terminal_raw_sha256,
      "scope claim.host_qualification_terminal_raw_sha256",
    ),
  });
  const qualificationCore = { ...qualificationPayload };
  const embeddedTerminalId = qualificationCore.terminal_id;
  delete qualificationCore.terminal_id;
  if (
    qualificationPayload.schema_version !== HOST_QUALIFICATION_TERMINAL_SCHEMA_VERSION ||
    qualificationPayload.passed !== true ||
    qualificationPayload.real_cuda_evidence_authorized !== true ||
    sha256Bytes(canonicalBytes(qualificationPayload)) !== qualification.rawSha256 ||
    embeddedTerminalId !== sha256Bytes(canonicalBytes(qualificationCore))
  ) {
    throw new Error("embedded host qualification terminal drift");
  }
  for (const key of [
    "qualification_protocol_id",
    "artifact_id",
    "manifest_sha256",
    "host_identity_sha256",
    "boot_identity_sha256",
    "terminal_id",
  ]) {
    requireSha256(qualificationPayload[key], `scope claim.host_qualification.${key}`);
  }
  requireUtcTimestamp(
    qualificationPayload.completed_at_utc,
    "scope claim.host_qualification.completed_at_utc",
  );
  const expectedScopeId = computeScopeId(
    protocol,
    childProtocol,
    qualification,
    scopeLoaded.value.execution_backend_id,
  );
  if (scopeId !== expectedScopeId) throw new Error("campaign deterministic scope ID drift");
  validateScopeClaim(scopeLoaded.value, {
    protocol,
    childProtocol,
    qualification,
    campaignRoot: root,
    scopeId,
    allowSyntheticTestBackend,
  });
  const baselineLoaded = loadStrictJsonFile(
    path.join(root, RECEIPT_FILES.eventLogBaseline),
    "Event Log baseline receipt",
    true,
  );
  validateBaselineReceipt(baselineLoaded.value, {
    protocol,
    qualification,
    scopeId,
    previousReceiptSha256: scopeLoaded.rawSha256,
  });
  const preregLoaded = loadStrictJsonFile(
    path.join(root, RECEIPT_FILES.preregistration),
    "campaign preregistration",
    true,
  );
  validatePreregistrationReceipt(preregLoaded.value, {
    protocol,
    childProtocol,
    qualification,
    scopeId,
    campaignRoot: root,
    previousReceiptSha256: baselineLoaded.rawSha256,
  });
  const leaseId = preregLoaded.rawSha256;
  const anchorLoaded = loadStrictJsonFile(
    path.join(root, RECEIPT_FILES.preregistrationAnchor),
    "preregistration Event Log anchor",
    true,
  );
  const anchorMessageValue = validateEventAnchorReceipt(anchorLoaded.value, {
    protocol,
    scopeId,
    sequence: 3,
    kind: "preregistration",
    previousReceiptSha256: leaseId,
    leaseId,
    label: "preregistration Event Log anchor",
  });
  const qualificationTime = requireUtcTimestamp(
    qualification.payload.completed_at_utc,
    "host qualification completed_at_utc",
  );
  const claimedTime = requireUtcTimestamp(
    scopeLoaded.value.claimed_at_utc,
    "scope claim.claimed_at_utc",
  );
  const baselineStarted = requireUtcTimestamp(
    baselineLoaded.value.collection_started_at_utc,
    "baseline collection_started_at_utc",
  );
  const baselineCompleted = requireUtcTimestamp(
    baselineLoaded.value.collection_completed_at_utc,
    "baseline collection_completed_at_utc",
  );
  const preregisteredTime = requireUtcTimestamp(
    preregLoaded.value.preregistered_at_utc,
    "campaign preregistered_at_utc",
  );
  const preregAnchorTime = requireUtcTimestamp(
    anchorLoaded.value.observation.time_created_utc,
    "preregistration Event Log anchor time",
  );
  if (
    qualificationTime > claimedTime ||
    claimedTime > baselineStarted ||
    baselineStarted > baselineCompleted ||
    baselineCompleted > preregisteredTime ||
    preregisteredTime > preregAnchorTime
  ) {
    throw new Error("campaign preregistration time chain drift");
  }
  const applicationBaseline = baselineLoaded.value.channels[0];
  if (
    applicationBaseline.log_name !== "Application" ||
    requireInteger(
      anchorLoaded.value.observation.record_id,
      "preregistration anchor record ID",
    ) <= requireInteger(applicationBaseline.newest_record_id, "Application baseline record ID")
  ) {
    throw new Error("preregistration anchor does not follow the Application baseline cursor");
  }
  return Object.freeze({
    root,
    protocol,
    childProtocol,
    qualification,
    scopeId,
    leaseId,
    scopeLoaded,
    baselineLoaded,
    preregLoaded,
    anchorLoaded,
    anchorMessage: anchorMessageValue,
  });
}

function preregisterHostCampaignCore({
  hostQualificationTerminalPath,
  pythonExecutable,
  campaignBaseDir = DEFAULT_CAMPAIGN_BASE_DIR,
  protocolPath = DEFAULT_PROTOCOL_PATH,
  repositoryRoot = REPOSITORY_ROOT,
  eventLogCollector,
  verifySources = true,
  executionBackendId,
  realExecutionObservationAuthorized,
  allowUnvalidatedQualificationForTesting = false,
} = {}) {
  if (!hostQualificationTerminalPath) {
    throw new TypeError("hostQualificationTerminalPath is required");
  }
  if (!pythonExecutable) throw new TypeError("pythonExecutable is required");
  const protocol = loadHostCampaignProtocol({ protocolPath, repositoryRoot, verifySources });
  if (allowUnvalidatedQualificationForTesting) {
    if (
      executionBackendId !== SYNTHETIC_TEST_BACKEND_ID ||
      realExecutionObservationAuthorized !== false
    ) {
      throw new Error("test dependency injection must use the non-evidence synthetic backend");
    }
  } else {
    if (
      eventLogCollector !== defaultWindowsEventLogCollector ||
      executionBackendId !== PRODUCTION_BACKEND_ID ||
      realExecutionObservationAuthorized !== true ||
      verifySources !== true
    ) {
      throw new Error("production preregistration requires the bundled verified backend");
    }
    throw new Error(
      "production preregistration is disabled until a pinned host-qualification " +
        "protocol and full artifact validator close the qualification-to-baseline interval",
    );
  }
  const childProtocol = loadChildProtocol(protocol);
  const qualification = loadHostQualificationTerminal(hostQualificationTerminalPath);
  const scopeId = computeScopeId(protocol, childProtocol, qualification, executionBackendId);
  const campaignRoot = campaignRootForScope(campaignBaseDir, scopeId);
  if (fs.existsSync(campaignRoot)) {
    throw new Error(`deterministic host campaign scope already exists: ${campaignRoot}`);
  }
  const executable = inspectExecutable(pythonExecutable);
  ensureEmptyCreateOnlyRoot(campaignRoot);
  const scopeClaim = {
    ...chainBase({
      schemaVersion: "windows-cuda-strict-32k-host-campaign-scope-claim.v1",
      sequence: 0,
      scopeId,
      previousReceiptSha256: null,
    }),
    scope_id_method: protocol.payload.scope.scope_id_method,
    outer_protocol_id: protocol.protocolId,
    outer_protocol_raw_sha256: protocol.protocolRawSha256,
    child_protocol_id: childProtocol.protocolId,
    child_protocol_raw_sha256: childProtocol.protocolRawSha256,
    host_qualification: qualification.payload,
    host_qualification_terminal_raw_sha256: qualification.rawSha256,
    host_identity_sha256: qualification.payload.host_identity_sha256,
    campaign_root: normalizedAbsolutePath(campaignRoot),
    child_output_relative_path: protocol.payload.child.output_relative_path,
    attempt_budget: 1,
    retry_budget: 0,
    execution_backend_id: executionBackendId,
    real_execution_observation_authorized: realExecutionObservationAuthorized,
    claimed_at_utc: nowUtc(),
    evidence_firewall: OUTER_EVIDENCE_FIREWALL,
    claim_boundary: OUTER_CLAIM_BOUNDARY,
  };
  const scopeWritten = writeCreateJson(path.join(campaignRoot, RECEIPT_FILES.scopeClaim), scopeClaim);
  const existingAnchors = normalizeAnchorInventory(
    eventLogCollector.findAnchors({ protocol, scopeId }),
    { protocol, scopeId },
  );
  if (existingAnchors.length !== 0) {
    throw new Error("deterministic host campaign scope already has a local Event Log anchor");
  }
  const baseline = normalizeBaselineCollectorPayload(
    eventLogCollector.captureBaseline({ protocol }),
    protocol,
  );
  if (
    baseline.host.machine_identity_sha256 !== qualification.payload.host_identity_sha256 ||
    baseline.host.boot_identity_sha256 !== qualification.payload.boot_identity_sha256
  ) {
    throw new Error("host qualification terminal is not for the current machine and boot");
  }
  if (
    requireUtcTimestamp(qualification.payload.completed_at_utc, "qualification completed_at_utc") >
    requireUtcTimestamp(baseline.collection_started_at_utc, "baseline collection_started_at_utc")
  ) {
    throw new Error("host qualification terminal completion must precede campaign baseline");
  }
  const baselineReceipt = {
    ...chainBase({
      schemaVersion: "windows-cuda-strict-32k-host-campaign-event-log-baseline.v1",
      sequence: 1,
      scopeId,
      previousReceiptSha256: scopeWritten.rawSha256,
    }),
    collection_started_at_utc: baseline.collection_started_at_utc,
    collection_completed_at_utc: baseline.collection_completed_at_utc,
    host: baseline.host,
    channels: baseline.channels,
    host_qualification_artifact_id: qualification.payload.artifact_id,
    same_qualified_machine: true,
    same_qualified_boot: true,
  };
  const baselineWritten = writeCreateJson(
    path.join(campaignRoot, RECEIPT_FILES.eventLogBaseline),
    baselineReceipt,
  );
  const preregistration = {
    ...chainBase({
      schemaVersion: "windows-cuda-strict-32k-host-campaign-preregistration.v1",
      sequence: 2,
      scopeId,
      previousReceiptSha256: baselineWritten.rawSha256,
    }),
    outer_protocol_id: protocol.protocolId,
    outer_protocol_raw_sha256: protocol.protocolRawSha256,
    child_protocol_id: childProtocol.protocolId,
    child_protocol_raw_sha256: childProtocol.protocolRawSha256,
    host_qualification_artifact_id: qualification.payload.artifact_id,
    host_identity_sha256: qualification.payload.host_identity_sha256,
    boot_identity_sha256: qualification.payload.boot_identity_sha256,
    campaign_root: normalizedAbsolutePath(campaignRoot),
    child_output_relative_path: protocol.payload.child.output_relative_path,
    python_executable: executable,
    argv_template: protocol.payload.child_process.argv_template,
    environment_overrides: normalizedEnvironmentOverrides(protocol),
    environment_removed: protocol.payload.child_process.environment_removed,
    attempt_budget: 1,
    retry_budget: 0,
    retry_scope: protocol.payload.scope.retry_scope,
    timeout_seconds: requireInteger(
      protocol.payload.child_process.timeout_seconds,
      "protocol.child_process.timeout_seconds",
    ),
    expected_child_complete_files: CHILD_REQUIRED_FILES,
    success_exit_code: 0,
    diagnostic_failure_exit_code: 2,
    preregistered_at_utc: nowUtc(),
    evidence_firewall: OUTER_EVIDENCE_FIREWALL,
    claim_boundary: OUTER_CLAIM_BOUNDARY,
  };
  const preregWritten = writeCreateJson(
    path.join(campaignRoot, RECEIPT_FILES.preregistration),
    preregistration,
  );
  const leaseId = preregWritten.rawSha256;
  const message = anchorMessage({
    protocol,
    scopeId,
    kind: "preregistration",
    sequence: 3,
    receiptSha256: leaseId,
    leaseId,
  });
  const anchorObservation = eventLogCollector.writeAnchor({ protocol, message });
  const anchorReceipt = buildEventAnchorReceipt({
    protocol,
    scopeId,
    sequence: 3,
    kind: "preregistration",
    previousReceiptSha256: leaseId,
    leaseId,
    observation: anchorObservation,
  });
  const anchorWritten = writeCreateJson(
    path.join(campaignRoot, RECEIPT_FILES.preregistrationAnchor),
    anchorReceipt,
  );
  return Object.freeze({
    status: "preregistered",
    scopeId,
    leaseId,
    campaignRoot,
    protocolId: protocol.protocolId,
    childProtocolId: childProtocol.protocolId,
    executionBackendId,
    realExecutionObservationAuthorized,
    receiptChainHeadSha256: anchorWritten.rawSha256,
  });
}

const CHILD_CHECK_KEYS = Object.freeze([
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
]);

function verifyChildCriticalSources(childProtocol, repositoryRoot) {
  if (childProtocol.payload.source_hash_mode !== "utf8_lf_canonical_v1") {
    throw new Error("strict 32K child source hash mode drift");
  }
  const sourceHashes = requireObject(childProtocol.payload.source_sha256, "child protocol.source_sha256");
  for (const [relative, expected] of Object.entries(sourceHashes)) {
    requireSha256(expected, `child protocol.source_sha256.${relative}`);
    const sourcePath = resolveRepositoryPath(repositoryRoot, relative);
    const stat = fs.lstatSync(sourcePath);
    if (!stat.isFile() || stat.isSymbolicLink()) {
      throw new Error(`strict 32K child source is missing or linked: ${relative}`);
    }
    if (sourceTextSha256(sourcePath) !== expected) {
      throw new Error(`strict 32K child source SHA-256 drift: ${relative}`);
    }
  }
}

function validateChildProtocolLineage(payload, childProtocol, label, includeEvidence = true) {
  if (
    payload.protocol_id !== childProtocol.protocolId ||
    payload.protocol_raw_sha256 !== childProtocol.protocolRawSha256 ||
    payload.source_hash_mode !== childProtocol.payload.source_hash_mode
  ) {
    throw new Error(`${label} child protocol lineage drift`);
  }
  deepExact(payload.source_sha256, childProtocol.payload.source_sha256, `${label}.source_sha256`);
  if (includeEvidence) {
    deepExact(
      payload.evidence_firewall,
      childProtocol.payload.evidence_firewall,
      `${label}.evidence_firewall`,
    );
    if (payload.claim_boundary !== childProtocol.payload.claim_boundary) {
      throw new Error(`${label} child claim boundary drift`);
    }
  }
}

function validateChildAttestation(attestation, childProtocol) {
  exactKeys(attestation, CHILD_ATTESTATION_KEYS, "child execution attestation");
  const attestationId = requireSha256(
    attestation.attestation_id,
    "child execution attestation.attestation_id",
  );
  const core = { ...attestation };
  delete core.attestation_id;
  if (attestationId !== sha256Bytes(canonicalBytes(core, false))) {
    throw new Error("child execution attestation ID drift");
  }
  const profile = childProtocol.payload.execution_profile;
  if (attestationId !== profile.expected_execution_attestation_id) {
    throw new Error("child execution attestation differs from protocol");
  }
  const model = childProtocol.payload.model;
  const expected = {
    profile_id: profile.profile_id,
    preset_name: profile.preset_name,
    model_id: model.model_id,
    model_revision: model.verified_revision,
    model_weights_sha256: model.model_weights_sha256,
    execution_assets_sha256: model.execution_assets_sha256,
    runtime_origin: "hf-local",
    platform_system: "Windows",
    attention_implementation: profile.attention_implementation,
    sdpa_backend: profile.sdpa_backend,
    sdpa_backend_policy: profile.sdpa_backend_policy,
    sdpa_backend_exclusive: profile.sdpa_backend_exclusive,
    generation_use_cache: profile.generation_use_cache,
    require_generation_chat_template: profile.require_generation_chat_template,
    generation_capture_strategy: profile.generation_capture_strategy,
    capture_failure_mode: profile.capture_failure_mode,
    context_window_tokens: profile.context_window_tokens,
    local_files_only: profile.local_files_only,
    fallback_mode: profile.fallback_mode,
    fail_on_truncation: profile.fail_on_truncation,
    model_dtype: profile.model_dtype,
    hidden_size: childProtocol.payload.diagnostic.activation_width,
    model_max_position_embeddings: profile.context_window_tokens,
    hook_layer_indices: childProtocol.payload.diagnostic.expected_capture.layer_indices,
  };
  for (const [key, value] of Object.entries(expected)) {
    deepExact(attestation[key], value, `child execution attestation.${key}`);
  }
  const device = requireText(attestation.device, "child execution attestation.device");
  if (device !== "cuda" && !/^cuda:\d+$/.test(device)) {
    throw new Error("child execution attestation device is not CUDA");
  }
}

function validateChildCapture(capture, childProtocol) {
  const expected = childProtocol.payload.diagnostic.expected_capture;
  const value = requireObject(capture, "child report.observation.capture");
  exactKeys(
    value,
    [
      "schema_version",
      "residual_sequence_length",
      "residual_step_continuity_exact",
      "capture_layer_exact",
      "capture_width_exact",
      "residual_activation_value_count",
      "finite_residual_activation_value_count",
      "capture_values_all_finite",
      "residual_sequence_sha256",
      "latest_activation_width",
      "latest_activation_sha256",
      "latest_matches_sequence_exact",
      "top_logit_count",
      "top_logits_finite_nonempty",
      "top_logits_sha256",
      "selected_feature_values",
      "description_sha256",
    ],
    "child report.observation.capture",
  );
  if (value.schema_version !== expected.audit_summary_schema_version) {
    throw new Error("child capture audit schema drift");
  }
  for (const key of [
    "residual_sequence_sha256",
    "latest_activation_sha256",
    "top_logits_sha256",
    "description_sha256",
  ]) {
    requireSha256(value[key], `child capture.${key}`);
  }
  for (const key of [
    "residual_step_continuity_exact",
    "capture_layer_exact",
    "capture_width_exact",
    "capture_values_all_finite",
    "latest_matches_sequence_exact",
    "top_logits_finite_nonempty",
  ]) {
    requireBoolean(value[key], `child capture.${key}`);
  }
  for (const key of [
    "residual_sequence_length",
    "residual_activation_value_count",
    "finite_residual_activation_value_count",
    "latest_activation_width",
    "top_logit_count",
  ]) {
    if (requireInteger(value[key], `child capture.${key}`) < 0) {
      throw new Error(`child capture.${key} must be non-negative`);
    }
  }
  const features = requireObject(value.selected_feature_values, "child capture.selected_feature_values");
  exactKeys(
    features,
    [
      "hook_layer_coverage",
      "hook_fire_rate",
      "token_step_coverage",
      "residual_sequence_present",
      "fallback_active",
    ],
    "child capture.selected_feature_values",
  );
  for (const [key, feature] of Object.entries(features)) {
    if (feature !== null) requireNumber(feature, `child capture.selected_feature_values.${key}`);
  }
  return value;
}

function recomputeChildChecks(report, childProtocol) {
  const diagnostic = childProtocol.payload.diagnostic;
  const profile = childProtocol.payload.execution_profile;
  const observation = requireObject(report.observation, "child report.observation");
  const budget = requireObject(observation.context_budget, "child report.observation.context_budget");
  const expectedBudget = {
    ...diagnostic.expected_context_budget,
    execution_attestation_id: profile.expected_execution_attestation_id,
  };
  const budgetExact = canonicalJson(budget) === canonicalJson(expectedBudget);
  const capture = validateChildCapture(observation.capture, childProtocol);
  const expectedCapture = diagnostic.expected_capture;
  const features = capture.selected_feature_values;
  const flags = requireObject(observation.application_flags, "child report.application_flags");
  exactKeys(
    flags,
    [
      "personal_conditioning_applied",
      "conditioning_bank_carrier_count",
      "character_prefix_applied",
      "character_residual_applied",
      "steering_intervention_applied",
    ],
    "child report.application_flags",
  );
  const carrierCount = requireInteger(
    flags.conditioning_bank_carrier_count,
    "child report.application_flags.conditioning_bank_carrier_count",
  );
  const noIntervention =
    requireBoolean(flags.personal_conditioning_applied, "personal_conditioning_applied") === false &&
    carrierCount === 0 &&
    requireBoolean(flags.character_prefix_applied, "character_prefix_applied") === false &&
    requireBoolean(flags.character_residual_applied, "character_residual_applied") === false &&
    requireBoolean(flags.steering_intervention_applied, "steering_intervention_applied") === false;
  const checks = {
    execution_attestation_exact:
      report.execution_attestation_id === profile.expected_execution_attestation_id,
    generation_execution_lineage_exact:
      budget.execution_attestation_id === report.execution_attestation_id,
    rendered_prompt_hash_exact:
      observation.rendered_prompt_sha256 ===
      diagnostic.prompt_recipe.expected_rendered_prompt_sha256,
    input_token_count_exact:
      requireInteger(observation.input_token_count, "child observation.input_token_count") ===
      requireInteger(
        diagnostic.expected_context_budget.input_token_count,
        "child diagnostic expected input count",
      ),
    context_budget_exact: budgetExact,
    generated_token_count_exact:
      requireInteger(observation.generated_token_count, "child observation.generated_token_count") ===
      requireInteger(diagnostic.generation_call.max_new_tokens, "child max new tokens"),
    capture_present: true,
    residual_sequence_length_exact:
      requireInteger(capture.residual_sequence_length, "child capture.residual_sequence_length") ===
      requireInteger(expectedCapture.residual_sequence_length, "expected residual sequence length"),
    residual_step_continuity_exact: capture.residual_step_continuity_exact === true,
    capture_layer_exact: capture.capture_layer_exact === true,
    capture_width_exact:
      capture.capture_width_exact === true &&
      requireInteger(capture.latest_activation_width, "child capture.latest_activation_width") ===
        requireInteger(expectedCapture.activation_width, "expected activation width"),
    latest_capture_matches_sequence_exact:
      capture.latest_matches_sequence_exact === expectedCapture.latest_matches_sequence_exact,
    capture_values_all_finite:
      capture.capture_values_all_finite === true &&
      requireInteger(
        capture.residual_activation_value_count,
        "child capture.residual_activation_value_count",
      ) ===
        requireInteger(expectedCapture.residual_sequence_length, "expected sequence length") *
          requireInteger(expectedCapture.activation_width, "expected activation width") &&
      requireInteger(
        capture.finite_residual_activation_value_count,
        "child capture.finite_residual_activation_value_count",
      ) ===
        requireInteger(
          capture.residual_activation_value_count,
          "child capture.residual_activation_value_count",
        ),
    top_logits_finite_nonempty:
      capture.top_logits_finite_nonempty === true &&
      requireInteger(capture.top_logit_count, "child capture.top_logit_count") > 0,
    hook_layer_coverage_exact:
      canonicalJson(features.hook_layer_coverage) ===
      canonicalJson(expectedCapture.hook_layer_coverage),
    hook_fire_rate_exact:
      canonicalJson(features.hook_fire_rate) === canonicalJson(expectedCapture.hook_fire_rate),
    token_step_coverage_exact:
      canonicalJson(features.token_step_coverage) ===
      canonicalJson(expectedCapture.token_step_coverage),
    residual_sequence_present_exact:
      canonicalJson(features.residual_sequence_present) ===
      canonicalJson(expectedCapture.residual_sequence_present),
    fallback_inactive:
      canonicalJson(features.fallback_active) === canonicalJson(expectedCapture.fallback_active),
    no_conditioning_or_steering_applied: noIntervention,
  };
  if (Object.keys(checks).some((key, index) => key !== CHILD_CHECK_KEYS[index])) {
    throw new Error("child check order/set drift in outer validator");
  }
  return checks;
}

function validateChildReport(report, childProtocol, expectedLeaseId) {
  exactKeys(
    report,
    [
      "schema_version",
      "attempt_id",
      "outer_attempt_lease_id",
      "protocol_id",
      "protocol_raw_sha256",
      "source_hash_mode",
      "source_sha256",
      "execution_attestation_id",
      "generation_call",
      "observation",
      "checks",
      "passed",
      "verdict",
      "evidence_firewall",
      "claim_boundary",
    ],
    "child report",
  );
  if (report.schema_version !== "windows-cuda-strict-32k-smoke-report.v1") {
    throw new Error("child report schema drift");
  }
  requireSha256(report.attempt_id, "child report.attempt_id");
  if (report.outer_attempt_lease_id !== expectedLeaseId) throw new Error("child report lease drift");
  validateChildProtocolLineage(report, childProtocol, "child report");
  requireSha256(report.execution_attestation_id, "child report.execution_attestation_id");
  deepExact(
    report.generation_call,
    childProtocol.payload.diagnostic.generation_call,
    "child report.generation_call",
  );
  const observation = requireObject(report.observation, "child report.observation");
  exactKeys(
    observation,
    [
      "generated_text_sha256",
      "generated_text_byte_count",
      "generated_token_count",
      "input_token_count",
      "rendered_prompt_sha256",
      "context_budget",
      "capture",
      "application_flags",
    ],
    "child report.observation",
  );
  requireSha256(observation.generated_text_sha256, "child observation.generated_text_sha256");
  requireSha256(observation.rendered_prompt_sha256, "child observation.rendered_prompt_sha256");
  for (const key of [
    "generated_text_byte_count",
    "generated_token_count",
    "input_token_count",
  ]) {
    if (requireInteger(observation[key], `child observation.${key}`) < 0) {
      throw new Error(`child observation.${key} must be non-negative`);
    }
  }
  const recomputed = recomputeChildChecks(report, childProtocol);
  const checks = requireObject(report.checks, "child report.checks");
  exactKeys(checks, CHILD_CHECK_KEYS, "child report.checks");
  for (const key of CHILD_CHECK_KEYS) {
    if (requireBoolean(checks[key], `child report.checks.${key}`) !== recomputed[key]) {
      throw new Error(`child report check drift: ${key}`);
    }
  }
  const passed = requireBoolean(report.passed, "child report.passed");
  if (passed !== CHILD_CHECK_KEYS.every((key) => checks[key] === true)) {
    throw new Error("child report passed/check drift");
  }
  const expectedVerdict = passed
    ? "passed_exact_strict_32767_plus_1_engineering_diagnostic"
    : "failed_diagnostic_stop_no_retry";
  if (report.verdict !== expectedVerdict) throw new Error("child report verdict drift");
  return { passed, verdict: expectedVerdict };
}

function validateChildLaunch(launch, childProtocol, expectedLeaseId) {
  exactKeys(
    launch,
    [
      "schema_version",
      "protocol_id",
      "protocol_raw_sha256",
      "source_hash_mode",
      "source_sha256",
      "attempt_budget",
      "retry_budget",
      "attempt_budget_scope",
      "retry_enforcement_owner",
      "outer_attempt_lease_id",
      "process_id",
      "started_at_utc",
      "attempt_id",
    ],
    "child launch receipt",
  );
  if (launch.schema_version !== "windows-cuda-strict-32k-smoke-launch.v1") {
    throw new Error("child launch receipt schema drift");
  }
  validateChildProtocolLineage(launch, childProtocol, "child launch receipt", false);
  if (
    requireInteger(launch.attempt_budget, "child launch.attempt_budget") !== 1 ||
    requireInteger(launch.retry_budget, "child launch.retry_budget") !== 0 ||
    launch.attempt_budget_scope !== "per_frozen_output_root" ||
    launch.retry_enforcement_owner !== "outer_host_campaign" ||
    launch.outer_attempt_lease_id !== expectedLeaseId ||
    requireInteger(launch.process_id, "child launch.process_id") <= 0
  ) {
    throw new Error("child launch receipt contract drift");
  }
  const startedAt = requireUtcTimestamp(launch.started_at_utc, "child launch.started_at_utc");
  const attemptId = requireSha256(launch.attempt_id, "child launch.attempt_id");
  const core = { ...launch };
  delete core.attempt_id;
  if (attemptId !== sha256Bytes(canonicalBytes(core))) {
    throw new Error("child launch attempt ID drift");
  }
  return { attemptId, processId: requireInteger(launch.process_id, "child launch.process_id"), startedAt };
}

export function validateStrict32KChildArtifact({
  outputDir,
  expectedLeaseId,
  outerProtocol,
  verifySources = true,
}) {
  const leaseId = requireSha256(expectedLeaseId, "expected child lease ID");
  const childProtocol = loadChildProtocol(outerProtocol);
  if (verifySources) verifyChildCriticalSources(childProtocol, outerProtocol.repositoryRoot);
  const root = path.resolve(outputDir);
  const stat = fs.lstatSync(root);
  if (!stat.isDirectory() || stat.isSymbolicLink()) {
    throw new Error("strict 32K child root is missing or linked");
  }
  ensureExactTopLevelEntries(root, CHILD_REQUIRED_FILES, "strict 32K child artifact");
  const loaded = Object.create(null);
  for (const fileName of CHILD_REQUIRED_FILES) {
    loaded[fileName] = loadStrictJsonFile(
      path.join(root, fileName),
      `strict 32K child ${fileName}`,
      true,
    );
  }
  const launchValidation = validateChildLaunch(
    loaded["launch_receipt.json"].value,
    childProtocol,
    leaseId,
  );
  const manifest = loaded["manifest.json"].value;
  exactKeys(
    manifest,
    [
      "schema_version",
      "attempt_id",
      "outer_attempt_lease_id",
      "protocol_id",
      "protocol_raw_sha256",
      "source_hash_mode",
      "source_sha256",
      "execution_attestation_id",
      "passed",
      "verdict",
      "files",
      "evidence_firewall",
      "claim_boundary",
      "artifact_id",
    ],
    "child manifest",
  );
  if (manifest.schema_version !== "windows-cuda-strict-32k-smoke-manifest.v1") {
    throw new Error("child manifest schema drift");
  }
  validateChildProtocolLineage(manifest, childProtocol, "child manifest");
  if (
    manifest.attempt_id !== launchValidation.attemptId ||
    manifest.outer_attempt_lease_id !== leaseId
  ) {
    throw new Error("child launch/manifest lineage drift");
  }
  const artifactId = requireSha256(manifest.artifact_id, "child manifest.artifact_id");
  const manifestCore = { ...manifest };
  delete manifestCore.artifact_id;
  if (artifactId !== sha256Bytes(canonicalBytes(manifestCore))) {
    throw new Error("child artifact ID drift");
  }
  const records = requireArray(manifest.files, "child manifest.files");
  if (records.length !== CHILD_PAYLOAD_FILES.length) throw new Error("child manifest file count drift");
  records.forEach((rawRecord, index) => {
    const record = requireObject(rawRecord, `child manifest.files[${index}]`);
    exactKeys(record, ["path", "byte_count", "sha256"], `child manifest.files[${index}]`);
    const expectedName = CHILD_PAYLOAD_FILES[index];
    if (record.path !== expectedName) throw new Error("child manifest file order/path drift");
    const payload = loaded[expectedName].raw;
    if (
      requireInteger(record.byte_count, `child manifest ${expectedName} byte_count`) !== payload.length ||
      requireSha256(record.sha256, `child manifest ${expectedName} sha256`) !== sha256Bytes(payload)
    ) {
      throw new Error(`child manifest payload drift: ${expectedName}`);
    }
  });
  const attestation = loaded["execution_attestation.json"].value;
  validateChildAttestation(attestation, childProtocol);
  const report = loaded["strict_32k_smoke_report.json"].value;
  const reportValidation = validateChildReport(report, childProtocol, leaseId);
  if (
    report.attempt_id !== launchValidation.attemptId ||
    manifest.execution_attestation_id !== attestation.attestation_id ||
    report.execution_attestation_id !== attestation.attestation_id ||
    manifest.passed !== reportValidation.passed ||
    manifest.verdict !== reportValidation.verdict
  ) {
    throw new Error("child manifest/report/attestation lineage drift");
  }
  const completion = loaded["completion_receipt.json"].value;
  exactKeys(
    completion,
    [
      "schema_version",
      "attempt_id",
      "outer_attempt_lease_id",
      "artifact_id",
      "protocol_id",
      "execution_attestation_id",
      "passed",
      "verdict",
      "completed_at_utc",
      "completion_id",
    ],
    "child completion receipt",
  );
  if (
    completion.schema_version !== "windows-cuda-strict-32k-smoke-completion.v1" ||
    completion.attempt_id !== launchValidation.attemptId ||
    completion.outer_attempt_lease_id !== leaseId ||
    completion.artifact_id !== artifactId ||
    completion.protocol_id !== childProtocol.protocolId ||
    completion.execution_attestation_id !== attestation.attestation_id ||
    completion.passed !== reportValidation.passed ||
    completion.verdict !== reportValidation.verdict
  ) {
    throw new Error("child completion receipt lineage drift");
  }
  const completedAt = requireUtcTimestamp(
    completion.completed_at_utc,
    "child completion.completed_at_utc",
  );
  if (completedAt < launchValidation.startedAt) throw new Error("child completion predates launch");
  const completionId = requireSha256(completion.completion_id, "child completion.completion_id");
  const completionCore = { ...completion };
  delete completionCore.completion_id;
  if (completionId !== sha256Bytes(canonicalBytes(completionCore))) {
    throw new Error("child completion ID drift");
  }
  return Object.freeze({
    state: "complete_valid",
    artifactId,
    attemptId: launchValidation.attemptId,
    processId: launchValidation.processId,
    executionAttestationId: attestation.attestation_id,
    protocolId: childProtocol.protocolId,
    passed: reportValidation.passed,
    verdict: reportValidation.verdict,
    startedAt: launchValidation.startedAt,
    completedAt,
  });
}

function normalizeEventData(raw, label) {
  return requireArray(raw, label).map((rawEntry, index) => {
    const entry = requireObject(rawEntry, `${label}[${index}]`);
    exactKeys(entry, ["name", "value"], `${label}[${index}]`);
    if (typeof entry.name !== "string" || typeof entry.value !== "string") {
      throw new TypeError(`${label}[${index}] name/value must be exact text`);
    }
    return { name: entry.name, value: entry.value };
  });
}

function normalizeEventRecord(raw, label) {
  const event = requireObject(raw, label);
  exactKeys(
    event,
    [
      "log_name",
      "provider_name",
      "event_id",
      "record_id",
      "level",
      "time_created_utc",
      "xml_sha256",
      "payload_kind",
      "event_data",
    ],
    label,
  );
  if (!new Set(["Application", "System"]).has(event.log_name)) {
    throw new Error(`${label}.log_name drift`);
  }
  const provider = requireText(event.provider_name, `${label}.provider_name`);
  const eventId = requireInteger(event.event_id, `${label}.event_id`);
  const recordId = requireInteger(event.record_id, `${label}.record_id`);
  const level = event.level === null ? null : requireInteger(event.level, `${label}.level`);
  requireUtcTimestamp(event.time_created_utc, `${label}.time_created_utc`);
  requireSha256(event.xml_sha256, `${label}.xml_sha256`);
  const payloadKind = requireText(event.payload_kind, `${label}.payload_kind`);
  if (!new Set(["event_data", "user_data", "binary_event_data", "none"]).has(payloadKind)) {
    throw new Error(`${label}.payload_kind drift`);
  }
  return {
    log_name: event.log_name,
    provider_name: provider,
    event_id: eventId,
    record_id: recordId,
    level,
    time_created_utc: event.time_created_utc,
    xml_sha256: event.xml_sha256,
    payload_kind: payloadKind,
    event_data: normalizeEventData(event.event_data, `${label}.event_data`),
  };
}

function classifyEvent(event, protocol) {
  const classifications = [];
  const rules = requireArray(
    protocol.payload.event_log.fault_classification,
    "protocol.event_log.fault_classification",
  );
  for (const [index, rawRule] of rules.entries()) {
    const rule = requireObject(rawRule, `fault classification[${index}]`);
    exactKeys(
      rule,
      ["failure_code", "log_name", "provider_name", "event_ids"],
      `fault classification[${index}]`,
    );
    if (event.log_name !== rule.log_name || event.provider_name !== rule.provider_name) continue;
    const ids = rule.event_ids;
    const idMatches =
      ids === null ||
      requireArray(ids, `fault classification[${index}].event_ids`).some(
        (value) => requireInteger(value, "fault event ID") === event.event_id,
      );
    if (idMatches) classifications.push(rule.failure_code);
  }
  return classifications;
}

function normalizeDeltaCollectorPayload(raw, { protocol, baseline }) {
  const payload = requireObject(raw, "Event Log delta collector output");
  exactKeys(
    payload,
    ["schema_version", "collection_started_at_utc", "collection_completed_at_utc", "host", "channels"],
    "Event Log delta collector output",
  );
  if (payload.schema_version !== "windows-host-event-log-delta-collector.v1") {
    throw new Error("Event Log delta collector schema drift");
  }
  const started = requireUtcTimestamp(payload.collection_started_at_utc, "Event Log delta start");
  const completed = requireUtcTimestamp(payload.collection_completed_at_utc, "Event Log delta completion");
  if (completed < started) throw new Error("Event Log delta completion predates start");
  const host = validateHostObservation(payload.host, "Event Log delta.host");
  const baselineChannels = requireArray(baseline.channels, "baseline.channels").map(
    (channel, index) => validateEventChannelCursor(channel, `baseline.channels[${index}]`),
  );
  const channels = requireArray(payload.channels, "Event Log delta.channels");
  if (channels.length !== 2) throw new Error("Event Log delta channel count drift");
  const normalizedChannels = channels.map((rawChannel, index) => {
    const channel = requireObject(rawChannel, `Event Log delta.channels[${index}]`);
    exactKeys(
      channel,
      [
        "log_name",
        "baseline_newest_record_id",
        "baseline_boundary_present",
        "baseline_boundary_xml_sha256",
        "end_cursor",
        "channel_configuration_stable",
        "end_cursor_hash_exact",
        "scanned_record_count",
        "record_id_range_complete",
        "truncated",
        "events",
      ],
      `Event Log delta.channels[${index}]`,
    );
    const baselineChannel = baselineChannels[index];
    if (
      channel.log_name !== baselineChannel.log_name ||
      requireInteger(channel.baseline_newest_record_id, "delta baseline newest record ID") !==
        requireInteger(baselineChannel.newest_record_id, "baseline newest record ID")
    ) {
      throw new Error("Event Log delta/baseline channel lineage drift");
    }
    const boundaryPresent = requireBoolean(
      channel.baseline_boundary_present,
      "delta baseline_boundary_present",
    );
    const boundaryHash =
      channel.baseline_boundary_xml_sha256 === null
        ? null
        : requireSha256(
            channel.baseline_boundary_xml_sha256,
            "delta baseline_boundary_xml_sha256",
          );
    const endCursor = validateEventChannelCursor(
      channel.end_cursor,
      `Event Log delta.channels[${index}].end_cursor`,
    );
    if (endCursor.log_name !== channel.log_name) {
      throw new Error("Event Log delta end cursor log-name drift");
    }
    const events = requireArray(channel.events, `Event Log delta.channels[${index}].events`).map(
      (event, eventIndex) =>
        normalizeEventRecord(event, `Event Log delta.channels[${index}].events[${eventIndex}]`),
    );
    const seen = new Set();
    for (const event of events) {
      if (event.log_name !== channel.log_name) throw new Error("Event Log delta event log drift");
      if (seen.has(event.record_id)) throw new Error("Event Log delta duplicate record ID");
      seen.add(event.record_id);
      if (
        event.record_id <= requireInteger(channel.baseline_newest_record_id, "delta baseline ID") ||
        event.record_id > requireInteger(endCursor.newest_record_id, "delta end newest ID")
      ) {
        throw new Error("Event Log delta event falls outside frozen cursor interval");
      }
    }
    const sortedIds = [...seen].sort((left, right) => left - right);
    if (events.some((event, eventIndex) => event.record_id !== sortedIds[eventIndex])) {
      throw new Error("Event Log delta events must be sorted by RecordID");
    }
    const scannedCount = requireInteger(
      channel.scanned_record_count,
      "Event Log delta scanned_record_count",
    );
    if (scannedCount !== events.length) throw new Error("Event Log delta scanned count drift");
    const baselineNewest = requireInteger(
      channel.baseline_newest_record_id,
      "delta baseline newest record ID",
    );
    const difference = endCursor.newest_record_id - baselineNewest;
    const maxRecords = requireInteger(
      protocol.payload.event_log.max_new_records_per_channel,
      "protocol max new Event Log records",
    );
    const expectedTruncated = difference > maxRecords;
    const reportedTruncated = requireBoolean(channel.truncated, "delta truncated");
    if (reportedTruncated !== expectedTruncated) {
      throw new Error("Event Log delta truncated flag is not derivable from frozen cursors");
    }
    const configurationStable =
      endCursor.log_mode === baselineChannel.log_mode &&
      endCursor.maximum_size_bytes === baselineChannel.maximum_size_bytes;
    if (
      requireBoolean(
        channel.channel_configuration_stable,
        "delta channel_configuration_stable",
      ) !== configurationStable
    ) {
      throw new Error("Event Log delta channel configuration flag drift");
    }
    let idsComplete = difference >= 0 && !expectedTruncated && events.length === difference;
    if (idsComplete) {
      idsComplete = events.every(
        (event, eventIndex) => event.record_id === baselineNewest + eventIndex + 1,
      );
    }
    if (expectedTruncated && events.length !== 0) {
      throw new Error("truncated Event Log delta must not publish a selected partial event window");
    }
    const expectedEndHashExact =
      difference === 0
        ? boundaryPresent && boundaryHash === endCursor.newest_record_xml_sha256
        : difference > 0 && !expectedTruncated && events.length > 0
          ? events.at(-1).xml_sha256 === endCursor.newest_record_xml_sha256
          : false;
    if (
      requireBoolean(channel.end_cursor_hash_exact, "delta end_cursor_hash_exact") !==
      expectedEndHashExact
    ) {
      throw new Error("Event Log delta end-cursor hash flag drift");
    }
    const expectedRangeComplete =
      idsComplete &&
      configurationStable &&
      boundaryPresent &&
      boundaryHash === baselineChannel.newest_record_xml_sha256 &&
      expectedEndHashExact;
    if (
      requireBoolean(channel.record_id_range_complete, "delta record_id_range_complete") !==
      expectedRangeComplete
    ) {
      throw new Error("Event Log delta range-complete flag is not derivable from stored records");
    }
    return {
      log_name: channel.log_name,
      baseline_newest_record_id: requireInteger(
        channel.baseline_newest_record_id,
        "delta baseline newest record ID",
      ),
      baseline_boundary_present: boundaryPresent,
      baseline_boundary_xml_sha256: boundaryHash,
      end_cursor: endCursor,
      channel_configuration_stable: configurationStable,
      end_cursor_hash_exact: expectedEndHashExact,
      scanned_record_count: scannedCount,
      record_id_range_complete: expectedRangeComplete,
      truncated: expectedTruncated,
      events,
    };
  });
  if (
    normalizedChannels[0].log_name !== "Application" ||
    normalizedChannels[1].log_name !== "System"
  ) {
    throw new Error("Event Log delta channel order drift");
  }
  const sameMachine = host.machine_identity_sha256 === baseline.host.machine_identity_sha256;
  const sameBoot = host.boot_identity_sha256 === baseline.host.boot_identity_sha256;
  const classifiedFaults = [];
  for (const channel of normalizedChannels) {
    for (const event of channel.events) {
      for (const failureCode of classifyEvent(event, protocol)) {
        classifiedFaults.push({
          failure_code: failureCode,
          log_name: event.log_name,
          record_id: event.record_id,
          provider_name: event.provider_name,
          event_id: event.event_id,
          xml_sha256: event.xml_sha256,
        });
      }
    }
  }
  const continuityChecks = {
    same_machine: sameMachine,
    same_boot: sameBoot,
    application_boundary_present:
      normalizedChannels[0].baseline_boundary_present === true,
    application_boundary_hash_exact:
      normalizedChannels[0].baseline_boundary_xml_sha256 ===
      baselineChannels[0].newest_record_xml_sha256,
    application_cursor_monotonic:
      normalizedChannels[0].end_cursor.newest_record_id >=
      normalizedChannels[0].baseline_newest_record_id,
    application_no_rollover_gap:
      normalizedChannels[0].end_cursor.oldest_record_id <=
      normalizedChannels[0].baseline_newest_record_id,
    application_range_complete:
      normalizedChannels[0].record_id_range_complete === true &&
      normalizedChannels[0].truncated === false,
    system_boundary_present: normalizedChannels[1].baseline_boundary_present === true,
    system_boundary_hash_exact:
      normalizedChannels[1].baseline_boundary_xml_sha256 ===
      baselineChannels[1].newest_record_xml_sha256,
    system_cursor_monotonic:
      normalizedChannels[1].end_cursor.newest_record_id >=
      normalizedChannels[1].baseline_newest_record_id,
    system_no_rollover_gap:
      normalizedChannels[1].end_cursor.oldest_record_id <=
      normalizedChannels[1].baseline_newest_record_id,
    system_range_complete:
      normalizedChannels[1].record_id_range_complete === true &&
      normalizedChannels[1].truncated === false,
  };
  return {
    collection_started_at_utc: payload.collection_started_at_utc,
    collection_completed_at_utc: payload.collection_completed_at_utc,
    host,
    channels: normalizedChannels,
    classified_faults: classifiedFaults,
    continuity_checks: continuityChecks,
  };
}

function defaultChildExecutor({ argv, environment, stdoutFd, stderrFd, timeoutMilliseconds }) {
  const started = Date.now();
  const result = childProcess.spawnSync(argv[0], argv.slice(1), {
    cwd: REPOSITORY_ROOT,
    env: environment,
    windowsHide: true,
    shell: false,
    stdio: ["ignore", stdoutFd, stderrFd],
    timeout: timeoutMilliseconds,
    killSignal: "SIGKILL",
  });
  const completed = Date.now();
  return {
    process_started: Number.isSafeInteger(result.pid) && result.pid > 0,
    process_id: Number.isSafeInteger(result.pid) && result.pid > 0 ? result.pid : null,
    exit_code: Number.isSafeInteger(result.status) ? result.status : null,
    signal: result.signal === null ? null : String(result.signal),
    error_code: result.error?.code ? String(result.error.code) : null,
    timed_out: result.error?.code === "ETIMEDOUT",
    duration_milliseconds: Math.max(0, completed - started),
  };
}

function inspectChildArtifact({ campaign, childValidator, verifySources }) {
  const outputDir = path.join(
    campaign.root,
    ...campaign.protocol.payload.child.output_relative_path.split("/"),
  );
  if (!fs.existsSync(outputDir)) {
    return {
      state: "missing",
      validation_error_name: null,
      result: null,
    };
  }
  const stat = fs.lstatSync(outputDir);
  if (!stat.isDirectory() || stat.isSymbolicLink()) {
    throw new Error("child output path is not a regular directory");
  }
  const actual = fs.readdirSync(outputDir).sort();
  const complete = [...CHILD_REQUIRED_FILES].sort();
  if (
    actual.length !== complete.length ||
    actual.some((name, index) => name !== complete[index])
  ) {
    return {
      state: "incomplete",
      validation_error_name: null,
      result: null,
    };
  }
  try {
    const result = childValidator({
      outputDir,
      expectedLeaseId: campaign.leaseId,
      outerProtocol: campaign.protocol,
      verifySources,
    });
    return { state: "complete_valid", validation_error_name: null, result };
  } catch (error) {
    if (!(error instanceof Error)) throw error;
    return {
      state: "complete_invalid",
      validation_error_name: error.name,
      result: null,
    };
  }
}

function readLogObservation(filePath) {
  const stat = fs.lstatSync(filePath);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1) {
    throw new Error(`child log is not one regular file: ${filePath}`);
  }
  const raw = fs.readFileSync(filePath);
  return { byte_count: raw.length, sha256: sha256Bytes(raw) };
}

function validateChildTerminalStdout(stdoutPath, childInspection, campaign) {
  if (childInspection.state !== "complete_valid") return false;
  const text = new TextDecoder("utf-8", { fatal: true }).decode(fs.readFileSync(stdoutPath));
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line !== "");
  if (lines.length === 0) return false;
  let terminal;
  try {
    terminal = parseJsonStrict(lines[lines.length - 1], "child terminal stdout JSON");
  } catch (error) {
    if (!(error instanceof Error)) throw error;
    return false;
  }
  exactKeys(
    terminal,
    [
      "artifact_id",
      "attempt_id",
      "outer_attempt_lease_id",
      "protocol_id",
      "execution_attestation_id",
      "passed",
      "verdict",
    ],
    "child terminal stdout JSON",
  );
  const result = childInspection.result;
  return (
    terminal.artifact_id === result.artifactId &&
    terminal.attempt_id === result.attemptId &&
    terminal.outer_attempt_lease_id === campaign.leaseId &&
    terminal.protocol_id === result.protocolId &&
    terminal.execution_attestation_id === result.executionAttestationId &&
    terminal.passed === result.passed &&
    terminal.verdict === result.verdict
  );
}

function normalizeProcessObservation(raw) {
  const observation = requireObject(raw, "child process observation");
  exactKeys(
    observation,
    [
      "process_started",
      "process_id",
      "exit_code",
      "signal",
      "error_code",
      "timed_out",
      "duration_milliseconds",
    ],
    "child process observation",
  );
  const processStarted = requireBoolean(observation.process_started, "process_started");
  const processId = observation.process_id === null ? null : requireInteger(observation.process_id, "process_id");
  const exitCode = observation.exit_code === null ? null : requireInteger(observation.exit_code, "exit_code");
  const signal = observation.signal === null ? null : requireText(observation.signal, "signal");
  const errorCode = observation.error_code === null ? null : requireText(observation.error_code, "error_code");
  const timedOut = requireBoolean(observation.timed_out, "timed_out");
  const duration = requireInteger(observation.duration_milliseconds, "duration_milliseconds");
  if (duration < 0) throw new Error("child process duration must be non-negative");
  if (processStarted !== (processId !== null && processId > 0)) {
    throw new Error("child process started/PID drift");
  }
  if (
    (exitCode === 0 || exitCode === 2) &&
    (!processStarted || signal !== null || errorCode !== null || timedOut)
  ) {
    throw new Error("terminal child exit code conflicts with signal/error/timeout observation");
  }
  if (exitCode === null && signal === null && errorCode === null && !timedOut) {
    throw new Error("child process observation has no terminal outcome");
  }
  return {
    process_started: processStarted,
    process_id: processId,
    exit_code: exitCode,
    signal,
    error_code: errorCode,
    timed_out: timedOut,
    duration_milliseconds: duration,
  };
}

function orderedFailureCodes(codes) {
  const unique = new Set(codes);
  for (const code of unique) {
    if (!FAILURE_CODE_ORDER.includes(code)) throw new Error(`unknown host campaign failure code: ${code}`);
  }
  return FAILURE_CODE_ORDER.filter((code) => unique.has(code));
}

function buildCampaignReportPayload({
  campaign,
  previousReceiptSha256,
  processObservation,
  childInspection,
  childTerminalStdoutValid,
  delta,
  anchorInventory,
  evaluation,
}) {
  return {
    ...chainBase({
      schemaVersion: "windows-cuda-strict-32k-host-campaign-report.v1",
      sequence: 8,
      scopeId: campaign.scopeId,
      previousReceiptSha256,
    }),
    lease_id: campaign.leaseId,
    outer_protocol_id: campaign.protocol.protocolId,
    child_protocol_id: campaign.childProtocol.protocolId,
    host_qualification_artifact_id: campaign.qualification.payload.artifact_id,
    execution_backend_id: campaign.scopeLoaded.value.execution_backend_id,
    real_execution_observation_authorized:
      campaign.scopeLoaded.value.real_execution_observation_authorized,
    process: processObservation,
    child: {
      state: childInspection.state,
      validation_error_name: childInspection.validation_error_name,
      artifact_id: childInspection.result?.artifactId ?? null,
      attempt_id: childInspection.result?.attemptId ?? null,
      process_id: childInspection.result?.processId ?? null,
      execution_attestation_id: childInspection.result?.executionAttestationId ?? null,
      passed: childInspection.result?.passed ?? null,
      verdict: childInspection.result?.verdict ?? null,
      terminal_stdout_valid: childTerminalStdoutValid,
    },
    event_log: {
      collection_status: delta.collection_status,
      continuity_checks: delta.continuity_checks,
      classified_fault_count: delta.classified_faults.length,
    },
    local_anchor_inventory: anchorInventory.map((anchor) => ({
      anchor_kind: anchor.message.anchor_kind,
      receipt_sha256: anchor.message.receipt_sha256,
      record_id: anchor.record_id,
    })),
    passed: evaluation.passed,
    verdict: evaluation.verdict,
    failure_codes: evaluation.failure_codes,
    retry_permitted: false,
    evidence_firewall: OUTER_EVIDENCE_FIREWALL,
    claim_boundary: OUTER_CLAIM_BOUNDARY,
  };
}

function evaluateCampaign({
  protocol,
  campaign,
  processObservation,
  childInspection,
  childTerminalStdoutValid,
  delta,
  anchorInventory,
}) {
  const codes = [];
  if (campaign.scopeLoaded.value.real_execution_observation_authorized !== true) {
    codes.push("synthetic_test_backend_not_evidence");
  }
  if (!processObservation.process_started) codes.push("process_start_failed");
  if (processObservation.timed_out) codes.push("child_timeout");
  if (processObservation.exit_code === 2) {
    codes.push("child_diagnostic_failed_exit_2");
  } else if (processObservation.exit_code !== 0) {
    codes.push("child_runtime_failed_nonzero");
  }
  if (childInspection.state === "missing" || childInspection.state === "incomplete") {
    codes.push("child_root_missing_or_incomplete");
  } else if (childInspection.state === "complete_invalid") {
    codes.push("child_lineage_mismatch");
  } else {
    const expectedChildPass =
      processObservation.exit_code === 0
        ? true
        : processObservation.exit_code === 2
          ? false
          : null;
    if (
      childInspection.result.processId !== processObservation.process_id ||
      childTerminalStdoutValid !== true ||
      (expectedChildPass !== null && childInspection.result.passed !== expectedChildPass)
    ) {
      codes.push("child_lineage_mismatch");
    }
  }
  if (delta.collection_status !== "complete") {
    codes.push("event_log_collection_failed");
  } else {
    if (Object.values(delta.continuity_checks).some((value) => value !== true)) {
      codes.push("event_log_continuity_lost");
    }
    const rolloverKeys = [
      "application_boundary_present",
      "application_boundary_hash_exact",
      "application_cursor_monotonic",
      "application_no_rollover_gap",
      "application_range_complete",
      "system_boundary_present",
      "system_boundary_hash_exact",
      "system_cursor_monotonic",
      "system_no_rollover_gap",
      "system_range_complete",
    ];
    if (rolloverKeys.some((key) => delta.continuity_checks[key] !== true)) {
      codes.push("event_log_cleared_or_rolled_over");
    }
    for (const fault of delta.classified_faults) codes.push(fault.failure_code);
  }
  const preregAnchors = anchorInventory.filter(
    (anchor) =>
      anchor.message.anchor_kind === "preregistration" &&
      anchor.message.receipt_sha256 === campaign.leaseId &&
      anchor.message.lease_id === campaign.leaseId,
  );
  const launchAnchors = anchorInventory.filter(
    (anchor) =>
      anchor.message.anchor_kind === "launch" &&
      anchor.message.receipt_sha256 === campaign.launchLoaded.rawSha256 &&
      anchor.message.lease_id === campaign.leaseId,
  );
  const terminalAnchors = anchorInventory.filter(
    (anchor) => anchor.message.anchor_kind === "terminal",
  );
  if (
    anchorInventory.length !== 2 ||
    preregAnchors.length !== 1 ||
    launchAnchors.length !== 1 ||
    terminalAnchors.length !== 0
  ) {
    codes.push("local_anchor_mismatch");
  }
  const failureCodes = orderedFailureCodes(codes);
  return {
    passed: failureCodes.length === 0,
    verdict:
      failureCodes.length === 0
        ? "passed_single_preregistered_strict_32k_host_campaign"
        : "failed_host_campaign_stop_no_retry",
    failure_codes: failureCodes,
  };
}

function listRegularFiles(root) {
  const records = [];
  function visit(current, relative) {
    const stat = fs.lstatSync(current);
    if (stat.isSymbolicLink()) throw new Error(`campaign artifact contains a linked path: ${relative || "."}`);
    if (stat.isDirectory()) {
      for (const name of fs.readdirSync(current).sort()) {
        visit(path.join(current, name), relative ? `${relative}/${name}` : name);
      }
      return;
    }
    if (!stat.isFile() || stat.nlink !== 1) {
      throw new Error(`campaign artifact path is not one regular file: ${relative}`);
    }
    requireRelativePosixPath(relative, "campaign artifact relative file path");
    const raw = fs.readFileSync(current);
    records.push({ path: relative, byte_count: raw.length, sha256: sha256Bytes(raw) });
  }
  visit(root, "");
  return records;
}

function manifestPayloadRecords(campaignRoot) {
  const excluded = new Set(POST_MANIFEST_FILES);
  return listRegularFiles(campaignRoot).filter((record) => !excluded.has(record.path));
}

function assertExpectedPreManifestInventory(records, protocol, childState) {
  const paths = records.map((record) => record.path);
  const outerAllowed = new Set(PRE_MANIFEST_OUTER_FILES);
  for (const required of PRE_MANIFEST_OUTER_FILES) {
    if (!paths.includes(required)) throw new Error(`campaign manifest is missing required payload: ${required}`);
  }
  const childPrefix = `${protocol.payload.child.output_relative_path}/`;
  const childPaths = paths.filter((item) => item.startsWith(childPrefix));
  const childAllowed = new Set(CHILD_REQUIRED_FILES.map((name) => `${childPrefix}${name}`));
  for (const payloadPath of paths) {
    if (!outerAllowed.has(payloadPath) && !childAllowed.has(payloadPath)) {
      throw new Error(`campaign manifest contains an unexpected payload: ${payloadPath}`);
    }
  }
  if (childState === "missing" && childPaths.length !== 0) {
    throw new Error("missing child state has child payload files");
  }
  if (childState === "complete_valid" || childState === "complete_invalid") {
    const expected = CHILD_REQUIRED_FILES.map((name) => `${childPrefix}${name}`).sort();
    const actual = [...childPaths].sort();
    if (actual.length !== expected.length || actual.some((name, index) => name !== expected[index])) {
      throw new Error("complete child state file set drift");
    }
  }
}

function runHostCampaignCore({
  campaignRoot,
  protocolPath = DEFAULT_PROTOCOL_PATH,
  repositoryRoot = REPOSITORY_ROOT,
  eventLogCollector,
  childExecutor,
  childValidator,
  verifySources = true,
  allowSyntheticTestBackend = false,
} = {}) {
  const campaign = loadPreregisteredCampaign({
    campaignRoot,
    protocolPath,
    repositoryRoot,
    verifySources,
    allowSyntheticTestBackend,
  });
  const isSynthetic =
    campaign.scopeLoaded.value.execution_backend_id === SYNTHETIC_TEST_BACKEND_ID;
  if (allowSyntheticTestBackend !== isSynthetic) {
    throw new Error("campaign execution backend does not match the selected runner");
  }
  if (
    !allowSyntheticTestBackend &&
    (eventLogCollector !== defaultWindowsEventLogCollector ||
      childExecutor !== defaultChildExecutor ||
      childValidator !== validateStrict32KChildArtifact ||
      verifySources !== true)
  ) {
    throw new Error("production execution requires the bundled verified backend");
  }
  if (
    !allowSyntheticTestBackend &&
    campaign.protocol.payload.host_qualification.production_preregistration_enabled !== true
  ) {
    throw new Error("this protocol does not authorize production campaign execution");
  }
  const executable = inspectExecutable(campaign.preregLoaded.value.python_executable.absolute_path);
  deepExact(
    executable,
    campaign.preregLoaded.value.python_executable,
    "preregistered Python executable",
  );
  if (verifySources) {
    verifyChildCriticalSources(campaign.childProtocol, campaign.protocol.repositoryRoot);
  }
  const livePreregAnchor = normalizeAnchorObservation(
    eventLogCollector.verifyAnchor({
      protocol: campaign.protocol,
      message: campaign.anchorMessage,
      recordId: campaign.anchorLoaded.value.observation.record_id,
    }),
    campaign.anchorMessage,
    campaign.protocol,
  );
  if (
    livePreregAnchor.record_id !==
    requireInteger(
      campaign.anchorLoaded.value.observation.record_id,
      "preregistration Event Log anchor record ID",
    )
  ) {
    throw new Error("live preregistration Event Log anchor record drift");
  }
  const prelaunchInventory = normalizeAnchorInventory(
    eventLogCollector.findAnchors({ protocol: campaign.protocol, scopeId: campaign.scopeId }),
    { protocol: campaign.protocol, scopeId: campaign.scopeId },
  );
  if (
    prelaunchInventory.length !== 1 ||
    prelaunchInventory[0].message.anchor_kind !== "preregistration" ||
    prelaunchInventory[0].message.receipt_sha256 !== campaign.leaseId
  ) {
    throw new Error("prelaunch Event Log anchor inventory is not unique");
  }
  const prelaunch = normalizePrelaunchCollectorPayload(
    eventLogCollector.capturePrelaunch({
      protocol: campaign.protocol,
      baselinePath: path.join(campaign.root, RECEIPT_FILES.eventLogBaseline),
    }),
    {
      protocol: campaign.protocol,
      baseline: campaign.baselineLoaded.value,
      qualification: campaign.qualification,
    },
  );
  const argv = buildRealizedChildArgv({
    protocol: campaign.protocol,
    preregistration: campaign.preregLoaded.value,
    leaseId: campaign.leaseId,
    campaignRoot: campaign.root,
  });
  const environmentOverrides = normalizedEnvironmentOverrides(campaign.protocol);
  const launch = {
    ...chainBase({
      schemaVersion: "windows-cuda-strict-32k-host-campaign-launch.v1",
      sequence: 4,
      scopeId: campaign.scopeId,
      previousReceiptSha256: campaign.anchorLoaded.rawSha256,
    }),
    lease_id: campaign.leaseId,
    attempt_ordinal: 1,
    launch_commit_semantics: "create_new_file_fsync_before_process_creation",
    parent_process_id: process.pid,
    python_executable: executable,
    argv,
    environment_overrides: environmentOverrides,
    environment_removed: campaign.protocol.payload.child_process.environment_removed,
    prelaunch,
    child_output_relative_path: campaign.protocol.payload.child.output_relative_path,
    launch_committed_at_utc: nowUtc(),
  };
  const launchWritten = writeCreateJson(path.join(campaign.root, RECEIPT_FILES.launch), launch);
  const launchedCampaign = Object.freeze({
    ...campaign,
    launchLoaded: Object.freeze({
      value: launch,
      raw: launchWritten.raw,
      rawSha256: launchWritten.rawSha256,
    }),
  });
  const launchMessage = anchorMessage({
    protocol: campaign.protocol,
    scopeId: campaign.scopeId,
    kind: "launch",
    sequence: 5,
    receiptSha256: launchWritten.rawSha256,
    leaseId: campaign.leaseId,
  });
  const launchAnchorReceipt = buildEventAnchorReceipt({
    protocol: campaign.protocol,
    scopeId: campaign.scopeId,
    sequence: 5,
    kind: "launch",
    previousReceiptSha256: launchWritten.rawSha256,
    leaseId: campaign.leaseId,
    observation: eventLogCollector.writeAnchor({ protocol: campaign.protocol, message: launchMessage }),
  });
  const launchAnchorWritten = writeCreateJson(
    path.join(campaign.root, RECEIPT_FILES.launchAnchor),
    launchAnchorReceipt,
  );
  const streamsDir = path.join(campaign.root, "streams");
  fs.mkdirSync(streamsDir, { recursive: false });
  const stdoutPath = path.join(streamsDir, "child.stdout.log");
  const stderrPath = path.join(streamsDir, "child.stderr.log");
  const stdoutFd = fs.openSync(stdoutPath, "wx", 0o600);
  const stderrFd = fs.openSync(stderrPath, "wx", 0o600);
  let rawProcessObservation;
  try {
    rawProcessObservation = childExecutor({
      argv,
      environment: isolatedChildEnvironment(campaign.protocol),
      stdoutFd,
      stderrFd,
      timeoutMilliseconds:
        requireInteger(campaign.protocol.payload.child_process.timeout_seconds, "child timeout") * 1000,
      campaignRoot: campaign.root,
      childOutputDir: path.join(
        campaign.root,
        ...campaign.protocol.payload.child.output_relative_path.split("/"),
      ),
      leaseId: campaign.leaseId,
      protocol: campaign.protocol,
    });
  } finally {
    fs.fsyncSync(stdoutFd);
    fs.fsyncSync(stderrFd);
    fs.closeSync(stdoutFd);
    fs.closeSync(stderrFd);
  }
  const processObservation = normalizeProcessObservation(rawProcessObservation);
  const stdoutObservation = readLogObservation(stdoutPath);
  const stderrObservation = readLogObservation(stderrPath);
  const processExit = {
    ...chainBase({
      schemaVersion: "windows-cuda-strict-32k-host-campaign-process-exit.v1",
      sequence: 6,
      scopeId: campaign.scopeId,
      previousReceiptSha256: launchAnchorWritten.rawSha256,
    }),
    lease_id: campaign.leaseId,
    argv,
    environment_overrides: environmentOverrides,
    environment_removed: campaign.protocol.payload.child_process.environment_removed,
    ...processObservation,
    stdout: stdoutObservation,
    stderr: stderrObservation,
    observed_at_utc: nowUtc(),
  };
  const processWritten = writeCreateJson(
    path.join(campaign.root, RECEIPT_FILES.processExit),
    processExit,
  );
  let delta;
  try {
    const normalized = normalizeDeltaCollectorPayload(
      eventLogCollector.captureDelta({
        protocol: campaign.protocol,
        baselinePath: path.join(campaign.root, RECEIPT_FILES.eventLogBaseline),
      }),
      { protocol: campaign.protocol, baseline: campaign.baselineLoaded.value },
    );
    delta = {
      ...chainBase({
        schemaVersion: "windows-cuda-strict-32k-host-campaign-event-log-delta.v1",
        sequence: 7,
        scopeId: campaign.scopeId,
        previousReceiptSha256: processWritten.rawSha256,
      }),
      lease_id: campaign.leaseId,
      collection_status: "complete",
      collector_error_name: null,
      collector_error_sha256: null,
      ...normalized,
    };
  } catch (error) {
    if (!(error instanceof Error)) throw error;
    delta = {
      ...chainBase({
        schemaVersion: "windows-cuda-strict-32k-host-campaign-event-log-delta.v1",
        sequence: 7,
        scopeId: campaign.scopeId,
        previousReceiptSha256: processWritten.rawSha256,
      }),
      lease_id: campaign.leaseId,
      collection_status: "failed",
      collector_error_name: error.name,
      collector_error_sha256: sha256Bytes(Buffer.from(error.message, "utf8")),
      collection_started_at_utc: null,
      collection_completed_at_utc: null,
      host: null,
      channels: [],
      classified_faults: [],
      continuity_checks: {
        same_machine: false,
        same_boot: false,
        application_boundary_present: false,
        application_boundary_hash_exact: false,
        application_cursor_monotonic: false,
        application_no_rollover_gap: false,
        application_range_complete: false,
        system_boundary_present: false,
        system_boundary_hash_exact: false,
        system_cursor_monotonic: false,
        system_no_rollover_gap: false,
        system_range_complete: false,
      },
    };
  }
  const deltaWritten = writeCreateJson(
    path.join(campaign.root, RECEIPT_FILES.eventLogDelta),
    delta,
  );
  const childInspection = inspectChildArtifact({
    campaign: launchedCampaign,
    childValidator,
    verifySources,
  });
  if (
    childInspection.state === "complete_valid" &&
    (childInspection.result.startedAt <
      requireUtcTimestamp(launch.launch_committed_at_utc, "campaign launch time") ||
      childInspection.result.completedAt >
        requireUtcTimestamp(processExit.observed_at_utc, "campaign process observation time"))
  ) {
    throw new Error("child artifact time interval escapes the observed child process interval");
  }
  const childTerminalStdoutValid = validateChildTerminalStdout(
    stdoutPath,
    childInspection,
    launchedCampaign,
  );
  const anchorInventory = anchorInventoryFromStoredReceipts(
    launchedCampaign,
    {
      value: launchAnchorReceipt,
      rawSha256: launchAnchorWritten.rawSha256,
    },
    delta,
  );
  const evaluation = evaluateCampaign({
    protocol: campaign.protocol,
    campaign: launchedCampaign,
    processObservation,
    childInspection,
    childTerminalStdoutValid,
    delta,
    anchorInventory,
  });
  const report = buildCampaignReportPayload({
    campaign: launchedCampaign,
    previousReceiptSha256: deltaWritten.rawSha256,
    processObservation,
    childInspection,
    childTerminalStdoutValid,
    delta,
    anchorInventory,
    evaluation,
  });
  const reportWritten = writeCreateJson(
    path.join(campaign.root, RECEIPT_FILES.campaignReport),
    report,
  );
  const records = manifestPayloadRecords(campaign.root);
  assertExpectedPreManifestInventory(records, campaign.protocol, childInspection.state);
  const manifestCore = {
    ...chainBase({
      schemaVersion: "windows-cuda-strict-32k-host-campaign-manifest.v1",
      sequence: 9,
      scopeId: campaign.scopeId,
      previousReceiptSha256: reportWritten.rawSha256,
    }),
    lease_id: campaign.leaseId,
    outer_protocol_id: campaign.protocol.protocolId,
    child_protocol_id: campaign.childProtocol.protocolId,
    host_qualification_artifact_id: campaign.qualification.payload.artifact_id,
    execution_backend_id: campaign.scopeLoaded.value.execution_backend_id,
    real_execution_observation_authorized:
      campaign.scopeLoaded.value.real_execution_observation_authorized,
    passed: evaluation.passed,
    verdict: evaluation.verdict,
    failure_codes: evaluation.failure_codes,
    files: records,
    evidence_firewall: OUTER_EVIDENCE_FIREWALL,
    claim_boundary: OUTER_CLAIM_BOUNDARY,
  };
  const manifestId = sha256Bytes(canonicalBytes(manifestCore));
  const manifestWritten = writeCreateJson(
    path.join(campaign.root, RECEIPT_FILES.manifest),
    { ...manifestCore, manifest_id: manifestId },
  );
  const terminal = {
    ...chainBase({
      schemaVersion: "windows-cuda-strict-32k-host-campaign-terminal.v1",
      sequence: 10,
      scopeId: campaign.scopeId,
      previousReceiptSha256: manifestWritten.rawSha256,
    }),
    lease_id: campaign.leaseId,
    manifest_id: manifestId,
    execution_backend_id: campaign.scopeLoaded.value.execution_backend_id,
    real_execution_observation_authorized:
      campaign.scopeLoaded.value.real_execution_observation_authorized,
    passed: evaluation.passed,
    verdict: evaluation.verdict,
    failure_codes: evaluation.failure_codes,
    retry_permitted: false,
    completed_at_utc: nowUtc(),
  };
  const terminalWritten = writeCreateJson(
    path.join(campaign.root, RECEIPT_FILES.terminal),
    terminal,
  );
  const terminalMessage = anchorMessage({
    protocol: campaign.protocol,
    scopeId: campaign.scopeId,
    kind: "terminal",
    sequence: 11,
    receiptSha256: terminalWritten.rawSha256,
    leaseId: campaign.leaseId,
  });
  const terminalAnchorReceipt = buildEventAnchorReceipt({
    protocol: campaign.protocol,
    scopeId: campaign.scopeId,
    sequence: 11,
    kind: "terminal",
    previousReceiptSha256: terminalWritten.rawSha256,
    leaseId: campaign.leaseId,
    observation: eventLogCollector.writeAnchor({
      protocol: campaign.protocol,
      message: terminalMessage,
    }),
  });
  const terminalAnchorWritten = writeCreateJson(
    path.join(campaign.root, RECEIPT_FILES.terminalAnchor),
    terminalAnchorReceipt,
  );
  const seal = {
    ...chainBase({
      schemaVersion: "windows-cuda-strict-32k-host-campaign-seal.v1",
      sequence: 12,
      scopeId: campaign.scopeId,
      previousReceiptSha256: terminalAnchorWritten.rawSha256,
    }),
    lease_id: campaign.leaseId,
    manifest_id: manifestId,
    terminal_receipt_sha256: terminalWritten.rawSha256,
    terminal_event_anchor_sha256: terminalAnchorWritten.rawSha256,
    execution_backend_id: campaign.scopeLoaded.value.execution_backend_id,
    real_execution_observation_authorized:
      campaign.scopeLoaded.value.real_execution_observation_authorized,
    passed: evaluation.passed,
    verdict: evaluation.verdict,
    failure_codes: evaluation.failure_codes,
    sealed_at_utc: nowUtc(),
    evidence_firewall: OUTER_EVIDENCE_FIREWALL,
    claim_boundary: OUTER_CLAIM_BOUNDARY,
  };
  const sealWritten = writeCreateJson(path.join(campaign.root, RECEIPT_FILES.seal), seal);
  return Object.freeze({
    status: "complete",
    campaignArtifactId: sealWritten.rawSha256,
    scopeId: campaign.scopeId,
    leaseId: campaign.leaseId,
    protocolId: campaign.protocol.protocolId,
    childProtocolId: campaign.childProtocol.protocolId,
    executionBackendId: campaign.scopeLoaded.value.execution_backend_id,
    realExecutionObservationAuthorized:
      campaign.scopeLoaded.value.real_execution_observation_authorized,
    passed: evaluation.passed,
    verdict: evaluation.verdict,
    failureCodes: evaluation.failure_codes,
    campaignRoot: campaign.root,
  });
}

function validateLaunchReceipt(receipt, campaign, previousReceiptSha256) {
  exactKeys(
    receipt,
    [
      "schema_version",
      "sequence",
      "scope_id",
      "previous_receipt_sha256",
      "lease_id",
      "attempt_ordinal",
      "launch_commit_semantics",
      "parent_process_id",
      "python_executable",
      "argv",
      "environment_overrides",
      "environment_removed",
      "prelaunch",
      "child_output_relative_path",
      "launch_committed_at_utc",
    ],
    "campaign launch receipt",
  );
  validateReceiptChainBase(receipt, {
    schemaVersion: "windows-cuda-strict-32k-host-campaign-launch.v1",
    sequence: 4,
    scopeId: campaign.scopeId,
    previousReceiptSha256,
    label: "campaign launch receipt",
  });
  if (
    receipt.lease_id !== campaign.leaseId ||
    requireInteger(receipt.attempt_ordinal, "campaign launch.attempt_ordinal") !== 1 ||
    receipt.launch_commit_semantics !== "create_new_file_fsync_before_process_creation" ||
    requireInteger(receipt.parent_process_id, "campaign launch.parent_process_id") <= 0 ||
    receipt.child_output_relative_path !== campaign.protocol.payload.child.output_relative_path
  ) {
    throw new Error("campaign launch receipt contract drift");
  }
  deepExact(
    receipt.python_executable,
    campaign.preregLoaded.value.python_executable,
    "campaign launch.python_executable",
  );
  validateFixedChildArgv(receipt.argv, {
    protocol: campaign.protocol,
    preregistration: campaign.preregLoaded.value,
    leaseId: campaign.leaseId,
    campaignRoot: campaign.root,
    label: "campaign launch.argv",
  });
  deepExact(
    receipt.environment_overrides,
    campaign.preregLoaded.value.environment_overrides,
    "campaign launch.environment_overrides",
  );
  deepExact(
    receipt.environment_removed,
    campaign.preregLoaded.value.environment_removed,
    "campaign launch.environment_removed",
  );
  const prelaunch = normalizePrelaunchCollectorPayload(receipt.prelaunch, {
    protocol: campaign.protocol,
    baseline: campaign.baselineLoaded.value,
    qualification: campaign.qualification,
  });
  deepExact(receipt.prelaunch, prelaunch, "campaign launch.prelaunch");
  const launchTime = requireUtcTimestamp(
    receipt.launch_committed_at_utc,
    "campaign launch.launch_committed_at_utc",
  );
  const preregTime = requireUtcTimestamp(
    campaign.preregLoaded.value.preregistered_at_utc,
    "campaign preregistration.preregistered_at_utc",
  );
  if (launchTime < preregTime) throw new Error("campaign launch predates preregistration");
  if (
    launchTime <
    requireUtcTimestamp(
      campaign.anchorLoaded.value.observation.time_created_utc,
      "preregistration Event Log anchor time",
    )
  ) {
    throw new Error("campaign launch predates preregistration Event Log anchor");
  }
  if (
    launchTime <
    requireUtcTimestamp(
      prelaunch.collection_completed_at_utc,
      "campaign launch.prelaunch.collection_completed_at_utc",
    )
  ) {
    throw new Error("campaign launch predates the prelaunch safety observation");
  }
  return launchTime;
}

function validateProcessExitReceipt(receipt, {
  campaign,
  previousReceiptSha256,
  launchTime,
}) {
  exactKeys(
    receipt,
    [
      "schema_version",
      "sequence",
      "scope_id",
      "previous_receipt_sha256",
      "lease_id",
      "argv",
      "environment_overrides",
      "environment_removed",
      "process_started",
      "process_id",
      "exit_code",
      "signal",
      "error_code",
      "timed_out",
      "duration_milliseconds",
      "stdout",
      "stderr",
      "observed_at_utc",
    ],
    "campaign process-exit receipt",
  );
  validateReceiptChainBase(receipt, {
    schemaVersion: "windows-cuda-strict-32k-host-campaign-process-exit.v1",
    sequence: 6,
    scopeId: campaign.scopeId,
    previousReceiptSha256,
    label: "campaign process-exit receipt",
  });
  if (receipt.lease_id !== campaign.leaseId) throw new Error("campaign process-exit lease drift");
  validateFixedChildArgv(receipt.argv, {
    protocol: campaign.protocol,
    preregistration: campaign.preregLoaded.value,
    leaseId: campaign.leaseId,
    campaignRoot: campaign.root,
    label: "campaign process-exit.argv",
  });
  deepExact(
    receipt.environment_overrides,
    campaign.preregLoaded.value.environment_overrides,
    "campaign process-exit.environment_overrides",
  );
  deepExact(
    receipt.environment_removed,
    campaign.preregLoaded.value.environment_removed,
    "campaign process-exit.environment_removed",
  );
  const observation = normalizeProcessObservation({
    process_started: receipt.process_started,
    process_id: receipt.process_id,
    exit_code: receipt.exit_code,
    signal: receipt.signal,
    error_code: receipt.error_code,
    timed_out: receipt.timed_out,
    duration_milliseconds: receipt.duration_milliseconds,
  });
  for (const stream of ["stdout", "stderr"]) {
    const streamValue = requireObject(receipt[stream], `campaign process-exit.${stream}`);
    exactKeys(streamValue, ["byte_count", "sha256"], `campaign process-exit.${stream}`);
    if (requireInteger(streamValue.byte_count, `${stream}.byte_count`) < 0) {
      throw new Error(`${stream} byte count must be non-negative`);
    }
    requireSha256(streamValue.sha256, `${stream}.sha256`);
    const actual = readLogObservation(
      path.join(campaign.root, "streams", `child.${stream}.log`),
    );
    deepExact(streamValue, actual, `campaign process-exit.${stream}`);
  }
  const observedAt = requireUtcTimestamp(
    receipt.observed_at_utc,
    "campaign process-exit.observed_at_utc",
  );
  if (observedAt < launchTime) throw new Error("campaign process-exit predates launch");
  return { observation, observedAt };
}

const CONTINUITY_CHECK_KEYS = Object.freeze([
  "same_machine",
  "same_boot",
  "application_boundary_present",
  "application_boundary_hash_exact",
  "application_cursor_monotonic",
  "application_no_rollover_gap",
  "application_range_complete",
  "system_boundary_present",
  "system_boundary_hash_exact",
  "system_cursor_monotonic",
  "system_no_rollover_gap",
  "system_range_complete",
]);

function validateStoredDeltaReceipt(receipt, {
  campaign,
  previousReceiptSha256,
  processObservedAt,
}) {
  exactKeys(
    receipt,
    [
      "schema_version",
      "sequence",
      "scope_id",
      "previous_receipt_sha256",
      "lease_id",
      "collection_status",
      "collector_error_name",
      "collector_error_sha256",
      "collection_started_at_utc",
      "collection_completed_at_utc",
      "host",
      "channels",
      "classified_faults",
      "continuity_checks",
    ],
    "campaign Event Log delta receipt",
  );
  validateReceiptChainBase(receipt, {
    schemaVersion: "windows-cuda-strict-32k-host-campaign-event-log-delta.v1",
    sequence: 7,
    scopeId: campaign.scopeId,
    previousReceiptSha256,
    label: "campaign Event Log delta receipt",
  });
  if (receipt.lease_id !== campaign.leaseId) throw new Error("campaign Event Log delta lease drift");
  exactKeys(receipt.continuity_checks, CONTINUITY_CHECK_KEYS, "campaign delta.continuity_checks");
  for (const key of CONTINUITY_CHECK_KEYS) {
    requireBoolean(receipt.continuity_checks[key], `campaign delta.continuity_checks.${key}`);
  }
  if (receipt.collection_status === "failed") {
    requireText(receipt.collector_error_name, "campaign delta.collector_error_name");
    requireSha256(receipt.collector_error_sha256, "campaign delta.collector_error_sha256");
    if (
      receipt.collection_started_at_utc !== null ||
      receipt.collection_completed_at_utc !== null ||
      receipt.host !== null ||
      requireArray(receipt.channels, "campaign delta.channels").length !== 0 ||
      requireArray(receipt.classified_faults, "campaign delta.classified_faults").length !== 0 ||
      CONTINUITY_CHECK_KEYS.some((key) => receipt.continuity_checks[key] !== false)
    ) {
      throw new Error("failed Event Log delta must carry a closed negative observation");
    }
    return receipt;
  }
  if (receipt.collection_status !== "complete") {
    throw new Error("campaign Event Log delta collection status drift");
  }
  if (receipt.collector_error_name !== null || receipt.collector_error_sha256 !== null) {
    throw new Error("complete Event Log delta carries a collector error");
  }
  const normalized = normalizeDeltaCollectorPayload(
    {
      schema_version: "windows-host-event-log-delta-collector.v1",
      collection_started_at_utc: receipt.collection_started_at_utc,
      collection_completed_at_utc: receipt.collection_completed_at_utc,
      host: receipt.host,
      channels: receipt.channels,
    },
    { protocol: campaign.protocol, baseline: campaign.baselineLoaded.value },
  );
  deepExact(receipt.classified_faults, normalized.classified_faults, "campaign delta.classified_faults");
  deepExact(receipt.continuity_checks, normalized.continuity_checks, "campaign delta.continuity_checks");
  if (
    requireUtcTimestamp(
      receipt.collection_started_at_utc,
      "campaign delta.collection_started_at_utc",
    ) < processObservedAt
  ) {
    throw new Error("Event Log delta collection predates child process observation");
  }
  return {
    ...receipt,
    host: normalized.host,
    channels: normalized.channels,
    classified_faults: normalized.classified_faults,
    continuity_checks: normalized.continuity_checks,
  };
}

function anchorInventoryFromStoredReceipts(campaign, launchAnchor, delta) {
  if (delta.collection_status !== "complete") return [];
  const application = delta.channels.find((channel) => channel.log_name === "Application");
  if (!application) throw new Error("complete Event Log delta is missing Application channel");
  const eventIds = new Set(
    Object.values(campaign.protocol.payload.event_log.anchor_event_ids).map((value) =>
      requireInteger(value, "protocol anchor event ID"),
    ),
  );
  const inventory = [];
  for (const event of application.events) {
    if (
      event.provider_name !== campaign.protocol.payload.event_log.application_anchor_source ||
      !eventIds.has(event.event_id)
    ) {
      continue;
    }
    if (
      event.payload_kind !== "event_data" ||
      event.event_data.length !== 1 ||
      event.event_data[0].name !== ""
    ) {
      throw new Error("Event Log delta contains a malformed Volvence anchor payload shape");
    }
    const payload = Buffer.from(event.event_data[0].value, "utf8");
    const message = parseJsonStrict(payload.toString("utf8"), "Event Log delta anchor payload");
    exactKeys(
      message,
      [
        "schema_version",
        "outer_protocol_id",
        "scope_id",
        "anchor_kind",
        "sequence",
        "receipt_sha256",
        "lease_id",
      ],
      "Event Log delta anchor payload",
    );
    if (!payload.equals(canonicalBytes(message, false))) {
      throw new Error("Event Log delta anchor payload is not canonical JSON");
    }
    if (message.schema_version !== "volvence-local-event-anchor.v1") {
      throw new Error("Event Log delta anchor schema drift");
    }
    if (
      message.outer_protocol_id !== campaign.protocol.protocolId ||
      message.scope_id !== campaign.scopeId
    ) {
      continue;
    }
    if (!new Set(["preregistration", "launch", "terminal"]).has(message.anchor_kind)) {
      throw new Error("Event Log delta anchor kind drift");
    }
    if (
      event.event_id !==
      requireInteger(
        campaign.protocol.payload.event_log.anchor_event_ids[message.anchor_kind],
        "Event Log delta expected anchor event ID",
      )
    ) {
      throw new Error("Event Log delta anchor event-ID/kind drift");
    }
    requireInteger(message.sequence, "Event Log delta anchor sequence");
    requireSha256(message.receipt_sha256, "Event Log delta anchor receipt SHA-256");
    requireSha256(message.lease_id, "Event Log delta anchor lease ID");
    inventory.push({
      log_name: event.log_name,
      provider_name: event.provider_name,
      event_id: event.event_id,
      record_id: event.record_id,
      time_created_utc: event.time_created_utc,
      xml_sha256: event.xml_sha256,
      payload_base64: payload.toString("base64"),
      message,
    });
  }
  const expectedStored = [
    {
      observation: campaign.anchorLoaded.value.observation,
      message: campaign.anchorMessage,
    },
    {
      observation: launchAnchor.value.observation,
      message: anchorMessage({
        protocol: campaign.protocol,
        scopeId: campaign.scopeId,
        kind: "launch",
        sequence: 5,
        receiptSha256: campaign.launchLoaded.rawSha256,
        leaseId: campaign.leaseId,
      }),
    },
  ];
  for (const expected of expectedStored) {
    const matches = inventory.filter(
      (anchor) =>
        anchor.record_id ===
          requireInteger(expected.observation.record_id, "stored anchor record ID") &&
        anchor.message.anchor_kind === expected.message.anchor_kind,
    );
    if (matches.length !== 1) {
      throw new Error("stored Event Log anchor is not cross-bound to the exact delta interval");
    }
    const actual = matches[0];
    deepExact(actual.message, expected.message, "stored/delta Event Log anchor message");
    for (const key of [
      "log_name",
      "provider_name",
      "event_id",
      "record_id",
      "time_created_utc",
      "xml_sha256",
      "payload_base64",
    ]) {
      deepExact(actual[key], expected.observation[key], `stored/delta Event Log anchor.${key}`);
    }
  }
  return inventory;
}

function assertNoUnexpectedDirectories(root, protocol) {
  const actual = [];
  function visit(current, relative) {
    const stat = fs.lstatSync(current);
    if (stat.isSymbolicLink()) throw new Error(`campaign tree contains linked directory: ${relative}`);
    if (!stat.isDirectory()) return;
    if (relative) actual.push(relative);
    for (const name of fs.readdirSync(current).sort()) {
      const child = path.join(current, name);
      if (fs.lstatSync(child).isDirectory()) visit(child, relative ? `${relative}/${name}` : name);
    }
  }
  visit(root, "");
  const childParts = protocol.payload.child.output_relative_path.split("/");
  const allowed = new Set(["streams"]);
  let prefix = "";
  for (const part of childParts) {
    prefix = prefix ? `${prefix}/${part}` : part;
    allowed.add(prefix);
  }
  for (const directory of actual) {
    if (!allowed.has(directory)) throw new Error(`unexpected campaign artifact directory: ${directory}`);
  }
}

function validateManifestReceipt(manifest, {
  campaign,
  previousReceiptSha256,
  evaluation,
  childState,
  recomputedRecords,
}) {
  exactKeys(
    manifest,
    [
      "schema_version",
      "sequence",
      "scope_id",
      "previous_receipt_sha256",
      "lease_id",
      "outer_protocol_id",
      "child_protocol_id",
      "host_qualification_artifact_id",
      "execution_backend_id",
      "real_execution_observation_authorized",
      "passed",
      "verdict",
      "failure_codes",
      "files",
      "evidence_firewall",
      "claim_boundary",
      "manifest_id",
    ],
    "campaign manifest",
  );
  validateReceiptChainBase(manifest, {
    schemaVersion: "windows-cuda-strict-32k-host-campaign-manifest.v1",
    sequence: 9,
    scopeId: campaign.scopeId,
    previousReceiptSha256,
    label: "campaign manifest",
  });
  if (
    manifest.lease_id !== campaign.leaseId ||
    manifest.outer_protocol_id !== campaign.protocol.protocolId ||
    manifest.child_protocol_id !== campaign.childProtocol.protocolId ||
    manifest.host_qualification_artifact_id !== campaign.qualification.payload.artifact_id ||
    manifest.execution_backend_id !== campaign.scopeLoaded.value.execution_backend_id ||
    manifest.real_execution_observation_authorized !==
      campaign.scopeLoaded.value.real_execution_observation_authorized ||
    manifest.passed !== evaluation.passed ||
    manifest.verdict !== evaluation.verdict
  ) {
    throw new Error("campaign manifest lineage drift");
  }
  deepExact(manifest.failure_codes, evaluation.failure_codes, "campaign manifest.failure_codes");
  assertExpectedPreManifestInventory(recomputedRecords, campaign.protocol, childState);
  deepExact(manifest.files, recomputedRecords, "campaign manifest.files");
  deepExact(manifest.evidence_firewall, OUTER_EVIDENCE_FIREWALL, "campaign manifest.evidence_firewall");
  if (manifest.claim_boundary !== OUTER_CLAIM_BOUNDARY) throw new Error("campaign manifest boundary drift");
  const manifestId = requireSha256(manifest.manifest_id, "campaign manifest.manifest_id");
  const core = { ...manifest };
  delete core.manifest_id;
  if (manifestId !== sha256Bytes(canonicalBytes(core))) throw new Error("campaign manifest ID drift");
  return manifestId;
}

function validateHostCampaignCore({
  campaignRoot,
  protocolPath = DEFAULT_PROTOCOL_PATH,
  repositoryRoot = REPOSITORY_ROOT,
  childValidator,
  verifySources = true,
  allowSyntheticTestBackend = false,
} = {}) {
  const root = path.resolve(campaignRoot);
  const campaign = loadPreregisteredCampaign({
    campaignRoot: root,
    protocolPath,
    repositoryRoot,
    verifySources,
    requireExactTopLevel: false,
    allowSyntheticTestBackend,
  });
  const isSynthetic =
    campaign.scopeLoaded.value.execution_backend_id === SYNTHETIC_TEST_BACKEND_ID;
  if (allowSyntheticTestBackend !== isSynthetic) {
    throw new Error("campaign execution backend is not authorized by this validator");
  }
  if (
    !allowSyntheticTestBackend &&
    (childValidator !== validateStrict32KChildArtifact || verifySources !== true)
  ) {
    throw new Error("production validation requires the bundled verified validator");
  }
  if (
    !allowSyntheticTestBackend &&
    campaign.protocol.payload.host_qualification.production_preregistration_enabled !== true
  ) {
    throw new Error("this protocol does not authorize production campaign artifacts");
  }
  const topLevelEntries = fs.readdirSync(root).sort();
  if (!topLevelEntries.includes(RECEIPT_FILES.launch)) {
    ensureExactTopLevelEntries(root, PREREGISTERED_FILES, "preregistered campaign");
    return Object.freeze({
      status: "preregistered",
      campaignArtifactId: null,
      scopeId: campaign.scopeId,
      leaseId: campaign.leaseId,
      protocolId: campaign.protocol.protocolId,
      childProtocolId: campaign.childProtocol.protocolId,
      executionBackendId: campaign.scopeLoaded.value.execution_backend_id,
      realExecutionObservationAuthorized:
        campaign.scopeLoaded.value.real_execution_observation_authorized,
      passed: false,
      verdict: "preregistered_not_launched",
      failureCodes: [],
      campaignRoot: root,
    });
  }
  const launchLoaded = loadStrictJsonFile(
    path.join(root, RECEIPT_FILES.launch),
    "campaign launch receipt",
    true,
  );
  const launchTime = validateLaunchReceipt(
    launchLoaded.value,
    campaign,
    campaign.anchorLoaded.rawSha256,
  );
  const launchedCampaign = Object.freeze({ ...campaign, launchLoaded });
  if (!topLevelEntries.includes(RECEIPT_FILES.seal)) {
    listRegularFiles(root);
    return Object.freeze({
      status: "incomplete_consumed",
      campaignArtifactId: null,
      scopeId: campaign.scopeId,
      leaseId: campaign.leaseId,
      protocolId: campaign.protocol.protocolId,
      childProtocolId: campaign.childProtocol.protocolId,
      executionBackendId: campaign.scopeLoaded.value.execution_backend_id,
      realExecutionObservationAuthorized:
        campaign.scopeLoaded.value.real_execution_observation_authorized,
      passed: false,
      verdict: "incomplete_consumed_stop_no_retry",
      failureCodes: [],
      campaignRoot: root,
    });
  }
  assertNoUnexpectedDirectories(root, campaign.protocol);
  const launchAnchorLoaded = loadStrictJsonFile(
    path.join(root, RECEIPT_FILES.launchAnchor),
    "launch Event Log anchor",
    true,
  );
  validateEventAnchorReceipt(launchAnchorLoaded.value, {
    protocol: campaign.protocol,
    scopeId: campaign.scopeId,
    sequence: 5,
    kind: "launch",
    previousReceiptSha256: launchLoaded.rawSha256,
    leaseId: campaign.leaseId,
    label: "launch Event Log anchor",
  });
  const launchAnchorTime = requireUtcTimestamp(
    launchAnchorLoaded.value.observation.time_created_utc,
    "launch Event Log anchor time",
  );
  if (
    launchAnchorTime < launchTime ||
    requireInteger(launchAnchorLoaded.value.observation.record_id, "launch anchor record ID") <=
      requireInteger(
        campaign.anchorLoaded.value.observation.record_id,
        "preregistration anchor record ID",
      )
  ) {
    throw new Error("launch Event Log anchor order drift");
  }
  const processLoaded = loadStrictJsonFile(
    path.join(root, RECEIPT_FILES.processExit),
    "campaign process-exit receipt",
    true,
  );
  const processValidation = validateProcessExitReceipt(processLoaded.value, {
    campaign: launchedCampaign,
    previousReceiptSha256: launchAnchorLoaded.rawSha256,
    launchTime,
  });
  if (processValidation.observedAt < launchAnchorTime) {
    throw new Error("child process observation predates launch Event Log anchor");
  }
  const deltaLoaded = loadStrictJsonFile(
    path.join(root, RECEIPT_FILES.eventLogDelta),
    "campaign Event Log delta receipt",
    true,
  );
  const delta = validateStoredDeltaReceipt(deltaLoaded.value, {
    campaign: launchedCampaign,
    previousReceiptSha256: processLoaded.rawSha256,
    processObservedAt: processValidation.observedAt,
  });
  const childInspection = inspectChildArtifact({
    campaign: launchedCampaign,
    childValidator,
    verifySources,
  });
  const childTerminalStdoutValid = validateChildTerminalStdout(
    path.join(root, "streams", "child.stdout.log"),
    childInspection,
    campaign,
  );
  if (
    childInspection.state === "complete_valid" &&
    (childInspection.result.startedAt < launchTime ||
      childInspection.result.completedAt > processValidation.observedAt)
  ) {
    throw new Error("child artifact time interval escapes the outer process observation");
  }
  const anchorInventory = anchorInventoryFromStoredReceipts(
    launchedCampaign,
    launchAnchorLoaded,
    delta,
  );
  const evaluation = evaluateCampaign({
    protocol: campaign.protocol,
    campaign: launchedCampaign,
    processObservation: processValidation.observation,
    childInspection,
    childTerminalStdoutValid,
    delta,
    anchorInventory,
  });
  const reportLoaded = loadStrictJsonFile(
    path.join(root, RECEIPT_FILES.campaignReport),
    "campaign report",
    true,
  );
  const expectedReport = buildCampaignReportPayload({
    campaign: launchedCampaign,
    previousReceiptSha256: deltaLoaded.rawSha256,
    processObservation: processValidation.observation,
    childInspection,
    childTerminalStdoutValid,
    delta,
    anchorInventory,
    evaluation,
  });
  deepExact(reportLoaded.value, expectedReport, "campaign report");
  const recomputedRecords = manifestPayloadRecords(root);
  const manifestLoaded = loadStrictJsonFile(
    path.join(root, RECEIPT_FILES.manifest),
    "campaign manifest",
    true,
  );
  const manifestId = validateManifestReceipt(manifestLoaded.value, {
    campaign: launchedCampaign,
    previousReceiptSha256: reportLoaded.rawSha256,
    evaluation,
    childState: childInspection.state,
    recomputedRecords,
  });
  const terminalLoaded = loadStrictJsonFile(
    path.join(root, RECEIPT_FILES.terminal),
    "campaign terminal receipt",
    true,
  );
  const terminal = terminalLoaded.value;
  exactKeys(
    terminal,
    [
      "schema_version",
      "sequence",
      "scope_id",
      "previous_receipt_sha256",
      "lease_id",
      "manifest_id",
      "execution_backend_id",
      "real_execution_observation_authorized",
      "passed",
      "verdict",
      "failure_codes",
      "retry_permitted",
      "completed_at_utc",
    ],
    "campaign terminal receipt",
  );
  validateReceiptChainBase(terminal, {
    schemaVersion: "windows-cuda-strict-32k-host-campaign-terminal.v1",
    sequence: 10,
    scopeId: campaign.scopeId,
    previousReceiptSha256: manifestLoaded.rawSha256,
    label: "campaign terminal receipt",
  });
  if (
    terminal.lease_id !== campaign.leaseId ||
    terminal.manifest_id !== manifestId ||
    terminal.execution_backend_id !== campaign.scopeLoaded.value.execution_backend_id ||
    terminal.real_execution_observation_authorized !==
      campaign.scopeLoaded.value.real_execution_observation_authorized ||
    terminal.passed !== evaluation.passed ||
    terminal.verdict !== evaluation.verdict ||
    terminal.retry_permitted !== false
  ) {
    throw new Error("campaign terminal receipt lineage drift");
  }
  deepExact(terminal.failure_codes, evaluation.failure_codes, "campaign terminal.failure_codes");
  const terminalTime = requireUtcTimestamp(
    terminal.completed_at_utc,
    "campaign terminal.completed_at_utc",
  );
  if (terminalTime < processValidation.observedAt) {
    throw new Error("campaign terminal predates child process observation");
  }
  if (
    delta.collection_status === "complete" &&
    terminalTime <
      requireUtcTimestamp(
        delta.collection_completed_at_utc,
        "campaign Event Log delta completion",
      )
  ) {
    throw new Error("campaign terminal predates Event Log delta completion");
  }
  const terminalAnchorLoaded = loadStrictJsonFile(
    path.join(root, RECEIPT_FILES.terminalAnchor),
    "terminal Event Log anchor",
    true,
  );
  validateEventAnchorReceipt(terminalAnchorLoaded.value, {
    protocol: campaign.protocol,
    scopeId: campaign.scopeId,
    sequence: 11,
    kind: "terminal",
    previousReceiptSha256: terminalLoaded.rawSha256,
    leaseId: campaign.leaseId,
    label: "terminal Event Log anchor",
  });
  if (
    requireUtcTimestamp(
      terminalAnchorLoaded.value.observation.time_created_utc,
      "terminal Event Log anchor time",
    ) < terminalTime
  ) {
    throw new Error("terminal Event Log anchor predates terminal receipt");
  }
  if (
    requireInteger(
      terminalAnchorLoaded.value.observation.record_id,
      "terminal Event Log anchor record ID",
    ) <=
    requireInteger(
      launchAnchorLoaded.value.observation.record_id,
      "launch Event Log anchor record ID",
    )
  ) {
    throw new Error("terminal Event Log anchor RecordID does not follow launch anchor");
  }
  const sealLoaded = loadStrictJsonFile(
    path.join(root, RECEIPT_FILES.seal),
    "campaign seal",
    true,
  );
  const seal = sealLoaded.value;
  exactKeys(
    seal,
    [
      "schema_version",
      "sequence",
      "scope_id",
      "previous_receipt_sha256",
      "lease_id",
      "manifest_id",
      "terminal_receipt_sha256",
      "terminal_event_anchor_sha256",
      "execution_backend_id",
      "real_execution_observation_authorized",
      "passed",
      "verdict",
      "failure_codes",
      "sealed_at_utc",
      "evidence_firewall",
      "claim_boundary",
    ],
    "campaign seal",
  );
  validateReceiptChainBase(seal, {
    schemaVersion: "windows-cuda-strict-32k-host-campaign-seal.v1",
    sequence: 12,
    scopeId: campaign.scopeId,
    previousReceiptSha256: terminalAnchorLoaded.rawSha256,
    label: "campaign seal",
  });
  if (
    seal.lease_id !== campaign.leaseId ||
    seal.manifest_id !== manifestId ||
    seal.terminal_receipt_sha256 !== terminalLoaded.rawSha256 ||
    seal.terminal_event_anchor_sha256 !== terminalAnchorLoaded.rawSha256 ||
    seal.execution_backend_id !== campaign.scopeLoaded.value.execution_backend_id ||
    seal.real_execution_observation_authorized !==
      campaign.scopeLoaded.value.real_execution_observation_authorized ||
    seal.passed !== evaluation.passed ||
    seal.verdict !== evaluation.verdict
  ) {
    throw new Error("campaign seal lineage drift");
  }
  deepExact(seal.failure_codes, evaluation.failure_codes, "campaign seal.failure_codes");
  deepExact(seal.evidence_firewall, OUTER_EVIDENCE_FIREWALL, "campaign seal.evidence_firewall");
  if (seal.claim_boundary !== OUTER_CLAIM_BOUNDARY) throw new Error("campaign seal boundary drift");
  const sealTime = requireUtcTimestamp(seal.sealed_at_utc, "campaign seal.sealed_at_utc");
  if (sealTime < terminalTime) throw new Error("campaign seal predates terminal receipt");
  if (
    sealTime <
    requireUtcTimestamp(
      terminalAnchorLoaded.value.observation.time_created_utc,
      "terminal Event Log anchor time",
    )
  ) {
    throw new Error("campaign seal predates terminal Event Log anchor");
  }
  const allRecords = listRegularFiles(root);
  const expectedPaths = new Set([
    ...recomputedRecords.map((record) => record.path),
    ...POST_MANIFEST_FILES,
  ]);
  if (
    allRecords.length !== expectedPaths.size ||
    allRecords.some((record) => !expectedPaths.has(record.path))
  ) {
    throw new Error("complete campaign root file set drift");
  }
  return Object.freeze({
    status: "complete",
    campaignArtifactId: sealLoaded.rawSha256,
    scopeId: campaign.scopeId,
    leaseId: campaign.leaseId,
    protocolId: campaign.protocol.protocolId,
    childProtocolId: campaign.childProtocol.protocolId,
    executionBackendId: campaign.scopeLoaded.value.execution_backend_id,
    realExecutionObservationAuthorized:
      campaign.scopeLoaded.value.real_execution_observation_authorized,
    passed: evaluation.passed,
    verdict: evaluation.verdict,
    failureCodes: evaluation.failure_codes,
    campaignRoot: root,
  });
}

function assertAllowedOptionKeys(options, allowed, label) {
  const value = requireObject(options, label);
  const allowedSet = new Set(allowed);
  for (const key of Object.keys(value)) {
    if (!allowedSet.has(key)) throw new TypeError(`${label} does not accept option ${key}`);
  }
  return value;
}

export function preregisterHostCampaign(options = {}) {
  const value = assertAllowedOptionKeys(
    options,
    ["hostQualificationTerminalPath", "pythonExecutable"],
    "production preregistration options",
  );
  return preregisterHostCampaignCore({
    hostQualificationTerminalPath: value.hostQualificationTerminalPath,
    pythonExecutable: value.pythonExecutable,
    campaignBaseDir: DEFAULT_CAMPAIGN_BASE_DIR,
    protocolPath: DEFAULT_PROTOCOL_PATH,
    repositoryRoot: REPOSITORY_ROOT,
    eventLogCollector: defaultWindowsEventLogCollector,
    verifySources: true,
    executionBackendId: PRODUCTION_BACKEND_ID,
    realExecutionObservationAuthorized: true,
    allowUnvalidatedQualificationForTesting: false,
  });
}

export function runHostCampaign(options = {}) {
  const value = assertAllowedOptionKeys(
    options,
    ["campaignRoot"],
    "production campaign-run options",
  );
  return runHostCampaignCore({
    campaignRoot: value.campaignRoot,
    protocolPath: DEFAULT_PROTOCOL_PATH,
    repositoryRoot: REPOSITORY_ROOT,
    eventLogCollector: defaultWindowsEventLogCollector,
    childExecutor: defaultChildExecutor,
    childValidator: validateStrict32KChildArtifact,
    verifySources: true,
    allowSyntheticTestBackend: false,
  });
}

export function validateHostCampaign(options = {}) {
  const value = assertAllowedOptionKeys(
    options,
    ["campaignRoot"],
    "production campaign-validation options",
  );
  return validateHostCampaignCore({
    campaignRoot: value.campaignRoot,
    protocolPath: DEFAULT_PROTOCOL_PATH,
    repositoryRoot: REPOSITORY_ROOT,
    childValidator: validateStrict32KChildArtifact,
    verifySources: true,
    allowSyntheticTestBackend: false,
  });
}

function preregisterSyntheticHostCampaign(options = {}) {
  const value = assertAllowedOptionKeys(
    options,
    [
      "hostQualificationTerminalPath",
      "pythonExecutable",
      "campaignBaseDir",
      "protocolPath",
      "repositoryRoot",
      "eventLogCollector",
      "verifySources",
    ],
    "synthetic preregistration options",
  );
  if (!value.eventLogCollector) throw new TypeError("synthetic Event Log collector is required");
  if (value.eventLogCollector === defaultWindowsEventLogCollector) {
    throw new Error("synthetic tests cannot use the bundled live Event Log collector");
  }
  if (
    !value.campaignBaseDir ||
    path.resolve(value.campaignBaseDir) === path.resolve(DEFAULT_CAMPAIGN_BASE_DIR)
  ) {
    throw new Error("synthetic tests require a non-production campaign base directory");
  }
  return preregisterHostCampaignCore({
    ...value,
    executionBackendId: SYNTHETIC_TEST_BACKEND_ID,
    realExecutionObservationAuthorized: false,
    allowUnvalidatedQualificationForTesting: true,
  });
}

function runSyntheticHostCampaign(options = {}) {
  const value = assertAllowedOptionKeys(
    options,
    [
      "campaignRoot",
      "protocolPath",
      "repositoryRoot",
      "eventLogCollector",
      "childExecutor",
      "childValidator",
      "verifySources",
    ],
    "synthetic campaign-run options",
  );
  if (!value.eventLogCollector) throw new TypeError("synthetic Event Log collector is required");
  if (!value.childExecutor) throw new TypeError("synthetic child executor is required");
  return runHostCampaignCore({
    ...value,
    childValidator: value.childValidator ?? validateStrict32KChildArtifact,
    allowSyntheticTestBackend: true,
  });
}

function validateSyntheticHostCampaign(options = {}) {
  const value = assertAllowedOptionKeys(
    options,
    ["campaignRoot", "protocolPath", "repositoryRoot", "childValidator", "verifySources"],
    "synthetic campaign-validation options",
  );
  return validateHostCampaignCore({
    ...value,
    childValidator: value.childValidator ?? validateStrict32KChildArtifact,
    allowSyntheticTestBackend: true,
  });
}

export const hostCampaignPaths = Object.freeze({
  repositoryRoot: REPOSITORY_ROOT,
  defaultProtocolPath: DEFAULT_PROTOCOL_PATH,
  defaultCampaignBaseDir: DEFAULT_CAMPAIGN_BASE_DIR,
});

export const __testing = Object.freeze({
  JsonNumber,
  canonicalBytes,
  canonicalJson,
  domainSeparatedSha256,
  parseJsonStrict,
  preregisterSyntheticHostCampaign,
  requireUtcTimestamp,
  runSyntheticHostCampaign,
  sha256Bytes,
  sourceTextSha256,
  validateSyntheticHostCampaign,
});
