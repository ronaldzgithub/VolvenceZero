/**
 * Production-disabled, standalone Windows Event Log source Audit acquisition.
 *
 * The public acquisition entrypoint is deliberately a static fail-closed gate.
 * A private reviewed backend records one fixed Windows PowerShell `Audit`
 * invocation, while the exported offline validator and declarative synthetic
 * writer exercise only capture/quarantine integrity. This owner never projects
 * source configuration, qualifies a host, or authorizes CUDA or production.
 */

import { spawn } from "node:child_process";
import crypto from "node:crypto";
import { EventEmitter } from "node:events";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { PassThrough } from "node:stream";
import { fileURLToPath } from "node:url";

export const EVENT_LOG_SOURCE_AUDIT_ACQUISITION_PROTOCOL_SCHEMA_VERSION =
  "windows-event-log-source-audit-acquisition-protocol.v2";
export const EVENT_LOG_SOURCE_AUDIT_ACQUISITION_CLAIM_SCHEMA_VERSION =
  "windows-event-log-source-audit-acquisition-claim.v2";
export const EVENT_LOG_SOURCE_AUDIT_ACQUISITION_TERMINAL_SCHEMA_VERSION =
  "windows-event-log-source-audit-acquisition-terminal.v2";

const AUDIT_SCHEMA_VERSION = "volvence-evidence-event-log-provisioning-audit.v2";
const FAILURE_SCHEMA_VERSION = "volvence-evidence-event-log-provisioning-failure.v1";
const ADAPTER_CAPTURE_ENVELOPE_SCHEMA_VERSION =
  "windows-event-log-source-audit-capture-envelope.v1";
const STREAM_CAPTURE_SCHEMA_VERSION =
  "windows-event-log-source-audit-stream-capture.v1";
const ADAPTER_CAPTURE_ENVELOPE_FIELDS = Object.freeze([
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
const STREAM_CAPTURE_FIELDS = Object.freeze([
  "schema_version",
  "stdout_relative_path",
  "stdout_raw_sha256",
  "stdout_byte_count",
  "stderr_relative_path",
  "stderr_raw_sha256",
  "stderr_byte_count",
]);
const PRODUCTION_BACKEND_ID =
  "windows-powershell-same-buffer-audit-capture.v2";
const SYNTHETIC_BACKEND_ID = "synthetic-node-audit-acquisition-declaration.v1";
const PRODUCTION_DISABLED_MESSAGE =
  "Windows Event Log source Audit acquisition is production-disabled in protocol v2";

const MODULE_PATH = fileURLToPath(import.meta.url);
const MODULE_DIR = path.dirname(MODULE_PATH);
const REPOSITORY_ROOT = path.resolve(MODULE_DIR, "../../../../../");
const PROTOCOL_PATH = path.join(
  MODULE_DIR,
  "protocols",
  "windows_event_log_source_audit_acquisition_v2.json",
);
const CLAIM_FILE = "000_scope_claim.json";
const TERMINAL_FILE = "001_terminal.json";
const STDOUT_FILE = "streams/audit.stdout.bin";
const STDERR_FILE = "streams/audit.stderr.bin";
const PROVISIONER_RELATIVE_PATH =
  "packages/vz-runtime/src/volvence_zero/offline_evidence/provision_volvence_evidence_event_log.ps1";
const SOURCE_BINDING_SCHEMA_VERSION =
  "windows-event-log-source-audit-execution-binding.v2";
const STREAM_OUTCOME_SCHEMA_VERSION =
  "windows-event-log-source-audit-stream-outcome.v1";
const PROCESS_OBSERVATION_SCHEMA_VERSION =
  "windows-event-log-source-audit-process-observation.v2";
const COMPLETE_FILES = Object.freeze([
  CLAIM_FILE,
  TERMINAL_FILE,
  STDOUT_FILE,
  STDERR_FILE,
]);

function powerShellSingleQuoted(value, label) {
  if (!/^[A-Za-z0-9_./-]+$/u.test(value)) {
    throw new Error(`${label} is not safe for the frozen PowerShell launcher`);
  }
  return `'${value}'`;
}

function buildSourceBindingLauncherSource(sourceBinding) {
  const relativePath = powerShellSingleQuoted(
    sourceBinding.provisioner_relative_path,
    "provisioner relative path",
  );
  const expectedRaw = powerShellSingleQuoted(
    sourceBinding.provisioner_raw_sha256,
    "provisioner raw SHA-256",
  );
  const expectedLf = powerShellSingleQuoted(
    sourceBinding.provisioner_lf_canonical_sha256,
    "provisioner LF SHA-256",
  );
  return `${[
    "[CmdletBinding()]",
    "param()",
    "Set-StrictMode -Version Latest",
    '$ErrorActionPreference = "Stop"',
    '$ProgressPreference = "SilentlyContinue"',
    `$sourceRelativePath = ${relativePath}`,
    `$expectedRawSha256 = ${expectedRaw}`,
    `$expectedLfSha256 = ${expectedLf}`,
    "$sourcePath = [IO.Path]::GetFullPath([IO.Path]::Combine([Environment]::CurrentDirectory, $sourceRelativePath.Replace('/', [IO.Path]::DirectorySeparatorChar)))",
    "$stream = $null",
    "try {",
    "    $stream = [IO.FileStream]::new($sourcePath, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)",
    "    if ($stream.Length -lt 1 -or $stream.Length -gt 2097152) { throw \"reviewed provisioner size is outside the frozen launcher budget\" }",
    "    $rawBytes = New-Object byte[] ([int]$stream.Length)",
    "    $offset = 0",
    "    while ($offset -lt $rawBytes.Length) {",
    "        $read = $stream.Read($rawBytes, $offset, $rawBytes.Length - $offset)",
    "        if ($read -le 0) { throw \"reviewed provisioner same-handle read ended early\" }",
    "        $offset += $read",
    "    }",
    "    if ($stream.ReadByte() -ne -1) { throw \"reviewed provisioner grew during same-handle read\" }",
    "    $sha256 = [Security.Cryptography.SHA256]::Create()",
    "    try { $rawSha256 = ([BitConverter]::ToString($sha256.ComputeHash($rawBytes))).Replace('-', '').ToLowerInvariant() } finally { $sha256.Dispose() }",
    "    if ($rawSha256 -cne $expectedRawSha256) { throw \"reviewed provisioner raw SHA-256 drift\" }",
    "    if ($rawBytes.Length -ge 3 -and $rawBytes[0] -eq 0xef -and $rawBytes[1] -eq 0xbb -and $rawBytes[2] -eq 0xbf) { throw \"reviewed provisioner UTF-8 BOM is prohibited\" }",
    "    $strictUtf8 = [Text.UTF8Encoding]::new($false, $true)",
    "    $sourceText = $strictUtf8.GetString($rawBytes)",
    '    $lfText = $sourceText.Replace("`r`n", "`n").Replace("`r", "`n")',
    "    $lfBytes = [Text.Encoding]::UTF8.GetBytes($lfText)",
    "    $sha256 = [Security.Cryptography.SHA256]::Create()",
    "    try { $lfSha256 = ([BitConverter]::ToString($sha256.ComputeHash($lfBytes))).Replace('-', '').ToLowerInvariant() } finally { $sha256.Dispose() }",
    "    if ($lfSha256 -cne $expectedLfSha256) { throw \"reviewed provisioner LF SHA-256 drift\" }",
    "    $tokens = $null",
    "    $parseErrors = $null",
    "    $ast = [Management.Automation.Language.Parser]::ParseInput($sourceText, $sourcePath, [ref]$tokens, [ref]$parseErrors)",
    "    if ($parseErrors.Count -ne 0) { throw \"reviewed provisioner parse failure\" }",
    "    $scriptBlock = $ast.GetScriptBlock()",
    "    & $scriptBlock -Mode Audit",
    "    throw \"reviewed provisioner returned without explicit process exit\"",
    "} catch {",
    '    [Console]::Error.WriteLine("Volvence fixed source-binding launcher failed: {0}", $_.Exception.GetType().FullName)',
    "    exit 3",
    "} finally {",
    "    if ($null -ne $stream) { $stream.Dispose() }",
    "}",
  ].join("\n")}\n`;
}

function buildSourceBindingLauncher(sourceBinding) {
  const source = buildSourceBindingLauncherSource(sourceBinding);
  const sourceBytes = Buffer.from(source, "utf8");
  const utf16leBytes = Buffer.from(source, "utf16le");
  return Object.freeze({
    source,
    sourceUtf8Sha256: sha256Bytes(sourceBytes),
    utf16leSha256: sha256Bytes(utf16leBytes),
    encodedCommand: utf16leBytes.toString("base64"),
  });
}

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

function materializeJsonIntegers(value) {
  if (value instanceof JsonInteger) return value.value;
  if (Array.isArray(value)) return value.map(materializeJsonIntegers);
  if (value !== null && typeof value === "object") {
    const output = Object.create(null);
    for (const [key, item] of Object.entries(value)) {
      output[key] = materializeJsonIntegers(item);
    }
    return output;
  }
  return value;
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

function sha256Bytes(value) {
  return crypto.createHash("sha256").update(value).digest("hex");
}

function decodeStrictUtf8PreservingBom(raw) {
  return new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(raw);
}

function lfCanonicalSha256Bytes(raw) {
  const text = decodeStrictUtf8PreservingBom(raw);
  return sha256Bytes(Buffer.from(text.replace(/\r\n?/gu, "\n"), "utf8"));
}

function contentId(value) {
  return sha256Bytes(canonicalBytes(value, false));
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

function requireObject(value, label) {
  if (
    value === null ||
    typeof value !== "object" ||
    Array.isArray(value) ||
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

function requireString(value, label) {
  if (typeof value !== "string") throw new TypeError(`${label} must be a string`);
  return value;
}

function requireText(value, label) {
  const text = requireString(value, label);
  if (text.length === 0 || text.trim() !== text) {
    throw new TypeError(`${label} must be nonempty trimmed text`);
  }
  return text;
}

function requireBoolean(value, label) {
  if (typeof value !== "boolean") throw new TypeError(`${label} must be boolean`);
  return value;
}

function requireInteger(value, label) {
  if (!Number.isSafeInteger(value)) throw new TypeError(`${label} must be a safe integer`);
  return value;
}

function requireSha256(value, label) {
  if (typeof value !== "string" || !/^[0-9a-f]{64}$/u.test(value)) {
    throw new TypeError(`${label} must be one lowercase SHA-256`);
  }
  return value;
}

function exactKeys(value, keys, label) {
  const object = requireObject(value, label);
  const actual = Object.keys(object).sort();
  const expected = [...keys].sort();
  if (actual.length !== expected.length || actual.some((key, i) => key !== expected[i])) {
    throw new Error(`${label} key set drift`);
  }
  return object;
}

function deepExact(actual, expected, label) {
  if (Array.isArray(expected)) {
    if (!Array.isArray(actual) || actual.length !== expected.length) {
      throw new Error(`${label} array drift`);
    }
    expected.forEach((value, index) => deepExact(actual[index], value, `${label}[${index}]`));
    return;
  }
  if (expected !== null && typeof expected === "object") {
    exactKeys(actual, Object.keys(expected), label);
    for (const [key, value] of Object.entries(expected)) {
      deepExact(actual[key], value, `${label}.${key}`);
    }
    return;
  }
  if (actual !== expected) throw new Error(`${label} value drift`);
}

function deepFreeze(value) {
  if (value !== null && typeof value === "object" && !Object.isFrozen(value)) {
    Object.freeze(value);
    for (const item of Object.values(value)) deepFreeze(item);
  }
  return value;
}

function requireUtcTimestamp(value, label) {
  const text = requireString(value, label);
  const match = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})\.(\d{3,7})Z$/u.exec(
    text,
  );
  if (match === null || Number(match[1]) < 2000) {
    throw new TypeError(`${label} must be a strict UTC timestamp`);
  }
  const milliseconds = Date.UTC(
    Number(match[1]),
    Number(match[2]) - 1,
    Number(match[3]),
    Number(match[4]),
    Number(match[5]),
    Number(match[6]),
    Number(match[7].slice(0, 3)),
  );
  const date = new Date(milliseconds);
  if (
    date.getUTCFullYear() !== Number(match[1]) ||
    date.getUTCMonth() + 1 !== Number(match[2]) ||
    date.getUTCDate() !== Number(match[3]) ||
    date.getUTCHours() !== Number(match[4]) ||
    date.getUTCMinutes() !== Number(match[5]) ||
    date.getUTCSeconds() !== Number(match[6])
  ) {
    throw new TypeError(`${label} is not a real UTC calendar time`);
  }
  return BigInt(milliseconds) * 10000n + BigInt(match[7].padEnd(7, "0").slice(3));
}

function nowUtc() {
  return new Date().toISOString().replace("Z", "0000Z");
}

function resolveRepositoryPath(relativePath) {
  const relative = requireText(relativePath, "repository-relative path");
  if (
    relative.includes("\\") ||
    path.posix.isAbsolute(relative) ||
    relative.split("/").some((segment) => segment === "" || segment === "." || segment === "..")
  ) {
    throw new Error("repository-relative path is not canonical POSIX relative text");
  }
  const resolved = path.resolve(REPOSITORY_ROOT, ...relative.split("/"));
  const back = path.relative(REPOSITORY_ROOT, resolved);
  if (back.startsWith("..") || path.isAbsolute(back)) {
    throw new Error("repository-relative path escapes the repository root");
  }
  return resolved;
}

function loadStrictJsonFile(filePath, label, canonicalRequired = false) {
  const stat = fs.lstatSync(filePath);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1) {
    throw new Error(`${label} must be one regular single-link file`);
  }
  const raw = fs.readFileSync(filePath);
  const text = decodeStrictUtf8PreservingBom(raw);
  const parsed = parseJsonStrict(text, label);
  const value = materializeJsonIntegers(parsed);
  if (canonicalRequired && !raw.equals(canonicalBytes(value))) {
    throw new Error(`${label} must use canonical compact JSON plus one LF`);
  }
  return Object.freeze({ raw, rawSha256: sha256Bytes(raw), value });
}

function validateProtocol(protocol) {
  exactKeys(
    protocol,
    [
      "schema_version",
      "owner",
      "qualification_compatibility",
      "scope",
      "identity_binding",
      "capture_roles",
      "execution",
      "budgets",
      "exit_matrix",
      "capture_envelope_contract",
      "source_hash_mode",
      "source_sha256",
      "output_contract",
      "evidence_firewall",
      "claim_boundary",
    ],
    "acquisition protocol",
  );
  if (protocol.schema_version !== EVENT_LOG_SOURCE_AUDIT_ACQUISITION_PROTOCOL_SCHEMA_VERSION) {
    throw new Error("acquisition protocol schema drift");
  }
  deepExact(
    protocol.owner,
    {
      wheel: "vz-runtime",
      owner: "volvence_zero.offline_evidence.windows_event_log_source_audit_acquisition",
      mode: "standalone_explicit_operator_invoked_audit_acquisition",
      distribution_scope: "repository-source-checkout-only",
      qualification_or_campaign_auto_invocation: false,
    },
    "protocol.owner",
  );
  deepExact(
    protocol.qualification_compatibility,
    {
      qualification_protocol_relative_path:
        "packages/vz-runtime/src/volvence_zero/offline_evidence/protocols/windows_cuda_host_stability_qualification_v1.json",
      qualification_protocol_id:
        "32f35e4f7027e9519522e099efb696fb352a48faf3ba69be861929304fae1d5f",
      qualification_protocol_raw_sha256:
        "30a881838b41fa5b7e6de5aba6bc94131245796126be5b49c4ebab539f8c4132",
      adapter_imported_or_called: false,
      qualification_projection_emitted: false,
    },
    "protocol.qualification_compatibility",
  );
  deepExact(
    protocol.scope,
    {
      method: "sha256_domain_separated_length_framed_v1",
      domain_separator: "volvence.windows-event-log-source-audit-acquisition-scope.v1",
      components: [
        "acquisition_protocol_id",
        "qualification_protocol_id",
        "operator_scope_binding_id",
        "machine_identity_sha256",
        "boot_identity_sha256",
        "capture_role",
        "execution_backend_id",
      ],
      attempt_number: 1,
      attempt_budget: 1,
      retry_budget: 0,
      attempt_budget_applies_to: "single_artifact_root_only",
      same_root_overwrite_or_retry_permitted: false,
      cross_root_duplicate_scope_excluded: false,
      scope_global_no_retry_proven: false,
      cross_scope_selection_excluded: false,
    },
    "protocol.scope",
  );
  deepExact(
    protocol.identity_binding,
    {
      machine_identity_field: "machine_identity_sha256",
      boot_identity_field: "boot_identity_sha256",
      input_owner: "explicit_caller",
      live_machine_queried: false,
      authoritative: false,
    },
    "protocol.identity_binding",
  );
  deepExact(
    protocol.capture_roles,
    ["qualification_source_audit_before", "qualification_source_audit_after"],
    "protocol.capture_roles",
  );
  const sourceBinding = exactKeys(
    requireObject(
      protocol.execution.source_execution_binding,
      "protocol.execution.source_execution_binding",
    ),
    [
      "schema_version",
      "method",
      "provisioner_relative_path",
      "provisioner_raw_sha256",
      "provisioner_lf_canonical_sha256",
      "strict_utf8_decoding",
      "utf8_bom_permitted",
      "maximum_source_bytes",
      "file_mode",
      "file_access",
      "file_share",
      "same_handle_read_hash_execute",
      "handle_held_through_script_execution_and_exit_unwind",
      "handle_held_until_os_process_exit_attested",
      "path_reopened_for_execution",
      "parser_filename_sets_pscommandpath_fixture_verified",
      "launcher_source_utf8_sha256",
      "launcher_utf16le_sha256",
      "fixed_mode",
      "requested_binding_frozen",
      "realized_source_execution_attested",
      "executable_image_identity_attested",
      "ifeo_excluded",
      "administrator_or_kernel_adversary_excluded",
      "source_content_toctou_excluded_under_declared_threat_model",
    ],
    "protocol.execution.source_execution_binding",
  );
  requireSha256(
    sourceBinding.provisioner_raw_sha256,
    "protocol source-binding provisioner raw SHA-256",
  );
  requireSha256(
    sourceBinding.provisioner_lf_canonical_sha256,
    "protocol source-binding provisioner LF SHA-256",
  );
  requireSha256(
    sourceBinding.launcher_source_utf8_sha256,
    "protocol source-binding launcher source SHA-256",
  );
  requireSha256(
    sourceBinding.launcher_utf16le_sha256,
    "protocol source-binding launcher UTF-16LE SHA-256",
  );
  const launcher = buildSourceBindingLauncher(sourceBinding);
  if (
    launcher.sourceUtf8Sha256 !== sourceBinding.launcher_source_utf8_sha256 ||
    launcher.utf16leSha256 !== sourceBinding.launcher_utf16le_sha256
  ) {
    throw new Error("frozen same-buffer PowerShell launcher identity drift");
  }
  deepExact(
    sourceBinding,
    {
      schema_version: SOURCE_BINDING_SCHEMA_VERSION,
      method: "windows-powershell-parser-same-buffer.v1",
      provisioner_relative_path: PROVISIONER_RELATIVE_PATH,
      provisioner_raw_sha256: sourceBinding.provisioner_raw_sha256,
      provisioner_lf_canonical_sha256:
        sourceBinding.provisioner_lf_canonical_sha256,
      strict_utf8_decoding: true,
      utf8_bom_permitted: false,
      maximum_source_bytes: 2_097_152,
      file_mode: "Open",
      file_access: "Read",
      file_share: "Read",
      same_handle_read_hash_execute: true,
      handle_held_through_script_execution_and_exit_unwind: true,
      handle_held_until_os_process_exit_attested: false,
      path_reopened_for_execution: false,
      parser_filename_sets_pscommandpath_fixture_verified: true,
      launcher_source_utf8_sha256: sourceBinding.launcher_source_utf8_sha256,
      launcher_utf16le_sha256: sourceBinding.launcher_utf16le_sha256,
      fixed_mode: "Audit",
      requested_binding_frozen: true,
      realized_source_execution_attested: false,
      executable_image_identity_attested: false,
      ifeo_excluded: false,
      administrator_or_kernel_adversary_excluded: false,
      source_content_toctou_excluded_under_declared_threat_model: false,
    },
    "protocol.execution.source_execution_binding",
  );
  deepExact(
    protocol.execution,
    {
      production_entrypoint_enabled: false,
      production_backend_id: PRODUCTION_BACKEND_ID,
      synthetic_backend_id: SYNTHETIC_BACKEND_ID,
      platform_system: "Windows",
      executable_template:
        "{SystemRoot}\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",
      argv_template: [
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-EncodedCommand",
        "{frozen_same_buffer_launcher_utf16le_base64}",
      ],
      invocation_fields: [
        "backend_id",
        "executable_template",
        "requested_executable",
        "argv_template",
        "requested_argv",
        "cwd",
        "shell",
        "windows_hide",
        "mode",
        "invocation_realization",
        "environment_inherited",
        "realized_environment_attested",
        "provision_or_allow_source_creation_present",
        "source_execution_binding",
      ],
      requested_argv_launcher_resolution:
        "frozen_protocol_binding_to_utf16le_base64",
      requested_executable_endpoint_binding_required: true,
      synthetic_invocation_realized: false,
      shell: false,
      windows_hide: true,
      working_directory: "repository_root",
      environment_inherited: true,
      realized_environment_attested: false,
      inherited_environment_fully_frozen: false,
      process_started_at_utc_semantics:
        "lower_bound_immediately_before_spawn_call_not_os_creation_attestation",
      job_object_used: false,
      descendants_contained: false,
      prohibited_tokens: [
        "-Command",
        "-ExecutionPolicy",
        "Provision",
        "-AllowSourceCreation",
      ],
      timeout_kill_once: true,
      post_kill_hard_cutoff_required: true,
      overall_supervision_deadline_required: true,
      source_and_executable_pre_post_endpoint_equality_required: true,
      endpoint_equality_proves_continuous_stability: false,
      source_execution_binding: sourceBinding,
    },
    "protocol.execution",
  );
  const budgets = exactKeys(
    protocol.budgets,
    [
      "stdout_max_bytes",
      "stderr_max_bytes",
      "timeout_milliseconds",
      "post_kill_pipe_drain_grace_milliseconds",
      "overall_supervision_deadline_milliseconds",
    ],
    "protocol.budgets",
  );
  for (const key of [
    "stdout_max_bytes",
    "stderr_max_bytes",
    "timeout_milliseconds",
    "post_kill_pipe_drain_grace_milliseconds",
    "overall_supervision_deadline_milliseconds",
  ]) {
    const value = requireInteger(budgets[key], `protocol.budgets.${key}`);
    if (value < 1 || value > 3_600_000) throw new Error(`protocol budget drift: ${key}`);
  }
  if (
    budgets.overall_supervision_deadline_milliseconds !==
    budgets.timeout_milliseconds + budgets.post_kill_pipe_drain_grace_milliseconds
  ) {
    throw new Error("overall supervision deadline must equal timeout plus drain grace");
  }
  deepExact(
    protocol.exit_matrix,
    [
      {
        exit_code: 0,
        stdout_schema_version: AUDIT_SCHEMA_VERSION,
        stderr_must_be_empty: true,
        disposition: "audit_v2_conformant_capture_candidate",
      },
      {
        exit_code: 2,
        stdout_schema_version: AUDIT_SCHEMA_VERSION,
        stderr_must_be_empty: true,
        disposition: "audit_v2_nonconformant_capture_candidate",
      },
      {
        exit_code: 3,
        stdout_schema_version: FAILURE_SCHEMA_VERSION,
        stderr_must_be_empty: false,
        disposition: "failure_v1_quarantined",
      },
      {
        exit_code: null,
        stdout_schema_version: null,
        stderr_must_be_empty: false,
        disposition: "unclassified_process_or_output_quarantined",
      },
    ],
    "protocol.exit_matrix",
  );
  deepExact(
    protocol.capture_envelope_contract,
    {
      schema_version: ADAPTER_CAPTURE_ENVELOPE_SCHEMA_VERSION,
      ordered_fields: ADAPTER_CAPTURE_ENVELOPE_FIELDS,
      candidate_dispositions: [
        "audit_v2_conformant_capture_candidate",
        "audit_v2_nonconformant_capture_candidate",
      ],
      quarantined_value: null,
      capture_authoritative: false,
    },
    "protocol.capture_envelope_contract",
  );
  if (protocol.source_hash_mode !== "utf8_lf_canonical_v1") {
    throw new Error("protocol source hash mode drift");
  }
  const sourcePins = requireObject(protocol.source_sha256, "protocol.source_sha256");
  const requiredSources = [
    ".gitattributes",
    "packages/vz-runtime/src/volvence_zero/offline_evidence/windows_event_log_source_audit_acquisition.mjs",
    "packages/vz-runtime/src/volvence_zero/offline_evidence/provision_volvence_evidence_event_log.ps1",
    "scripts/run_windows_event_log_source_audit_acquisition.mjs",
  ];
  deepExact(Object.keys(sourcePins).sort(), [...requiredSources].sort(), "protocol source paths");
  for (const relative of requiredSources) {
    requireSha256(sourcePins[relative], `protocol.source_sha256.${relative}`);
  }
  deepExact(
    protocol.output_contract,
    {
      create_only: true,
      claim_file: CLAIM_FILE,
      terminal_file: TERMINAL_FILE,
      stdout_file: STDOUT_FILE,
      stderr_file: STDERR_FILE,
      exact_files: COMPLETE_FILES,
      stream_capture_schema_version: STREAM_CAPTURE_SCHEMA_VERSION,
      stream_capture_fields: STREAM_CAPTURE_FIELDS,
      process_observation_schema_version: PROCESS_OBSERVATION_SCHEMA_VERSION,
      stream_outcome_schema_version: STREAM_OUTCOME_SCHEMA_VERSION,
      stream_outcome_fields: [
        "schema_version",
        "pipe_end_observed",
        "pipe_close_observed",
        "forcibly_detached",
        "observed_byte_count",
        "persisted_byte_count",
        "persistence_complete",
        "capture_error_stage",
        "capture_error_name",
        "capture_error_message_sha256",
        "persistence_error_stage",
        "persistence_error_name",
        "persistence_error_message_sha256",
      ],
      artifact_parent_must_preexist_as_non_symlink_directory: true,
      artifact_parent_chain_reparse_points_excluded: false,
      claim_file_descriptor_content_fsync_before_process_creation: true,
      directory_entry_durability_guaranteed: false,
      lifecycle_finalization_then_stream_fsync_and_same_descriptor_bounded_read_before_descriptor_close:
        true,
      normal_candidate_requires_child_exit_and_close: true,
      hard_cutoff_persists_bounded_prefix_for_quarantine: true,
      async_process_supervision_deadline_excludes_synchronous_persistence: true,
      stream_readback_or_terminal_write_failure_can_leave_incomplete_root: true,
      capture_envelope_null_for_quarantine: true,
      terminal_id_method: "sha256_of_canonical_terminal_core_without_terminal_id",
      terminal_raw_sha256_is_full_root_identity: true,
      offline_validator_queries_live_state: false,
    },
    "protocol.output_contract",
  );
  deepExact(
    protocol.evidence_firewall,
    {
      capture_envelope_authoritative: false,
      real_process_observation: false,
      real_provisioner_observation: false,
      source_config_projection_emitted: false,
      eligible_as_host_qualification_input: false,
      qualification_authorized: false,
      cuda_execution_authorized: false,
      formal_evidence_authorized: false,
      production_active_authorized: false,
      appendable_proven: false,
      readable_proven: false,
      learnable_proven: false,
      steerable_proven: false,
      four_capability_claim_authorized: false,
      tamper_resistance_proven: false,
      path_parent_reparse_trust_proven: false,
      directory_entry_durability_guaranteed: false,
      continuous_endpoint_stability_proven: false,
    },
    "protocol.evidence_firewall",
  );
  requireText(protocol.claim_boundary, "protocol.claim_boundary");
}

function loadProtocol({ verifySources = true } = {}) {
  const loaded = loadStrictJsonFile(PROTOCOL_PATH, "acquisition protocol", false);
  validateProtocol(loaded.value);
  const protocolId = contentId(loaded.value);
  const compatibilityPath = resolveRepositoryPath(
    loaded.value.qualification_compatibility.qualification_protocol_relative_path,
  );
  const compatibility = loadStrictJsonFile(
    compatibilityPath,
    "bound qualification protocol",
    false,
  );
  if (
    contentId(compatibility.value) !==
      loaded.value.qualification_compatibility.qualification_protocol_id ||
    compatibility.rawSha256 !==
      loaded.value.qualification_compatibility.qualification_protocol_raw_sha256
  ) {
    throw new Error("bound qualification protocol identity drift");
  }
  if (verifySources) {
    for (const [relative, expected] of Object.entries(loaded.value.source_sha256)) {
      const sourcePath = resolveRepositoryPath(relative);
      const endpoint = observeLocalFileEndpoint(sourcePath, true);
      if (endpoint.lf_canonical_sha256 !== expected) {
        throw new Error(`critical acquisition source SHA-256 drift: ${relative}`);
      }
      if (
        relative === PROVISIONER_RELATIVE_PATH &&
        (endpoint.raw_sha256 !==
          loaded.value.execution.source_execution_binding.provisioner_raw_sha256 ||
          endpoint.lf_canonical_sha256 !==
            loaded.value.execution.source_execution_binding
              .provisioner_lf_canonical_sha256)
      ) {
        throw new Error("reviewed provisioner source-binding pin drift");
      }
    }
  }
  return Object.freeze({
    protocol: loaded.value,
    protocolId,
    protocolRawSha256: loaded.rawSha256,
  });
}

function writeCreateFile(filePath, bytes) {
  const descriptor = fs.openSync(filePath, "wx");
  try {
    fs.writeFileSync(descriptor, bytes);
    fs.fsyncSync(descriptor);
  } finally {
    fs.closeSync(descriptor);
  }
  return Object.freeze({
    rawSha256: sha256Bytes(bytes),
    byteCount: bytes.length,
  });
}

function writeCreateJson(filePath, value) {
  const raw = canonicalBytes(value);
  return Object.freeze({ ...writeCreateFile(filePath, raw), raw });
}

function ensureCreateOnlyRoot(root) {
  if (fs.existsSync(root)) throw new Error(`acquisition root already exists: ${root}`);
  const parent = path.dirname(root);
  if (!fs.existsSync(parent)) {
    throw new Error("acquisition root parent must preexist as a non-symlink directory");
  }
  const parentStat = fs.lstatSync(parent);
  if (!parentStat.isDirectory() || parentStat.isSymbolicLink()) {
    throw new Error("acquisition root parent must preexist as a non-symlink directory");
  }
  fs.mkdirSync(root, { recursive: false });
  const rootStat = fs.lstatSync(root);
  if (!rootStat.isDirectory() || rootStat.isSymbolicLink()) {
    throw new Error("new acquisition root is not a non-symlink directory");
  }
  fs.mkdirSync(path.join(root, "streams"), { recursive: false });
}

function normalizedAbsolutePath(value) {
  return path.resolve(value).replace(/\\/gu, "/");
}

function syntheticEndpoint(label) {
  const raw = Buffer.from(`synthetic-endpoint:${label}\n`, "utf8");
  return {
    schema_version: "windows-event-log-source-audit-file-endpoint.v1",
    observation_kind: "synthetic_declaration",
    absolute_path: `synthetic://${label}`,
    real_path: `synthetic://${label}`,
    byte_count: raw.length,
    raw_sha256: sha256Bytes(raw),
    lf_canonical_sha256: sha256Bytes(raw),
    device_id: null,
    inode_id: null,
    mtime_nanoseconds: null,
    regular_file: false,
    symbolic_link: false,
  };
}

function validateEndpoint(endpoint, label) {
  exactKeys(
    endpoint,
    [
      "schema_version",
      "observation_kind",
      "absolute_path",
      "real_path",
      "byte_count",
      "raw_sha256",
      "lf_canonical_sha256",
      "device_id",
      "inode_id",
      "mtime_nanoseconds",
      "regular_file",
      "symbolic_link",
    ],
    label,
  );
  if (endpoint.schema_version !== "windows-event-log-source-audit-file-endpoint.v1") {
    throw new Error(`${label} schema drift`);
  }
  if (!["synthetic_declaration", "local_file_observation"].includes(endpoint.observation_kind)) {
    throw new Error(`${label} observation kind drift`);
  }
  requireText(endpoint.absolute_path, `${label}.absolute_path`);
  requireText(endpoint.real_path, `${label}.real_path`);
  if (requireInteger(endpoint.byte_count, `${label}.byte_count`) < 0) {
    throw new Error(`${label} byte count is negative`);
  }
  requireSha256(endpoint.raw_sha256, `${label}.raw_sha256`);
  if (endpoint.lf_canonical_sha256 !== null) {
    requireSha256(endpoint.lf_canonical_sha256, `${label}.lf_canonical_sha256`);
  }
  for (const key of ["device_id", "inode_id", "mtime_nanoseconds"]) {
    if (endpoint[key] !== null) requireText(endpoint[key], `${label}.${key}`);
  }
  requireBoolean(endpoint.regular_file, `${label}.regular_file`);
  requireBoolean(endpoint.symbolic_link, `${label}.symbolic_link`);
  return endpoint;
}

function observeLocalFileEndpoint(filePath, includeLfHash) {
  const before = fs.lstatSync(filePath, { bigint: true });
  if (!before.isFile() || before.isSymbolicLink() || before.nlink !== 1n) {
    throw new Error(`endpoint is not one regular single-link file: ${filePath}`);
  }
  const descriptor = fs.openSync(filePath, "r");
  try {
    const opened = fs.fstatSync(descriptor, { bigint: true });
    if (
      opened.dev !== before.dev ||
      opened.ino !== before.ino ||
      opened.size !== before.size ||
      opened.mtimeNs !== before.mtimeNs
    ) {
      throw new Error(`endpoint changed before descriptor read: ${filePath}`);
    }
    const raw = fs.readFileSync(descriptor);
    const after = fs.fstatSync(descriptor, { bigint: true });
    if (
      after.dev !== opened.dev ||
      after.ino !== opened.ino ||
      after.size !== opened.size ||
      after.mtimeNs !== opened.mtimeNs
    ) {
      throw new Error(`endpoint changed during descriptor read: ${filePath}`);
    }
    let lfHash = null;
    if (includeLfHash) {
      lfHash = lfCanonicalSha256Bytes(raw);
    }
    return {
      schema_version: "windows-event-log-source-audit-file-endpoint.v1",
      observation_kind: "local_file_observation",
      absolute_path: normalizedAbsolutePath(filePath),
      real_path: normalizedAbsolutePath(fs.realpathSync(filePath)),
      byte_count: raw.length,
      raw_sha256: sha256Bytes(raw),
      lf_canonical_sha256: lfHash,
      device_id: opened.dev.toString(),
      inode_id: opened.ino.toString(),
      mtime_nanoseconds: opened.mtimeNs.toString(),
      regular_file: true,
      symbolic_link: false,
    };
  } finally {
    fs.closeSync(descriptor);
  }
}

function decodeBase64Strict(value, label) {
  const text = requireString(value, label);
  if (text.length % 4 !== 0 || !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/u.test(text)) {
    throw new TypeError(`${label} must be canonical base64`);
  }
  const raw = Buffer.from(text, "base64");
  if (raw.toString("base64") !== text) throw new TypeError(`${label} base64 round-trip drift`);
  return raw;
}

function parseCompactReceiptDiscriminator(raw) {
  const empty = {
    strict_compact_single_lf_utf8: false,
    schema_version: null,
    mode: null,
    receipt_exit_code: null,
    overall_conformant: null,
    completed: null,
    parse_error_sha256: null,
  };
  try {
    if (
      raw.length >= 3 &&
      raw[0] === 0xef &&
      raw[1] === 0xbb &&
      raw[2] === 0xbf
    ) {
      throw new SyntaxError("stdout must not begin with a UTF-8 BOM");
    }
    const text = decodeStrictUtf8PreservingBom(raw);
    if (
      !text.endsWith("\n") ||
      text.slice(0, -1).includes("\n") ||
      text.includes("\r") ||
      text.startsWith("\uFEFF")
    ) {
      throw new SyntaxError("stdout must be one compact UTF-8 JSON line with one terminal LF");
    }
    const body = text.slice(0, -1);
    const parsed = materializeJsonIntegers(parseJsonStrict(body, "captured stdout JSON"));
    const object = requireObject(parsed, "captured stdout JSON");
    if (JSON.stringify(object) !== body) {
      throw new SyntaxError("stdout JSON is not compact or uses a noncanonical lexical form");
    }
    return {
      strict_compact_single_lf_utf8: true,
      schema_version:
        typeof object.schema_version === "string" ? object.schema_version : null,
      mode: typeof object.mode === "string" ? object.mode : null,
      receipt_exit_code:
        Number.isSafeInteger(object.process_exit_code) ? object.process_exit_code : null,
      overall_conformant:
        typeof object.overall_conformant === "boolean" ? object.overall_conformant : null,
      completed: typeof object.completed === "boolean" ? object.completed : null,
      parse_error_sha256: null,
    };
  } catch (error) {
    return {
      ...empty,
      parse_error_sha256: sha256Bytes(
        Buffer.from(`${error.name}:${error.message}`, "utf8"),
      ),
    };
  }
}

function classifyOutcome(processObservation, stdout, stderr) {
  const discriminator = parseCompactReceiptDiscriminator(stdout);
  const realLifecycleComplete =
    processObservation.process_started_at_semantics ===
      "lower_bound_immediately_before_spawn_call" &&
    processObservation.process_exit_observation === "child_exit_event" &&
    processObservation.streams_close_observation === "child_close_event" &&
    processObservation.finalization_reason === "child_close";
  const syntheticLifecycleDeclared =
    processObservation.process_started_at_semantics === "synthetic_declaration" &&
    processObservation.process_exit_observation === "synthetic_declaration" &&
    processObservation.streams_close_observation === "synthetic_declaration" &&
    processObservation.finalization_reason === "synthetic_declaration";
  const cleanProcess =
    (realLifecycleComplete || syntheticLifecycleDeclared) &&
    processObservation.capture_complete === true &&
    processObservation.signal === null &&
    processObservation.timed_out === false &&
    processObservation.hard_cutoff_at_utc === null &&
    processObservation.capture_detached_at_hard_cutoff === false &&
    processObservation.overflow_stream === null &&
    processObservation.kill_attempted === false &&
    processObservation.kill_attempt_count === 0 &&
    processObservation.spawn_error_name === null &&
    processObservation.spawn_error_message_sha256 === null;
  const cleanAuditV2 =
    cleanProcess &&
    stderr.length === 0 &&
    discriminator.strict_compact_single_lf_utf8 === true &&
    discriminator.schema_version === AUDIT_SCHEMA_VERSION &&
    discriminator.mode === "Audit";
  if (
    cleanAuditV2 &&
    processObservation.exit_code === 0 &&
    discriminator.receipt_exit_code === 0 &&
    discriminator.overall_conformant === true
  ) {
    return {
      disposition: "audit_v2_conformant_capture_candidate",
      capture_candidate: true,
      quarantined: false,
      discriminator,
    };
  }
  if (
    cleanAuditV2 &&
    processObservation.exit_code === 2 &&
    discriminator.receipt_exit_code === 2 &&
    discriminator.overall_conformant === false
  ) {
    return {
      disposition: "audit_v2_nonconformant_capture_candidate",
      capture_candidate: true,
      quarantined: false,
      discriminator,
    };
  }
  if (
    cleanProcess &&
    processObservation.exit_code === 3 &&
    discriminator.strict_compact_single_lf_utf8 === true &&
    discriminator.schema_version === FAILURE_SCHEMA_VERSION &&
    discriminator.mode === "Audit" &&
    discriminator.receipt_exit_code === 3 &&
    discriminator.overall_conformant === false &&
    discriminator.completed === false
  ) {
    return {
      disposition: "failure_v1_quarantined",
      capture_candidate: false,
      quarantined: true,
      discriminator,
    };
  }
  return {
    disposition: "unclassified_process_or_output_quarantined",
    capture_candidate: false,
    quarantined: true,
    discriminator,
  };
}

function validateStreamOutcome(value, label) {
  exactKeys(
    value,
    [
      "schema_version",
      "pipe_end_observed",
      "pipe_close_observed",
      "forcibly_detached",
      "observed_byte_count",
      "persisted_byte_count",
      "persistence_complete",
      "capture_error_stage",
      "capture_error_name",
      "capture_error_message_sha256",
      "persistence_error_stage",
      "persistence_error_name",
      "persistence_error_message_sha256",
    ],
    label,
  );
  if (value.schema_version !== STREAM_OUTCOME_SCHEMA_VERSION) {
    throw new Error(`${label} schema drift`);
  }
  requireBoolean(value.pipe_end_observed, `${label}.pipe_end_observed`);
  requireBoolean(value.pipe_close_observed, `${label}.pipe_close_observed`);
  requireBoolean(value.forcibly_detached, `${label}.forcibly_detached`);
  const observed = requireInteger(
    value.observed_byte_count,
    `${label}.observed_byte_count`,
  );
  const persisted = requireInteger(
    value.persisted_byte_count,
    `${label}.persisted_byte_count`,
  );
  if (observed < 0 || persisted < 0 || persisted > observed) {
    throw new Error(`${label} byte-count chronology drift`);
  }
  requireBoolean(value.persistence_complete, `${label}.persistence_complete`);
  const captureErrorNull =
    value.capture_error_stage === null &&
    value.capture_error_name === null &&
    value.capture_error_message_sha256 === null;
  const captureErrorPresent =
    ["pipe", "buffer"].includes(value.capture_error_stage) &&
    typeof value.capture_error_name === "string" &&
    typeof value.capture_error_message_sha256 === "string";
  if (!captureErrorNull && !captureErrorPresent) {
    throw new Error(`${label} capture error fields must be all-null or frozen`);
  }
  if (captureErrorPresent) {
    requireText(value.capture_error_name, `${label}.capture_error_name`);
    requireSha256(
      value.capture_error_message_sha256,
      `${label}.capture_error_message_sha256`,
    );
  }
  const persistenceErrorNull =
    value.persistence_error_stage === null &&
    value.persistence_error_name === null &&
    value.persistence_error_message_sha256 === null;
  const persistenceErrorPresent =
    ["write", "fsync"].includes(value.persistence_error_stage) &&
    typeof value.persistence_error_name === "string" &&
    typeof value.persistence_error_message_sha256 === "string";
  if (!persistenceErrorNull && !persistenceErrorPresent) {
    throw new Error(`${label} persistence error fields must be all-null or frozen`);
  }
  if (persistenceErrorPresent) {
    requireText(value.persistence_error_name, `${label}.persistence_error_name`);
    requireSha256(
      value.persistence_error_message_sha256,
      `${label}.persistence_error_message_sha256`,
    );
  }
  if (value.persistence_complete !== (persisted === observed && persistenceErrorNull)) {
    throw new Error(`${label} persistence status drift`);
  }
  return value;
}

function validateProcessObservation(value, label) {
  exactKeys(
    value,
    [
      "schema_version",
      "spawn_attempted",
      "process_id",
      "process_started_at_utc",
      "process_started_at_semantics",
      "os_process_creation_time_attested",
      "process_exit_observation",
      "streams_close_observation",
      "process_exited_at_utc",
      "streams_closed_at_utc",
      "stdout_captured_at_utc",
      "exit_code",
      "signal",
      "timed_out",
      "hard_cutoff_at_utc",
      "overall_deadline_exceeded",
      "finalization_reason",
      "capture_detached_at_hard_cutoff",
      "termination_confirmed",
      "process_may_remain_running",
      "overflow_stream",
      "kill_attempted",
      "kill_attempt_count",
      "kill_request_accepted",
      "kill_error_name",
      "kill_error_message_sha256",
      "descendants_contained",
      "spawn_error_name",
      "spawn_error_message_sha256",
      "capture_complete",
      "stream_outcomes",
    ],
    label,
  );
  if (value.schema_version !== PROCESS_OBSERVATION_SCHEMA_VERSION) {
    throw new Error(`${label} schema drift`);
  }
  requireBoolean(value.spawn_attempted, `${label}.spawn_attempted`);
  if (value.process_id !== null) {
    const processId = requireInteger(value.process_id, `${label}.process_id`);
    if (processId <= 0 || value.spawn_attempted !== true) {
      throw new Error(`${label} process ID/spawn-attempt drift`);
    }
  }
  const started = requireUtcTimestamp(value.process_started_at_utc, `${label}.process_started_at_utc`);
  if (
    ![
      "lower_bound_immediately_before_spawn_call",
      "synthetic_declaration",
    ].includes(value.process_started_at_semantics)
  ) {
    throw new Error(`${label}.process_started_at_semantics drift`);
  }
  if (value.os_process_creation_time_attested !== false) {
    throw new Error(`${label} must not attest an OS process-creation timestamp`);
  }
  const synthetic = value.process_started_at_semantics === "synthetic_declaration";
  const allowedExitObservations = synthetic
    ? ["synthetic_declaration"]
    : ["child_exit_event", "child_close_fallback", "not_observed"];
  const allowedCloseObservations = synthetic
    ? ["synthetic_declaration"]
    : ["child_close_event", "not_observed"];
  if (!allowedExitObservations.includes(value.process_exit_observation)) {
    throw new Error(`${label}.process_exit_observation drift`);
  }
  if (!allowedCloseObservations.includes(value.streams_close_observation)) {
    throw new Error(`${label}.streams_close_observation drift`);
  }
  let exited = null;
  if (value.process_exited_at_utc !== null) {
    exited = requireUtcTimestamp(
      value.process_exited_at_utc,
      `${label}.process_exited_at_utc`,
    );
  }
  let closed = null;
  if (value.streams_closed_at_utc !== null) {
    closed = requireUtcTimestamp(
      value.streams_closed_at_utc,
      `${label}.streams_closed_at_utc`,
    );
  }
  if (
    (value.process_exit_observation === "not_observed") !== (exited === null) ||
    (value.streams_close_observation === "not_observed") !== (closed === null)
  ) {
    throw new Error(`${label} observed child events/timestamps drift`);
  }
  if (closed !== null && exited === null) {
    throw new Error(`${label} child close requires an exit upper bound`);
  }
  const captured = requireUtcTimestamp(
    value.stdout_captured_at_utc,
    `${label}.stdout_captured_at_utc`,
  );
  if (
    (exited !== null && (started > exited || exited > captured)) ||
    (closed !== null && (started > closed || closed > captured)) ||
    (exited !== null && closed !== null && exited > closed)
  ) {
    throw new Error(`${label} chronology drift`);
  }
  if (value.exit_code !== null) requireInteger(value.exit_code, `${label}.exit_code`);
  if (value.signal !== null) requireText(value.signal, `${label}.signal`);
  if (
    value.process_exit_observation === "not_observed" &&
    (value.exit_code !== null || value.signal !== null)
  ) {
    throw new Error(`${label} unobserved exit cannot publish code or signal`);
  }
  requireBoolean(value.timed_out, `${label}.timed_out`);
  let hardCutoffAt = null;
  if (value.hard_cutoff_at_utc !== null) {
    hardCutoffAt = requireUtcTimestamp(
      value.hard_cutoff_at_utc,
      `${label}.hard_cutoff_at_utc`,
    );
  }
  if (hardCutoffAt !== null && (hardCutoffAt < started || hardCutoffAt > captured)) {
    throw new Error(`${label} hard-cutoff chronology drift`);
  }
  const hardCutoffFinalization = [
    "post_kill_grace_cutoff",
    "overall_hard_cutoff",
  ].includes(value.finalization_reason);
  if (
    ![
      "synthetic_declaration",
      "spawn_throw",
      "child_close",
      "post_kill_grace_cutoff",
      "overall_hard_cutoff",
    ].includes(value.finalization_reason) ||
    hardCutoffFinalization !== (hardCutoffAt !== null)
  ) {
    throw new Error(`${label}.finalization_reason drift`);
  }
  requireBoolean(
    value.overall_deadline_exceeded,
    `${label}.overall_deadline_exceeded`,
  );
  if (
    value.overall_deadline_exceeded !==
    (value.finalization_reason === "overall_hard_cutoff")
  ) {
    throw new Error(`${label} overall deadline status drift`);
  }
  requireBoolean(
    value.capture_detached_at_hard_cutoff,
    `${label}.capture_detached_at_hard_cutoff`,
  );
  if (value.capture_detached_at_hard_cutoff !== hardCutoffFinalization) {
    throw new Error(`${label} hard-cutoff detach status drift`);
  }
  if (hardCutoffFinalization && value.streams_close_observation !== "not_observed") {
    throw new Error(`${label} hard cutoff cannot claim child close`);
  }
  requireBoolean(value.termination_confirmed, `${label}.termination_confirmed`);
  requireBoolean(
    value.process_may_remain_running,
    `${label}.process_may_remain_running`,
  );
  if (
    value.termination_confirmed !==
      (value.spawn_attempted &&
        value.process_exit_observation !== "not_observed") ||
    value.process_may_remain_running !==
      (value.spawn_attempted && value.process_exit_observation === "not_observed")
  ) {
    throw new Error(`${label} process termination status drift`);
  }
  if (value.overflow_stream !== null && !["stdout", "stderr"].includes(value.overflow_stream)) {
    throw new Error(`${label}.overflow_stream drift`);
  }
  requireBoolean(value.kill_attempted, `${label}.kill_attempted`);
  const killCount = requireInteger(value.kill_attempt_count, `${label}.kill_attempt_count`);
  if (killCount < 0 || killCount > 1 || value.kill_attempted !== (killCount === 1)) {
    throw new Error(`${label} kill-once contract drift`);
  }
  if (
    (value.timed_out ||
      value.overflow_stream !== null ||
      hardCutoffFinalization ||
      (value.spawn_attempted &&
        value.spawn_error_name !== null &&
        value.finalization_reason !== "spawn_throw")) &&
    killCount !== 1
  ) {
    throw new Error(`${label} failed process supervision must attempt exactly one kill`);
  }
  if (
    (killCount === 0 && value.kill_request_accepted !== null) ||
    (killCount === 1 && typeof value.kill_request_accepted !== "boolean")
  ) {
    throw new Error(`${label} kill-request result drift`);
  }
  const killErrorNameNull = value.kill_error_name === null;
  const killErrorHashNull = value.kill_error_message_sha256 === null;
  if (killErrorNameNull !== killErrorHashNull || (killCount === 0 && !killErrorNameNull)) {
    throw new Error(`${label} kill error all-or-none drift`);
  }
  if (!killErrorNameNull) {
    requireText(value.kill_error_name, `${label}.kill_error_name`);
    requireSha256(value.kill_error_message_sha256, `${label}.kill_error_message_sha256`);
    if (value.kill_request_accepted !== false) {
      throw new Error(`${label} throwing kill request cannot be accepted`);
    }
  }
  if (value.descendants_contained !== false) {
    throw new Error(`${label} must not claim descendant containment without a Job Object`);
  }
  const spawnNameNull = value.spawn_error_name === null;
  const spawnHashNull = value.spawn_error_message_sha256 === null;
  if (spawnNameNull !== spawnHashNull) throw new Error(`${label} spawn error all-or-none drift`);
  if (!spawnNameNull) {
    requireText(value.spawn_error_name, `${label}.spawn_error_name`);
    requireSha256(value.spawn_error_message_sha256, `${label}.spawn_error_message_sha256`);
  }
  requireBoolean(value.capture_complete, `${label}.capture_complete`);
  const streamOutcomes = exactKeys(
    requireObject(value.stream_outcomes, `${label}.stream_outcomes`),
    ["stdout", "stderr"],
    `${label}.stream_outcomes`,
  );
  const stdoutOutcome = validateStreamOutcome(
    streamOutcomes.stdout,
    `${label}.stream_outcomes.stdout`,
  );
  const stderrOutcome = validateStreamOutcome(
    streamOutcomes.stderr,
    `${label}.stream_outcomes.stderr`,
  );
  if (stdoutOutcome.forcibly_detached !== stderrOutcome.forcibly_detached) {
    throw new Error(`${label} detached-pipe outcomes must agree`);
  }
  if (
    value.capture_detached_at_hard_cutoff !==
    (stdoutOutcome.forcibly_detached && stderrOutcome.forcibly_detached)
  ) {
    throw new Error(`${label} detached-pipe summary drift`);
  }
  const streamsClean = [stdoutOutcome, stderrOutcome].every(
    (outcome) =>
      outcome.persistence_complete &&
      outcome.capture_error_stage === null &&
      outcome.persistence_error_stage === null &&
      outcome.forcibly_detached === false,
  );
  const streamTerminalsComplete =
    synthetic ||
    [stdoutOutcome, stderrOutcome].every(
      (outcome) => outcome.pipe_end_observed && outcome.pipe_close_observed,
    );
  const lifecycleComplete = synthetic
    ? value.finalization_reason === "synthetic_declaration" &&
      value.process_exit_observation === "synthetic_declaration" &&
      value.streams_close_observation === "synthetic_declaration"
    : value.process_exit_observation === "child_exit_event" &&
      value.streams_close_observation === "child_close_event" &&
      value.finalization_reason === "child_close";
  const expectedCaptureComplete =
    lifecycleComplete &&
    streamsClean &&
    streamTerminalsComplete &&
    value.timed_out === false &&
    hardCutoffAt === null &&
    value.overflow_stream === null &&
    value.spawn_error_name === null &&
    value.signal === null &&
    value.kill_attempt_count === 0;
  if (value.capture_complete !== expectedCaptureComplete) {
    throw new Error(`${label} complete-capture derivation drift`);
  }
  return value;
}

function fixedRequestedArgv(protocol) {
  const requestedArgv = [...protocol.execution.argv_template];
  requestedArgv[4] = buildSourceBindingLauncher(
    protocol.execution.source_execution_binding,
  ).encodedCommand;
  return requestedArgv;
}

function invocationContract(protocol, backendId, requestedExecutable, requestedArgv) {
  deepExact(requestedArgv, fixedRequestedArgv(protocol), "requested Audit argv");
  if (backendId === PRODUCTION_BACKEND_ID) {
    const executable = requireText(requestedExecutable, "requestedExecutable");
    if (
      !path.win32.isAbsolute(executable) ||
      !executable.toLowerCase().endsWith(
        "\\system32\\windowspowershell\\v1.0\\powershell.exe",
      )
    ) {
      throw new Error("requested production executable does not realize the frozen template");
    }
    return {
      backend_id: backendId,
      executable_template: protocol.execution.executable_template,
      requested_executable: executable,
      argv_template: protocol.execution.argv_template,
      requested_argv: requestedArgv,
      cwd: REPOSITORY_ROOT,
      shell: false,
      windows_hide: true,
      mode: "Audit",
      invocation_realization: "terminal_process_observation",
      environment_inherited: true,
      realized_environment_attested: false,
      provision_or_allow_source_creation_present: false,
      source_execution_binding: protocol.execution.source_execution_binding,
    };
  }
  if (backendId === SYNTHETIC_BACKEND_ID) {
    if (requestedExecutable !== "synthetic://windows-powershell-5.1") {
      throw new Error("synthetic requested executable drift");
    }
    return {
      backend_id: backendId,
      executable_template: protocol.execution.executable_template,
      requested_executable: requestedExecutable,
      argv_template: protocol.execution.argv_template,
      requested_argv: requestedArgv,
      cwd: REPOSITORY_ROOT,
      shell: false,
      windows_hide: true,
      mode: "Audit",
      invocation_realization: "synthetic_not_realized",
      environment_inherited: false,
      realized_environment_attested: false,
      provision_or_allow_source_creation_present: false,
      source_execution_binding: protocol.execution.source_execution_binding,
    };
  }
  throw new Error("unsupported acquisition backend");
}

function validateRequestedInvocationEndpoints(
  protocol,
  invocation,
  backendId,
  sourceEndpointBefore,
  executableEndpointBefore,
) {
  if (backendId !== PRODUCTION_BACKEND_ID) return;
  for (const [label, endpoint] of [
    ["requested source", sourceEndpointBefore],
    ["requested executable", executableEndpointBefore],
  ]) {
    if (
      endpoint.observation_kind !== "local_file_observation" ||
      endpoint.regular_file !== true ||
      endpoint.symbolic_link !== false
    ) {
      throw new Error(`${label} must be a directly observed regular non-symlink file`);
    }
  }
  const requestedExecutableNormalized = normalizedAbsolutePath(
    invocation.requested_executable,
  );
  const requestedSourceNormalized = normalizedAbsolutePath(
    resolveRepositoryPath(
      protocol.execution.source_execution_binding.provisioner_relative_path,
    ),
  );
  if (
    requestedExecutableNormalized.toLowerCase() !==
      executableEndpointBefore.absolute_path.toLowerCase() ||
    requestedExecutableNormalized.toLowerCase() !==
      executableEndpointBefore.real_path.toLowerCase() ||
    requestedSourceNormalized.toLowerCase() !==
      sourceEndpointBefore.absolute_path.toLowerCase() ||
    requestedSourceNormalized.toLowerCase() !== sourceEndpointBefore.real_path.toLowerCase() ||
    sourceEndpointBefore.raw_sha256 !==
      protocol.execution.source_execution_binding.provisioner_raw_sha256 ||
    sourceEndpointBefore.lf_canonical_sha256 !==
      protocol.execution.source_execution_binding.provisioner_lf_canonical_sha256
  ) {
    throw new Error(
      "requested invocation/source binding does not match its pre-spawn file endpoints",
    );
  }
}

function computeScopeId(
  protocolState,
  operatorScopeBindingId,
  machineIdentitySha256,
  bootIdentitySha256,
  captureRole,
  backendId,
) {
  return domainSeparatedSha256(protocolState.protocol.scope.domain_separator, [
    protocolState.protocolId,
    protocolState.protocol.qualification_compatibility.qualification_protocol_id,
    operatorScopeBindingId,
    machineIdentitySha256,
    bootIdentitySha256,
    captureRole,
    backendId,
  ]);
}

function makeClaim({
  protocolState,
  artifactRoot,
  operatorScopeBindingId,
  machineIdentitySha256,
  bootIdentitySha256,
  captureRole,
  backendId,
  requestedExecutable,
  requestedArgv,
  sourceEndpointBefore,
  executableEndpointBefore,
  claimedAtUtc,
}) {
  if (!protocolState.protocol.capture_roles.includes(captureRole)) {
    throw new Error("capture role is not frozen in the acquisition protocol");
  }
  requireSha256(operatorScopeBindingId, "operatorScopeBindingId");
  requireSha256(machineIdentitySha256, "machineIdentitySha256");
  requireSha256(bootIdentitySha256, "bootIdentitySha256");
  validateEndpoint(sourceEndpointBefore, "sourceEndpointBefore");
  validateEndpoint(executableEndpointBefore, "executableEndpointBefore");
  requireUtcTimestamp(claimedAtUtc, "claimedAtUtc");
  const invocation = invocationContract(
    protocolState.protocol,
    backendId,
    requestedExecutable,
    requestedArgv,
  );
  validateRequestedInvocationEndpoints(
    protocolState.protocol,
    invocation,
    backendId,
    sourceEndpointBefore,
    executableEndpointBefore,
  );
  const scopeId = computeScopeId(
    protocolState,
    operatorScopeBindingId,
    machineIdentitySha256,
    bootIdentitySha256,
    captureRole,
    backendId,
  );
  return {
    schema_version: EVENT_LOG_SOURCE_AUDIT_ACQUISITION_CLAIM_SCHEMA_VERSION,
    sequence: 0,
    scope_id: scopeId,
    previous_receipt_sha256: null,
    acquisition_protocol_id: protocolState.protocolId,
    acquisition_protocol_raw_sha256: protocolState.protocolRawSha256,
    qualification_protocol_id:
      protocolState.protocol.qualification_compatibility.qualification_protocol_id,
    qualification_protocol_raw_sha256:
      protocolState.protocol.qualification_compatibility.qualification_protocol_raw_sha256,
    operator_scope_binding_id: operatorScopeBindingId,
    machine_identity_sha256: machineIdentitySha256,
    boot_identity_sha256: bootIdentitySha256,
    identity_binding_authoritative: false,
    capture_role: captureRole,
    execution_backend_id: backendId,
    attempt_number: 1,
    attempt_budget: 1,
    retry_budget: 0,
    attempt_budget_applies_to: "single_artifact_root_only",
    same_root_overwrite_or_retry_permitted: false,
    cross_root_duplicate_scope_excluded: false,
    scope_global_no_retry_proven: false,
    claim_file_descriptor_content_fsync_required_before_process_creation: true,
    directory_entry_durability_guaranteed: false,
    acquisition_root: normalizedAbsolutePath(artifactRoot),
    invocation,
    source_endpoint_before: sourceEndpointBefore,
    executable_endpoint_before: executableEndpointBefore,
    claimed_at_utc: claimedAtUtc,
    evidence_firewall: protocolState.protocol.evidence_firewall,
    claim_boundary: protocolState.protocol.claim_boundary,
  };
}

function makeStreamCapture(stdout, stderr) {
  return {
    schema_version: STREAM_CAPTURE_SCHEMA_VERSION,
    stdout_relative_path: STDOUT_FILE,
    stdout_raw_sha256: sha256Bytes(stdout),
    stdout_byte_count: stdout.length,
    stderr_relative_path: STDERR_FILE,
    stderr_raw_sha256: sha256Bytes(stderr),
    stderr_byte_count: stderr.length,
  };
}

// Property insertion order intentionally matches the existing qualification
// adapter's ordered capture-envelope v1 contract. The on-disk terminal is
// canonicalized independently; validator outcomes reconstruct this exact
// adapter-facing order rather than exposing the parsed terminal object.
function makeAdapterCaptureEnvelope(claim, processObservation, stdout, stderr) {
  return {
    schema_version: ADAPTER_CAPTURE_ENVELOPE_SCHEMA_VERSION,
    capture_role: claim.capture_role,
    stdout_raw_sha256: sha256Bytes(stdout),
    stdout_byte_count: stdout.length,
    stderr_raw_sha256: sha256Bytes(stderr),
    stderr_byte_count: stderr.length,
    process_exit_code: processObservation.exit_code,
    process_started_at_utc: processObservation.process_started_at_utc,
    process_exited_at_utc: processObservation.process_exited_at_utc,
    stdout_captured_at_utc: processObservation.stdout_captured_at_utc,
    machine_identity_sha256: claim.machine_identity_sha256,
    boot_identity_sha256: claim.boot_identity_sha256,
    capture_authoritative: false,
  };
}

function makeTerminal({
  protocolState,
  claim,
  claimRawSha256,
  processObservation,
  stdout,
  stderr,
  sourceEndpointAfter,
  executableEndpointAfter,
  completedAtUtc,
}) {
  validateProcessObservation(processObservation, "process observation");
  validateEndpoint(sourceEndpointAfter, "sourceEndpointAfter");
  validateEndpoint(executableEndpointAfter, "executableEndpointAfter");
  requireUtcTimestamp(completedAtUtc, "completedAtUtc");
  if (
    requireUtcTimestamp(processObservation.stdout_captured_at_utc, "stdout captured") >
    requireUtcTimestamp(completedAtUtc, "completedAtUtc")
  ) {
    throw new Error("terminal completion predates bounded stdout capture");
  }
  const sourceEqual = canonicalJson(claim.source_endpoint_before) === canonicalJson(sourceEndpointAfter);
  const executableEqual =
    canonicalJson(claim.executable_endpoint_before) === canonicalJson(executableEndpointAfter);
  let classification = classifyOutcome(processObservation, stdout, stderr);
  if (!sourceEqual || !executableEqual) {
    classification = {
      ...classification,
      disposition: "unclassified_process_or_output_quarantined",
      capture_candidate: false,
      quarantined: true,
    };
  }
  const captureEnvelope = classification.capture_candidate
    ? makeAdapterCaptureEnvelope(claim, processObservation, stdout, stderr)
    : null;
  const terminalCore = {
    schema_version: EVENT_LOG_SOURCE_AUDIT_ACQUISITION_TERMINAL_SCHEMA_VERSION,
    sequence: 1,
    scope_id: claim.scope_id,
    previous_receipt_sha256: claimRawSha256,
    acquisition_protocol_id: protocolState.protocolId,
    acquisition_protocol_raw_sha256: protocolState.protocolRawSha256,
    qualification_protocol_id: claim.qualification_protocol_id,
    qualification_protocol_raw_sha256: claim.qualification_protocol_raw_sha256,
    capture_role: claim.capture_role,
    execution_backend_id: claim.execution_backend_id,
    process_observation: processObservation,
    stream_capture: makeStreamCapture(stdout, stderr),
    capture_envelope: captureEnvelope,
    audit_discriminator: classification.discriminator,
    disposition: classification.disposition,
    capture_candidate: classification.capture_candidate,
    quarantined: classification.quarantined,
    source_endpoint_before: claim.source_endpoint_before,
    source_endpoint_after: sourceEndpointAfter,
    source_endpoint_equal: sourceEqual,
    executable_endpoint_before: claim.executable_endpoint_before,
    executable_endpoint_after: executableEndpointAfter,
    executable_endpoint_equal: executableEqual,
    continuous_endpoint_stability_proven: false,
    evidence_firewall: protocolState.protocol.evidence_firewall,
    completed_at_utc: completedAtUtc,
  };
  return {
    ...terminalCore,
    terminal_id: contentId(terminalCore),
  };
}

function readRegularFile(rootReal, relativePath, maximumBytes, allowEmpty) {
  const candidate = path.resolve(rootReal, ...relativePath.split("/"));
  const back = path.relative(rootReal, candidate);
  if (back.startsWith("..") || path.isAbsolute(back)) {
    throw new Error(`artifact file escapes root: ${relativePath}`);
  }
  const before = fs.lstatSync(candidate, { bigint: true });
  if (!before.isFile() || before.isSymbolicLink() || before.nlink !== 1n) {
    throw new Error(`artifact member must be one regular single-link file: ${relativePath}`);
  }
  if ((!allowEmpty && before.size === 0n) || before.size > BigInt(maximumBytes)) {
    throw new Error(`artifact member size is outside its budget: ${relativePath}`);
  }
  const descriptor = fs.openSync(candidate, "r");
  try {
    const opened = fs.fstatSync(descriptor, { bigint: true });
    if (
      opened.dev !== before.dev ||
      opened.ino !== before.ino ||
      opened.size !== before.size ||
      opened.mtimeNs !== before.mtimeNs
    ) {
      throw new Error(`artifact member changed before descriptor read: ${relativePath}`);
    }
    const raw = fs.readFileSync(descriptor);
    const after = fs.fstatSync(descriptor, { bigint: true });
    if (
      after.dev !== opened.dev ||
      after.ino !== opened.ino ||
      after.size !== opened.size ||
      after.mtimeNs !== opened.mtimeNs
    ) {
      throw new Error(`artifact member changed during descriptor read: ${relativePath}`);
    }
    return raw;
  } finally {
    fs.closeSync(descriptor);
  }
}

function assertExactArtifactTree(root) {
  const rootStat = fs.lstatSync(root);
  if (!rootStat.isDirectory() || rootStat.isSymbolicLink()) {
    throw new Error("acquisition artifact root must be one real directory");
  }
  const rootReal = fs.realpathSync(root);
  const top = fs.readdirSync(rootReal).sort();
  deepExact(top, [CLAIM_FILE, TERMINAL_FILE, "streams"].sort(), "artifact top-level entries");
  const streamsPath = path.join(rootReal, "streams");
  const streamsStat = fs.lstatSync(streamsPath);
  if (!streamsStat.isDirectory() || streamsStat.isSymbolicLink()) {
    throw new Error("artifact streams member must be one real directory");
  }
  deepExact(
    fs.readdirSync(streamsPath).sort(),
    [path.basename(STDOUT_FILE), path.basename(STDERR_FILE)].sort(),
    "artifact stream entries",
  );
  return rootReal;
}

function validateClaim(claim, protocolState, artifactRoot) {
  exactKeys(
    claim,
    [
      "schema_version",
      "sequence",
      "scope_id",
      "previous_receipt_sha256",
      "acquisition_protocol_id",
      "acquisition_protocol_raw_sha256",
      "qualification_protocol_id",
      "qualification_protocol_raw_sha256",
      "operator_scope_binding_id",
      "machine_identity_sha256",
      "boot_identity_sha256",
      "identity_binding_authoritative",
      "capture_role",
      "execution_backend_id",
      "attempt_number",
      "attempt_budget",
      "retry_budget",
      "attempt_budget_applies_to",
      "same_root_overwrite_or_retry_permitted",
      "cross_root_duplicate_scope_excluded",
      "scope_global_no_retry_proven",
      "claim_file_descriptor_content_fsync_required_before_process_creation",
      "directory_entry_durability_guaranteed",
      "acquisition_root",
      "invocation",
      "source_endpoint_before",
      "executable_endpoint_before",
      "claimed_at_utc",
      "evidence_firewall",
      "claim_boundary",
    ],
    "acquisition claim",
  );
  if (
    claim.schema_version !== EVENT_LOG_SOURCE_AUDIT_ACQUISITION_CLAIM_SCHEMA_VERSION ||
    claim.sequence !== 0 ||
    claim.previous_receipt_sha256 !== null ||
    claim.acquisition_protocol_id !== protocolState.protocolId ||
    claim.acquisition_protocol_raw_sha256 !== protocolState.protocolRawSha256 ||
    claim.qualification_protocol_id !==
      protocolState.protocol.qualification_compatibility.qualification_protocol_id ||
    claim.qualification_protocol_raw_sha256 !==
      protocolState.protocol.qualification_compatibility.qualification_protocol_raw_sha256 ||
    claim.attempt_number !== 1 ||
    claim.attempt_budget !== 1 ||
    claim.retry_budget !== 0 ||
    claim.attempt_budget_applies_to !== "single_artifact_root_only" ||
    claim.same_root_overwrite_or_retry_permitted !== false ||
    claim.cross_root_duplicate_scope_excluded !== false ||
    claim.scope_global_no_retry_proven !== false ||
    claim.claim_file_descriptor_content_fsync_required_before_process_creation !== true ||
    claim.directory_entry_durability_guaranteed !== false ||
    claim.identity_binding_authoritative !== false ||
    claim.acquisition_root !== normalizedAbsolutePath(artifactRoot) ||
    claim.claim_boundary !== protocolState.protocol.claim_boundary
  ) {
    throw new Error("acquisition claim fixed contract drift");
  }
  requireSha256(claim.scope_id, "claim.scope_id");
  requireSha256(claim.operator_scope_binding_id, "claim.operator_scope_binding_id");
  requireSha256(claim.machine_identity_sha256, "claim.machine_identity_sha256");
  requireSha256(claim.boot_identity_sha256, "claim.boot_identity_sha256");
  if (!protocolState.protocol.capture_roles.includes(claim.capture_role)) {
    throw new Error("claim capture role drift");
  }
  if (![PRODUCTION_BACKEND_ID, SYNTHETIC_BACKEND_ID].includes(claim.execution_backend_id)) {
    throw new Error("claim backend drift");
  }
  const expectedScopeId = computeScopeId(
    protocolState,
    claim.operator_scope_binding_id,
    claim.machine_identity_sha256,
    claim.boot_identity_sha256,
    claim.capture_role,
    claim.execution_backend_id,
  );
  if (claim.scope_id !== expectedScopeId) throw new Error("claim deterministic scope drift");
  const claimedInvocation = requireObject(claim.invocation, "claim.invocation");
  const expectedInvocation = invocationContract(
    protocolState.protocol,
    claim.execution_backend_id,
    claimedInvocation.requested_executable,
    claimedInvocation.requested_argv,
  );
  deepExact(claimedInvocation, expectedInvocation, "claim.invocation");
  validateEndpoint(claim.source_endpoint_before, "claim.source_endpoint_before");
  validateEndpoint(claim.executable_endpoint_before, "claim.executable_endpoint_before");
  validateRequestedInvocationEndpoints(
    protocolState.protocol,
    claimedInvocation,
    claim.execution_backend_id,
    claim.source_endpoint_before,
    claim.executable_endpoint_before,
  );
  requireUtcTimestamp(claim.claimed_at_utc, "claim.claimed_at_utc");
  deepExact(claim.evidence_firewall, protocolState.protocol.evidence_firewall, "claim firewall");
}

function validateTerminal(
  terminal,
  protocolState,
  claim,
  claimRawSha256,
  stdout,
  stderr,
) {
  exactKeys(
    terminal,
    [
      "schema_version",
      "sequence",
      "scope_id",
      "previous_receipt_sha256",
      "acquisition_protocol_id",
      "acquisition_protocol_raw_sha256",
      "qualification_protocol_id",
      "qualification_protocol_raw_sha256",
      "capture_role",
      "execution_backend_id",
      "process_observation",
      "stream_capture",
      "capture_envelope",
      "audit_discriminator",
      "disposition",
      "capture_candidate",
      "quarantined",
      "source_endpoint_before",
      "source_endpoint_after",
      "source_endpoint_equal",
      "executable_endpoint_before",
      "executable_endpoint_after",
      "executable_endpoint_equal",
      "continuous_endpoint_stability_proven",
      "evidence_firewall",
      "completed_at_utc",
      "terminal_id",
    ],
    "acquisition terminal",
  );
  if (
    terminal.schema_version !== EVENT_LOG_SOURCE_AUDIT_ACQUISITION_TERMINAL_SCHEMA_VERSION ||
    terminal.sequence !== 1 ||
    terminal.scope_id !== claim.scope_id ||
    terminal.previous_receipt_sha256 !== claimRawSha256 ||
    terminal.acquisition_protocol_id !== protocolState.protocolId ||
    terminal.acquisition_protocol_raw_sha256 !== protocolState.protocolRawSha256 ||
    terminal.qualification_protocol_id !== claim.qualification_protocol_id ||
    terminal.qualification_protocol_raw_sha256 !== claim.qualification_protocol_raw_sha256 ||
    terminal.capture_role !== claim.capture_role ||
    terminal.execution_backend_id !== claim.execution_backend_id ||
    terminal.continuous_endpoint_stability_proven !== false
  ) {
    throw new Error("acquisition terminal lineage drift");
  }
  validateProcessObservation(terminal.process_observation, "terminal.process_observation");
  if (
    terminal.process_observation.stream_outcomes.stdout.persisted_byte_count !==
      stdout.length ||
    terminal.process_observation.stream_outcomes.stderr.persisted_byte_count !==
      stderr.length
  ) {
    throw new Error("terminal stream outcome/file byte-count drift");
  }
  const syntheticBackend = terminal.execution_backend_id === SYNTHETIC_BACKEND_ID;
  if (
    terminal.process_observation.spawn_attempted !== !syntheticBackend ||
    terminal.process_observation.process_started_at_semantics !==
      (syntheticBackend
        ? "synthetic_declaration"
        : "lower_bound_immediately_before_spawn_call") ||
    (syntheticBackend && terminal.process_observation.process_id !== null)
  ) {
    throw new Error("terminal process/backend observation drift");
  }
  exactKeys(
    terminal.stream_capture,
    STREAM_CAPTURE_FIELDS,
    "terminal.stream_capture",
  );
  const expectedStreamCapture = makeStreamCapture(stdout, stderr);
  deepExact(terminal.stream_capture, expectedStreamCapture, "terminal.stream_capture");
  if (
    terminal.stream_capture.schema_version !== STREAM_CAPTURE_SCHEMA_VERSION ||
    terminal.stream_capture.stdout_relative_path !== STDOUT_FILE ||
    terminal.stream_capture.stderr_relative_path !== STDERR_FILE
  ) {
    throw new Error("terminal stream capture drift");
  }
  const classification = classifyOutcome(terminal.process_observation, stdout, stderr);
  const sourceBefore = validateEndpoint(
    terminal.source_endpoint_before,
    "terminal.source_endpoint_before",
  );
  const sourceAfter = validateEndpoint(
    terminal.source_endpoint_after,
    "terminal.source_endpoint_after",
  );
  const executableBefore = validateEndpoint(
    terminal.executable_endpoint_before,
    "terminal.executable_endpoint_before",
  );
  const executableAfter = validateEndpoint(
    terminal.executable_endpoint_after,
    "terminal.executable_endpoint_after",
  );
  deepExact(sourceBefore, claim.source_endpoint_before, "terminal/claim source endpoint");
  deepExact(
    executableBefore,
    claim.executable_endpoint_before,
    "terminal/claim executable endpoint",
  );
  const sourceEqual = canonicalJson(sourceBefore) === canonicalJson(sourceAfter);
  const executableEqual = canonicalJson(executableBefore) === canonicalJson(executableAfter);
  if (!sourceEqual || !executableEqual) {
    classification.disposition = "unclassified_process_or_output_quarantined";
    classification.capture_candidate = false;
    classification.quarantined = true;
  }
  if (
    terminal.source_endpoint_equal !== sourceEqual ||
    terminal.executable_endpoint_equal !== executableEqual ||
    terminal.disposition !== classification.disposition ||
    terminal.capture_candidate !== classification.capture_candidate ||
    terminal.quarantined !== classification.quarantined
  ) {
    throw new Error("terminal derived disposition drift");
  }
  const expectedCaptureEnvelope = classification.capture_candidate
    ? makeAdapterCaptureEnvelope(claim, terminal.process_observation, stdout, stderr)
    : null;
  if (expectedCaptureEnvelope === null) {
    if (terminal.capture_envelope !== null) {
      throw new Error("quarantined terminal must not expose an adapter capture envelope");
    }
  } else {
    exactKeys(
      terminal.capture_envelope,
      ADAPTER_CAPTURE_ENVELOPE_FIELDS,
      "terminal.capture_envelope",
    );
    deepExact(
      terminal.capture_envelope,
      expectedCaptureEnvelope,
      "terminal.capture_envelope",
    );
  }
  deepExact(terminal.audit_discriminator, classification.discriminator, "terminal discriminator");
  deepExact(
    terminal.evidence_firewall,
    protocolState.protocol.evidence_firewall,
    "terminal firewall",
  );
  const claimedAt = requireUtcTimestamp(claim.claimed_at_utc, "claim.claimed_at_utc");
  const completedAt = requireUtcTimestamp(terminal.completed_at_utc, "terminal.completed_at_utc");
  const processStartedAt = requireUtcTimestamp(
    terminal.process_observation.process_started_at_utc,
    "terminal process start",
  );
  const stdoutCapturedAt = requireUtcTimestamp(
    terminal.process_observation.stdout_captured_at_utc,
    "terminal stdout capture",
  );
  if (
    processStartedAt < claimedAt ||
    completedAt < claimedAt ||
    completedAt < stdoutCapturedAt
  ) {
    throw new Error("process or terminal predates claim");
  }
  const terminalId = requireSha256(terminal.terminal_id, "terminal.terminal_id");
  const core = { ...terminal };
  delete core.terminal_id;
  if (terminalId !== contentId(core)) throw new Error("terminal ID drift");
  return {
    classification,
    captureEnvelope: expectedCaptureEnvelope,
    streamCapture: expectedStreamCapture,
  };
}

export function validateEventLogSourceAuditAcquisitionArtifact(options) {
  const input = exactKeys(
    requireObject(options, "acquisition validator options"),
    ["artifactRoot"],
    "acquisition validator options",
  );
  const artifactRoot = path.resolve(requireText(input.artifactRoot, "artifactRoot"));
  const protocolState = loadProtocol({ verifySources: true });
  const rootReal = assertExactArtifactTree(artifactRoot);
  const claimRaw = readRegularFile(rootReal, CLAIM_FILE, 1_048_576, false);
  const terminalRaw = readRegularFile(rootReal, TERMINAL_FILE, 1_048_576, false);
  const stdout = readRegularFile(
    rootReal,
    STDOUT_FILE,
    protocolState.protocol.budgets.stdout_max_bytes,
    true,
  );
  const stderr = readRegularFile(
    rootReal,
    STDERR_FILE,
    protocolState.protocol.budgets.stderr_max_bytes,
    true,
  );
  const claimParsed = materializeJsonIntegers(
    parseJsonStrict(decodeStrictUtf8PreservingBom(claimRaw), "claim"),
  );
  const terminalParsed = materializeJsonIntegers(
    parseJsonStrict(decodeStrictUtf8PreservingBom(terminalRaw), "terminal"),
  );
  if (!claimRaw.equals(canonicalBytes(claimParsed))) {
    throw new Error("claim must be canonical compact JSON plus one LF");
  }
  if (!terminalRaw.equals(canonicalBytes(terminalParsed))) {
    throw new Error("terminal must be canonical compact JSON plus one LF");
  }
  validateClaim(claimParsed, protocolState, artifactRoot);
  const terminalValidation = validateTerminal(
    terminalParsed,
    protocolState,
    claimParsed,
    sha256Bytes(claimRaw),
    stdout,
    stderr,
  );
  return deepFreeze({
    integrityValid: true,
    protocolId: protocolState.protocolId,
    protocolRawSha256: protocolState.protocolRawSha256,
    scopeId: claimParsed.scope_id,
    sameRootOverwriteOrRetryPermitted: false,
    crossRootDuplicateScopeExcluded: false,
    scopeGlobalNoRetryProven: false,
    claimFileDescriptorContentFsyncRequiredBeforeProcessCreation: true,
    directoryEntryDurabilityGuaranteed: false,
    executionBackendId: claimParsed.execution_backend_id,
    disposition: terminalValidation.classification.disposition,
    captureCandidate: terminalValidation.classification.capture_candidate,
    quarantined: terminalValidation.classification.quarantined,
    streamCapture: terminalValidation.streamCapture,
    captureEnvelope: terminalValidation.captureEnvelope,
    terminalId: terminalParsed.terminal_id,
    terminalRawSha256: sha256Bytes(terminalRaw),
    fullRootIdentity: sha256Bytes(terminalRaw),
    captureEnvelopeAuthoritative: false,
    realProcessObservation: false,
    realProvisionerObservation: false,
    sourceConfigProjectionEmitted: false,
    eligibleAsHostQualificationInput: false,
    validatedEligibleAsHostQualificationInput: false,
    qualificationAuthorized: false,
    cudaExecutionAuthorized: false,
    formalEvidenceAuthorized: false,
    productionActiveAuthorized: false,
    appendableProven: false,
    readableProven: false,
    learnableProven: false,
    steerableProven: false,
    fourCapabilityClaimAuthorized: false,
    tamperResistanceProven: false,
    pathParentReparseTrustProven: false,
    continuousEndpointStabilityProven: false,
  });
}

function normalizeSyntheticProcessOutcome(value, protocol) {
  const input = exactKeys(
    requireObject(value, "synthetic processOutcome"),
    [
      "processStartedAtUtc",
      "processExitedAtUtc",
      "streamsClosedAtUtc",
      "exitCode",
      "signal",
      "timedOut",
      "overflowStream",
      "killAttempted",
      "killAttemptCount",
      "spawnErrorName",
      "spawnErrorMessage",
      "stdoutBase64",
      "stderrBase64",
    ],
    "synthetic processOutcome",
  );
  const stdout = decodeBase64Strict(input.stdoutBase64, "processOutcome.stdoutBase64");
  const stderr = decodeBase64Strict(input.stderrBase64, "processOutcome.stderrBase64");
  if (
    stdout.length > protocol.budgets.stdout_max_bytes ||
    stderr.length > protocol.budgets.stderr_max_bytes
  ) {
    throw new Error("synthetic stream exceeds the frozen bounded capture size");
  }
  const spawnErrorName = input.spawnErrorName;
  const spawnErrorMessage = input.spawnErrorMessage;
  if ((spawnErrorName === null) !== (spawnErrorMessage === null)) {
    throw new Error("synthetic spawn error name/message must be all-or-none");
  }
  if (spawnErrorName !== null) {
    requireText(spawnErrorName, "processOutcome.spawnErrorName");
    requireString(spawnErrorMessage, "processOutcome.spawnErrorMessage");
  }
  const processObservation = {
    schema_version: PROCESS_OBSERVATION_SCHEMA_VERSION,
    spawn_attempted: false,
    process_id: null,
    process_started_at_utc: input.processStartedAtUtc,
    process_started_at_semantics: "synthetic_declaration",
    os_process_creation_time_attested: false,
    process_exit_observation: "synthetic_declaration",
    streams_close_observation: "synthetic_declaration",
    process_exited_at_utc: input.processExitedAtUtc,
    streams_closed_at_utc: input.streamsClosedAtUtc,
    stdout_captured_at_utc: input.streamsClosedAtUtc,
    exit_code: input.exitCode,
    signal: input.signal,
    timed_out: input.timedOut,
    hard_cutoff_at_utc: null,
    overall_deadline_exceeded: false,
    finalization_reason: "synthetic_declaration",
    capture_detached_at_hard_cutoff: false,
    termination_confirmed: false,
    process_may_remain_running: false,
    overflow_stream: input.overflowStream,
    kill_attempted: input.killAttempted,
    kill_attempt_count: input.killAttemptCount,
    kill_request_accepted: input.killAttemptCount === 1 ? false : null,
    kill_error_name: null,
    kill_error_message_sha256: null,
    descendants_contained: false,
    spawn_error_name: spawnErrorName,
    spawn_error_message_sha256:
      spawnErrorMessage === null
        ? null
        : sha256Bytes(Buffer.from(spawnErrorMessage, "utf8")),
    capture_complete:
      input.signal === null &&
      input.timedOut === false &&
      input.overflowStream === null &&
      input.killAttemptCount === 0 &&
      spawnErrorName === null,
    stream_outcomes: {
      stdout: {
        schema_version: STREAM_OUTCOME_SCHEMA_VERSION,
        pipe_end_observed: false,
        pipe_close_observed: false,
        forcibly_detached: false,
        observed_byte_count: stdout.length,
        persisted_byte_count: stdout.length,
        persistence_complete: true,
        capture_error_stage: null,
        capture_error_name: null,
        capture_error_message_sha256: null,
        persistence_error_stage: null,
        persistence_error_name: null,
        persistence_error_message_sha256: null,
      },
      stderr: {
        schema_version: STREAM_OUTCOME_SCHEMA_VERSION,
        pipe_end_observed: false,
        pipe_close_observed: false,
        forcibly_detached: false,
        observed_byte_count: stderr.length,
        persisted_byte_count: stderr.length,
        persistence_complete: true,
        capture_error_stage: null,
        capture_error_name: null,
        capture_error_message_sha256: null,
        persistence_error_stage: null,
        persistence_error_name: null,
        persistence_error_message_sha256: null,
      },
    },
  };
  validateProcessObservation(processObservation, "synthetic process observation");
  return { processObservation, stdout, stderr };
}

function createSyntheticEventLogSourceAuditAcquisitionArtifact(options) {
  const input = exactKeys(
    requireObject(options, "synthetic acquisition options"),
    [
      "artifactRoot",
      "captureRole",
      "operatorScopeBindingId",
      "machineIdentitySha256",
      "bootIdentitySha256",
      "processOutcome",
    ],
    "synthetic acquisition options",
  );
  const protocolState = loadProtocol({ verifySources: true });
  const artifactRoot = path.resolve(requireText(input.artifactRoot, "artifactRoot"));
  const captureRole = requireText(input.captureRole, "captureRole");
  const operatorScopeBindingId = requireSha256(
    input.operatorScopeBindingId,
    "operatorScopeBindingId",
  );
  const machineIdentitySha256 = requireSha256(
    input.machineIdentitySha256,
    "machineIdentitySha256",
  );
  const bootIdentitySha256 = requireSha256(
    input.bootIdentitySha256,
    "bootIdentitySha256",
  );
  const normalized = normalizeSyntheticProcessOutcome(
    input.processOutcome,
    protocolState.protocol,
  );
  const requestedExecutable = "synthetic://windows-powershell-5.1";
  const requestedArgv = fixedRequestedArgv(protocolState.protocol);
  const sourceEndpoint = syntheticEndpoint("reviewed-provisioner-source");
  const executableEndpoint = syntheticEndpoint("fixed-windows-powershell-executable");
  ensureCreateOnlyRoot(artifactRoot);
  const claim = makeClaim({
    protocolState,
    artifactRoot,
    operatorScopeBindingId,
    machineIdentitySha256,
    bootIdentitySha256,
    captureRole,
    backendId: SYNTHETIC_BACKEND_ID,
    requestedExecutable,
    requestedArgv,
    sourceEndpointBefore: sourceEndpoint,
    executableEndpointBefore: executableEndpoint,
    claimedAtUtc: normalized.processObservation.process_started_at_utc,
  });
  const claimWritten = writeCreateJson(path.join(artifactRoot, CLAIM_FILE), claim);
  writeCreateFile(path.join(artifactRoot, ...STDOUT_FILE.split("/")), normalized.stdout);
  writeCreateFile(path.join(artifactRoot, ...STDERR_FILE.split("/")), normalized.stderr);
  const terminal = makeTerminal({
    protocolState,
    claim,
    claimRawSha256: claimWritten.rawSha256,
    processObservation: normalized.processObservation,
    stdout: normalized.stdout,
    stderr: normalized.stderr,
    sourceEndpointAfter: sourceEndpoint,
    executableEndpointAfter: executableEndpoint,
    completedAtUtc: normalized.processObservation.streams_closed_at_utc,
  });
  writeCreateJson(path.join(artifactRoot, TERMINAL_FILE), terminal);
  return validateEventLogSourceAuditAcquisitionArtifact({ artifactRoot });
}

function fixedPowerShellExecutablePath() {
  const systemRoot = process.env.SystemRoot;
  if (
    typeof systemRoot !== "string" ||
    !/^[A-Za-z]:\\[^\0]*$/u.test(systemRoot) ||
    systemRoot.endsWith("\\")
  ) {
    throw new Error("SystemRoot is unavailable or noncanonical");
  }
  const executable = path.win32.join(
    systemRoot,
    "System32",
    "WindowsPowerShell",
    "v1.0",
    "powershell.exe",
  );
  const expected = `${systemRoot}\\System32\\WindowsPowerShell\\v1.0\\powershell.exe`;
  if (executable.toLowerCase() !== expected.toLowerCase()) {
    throw new Error("fixed Windows PowerShell path construction drift");
  }
  return executable;
}

function normalizeThrownError(error) {
  return error instanceof Error ? error : new Error(`non-Error thrown: ${String(error)}`);
}

function frozenErrorIdentity(error) {
  const normalized = normalizeThrownError(error);
  return Object.freeze({
    name: normalized.name || "Error",
    messageSha256: sha256Bytes(Buffer.from(normalized.message, "utf8")),
  });
}

function reportLateErrorWarning(origin, error) {
  const identity = frozenErrorIdentity(error);
  process.emitWarning(
    `late ${origin} error after Audit acquisition finalization: ${identity.name}:${identity.messageSha256}`,
    {
      code: "VOLVENCE_AUDIT_ACQUISITION_LATE_ERROR",
      type: "VolvenceAuditAcquisitionWarning",
    },
  );
}

function createBoundedStreamState(maximumBytes) {
  return {
    maximumBytes,
    chunks: [],
    observedByteCount: 0,
    overflowed: false,
    pipeEndObserved: false,
    pipeCloseObserved: false,
    forciblyDetached: false,
    captureError: null,
  };
}

function recordCaptureError(state, stage, error) {
  if (state.captureError !== null) return;
  state.captureError = Object.freeze({ stage, ...frozenErrorIdentity(error) });
}

function appendBoundedChunk(state, chunk, streamName, onDisqualifyingFailure) {
  try {
    const bytes = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    const remaining = state.maximumBytes - state.observedByteCount;
    if (remaining > 0) {
      const accepted = bytes.subarray(0, remaining);
      if (accepted.length > 0) {
        const stableCopy = Buffer.from(accepted);
        state.chunks.push(stableCopy);
        state.observedByteCount += stableCopy.length;
      }
    }
    if (bytes.length > remaining && !state.overflowed) {
      state.overflowed = true;
      onDisqualifyingFailure({ overflowStream: streamName });
    }
  } catch (error) {
    recordCaptureError(state, "buffer", error);
    onDisqualifyingFailure({ overflowStream: null });
  }
}

function readBoundedCaptureDescriptor(descriptor, expectedByteCount, maximumBytes, label) {
  if (expectedByteCount < 0 || expectedByteCount > maximumBytes) {
    throw new Error(`${label} same-descriptor read exceeds its frozen budget`);
  }
  const before = fs.fstatSync(descriptor, { bigint: true });
  if (
    !before.isFile() ||
    before.nlink !== 1n ||
    before.size !== BigInt(expectedByteCount)
  ) {
    throw new Error(`${label} same-descriptor identity/size drift before read`);
  }
  const raw = Buffer.alloc(expectedByteCount);
  let offset = 0;
  while (offset < raw.length) {
    const read = fs.readSync(descriptor, raw, offset, raw.length - offset, offset);
    if (read <= 0) throw new Error(`${label} same-descriptor read ended early`);
    offset += read;
  }
  const after = fs.fstatSync(descriptor, { bigint: true });
  if (
    after.dev !== before.dev ||
    after.ino !== before.ino ||
    after.size !== before.size ||
    after.mtimeNs !== before.mtimeNs ||
    after.nlink !== before.nlink
  ) {
    throw new Error(`${label} descriptor changed during bounded read`);
  }
  sha256Bytes(raw);
  return raw;
}

async function runFixedAuditProcess({
  executable,
  argv,
  cwd,
  stdoutDescriptor,
  stderrDescriptor,
  protocol,
  spawnProcess = spawn,
  persistenceHooks = null,
  lateErrorReporter = reportLateErrorWarning,
}) {
  const stdoutState = createBoundedStreamState(protocol.budgets.stdout_max_bytes);
  const stderrState = createBoundedStreamState(protocol.budgets.stderr_max_bytes);
  let child = null;
  let spawnAttempted = false;
  let timedOut = false;
  let killAttemptCount = 0;
  let killRequestAccepted = null;
  let killError = null;
  let spawnError = null;
  let processId = null;
  let startedAtUtc = null;
  let exitedAtUtc = null;
  let streamsClosedAtUtc = null;
  let exitCode = null;
  let signal = null;
  let processExitObservation = "not_observed";
  let streamsCloseObservation = "not_observed";
  let finalizationReason = null;
  let hardCutoffAtUtc = null;
  let overflowStream = null;
  let lifecycleSettled = false;
  let captureActive = true;
  let timeoutTimer = null;
  let postKillTimer = null;
  let overallTimer = null;
  const streamRegistrations = [];
  const childRegistrations = [];
  const killOnce = () => {
    if (killAttemptCount !== 0) return;
    killAttemptCount = 1;
    if (child === null) {
      killRequestAccepted = false;
      return;
    }
    try {
      killRequestAccepted = child.kill();
    } catch (error) {
      killRequestAccepted = false;
      killError = normalizeThrownError(error);
    }
  };
  const clearSupervisionTimers = () => {
    if (timeoutTimer !== null) clearTimeout(timeoutTimer);
    if (postKillTimer !== null) clearTimeout(postKillTimer);
    if (overallTimer !== null) clearTimeout(overallTimer);
  };
  let settleLifecycle;
  const lifecyclePromise = new Promise((resolve) => {
    settleLifecycle = resolve;
  });
  const detachCaptureAtHardCutoff = () => {
    captureActive = false;
    stdoutState.forciblyDetached = true;
    stderrState.forciblyDetached = true;
    for (const registration of streamRegistrations) {
      try {
        registration.stream.destroy();
      } catch (error) {
        recordCaptureError(registration.state, "pipe", error);
      }
    }
    if (child !== null && typeof child.unref === "function") {
      try {
        child.unref();
      } catch (error) {
        if (spawnError === null) spawnError = normalizeThrownError(error);
      }
    }
  };
  const finalizeOnce = (reason) => {
    if (lifecycleSettled) return;
    lifecycleSettled = true;
    finalizationReason = reason;
    if (["post_kill_grace_cutoff", "overall_hard_cutoff"].includes(reason)) {
      hardCutoffAtUtc = nowUtc();
      detachCaptureAtHardCutoff();
    } else {
      captureActive = false;
    }
    clearSupervisionTimers();
    settleLifecycle();
  };
  const schedulePostKillCutoff = () => {
    if (postKillTimer !== null || lifecycleSettled) return;
    postKillTimer = setTimeout(
      () => finalizeOnce("post_kill_grace_cutoff"),
      protocol.budgets.post_kill_pipe_drain_grace_milliseconds,
    );
  };
  const requestStop = ({ requestedOverflowStream = null } = {}) => {
    if (requestedOverflowStream !== null && overflowStream === null) {
      overflowStream = requestedOverflowStream;
    }
    killOnce();
    schedulePostKillCutoff();
  };
  const registerStream = (stream, state, streamName) => {
    const onData = (chunk) => {
      if (!captureActive) return;
      appendBoundedChunk(state, chunk, streamName, ({ overflowStream: observed }) =>
        requestStop({ requestedOverflowStream: observed }),
      );
    };
    const onError = (error) => {
      if (!captureActive) {
        lateErrorReporter(`${streamName} pipe`, error);
        return;
      }
      recordCaptureError(state, "pipe", error);
      requestStop();
    };
    const onEnd = () => {
      if (captureActive) state.pipeEndObserved = true;
    };
    const onClose = () => {
      if (captureActive) state.pipeCloseObserved = true;
    };
    stream.on("data", onData);
    stream.on("error", onError);
    stream.on("end", onEnd);
    stream.on("close", onClose);
    streamRegistrations.push({
      stream,
      state,
      handlers: { data: onData, error: onError, end: onEnd, close: onClose },
    });
  };
  const registerChild = (eventName, handler) => {
    child.on(eventName, handler);
    childRegistrations.push({ eventName, handler });
  };
  const removeLateErrorGuardians = () => {
    for (const registration of streamRegistrations) {
      registration.stream.removeListener("error", registration.handlers.error);
    }
    if (child !== null) {
      for (const registration of childRegistrations) {
        if (["error", "close"].includes(registration.eventName)) {
          child.removeListener(registration.eventName, registration.handler);
        }
      }
    }
  };
  try {
    try {
      // This timestamp is only a caller-side lower bound immediately before
      // spawn(). It is not an OS process-creation-time attestation.
      startedAtUtc = nowUtc();
      spawnAttempted = true;
      child = spawnProcess(executable, argv, {
        cwd,
        shell: false,
        windowsHide: true,
        stdio: ["ignore", "pipe", "pipe"],
      });
      processId = Number.isSafeInteger(child.pid) && child.pid > 0 ? child.pid : null;
    } catch (error) {
      spawnError = normalizeThrownError(error);
      finalizeOnce("spawn_throw");
    }
    if (!lifecycleSettled) {
      if (child.stdout === null || child.stderr === null) {
        spawnError = new Error("fixed spawn did not expose both captured stream pipes");
        if (child.stdout === null) {
          recordCaptureError(stdoutState, "pipe", spawnError);
        }
        if (child.stderr === null) {
          recordCaptureError(stderrState, "pipe", spawnError);
        }
        requestStop();
      } else {
        registerStream(child.stdout, stdoutState, "stdout");
        registerStream(child.stderr, stderrState, "stderr");
      }
      registerChild("error", (error) => {
        if (lifecycleSettled) {
          lateErrorReporter("child process", error);
          return;
        }
        if (spawnError === null) spawnError = normalizeThrownError(error);
        requestStop();
      });
      registerChild("exit", (code, exitSignal) => {
        if (lifecycleSettled) return;
        processExitObservation = "child_exit_event";
        exitCode = code;
        signal = exitSignal;
        exitedAtUtc = nowUtc();
      });
      registerChild("close", (code, closeSignal) => {
        if (lifecycleSettled) {
          removeLateErrorGuardians();
          return;
        }
        streamsCloseObservation = "child_close_event";
        streamsClosedAtUtc = nowUtc();
        if (processExitObservation === "not_observed") {
          processExitObservation = "child_close_fallback";
          exitedAtUtc = streamsClosedAtUtc;
          exitCode = code;
          signal = closeSignal;
        }
        finalizeOnce("child_close");
      });
      timeoutTimer = setTimeout(() => {
        if (lifecycleSettled) return;
        timedOut = true;
        requestStop();
      }, protocol.budgets.timeout_milliseconds);
      overallTimer = setTimeout(
        () => finalizeOnce("overall_hard_cutoff"),
        protocol.budgets.overall_supervision_deadline_milliseconds,
      );
    }
    await lifecyclePromise;
    await new Promise((resolve) => setImmediate(resolve));
  } finally {
    clearSupervisionTimers();
    for (const registration of streamRegistrations) {
      for (const [eventName, handler] of Object.entries(registration.handlers)) {
        if (hardCutoffAtUtc !== null && eventName === "error") continue;
        registration.stream.removeListener(eventName, handler);
      }
    }
    if (child !== null) {
      for (const registration of childRegistrations) {
        if (
          hardCutoffAtUtc !== null &&
          ["error", "close"].includes(registration.eventName)
        ) {
          continue;
        }
        child.removeListener(registration.eventName, registration.handler);
      }
    }
  }
  const persistStream = (descriptor, state, label) => {
    const writeSync = persistenceHooks?.writeSync ?? fs.writeSync;
    const fsyncSync = persistenceHooks?.fsyncSync ?? fs.fsyncSync;
    let persistedByteCount = 0;
    let persistenceError = null;
    writeChunks: for (const chunk of state.chunks) {
      let offset = 0;
      while (offset < chunk.length) {
        let written;
        try {
          written = writeSync(descriptor, chunk, offset, chunk.length - offset);
          if (written <= 0) {
            throw new Error(`${label} descriptor made no write progress`);
          }
        } catch (error) {
          persistenceError = Object.freeze({
            stage: "write",
            ...frozenErrorIdentity(error),
          });
          break writeChunks;
        }
        offset += written;
        persistedByteCount += written;
      }
    }
    try {
      fsyncSync(descriptor);
    } catch (error) {
      if (persistenceError === null) {
        persistenceError = Object.freeze({
          stage: "fsync",
          ...frozenErrorIdentity(error),
        });
      }
    }
    const readCapture =
      persistenceHooks?.readBoundedCaptureDescriptor ??
      readBoundedCaptureDescriptor;
    const raw = readCapture(
      descriptor,
      persistedByteCount,
      state.maximumBytes,
      label,
    );
    return {
      raw,
      outcome: {
        schema_version: STREAM_OUTCOME_SCHEMA_VERSION,
        pipe_end_observed: state.pipeEndObserved,
        pipe_close_observed: state.pipeCloseObserved,
        forcibly_detached: state.forciblyDetached,
        observed_byte_count: state.observedByteCount,
        persisted_byte_count: persistedByteCount,
        persistence_complete:
          persistenceError === null &&
          persistedByteCount === state.observedByteCount,
        capture_error_stage:
          state.captureError === null ? null : state.captureError.stage,
        capture_error_name:
          state.captureError === null ? null : state.captureError.name,
        capture_error_message_sha256:
          state.captureError === null ? null : state.captureError.messageSha256,
        persistence_error_stage:
          persistenceError === null ? null : persistenceError.stage,
        persistence_error_name:
          persistenceError === null ? null : persistenceError.name,
        persistence_error_message_sha256:
          persistenceError === null ? null : persistenceError.messageSha256,
      },
    };
  };
  const stdoutPersisted = persistStream(stdoutDescriptor, stdoutState, "stdout");
  const stderrPersisted = persistStream(stderrDescriptor, stderrState, "stderr");
  const stdout = stdoutPersisted.raw;
  const stderr = stderrPersisted.raw;
  const stdoutCapturedAtUtc = nowUtc();
  const streamOutcomes = {
    stdout: stdoutPersisted.outcome,
    stderr: stderrPersisted.outcome,
  };
  const streamsClean = Object.values(streamOutcomes).every(
    (outcome) =>
      outcome.pipe_end_observed &&
      outcome.pipe_close_observed &&
      outcome.persistence_complete &&
      outcome.capture_error_stage === null &&
      outcome.persistence_error_stage === null &&
      outcome.forcibly_detached === false,
  );
  const captureComplete =
    finalizationReason === "child_close" &&
    processExitObservation === "child_exit_event" &&
    streamsCloseObservation === "child_close_event" &&
    streamsClean &&
    timedOut === false &&
    hardCutoffAtUtc === null &&
    overflowStream === null &&
    spawnError === null &&
    signal === null &&
    killAttemptCount === 0;
  return {
    processObservation: {
      schema_version: PROCESS_OBSERVATION_SCHEMA_VERSION,
      spawn_attempted: spawnAttempted,
      process_id: processId,
      process_started_at_utc: startedAtUtc,
      process_started_at_semantics: "lower_bound_immediately_before_spawn_call",
      os_process_creation_time_attested: false,
      process_exit_observation: processExitObservation,
      streams_close_observation: streamsCloseObservation,
      process_exited_at_utc: exitedAtUtc,
      streams_closed_at_utc: streamsClosedAtUtc,
      stdout_captured_at_utc: stdoutCapturedAtUtc,
      exit_code: exitCode,
      signal,
      timed_out: timedOut,
      hard_cutoff_at_utc: hardCutoffAtUtc,
      overall_deadline_exceeded: finalizationReason === "overall_hard_cutoff",
      finalization_reason: finalizationReason,
      capture_detached_at_hard_cutoff: hardCutoffAtUtc !== null,
      termination_confirmed:
        spawnAttempted && processExitObservation !== "not_observed",
      process_may_remain_running:
        spawnAttempted && processExitObservation === "not_observed",
      overflow_stream: overflowStream,
      kill_attempted: killAttemptCount === 1,
      kill_attempt_count: killAttemptCount,
      kill_request_accepted: killRequestAccepted,
      kill_error_name: killError === null ? null : killError.name,
      kill_error_message_sha256:
        killError === null
          ? null
          : sha256Bytes(Buffer.from(killError.message, "utf8")),
      descendants_contained: false,
      spawn_error_name: spawnError === null ? null : spawnError.name,
      spawn_error_message_sha256:
        spawnError === null
          ? null
          : sha256Bytes(Buffer.from(spawnError.message, "utf8")),
      capture_complete: captureComplete,
      stream_outcomes: streamOutcomes,
    },
    stdout,
    stderr,
  };
}

// This backend is deliberately private and has no exported or __testing route.
// It is retained so a later audited protocol can review the exact process
// boundary without allowing the current public surface to execute it.
async function acquireRealEventLogSourceAudit({
  artifactRoot,
  captureRole,
  operatorScopeBindingId,
  machineIdentitySha256,
  bootIdentitySha256,
}) {
  const protocolState = loadProtocol({ verifySources: true });
  if (protocolState.protocol.execution.production_entrypoint_enabled !== false) {
    throw new Error("private backend requires the production-disabled protocol revision");
  }
  const root = path.resolve(requireText(artifactRoot, "artifactRoot"));
  const executable = fixedPowerShellExecutablePath();
  const requestedArgv = fixedRequestedArgv(protocolState.protocol);
  deepExact(
    requestedArgv,
    [
      "-NoLogo",
      "-NoProfile",
      "-NonInteractive",
      "-EncodedCommand",
      buildSourceBindingLauncher(
        protocolState.protocol.execution.source_execution_binding,
      ).encodedCommand,
    ],
    "private fixed Audit argv",
  );
  const sourcePath = resolveRepositoryPath(
    protocolState.protocol.execution.source_execution_binding.provisioner_relative_path,
  );
  const sourceEndpointBefore = observeLocalFileEndpoint(sourcePath, true);
  const executableEndpointBefore = observeLocalFileEndpoint(executable, false);
  ensureCreateOnlyRoot(root);
  const claim = makeClaim({
    protocolState,
    artifactRoot: root,
    operatorScopeBindingId,
    machineIdentitySha256,
    bootIdentitySha256,
    captureRole,
    backendId: PRODUCTION_BACKEND_ID,
    requestedExecutable: executable,
    requestedArgv,
    sourceEndpointBefore,
    executableEndpointBefore,
    claimedAtUtc: nowUtc(),
  });
  const claimWritten = writeCreateJson(path.join(root, CLAIM_FILE), claim);
  const stdoutPath = path.join(root, ...STDOUT_FILE.split("/"));
  const stderrPath = path.join(root, ...STDERR_FILE.split("/"));
  const stdoutDescriptor = fs.openSync(stdoutPath, "wx+");
  let stderrDescriptor;
  try {
    stderrDescriptor = fs.openSync(stderrPath, "wx+");
  } catch (error) {
    fs.closeSync(stdoutDescriptor);
    throw error;
  }
  let processResult;
  try {
    processResult = await runFixedAuditProcess({
      executable,
      argv: requestedArgv,
      cwd: REPOSITORY_ROOT,
      stdoutDescriptor,
      stderrDescriptor,
      protocol: protocolState.protocol,
    });
  } finally {
    fs.closeSync(stdoutDescriptor);
    fs.closeSync(stderrDescriptor);
  }
  const sourceEndpointAfter = observeLocalFileEndpoint(sourcePath, true);
  const executableEndpointAfter = observeLocalFileEndpoint(executable, false);
  const terminal = makeTerminal({
    protocolState,
    claim,
    claimRawSha256: claimWritten.rawSha256,
    processObservation: processResult.processObservation,
    stdout: processResult.stdout,
    stderr: processResult.stderr,
    sourceEndpointAfter,
    executableEndpointAfter,
    completedAtUtc: nowUtc(),
  });
  writeCreateJson(path.join(root, TERMINAL_FILE), terminal);
  return validateEventLogSourceAuditAcquisitionArtifact({ artifactRoot: root });
}

void acquireRealEventLogSourceAudit;

export function acquireEventLogSourceAudit() {
  throw new Error(PRODUCTION_DISABLED_MESSAGE);
}

function frozenSourceBindingLauncherObservationForTesting() {
  const loaded = loadStrictJsonFile(PROTOCOL_PATH, "acquisition protocol", false);
  const binding = requireObject(
    requireObject(loaded.value.execution, "protocol.execution")
      .source_execution_binding,
    "protocol.execution.source_execution_binding",
  );
  const launcher = buildSourceBindingLauncher(binding);
  return deepFreeze({
    sourceUtf8Sha256: launcher.sourceUtf8Sha256,
    utf16leSha256: launcher.utf16leSha256,
    encodedCommandSha256: sha256Bytes(Buffer.from(launcher.encodedCommand, "ascii")),
    encodedCommand: launcher.encodedCommand,
  });
}

function frozenUtf8BomCanonicalizationObservationForTesting() {
  const withoutBom = Buffer.from("fixture\r\n", "utf8");
  const withBom = Buffer.concat([Buffer.from([0xef, 0xbb, 0xbf]), withoutBom]);
  return deepFreeze({
    withBomLfCanonicalSha256: lfCanonicalSha256Bytes(withBom),
    withoutBomLfCanonicalSha256: lfCanonicalSha256Bytes(withoutBom),
    expectedPreservedBomLfCanonicalSha256: sha256Bytes(
      Buffer.from("\ufefffixture\n", "utf8"),
    ),
  });
}

function buildWindowsDescendantPipeFixtureEncodedCommand() {
  const source = `Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
using System.Text;
public static class VolvencePipeFixture {
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    public struct STARTUPINFO {
        public int cb; public string lpReserved; public string lpDesktop; public string lpTitle;
        public int dwX; public int dwY; public int dwXSize; public int dwYSize;
        public int dwXCountChars; public int dwYCountChars; public int dwFillAttribute;
        public int dwFlags; public short wShowWindow; public short cbReserved2;
        public IntPtr lpReserved2; public IntPtr hStdInput; public IntPtr hStdOutput; public IntPtr hStdError;
    }
    [StructLayout(LayoutKind.Sequential)]
    public struct PROCESS_INFORMATION {
        public IntPtr hProcess; public IntPtr hThread; public int dwProcessId; public int dwThreadId;
    }
    [DllImport("kernel32.dll", SetLastError = true)] static extern IntPtr GetStdHandle(int kind);
    [DllImport("kernel32.dll", SetLastError = true)] static extern bool SetHandleInformation(IntPtr handle, int mask, int flags);
    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    static extern bool CreateProcess(string application, StringBuilder commandLine, IntPtr processAttributes,
        IntPtr threadAttributes, bool inheritHandles, int creationFlags, IntPtr environment,
        string currentDirectory, ref STARTUPINFO startupInfo, out PROCESS_INFORMATION processInformation);
    [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr handle);
    public static int Spawn(string executable) {
        IntPtr stdoutHandle = GetStdHandle(-11);
        IntPtr stderrHandle = GetStdHandle(-12);
        if (!SetHandleInformation(stdoutHandle, 1, 1) || !SetHandleInformation(stderrHandle, 1, 1)) {
            throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error());
        }
        STARTUPINFO startup = new STARTUPINFO();
        startup.cb = Marshal.SizeOf(typeof(STARTUPINFO));
        startup.dwFlags = 0x00000100;
        startup.hStdInput = GetStdHandle(-10);
        startup.hStdOutput = stdoutHandle;
        startup.hStdError = stderrHandle;
        PROCESS_INFORMATION process;
        StringBuilder command = new StringBuilder("\\\"" + executable + "\\\" -n 30 127.0.0.1");
        if (!CreateProcess(executable, command, IntPtr.Zero, IntPtr.Zero, true, 0x08000000,
            IntPtr.Zero, null, ref startup, out process)) {
            throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error());
        }
        CloseHandle(process.hThread);
        CloseHandle(process.hProcess);
        return process.dwProcessId;
    }
}
"@
$executable = "$env:SystemRoot\\System32\\PING.EXE"
$descendantProcessId = [VolvencePipeFixture]::Spawn($executable)
[Console]::Out.WriteLine("descendant_pid={0}", $descendantProcessId)
exit 0
`;
  return Buffer.from(source, "utf16le").toString("base64");
}

async function exerciseFixedAuditLifecycleScenarioForTesting(scenario) {
  const allowed = [
    "clean_close",
    "kill_false_never_close",
    "kill_throw_never_close",
    "kill_true_never_close",
    "exit_without_close",
    "stdout_pipe_error",
    "close_without_end",
    "late_error_after_cutoff",
    "write_failure",
    "fsync_failure",
    "readback_failure",
    "descendant_holds_pipe",
  ];
  if (!allowed.includes(scenario)) throw new Error("unsupported fixed lifecycle scenario");
  if (scenario === "descendant_holds_pipe" && process.platform !== "win32") {
    throw new Error("descendant_holds_pipe fixture requires Windows");
  }
  const loaded = loadStrictJsonFile(PROTOCOL_PATH, "acquisition protocol", false);
  const descendantFixture = scenario === "descendant_holds_pipe";
  const protocol = {
    ...loaded.value,
    budgets: {
      stdout_max_bytes: 4096,
      stderr_max_bytes: 4096,
      timeout_milliseconds: descendantFixture ? 6000 : 30,
      post_kill_pipe_drain_grace_milliseconds: descendantFixture ? 2000 : 30,
      overall_supervision_deadline_milliseconds: descendantFixture ? 8000 : 60,
    },
  };
  const nonce = `${process.pid}-${crypto.randomUUID()}`;
  const stdoutPath = path.join(os.tmpdir(), `volvence-audit-stdout-${nonce}.bin`);
  const stderrPath = path.join(os.tmpdir(), `volvence-audit-stderr-${nonce}.bin`);
  let stdoutDescriptor = null;
  let stderrDescriptor = null;
  const scheduled = [];
  let fakeChild = null;
  let killCallCount = 0;
  const lateErrors = [];
  const schedule = (callback, milliseconds) => {
    const timer = setTimeout(callback, milliseconds);
    scheduled.push(timer);
  };
  const makeFakeChild = () => {
    const child = new EventEmitter();
    child.stdout = new PassThrough();
    child.stderr = new PassThrough();
    child.pid = 4242;
    child.unref = () => undefined;
    child.kill = () => {
      killCallCount += 1;
      if (scenario === "kill_throw_never_close") {
        throw new Error("fixture kill failure");
      }
      return scenario !== "kill_false_never_close" && scenario !== "exit_without_close";
    };
    if (
      ["clean_close", "write_failure", "fsync_failure", "readback_failure"].includes(
        scenario,
      )
    ) {
      schedule(() => {
        child.stdout.end(Buffer.from("ok", "utf8"));
        child.stderr.end();
      }, 1);
      schedule(() => child.emit("exit", 0, null), 8);
      schedule(() => child.emit("close", 0, null), 12);
    } else if (scenario === "close_without_end") {
      schedule(() => {
        child.stdout.emit("close");
        child.stderr.emit("close");
      }, 1);
      schedule(() => child.emit("exit", 0, null), 8);
      schedule(() => child.emit("close", 0, null), 12);
    } else if (scenario === "exit_without_close") {
      schedule(() => child.emit("exit", 0, null), 5);
    } else if (scenario === "stdout_pipe_error") {
      schedule(() => child.stdout.emit("error", new Error("fixture stdout error")), 5);
    }
    return child;
  };
  const spawnProcess =
    scenario === "descendant_holds_pipe"
      ? spawn
      : () => {
          fakeChild = makeFakeChild();
          return fakeChild;
        };
  const persistenceHooks =
    scenario === "write_failure"
      ? {
          writeSync(descriptor, buffer, offset, length) {
            if (descriptor === stdoutDescriptor && offset > 0) {
              throw new Error("fixture write failure");
            }
            const boundedLength =
              descriptor === stdoutDescriptor ? Math.min(1, length) : length;
            return fs.writeSync(descriptor, buffer, offset, boundedLength);
          },
        }
      : scenario === "fsync_failure"
        ? {
            fsyncSync(descriptor) {
              if (descriptor === stdoutDescriptor) {
                throw new Error("fixture fsync failure");
              }
              return fs.fsyncSync(descriptor);
            },
          }
        : scenario === "readback_failure"
          ? {
              readBoundedCaptureDescriptor() {
                throw new Error("fixture same-descriptor readback failure");
              },
            }
        : null;
  const startedMilliseconds = Date.now();
  let descendantPid = null;
  let descendantCleanup = null;
  const ensureDescendantStopped = async () => {
    if (descendantPid === null || descendantCleanup?.terminationConfirmed === true) return;
    let killAccepted = false;
    try {
      killAccepted = process.kill(descendantPid);
    } catch (error) {
      if (error?.code !== "ESRCH") throw error;
    }
    let terminationConfirmed = false;
    for (let attempt = 0; attempt < 80; attempt += 1) {
      try {
        process.kill(descendantPid, 0);
      } catch (error) {
        if (error?.code === "ESRCH") {
          terminationConfirmed = true;
          break;
        }
        throw error;
      }
      await new Promise((resolve) => setTimeout(resolve, 25));
    }
    descendantCleanup = { descendantPid, killAccepted, terminationConfirmed };
    if (!terminationConfirmed) {
      throw new Error("descendant fixture cleanup was not confirmed");
    }
  };
  try {
    stdoutDescriptor = fs.openSync(stdoutPath, "wx+");
    stderrDescriptor = fs.openSync(stderrPath, "wx+");
    let result;
    try {
      result = await runFixedAuditProcess({
        executable: descendantFixture
          ? fixedPowerShellExecutablePath()
          : process.execPath,
        argv:
          descendantFixture
            ? [
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-EncodedCommand",
                buildWindowsDescendantPipeFixtureEncodedCommand(),
              ]
            : ["synthetic-fixed-lifecycle-fixture"],
        cwd: REPOSITORY_ROOT,
        stdoutDescriptor,
        stderrDescriptor,
        protocol,
        spawnProcess,
        persistenceHooks,
        lateErrorReporter(origin, error) {
          lateErrors.push({ origin, ...frozenErrorIdentity(error) });
        },
      });
    } catch (error) {
      if (scenario !== "readback_failure") throw error;
      return deepFreeze({
        scenario,
        elapsedMilliseconds: Date.now() - startedMilliseconds,
        killCallCount,
        boundedFailure: frozenErrorIdentity(error),
      });
    }
    if (descendantFixture) {
      const stdoutText = result.stdout.toString("utf8");
      const match = /(?:^|\r?\n)descendant_pid=([1-9][0-9]*)(?:\r?\n|$)/u.exec(
        stdoutText,
      );
      if (match === null) throw new Error("descendant fixture did not publish its PID");
      descendantPid = Number(match[1]);
    }
    validateProcessObservation(result.processObservation, "fixed lifecycle fixture");
    let lateErrorGuardianCountsBeforeClose = null;
    let lateErrorGuardianCountsAfterClose = null;
    if (scenario === "late_error_after_cutoff") {
      fakeChild.emit("error", new Error("fixture late child error"));
      fakeChild.stdout.emit("error", new Error("fixture late stdout error"));
      fakeChild.stderr.emit("error", new Error("fixture late stderr error"));
      lateErrorGuardianCountsBeforeClose = {
        child: fakeChild.listenerCount("error"),
        stdout: fakeChild.stdout.listenerCount("error"),
        stderr: fakeChild.stderr.listenerCount("error"),
      };
      fakeChild.emit("close", null, null);
      lateErrorGuardianCountsAfterClose = {
        child: fakeChild.listenerCount("error"),
        stdout: fakeChild.stdout.listenerCount("error"),
        stderr: fakeChild.stderr.listenerCount("error"),
      };
    }
    if (descendantFixture) {
      await ensureDescendantStopped();
    }
    return deepFreeze({
      scenario,
      elapsedMilliseconds: Date.now() - startedMilliseconds,
      killCallCount,
      processObservation: result.processObservation,
      stdoutBase64: result.stdout.toString("base64"),
      stderrBase64: result.stderr.toString("base64"),
      descendantCleanup,
      lateErrors,
      lateErrorGuardianCountsBeforeClose,
      lateErrorGuardianCountsAfterClose,
      fakeChildListenerCounts:
        fakeChild === null
          ? null
          : {
              error: fakeChild.listenerCount("error"),
              exit: fakeChild.listenerCount("exit"),
              close: fakeChild.listenerCount("close"),
            },
      fakeStreamListenerCounts:
        fakeChild === null
          ? null
          : {
              stdoutError: fakeChild.stdout.listenerCount("error"),
              stderrError: fakeChild.stderr.listenerCount("error"),
            },
    });
  } finally {
    const cleanupFailures = [];
    for (const timer of scheduled) clearTimeout(timer);
    try {
      await ensureDescendantStopped();
    } catch (error) {
      cleanupFailures.push(normalizeThrownError(error));
    }
    for (const descriptor of [stdoutDescriptor, stderrDescriptor]) {
      if (descriptor === null) continue;
      try {
        fs.closeSync(descriptor);
      } catch (error) {
        cleanupFailures.push(normalizeThrownError(error));
      }
    }
    for (const candidate of [stdoutPath, stderrPath]) {
      try {
        if (fs.existsSync(candidate)) fs.unlinkSync(candidate);
      } catch (error) {
        cleanupFailures.push(normalizeThrownError(error));
      }
    }
    if (cleanupFailures.length > 0) {
      throw new AggregateError(cleanupFailures, "fixed lifecycle fixture cleanup failed");
    }
  }
}

async function exerciseSourceBindingLauncherFixtureForTesting(scenario) {
  const allowed = [
    "valid_exit_2_and_lock",
    "normal_return_with_stale_last_exit_code",
    "raw_mismatch_lf_equal",
    "lf_mismatch_raw_equal",
    "utf8_bom",
    "invalid_utf8",
  ];
  if (!allowed.includes(scenario)) throw new Error("unsupported source-binding fixture");
  if (process.platform !== "win32") {
    throw new Error("source-binding launcher fixture requires Windows");
  }
  const nonce = `${process.pid}-${crypto.randomUUID()}`;
  const sourceLeaf = `volvence-source-binding-${nonce}.ps1`;
  const sourcePath = path.join(REPOSITORY_ROOT, sourceLeaf);
  const movedPath = `${sourcePath}.moved`;
  const markerPath = `${sourcePath}.opened`;
  const stdoutPath = path.join(os.tmpdir(), `volvence-binding-stdout-${nonce}.bin`);
  const stderrPath = path.join(os.tmpdir(), `volvence-binding-stderr-${nonce}.bin`);
  let sourceBytes;
  if (scenario === "invalid_utf8") {
    sourceBytes = Buffer.from([0xc3, 0x28]);
  } else if (scenario === "utf8_bom") {
    sourceBytes = Buffer.concat([
      Buffer.from([0xef, 0xbb, 0xbf]),
      Buffer.from("param([ValidateSet('Audit')][string]$Mode)\nexit 0\n", "utf8"),
    ]);
  } else if (scenario === "normal_return_with_stale_last_exit_code") {
    sourceBytes = Buffer.from(
      [
        "param([ValidateSet('Audit')][string]$Mode)",
        "$global:LASTEXITCODE = 0",
        "return",
        "",
      ].join("\n"),
      "utf8",
    );
  } else {
    const lines = [
      "param([ValidateSet('Audit')][string]$Mode)",
      "[IO.File]::WriteAllText(($PSCommandPath + '.opened'), 'opened')",
      "Start-Sleep -Milliseconds 500",
      "[pscustomobject]@{Mode=$Mode;Path=$PSCommandPath;Root=$PSScriptRoot}|ConvertTo-Json -Compress",
      "exit 2",
      "",
    ];
    sourceBytes = Buffer.from(
      scenario === "raw_mismatch_lf_equal"
        ? lines.join("\r\n")
        : lines.join("\n"),
      "utf8",
    );
  }
  const decodedForLf =
    scenario === "invalid_utf8"
      ? null
      : decodeStrictUtf8PreservingBom(sourceBytes);
  const lfBytes =
    decodedForLf === null
      ? sourceBytes
      : Buffer.from(decodedForLf.replace(/\r\n?/gu, "\n"), "utf8");
  const expectedRawSha256 =
    scenario === "raw_mismatch_lf_equal"
      ? sha256Bytes(lfBytes)
      : sha256Bytes(sourceBytes);
  const expectedLfCanonicalSha256 =
    scenario === "lf_mismatch_raw_equal"
      ? sha256Bytes(Buffer.from("fixture mismatched LF pin\n", "utf8"))
      : sha256Bytes(lfBytes);
  const binding = {
    provisioner_relative_path: sourceLeaf,
    provisioner_raw_sha256: expectedRawSha256,
    provisioner_lf_canonical_sha256: expectedLfCanonicalSha256,
  };
  const encodedCommand = buildSourceBindingLauncher(binding).encodedCommand;
  let stdoutDescriptor = null;
  let stderrDescriptor = null;
  const protocol = {
    budgets: {
      stdout_max_bytes: 65_536,
      stderr_max_bytes: 65_536,
      timeout_milliseconds: 10_000,
      post_kill_pipe_drain_grace_milliseconds: 2_000,
      overall_supervision_deadline_milliseconds: 12_000,
    },
  };
  let renameBlockedWhileHandleHeld = null;
  let renameBlockErrorCode = null;
  let handleReleasedAfterExit = false;
  try {
    writeCreateFile(sourcePath, sourceBytes);
    stdoutDescriptor = fs.openSync(stdoutPath, "wx+");
    stderrDescriptor = fs.openSync(stderrPath, "wx+");
    const processPromise = runFixedAuditProcess({
      executable: fixedPowerShellExecutablePath(),
      argv: [
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-EncodedCommand",
        encodedCommand,
      ],
      cwd: REPOSITORY_ROOT,
      stdoutDescriptor,
      stderrDescriptor,
      protocol,
    });
    const lockProbePromise =
      scenario === "valid_exit_2_and_lock"
        ? (async () => {
            for (let attempt = 0; attempt < 160; attempt += 1) {
              if (fs.existsSync(markerPath)) break;
              await new Promise((resolve) => setTimeout(resolve, 25));
            }
            if (!fs.existsSync(markerPath)) {
              renameBlockErrorCode = "marker_missing";
              return;
            }
            try {
              fs.renameSync(sourcePath, movedPath);
              renameBlockedWhileHandleHeld = false;
              fs.renameSync(movedPath, sourcePath);
            } catch (error) {
              renameBlockedWhileHandleHeld = true;
              renameBlockErrorCode = error?.code ?? error?.name ?? "unknown";
            }
          })()
        : Promise.resolve();
    const [processSettled, lockProbeSettled] = await Promise.allSettled([
      processPromise,
      lockProbePromise,
    ]);
    if (processSettled.status === "rejected") throw processSettled.reason;
    if (lockProbeSettled.status === "rejected") throw lockProbeSettled.reason;
    const result = processSettled.value;
    validateProcessObservation(result.processObservation, "source-binding fixture");
    fs.renameSync(sourcePath, movedPath);
    fs.renameSync(movedPath, sourcePath);
    handleReleasedAfterExit = true;
    return deepFreeze({
      scenario,
      processObservation: result.processObservation,
      stdoutUtf8: result.stdout.toString("utf8"),
      stderrUtf8: result.stderr.toString("utf8"),
      renameBlockedWhileHandleHeld,
      renameBlockErrorCode,
      handleReleasedAfterExit,
      sourceRawSha256: sha256Bytes(sourceBytes),
      expectedRawSha256,
      sourceLfCanonicalSha256: sha256Bytes(lfBytes),
      expectedLfCanonicalSha256: binding.provisioner_lf_canonical_sha256,
    });
  } finally {
    const cleanupFailures = [];
    for (const descriptor of [stdoutDescriptor, stderrDescriptor]) {
      if (descriptor === null) continue;
      try {
        fs.closeSync(descriptor);
      } catch (error) {
        cleanupFailures.push(normalizeThrownError(error));
      }
    }
    for (const candidate of [
      sourcePath,
      movedPath,
      markerPath,
      stdoutPath,
      stderrPath,
    ]) {
      try {
        if (fs.existsSync(candidate)) fs.unlinkSync(candidate);
      } catch (error) {
        cleanupFailures.push(normalizeThrownError(error));
      }
    }
    if (cleanupFailures.length > 0) {
      throw new AggregateError(cleanupFailures, "source-binding fixture cleanup failed");
    }
  }
}

export const __testing = Object.freeze({
  createSyntheticEventLogSourceAuditAcquisitionArtifact,
  exerciseFixedAuditLifecycleScenarioForTesting,
  exerciseSourceBindingLauncherFixtureForTesting,
  frozenSourceBindingLauncherObservationForTesting,
  frozenUtf8BomCanonicalizationObservationForTesting,
});
