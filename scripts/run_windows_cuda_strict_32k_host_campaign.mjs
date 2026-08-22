#!/usr/bin/env node
/** CLI for the independent Windows/CUDA strict 32K outer host campaign. */

import path from "node:path";
import process from "node:process";

import {
  preregisterHostCampaign,
  runHostCampaign,
  validateHostCampaign,
} from "../packages/vz-runtime/src/volvence_zero/offline_evidence/windows_cuda_strict_32k_host_campaign.mjs";

function usage() {
  return [
    "Usage:",
    "  node scripts/run_windows_cuda_strict_32k_host_campaign.mjs preregister --host-qualification-terminal PATH --python-executable PATH",
    "  node scripts/run_windows_cuda_strict_32k_host_campaign.mjs run --campaign-root PATH",
    "  node scripts/run_windows_cuda_strict_32k_host_campaign.mjs validate-existing --campaign-root PATH",
  ].join("\n");
}

function parseArguments(argv) {
  if (argv.length === 0) throw new Error(usage());
  const command = argv[0];
  if (!new Set(["preregister", "run", "validate-existing"]).has(command)) {
    throw new Error(`unknown command: ${command}\n${usage()}`);
  }
  const options = Object.create(null);
  for (let index = 1; index < argv.length; index += 2) {
    const name = argv[index];
    const value = argv[index + 1];
    if (!name?.startsWith("--") || value === undefined || value.startsWith("--")) {
      throw new Error(`invalid command argument near ${name ?? "<end>"}\n${usage()}`);
    }
    const key = name.slice(2);
    if (Object.hasOwn(options, key)) throw new Error(`duplicate option: ${name}`);
    options[key] = value;
  }
  const allowed =
    command === "preregister"
      ? new Set(["host-qualification-terminal", "python-executable"])
      : new Set(["campaign-root"]);
  for (const key of Object.keys(options)) {
    if (!allowed.has(key)) throw new Error(`unsupported option for ${command}: --${key}`);
  }
  if (command === "preregister") {
    if (!options["host-qualification-terminal"] || !options["python-executable"]) {
      throw new Error(`preregister requires qualification terminal and Python executable\n${usage()}`);
    }
  } else if (!options["campaign-root"]) {
    throw new Error(`${command} requires --campaign-root\n${usage()}`);
  }
  return { command, options };
}

function printableResult(result) {
  return {
    status: result.status,
    campaign_artifact_id: result.campaignArtifactId ?? null,
    scope_id: result.scopeId,
    lease_id: result.leaseId,
    protocol_id: result.protocolId,
    child_protocol_id: result.childProtocolId,
    passed: result.passed ?? false,
    verdict: result.verdict ?? result.status,
    failure_codes: result.failureCodes ?? [],
    campaign_root: path.resolve(result.campaignRoot),
  };
}

function main(argv) {
  const { command, options } = parseArguments(argv);
  let result;
  if (command === "preregister") {
    result = preregisterHostCampaign({
      hostQualificationTerminalPath: path.resolve(options["host-qualification-terminal"]),
      pythonExecutable: path.resolve(options["python-executable"]),
    });
  } else if (command === "run") {
    result = runHostCampaign({
      campaignRoot: path.resolve(options["campaign-root"]),
    });
  } else {
    result = validateHostCampaign({
      campaignRoot: path.resolve(options["campaign-root"]),
    });
  }
  process.stdout.write(`${JSON.stringify(printableResult(result))}\n`);
  if (result.status === "preregistered") return command === "preregister" ? 0 : 3;
  if (result.status === "incomplete_consumed") return 3;
  return result.passed ? 0 : 2;
}

try {
  process.exitCode = main(process.argv.slice(2));
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  process.stderr.write(`${JSON.stringify({ error: message })}\n`);
  process.exitCode = process.argv[2] === "validate-existing" ? 4 : 1;
}
