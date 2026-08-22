#!/usr/bin/env node

import { acquireEventLogSourceAudit } from "../packages/vz-runtime/src/volvence_zero/offline_evidence/windows_event_log_source_audit_acquisition.mjs";

try {
  acquireEventLogSourceAudit();
} catch (error) {
  process.stderr.write(`${error.name}: ${error.message}\n`);
  process.exitCode = 2;
}
