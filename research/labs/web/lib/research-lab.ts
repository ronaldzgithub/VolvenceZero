export type LifecycleStage =
  | 'NEEDS_TASK_DESIGN'
  | 'AWAITING_A0'
  | 'PREFLIGHT'
  | 'RESEARCH_RUNNING'
  | 'RESEARCH_COMPLETE'
  | 'CANDIDATE_RETAINED'
  | 'FORMAL_VALIDATION'
  | 'AWAITING_A1'
  | 'SHADOW'
  | 'AWAITING_A2'
  | 'ACTIVE'
  | 'ROLLED_BACK'
  | 'BLOCKED';

export interface NamedCount {
  name: string;
  count: number;
}

export interface PortalWarning {
  code: string;
  message: string;
  source: string;
  severity: 'warning' | 'error';
  task_id: string | null;
}

export interface ArtifactRef {
  kind: string;
  locator: string;
  sha256: string;
  artifact_id: string | null;
}

export interface PraxistRunSnapshot {
  run_id: string;
  state: string;
  source: string;
  pid: number | null;
  task_path: string;
  run_dir: string | null;
  generation: number | null;
  findings_total: number;
  peers_total: number;
  peer_health: NamedCount[];
  runtime: string | null;
  model_provider: string | null;
  model: string | null;
  started_at: string | null;
  updated_at: string | null;
}

export interface ResearchLabItem {
  item_id: string;
  task_id: string;
  research_mode: 'volvence_promotion' | 'external_simulation';
  claim_id: string;
  title: string;
  objective: string;
  owner: string;
  capability_axes: string[];
  release_target: string;
  lifecycle: {
    stage: LifecycleStage;
    next_stage: LifecycleStage | null;
    blocking_reason: string | null;
    last_transition_at: string | null;
  };
  authority: {
    a0_research_start_authorized: boolean;
    formal_validation_status: string;
    modification_gate_decision: string;
    authorized_wiring: string;
    runtime_wiring: string;
    target_adapter_apply_required: boolean;
    production_default_changed: boolean;
    evaluation_is_learning_source: boolean;
  };
  evidence: {
    development: string;
    formal: string;
    shadow: string;
    canary: string;
  };
  bindings: ArtifactRef[];
  run: PraxistRunSnapshot | null;
  available_actions: string[];
  warnings: PortalWarning[];
  updated_at: string | null;
}

export interface ResearchLabSnapshot {
  schema_version: 'volvence-research-lab-snapshot.v1';
  generated_at: string;
  revision: string;
  repo_revision: string;
  summary: {
    registered_tasks: number;
    stage_counts: NamedCount[];
    active_runs: number;
    blocked: number;
    awaiting_human: number;
    production_active: number;
  };
  source_health: Array<{
    source: string;
    status: 'healthy' | 'degraded' | 'unavailable';
    artifacts_seen: number;
    detail: string;
  }>;
  items: ResearchLabItem[];
  warnings: PortalWarning[];
}

export type SupportedCommandAction =
  | 'submit_external'
  | 'review_a0'
  | 'reconcile'
  | 'record_external_handoff'
  | 'import_candidate'
  | 'authorize_shadow'
  | 'authorize_active'
  | 'rollback';

export interface ResearchLabSession {
  schema_version: 'volvence-research-lab-session.v1';
  mutations_enabled: boolean;
  csrf_token: string | null;
  supported_actions: SupportedCommandAction[];
}

interface BaseCommandPayload {
  snapshot_revision: string;
  task_id: string;
  actor: string;
  reason: string;
}

export interface SubmitExternalCommandPayload {
  snapshot_revision: string;
  domain_id: string;
  descriptor_locator: string;
  descriptor_id: string;
  descriptor_sha256: string;
  actor: string;
  reason: string;
}

export interface A0CommandPayload extends BaseCommandPayload {
  artifact_id: string;
  artifact_sha256: string;
  decision: 'approve' | 'reject';
}

export interface ReconcileCommandPayload extends BaseCommandPayload {
  artifact_id: string;
  artifact_sha256: string;
}

export interface ImportCandidateCommandPayload extends BaseCommandPayload {
  task_artifact_id: string;
  task_sha256: string;
  handoff_sha256: string;
  run_id: string;
}

export interface AuthorizeCommandPayload extends BaseCommandPayload {
  task_artifact_id: string;
  task_sha256: string;
  candidate_artifact_id: string;
  candidate_sha256: string;
  validation_sha256: string;
  gate_sha256: string;
  previous_receipt_id: string | null;
  previous_receipt_sha256: string | null;
}

export interface RollbackCommandPayload extends BaseCommandPayload {
  receipt_id: string;
  receipt_sha256: string;
}

export interface ResearchLabCommandPayloadByAction {
  submit_external: SubmitExternalCommandPayload;
  review_a0: A0CommandPayload;
  reconcile: ReconcileCommandPayload;
  record_external_handoff: ReconcileCommandPayload;
  import_candidate: ImportCandidateCommandPayload;
  authorize_shadow: AuthorizeCommandPayload;
  authorize_active: AuthorizeCommandPayload;
  rollback: RollbackCommandPayload;
}

export interface ResearchLabCommandResult {
  schema_version: 'volvence-research-lab-command-result.v1';
  action: SupportedCommandAction;
  task_id: string;
  outcome: string;
  message: string;
  previous_revision: string;
  current_revision: string;
  binding: {
    kind: string;
    artifact_id: string | null;
    sha256: string;
  };
  input_bindings: Array<{
    kind: string;
    artifact_id: string | null;
    sha256: string;
  }>;
}

export async function fetchResearchLabSnapshot(
  signal?: AbortSignal,
): Promise<ResearchLabSnapshot> {
  const response = await fetch('/api/v1/snapshot', {
    cache: 'no-store',
    headers: { Accept: 'application/json' },
    signal,
  });
  if (!response.ok) {
    throw new Error(`Research Lab API returned HTTP ${response.status}`);
  }
  const payload: unknown = await response.json();
  if (!isResearchLabSnapshot(payload)) {
    throw new Error('Research Lab API returned an incompatible snapshot');
  }
  return payload;
}

export async function fetchResearchLabSession(
  signal?: AbortSignal,
): Promise<ResearchLabSession> {
  const response = await fetch('/api/v1/session', {
    cache: 'no-store',
    headers: { Accept: 'application/json' },
    signal,
  });
  if (!response.ok) {
    throw new Error(`Research Lab session returned HTTP ${response.status}`);
  }
  const payload: unknown = await response.json();
  if (!isSession(payload)) {
    throw new Error('Research Lab API returned an incompatible session');
  }
  return payload;
}

export async function submitResearchLabCommand<
  Action extends SupportedCommandAction,
>(
  action: Action,
  session: ResearchLabSession,
  payload: ResearchLabCommandPayloadByAction[Action],
): Promise<ResearchLabCommandResult> {
  if (!session.mutations_enabled || !session.csrf_token) {
    throw new Error('Controlled mutations are disabled on the local API');
  }
  if (!session.supported_actions.includes(action)) {
    throw new Error(`${action} is not enabled by the local API session`);
  }
  const endpoints: Record<SupportedCommandAction, string> = {
    submit_external: '/api/v1/external/requests',
    review_a0: '/api/v1/a0/review',
    reconcile: '/api/v1/reconcile',
    record_external_handoff: '/api/v1/external/handoff',
    import_candidate: '/api/v1/candidates/import',
    authorize_shadow: '/api/v1/a1/authorize-shadow',
    authorize_active: '/api/v1/a2/authorize-active',
    rollback: '/api/v1/rollback',
  };
  const endpoint = endpoints[action];
  const response = await fetch(endpoint, {
    method: 'POST',
    cache: 'no-store',
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
      'X-Research-Lab-CSRF': session.csrf_token,
    },
    body: JSON.stringify(payload),
  });
  const body: unknown = await response.json();
  if (!response.ok) {
    const message =
      isRecord(body) && typeof body.message === 'string'
        ? body.message
        : `Research Lab command returned HTTP ${response.status}`;
    throw new Error(message);
  }
  if (!isCommandResult(body)) {
    throw new Error('Research Lab API returned an incompatible command result');
  }
  return body;
}

export function isResearchLabSnapshot(
  value: unknown,
): value is ResearchLabSnapshot {
  if (!isRecord(value)) return false;
  return (
    value.schema_version === 'volvence-research-lab-snapshot.v1' &&
    typeof value.generated_at === 'string' &&
    typeof value.revision === 'string' &&
    typeof value.repo_revision === 'string' &&
    isSnapshotSummary(value.summary) &&
    Array.isArray(value.source_health) &&
    value.source_health.every(isSourceHealth) &&
    Array.isArray(value.items) &&
    value.items.every(isResearchLabItem) &&
    Array.isArray(value.warnings) &&
    value.warnings.every(isPortalWarning)
  );
}

function isSnapshotSummary(value: unknown): boolean {
  return (
    isRecord(value) &&
    isNumber(value.registered_tasks) &&
    Array.isArray(value.stage_counts) &&
    value.stage_counts.every(isNamedCount) &&
    isNumber(value.active_runs) &&
    isNumber(value.blocked) &&
    isNumber(value.awaiting_human) &&
    isNumber(value.production_active)
  );
}

function isSourceHealth(value: unknown): boolean {
  return (
    isRecord(value) &&
    typeof value.source === 'string' &&
    (value.status === 'healthy' ||
      value.status === 'degraded' ||
      value.status === 'unavailable') &&
    isNumber(value.artifacts_seen) &&
    typeof value.detail === 'string'
  );
}

function isResearchLabItem(value: unknown): boolean {
  if (!isRecord(value)) return false;
  return (
    typeof value.item_id === 'string' &&
    typeof value.task_id === 'string' &&
    (value.research_mode === 'volvence_promotion' ||
      value.research_mode === 'external_simulation') &&
    typeof value.claim_id === 'string' &&
    typeof value.title === 'string' &&
    typeof value.objective === 'string' &&
    typeof value.owner === 'string' &&
    isStringArray(value.capability_axes) &&
    typeof value.release_target === 'string' &&
    isLifecycle(value.lifecycle) &&
    isAuthority(value.authority) &&
    isEvidence(value.evidence) &&
    Array.isArray(value.bindings) &&
    value.bindings.every(isArtifactRef) &&
    (value.run === null || isPraxistRun(value.run)) &&
    isStringArray(value.available_actions) &&
    Array.isArray(value.warnings) &&
    value.warnings.every(isPortalWarning) &&
    isNullableString(value.updated_at)
  );
}

function isLifecycle(value: unknown): boolean {
  return (
    isRecord(value) &&
    isLifecycleStage(value.stage) &&
    (value.next_stage === null || isLifecycleStage(value.next_stage)) &&
    isNullableString(value.blocking_reason) &&
    isNullableString(value.last_transition_at)
  );
}

function isAuthority(value: unknown): boolean {
  return (
    isRecord(value) &&
    typeof value.a0_research_start_authorized === 'boolean' &&
    typeof value.formal_validation_status === 'string' &&
    typeof value.modification_gate_decision === 'string' &&
    typeof value.authorized_wiring === 'string' &&
    typeof value.runtime_wiring === 'string' &&
    typeof value.target_adapter_apply_required === 'boolean' &&
    typeof value.production_default_changed === 'boolean' &&
    typeof value.evaluation_is_learning_source === 'boolean'
  );
}

function isEvidence(value: unknown): boolean {
  return (
    isRecord(value) &&
    typeof value.development === 'string' &&
    typeof value.formal === 'string' &&
    typeof value.shadow === 'string' &&
    typeof value.canary === 'string'
  );
}

function isArtifactRef(value: unknown): boolean {
  return (
    isRecord(value) &&
    typeof value.kind === 'string' &&
    typeof value.locator === 'string' &&
    typeof value.sha256 === 'string' &&
    isNullableString(value.artifact_id)
  );
}

function isPraxistRun(value: unknown): boolean {
  return (
    isRecord(value) &&
    typeof value.run_id === 'string' &&
    typeof value.state === 'string' &&
    typeof value.source === 'string' &&
    isNullableNumber(value.pid) &&
    typeof value.task_path === 'string' &&
    isNullableString(value.run_dir) &&
    isNullableNumber(value.generation) &&
    isNumber(value.findings_total) &&
    isNumber(value.peers_total) &&
    Array.isArray(value.peer_health) &&
    value.peer_health.every(isNamedCount) &&
    isNullableString(value.runtime) &&
    isNullableString(value.model_provider) &&
    isNullableString(value.model) &&
    isNullableString(value.started_at) &&
    isNullableString(value.updated_at)
  );
}

function isPortalWarning(value: unknown): boolean {
  return (
    isRecord(value) &&
    typeof value.code === 'string' &&
    typeof value.message === 'string' &&
    typeof value.source === 'string' &&
    (value.severity === 'warning' || value.severity === 'error') &&
    isNullableString(value.task_id)
  );
}

function isNamedCount(value: unknown): boolean {
  return (
    isRecord(value) && typeof value.name === 'string' && isNumber(value.count)
  );
}

function isLifecycleStage(value: unknown): value is LifecycleStage {
  return (
    value === 'NEEDS_TASK_DESIGN' ||
    value === 'AWAITING_A0' ||
    value === 'PREFLIGHT' ||
    value === 'RESEARCH_RUNNING' ||
    value === 'RESEARCH_COMPLETE' ||
    value === 'CANDIDATE_RETAINED' ||
    value === 'FORMAL_VALIDATION' ||
    value === 'AWAITING_A1' ||
    value === 'SHADOW' ||
    value === 'AWAITING_A2' ||
    value === 'ACTIVE' ||
    value === 'ROLLED_BACK' ||
    value === 'BLOCKED'
  );
}

function isStringArray(value: unknown): value is string[] {
  return (
    Array.isArray(value) && value.every((item) => typeof item === 'string')
  );
}

function isNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function isNullableNumber(value: unknown): value is number | null {
  return value === null || isNumber(value);
}

function isNullableString(value: unknown): value is string | null {
  return value === null || typeof value === 'string';
}

function isSession(value: unknown): value is ResearchLabSession {
  if (!isRecord(value)) return false;
  return (
    value.schema_version === 'volvence-research-lab-session.v1' &&
    typeof value.mutations_enabled === 'boolean' &&
    (typeof value.csrf_token === 'string' || value.csrf_token === null) &&
    Array.isArray(value.supported_actions) &&
    value.supported_actions.every(isSupportedCommandAction)
  );
}

function isCommandResult(value: unknown): value is ResearchLabCommandResult {
  if (!isRecord(value)) return false;
  return (
    value.schema_version === 'volvence-research-lab-command-result.v1' &&
    isSupportedCommandAction(value.action) &&
    typeof value.task_id === 'string' &&
    typeof value.outcome === 'string' &&
    typeof value.message === 'string' &&
    typeof value.previous_revision === 'string' &&
    typeof value.current_revision === 'string' &&
    isCommandBinding(value.binding) &&
    Array.isArray(value.input_bindings) &&
    value.input_bindings.every(isCommandBinding)
  );
}

function isCommandBinding(value: unknown): boolean {
  return (
    isRecord(value) &&
    typeof value.kind === 'string' &&
    (typeof value.artifact_id === 'string' || value.artifact_id === null) &&
    typeof value.sha256 === 'string'
  );
}

function isSupportedCommandAction(
  value: unknown,
): value is SupportedCommandAction {
  return (
    value === 'submit_external' ||
    value === 'review_a0' ||
    value === 'reconcile' ||
    value === 'record_external_handoff' ||
    value === 'import_candidate' ||
    value === 'authorize_shadow' ||
    value === 'authorize_active' ||
    value === 'rollback'
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
