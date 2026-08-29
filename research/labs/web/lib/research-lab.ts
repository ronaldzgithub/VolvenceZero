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
  | 'review_a0'
  | 'reconcile'
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
  review_a0: A0CommandPayload;
  reconcile: ReconcileCommandPayload;
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
  if (!isSnapshot(payload)) {
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
    review_a0: '/api/v1/a0/review',
    reconcile: '/api/v1/reconcile',
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

function isSnapshot(value: unknown): value is ResearchLabSnapshot {
  if (!isRecord(value)) return false;
  if (value.schema_version !== 'volvence-research-lab-snapshot.v1')
    return false;
  if (
    typeof value.revision !== 'string' ||
    typeof value.generated_at !== 'string'
  )
    return false;
  if (!Array.isArray(value.items) || !Array.isArray(value.source_health))
    return false;
  if (!isRecord(value.summary)) return false;
  return (
    typeof value.summary.registered_tasks === 'number' &&
    typeof value.summary.active_runs === 'number' &&
    typeof value.summary.awaiting_human === 'number' &&
    typeof value.summary.production_active === 'number'
  );
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
    value === 'review_a0' ||
    value === 'reconcile' ||
    value === 'import_candidate' ||
    value === 'authorize_shadow' ||
    value === 'authorize_active' ||
    value === 'rollback'
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
