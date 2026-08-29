'use client';

import { useMemo, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  Check,
  CircleOff,
  PackagePlus,
  Play,
  RotateCcw,
  ShieldCheck,
  UserRoundCheck,
} from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import {
  submitResearchLabCommand,
  type ArtifactRef,
  type ResearchLabCommandResult,
  type ResearchLabItem,
  type ResearchLabSession,
  type ResearchLabSnapshot,
  type SupportedCommandAction,
} from '@/lib/research-lab';

interface ResearchCommandWorkbenchProps {
  item: ResearchLabItem | null;
  snapshot: ResearchLabSnapshot | null;
  session: ResearchLabSession | null;
  sessionError: string | null;
  onRefresh: () => void;
}

interface CommandBindings {
  request: ArtifactRef | null;
  task: ArtifactRef | null;
  handoff: ArtifactRef | null;
  candidate: ArtifactRef | null;
  validation: ArtifactRef | null;
  gate: ArtifactRef | null;
  receipt: ArtifactRef | null;
}

type ReviewDecision = 'approve' | 'reject';

const commandCopy: Record<
  SupportedCommandAction,
  { title: string; description: string; submit: string }
> = {
  review_a0: {
    title: 'Review exact A0 scope',
    description:
      'This writes one immutable APPROVE or REJECT decision for the frozen ResearchRequest. It does not authorize formal validation or production wiring.',
    submit: 'Submit exact A0 review',
  },
  reconcile: {
    title: 'Run one bounded reconciliation',
    description:
      'Forge will recheck approval, binding bytes, host capacity, doctor, resolve, and the global reconcile lock. A start is possible only when every owner gate passes.',
    submit: 'Run one reconcile pass',
  },
  import_candidate: {
    title: 'Seal completed Praxist handoff',
    description:
      'Forge will revalidate the exact Task, run root, generation boundary, result summary, and every candidate byte. The resulting Candidate remains DISABLED.',
    submit: 'Import exact Candidate',
  },
  authorize_shadow: {
    title: 'Authorize A1 SHADOW boundary',
    description:
      'This issues an exact Forge authorization receipt from loop-external formal evidence and ModificationGate review. Target-owned SHADOW apply is still separate.',
    submit: 'Authorize exact SHADOW',
  },
  authorize_active: {
    title: 'Authorize A2 ACTIVE boundary',
    description:
      'This requires fresh formal/canary evidence, a fresh Gate review, and the exact previous SHADOW receipt. Target-owned ACTIVE apply remains separate.',
    submit: 'Authorize exact ACTIVE',
  },
  rollback: {
    title: 'Authorize adjacent rollback',
    description:
      'Forge derives the only legal downgrade from the current receipt: ACTIVE to SHADOW or SHADOW to DISABLED. This cannot increase runtime authority.',
    submit: 'Authorize exact rollback',
  },
};

export function ResearchCommandWorkbench({
  item,
  snapshot,
  session,
  sessionError,
  onRefresh,
}: ResearchCommandWorkbenchProps) {
  const [open, setOpen] = useState(false);
  const [action, setAction] = useState<SupportedCommandAction | null>(null);
  const [actor, setActor] = useState('');
  const [reason, setReason] = useState('');
  const [decision, setDecision] = useState<ReviewDecision>('approve');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ResearchLabCommandResult | null>(null);

  const bindings = useMemo(
    () => ({
      request: findBinding(item, 'research request'),
      task: findBinding(item, 'task'),
      handoff: findBinding(item, 'praxist handoff'),
      candidate: findBinding(item, 'candidate'),
      validation: findBinding(item, 'validation'),
      gate: findBinding(item, 'gate'),
      receipt: findBinding(item, 'receipt'),
    }),
    [item],
  );
  const mutationsReady = Boolean(
    session?.mutations_enabled && session.csrf_token,
  );
  const available = new Set(item?.available_actions ?? []);

  const begin = (nextAction: SupportedCommandAction) => {
    setAction(nextAction);
    setActor('');
    setReason('');
    setDecision('approve');
    setError(null);
    setResult(null);
    setOpen(true);
  };

  const submit = async () => {
    if (!action || !item || !snapshot || !session) return;
    if (!commandBindingsReady(action, item, bindings)) {
      setError('The current snapshot is missing an exact command binding.');
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const base = {
        snapshot_revision: snapshot.revision,
        task_id: item.task_id,
        actor: actor.trim(),
        reason: reason.trim(),
      };
      let response: ResearchLabCommandResult;
      switch (action) {
        case 'review_a0':
          response = await submitResearchLabCommand(action, session, {
            ...base,
            artifact_id: requiredId(bindings.request),
            artifact_sha256: requiredSha(bindings.request),
            decision,
          });
          break;
        case 'reconcile':
          response = await submitResearchLabCommand(action, session, {
            ...base,
            artifact_id: requiredId(bindings.request),
            artifact_sha256: requiredSha(bindings.request),
          });
          break;
        case 'import_candidate':
          response = await submitResearchLabCommand(action, session, {
            ...base,
            task_artifact_id: requiredId(bindings.task),
            task_sha256: requiredSha(bindings.task),
            handoff_sha256: requiredSha(bindings.handoff),
            run_id: item.run?.run_id ?? '',
          });
          break;
        case 'authorize_shadow': {
          const previous =
            item.authority.target_adapter_apply_required &&
            item.authority.authorized_wiring === 'disabled'
              ? bindings.receipt
              : null;
          response = await submitResearchLabCommand(action, session, {
            ...base,
            ...authorizationPayload(bindings, previous),
          });
          break;
        }
        case 'authorize_active':
          response = await submitResearchLabCommand(action, session, {
            ...base,
            ...authorizationPayload(bindings, bindings.receipt),
          });
          break;
        case 'rollback':
          response = await submitResearchLabCommand(action, session, {
            ...base,
            receipt_id: requiredId(bindings.receipt),
            receipt_sha256: requiredSha(bindings.receipt),
          });
          break;
      }
      setResult(response);
      onRefresh();
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : 'The local owner command did not complete.',
      );
    } finally {
      setSubmitting(false);
    }
  };

  const canSubmit =
    mutationsReady &&
    Boolean(
      action &&
      session?.supported_actions.includes(action) &&
      item &&
      commandBindingsReady(action, item, bindings) &&
      actor.trim() &&
      reason.trim(),
    ) &&
    !submitting &&
    !result;
  const exactBindings = action ? commandBindings(action, item, bindings) : [];
  const copy = action ? commandCopy[action] : null;
  const commandEnabled = (value: SupportedCommandAction) =>
    mutationsReady && (session?.supported_actions.includes(value) ?? false);

  return (
    <>
      <div className="mt-5 rounded-xl border border-white/[0.07] bg-white/[0.02] p-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <p className="flex items-center gap-2 text-xs text-slate-300">
              <ShieldCheck className="size-4 text-cyan-300" /> Control actions
            </p>
            <p className="mt-1 text-[11px] leading-relaxed text-slate-600">
              Fresh revision + exact artifact hash + named operator.
            </p>
          </div>
          <Badge
            variant="outline"
            className={
              mutationsReady
                ? 'border-emerald-300/20 bg-emerald-300/[0.06] font-mono text-[9px] text-emerald-200'
                : 'border-white/10 font-mono text-[9px] text-slate-500'
            }
          >
            {mutationsReady ? 'controlled' : 'read-only'}
          </Badge>
        </div>

        {sessionError && (
          <p className="mt-3 rounded-lg border border-amber-300/15 bg-amber-300/[0.04] px-3 py-2 text-[10px] leading-relaxed text-amber-100/60">
            {sessionError}
          </p>
        )}

        <div className="mt-4 space-y-2">
          {available.has('review_a0') && (
            <Button
              className="w-full bg-cyan-300 text-slate-950 hover:bg-cyan-200"
              onClick={() => begin('review_a0')}
              disabled={!commandEnabled('review_a0')}
            >
              <UserRoundCheck className="size-4" /> Review exact A0
            </Button>
          )}
          {available.has('reconcile') && (
            <Button
              className="w-full bg-cyan-300 text-slate-950 hover:bg-cyan-200"
              onClick={() => begin('reconcile')}
              disabled={!commandEnabled('reconcile')}
            >
              <Play className="size-4" /> Run one reconcile pass
            </Button>
          )}
          {available.has('import_candidate') && (
            <Button
              className="w-full bg-cyan-300 text-slate-950 hover:bg-cyan-200"
              onClick={() => begin('import_candidate')}
              disabled={!commandEnabled('import_candidate')}
            >
              <PackagePlus className="size-4" /> Import exact Candidate
            </Button>
          )}
          {available.has('view_run') && (
            <ActionNotice
              icon={Activity}
              title="Praxist takeover is running"
              detail="No start or reconcile command is exposed while this exact run is active."
              tone="live"
            />
          )}
          {available.has('authorize_shadow') && (
            <Button
              className="w-full bg-violet-300 text-slate-950 hover:bg-violet-200"
              onClick={() => begin('authorize_shadow')}
              disabled={!commandEnabled('authorize_shadow')}
            >
              <ShieldCheck className="size-4" /> Authorize exact SHADOW
            </Button>
          )}
          {available.has('authorize_active') && (
            <Button
              className="w-full bg-emerald-300 text-slate-950 hover:bg-emerald-200"
              onClick={() => begin('authorize_active')}
              disabled={!commandEnabled('authorize_active')}
            >
              <ShieldCheck className="size-4" /> Authorize exact ACTIVE
            </Button>
          )}
          {available.has('rollback') && (
            <Button
              variant="outline"
              className="w-full border-amber-300/25 bg-amber-300/[0.05] text-amber-100 hover:bg-amber-300/10 hover:text-amber-50"
              onClick={() => begin('rollback')}
              disabled={!commandEnabled('rollback')}
            >
              <RotateCcw className="size-4" /> Authorize adjacent rollback
            </Button>
          )}
          {available.has('run_formal_validation') && (
            <ActionNotice
              icon={CircleOff}
              title="Formal validator is external"
              detail="The task-owned sealed validator must publish exact evidence before the Portal can expose A1."
            />
          )}
          {available.has('view_formal_evidence') && (
            <ActionNotice
              icon={AlertTriangle}
              title="Gate review is still required"
              detail="Formal evidence is visible, but no exact ModificationGate artifact binds this validation round."
            />
          )}
          {available.has('inspect_handoff') && (
            <ActionNotice
              icon={AlertTriangle}
              title="Committed handoff is missing"
              detail="The completed process is visible, but Candidate import stays closed until the task-local exporter publishes the canonical handoff."
            />
          )}
          {available.has('inspect_blocker') && (
            <ActionNotice
              icon={AlertTriangle}
              title="Owner gate blocked this transition"
              detail={
                item?.lifecycle.blocking_reason ??
                'Inspect the exact negative artifact before producing fresh evidence.'
              }
            />
          )}
          {!item && (
            <ActionNotice
              icon={AlertTriangle}
              title="No exact task selected"
              detail="The control surface stays closed until a valid Task snapshot is present."
            />
          )}
          {item && item.available_actions.length === 0 && (
            <ActionNotice
              icon={CircleOff}
              title="No command available"
              detail="The current owner artifacts do not authorize a lifecycle transition."
            />
          )}
        </div>
      </div>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="border border-white/10 bg-slate-950 text-slate-100 shadow-2xl sm:max-w-lg">
          <DialogHeader>
            <DialogTitle>{copy?.title ?? 'Review exact command'}</DialogTitle>
            <DialogDescription className="leading-relaxed text-slate-500">
              {copy?.description}
            </DialogDescription>
          </DialogHeader>

          <div className="rounded-lg border border-white/[0.07] bg-white/[0.025] p-3">
            <p className="text-[10px] uppercase tracking-[0.16em] text-slate-600">
              Exact binding
            </p>
            <div className="mt-2 space-y-2">
              {exactBindings.length ? (
                exactBindings.map((binding) => (
                  <div
                    key={`${binding.kind}:${binding.sha256}`}
                    className="rounded-md border border-white/[0.05] bg-black/10 px-2.5 py-2"
                  >
                    <div className="flex items-center justify-between gap-3">
                      <span className="text-[9px] uppercase tracking-[0.12em] text-slate-600">
                        {binding.kind}
                      </span>
                      <span className="max-w-[250px] truncate font-mono text-[9px] text-slate-400">
                        {binding.artifact_id ?? 'content-addressed bytes'}
                      </span>
                    </div>
                    <p className="mt-1 break-all font-mono text-[8px] text-slate-700">
                      sha256:{binding.sha256}
                    </p>
                  </div>
                ))
              ) : (
                <p className="font-mono text-[10px] text-rose-300">
                  missing exact command bindings
                </p>
              )}
            </div>
            <p className="mt-2 font-mono text-[9px] text-slate-700">
              snapshot:{shortRevision(snapshot?.revision)}
            </p>
          </div>

          {!result && (
            <div className="space-y-4">
              {action === 'review_a0' && (
                <div className="grid grid-cols-2 gap-2">
                  <Button
                    type="button"
                    variant={decision === 'approve' ? 'default' : 'outline'}
                    className={
                      decision === 'approve'
                        ? 'bg-emerald-300 text-slate-950 hover:bg-emerald-200'
                        : 'border-white/10 bg-white/[0.02] text-slate-400'
                    }
                    onClick={() => setDecision('approve')}
                  >
                    <Check className="size-4" /> Approve scope
                  </Button>
                  <Button
                    type="button"
                    variant={decision === 'reject' ? 'destructive' : 'outline'}
                    className="border-white/10"
                    onClick={() => setDecision('reject')}
                  >
                    <CircleOff className="size-4" /> Reject scope
                  </Button>
                </div>
              )}

              <div className="space-y-2">
                <Label
                  htmlFor="command-actor"
                  className="text-xs text-slate-300"
                >
                  Named reviewer / operator
                </Label>
                <Input
                  id="command-actor"
                  value={actor}
                  onChange={(event) => setActor(event.target.value)}
                  placeholder="Your full name"
                  autoComplete="name"
                  maxLength={160}
                  className="border-white/10 bg-white/[0.03] text-slate-100"
                />
              </div>

              <div className="space-y-2">
                <Label
                  htmlFor="command-reason"
                  className="text-xs text-slate-300"
                >
                  Review reason
                </Label>
                <Textarea
                  id="command-reason"
                  value={reason}
                  onChange={(event) => setReason(event.target.value)}
                  placeholder="Why this exact scope should move to the next owner gate"
                  maxLength={2000}
                  className="min-h-24 border-white/10 bg-white/[0.03] text-slate-100"
                />
              </div>
            </div>
          )}

          {error && (
            <div className="rounded-lg border border-rose-300/20 bg-rose-300/[0.06] px-3 py-2 text-xs text-rose-100">
              {error}
            </div>
          )}
          {result && (
            <div
              className={`rounded-lg border px-3 py-3 text-xs ${
                result.outcome === 'blocked'
                  ? 'border-amber-300/20 bg-amber-300/[0.06] text-amber-100'
                  : 'border-emerald-300/20 bg-emerald-300/[0.06] text-emerald-100'
              }`}
            >
              <p className="flex items-center gap-2 font-medium">
                {result.outcome === 'blocked' ? (
                  <AlertTriangle className="size-4" />
                ) : (
                  <Check className="size-4" />
                )}{' '}
                {result.outcome}
              </p>
              <p className="mt-1 opacity-60">{result.message}</p>
            </div>
          )}

          <DialogFooter className="border-white/[0.07] bg-white/[0.02]">
            <Button
              variant="outline"
              className="border-white/10 bg-transparent text-slate-300"
              onClick={() => setOpen(false)}
            >
              {result ? 'Close' : 'Cancel'}
            </Button>
            {!result && (
              <Button
                onClick={submit}
                disabled={!canSubmit}
                className={
                  (action === 'review_a0' && decision === 'reject') ||
                  action === 'rollback'
                    ? 'bg-rose-300 text-slate-950 hover:bg-rose-200'
                    : 'bg-cyan-300 text-slate-950 hover:bg-cyan-200'
                }
              >
                {submitting
                  ? 'Submitting exact command…'
                  : action === 'review_a0'
                    ? `${decision === 'approve' ? 'Approve' : 'Reject'} exact A0`
                    : copy?.submit}
              </Button>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}

function ActionNotice({
  icon: Icon,
  title,
  detail,
  tone = 'blocked',
}: {
  icon: typeof Activity;
  title: string;
  detail: string;
  tone?: 'live' | 'blocked';
}) {
  return (
    <div
      className={`rounded-lg border px-3 py-3 ${
        tone === 'live'
          ? 'border-cyan-300/15 bg-cyan-300/[0.04]'
          : 'border-white/[0.06] bg-white/[0.02]'
      }`}
    >
      <p
        className={`flex items-center gap-2 text-[11px] ${
          tone === 'live' ? 'text-cyan-100' : 'text-slate-400'
        }`}
      >
        <Icon className="size-3.5" /> {title}
      </p>
      <p className="mt-1 text-[10px] leading-relaxed text-slate-600">
        {detail}
      </p>
    </div>
  );
}

function shortRevision(value: string | undefined): string {
  return value ? `${value.slice(0, 12)}…${value.slice(-8)}` : 'missing';
}

function findBinding(
  item: ResearchLabItem | null,
  kind: string,
): ArtifactRef | null {
  return item?.bindings.find((binding) => binding.kind === kind) ?? null;
}

function requiredId(binding: ArtifactRef | null): string {
  if (!binding?.artifact_id) throw new Error('Missing exact artifact identity');
  return binding.artifact_id;
}

function requiredSha(binding: ArtifactRef | null): string {
  if (!binding?.sha256) throw new Error('Missing exact artifact digest');
  return binding.sha256;
}

function authorizationPayload(
  bindings: CommandBindings,
  previous: ArtifactRef | null,
) {
  return {
    task_artifact_id: requiredId(bindings.task),
    task_sha256: requiredSha(bindings.task),
    candidate_artifact_id: requiredId(bindings.candidate),
    candidate_sha256: requiredSha(bindings.candidate),
    validation_sha256: requiredSha(bindings.validation),
    gate_sha256: requiredSha(bindings.gate),
    previous_receipt_id: previous ? requiredId(previous) : null,
    previous_receipt_sha256: previous ? requiredSha(previous) : null,
  };
}

function commandBindings(
  action: SupportedCommandAction,
  item: ResearchLabItem | null,
  bindings: CommandBindings,
): ArtifactRef[] {
  switch (action) {
    case 'review_a0':
    case 'reconcile':
      return bindings.request ? [bindings.request] : [];
    case 'import_candidate':
      return [bindings.task, bindings.handoff].filter(isArtifactRef);
    case 'authorize_shadow': {
      const previous =
        item?.authority.target_adapter_apply_required &&
        item.authority.authorized_wiring === 'disabled'
          ? bindings.receipt
          : null;
      return [
        bindings.task,
        bindings.candidate,
        bindings.validation,
        bindings.gate,
        previous,
      ].filter(isArtifactRef);
    }
    case 'authorize_active':
      return [
        bindings.task,
        bindings.candidate,
        bindings.validation,
        bindings.gate,
        bindings.receipt,
      ].filter(isArtifactRef);
    case 'rollback':
      return bindings.receipt ? [bindings.receipt] : [];
  }
}

function commandBindingsReady(
  action: SupportedCommandAction,
  item: ResearchLabItem,
  bindings: CommandBindings,
): boolean {
  const hasId = (binding: ArtifactRef | null) => Boolean(binding?.artifact_id);
  switch (action) {
    case 'review_a0':
    case 'reconcile':
      return hasId(bindings.request);
    case 'import_candidate':
      return (
        hasId(bindings.task) && Boolean(bindings.handoff && item.run?.run_id)
      );
    case 'authorize_shadow': {
      const needsPrevious =
        item.authority.target_adapter_apply_required &&
        item.authority.authorized_wiring === 'disabled';
      return (
        hasId(bindings.task) &&
        hasId(bindings.candidate) &&
        Boolean(bindings.validation && bindings.gate) &&
        (!needsPrevious || hasId(bindings.receipt))
      );
    }
    case 'authorize_active':
      return (
        hasId(bindings.task) &&
        hasId(bindings.candidate) &&
        Boolean(bindings.validation && bindings.gate) &&
        hasId(bindings.receipt)
      );
    case 'rollback':
      return hasId(bindings.receipt);
  }
}

function isArtifactRef(value: ArtifactRef | null): value is ArtifactRef {
  return value !== null;
}
