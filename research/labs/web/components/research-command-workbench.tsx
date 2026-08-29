'use client';

import { useMemo, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  Check,
  CircleOff,
  Play,
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

type ReviewDecision = 'approve' | 'reject';

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

  const request = useMemo(
    () => item?.bindings.find((binding) => binding.kind === 'research request'),
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
    if (!request?.artifact_id) {
      setError('The current snapshot has no exact ResearchRequest identity.');
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const response = await submitResearchLabCommand(action, session, {
        snapshot_revision: snapshot.revision,
        task_id: item.task_id,
        artifact_id: request.artifact_id,
        artifact_sha256: request.sha256,
        actor: actor.trim(),
        reason: reason.trim(),
        ...(action === 'review_a0' ? { decision } : {}),
      });
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
    Boolean(action && request?.artifact_id && actor.trim() && reason.trim()) &&
    !submitting &&
    !result;

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
              disabled={!mutationsReady}
            >
              <UserRoundCheck className="size-4" /> Review exact A0
            </Button>
          )}
          {available.has('reconcile') && (
            <Button
              className="w-full bg-cyan-300 text-slate-950 hover:bg-cyan-200"
              onClick={() => begin('reconcile')}
              disabled={!mutationsReady}
            >
              <Play className="size-4" /> Run one reconcile pass
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
            <ActionNotice
              icon={CircleOff}
              title="A1 authorization is not connected"
              detail="The portal needs exact formal validation and ModificationGate bindings before this owner seam can be enabled."
            />
          )}
          {available.has('authorize_active') && (
            <ActionNotice
              icon={CircleOff}
              title="A2 authorization is not connected"
              detail="SHADOW observation, canary evidence, and target-owned apply receipts remain mandatory."
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
            <DialogTitle>
              {action === 'review_a0'
                ? 'Review exact A0 scope'
                : 'Run one bounded reconciliation'}
            </DialogTitle>
            <DialogDescription className="leading-relaxed text-slate-500">
              {action === 'review_a0'
                ? 'This writes one immutable APPROVE or REJECT decision for the frozen ResearchRequest. It does not authorize formal validation or production wiring.'
                : 'Forge will recheck approval, binding bytes, host capacity, doctor, resolve, and the global reconcile lock. A start is possible only when every owner gate passes.'}
            </DialogDescription>
          </DialogHeader>

          <div className="rounded-lg border border-white/[0.07] bg-white/[0.025] p-3">
            <p className="text-[10px] uppercase tracking-[0.16em] text-slate-600">
              Exact binding
            </p>
            <p className="mt-2 break-all font-mono text-[10px] text-slate-400">
              {request?.artifact_id ?? 'missing request id'}
            </p>
            <p className="mt-1 break-all font-mono text-[9px] text-slate-700">
              sha256:{request?.sha256 ?? 'missing'}
            </p>
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
            <div className="rounded-lg border border-emerald-300/20 bg-emerald-300/[0.06] px-3 py-3 text-xs text-emerald-100">
              <p className="flex items-center gap-2 font-medium">
                <Check className="size-4" /> {result.outcome}
              </p>
              <p className="mt-1 text-emerald-100/60">{result.message}</p>
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
                  action === 'review_a0' && decision === 'reject'
                    ? 'bg-rose-300 text-slate-950 hover:bg-rose-200'
                    : 'bg-cyan-300 text-slate-950 hover:bg-cyan-200'
                }
              >
                {submitting
                  ? 'Submitting exact command…'
                  : action === 'review_a0'
                    ? `${decision === 'approve' ? 'Approve' : 'Reject'} exact A0`
                    : 'Run one reconcile pass'}
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
