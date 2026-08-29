'use client';

import { useMemo, useState } from 'react';
import {
  Bot,
  Check,
  CircleOff,
  FileSearch,
  GitBranch,
  LockKeyhole,
  ShieldCheck,
} from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
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
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Textarea } from '@/components/ui/textarea';
import {
  submitResearchLabCommand,
  type ResearchDemand,
  type ResearchLabCommandResult,
  type ResearchLabSession,
  type ResearchLabSnapshot,
  type ResearchTopicProposal,
} from '@/lib/research-lab';

interface SelectedTopic {
  demand: ResearchDemand;
  proposal: ResearchTopicProposal;
}

export function ResearchDiscoveryInbox({
  snapshot,
  session,
  onRefresh,
}: {
  snapshot: ResearchLabSnapshot | null;
  session: ResearchLabSession | null;
  onRefresh: () => void;
}) {
  const topics = useMemo(
    () =>
      (snapshot?.discovery.demands ?? []).flatMap((demand) =>
        demand.proposals.map((proposal) => ({ demand, proposal })),
      ),
    [snapshot],
  );
  const [selected, setSelected] = useState<SelectedTopic | null>(null);
  const [actor, setActor] = useState('');
  const [reason, setReason] = useState('');
  const [mappingId, setMappingId] = useState('');
  const [decision, setDecision] = useState<'approve' | 'reject'>('approve');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ResearchLabCommandResult | null>(null);

  function begin(topic: SelectedTopic) {
    setSelected(topic);
    setMappingId(topic.demand.requested_mapping_id ?? '');
    setDecision('approve');
    setReason('');
    setError(null);
    setResult(null);
  }

  function close() {
    setSelected(null);
    setError(null);
    setResult(null);
  }

  async function submit() {
    if (!snapshot || !session || !selected || !snapshot.discovery.registry) {
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const commandResult = await submitResearchLabCommand(
        'bind_topic',
        session,
        {
          snapshot_revision: snapshot.revision,
          demand_id: selected.demand.demand_id,
          demand_sha256: selected.demand.artifact.sha256,
          proposal_id: selected.proposal.proposal_id,
          proposal_sha256: selected.proposal.artifact.sha256,
          registry_sha256: snapshot.discovery.registry.sha256,
          mapping_id: mappingId.trim(),
          actor: actor.trim(),
          reason: reason.trim(),
          decision,
        },
      );
      setResult(commandResult);
      onRefresh();
    } catch (value) {
      setError(value instanceof Error ? value.message : 'Binding failed');
    } finally {
      setSubmitting(false);
    }
  }

  const commandEnabled = Boolean(
    session?.mutations_enabled &&
      session.supported_actions.includes('bind_topic') &&
      snapshot?.discovery.registry,
  );
  const canSubmit = Boolean(
    commandEnabled &&
      selected &&
      actor.trim() &&
      reason.trim() &&
      mappingId.trim() &&
      !submitting,
  );

  return (
    <>
      <Card className="mt-5 border-0 bg-card/80 ring-white/[0.07]">
        <CardHeader className="border-b border-white/[0.06]">
          <CardTitle className="flex items-center gap-2 text-sm text-slate-200">
            <FileSearch className="size-4 text-cyan-300" /> Demand discovery
            inbox
          </CardTitle>
          <CardDescription>
            Codex proposes against frozen Volvence needs. A named human must
            bind one exact topic before the separate A0 gate exists.
          </CardDescription>
          <CardAction className="flex gap-2">
            <Badge
              variant="outline"
              className="border-amber-300/20 bg-amber-300/[0.05] font-mono text-[10px] text-amber-200"
            >
              {snapshot?.discovery.awaiting_binding_count ?? 0} unbound
            </Badge>
            <Badge
              variant="outline"
              className="border-white/10 font-mono text-[10px] text-slate-500"
            >
              {topics.length} proposals
            </Badge>
          </CardAction>
        </CardHeader>
        <CardContent className="px-0">
          <Table>
            <TableHeader>
              <TableRow className="border-white/[0.06] hover:bg-transparent">
                {['Demand / topic', 'Owner', 'Codex run', 'State', 'Action'].map(
                  (label, index) => (
                    <TableHead
                      key={label}
                      className={`${index === 0 ? 'pl-4' : ''} ${index === 4 ? 'pr-4 text-right' : ''} text-[10px] uppercase tracking-wider text-slate-600`}
                    >
                      {label}
                    </TableHead>
                  ),
                )}
              </TableRow>
            </TableHeader>
            <TableBody>
              {topics.length ? (
                topics.map(({ demand, proposal }) => (
                  <TableRow
                    key={proposal.proposal_id}
                    className="border-white/[0.06]"
                  >
                    <TableCell className="max-w-xl pl-4">
                      <p className="text-[10px] uppercase tracking-[0.14em] text-cyan-300/60">
                        {demand.title}
                      </p>
                      <p className="mt-1 font-medium text-slate-200">
                        {proposal.title}
                      </p>
                      <p className="mt-1 line-clamp-2 text-[11px] leading-relaxed text-slate-500">
                        {proposal.demand_relevance}
                      </p>
                    </TableCell>
                    <TableCell>
                      <p className="font-mono text-[10px] text-slate-400">
                        {demand.owner}
                      </p>
                      <p className="mt-1 text-[10px] text-slate-600">
                        {demand.capability_axes.join(' · ')}
                      </p>
                    </TableCell>
                    <TableCell>
                      <p className="flex items-center gap-1.5 font-mono text-[10px] text-slate-400">
                        <Bot className="size-3 text-cyan-300" />
                        {demand.run_model ?? 'pending'}
                      </p>
                      <p className="mt-1 text-[10px] text-slate-600">
                        {proposal.source_refs.length} exact sources
                      </p>
                    </TableCell>
                    <TableCell>
                      <Badge
                        variant="outline"
                        className={topicStateClass(proposal.effective_state)}
                      >
                        {topicStateLabel(proposal.effective_state)}
                      </Badge>
                      {proposal.mapping_id && (
                        <p className="mt-1 font-mono text-[9px] text-slate-600">
                          {proposal.mapping_id}
                        </p>
                      )}
                    </TableCell>
                    <TableCell className="pr-4 text-right">
                      {proposal.available_actions.includes('bind_topic') ? (
                        <Button
                          size="sm"
                          variant="outline"
                          className="border-cyan-300/20 bg-cyan-300/[0.05] text-cyan-100"
                          onClick={() => begin({ demand, proposal })}
                          disabled={!commandEnabled}
                        >
                          <GitBranch className="size-3.5" /> Bind topic
                        </Button>
                      ) : (
                        <span className="font-mono text-[9px] text-slate-700">
                          immutable
                        </span>
                      )}
                    </TableCell>
                  </TableRow>
                ))
              ) : (
                <TableRow className="border-white/[0.06] hover:bg-transparent">
                  <TableCell
                    colSpan={5}
                    className="h-28 text-center text-xs text-slate-600"
                  >
                    No completed TopicProposal is visible. The automatic worker
                    will inspect the next new or changed OPEN Demand.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Dialog open={selected !== null} onOpenChange={(open) => !open && close()}>
        <DialogContent className="max-h-[90vh] overflow-y-auto border-white/[0.08] bg-slate-950 sm:max-w-2xl">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-slate-100">
              <GitBranch className="size-4 text-cyan-300" /> Exact topic binding
            </DialogTitle>
            <DialogDescription>
              This decision only permits Forge to submit a ResearchRequest. It
              does not approve A0, start Praxist, import a Candidate, or change
              runtime wiring.
            </DialogDescription>
          </DialogHeader>

          {selected && (
            <div className="space-y-4">
              <div className="rounded-xl border border-white/[0.07] bg-white/[0.025] p-4">
                <p className="text-[10px] uppercase tracking-[0.14em] text-cyan-300/60">
                  {selected.demand.title}
                </p>
                <p className="mt-1 text-sm font-medium text-slate-100">
                  {selected.proposal.title}
                </p>
                <p className="mt-2 text-xs leading-relaxed text-slate-500">
                  {selected.proposal.hypothesis}
                </p>
                <div className="mt-3 grid gap-2 sm:grid-cols-3">
                  {[
                    ['Demand', selected.demand.artifact.sha256],
                    ['Proposal', selected.proposal.artifact.sha256],
                    ['Registry', snapshot?.discovery.registry?.sha256 ?? 'missing'],
                  ].map(([label, value]) => (
                    <div
                      key={label}
                      className="rounded-lg border border-white/[0.05] bg-black/10 p-2.5"
                    >
                      <p className="text-[9px] uppercase tracking-wider text-slate-600">
                        {label}
                      </p>
                      <p className="mt-1 truncate font-mono text-[9px] text-slate-400">
                        {value}
                      </p>
                    </div>
                  ))}
                </div>
              </div>

              {!result && (
                <>
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
                      <Check className="size-4" /> Bind for A0 submission
                    </Button>
                    <Button
                      type="button"
                      variant={decision === 'reject' ? 'destructive' : 'outline'}
                      className="border-white/10"
                      onClick={() => setDecision('reject')}
                    >
                      <CircleOff className="size-4" /> Reject topic
                    </Button>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="topic-mapping" className="text-xs text-slate-300">
                      Exact registry mapping
                    </Label>
                    <Input
                      id="topic-mapping"
                      value={mappingId}
                      onChange={(event) => setMappingId(event.target.value)}
                      placeholder="registered_mapping_id"
                      className="border-white/10 bg-white/[0.03] font-mono text-slate-100"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="topic-actor" className="text-xs text-slate-300">
                      Named human reviewer
                    </Label>
                    <Input
                      id="topic-actor"
                      value={actor}
                      onChange={(event) => setActor(event.target.value)}
                      placeholder="Your full name"
                      autoComplete="name"
                      maxLength={160}
                      className="border-white/10 bg-white/[0.03] text-slate-100"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="topic-reason" className="text-xs text-slate-300">
                      Binding reason
                    </Label>
                    <Textarea
                      id="topic-reason"
                      value={reason}
                      onChange={(event) => setReason(event.target.value)}
                      placeholder="Why this exact hypothesis belongs to this registered Volvence task"
                      maxLength={2000}
                      className="min-h-24 border-white/10 bg-white/[0.03] text-slate-100"
                    />
                  </div>
                </>
              )}

              {error && (
                <div className="rounded-lg border border-rose-300/20 bg-rose-300/[0.06] px-3 py-2 text-xs text-rose-100">
                  {error}
                </div>
              )}
              {result && (
                <div className="rounded-lg border border-emerald-300/20 bg-emerald-300/[0.06] px-3 py-3 text-xs text-emerald-100">
                  <p className="flex items-center gap-2 font-medium">
                    <ShieldCheck className="size-4" /> {result.outcome}
                  </p>
                  <p className="mt-1 text-emerald-100/60">{result.message}</p>
                </div>
              )}
            </div>
          )}

          <DialogFooter className="border-white/[0.07] bg-white/[0.02]">
            <Button
              variant="outline"
              className="border-white/10 bg-transparent text-slate-300"
              onClick={close}
            >
              {result ? 'Close' : 'Cancel'}
            </Button>
            {!result && (
              <Button
                onClick={submit}
                disabled={!canSubmit}
                className={
                  decision === 'reject'
                    ? 'bg-rose-300 text-slate-950 hover:bg-rose-200'
                    : 'bg-cyan-300 text-slate-950 hover:bg-cyan-200'
                }
              >
                {submitting
                  ? 'Sealing exact decision…'
                  : decision === 'approve'
                    ? 'Bind exact topic'
                    : 'Reject exact topic'}
                <LockKeyhole className="size-4" />
              </Button>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}

function topicStateLabel(state: string): string {
  const labels: Record<string, string> = {
    UNBOUND: 'Needs binding',
    BOUND_FOR_A0: 'Queued for A0',
    AWAITING_A0: 'Awaiting A0',
    PREFLIGHT: 'Preflight',
    RESEARCH_RUNNING: 'Research running',
    RESEARCH_COMPLETE: 'Research complete',
    REJECTED: 'Rejected',
    STALE_SOURCE: 'Source changed',
    BINDING_AMBIGUOUS: 'Binding conflict',
    REQUEST_AMBIGUOUS: 'Request conflict',
  };
  return labels[state] ?? state.replaceAll('_', ' ').toLowerCase();
}

function topicStateClass(state: string): string {
  if (state === 'UNBOUND')
    return 'border-amber-300/20 bg-amber-300/[0.06] text-amber-200';
  if (state === 'RESEARCH_RUNNING')
    return 'border-cyan-300/20 bg-cyan-300/[0.06] text-cyan-200';
  if (state === 'RESEARCH_COMPLETE')
    return 'border-emerald-300/20 bg-emerald-300/[0.06] text-emerald-200';
  if (
    state === 'REJECTED' ||
    state === 'STALE_SOURCE' ||
    state.endsWith('AMBIGUOUS')
  )
    return 'border-rose-300/20 bg-rose-300/[0.06] text-rose-200';
  return 'border-white/10 bg-white/[0.025] text-slate-400';
}
