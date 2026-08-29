'use client';

import {
  Activity,
  AlertTriangle,
  Boxes,
  Check,
  ChevronRight,
  CircleDot,
  Cpu,
  Database,
  FlaskConical,
  GitBranch,
  LayoutDashboard,
  LockKeyhole,
  RefreshCw,
  Search,
  ShieldCheck,
  UserRoundCheck,
  WifiOff,
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
import { Progress } from '@/components/ui/progress';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { PromotionReadiness } from '@/components/promotion-readiness';
import { ResearchCommandWorkbench } from '@/components/research-command-workbench';
import { useResearchLab } from '@/hooks/use-research-lab';
import type {
  LifecycleStage,
  ResearchLabItem,
  ResearchLabSnapshot,
} from '@/lib/research-lab';

type StageState = 'complete' | 'current' | 'locked';

interface StageNodeData {
  label: string;
  detail: string;
  state: StageState;
}

const stageBlueprints = [
  { label: 'Forge', detail: 'Opportunity' },
  { label: 'A0', detail: 'Human review' },
  { label: 'Praxist', detail: 'Research run' },
  { label: 'Formal', detail: 'Sealed validation' },
  { label: 'Gate', detail: 'ModificationGate' },
  { label: 'Shadow', detail: 'A1 observation' },
  { label: 'Active', detail: 'A2 canary' },
] as const;

const stageLabels: Record<LifecycleStage, string> = {
  NEEDS_TASK_DESIGN: 'Needs task design',
  AWAITING_A0: 'Awaiting A0',
  PREFLIGHT: 'Preflight',
  RESEARCH_RUNNING: 'Research running',
  RESEARCH_COMPLETE: 'Research complete',
  CANDIDATE_RETAINED: 'Candidate retained',
  FORMAL_VALIDATION: 'Formal validation',
  AWAITING_A1: 'Awaiting A1',
  SHADOW: 'Shadow',
  AWAITING_A2: 'Awaiting A2',
  ACTIVE: 'Active',
  ROLLED_BACK: 'Rolled back',
  BLOCKED: 'Blocked',
};

function StageNode({ stage, index }: { stage: StageNodeData; index: number }) {
  const current = stage.state === 'current';
  const complete = stage.state === 'complete';

  return (
    <div className="relative min-w-[132px] flex-1">
      <div
        className={`relative z-10 rounded-xl border px-3 py-3 transition-colors ${
          current
            ? 'border-cyan-300/60 bg-cyan-300/10 shadow-[0_0_0_1px_rgba(103,232,249,.08),0_12px_32px_rgba(0,0,0,.16)]'
            : complete
              ? 'border-emerald-300/30 bg-emerald-300/[0.07]'
              : 'border-white/[0.07] bg-white/[0.025]'
        }`}
      >
        <div className="flex items-center justify-between">
          <span
            className={`flex size-6 items-center justify-center rounded-full border font-mono text-[10px] ${
              complete
                ? 'border-emerald-300/50 bg-emerald-300/15 text-emerald-200'
                : current
                  ? 'border-cyan-300/60 bg-cyan-300/15 text-cyan-100'
                  : 'border-white/10 bg-white/[0.03] text-slate-500'
            }`}
          >
            {complete ? <Check className="size-3" /> : index + 1}
          </span>
          {current && (
            <span className="flex items-center gap-1 font-mono text-[9px] uppercase tracking-[0.16em] text-cyan-200">
              <CircleDot className="size-3" /> current
            </span>
          )}
          {stage.state === 'locked' && (
            <LockKeyhole className="size-3 text-slate-600" />
          )}
        </div>
        <p className="mt-3 text-sm font-semibold text-slate-100">
          {stage.label}
        </p>
        <p className="mt-0.5 text-[11px] text-slate-500">{stage.detail}</p>
      </div>
      {index < stageBlueprints.length - 1 && (
        <ChevronRight className="absolute -right-3.5 top-1/2 z-20 size-3 -translate-y-1/2 text-slate-600" />
      )}
    </div>
  );
}

export default function Home() {
  const {
    snapshot,
    session,
    connection,
    error,
    sessionError,
    refreshing,
    refresh,
  } = useResearchLab();
  const primaryItem = snapshot?.items[0] ?? null;
  const stages = buildStages(primaryItem);
  const navItems = [
    { label: 'Pipeline', icon: LayoutDashboard, active: true },
    {
      label: 'Approvals',
      icon: UserRoundCheck,
      count: snapshot?.summary.awaiting_human ?? 0,
    },
    {
      label: 'Runs',
      icon: Activity,
      count: snapshot?.summary.active_runs ?? 0,
    },
    { label: 'Evidence', icon: Database },
    { label: 'System', icon: Cpu },
  ] as const;
  const metrics = buildMetrics(snapshot, primaryItem);
  const systemChecks = buildSystemChecks(snapshot, session?.mutations_enabled);
  const inspectorFacts = buildInspectorFacts(primaryItem);

  return (
    <main className="min-h-screen bg-background text-foreground">
      <header className="sticky top-0 z-40 border-b border-white/[0.07] bg-background/90 backdrop-blur-xl">
        <div className="flex h-16 items-center gap-4 px-4 sm:px-6">
          <div className="flex items-center gap-3 pr-3 sm:border-r sm:border-white/[0.08] sm:pr-6">
            <span className="flex size-9 items-center justify-center rounded-xl border border-cyan-300/20 bg-cyan-300/10 text-cyan-200 shadow-[inset_0_0_20px_rgba(34,211,238,.06)]">
              <FlaskConical className="size-4" />
            </span>
            <div>
              <p className="text-sm font-semibold tracking-tight">
                Research Lab
              </p>
              <p className="font-mono text-[9px] uppercase tracking-[0.18em] text-slate-500">
                Volvence control plane
              </p>
            </div>
          </div>

          <div className="hidden max-w-md flex-1 items-center gap-2 rounded-lg border border-white/[0.07] bg-white/[0.025] px-3 py-2 text-xs text-slate-500 md:flex">
            <Search className="size-3.5" />
            Search task, claim, request or run
            <kbd className="ml-auto rounded border border-white/10 px-1.5 py-0.5 font-mono text-[9px] text-slate-600">
              ⌘ K
            </kbd>
          </div>

          <div className="ml-auto flex items-center gap-2">
            <ConnectionBadge connection={connection} />
            <Button
              variant="ghost"
              size="icon"
              aria-label="Refresh Research Lab snapshot"
              onClick={refresh}
              disabled={refreshing}
            >
              <RefreshCw
                className={`size-4 text-slate-400 ${refreshing ? 'animate-spin' : ''}`}
              />
            </Button>
            <div className="flex size-8 items-center justify-center rounded-full border border-white/10 bg-slate-800 font-mono text-[10px] text-slate-300">
              MF
            </div>
          </div>
        </div>
      </header>

      <div className="grid min-h-[calc(100vh-64px)] grid-cols-1 xl:grid-cols-[210px_minmax(720px,1fr)_360px]">
        <aside className="hidden border-r border-white/[0.07] px-3 py-5 xl:flex xl:flex-col">
          <nav className="space-y-1" aria-label="Research Lab navigation">
            {navItems.map((item) => {
              const Icon = item.icon;
              return (
                <Button
                  key={item.label}
                  variant="ghost"
                  className={`h-9 w-full justify-start px-3 ${
                    'active' in item && item.active
                      ? 'bg-cyan-300/[0.08] text-cyan-100 hover:bg-cyan-300/[0.1] hover:text-cyan-100'
                      : 'text-slate-500 hover:bg-white/[0.04] hover:text-slate-200'
                  }`}
                >
                  <Icon className="size-4" />
                  {item.label}
                  {'count' in item && (
                    <span className="ml-auto font-mono text-[10px] text-slate-600">
                      {item.count}
                    </span>
                  )}
                </Button>
              );
            })}
          </nav>

          <div className="mt-auto rounded-xl border border-white/[0.07] bg-white/[0.025] p-3">
            <div className="flex items-center gap-2 text-xs text-slate-300">
              <ShieldCheck className="size-4 text-emerald-300" /> Authority
              guard
            </div>
            <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
              Evaluation is read-only. Every transition requires an exact owner
              artifact.
            </p>
          </div>
        </aside>

        <section className="min-w-0 px-4 py-5 sm:px-6 lg:px-8">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-cyan-300/70">
                Pipeline board
              </p>
              <h1 className="mt-1 text-2xl font-semibold tracking-[-0.03em] text-slate-50">
                Research to production, one gate at a time
              </h1>
              <p className="mt-1 max-w-2xl text-sm text-slate-500">
                Owner artifacts stay separate. This view joins their lineage
                without granting new authority.
              </p>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                className="border-white/10 bg-white/[0.025] text-slate-300"
              >
                <GitBranch className="size-4" /> View lineage
              </Button>
              <Button
                className="bg-cyan-300 text-slate-950 hover:bg-cyan-200"
                onClick={refresh}
              >
                Refresh snapshot <RefreshCw className="size-4" />
              </Button>
            </div>
          </div>

          {error && (
            <div className="mt-5 flex items-start gap-3 rounded-xl border border-rose-300/20 bg-rose-300/[0.06] px-4 py-3 text-xs text-rose-100">
              <WifiOff className="mt-0.5 size-4 shrink-0 text-rose-300" />
              <div>
                <p className="font-medium">Live snapshot refresh failed</p>
                <p className="mt-1 text-rose-100/55">
                  {error}. Last valid snapshot remains visible when available.
                </p>
              </div>
            </div>
          )}

          <div className="mt-6 overflow-x-auto pb-2">
            <div className="flex min-w-[980px] gap-3">
              {stages.map((stage, index) => (
                <StageNode key={stage.label} stage={stage} index={index} />
              ))}
            </div>
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {metrics.map((metric) => {
              const Icon = metric.icon;
              return (
                <Card
                  key={metric.label}
                  className="border-0 bg-card/80 ring-white/[0.07]"
                >
                  <CardHeader>
                    <CardTitle className="text-xs font-medium text-slate-500">
                      {metric.label}
                    </CardTitle>
                    <CardAction>
                      <Icon className="size-4 text-slate-600" />
                    </CardAction>
                  </CardHeader>
                  <CardContent>
                    <p className="font-mono text-2xl font-semibold text-slate-100">
                      {metric.value}
                    </p>
                    <p className="mt-1 text-[11px] text-slate-600">
                      {metric.detail}
                    </p>
                  </CardContent>
                </Card>
              );
            })}
          </div>

          <Card className="mt-5 border-0 bg-card/80 ring-white/[0.07]">
            <CardHeader className="border-b border-white/[0.06]">
              <CardTitle className="flex items-center gap-2 text-sm text-slate-200">
                <FlaskConical className="size-4 text-cyan-300" /> Research queue
              </CardTitle>
              <CardDescription>
                Exact tasks ordered by the next human or system gate.
              </CardDescription>
              <CardAction>
                <Badge
                  variant="outline"
                  className="border-white/10 font-mono text-[10px] text-slate-500"
                >
                  {snapshot?.items.length ?? 0} items
                </Badge>
              </CardAction>
            </CardHeader>
            <CardContent className="px-0">
              <Table>
                <TableHeader>
                  <TableRow className="border-white/[0.06] hover:bg-transparent">
                    <TableHead className="pl-4 text-[10px] uppercase tracking-wider text-slate-600">
                      Task
                    </TableHead>
                    <TableHead className="text-[10px] uppercase tracking-wider text-slate-600">
                      Owner
                    </TableHead>
                    <TableHead className="text-[10px] uppercase tracking-wider text-slate-600">
                      Stage
                    </TableHead>
                    <TableHead className="text-[10px] uppercase tracking-wider text-slate-600">
                      Evidence
                    </TableHead>
                    <TableHead className="text-[10px] uppercase tracking-wider text-slate-600">
                      Next gate
                    </TableHead>
                    <TableHead className="pr-4 text-right text-[10px] uppercase tracking-wider text-slate-600">
                      Updated
                    </TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {snapshot?.items.length ? (
                    snapshot.items.map((item) => (
                      <ResearchRow key={item.item_id} item={item} />
                    ))
                  ) : (
                    <TableRow className="border-white/[0.06] hover:bg-transparent">
                      <TableCell
                        colSpan={6}
                        className="h-28 text-center text-xs text-slate-600"
                      >
                        {connection === 'loading'
                          ? 'Loading immutable owner artifacts…'
                          : 'No valid Research Task is currently available.'}
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </CardContent>
          </Card>

          <div className="mt-5 grid gap-3 lg:grid-cols-[minmax(0,1.45fr)_minmax(280px,.55fr)]">
            <Card className="border-0 bg-card/80 ring-white/[0.07]">
              <CardContent>
                <PromotionReadiness item={primaryItem} />
              </CardContent>
            </Card>

            <Card className="border-0 bg-card/80 ring-white/[0.07]">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-sm text-slate-200">
                  <Cpu className="size-4 text-emerald-300" /> Local system
                </CardTitle>
                <CardDescription>
                  Readiness from the latest aggregate snapshot
                </CardDescription>
              </CardHeader>
              <CardContent className="grid grid-cols-2 gap-3">
                {systemChecks.map((check) => (
                  <div
                    key={check.label}
                    className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[11px] text-slate-500">
                        {check.label}
                      </span>
                      <span
                        className={`size-1.5 rounded-full ${healthDot(check.state)}`}
                      />
                    </div>
                    <p
                      className={`mt-2 font-mono text-[10px] uppercase ${healthText(check.state)}`}
                    >
                      {check.state}
                    </p>
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>
        </section>

        <aside className="border-l border-white/[0.07] bg-black/10 px-5 py-5 xl:block">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-mono text-[9px] uppercase tracking-[0.18em] text-slate-600">
                Inspector
              </p>
              <h2 className="mt-1 text-sm font-semibold text-slate-200">
                {primaryItem?.title ?? 'No selected research task'}
              </h2>
            </div>
            <Badge className={stageBadge(primaryItem?.lifecycle.stage)}>
              {primaryItem
                ? stageLabels[primaryItem.lifecycle.stage]
                : connection}
            </Badge>
          </div>

          <InspectorBanner item={primaryItem} connection={connection} />

          <dl className="mt-5 space-y-4">
            {inspectorFacts.map(([term, value]) => (
              <div
                key={term}
                className="flex items-start justify-between gap-4 border-b border-white/[0.05] pb-3"
              >
                <dt className="text-[11px] text-slate-600">{term}</dt>
                <dd className="max-w-[220px] break-all text-right font-mono text-[10px] text-slate-400">
                  {value}
                </dd>
              </div>
            ))}
          </dl>

          <div className="mt-5">
            <p className="font-mono text-[9px] uppercase tracking-[0.16em] text-slate-600">
              Live profile
            </p>
            <div className="mt-3 grid grid-cols-3 gap-2">
              {[
                ['Peers', String(primaryItem?.run?.peers_total ?? '—')],
                ['Generation', String(primaryItem?.run?.generation ?? '—')],
                ['Findings', String(primaryItem?.run?.findings_total ?? '—')],
              ].map(([label, value]) => (
                <div
                  key={label}
                  className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-2.5 text-center"
                >
                  <p className="font-mono text-sm text-slate-300">{value}</p>
                  <p className="mt-1 text-[9px] text-slate-600">{label}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="mt-5 rounded-xl border border-white/[0.07] bg-white/[0.02] p-4">
            <p className="flex items-center gap-2 text-xs text-slate-300">
              <ShieldCheck className="size-4 text-cyan-300" /> Authority
              boundary
            </p>
            <ul className="mt-3 space-y-2 text-[11px] text-slate-500">
              <AuthorityLine
                allowed={
                  primaryItem?.authority.a0_research_start_authorized ?? false
                }
                text="A0 exact research authorization"
              />
              <AuthorityLine
                allowed={
                  primaryItem?.authority.formal_validation_status === 'pass'
                }
                text={`Formal validation: ${primaryItem?.authority.formal_validation_status ?? 'unknown'}`}
              />
              <AuthorityLine
                allowed={
                  primaryItem?.authority.modification_gate_decision === 'allow'
                }
                text={`ModificationGate: ${primaryItem?.authority.modification_gate_decision ?? 'unknown'}`}
              />
              <AuthorityLine
                allowed={['shadow', 'active'].includes(
                  primaryItem?.authority.authorized_wiring ?? '',
                )}
                text={`A1 authorization: ${primaryItem?.authority.authorized_wiring ?? 'unknown'}`}
              />
              <AuthorityLine
                allowed={primaryItem?.authority.runtime_wiring === 'shadow'}
                text={`SHADOW applied: ${primaryItem?.authority.runtime_wiring ?? 'unknown'}`}
              />
              <AuthorityLine
                allowed={primaryItem?.authority.authorized_wiring === 'active'}
                text={`A2 authorization: ${primaryItem?.authority.authorized_wiring ?? 'unknown'}`}
              />
              <AuthorityLine
                allowed={primaryItem?.authority.runtime_wiring === 'active'}
                text={`ACTIVE applied: ${primaryItem?.authority.runtime_wiring ?? 'unknown'}`}
              />
            </ul>
          </div>

          <ResearchCommandWorkbench
            item={primaryItem}
            snapshot={snapshot}
            session={session}
            sessionError={sessionError}
            onRefresh={refresh}
          />

          <Button
            className="mt-5 w-full bg-cyan-300 text-slate-950 hover:bg-cyan-200"
            onClick={refresh}
            disabled={refreshing}
          >
            Refresh exact state <RefreshCw className="size-4" />
          </Button>
          <p className="mt-2 text-center text-[10px] text-slate-700">
            Commands delegate to owner gates · no direct runtime wiring
          </p>
        </aside>
      </div>
    </main>
  );
}

function ResearchRow({ item }: { item: ResearchLabItem }) {
  const running = item.lifecycle.stage === 'RESEARCH_RUNNING';
  const blocked = item.lifecycle.stage === 'BLOCKED';
  return (
    <TableRow className="border-white/[0.06] bg-cyan-300/[0.025] hover:bg-cyan-300/[0.045]">
      <TableCell className="pl-4">
        <div className="flex items-center gap-3">
          <span className="flex size-8 items-center justify-center rounded-lg border border-cyan-300/20 bg-cyan-300/[0.07] text-cyan-200">
            <Boxes className="size-4" />
          </span>
          <div>
            <p className="max-w-[260px] truncate font-medium text-slate-200">
              {item.title}
            </p>
            <p className="font-mono text-[10px] text-slate-600">
              {item.claim_id}
            </p>
          </div>
        </div>
      </TableCell>
      <TableCell>
        <Badge
          variant="outline"
          className="border-white/10 font-mono text-[10px] text-slate-400"
        >
          {item.owner}
        </Badge>
      </TableCell>
      <TableCell>
        <span
          className={`inline-flex items-center gap-1.5 text-xs ${
            blocked
              ? 'text-rose-200'
              : running
                ? 'text-cyan-200'
                : 'text-slate-300'
          }`}
        >
          <span
            className={`size-1.5 rounded-full ${
              blocked
                ? 'bg-rose-300'
                : running
                  ? 'animate-pulse bg-cyan-300'
                  : 'bg-slate-500'
            }`}
          />
          {stageLabels[item.lifecycle.stage]}
        </span>
      </TableCell>
      <TableCell>
        <div className="space-y-1.5">
          <div className="flex items-center justify-between gap-4 font-mono text-[10px] text-slate-500">
            <span>{item.evidence.development}</span>
            <span>
              {item.run
                ? `${item.run.peers_total} peers`
                : item.evidence.formal}
            </span>
          </div>
          <Progress
            value={pipelineProgress(item)}
            className="w-28 [&_[data-slot=progress-indicator]]:bg-cyan-300"
          />
        </div>
      </TableCell>
      <TableCell>
        <span className="inline-flex items-center gap-1.5 text-xs text-slate-300">
          <FlaskConical className="size-3.5 text-cyan-300" />
          {item.lifecycle.next_stage
            ? stageLabels[item.lifecycle.next_stage]
            : 'Inspect outcome'}
        </span>
      </TableCell>
      <TableCell className="pr-4 text-right font-mono text-[10px] text-slate-600">
        {formatTime(item.updated_at)}
      </TableCell>
    </TableRow>
  );
}

function InspectorBanner({
  item,
  connection,
}: {
  item: ResearchLabItem | null;
  connection: 'loading' | 'live' | 'degraded' | 'offline';
}) {
  if (connection === 'offline') {
    return (
      <div className="mt-5 rounded-xl border border-rose-300/20 bg-rose-300/[0.05] p-4">
        <div className="flex items-start gap-3">
          <WifiOff className="mt-0.5 size-4 text-rose-300" />
          <div>
            <p className="text-xs font-medium text-rose-100">
              Local API unavailable
            </p>
            <p className="mt-1 text-[11px] leading-relaxed text-rose-100/55">
              Start the loopback collector; no cached state is being presented
              as live.
            </p>
          </div>
        </div>
      </div>
    );
  }
  if (item?.run?.state === 'running') {
    return (
      <div className="mt-5 rounded-xl border border-cyan-300/20 bg-cyan-300/[0.05] p-4">
        <div className="flex items-start gap-3">
          <Activity className="mt-0.5 size-4 text-cyan-300" />
          <div>
            <p className="text-xs font-medium text-cyan-100">
              One live Praxist controller
            </p>
            <p className="mt-1 text-[11px] leading-relaxed text-cyan-100/55">
              PID {item.run.pid ?? 'unknown'} · generation{' '}
              {item.run.generation ?? 'unknown'} · {item.run.peers_total} peers.
              Production wiring remains {item.authority.runtime_wiring}.
            </p>
          </div>
        </div>
      </div>
    );
  }
  return (
    <div className="mt-5 rounded-xl border border-amber-300/20 bg-amber-300/[0.05] p-4">
      <div className="flex items-start gap-3">
        <AlertTriangle className="mt-0.5 size-4 text-amber-300" />
        <div>
          <p className="text-xs font-medium text-amber-100">
            {item
              ? stageLabels[item.lifecycle.stage]
              : 'Waiting for valid owner artifacts'}
          </p>
          <p className="mt-1 text-[11px] leading-relaxed text-amber-100/55">
            {item?.lifecycle.blocking_reason ??
              'The collector will publish the next exact state when available.'}
          </p>
        </div>
      </div>
    </div>
  );
}

function ConnectionBadge({
  connection,
}: {
  connection: 'loading' | 'live' | 'degraded' | 'offline';
}) {
  const color =
    connection === 'live'
      ? 'border-emerald-300/20 bg-emerald-300/[0.06] text-emerald-200'
      : connection === 'offline'
        ? 'border-rose-300/20 bg-rose-300/[0.06] text-rose-200'
        : 'border-amber-300/20 bg-amber-300/[0.06] text-amber-200';
  const dot =
    connection === 'live'
      ? 'bg-emerald-300'
      : connection === 'offline'
        ? 'bg-rose-300'
        : 'bg-amber-300';
  return (
    <Badge
      variant="outline"
      className={`hidden font-mono text-[10px] lg:flex ${color}`}
    >
      <span className={`size-1.5 rounded-full ${dot}`} /> {connection}
    </Badge>
  );
}

function AuthorityLine({ allowed, text }: { allowed: boolean; text: string }) {
  return (
    <li className="flex gap-2">
      {allowed ? (
        <Check className="mt-0.5 size-3 shrink-0 text-emerald-300" />
      ) : (
        <LockKeyhole className="mt-0.5 size-3 shrink-0 text-slate-600" />
      )}
      {text}
    </li>
  );
}

function buildStages(item: ResearchLabItem | null): StageNodeData[] {
  const currentIndex = pipelineIndex(item);
  return stageBlueprints.map((stage, index) => ({
    ...stage,
    state:
      index < currentIndex
        ? 'complete'
        : index === currentIndex
          ? 'current'
          : 'locked',
  }));
}

function pipelineIndex(item: ResearchLabItem | null): number {
  if (!item) return 0;
  if (
    item.authority.runtime_wiring === 'active' ||
    item.lifecycle.stage === 'AWAITING_A2'
  )
    return 6;
  if (
    item.authority.runtime_wiring === 'shadow' ||
    item.lifecycle.stage === 'AWAITING_A1'
  )
    return 5;
  if (item.authority.modification_gate_decision === 'allow') return 4;
  if (
    item.lifecycle.stage === 'CANDIDATE_RETAINED' ||
    item.lifecycle.stage === 'FORMAL_VALIDATION' ||
    item.authority.formal_validation_status === 'pass'
  )
    return 3;
  if (
    item.lifecycle.stage === 'PREFLIGHT' ||
    item.lifecycle.stage === 'RESEARCH_RUNNING' ||
    item.lifecycle.stage === 'RESEARCH_COMPLETE'
  )
    return 2;
  if (item.lifecycle.stage === 'AWAITING_A0') return 1;
  return 0;
}

function pipelineProgress(item: ResearchLabItem): number {
  const values = [8, 18, 36, 52, 66, 84, 100];
  return values[pipelineIndex(item)];
}

function buildMetrics(
  snapshot: ResearchLabSnapshot | null,
  item: ResearchLabItem | null,
) {
  return [
    {
      label: 'Registered tasks',
      value: String(snapshot?.summary.registered_tasks ?? 0),
      detail: snapshot
        ? `${snapshot.items.length} immutable views`
        : 'collector unavailable',
      icon: Boxes,
    },
    {
      label: 'Awaiting human',
      value: String(snapshot?.summary.awaiting_human ?? 0),
      detail:
        item?.lifecycle.stage === 'AWAITING_A0'
          ? 'A0 research start'
          : 'No pending exact review',
      icon: UserRoundCheck,
    },
    {
      label: 'Active runs',
      value: String(snapshot?.summary.active_runs ?? 0),
      detail: item?.run
        ? `PID ${item.run.pid ?? '—'} · gen ${item.run.generation ?? '—'}`
        : 'Praxist registry',
      icon: Activity,
    },
    {
      label: 'Production active',
      value: String(snapshot?.summary.production_active ?? 0),
      detail: item
        ? `runtime ${item.authority.runtime_wiring}`
        : 'No authority observed',
      icon: ShieldCheck,
    },
  ];
}

function buildSystemChecks(
  snapshot: ResearchLabSnapshot | null,
  mutationsEnabled: boolean | undefined,
) {
  const health = (source: string) =>
    snapshot?.source_health.find((value) => value.source === source);
  return [
    {
      label: 'Forge control',
      state: health('control')?.status ?? 'unavailable',
    },
    {
      label: 'Praxist status',
      state: health('praxist')?.status ?? 'unavailable',
    },
    {
      label: 'Promotion store',
      state: health('promotion')?.artifacts_seen
        ? (health('promotion')?.status ?? 'unavailable')
        : 'empty',
    },
    {
      label: 'Portal commands',
      state: mutationsEnabled ? 'controlled' : 'read-only',
    },
    { label: 'Target adapter', state: 'not wired' },
  ];
}

function buildInspectorFacts(
  item: ResearchLabItem | null,
): Array<[string, string]> {
  const binding = (kind: string) =>
    item?.bindings.find((value) => value.kind === kind);
  const request = binding('research request');
  const approval = binding('research approval');
  const task = binding('task');
  const candidate = binding('candidate');
  const validation = binding('validation');
  const gate = binding('gate');
  const receipt = binding('receipt');
  return [
    ['Request', shorten(request?.artifact_id)],
    ['Request SHA', shorten(request?.sha256)],
    ['Approval', shorten(approval?.artifact_id)],
    ['Task SHA', shorten(task?.sha256)],
    ['Candidate', shorten(candidate?.artifact_id)],
    ['Validation SHA', shorten(validation?.sha256)],
    ['Gate SHA', shorten(gate?.sha256)],
    ['Receipt', shorten(receipt?.artifact_id)],
    ['Run id', shorten(item?.run?.run_id, 28)],
    [
      'PID / state',
      item?.run ? `${item.run.pid ?? '—'} · ${item.run.state}` : '—',
    ],
    ['Runtime', stripNamespace(item?.run?.runtime)],
    ['Model', item?.run?.model ?? '—'],
    ['Authorized', item?.authority.authorized_wiring ?? '—'],
    ['Runtime wiring', item?.authority.runtime_wiring ?? '—'],
  ];
}

function shorten(value: string | null | undefined, width = 18): string {
  if (!value) return '—';
  if (value.length <= width) return value;
  const side = Math.max(4, Math.floor((width - 1) / 2));
  return `${value.slice(0, side)}…${value.slice(-side)}`;
}

function stripNamespace(value: string | null | undefined): string {
  if (!value) return '—';
  const separator = value.indexOf(':');
  return separator >= 0 ? value.slice(separator + 1) : value;
}

function formatTime(value: string | null): string {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return 'invalid time';
  return `${date.toISOString().slice(11, 16)} UTC`;
}

function stageBadge(stage: LifecycleStage | undefined): string {
  if (stage === 'RESEARCH_RUNNING') return 'bg-cyan-300/10 text-cyan-200';
  if (stage === 'BLOCKED') return 'bg-rose-300/10 text-rose-200';
  if (stage === 'ACTIVE' || stage === 'SHADOW')
    return 'bg-emerald-300/10 text-emerald-200';
  return 'bg-amber-300/10 text-amber-200';
}

function healthDot(state: string): string {
  if (state === 'healthy' || state === 'controlled') return 'bg-emerald-300';
  if (state === 'degraded') return 'bg-amber-300';
  return 'bg-slate-600';
}

function healthText(state: string): string {
  if (state === 'healthy' || state === 'controlled') return 'text-emerald-300';
  if (state === 'degraded') return 'text-amber-300';
  return 'text-slate-600';
}
