'use client';

import Link from 'next/link';
import { useParams } from 'next/navigation';
import { useState } from 'react';
import {
  Activity,
  AlertTriangle,
  Boxes,
  Check,
  ChevronRight,
  CircleDot,
  Cpu,
  Database,
  FileSearch,
  FlaskConical,
  GitBranch,
  LayoutDashboard,
  LockKeyhole,
  Minus,
  RefreshCw,
  Search,
  ShieldCheck,
  UserRoundCheck,
  WifiOff,
} from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button, buttonVariants } from '@/components/ui/button';
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Input } from '@/components/ui/input';
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
import { ResearchDiscoveryInbox } from '@/components/research-discovery-inbox';
import { useResearchLab } from '@/hooks/use-research-lab';
import type {
  LifecycleStage,
  ResearchLabItem,
  ResearchLabSnapshot,
} from '@/lib/research-lab';

type StageState = 'complete' | 'current' | 'locked' | 'not_applicable';
export type ResearchLabView =
  | 'pipeline'
  | 'discovery'
  | 'task'
  | 'approvals'
  | 'runs'
  | 'evidence'
  | 'system';

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

const viewCopy: Record<
  ResearchLabView,
  { eyebrow: string; title: string; description: string }
> = {
  pipeline: {
    eyebrow: 'Pipeline board',
    title: 'Research to production, one gate at a time',
    description:
      'Owner artifacts stay separate. This view joins their lineage without granting new authority.',
  },
  discovery: {
    eyebrow: 'Demand discovery',
    title: 'Volvence needs become auditable research candidates',
    description:
      'Codex reads only the frozen Demand corpus. Every proposal stays unbound until a named human maps it to an exact registered task.',
  },
  task: {
    eyebrow: 'Task lineage',
    title: 'One exact research task across every gate',
    description:
      'Inspect immutable bindings, run state, evidence tiers, authorization and the next owner action.',
  },
  approvals: {
    eyebrow: 'Review inbox',
    title: 'Human decisions that are actually waiting',
    description:
      'A0, A1 and A2 remain separate exact-bound decisions. This view never infers approval from progress.',
  },
  runs: {
    eyebrow: 'Run registry',
    title: 'Praxist and Lab execution in one operational view',
    description:
      'Process health is shown independently from research completion, retained results and production authority.',
  },
  evidence: {
    eyebrow: 'Evidence matrix',
    title: 'Development, formal, SHADOW and canary evidence stay distinct',
    description:
      'Evidence can support a gate, but it cannot grant wiring authority or become a learning source.',
  },
  system: {
    eyebrow: 'System readiness',
    title: 'Every local owner seam, with failures made visible',
    description:
      'Collector health, command mode and typed warnings are reported without silent fallback.',
  },
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
          {stage.state === 'not_applicable' && (
            <Minus
              className="size-3 text-slate-700"
              aria-label="Not applicable"
            />
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

export function ResearchLabPage({ view }: { view: ResearchLabView }) {
  const {
    snapshot,
    session,
    connection,
    error,
    sessionError,
    refreshing,
    refresh,
  } = useResearchLab();
  const params = useParams();
  const rawTaskId = params?.taskId;
  const taskId = Array.isArray(rawTaskId) ? rawTaskId[0] : rawTaskId;
  const [searchQuery, setSearchQuery] = useState('');
  const scopedItems =
    view === 'discovery'
      ? []
      : selectItemsForView(snapshot?.items ?? [], view, taskId);
  const visibleItems = filterItems(scopedItems, searchQuery);
  const primaryItem =
    view === 'discovery'
      ? null
      : (visibleItems[0] ??
        (view === 'task' || searchQuery.trim()
          ? null
          : (snapshot?.items[0] ?? null)));
  const stages = buildStages(primaryItem);
  const navItems = [
    { label: 'Pipeline', icon: LayoutDashboard, href: '/', key: 'pipeline' },
    {
      label: 'Discovery',
      icon: FileSearch,
      href: '/discovery',
      key: 'discovery',
      count: snapshot?.discovery.awaiting_binding_count ?? 0,
    },
    {
      label: 'Approvals',
      icon: UserRoundCheck,
      href: '/approvals',
      key: 'approvals',
      count: snapshot?.summary.awaiting_human ?? 0,
    },
    {
      label: 'Runs',
      icon: Activity,
      href: '/runs',
      key: 'runs',
      count: snapshot?.summary.active_runs ?? 0,
    },
    { label: 'Evidence', icon: Database, href: '/evidence', key: 'evidence' },
    { label: 'System', icon: Cpu, href: '/system', key: 'system' },
  ] as const;
  const metrics =
    view === 'discovery'
      ? buildDiscoveryMetrics(snapshot)
      : buildMetrics(snapshot, primaryItem);
  const systemChecks = buildSystemChecks(snapshot, session?.mutations_enabled);
  const inspectorFacts =
    view === 'discovery'
      ? buildDiscoveryInspectorFacts(snapshot)
      : buildInspectorFacts(primaryItem);

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

          <div className="relative hidden max-w-md flex-1 md:block">
            <Search className="pointer-events-none absolute left-3 top-1/2 z-10 size-3.5 -translate-y-1/2 text-slate-500" />
            <Input
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              aria-label="Search Research Lab tasks"
              placeholder="Search task, claim, request or run"
              className="h-9 border-white/[0.07] bg-white/[0.025] pl-9 font-mono text-xs text-slate-300 placeholder:text-slate-600"
            />
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

      <nav
        className="flex gap-1 overflow-x-auto border-b border-white/[0.07] px-3 py-2 xl:hidden"
        aria-label="Research Lab navigation"
      >
        {navItems.map((item) => {
          const Icon = item.icon;
          const active =
            item.key === view || (view === 'task' && item.key === 'pipeline');
          return (
            <Link
              key={item.key}
              href={item.href}
              aria-current={active ? 'page' : undefined}
              className={buttonVariants({
                variant: 'ghost',
                size: 'sm',
                className: active
                  ? 'bg-cyan-300/[0.08] text-cyan-100'
                  : 'text-slate-500',
              })}
            >
              <Icon className="size-3.5" /> {item.label}
            </Link>
          );
        })}
      </nav>

      <div className="grid min-h-[calc(100vh-64px)] grid-cols-1 xl:grid-cols-[210px_minmax(720px,1fr)_360px]">
        <aside className="hidden border-r border-white/[0.07] px-3 py-5 xl:flex xl:flex-col">
          <nav className="space-y-1" aria-label="Research Lab navigation">
            {navItems.map((item) => {
              const Icon = item.icon;
              const active =
                item.key === view ||
                (view === 'task' && item.key === 'pipeline');
              return (
                <Link
                  key={item.key}
                  href={item.href}
                  aria-current={active ? 'page' : undefined}
                  className={buttonVariants({
                    variant: 'ghost',
                    className: `h-9 w-full justify-start px-3 ${
                      active
                        ? 'bg-cyan-300/[0.08] text-cyan-100 hover:bg-cyan-300/[0.1] hover:text-cyan-100'
                        : 'text-slate-500 hover:bg-white/[0.04] hover:text-slate-200'
                    }`,
                  })}
                >
                  <Icon className="size-4" />
                  {item.label}
                  {'count' in item && (
                    <span className="ml-auto font-mono text-[10px] text-slate-600">
                      {item.count}
                    </span>
                  )}
                </Link>
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
                {viewCopy[view].eyebrow}
              </p>
              <h1 className="mt-1 text-2xl font-semibold tracking-[-0.03em] text-slate-50">
                {view === 'task' && primaryItem
                  ? primaryItem.title
                  : viewCopy[view].title}
              </h1>
              <p className="mt-1 max-w-2xl text-sm text-slate-500">
                {view === 'task' && primaryItem
                  ? primaryItem.objective
                  : viewCopy[view].description}
              </p>
            </div>
            <div className="flex items-center gap-2">
              {primaryItem && view !== 'task' && (
                <Link
                  href={`/tasks/${encodeURIComponent(primaryItem.task_id)}`}
                  className={buttonVariants({
                    variant: 'outline',
                    className:
                      'border-white/10 bg-white/[0.025] text-slate-300',
                  })}
                >
                  <GitBranch className="size-4" /> View lineage
                </Link>
              )}
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

          {(view === 'pipeline' || view === 'task') && (
            <div className="mt-6 overflow-x-auto pb-2">
              <div className="flex min-w-[980px] gap-3">
                {stages.map((stage, index) => (
                  <StageNode key={stage.label} stage={stage} index={index} />
                ))}
              </div>
            </div>
          )}

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

          <PrimaryWorkspace
            view={view}
            items={visibleItems}
            snapshot={snapshot}
            session={session}
            connection={connection}
            taskId={taskId}
            onRefresh={refresh}
          />

          {(view === 'pipeline' || view === 'task') && (
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
          )}
        </section>

        <aside className="border-l border-white/[0.07] bg-black/10 px-5 py-5 xl:block">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-mono text-[9px] uppercase tracking-[0.18em] text-slate-600">
                Inspector
              </p>
              <h2 className="mt-1 text-sm font-semibold text-slate-200">
                {view === 'discovery'
                  ? 'Discovery authority guard'
                  : (primaryItem?.title ?? 'No selected research task')}
              </h2>
              {primaryItem && (
                <div className="mt-2">
                  <TrackBadge mode={primaryItem.research_mode} />
                </div>
              )}
            </div>
            <Badge className={stageBadge(primaryItem?.lifecycle.stage)}>
              {view === 'discovery'
                ? `${snapshot?.discovery.awaiting_binding_count ?? 0} unbound`
                : primaryItem
                ? stageLabels[primaryItem.lifecycle.stage]
                : connection}
            </Badge>
          </div>

          {view === 'discovery' ? (
            <div className="mt-4 rounded-xl border border-cyan-300/15 bg-cyan-300/[0.04] p-4">
              <p className="flex items-center gap-2 text-xs text-cyan-100">
                <FileSearch className="size-4" /> Proposal-only discovery
              </p>
              <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
                Codex can inspect only the frozen corpus. It cannot choose the
                task mapping, approve A0, start Praxist, or promote a result.
              </p>
            </div>
          ) : (
            <InspectorBanner item={primaryItem} connection={connection} />
          )}

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

          {view !== 'discovery' && (
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
          )}

          <div className="mt-5 rounded-xl border border-white/[0.07] bg-white/[0.02] p-4">
            <p className="flex items-center gap-2 text-xs text-slate-300">
              <ShieldCheck className="size-4 text-cyan-300" /> Authority
              boundary
            </p>
            {view === 'discovery' ? (
              <ul className="mt-3 space-y-2 text-[11px] text-slate-500">
                <AuthorityLine
                  allowed={Boolean(snapshot?.discovery.open_demand_count)}
                  text="Volvence Demand published"
                />
                <AuthorityLine
                  allowed={Boolean(snapshot?.discovery.proposal_count)}
                  text="Read-only Codex proposal sealed"
                />
                <AuthorityLine
                  allowed={false}
                  text="Named-human topic binding"
                />
                <AuthorityLine allowed={false} text="Separate A0 approval" />
                <NeutralAuthorityLine text="Candidate import: not authorized" />
                <NeutralAuthorityLine text="SHADOW / ACTIVE: unchanged" />
              </ul>
            ) : primaryItem?.research_mode === 'external_simulation' ? (
              <ul className="mt-3 space-y-2 text-[11px] text-slate-500">
                <AuthorityLine
                  allowed={primaryItem.authority.a0_research_start_authorized}
                  text="A0 exact research authorization"
                />
                <AuthorityLine
                  allowed={primaryItem.bindings.some(
                    (binding) => binding.kind === 'external handoff',
                  )}
                  text="Immutable simulation handoff"
                />
                <NeutralAuthorityLine text="Formal validation: external domain owned" />
                <NeutralAuthorityLine text="ModificationGate: not applicable" />
                <NeutralAuthorityLine text="SHADOW / ACTIVE wiring: not applicable" />
                <NeutralAuthorityLine text="Adoption: proposal only, Foundry owned" />
              </ul>
            ) : (
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
                    primaryItem?.authority.modification_gate_decision ===
                    'allow'
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
                  allowed={
                    primaryItem?.authority.authorized_wiring === 'active'
                  }
                  text={`A2 authorization: ${primaryItem?.authority.authorized_wiring ?? 'unknown'}`}
                />
                <AuthorityLine
                  allowed={primaryItem?.authority.runtime_wiring === 'active'}
                  text={`ACTIVE applied: ${primaryItem?.authority.runtime_wiring ?? 'unknown'}`}
                />
              </ul>
            )}
          </div>

          {view !== 'discovery' && (
            <ResearchCommandWorkbench
              item={primaryItem}
              snapshot={snapshot}
              session={session}
              sessionError={sessionError}
              onRefresh={refresh}
            />
          )}

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

export default function Home() {
  return <ResearchLabPage view="pipeline" />;
}

function PrimaryWorkspace({
  view,
  items,
  snapshot,
  session,
  connection,
  taskId,
  onRefresh,
}: {
  view: ResearchLabView;
  items: ResearchLabItem[];
  snapshot: ResearchLabSnapshot | null;
  session: ReturnType<typeof useResearchLab>['session'];
  connection: 'loading' | 'live' | 'degraded' | 'offline';
  taskId: string | undefined;
  onRefresh: () => void;
}) {
  if (view === 'discovery')
    return (
      <ResearchDiscoveryInbox
        snapshot={snapshot}
        session={session}
        onRefresh={onRefresh}
      />
    );
  if (view === 'evidence')
    return <EvidenceMatrix items={items} connection={connection} />;
  if (view === 'system') {
    return (
      <SystemOverview
        snapshot={snapshot}
        mutationsEnabled={session?.mutations_enabled}
      />
    );
  }
  const titles: Record<
    'pipeline' | 'task' | 'approvals' | 'runs',
    [string, string]
  > = {
    pipeline: [
      'Research queue',
      'Exact tasks ordered by the next human or system gate.',
    ],
    task: [
      'Exact task lineage',
      'One task, its immutable bindings and current owner-published state.',
    ],
    approvals: [
      'Exact approval inbox',
      'Only tasks with an A0, A1 or A2 human decision currently available.',
    ],
    runs: [
      'Praxist and Lab runs',
      'Registered processes remain separate from evidence and wiring authority.',
    ],
  };
  const [title, description] = titles[view];
  const emptyMessage =
    view === 'task' && taskId
      ? `No exact Research Task matches ${taskId}.`
      : view === 'approvals'
        ? 'No exact human review is currently waiting.'
        : view === 'runs'
          ? 'No registered research run is currently visible.'
          : 'No valid Research Task is currently available.';
  return (
    <Card className="mt-5 border-0 bg-card/80 ring-white/[0.07]">
      <CardHeader className="border-b border-white/[0.06]">
        <CardTitle className="flex items-center gap-2 text-sm text-slate-200">
          <FlaskConical className="size-4 text-cyan-300" /> {title}
        </CardTitle>
        <CardDescription>{description}</CardDescription>
        <CardAction>
          <Badge
            variant="outline"
            className="border-white/10 font-mono text-[10px] text-slate-500"
          >
            {items.length} items
          </Badge>
        </CardAction>
      </CardHeader>
      <CardContent className="px-0">
        <Table>
          <TableHeader>
            <TableRow className="border-white/[0.06] hover:bg-transparent">
              {[
                'Task',
                'Owner',
                'Stage',
                'Evidence',
                'Next gate',
                'Updated',
              ].map((label, index) => (
                <TableHead
                  key={label}
                  className={`${index === 0 ? 'pl-4' : ''} ${index === 5 ? 'pr-4 text-right' : ''} text-[10px] uppercase tracking-wider text-slate-600`}
                >
                  {label}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {items.length ? (
              items.map((item) => (
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
                    : emptyMessage}
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}

function EvidenceMatrix({
  items,
  connection,
}: {
  items: ResearchLabItem[];
  connection: 'loading' | 'live' | 'degraded' | 'offline';
}) {
  return (
    <Card className="mt-5 border-0 bg-card/80 ring-white/[0.07]">
      <CardHeader className="border-b border-white/[0.06]">
        <CardTitle className="flex items-center gap-2 text-sm text-slate-200">
          <Database className="size-4 text-cyan-300" /> Evidence tiers
        </CardTitle>
        <CardDescription>
          No score or completion flag is allowed to collapse these four columns.
        </CardDescription>
      </CardHeader>
      <CardContent className="px-0">
        <Table>
          <TableHeader>
            <TableRow className="border-white/[0.06] hover:bg-transparent">
              {[
                'Task',
                'Track',
                'Development',
                'Formal',
                'SHADOW',
                'Canary',
              ].map((label, index) => (
                <TableHead
                  key={label}
                  className={`${index === 0 ? 'pl-4' : ''} text-[10px] uppercase tracking-wider text-slate-600`}
                >
                  {label}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {items.length ? (
              items.map((item) => (
                <TableRow key={item.item_id} className="border-white/[0.06]">
                  <TableCell className="pl-4">
                    <Link
                      href={`/tasks/${encodeURIComponent(item.task_id)}`}
                      className="font-medium text-slate-200 hover:text-cyan-200"
                    >
                      {item.title}
                    </Link>
                  </TableCell>
                  <TableCell>
                    <TrackBadge mode={item.research_mode} />
                  </TableCell>
                  {[
                    item.evidence.development,
                    item.evidence.formal,
                    item.evidence.shadow,
                    item.evidence.canary,
                  ].map((value, index) => (
                    <TableCell key={`${item.item_id}-${index}`}>
                      <span className="font-mono text-[10px] text-slate-400">
                        {value}
                      </span>
                    </TableCell>
                  ))}
                </TableRow>
              ))
            ) : (
              <TableRow className="border-white/[0.06] hover:bg-transparent">
                <TableCell
                  colSpan={6}
                  className="h-28 text-center text-xs text-slate-600"
                >
                  {connection === 'loading'
                    ? 'Loading evidence tiers…'
                    : 'No exact evidence bindings are visible.'}
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}

function SystemOverview({
  snapshot,
  mutationsEnabled,
}: {
  snapshot: ResearchLabSnapshot | null;
  mutationsEnabled: boolean | undefined;
}) {
  return (
    <div className="mt-5 grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(300px,.55fr)]">
      <Card className="border-0 bg-card/80 ring-white/[0.07]">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm text-slate-200">
            <Cpu className="size-4 text-emerald-300" /> Owner sources
          </CardTitle>
          <CardDescription>
            Each source publishes its own health and artifact count.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3 sm:grid-cols-2">
          {(snapshot?.source_health ?? []).map((source) => (
            <div
              key={source.source}
              className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-4"
            >
              <div className="flex items-center justify-between gap-3">
                <span className="font-mono text-xs text-slate-300">
                  {source.source}
                </span>
                <Badge variant="outline" className={healthBadge(source.status)}>
                  {source.status}
                </Badge>
              </div>
              <p className="mt-3 text-[11px] text-slate-500">{source.detail}</p>
              <p className="mt-2 font-mono text-[10px] text-slate-600">
                {source.artifacts_seen} artifacts
              </p>
            </div>
          ))}
          {!snapshot && (
            <p className="text-xs text-slate-600">
              Collector state is unavailable.
            </p>
          )}
        </CardContent>
      </Card>
      <Card className="border-0 bg-card/80 ring-white/[0.07]">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm text-slate-200">
            <AlertTriangle className="size-4 text-amber-300" /> Typed warnings
          </CardTitle>
          <CardDescription>
            Command mode:{' '}
            {mutationsEnabled ? 'controlled exact delegation' : 'read-only'}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {snapshot?.warnings.length ? (
            snapshot.warnings.map((warning, index) => (
              <div
                key={`${warning.code}-${index}`}
                className="rounded-lg border border-amber-300/15 bg-amber-300/[0.04] p-3"
              >
                <p className="font-mono text-[10px] text-amber-200">
                  {warning.code}
                </p>
                <p className="mt-1 text-[11px] leading-relaxed text-slate-500">
                  {warning.message}
                </p>
              </div>
            ))
          ) : (
            <p className="rounded-lg border border-emerald-300/10 bg-emerald-300/[0.03] p-3 text-xs text-emerald-200/70">
              No collector warning is currently published.
            </p>
          )}
        </CardContent>
      </Card>
    </div>
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
            <Link
              href={`/tasks/${encodeURIComponent(item.task_id)}`}
              className="block max-w-[260px] truncate font-medium text-slate-200 hover:text-cyan-200"
            >
              {item.title}
            </Link>
            <p className="font-mono text-[10px] text-slate-600">
              {item.claim_id}
            </p>
            <div className="mt-1">
              <TrackBadge mode={item.research_mode} />
            </div>
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

function TrackBadge({ mode }: { mode: ResearchLabItem['research_mode'] }) {
  const external = mode === 'external_simulation';
  return (
    <Badge
      variant="outline"
      className={
        external
          ? 'border-violet-300/20 bg-violet-300/[0.05] font-mono text-[9px] text-violet-200'
          : 'border-cyan-300/15 bg-cyan-300/[0.04] font-mono text-[9px] text-cyan-200/80'
      }
    >
      {external ? 'simulation' : 'volvence promotion'}
    </Badge>
  );
}

function selectItemsForView(
  items: ResearchLabItem[],
  view: ResearchLabView,
  taskId: string | undefined,
): ResearchLabItem[] {
  if (view === 'task') {
    return taskId ? items.filter((item) => item.task_id === taskId) : [];
  }
  if (view === 'approvals') {
    const approvalActions = new Set([
      'review_a0',
      'authorize_shadow',
      'authorize_active',
    ]);
    return items.filter(
      (item) =>
        ['AWAITING_A0', 'AWAITING_A1', 'AWAITING_A2'].includes(
          item.lifecycle.stage,
        ) ||
        item.available_actions.some((action) => approvalActions.has(action)),
    );
  }
  if (view === 'runs') return items.filter((item) => item.run !== null);
  return items;
}

function filterItems(
  items: ResearchLabItem[],
  rawQuery: string,
): ResearchLabItem[] {
  const query = rawQuery.trim().toLocaleLowerCase();
  if (!query) return items;
  return items.filter((item) => {
    const searchable = [
      item.task_id,
      item.title,
      item.claim_id,
      item.owner,
      item.run?.run_id,
      item.run?.model,
      ...item.bindings.flatMap((binding) => [
        binding.artifact_id,
        binding.sha256,
      ]),
    ];
    return searchable.some((value) =>
      value?.toLocaleLowerCase().includes(query),
    );
  });
}

function healthBadge(status: 'healthy' | 'degraded' | 'unavailable'): string {
  if (status === 'healthy') {
    return 'border-emerald-300/20 bg-emerald-300/[0.05] font-mono text-[9px] text-emerald-200';
  }
  if (status === 'degraded') {
    return 'border-amber-300/20 bg-amber-300/[0.05] font-mono text-[9px] text-amber-200';
  }
  return 'border-rose-300/20 bg-rose-300/[0.05] font-mono text-[9px] text-rose-200';
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

function NeutralAuthorityLine({ text }: { text: string }) {
  return (
    <li className="flex gap-2">
      <Minus className="mt-0.5 size-3 shrink-0 text-violet-300/70" />
      {text}
    </li>
  );
}

function buildStages(item: ResearchLabItem | null): StageNodeData[] {
  const currentIndex = pipelineIndex(item);
  return stageBlueprints.map((stage, index) => ({
    ...stage,
    state:
      item?.research_mode === 'external_simulation' && index > 2
        ? 'not_applicable'
        : index < currentIndex
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

function buildDiscoveryMetrics(snapshot: ResearchLabSnapshot | null) {
  return [
    {
      label: 'Open demands',
      value: String(snapshot?.discovery.open_demand_count ?? 0),
      detail: `${snapshot?.discovery.demand_count ?? 0} exact Demand artifacts`,
      icon: FileSearch,
    },
    {
      label: 'Topic proposals',
      value: String(snapshot?.discovery.proposal_count ?? 0),
      detail: 'Read-only Codex output',
      icon: FlaskConical,
    },
    {
      label: 'Needs binding',
      value: String(snapshot?.discovery.awaiting_binding_count ?? 0),
      detail: 'Named human decision',
      icon: GitBranch,
    },
    {
      label: 'Awaiting A0',
      value: String(snapshot?.discovery.awaiting_a0_count ?? 0),
      detail: 'Separate research-start approval',
      icon: UserRoundCheck,
    },
  ];
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

function buildDiscoveryInspectorFacts(
  snapshot: ResearchLabSnapshot | null,
): Array<[string, string]> {
  const discovery = snapshot?.discovery;
  const latestModel = discovery?.demands.find(
    (demand) => demand.run_model,
  )?.run_model;
  return [
    ['Open demands', String(discovery?.open_demand_count ?? 0)],
    ['Proposals', String(discovery?.proposal_count ?? 0)],
    ['Unbound', String(discovery?.awaiting_binding_count ?? 0)],
    ['Awaiting A0', String(discovery?.awaiting_a0_count ?? 0)],
    ['Registry SHA', shorten(discovery?.registry?.sha256)],
    ['Discovery model', latestModel ?? 'waiting for first run'],
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
