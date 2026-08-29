import {
  Activity,
  ArrowUpRight,
  Bell,
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
  Search,
  ShieldCheck,
  UserRoundCheck,
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

const stages = [
  { label: 'Forge', detail: 'Opportunity', state: 'complete' },
  { label: 'A0', detail: 'Human review', state: 'complete' },
  { label: 'Praxist', detail: 'Research run', state: 'current' },
  { label: 'Formal', detail: 'Sealed validation', state: 'locked' },
  { label: 'Gate', detail: 'ModificationGate', state: 'locked' },
  { label: 'Shadow', detail: 'A1 observation', state: 'locked' },
  { label: 'Active', detail: 'A2 canary', state: 'locked' },
] as const;

const navItems = [
  { label: 'Pipeline', icon: LayoutDashboard, active: true },
  { label: 'Approvals', icon: UserRoundCheck, count: 0 },
  { label: 'Runs', icon: Activity, count: 1 },
  { label: 'Evidence', icon: Database },
  { label: 'System', icon: Cpu },
] as const;

function StageNode({
  stage,
  index,
}: {
  stage: (typeof stages)[number];
  index: number;
}) {
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
      {index < stages.length - 1 && (
        <ChevronRight className="absolute -right-3.5 top-1/2 z-20 size-3 -translate-y-1/2 text-slate-600" />
      )}
    </div>
  );
}

export default function Home() {
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
            <Badge
              variant="outline"
              className="hidden border-emerald-300/20 bg-emerald-300/[0.06] font-mono text-[10px] text-emerald-200 lg:flex"
            >
              <span className="size-1.5 rounded-full bg-emerald-300" /> local
            </Badge>
            <Button variant="ghost" size="icon" aria-label="Notifications">
              <Bell className="size-4 text-slate-400" />
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
              <ShieldCheck className="size-4 text-emerald-300" />
              Authority guard
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
              <Button className="bg-cyan-300 text-slate-950 hover:bg-cyan-200">
                View live run <ArrowUpRight className="size-4" />
              </Button>
            </div>
          </div>

          <div className="mt-6 overflow-x-auto pb-2">
            <div className="flex min-w-[980px] gap-3">
              {stages.map((stage, index) => (
                <StageNode key={stage.label} stage={stage} index={index} />
              ))}
            </div>
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {[
              {
                label: 'Registered tasks',
                value: '1',
                detail: '1 exact mapping',
                icon: Boxes,
              },
              {
                label: 'Awaiting human',
                value: '0',
                detail: 'A0 completed',
                icon: UserRoundCheck,
              },
              {
                label: 'Active runs',
                value: '1',
                detail: 'PID 47372 · gen 0',
                icon: Activity,
              },
              {
                label: 'Production active',
                value: '0',
                detail: 'No authority granted',
                icon: ShieldCheck,
              },
            ].map((metric) => {
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
                  1 item
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
                  <TableRow className="border-white/[0.06] bg-cyan-300/[0.025] hover:bg-cyan-300/[0.045]">
                    <TableCell className="pl-4">
                      <div className="flex items-center gap-3">
                        <span className="flex size-8 items-center justify-center rounded-lg border border-cyan-300/20 bg-cyan-300/[0.07] text-cyan-200">
                          <Boxes className="size-4" />
                        </span>
                        <div>
                          <p className="font-medium text-slate-200">
                            Coding memory inheritance
                          </p>
                          <p className="font-mono text-[10px] text-slate-600">
                            claim:coding-memory-scaling
                          </p>
                        </div>
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge
                        variant="outline"
                        className="border-white/10 font-mono text-[10px] text-slate-400"
                      >
                        vz-memory
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <span className="inline-flex items-center gap-1.5 text-xs text-cyan-200">
                        <span className="size-1.5 animate-pulse rounded-full bg-cyan-300" />{' '}
                        Research running
                      </span>
                    </TableCell>
                    <TableCell>
                      <div className="space-y-1.5">
                        <div className="flex items-center justify-between gap-4 font-mono text-[10px] text-slate-500">
                          <span>generation 0</span>
                          <span>4 initializing</span>
                        </div>
                        <Progress
                          value={8}
                          className="w-28 [&_[data-slot=progress-indicator]]:bg-cyan-300"
                        />
                      </div>
                    </TableCell>
                    <TableCell>
                      <span className="inline-flex items-center gap-1.5 text-xs text-slate-300">
                        <FlaskConical className="size-3.5 text-cyan-300" />{' '}
                        Retained candidate
                      </span>
                    </TableCell>
                    <TableCell className="pr-4 text-right font-mono text-[10px] text-slate-600">
                      14:27 UTC
                    </TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </CardContent>
          </Card>

          <div className="mt-5 grid gap-3 lg:grid-cols-2">
            <Card className="border-0 bg-card/80 ring-white/[0.07]">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-sm text-slate-200">
                  <Database className="size-4 text-rose-300" /> Baseline finding
                </CardTitle>
                <CardDescription>
                  Public replay v2 · development evidence only
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <div className="mb-2 flex items-center justify-between text-xs">
                    <span className="text-slate-500">Context ratio</span>
                    <span className="font-mono text-rose-300">
                      0.1233 / 0.1000 max
                    </span>
                  </div>
                  <Progress
                    value={100}
                    className="[&_[data-slot=progress-indicator]]:bg-rose-400"
                  />
                </div>
                <div className="grid grid-cols-3 gap-3 border-t border-white/[0.06] pt-4">
                  <div>
                    <p className="font-mono text-sm text-slate-200">0.9720</p>
                    <p className="text-[10px] text-slate-600">
                      recalled retention
                    </p>
                  </div>
                  <div>
                    <p className="font-mono text-sm text-slate-200">0.9846</p>
                    <p className="text-[10px] text-slate-600">
                      failed retention
                    </p>
                  </div>
                  <div>
                    <p className="font-mono text-sm text-rose-300">0.0</p>
                    <p className="text-[10px] text-slate-600">
                      strict budget pass
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="border-0 bg-card/80 ring-white/[0.07]">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-sm text-slate-200">
                  <Cpu className="size-4 text-emerald-300" /> Local system
                </CardTitle>
                <CardDescription>
                  Readiness from exact host-bound checks
                </CardDescription>
              </CardHeader>
              <CardContent className="grid grid-cols-2 gap-3">
                {[
                  ['Forge scanner', 'ready'],
                  ['Praxist doctor', 'ready'],
                  ['Formal validator', 'blocked'],
                  ['Target adapter', 'blocked'],
                ].map(([label, state]) => (
                  <div
                    key={label}
                    className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[11px] text-slate-500">
                        {label}
                      </span>
                      <span
                        className={`size-1.5 rounded-full ${state === 'ready' ? 'bg-emerald-300' : 'bg-slate-600'}`}
                      />
                    </div>
                    <p
                      className={`mt-2 font-mono text-[10px] uppercase ${state === 'ready' ? 'text-emerald-300' : 'text-slate-600'}`}
                    >
                      {state}
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
                Praxist live run
              </h2>
            </div>
            <Badge className="bg-cyan-300/10 text-cyan-200">running</Badge>
          </div>

          <div className="mt-5 rounded-xl border border-cyan-300/20 bg-cyan-300/[0.05] p-4">
            <div className="flex items-start gap-3">
              <Activity className="mt-0.5 size-4 text-cyan-300" />
              <div>
                <p className="text-xs font-medium text-cyan-100">
                  A0 approved · run resumed safely
                </p>
                <p className="mt-1 text-[11px] leading-relaxed text-cyan-100/55">
                  The registry reports one live controller. Four peers are
                  initializing in generation 0; no duplicate run was started.
                </p>
              </div>
            </div>
          </div>

          <dl className="mt-5 space-y-4">
            {[
              ['Request', '8f44be1d…88ae6be9'],
              ['Request SHA', 'dd7ca6cb…97b68d61'],
              ['Task manifest', '093957ad…d4e6128'],
              ['Run id', 'run_2026-08…inheritance'],
              ['PID / state', '47372 · running'],
              ['Generation', '0 · 4 yellow peers'],
              ['Runtime', 'codex_sdk'],
              ['Model', 'gpt-5.6-luna'],
            ].map(([term, value]) => (
              <div
                key={term}
                className="flex items-start justify-between gap-4 border-b border-white/[0.05] pb-3"
              >
                <dt className="text-[11px] text-slate-600">{term}</dt>
                <dd className="text-right font-mono text-[10px] text-slate-400">
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
                ['Peers', '4'],
                ['Generation', '0'],
                ['Strategy', 'auto'],
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
              <li className="flex gap-2">
                <Check className="mt-0.5 size-3 text-emerald-300" /> A0 exact
                approval consumed
              </li>
              <li className="flex gap-2">
                <LockKeyhole className="mt-0.5 size-3 text-slate-600" /> No
                formal validation authority
              </li>
              <li className="flex gap-2">
                <LockKeyhole className="mt-0.5 size-3 text-slate-600" /> No
                SHADOW or ACTIVE authority
              </li>
            </ul>
          </div>

          <Button className="mt-5 w-full bg-cyan-300 text-slate-950 hover:bg-cyan-200">
            Open live run <ChevronRight className="size-4" />
          </Button>
          <p className="mt-2 text-center text-[10px] text-slate-700">
            Read-only preview · mutations not wired
          </p>
        </aside>
      </div>
    </main>
  );
}
