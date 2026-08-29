import {
  ArrowRight,
  Check,
  CircleDot,
  FileCheck2,
  LockKeyhole,
  ShieldCheck,
} from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import type { ResearchLabItem } from '@/lib/research-lab';

export function PromotionReadiness({ item }: { item: ResearchLabItem | null }) {
  const hasBinding = (kind: string) =>
    item?.bindings.some((binding) => binding.kind === kind) ?? false;
  const milestones = [
    {
      label: 'Candidate',
      value: hasBinding('candidate') ? 'retained' : 'not retained',
      ready: hasBinding('candidate'),
      detail: hasBinding('candidate')
        ? 'Exact Praxist handoff imported'
        : item?.run
          ? 'Waiting for mature handoff'
          : 'No retained research artifact',
    },
    {
      label: 'Formal',
      value: item?.authority.formal_validation_status ?? 'unknown',
      ready: item?.authority.formal_validation_status === 'pass',
      detail: hasBinding('validation')
        ? 'Loop-external evidence bound'
        : 'Sealed validator has not published evidence',
    },
    {
      label: 'Gate',
      value: item?.authority.modification_gate_decision ?? 'unknown',
      ready: item?.authority.modification_gate_decision === 'allow',
      detail: hasBinding('gate')
        ? 'ModificationGate artifact present'
        : 'No ALLOW/DENY owner artifact',
    },
    {
      label: 'Authorized',
      value: item?.authority.authorized_wiring ?? 'disabled',
      ready: ['shadow', 'active'].includes(
        item?.authority.authorized_wiring ?? '',
      ),
      detail: hasBinding('receipt')
        ? 'Forge receipt is present'
        : 'A1/A2 named-human receipt absent',
    },
    {
      label: 'Runtime',
      value: item?.authority.runtime_wiring ?? 'disabled',
      ready: ['shadow', 'active'].includes(
        item?.authority.runtime_wiring ?? '',
      ),
      detail:
        item?.authority.runtime_wiring === 'disabled'
          ? 'Target owner has not applied wiring'
          : 'Target-owned apply state',
    },
  ];

  return (
    <div>
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="flex items-center gap-2 text-sm font-medium text-slate-200">
            <ShieldCheck className="size-4 text-violet-300" /> Promotion
            readiness
          </p>
          <p className="mt-1 text-[11px] text-slate-600">
            Research evidence, authority, and actual runtime remain separate.
          </p>
        </div>
        <Badge
          variant="outline"
          className="border-white/10 font-mono text-[9px] text-slate-500"
        >
          fail closed
        </Badge>
      </div>

      <div className="mt-5 overflow-x-auto pb-1">
        <div className="flex min-w-[700px] items-stretch gap-2">
          {milestones.map((milestone, index) => (
            <div key={milestone.label} className="contents">
              <div
                className={`min-w-0 flex-1 rounded-lg border p-3 ${
                  milestone.ready
                    ? 'border-emerald-300/20 bg-emerald-300/[0.04]'
                    : 'border-white/[0.06] bg-white/[0.02]'
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="text-[10px] uppercase tracking-[0.12em] text-slate-600">
                    {milestone.label}
                  </span>
                  {milestone.ready ? (
                    <Check className="size-3 text-emerald-300" />
                  ) : (
                    <LockKeyhole className="size-3 text-slate-700" />
                  )}
                </div>
                <p
                  className={`mt-3 font-mono text-xs ${
                    milestone.ready ? 'text-emerald-200' : 'text-slate-400'
                  }`}
                >
                  {milestone.value}
                </p>
                <p className="mt-1 text-[9px] leading-relaxed text-slate-700">
                  {milestone.detail}
                </p>
              </div>
              {index < milestones.length - 1 && (
                <ArrowRight className="mt-12 size-3 shrink-0 text-slate-700" />
              )}
            </div>
          ))}
        </div>
      </div>

      <div className="mt-4 grid gap-2 border-t border-white/[0.06] pt-4 sm:grid-cols-3">
        <ReadinessFact
          icon={CircleDot}
          label="Development"
          value={item?.evidence.development ?? 'unavailable'}
        />
        <ReadinessFact
          icon={FileCheck2}
          label="SHADOW evidence"
          value={item?.evidence.shadow ?? 'unavailable'}
        />
        <ReadinessFact
          icon={ShieldCheck}
          label="Canary evidence"
          value={item?.evidence.canary ?? 'unavailable'}
        />
      </div>
    </div>
  );
}

function ReadinessFact({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof CircleDot;
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-lg border border-white/[0.05] bg-white/[0.015] p-3">
      <p className="flex items-center gap-2 text-[10px] text-slate-600">
        <Icon className="size-3" /> {label}
      </p>
      <p className="mt-2 font-mono text-[10px] text-slate-400">{value}</p>
    </div>
  );
}
