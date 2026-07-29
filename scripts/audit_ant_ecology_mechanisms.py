"""Run the P0 digital-ant ecology mechanism audit and emit its evidence bundle.

Exit codes are part of the contract: ``0`` only for a PASS verdict, ``1`` for a
BLOCK.  A driver that always returns ``0`` turns an honest BLOCK into a green
CI run, which is exactly how the vacuous v1 PASS survived.

``research/ant/05_ecology_p0_p1_p2_plan.md`` s2.1 and s2.3 fix what a run must
leave behind, and none of it is optional:

* a per-run directory ``research/ant/results/ecology_recovery/p0/<run-id>/`` --
  a fixed filename would silently overwrite the previous BLOCK artifact;
* the canonical JSON report, plus the sidecar integrity manifest carrying git
  SHA, dirty flag, config digest, dependency versions, device, training seeds,
  layout seeds and model fingerprint;
* a human-readable Markdown summary;
* the raw per-tick temporal trace.

Provenance, digests and the no-overwrite refusal come from
``volvence_ant.evidence.provenance`` -- this driver never re-implements them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from volvence_ant.evidence.provenance import (
    collect_ant_provenance,
    ensure_artifact_writable,
    file_digest,
    stable_json_digest,
    write_ant_artifact_bundle,
)
from volvence_ant.experiments.ecology_mechanism_audit import (
    ECOLOGY_MECHANISM_AUDIT_SCHEMA_VERSION,
    EcologyMechanismAuditConfig,
    EcologyMechanismAuditReport,
    ecology_mechanism_audit_seed_schedule,
    run_ecology_mechanism_audit,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_ROOT = Path("research/ant/results/ecology_recovery/p0")
# The artifact filename carries the schema version so a new shape can never be
# confused with a committed artifact of a different shape (plan 05 s2.1).
_SCHEMA_TOKEN = ECOLOGY_MECHANISM_AUDIT_SCHEMA_VERSION.rsplit(".", 1)[-1]
_ARTIFACT_NAME = f"ecology_mechanism_audit.{_SCHEMA_TOKEN}.json"
_SUMMARY_NAME = "summary.md"
_RAW_TRACE_NAME = "raw/temporal_ticks.jsonl"
_TEST_COMMAND = (
    'cd <repo> && export PYTHONPATH="$(ls -d packages/*/src | paste -sd: -)" '
    "&& .venv/bin/python -m pytest "
    "packages/vz-embodiment-ant/tests/test_ecology_mechanism_audit.py "
    "packages/vz-embodiment-ant/tests/test_ecology_action_chain_audit.py "
    "packages/vz-embodiment-ant/tests/test_ecology_temporal_switch_audit.py "
    "packages/vz-embodiment-ant/tests/test_ecology_frozen_evaluation.py "
    "-q --no-header -p no:cacheprovider"
)


def _repo_path(path: Path) -> Path:
    resolved = path if path.is_absolute() else _REPO_ROOT / path
    resolved.relative_to(_REPO_ROOT)
    return resolved


def _default_run_id(config: EcologyMechanismAuditConfig) -> str:
    """A new run-id per run: UTC second plus the config digest prefix.

    The digest makes two different configurations distinguishable at a glance;
    the timestamp makes two runs of the same configuration distinct rather
    than colliding on a name whose previous artifact would be destroyed.
    """

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = stable_json_digest(asdict(config))[:8]
    return f"{stamp}-seed{config.seed}-{digest}"


def _model_fingerprint(report: EcologyMechanismAuditReport) -> str:
    """Digest the checkpoints the verdict was actually taken on.

    plan 05 s2.1 requires a model fingerprint per run.  The gated artefact is
    the final learned colony, so its per-body owner-published fingerprints are
    what the record must bind to.
    """

    bodies = tuple(
        {
            "body_id": body.body_id,
            "checkpoint_id": body.checkpoint_id,
            "policy_fingerprint": body.policy_fingerprint,
            "temporal_learning_fingerprint": (
                body.temporal_learning_fingerprint
            ),
        }
        for body in report.final_learned_snapshot.body_reports
    )
    if not bodies:
        raise RuntimeError(
            "the final learned snapshot carries no body reports, so the run "
            "has no model fingerprint to record"
        )
    return stable_json_digest(list(bodies))


def _write_raw_temporal_trace(
    *,
    path: Path,
    report: EcologyMechanismAuditReport,
) -> None:
    """One JSON object per scripted tick, per control trace (plan 05 s2.3)."""

    traces = (
        report.temporal_switch.positive_control,
        report.temporal_switch.negative_control,
        report.temporal_switch.segment_credit_off_control,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "trace": trace.label,
                "segment_credit_enabled": trace.segment_credit_enabled,
                "checkpoint_id": trace.checkpoint_id,
                **asdict(tick),
            },
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        for trace in traces
        for tick in trace.ticks
    ]
    if not lines:
        raise RuntimeError(
            "the temporal switch audit produced no ticks; a P0 bundle without "
            "a raw trace does not satisfy plan 05 s2.3"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary_markdown(
    *,
    report: EcologyMechanismAuditReport,
    run_id: str,
    provenance,
) -> str:
    lines: list[str] = [
        f"# P0 ecology mechanism audit — {report.verdict}",
        "",
        f"- run-id: `{run_id}`",
        f"- schema: `{report.schema_version}`",
        f"- description: {report.description}",
        "",
        "## Provenance (plan 05 §2.1)",
        "",
        "| field | value |",
        "|---|---|",
        f"| git SHA | `{provenance.git_sha}` |",
        f"| git branch | `{provenance.git_branch}` |",
        f"| working tree dirty | {provenance.working_tree_dirty} |",
        f"| externally retainable | {provenance.externally_retainable} |",
        f"| config digest | `{provenance.config_digest}` |",
        f"| model fingerprint | `{provenance.model_fingerprint}` |",
        f"| requested device | `{provenance.requested_device}` |",
        f"| effective backend | `{provenance.effective_backend}` |",
        f"| python | {provenance.python_version} |",
        f"| platform | {provenance.platform} |",
        f"| training seeds | {list(provenance.training_seeds)} |",
        f"| layout seeds | {list(provenance.layout_seeds)} |",
        "| dependency versions | "
        + ", ".join(f"{name}={value}" for name, value in provenance.dependency_versions)
        + " |",
        "",
        "## Gates",
        "",
        "| gate | verdict | observed |",
        "|---|---|---|",
    ]
    for gate in report.gates:
        observed = gate.observed.replace("|", "\\|")
        if len(observed) > 400:
            observed = observed[:400] + " …"
        lines.append(
            f"| `{gate.name}` | {'PASS' if gate.passed else 'BLOCK'} | "
            f"{observed} |"
        )
    lines += [
        "",
        "## Breakpoints",
        "",
    ]
    if report.diagnostic_breakpoints:
        lines += [f"- `{name}`" for name in report.diagnostic_breakpoints]
    else:
        lines.append("- none")
    lines += [
        "",
        "## Gate thresholds",
        "",
    ]
    for gate in report.gates:
        lines.append(f"- `{gate.name}`: {gate.threshold}")
    lines += [
        "",
        "## First failing learned episode (plan 05:130 bisect trigger)",
        "",
        f"- {report.first_failing_learned_episode or 'none'}",
        "",
        "## Declared diagnostic-only surfaces",
        "",
    ]
    for surface in report.diagnostic_surfaces:
        lines.append(
            f"- `{surface.name}` — gated={surface.gated}: {surface.reason}"
        )
    lines += [
        "",
        "## Declared gaps against the plan",
        "",
    ]
    for gap in report.declared_gaps:
        lines += [
            f"### {gap.plan_reference}",
            "",
            f"- requirement: {gap.requirement}",
            f"- status: {gap.status}",
            f"- owner: {gap.owner}",
            f"- currently failing a gate: {gap.gate_failing}",
            "",
        ]
    lines += [
        "## Test command",
        "",
        "```text",
        _TEST_COMMAND,
        "```",
        "",
    ]
    return "\n".join(lines)


async def _run(args: argparse.Namespace) -> int:
    config = EcologyMechanismAuditConfig(
        n_ants=args.n_ants,
        temporal_latent_dim=args.temporal_latent_dim,
        episode_rounds=args.episode_rounds,
        episodes_per_stage=args.episodes_per_stage,
        evaluation_rounds=args.evaluation_rounds,
        seed=args.seed,
    )
    run_id = args.run_id or _default_run_id(config)
    run_directory = _repo_path(args.results_root / run_id)
    artifact_path = run_directory / _ARTIFACT_NAME
    # Refuse the name BEFORE paying for a training schedule: a collision that
    # is only discovered at write time has already burnt the budget.
    ensure_artifact_writable(artifact_path, overwrite=args.overwrite)
    # Validate and freeze the exact seed namespaces before paying for the
    # audit. This used to run only while assembling provenance after the full
    # learned/no-optimize workload, so the default seed-0 run spent its whole
    # budget before reporting the training/held-out collision at seed 101.
    training_seeds, layout_seeds = ecology_mechanism_audit_seed_schedule(
        config
    )

    report = await run_ecology_mechanism_audit(config)

    summary_path = run_directory / _SUMMARY_NAME
    raw_trace_path = run_directory / _RAW_TRACE_NAME
    _write_raw_temporal_trace(path=raw_trace_path, report=report)

    provenance = collect_ant_provenance(
        repo_root=_REPO_ROOT,
        seeds=training_seeds + layout_seeds,
        config=asdict(config),
        model_fingerprint=_model_fingerprint(report),
        device=args.device,
        training_seeds=training_seeds,
        layout_seeds=layout_seeds,
    )
    summary_path.write_text(
        _summary_markdown(
            report=report,
            run_id=run_id,
            provenance=provenance,
        ),
        encoding="utf-8",
    )
    payload = {
        **report.to_dict(),
        "run_id": run_id,
        # The sidecar files are outputs of this run, not inputs, so they are
        # bound to the bundle through the artifact payload the manifest hashes
        # rather than being mislabelled as manifest inputs.
        "bundle_files": [
            asdict(file_digest(path, relative_to=_REPO_ROOT))
            for path in (summary_path, raw_trace_path)
        ],
        "test_command": _TEST_COMMAND,
    }
    manifest_path = write_ant_artifact_bundle(
        artifact_path=artifact_path,
        payload=payload,
        provenance=provenance,
        repo_root=_REPO_ROOT,
        overwrite=args.overwrite,
    )

    print(report.description)
    print(f"artifact: {artifact_path.relative_to(_REPO_ROOT)}")
    print(f"manifest: {manifest_path.relative_to(_REPO_ROOT)}")
    print(f"summary:  {summary_path.relative_to(_REPO_ROOT)}")
    print(f"raw:      {raw_trace_path.relative_to(_REPO_ROOT)}")
    if report.verdict != "PASS":
        print(
            "BLOCK gates: " + ", ".join(report.diagnostic_breakpoints),
        )
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit digital-ant ecology action, temporal, and freeze chains"
        )
    )
    parser.add_argument("--n-ants", type=int, default=4)
    parser.add_argument("--temporal-latent-dim", type=int, default=16)
    parser.add_argument("--episode-rounds", type=int, default=12)
    parser.add_argument("--episodes-per-stage", type=int, default=3)
    parser.add_argument("--evaluation-rounds", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "device this run is asked to use; recorded in the manifest and "
            "resolved to the backend that actually runs"
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="explicit run-id directory name (default: UTC stamp + digest)",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=_RESULTS_ROOT,
        help="parent directory that receives the <run-id> bundle directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "destroy an existing artifact at this run-id; plan 05 s2.1 "
            "forbids overwriting an existing BLOCK artifact, so this is only "
            "for re-running a run-id you created yourself"
        ),
    )
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
