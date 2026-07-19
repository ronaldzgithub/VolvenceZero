"""volvence-labs CLI.

用法：
    python -m volvence_labs.cli run --probe pe-baseline-v0 --profile dev
    python -m volvence_labs.cli run --probe r15-rollback-v0 --profile dev
    python -m volvence_labs.cli ls
    python -m volvence_labs.cli ls --probe pe-baseline-v0
    python -m volvence_labs.cli rollback --run <run_id>
    python -m volvence_labs.cli snapshot-show --sha <sha>
    python -m volvence_labs.cli probes
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from typing import Optional

# Importing probes triggers registration.
from . import probes  # noqa: F401
from .framework.probe import get_registry
from .framework.scheduler import run_experiment
from .framework.snapshot import CASStore, RunLog, default_paths
from .framework.wiring import builtin_profiles, get_profile


def _cmd_run(args: argparse.Namespace) -> int:
    # --unit mode: run a single (probe, cell, seed) and output JSON.
    if args.unit:
        return _cmd_run_unit(args)

    # Parse --knob KEY=VALUE pairs into a dict
    knob_overrides = {}
    for kv in getattr(args, "knob", []):
        if "=" not in kv:
            print(f"error: --knob must be KEY=VALUE, got: {kv}", file=sys.stderr)
            return 1
        key, val = kv.split("=", 1)
        # Auto-convert types
        if val.lower() in ("true", "yes", "1"):
            knob_overrides[key] = True
        elif val.lower() in ("false", "no", "0"):
            knob_overrides[key] = False
        else:
            try:
                knob_overrides[key] = float(val) if "." in val else int(val)
            except ValueError:
                knob_overrides[key] = val

    profile = get_profile(args.profile)

    if args.cursor:
        from .framework.parallel import CursorRunner
        runner = CursorRunner(root=args.root)
        report = runner.run(args.probe, profile)
    else:
        report = run_experiment(
            args.probe,
            profile,
            parallel=args.parallel,
            root=args.root,
            knob_overrides=knob_overrides if knob_overrides else None,
        )
    payload = report.to_jsonable()
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    ok = sum(1 for u in payload["units"] if u["ok"])
    total = len(payload["units"])
    print(f"[run] probe={args.probe} profile={args.profile} units={ok}/{total}")
    for u in payload["units"]:
        status = "OK " if u["ok"] else "ERR"
        metrics = ", ".join(f"{k}={v:.4f}" for k, v in u["metrics"].items())
        print(
            f"  {status} {u['cell']:>16} seed={u['seed']} wiring={u['wiring']}  "
            f"{metrics}  run_id={u['run_id']}"
        )
        if not u["ok"] and u["error"]:
            print(f"      error: {u['error'].splitlines()[-1]}")
    gate = payload.get("gate")
    if gate is not None:
        gate_mark = "PASS" if gate["passed"] else "FAIL"
        print(f"[gate] {gate_mark} — {gate['reason']}")
    return 0 if all(u["ok"] for u in payload["units"]) else 1


def _cmd_run_unit(args: argparse.Namespace) -> int:
    """Run a single (probe, cell, seed) unit. Output JSON to stdout."""
    from .framework.scheduler.runner import _run_unit
    result = _run_unit(
        args.probe,
        args.cell,
        int(args.seed),
        args.wiring or "shadow",
        args.root or str(default_paths().root),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("ok") else 1


def _cmd_ls(args: argparse.Namespace) -> int:
    paths = default_paths(args.root)
    store = CASStore(paths)
    log = RunLog(paths, store)
    records = log.list(probe_id=args.probe, limit=args.limit)
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "run_id": r.run_id,
                        "probe_id": r.probe_id,
                        "cell": r.ablation_cell,
                        "wiring": r.wiring,
                        "seed": r.seed,
                        "created_at": r.created_at,
                    }
                    for r in records
                ],
                indent=2,
            )
        )
    else:
        if not records:
            print("(no runs)")
        for r in records:
            print(
                f"{r.created_at:.0f}  {r.run_id}  probe={r.probe_id}  "
                f"cell={r.ablation_cell}  wiring={r.wiring}  seed={r.seed}"
            )
    log.close()
    store.close()
    return 0


def _cmd_rollback(args: argparse.Namespace) -> int:
    """从 CAS + RunLog 重建 experiments/<run_id>/ 目录。"""
    paths = default_paths(args.root)
    store = CASStore(paths)
    log = RunLog(paths, store)
    record = log.get(args.run)
    exp_dir = paths.experiment_dir(record.run_id)
    if exp_dir.exists() and not args.force:
        print(f"refusing to overwrite existing {exp_dir} (use --force)", file=sys.stderr)
        return 2

    if exp_dir.exists():
        shutil.rmtree(exp_dir)

    manifest = store.get_obj(record.manifest_sha)
    readouts = store.get_obj(record.readouts_sha)

    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    ro_dir = exp_dir / "readouts"
    ro_dir.mkdir(exist_ok=True)
    (ro_dir / "readouts.json").write_text(
        json.dumps(readouts, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"rebuilt {exp_dir} from CAS (manifest={record.manifest_sha[:12]})")
    log.close()
    store.close()
    return 0


def _cmd_snapshot_show(args: argparse.Namespace) -> int:
    paths = default_paths(args.root)
    store = CASStore(paths)
    try:
        obj = store.get_obj(args.sha)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(obj, indent=2, ensure_ascii=False))
    store.close()
    return 0


def _cmd_probes(args: argparse.Namespace) -> int:
    reg = get_registry()
    for pid, cls in reg.all_items():
        probe = cls()
        print(f"{pid}")
        print(f"  primitive: {probe.primitive.value}")
        print(f"  r_ids: {list(probe.r_ids)}")
        print(f"  hypothesis: {probe.hypothesis}")
    profiles = builtin_profiles()
    print("\nProfiles:")
    for name, profile in profiles.items():
        print(
            f"  {name}: default={profile.default_level.value} "
            f"seeds={list(profile.seeds)} cells={[c.value for c in profile.cells]}"
        )
    return 0


def _cmd_dashboard(args: argparse.Namespace) -> int:
    from .framework.readout import generate_dashboard
    out = args.out or "docs/dash/index.html"
    html_content = generate_dashboard(out_path=out, root=args.root)
    print(f"dashboard written to {out} ({len(html_content)} bytes)")
    return 0


def _cmd_promote(args: argparse.Namespace) -> int:
    from dataclasses import dataclass
    from .framework.gate import GateAggregator, GateDecision
    from .framework.wiring.promotion import PromotionManager
    from .framework.snapshot import CASStore, RunLog, default_paths

    paths = default_paths(args.root)
    store = CASStore(paths)
    log = RunLog(paths, store)

    # Gather evidence: all runs for this probe
    records = log.list(probe_id=args.probe, limit=200)
    if not records:
        print(f"error: no runs found for probe {args.probe}", file=sys.stderr)
        return 1

    # Build unit-like objects for the aggregator
    @dataclass
    class _Unit:
        ok: bool
        run_id: str
        cell: str
        metrics: dict
        seed: int

    units = []
    for r in records:
        try:
            readouts = store.get_obj(r.readouts_sha)
            metrics = readouts.get("metrics", {})
        except Exception:
            metrics = {}
        units.append(_Unit(ok=True, run_id=r.run_id, cell=r.ablation_cell, metrics=metrics, seed=r.seed))

    # Evaluate gate
    agg = GateAggregator(
        sgm_sigma=0.1,  # tight for high-confidence probes
        sgm_delta=0.05,
        margin_delta=0.01,
        conservative_margin=1.0,
    )
    decision = agg.evaluate(
        args.probe,
        units,
        metric_name=args.metric,
        n_parameters=args.n_params,
        from_level=args.from_level,
        to_level=args.to_level,
        mode=args.mode,
        absolute_threshold=args.threshold,
    )

    print(f"[gate] {decision.decision.value}: {decision.reason}")
    if decision.capacity:
        print(f"  capacity: VC={decision.capacity.vc_dim_estimate:.1f} (threshold={decision.capacity.threshold})")
    if decision.margin:
        print(f"  margin: {decision.margin.margin:.4f} (delta={decision.margin.delta:.4f})")
    print(f"  sgm e-value: {decision.sgm_summary.get('e_value', 0):.4f} (threshold={decision.sgm_summary.get('threshold', 0):.1f})")

    if args.dry_run:
        print("[dry-run] no promotion recorded")
        return 0

    if decision.decision != GateDecision.APPROVE:
        print(f"[skip] gate did not approve — no promotion recorded")
        return 0

    mgr = PromotionManager(args.root)
    record = mgr.promote(args.probe, decision)
    print(f"[promoted] {record.probe_id}: {record.from_level} → {record.to_level}")
    print(f"  promotion_id: {record.promotion_id}")
    print(f"  sha: {record.sha}")
    mgr.close()
    log.close()
    store.close()
    return 0


def _cmd_demote(args: argparse.Namespace) -> int:
    from .framework.wiring.promotion import PromotionManager

    mgr = PromotionManager(args.root)
    promotions = mgr.list_promotions(probe_id=args.probe)
    if not promotions:
        print(f"error: no promotions found for probe {args.probe}", file=sys.stderr)
        mgr.close()
        return 1

    latest = promotions[-1]
    record = mgr.demote(args.probe, latest.sha, reason=args.reason)
    print(f"[demoted] {record.probe_id}: {record.from_level} → {record.to_level}")
    print(f"  demotion_id: {record.demotion_id}")
    print(f"  original_promotion_sha: {record.original_promotion_sha}")
    print(f"  reason: {record.reason}")
    mgr.close()
    return 0


def _cmd_metrics_server(args: argparse.Namespace) -> int:
    from .framework.readout.metrics_exporter import start_metrics_server
    print(f"[metrics] starting Prometheus exporter on {args.host}:{args.port}")
    start_metrics_server(host=args.host, port=args.port, root=args.root)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="volvence-labs")
    parser.add_argument("--root", default=None, help="override labs root dir")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run an experiment")
    p_run.add_argument("--probe", required=True, help="probe id (see `probes`)")
    p_run.add_argument(
        "--profile",
        default="dev",
        help=f"wiring profile (default: dev); known: {sorted(builtin_profiles())}",
    )
    p_run.add_argument("--parallel", action="store_true", help="use multiprocessing")
    p_run.add_argument("--cursor", action="store_true", help="use CursorRunner backend")
    p_run.add_argument("--json", action="store_true", help="print full report as json")
    # --unit mode: run a single (probe, cell, seed) unit
    p_run.add_argument("--unit", action="store_true", help="single-unit mode (for subagent use)")
    p_run.add_argument("--cell", default=None, help="ablation cell (for --unit mode)")
    p_run.add_argument("--seed", default=None, help="seed (for --unit mode)")
    p_run.add_argument("--wiring", default=None, help="wiring level (for --unit mode)")
    p_run.add_argument("--knob", action="append", default=[], metavar="KEY=VALUE",
                       help="override probe knobs (repeatable, e.g. --knob use_real_model=True --knob model_id=TinyLlama/TinyLlama-1.1B-Chat-v1.0)")
    p_run.set_defaults(func=_cmd_run)

    p_ls = sub.add_parser("ls", help="list runs")
    p_ls.add_argument("--probe", default=None, help="filter by probe id")
    p_ls.add_argument("--limit", type=int, default=50)
    p_ls.add_argument("--json", action="store_true")
    p_ls.set_defaults(func=_cmd_ls)

    p_rb = sub.add_parser("rollback", help="rebuild experiments/<run_id>/ from CAS")
    p_rb.add_argument("--run", required=True, dest="run")
    p_rb.add_argument("--force", action="store_true")
    p_rb.set_defaults(func=_cmd_rollback)

    p_ss = sub.add_parser("snapshot-show", help="pretty-print a snapshot object")
    p_ss.add_argument("--sha", required=True)
    p_ss.set_defaults(func=_cmd_snapshot_show)

    p_probes = sub.add_parser("probes", help="list registered probes and profiles")
    p_probes.set_defaults(func=_cmd_probes)

    p_dash = sub.add_parser("dashboard", help="generate static HTML readout dashboard")
    p_dash.add_argument("--out", default=None, help="output path (default: docs/dash/index.html)")
    p_dash.set_defaults(func=_cmd_dashboard)

    p_promote = sub.add_parser("promote", help="promote a probe from SHADOW to ACTIVE")
    p_promote.add_argument("--probe", required=True, help="probe ID to promote")
    p_promote.add_argument("--from", default="shadow", dest="from_level")
    p_promote.add_argument("--to", default="active", dest="to_level")
    p_promote.add_argument("--evidence", nargs="*", default=None, help="run_ids as evidence (optional, auto-discovers if omitted)")
    p_promote.add_argument("--metric", default="accuracy", help="metric for gate evaluation")
    p_promote.add_argument("--mode", default="relative", choices=["relative", "absolute"], help="relative (probe_on vs baseline) or absolute (vs threshold)")
    p_promote.add_argument("--threshold", type=float, default=0.8, help="for absolute mode: minimum acceptable value")
    p_promote.add_argument("--n-params", type=int, default=0, help="trainable parameters (0 for read-only)")
    p_promote.add_argument("--dry-run", action="store_true", help="evaluate gate without recording promotion")
    p_promote.set_defaults(func=_cmd_promote)

    p_demote = sub.add_parser("rollback-promotion", help="demote a probe back to SHADOW")
    p_demote.add_argument("--probe", required=True, help="probe ID to demote")
    p_demote.add_argument("--reason", default="manual rollback", help="reason for demotion")
    p_demote.set_defaults(func=_cmd_demote)

    p_metrics = sub.add_parser("metrics-server", help="start Prometheus metrics exporter")
    p_metrics.add_argument("--port", type=int, default=9090, help="HTTP port (default: 9090)")
    p_metrics.add_argument("--host", default="0.0.0.0", help="bind address")
    p_metrics.set_defaults(func=_cmd_metrics_server)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
