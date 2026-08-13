"""Packet 3：S3-E 编程域复刻 runner（六门 × 5 seed）。

子命令：
- ``freeze-prereg``：冻结 prereg JSON（六门阈值、seed 表、几何、语料
  纪律）。已存在则拒绝覆盖。
- ``run``：校验 prereg 一致性 → 从通过 episode 轨迹构建 junction 行
  → 重 fit reader/executor（几何绑定 Coder-1.5B：层 13、宽 1536）→
  REINFORCE 六门判词 → report + artifact manifest。输出已存在则拒绝。

前置：Packet 3 前置 b（margin 审计）三门全过后才允许 ``run``
（--skip-margin-check 仅供机制冒烟，正式判词禁用）。
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
for _pkg in (
    "vz-contracts",
    "vz-substrate",
    "vz-runtime",
    "vz-cognition",
    "vz-memory",
    "vz-temporal",
    "lifeform-domain-coding",
    "lifeform-evolution",
):
    _src = _REPO_ROOT / "packages" / _pkg / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from companion_test_plan_common import (  # noqa: E402
    exclusive_mps_lock,
    require_mps,
)

from lifeform_evolution.coding_lab_s3e import (  # noqa: E402
    ACTION_SURFACES,
    CODING_WHEN_TO_STEER_SCHEMA_VERSION,
    build_coding_junction_rows,
    rows_manifest,
    run_coding_when_to_steer_rl,
)

PREREG_SCHEMA_VERSION = "coding-lab-when-to-steer-prereg.v1"
DEFAULT_PREREG_PATH = "artifacts/coding_lab/coding_lab_packet3_prereg.json"
MPS_PLAN_ID = "coding-lab-when-to-steer-rl-mps.v1"
DEFAULT_TRAJECTORY_GLOBS = (
    "artifacts/coding_lab/*/chains/chain-*/trajectories/episode-*.jsonl",
    "artifacts/coding_lab/*/brain/chain-*/trajectories/episode-*.jsonl",
    "artifacts/coding_lab/*/steelman/chain-*/trajectories/episode-*.jsonl",
    "artifacts/coding_lab/*/stateless/chain-*/trajectories/episode-*.jsonl",
)

SOURCE_FILES = (
    "packages/lifeform-evolution/src/lifeform_evolution/coding_lab_s3e.py",
    "packages/lifeform-domain-coding/src/lifeform_domain_coding/lab/junctions.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_when_to_steer_rl.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_read_steer_prereq.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_conditional_steering_screen.py",
    "scripts/run_coding_lab_packet3_s3e.py",
)


def _sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cap_cases(rows: tuple, max_cases: int) -> tuple:
    """Keep the first ``max_cases`` whole routes (deterministic order)."""

    kept: list = []
    seen: dict[str, bool] = {}
    for row in rows:
        if row.case_id not in seen and len(seen) >= max_cases:
            continue
        seen[row.case_id] = True
        kept.append(row)
    return tuple(kept)


def _default_prereg() -> dict:
    return {
        "schema_version": PREREG_SCHEMA_VERSION,
        "claim_scope": "coding-junction-internal-rl-when-to-steer",
        "decision_rules": {
            "convergence": {"min_improvement_nll": 0.2},
            "gain_vs_noop": {
                "min_nll": 0.3,
                "require_bootstrap_lower_positive": True,
            },
            "gain_vs_always_on_belief": {"min_nll": 0.2},
            "gain_vs_random_gate": {"min_nll": 0.2},
            "gate_selectivity": {"min": 0.3},
            "structural_integrity": {
                "free_bias_present": False,
                "zero_code_strict_noop": True,
                "substrate_trainable_parameter_count": 0,
                "reader_frozen_during_rl": True,
                "executor_frozen_during_rl": True,
            },
        },
        "seed_schedule": [0, 1, 2, 3, 4],
        "policy": {
            "restarts": 4,
            "learning_rate": 0.1,
            "batch_cases": 8,
            "entropy_coef": 0.1,
            "init_noop_bias": 0.0,
            "max_online_episodes": 1200,
            "eval_every": 80,
            "baseline_beta": 0.9,
        },
        "geometry": {
            "model_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "injection_layer_index": 13,
            "residual_width": 1536,
            "steering_rank": 8,
            "model_dtype": "float32",
        },
        "corpus": {
            "trajectory_globs": list(DEFAULT_TRAJECTORY_GLOBS),
            "heldout_fraction": 0.3,
            "min_train_rows": 60,
            "min_heldout_rows": 20,
            "expert_source": "passing-episodes-only",
        },
        "credit_source": "route mean expert-action NLL (episode terminal)",
        "prohibited": [
            "evaluation/judge scores as reward",
            "token-space RL",
            "substrate weight updates",
            "reader/executor updates during RL",
        ],
    }


def _assert_prereg_consistency(prereg: dict, thresholds) -> None:
    rules = prereg["decision_rules"]
    pairs = (
        (thresholds.min_convergence_improvement_nll,
         rules["convergence"]["min_improvement_nll"]),
        (thresholds.min_gain_vs_noop_nll, rules["gain_vs_noop"]["min_nll"]),
        (thresholds.min_gain_vs_always_on_nll,
         rules["gain_vs_always_on_belief"]["min_nll"]),
        (thresholds.min_gain_vs_random_gate_nll,
         rules["gain_vs_random_gate"]["min_nll"]),
        (thresholds.min_gate_selectivity, rules["gate_selectivity"]["min"]),
    )
    for module_value, prereg_value in pairs:
        if abs(float(module_value) - float(prereg_value)) > 1e-12:
            raise ValueError(
                f"prereg drift: module threshold {module_value} != "
                f"prereg {prereg_value}"
            )
    if prereg["schema_version"] != PREREG_SCHEMA_VERSION:
        raise ValueError(f"unexpected prereg schema: {prereg['schema_version']}")
    if not prereg["seed_schedule"]:
        raise ValueError("prereg seed_schedule must be non-empty")


def _freeze_prereg(args: argparse.Namespace) -> int:
    path = _REPO_ROOT / args.prereg_path
    if path.exists():
        raise FileExistsError(f"prereg already frozen: {path!s}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _default_prereg()
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"prereg frozen: {path!s}")
    print(f"sha256: {_sha256_file(path)}")
    return 0


def _require_margin_pass(margin_run_id: str) -> dict:
    report_path = (
        _REPO_ROOT / "artifacts" / "coding_lab" / margin_run_id / "report.json"
    )
    if not report_path.is_file():
        raise FileNotFoundError(
            f"margin audit report missing: {report_path!s} — run "
            "run_coding_lab_packet3_margin.py first (前置 b 不过门不开 RL)"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report["verdicts"]["overall_pass"]:
        raise RuntimeError(
            "margin audit did not pass; Packet 3 RL stays closed "
            f"(verdicts={report['verdicts']})"
        )
    return {"path": str(report_path), "sha256": _sha256_file(report_path)}


def _run(args: argparse.Namespace) -> int:
    from volvence_zero.agent.eta_when_to_steer_rl import (  # noqa: PLC0415
        WhenToSteerThresholds,
    )

    prereg_path = _REPO_ROOT / args.prereg_path
    if not prereg_path.is_file():
        raise FileNotFoundError(
            f"prereg not frozen yet: {prereg_path!s} (run freeze-prereg first)"
        )
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    thresholds = WhenToSteerThresholds()
    _assert_prereg_consistency(prereg, thresholds)

    out_dir = _REPO_ROOT / "artifacts" / "coding_lab" / args.run_id
    for name in ("report.json", "report.md", "artifact_manifest.json"):
        if (out_dir / name).exists():
            raise FileExistsError(f"output already exists: {(out_dir / name)!s}")
    out_dir.mkdir(parents=True, exist_ok=True)

    margin_attestation: dict | None = None
    if args.skip_margin_check:
        if not args.smoke:
            raise ValueError("--skip-margin-check is only allowed with --smoke")
    else:
        margin_attestation = _require_margin_pass(args.margin_run_id)

    paths: list[pathlib.Path] = []
    for pattern in prereg["corpus"]["trajectory_globs"]:
        paths.extend(sorted(_REPO_ROOT.glob(pattern)))
    trajectories = tuple(pathlib.Path(p) for p in sorted({p.resolve() for p in paths}))
    train_rows, heldout_rows = build_coding_junction_rows(
        trajectories, heldout_fraction=prereg["corpus"]["heldout_fraction"]
    )
    if args.smoke:
        # Mechanism smoke: cap the corpus by whole cases (routes stay
        # intact) so capture cost is minutes, not hours.
        train_rows = _cap_cases(train_rows, args.smoke_train_cases)
        heldout_rows = _cap_cases(heldout_rows, args.smoke_heldout_cases)
    manifest = rows_manifest(train_rows, heldout_rows)
    manifest["source_trajectories"] = len(trajectories)
    print(json.dumps(manifest, ensure_ascii=False))
    if not args.smoke and (
        manifest["train_rows"] < prereg["corpus"]["min_train_rows"]
        or manifest["heldout_rows"] < prereg["corpus"]["min_heldout_rows"]
    ):
        raise RuntimeError(
            f"corpus below prereg minimum: {manifest} < "
            f"{prereg['corpus']['min_train_rows']}/"
            f"{prereg['corpus']['min_heldout_rows']}"
        )

    corpus_blob = json.dumps(
        [dataclasses.asdict(row) for row in (*train_rows, *heldout_rows)],
        sort_keys=True,
    ).encode("utf-8")
    corpus_fingerprint = int.from_bytes(
        hashlib.sha256(corpus_blob).digest()[:4], "big"
    )

    geometry = prereg["geometry"]
    policy = prereg["policy"]
    with contextlib.ExitStack() as stack:
        mps_attestation = None
        if args.device.startswith("mps"):
            stack.enter_context(
                exclusive_mps_lock(
                    pathlib.Path(args.mps_lock), plan_id=MPS_PLAN_ID
                )
            )
            mps_attestation = dataclasses.asdict(require_mps())

        from volvence_zero.substrate.residual_backend import (  # noqa: PLC0415
            TransformersOpenWeightResidualRuntime,
        )
        from volvence_zero.substrate.steered_action_scoring import (  # noqa: PLC0415
            SteeredActionOption,
        )

        runtime = TransformersOpenWeightResidualRuntime(
            model_id=geometry["model_id"],
            device=args.device,
            max_length=args.max_length,
            fail_on_truncation=True,
            activation_width=geometry["residual_width"],
            # Single-layer capture at the prereg injection layer (ETA
            # precedent): `_capture_one` requires exactly one full-width
            # residual; "middle" selection returns a 3-layer window.
            layer_indices=(geometry["injection_layer_index"],),
            allow_live_substrate_mutation=False,
            allow_offline_substrate_training=False,
            model_dtype=geometry["model_dtype"],
        )
        scorer = runtime.build_steered_action_scorer(
            action_options=tuple(
                SteeredActionOption(
                    action_id=f"move:{action}", surface_text=surface
                )
                for action, surface in ACTION_SURFACES.items()
            ),
            max_length=args.max_length,
        )

        report = run_coding_when_to_steer_rl(
            train_rows=train_rows,
            heldout_rows=heldout_rows,
            runtime=runtime,
            scorer=scorer,
            model_source=geometry["model_id"],
            device=args.device,
            corpus_fingerprint=corpus_fingerprint,
            injection_layer_index=geometry["injection_layer_index"],
            residual_width=geometry["residual_width"],
            steering_rank=geometry["steering_rank"],
            policy_learning_rate=policy["learning_rate"],
            policy_batch_cases=policy["batch_cases"],
            entropy_coef=policy["entropy_coef"],
            init_noop_bias=policy["init_noop_bias"],
            policy_restarts=policy["restarts"],
            max_online_episodes=(
                args.smoke_episodes if args.smoke else policy["max_online_episodes"]
            ),
            eval_every=policy["eval_every"],
            baseline_beta=policy["baseline_beta"],
            seed_schedule=(
                tuple(prereg["seed_schedule"][:1])
                if args.smoke
                else tuple(prereg["seed_schedule"])
            ),
            thresholds=thresholds,
            progress=print,
        )

    payload = dataclasses.asdict(report)
    payload["corpus_manifest"] = manifest
    payload["smoke"] = args.smoke
    (out_dir / "report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    admission = report.admission
    lines = [
        "# Packet 3：S3-E 编程域复刻判词",
        "",
        f"- smoke: {args.smoke}",
        f"- admitted: {admission.admitted}",
        f"- failed_conditions: {list(admission.failed_conditions)}",
        f"- rows: train={report.train_row_count} heldout={report.heldout_row_count}",
        f"- post_switch_fraction: {report.post_switch_fraction:.3f}",
        "",
        "```json",
        json.dumps(dataclasses.asdict(report.aggregate), indent=2, default=str),
        "```",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest_payload = {
        "schema_version": "coding-lab-when-to-steer-manifest.v1",
        "report_schema_version": CODING_WHEN_TO_STEER_SCHEMA_VERSION,
        "run_id": args.run_id,
        "smoke": args.smoke,
        "prereg_path": str(prereg_path.relative_to(_REPO_ROOT)),
        "prereg_sha256": _sha256_file(prereg_path),
        "margin_attestation": margin_attestation,
        "corpus_fingerprint": corpus_fingerprint,
        "source_files": {
            path: _sha256_file(_REPO_ROOT / path) for path in SOURCE_FILES
        },
        "result_files": {
            "report.json": _sha256_file(out_dir / "report.json"),
            "report.md": _sha256_file(out_dir / "report.md"),
        },
        "mps_attestation": mps_attestation,
        "admitted": admission.admitted,
        "production_promotion_authorized": False,
    }
    (out_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"admitted={admission.admitted}")
    print(f"failed_conditions={list(admission.failed_conditions)}")
    print(f"report: {out_dir / 'report.json'}")
    return 0 if admission.admitted else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    freeze = sub.add_parser("freeze-prereg")
    freeze.add_argument("--prereg-path", default=DEFAULT_PREREG_PATH)

    run = sub.add_parser("run")
    run.add_argument("--prereg-path", default=DEFAULT_PREREG_PATH)
    run.add_argument("--run-id", required=True)
    run.add_argument("--device", default="cpu")
    run.add_argument("--max-length", type=int, default=256)
    run.add_argument("--margin-run-id", default="coding_lab_packet3_margin")
    run.add_argument(
        "--mps-lock", default="artifacts/.companion-evidence-mps.lock"
    )
    run.add_argument(
        "--smoke",
        action="store_true",
        help="机制冒烟：单 seed、短训练、放宽语料量门；不产生正式判词。",
    )
    run.add_argument("--smoke-episodes", type=int, default=160)
    run.add_argument("--smoke-train-cases", type=int, default=8)
    run.add_argument("--smoke-heldout-cases", type=int, default=4)
    run.add_argument(
        "--skip-margin-check",
        action="store_true",
        help="仅与 --smoke 联用。",
    )

    args = parser.parse_args()
    if args.command == "freeze-prereg":
        return _freeze_prereg(args)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
