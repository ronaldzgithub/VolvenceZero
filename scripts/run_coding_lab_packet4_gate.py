"""Packet 4 runner：coding-lab artifact 提案过 ModificationGate.OFFLINE。

输入一个 **admitted** 的 Packet 3 运行目录（report.json +
artifact_manifest.json）。流程：

1. checkpoint 现任注册表指针（无现任 = genesis）；
2. cognition 拥有的 OFFLINE 门评审（fail-closed）；
3. ALLOW 时：写入候选指针 → **演示回滚**（恢复 checkpoint 并校验）→
   重新应用候选（终态 = 候选）；BLOCK 时注册表不动；
4. 落盘 review.json（评审 + 回滚演示证据）。

前序包判词不齐（候选未 admitted）时门会 BLOCK——不绕行。
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
for _pkg in (
    "vz-contracts",
    "vz-cognition",
    "vz-memory",
    "vz-temporal",
    "lifeform-evolution",
):
    _src = _REPO_ROOT / "packages" / _pkg / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from volvence_zero.credit.gate import GateDecision  # noqa: E402

from lifeform_evolution.coding_lab_packet4 import (  # noqa: E402
    CodingArtifactPointer,
    build_coding_modification_gate_review,
    read_registry,
    rollback_registry,
    write_registry,
)

DEFAULT_REGISTRY = "artifacts/coding_lab/active_artifact.json"


def _sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-run-id",
        required=True,
        help="Packet 3 运行目录名（artifacts/coding_lab/ 下）。",
    )
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--out-run-id",
        default=None,
        help="评审输出目录名；默认 <candidate>_gate_review。",
    )
    parser.add_argument(
        "--mechanism-probe",
        action="store_true",
        help="接受 smoke 候选并把 apply/rollback 演示打到 probe 注册表；"
        "不产生正式晋升。",
    )
    parser.add_argument(
        "--probe-registry",
        default="artifacts/coding_lab/probe_active_artifact.json",
    )
    args = parser.parse_args()

    candidate_dir = _REPO_ROOT / "artifacts" / "coding_lab" / args.candidate_run_id
    report_path = candidate_dir / "report.json"
    manifest_path = candidate_dir / "artifact_manifest.json"
    for path in (report_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"candidate artifact incomplete: {path!s}")
    candidate_report = json.loads(report_path.read_text(encoding="utf-8"))
    if candidate_report.get("smoke") and not args.mechanism_probe:
        raise ValueError(
            "smoke runs are mechanism probes, not promotion candidates "
            "(use --mechanism-probe to exercise the gate against a probe "
            "registry)"
        )
    if args.mechanism_probe:
        # Probe mode never touches the real registry: the whole
        # apply/rollback demonstration runs against a probe pointer file.
        args.registry = args.probe_registry
    manifest_sha = _sha256_file(manifest_path)
    report_sha = _sha256_file(report_path)

    registry_path = _REPO_ROOT / args.registry
    incumbent = read_registry(registry_path)

    review = build_coding_modification_gate_review(
        candidate_report=candidate_report,
        candidate_manifest_sha256=manifest_sha,
        candidate_report_sha256=report_sha,
        incumbent=incumbent,
    )

    rollback_demo: dict | None = None
    if review.decision is GateDecision.ALLOW:
        candidate_pointer = CodingArtifactPointer(
            run_id=args.candidate_run_id,
            manifest_sha256=manifest_sha,
            report_sha256=report_sha,
        )
        write_registry(registry_path, candidate_pointer)
        applied = read_registry(registry_path)
        assert applied == candidate_pointer
        # Candidate-bound rollback demonstration: restore incumbent,
        # verify byte-level equivalence of the restored state, re-apply.
        rollback_registry(registry_path, incumbent)
        restored = read_registry(registry_path)
        rollback_verified = restored == incumbent
        write_registry(registry_path, candidate_pointer)
        final = read_registry(registry_path)
        rollback_demo = {
            "rollback_verified": rollback_verified,
            "incumbent": (
                dataclasses.asdict(incumbent) if incumbent is not None else None
            ),
            "final_pointer": dataclasses.asdict(final),
        }
        if not rollback_verified:
            # Fail loudly and leave the registry rolled back.
            rollback_registry(registry_path, incumbent)
            raise RuntimeError("rollback demonstration failed; registry restored")

    out_dir = (
        _REPO_ROOT
        / "artifacts"
        / "coding_lab"
        / (args.out_run_id or f"{args.candidate_run_id}_gate_review")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "review": dataclasses.asdict(review),
        "mechanism_probe": args.mechanism_probe,
        "candidate_run_id": args.candidate_run_id,
        "candidate_manifest_sha256": manifest_sha,
        "candidate_report_sha256": report_sha,
        "registry": str(registry_path.relative_to(_REPO_ROOT)),
        "rollback_demonstration": rollback_demo,
    }
    (out_dir / "review.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"decision={review.decision.value}")
    print(f"blocking_reasons={list(review.blocking_reasons)}")
    print(f"review: {out_dir / 'review.json'}")
    return 0 if review.decision is GateDecision.ALLOW else 1


if __name__ == "__main__":
    raise SystemExit(main())
