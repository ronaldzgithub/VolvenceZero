"""Packet 4：coding-lab artifact 自改提案经 ModificationGate.OFFLINE。

前四包判词齐备后才有资格调用（谱系纪律：系统一旦改自己，被测对象
每轮都在变，前面所有 baseline 作废——所以自改放最后且受门）。

流程照 `steering_promotion_gate` 先例：候选 = 一次 **admitted** 的
Packet 3 S3-E 运行产出的 artifact bundle（reader/executor/gate 策略,
由其 `artifact_manifest.json` 内容寻址）；提案走 cognition 拥有的
`evaluate_gate_reasons`（OFFLINE 档要求 validation_delta ≥ 0.05 且
rollback 证据在场，fail-closed）；部署只改一个注册表指针文件，回滚
= 从 candidate-bound checkpoint 恢复指针并校验哈希（单字段回滚）。

三条安全读数（contract_integrity / rollback_resilience /
fallback_reliance）是只读结构证据，不是 PE、credit 或学习输入
（R12）。
"""

from __future__ import annotations

import json
import pathlib

from dataclasses import dataclass

from volvence_zero.credit.gate import (
    GateDecision,
    ModificationGate,
    ModificationProposal,
    evaluate_gate_reasons,
)
from volvence_zero.evaluation import EvaluationScore, EvaluationSnapshot

CODING_MODIFICATION_TARGET = "coding_lab.steering_artifact_bundle"
CODING_GATE_REVIEW_SCHEMA_VERSION = "coding-lab-modification-gate-review.v1"
GENESIS_HASH = "genesis"


@dataclass(frozen=True)
class CodingArtifactPointer:
    """Registry pointer: which artifact bundle is live for coding-lab."""

    run_id: str
    manifest_sha256: str
    report_sha256: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "run_id": self.run_id,
                "manifest_sha256": self.manifest_sha256,
                "report_sha256": self.report_sha256,
            },
            indent=2,
            sort_keys=True,
        )

    @staticmethod
    def from_json(text: str) -> "CodingArtifactPointer":
        payload = json.loads(text)
        return CodingArtifactPointer(
            run_id=payload["run_id"],
            manifest_sha256=payload["manifest_sha256"],
            report_sha256=payload["report_sha256"],
        )


@dataclass(frozen=True)
class CodingModificationGateReview:
    schema_version: str
    proposal_target: str
    desired_gate: ModificationGate
    old_value_hash: str
    new_value_hash: str
    validation_delta: float
    rollback_evidence: str
    contract_integrity: float
    rollback_resilience: float
    fallback_reliance: float
    decision: GateDecision
    blocking_reasons: tuple[str, ...]
    description: str


def verify_pointer_round_trip(pointer: CodingArtifactPointer) -> bool:
    """Checkpoint JSON round-trip: serialize → parse → equality."""

    return CodingArtifactPointer.from_json(pointer.to_json()) == pointer


def build_coding_modification_gate_review(
    *,
    candidate_report: dict,
    candidate_manifest_sha256: str,
    candidate_report_sha256: str,
    incumbent: CodingArtifactPointer | None,
) -> CodingModificationGateReview:
    """OFFLINE gate review for one admitted Packet 3 candidate bundle.

    ``validation_delta`` is the candidate's held-out gain vs noop in
    nats (worst-seed CI 下界，来自 S3-E 判词聚合)——环境结算出身的
    结构证据，不是 evaluation readout。
    """

    admission = candidate_report["admission"]
    aggregate = candidate_report["aggregate"]
    admitted = bool(admission["admitted"])
    # Worst-seed bootstrap CI lower bound: the most conservative
    # held-out improvement any seed demonstrated.
    validation_delta = float(aggregate["gain_vs_noop_ci_lower_min"])

    contract_integrity = float(
        admitted
        and not candidate_report["free_bias_present"]
        and candidate_report["zero_code_strict_noop"]
        and candidate_report["substrate_trainable_parameter_count"] == 0
        and not candidate_report["reader_parameters_changed"]
        and not candidate_report["executor_parameters_changed"]
        and not candidate_report["production_wiring_changed"]
        and not candidate_report["feedback_to_learning"]
    )

    checkpoint_ok = incumbent is None or verify_pointer_round_trip(incumbent)
    rollback_resilience = float(checkpoint_ok)
    rollback_evidence = (
        "coding-lab:pointer-checkpoint-round-trip:"
        f"{candidate_manifest_sha256}"
        if checkpoint_ok
        else ""
    )
    fallback_reliance = float(not admitted)

    proposal = ModificationProposal(
        target=CODING_MODIFICATION_TARGET,
        desired_gate=ModificationGate.OFFLINE,
        old_value_hash=(
            incumbent.manifest_sha256 if incumbent is not None else GENESIS_HASH
        ),
        new_value_hash=candidate_manifest_sha256,
        justification=(
            "Point the coding-lab registry at the admitted, content-"
            "addressed Packet 3 artifact bundle; single-field rollback "
            "to the checkpointed incumbent pointer."
        ),
        is_reversible=True,
        validation_delta=validation_delta,
        capacity_cost=0.0,
        rollback_evidence=rollback_evidence,
    )
    evaluation_snapshot = EvaluationSnapshot(
        turn_scores=(
            EvaluationScore(
                family="safety",
                metric_name="contract_integrity",
                value=contract_integrity,
                confidence=1.0,
                evidence=candidate_report_sha256,
            ),
            EvaluationScore(
                family="safety",
                metric_name="rollback_resilience",
                value=rollback_resilience,
                confidence=1.0,
                evidence=rollback_evidence or "coding-lab:rollback-evidence-missing",
            ),
            EvaluationScore(
                family="safety",
                metric_name="fallback_reliance",
                value=fallback_reliance,
                confidence=1.0,
                evidence=candidate_manifest_sha256,
            ),
        ),
        session_scores=(),
        alerts=(),
        structured_alerts=(),
        description=(
            "Coding-lab read-only structural release evidence; not a PE, "
            "credit, or learning input."
        ),
    )
    reasons = evaluate_gate_reasons(
        proposal=proposal,
        evaluation_snapshot=evaluation_snapshot,
        audit_required=False,
    )
    return CodingModificationGateReview(
        schema_version=CODING_GATE_REVIEW_SCHEMA_VERSION,
        proposal_target=proposal.target,
        desired_gate=proposal.desired_gate,
        old_value_hash=proposal.old_value_hash,
        new_value_hash=proposal.new_value_hash,
        validation_delta=proposal.validation_delta,
        rollback_evidence=proposal.rollback_evidence,
        contract_integrity=contract_integrity,
        rollback_resilience=rollback_resilience,
        fallback_reliance=fallback_reliance,
        decision=(GateDecision.BLOCK if reasons else GateDecision.ALLOW),
        blocking_reasons=reasons,
        description=(
            "ModificationGate.OFFLINE review for the coding-lab artifact "
            "bundle pointer. Deployment is a single registry-file write; "
            "rollback restores the checkpointed pointer."
        ),
    )


def read_registry(path: pathlib.Path) -> CodingArtifactPointer | None:
    if not path.is_file():
        return None
    return CodingArtifactPointer.from_json(path.read_text(encoding="utf-8"))


def write_registry(path: pathlib.Path, pointer: CodingArtifactPointer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pointer.to_json() + "\n", encoding="utf-8")


def rollback_registry(
    path: pathlib.Path, checkpoint: CodingArtifactPointer | None
) -> None:
    """Candidate-bound rollback: restore the checkpointed incumbent.

    ``None`` checkpoint = genesis (no live artifact) → the registry file
    is removed, returning the system to its pre-candidate state.
    """

    if checkpoint is None:
        path.unlink(missing_ok=True)
        return
    write_registry(path, checkpoint)


__all__ = [
    "CODING_GATE_REVIEW_SCHEMA_VERSION",
    "CODING_MODIFICATION_TARGET",
    "GENESIS_HASH",
    "CodingArtifactPointer",
    "CodingModificationGateReview",
    "build_coding_modification_gate_review",
    "read_registry",
    "rollback_registry",
    "verify_pointer_round_trip",
    "write_registry",
]
