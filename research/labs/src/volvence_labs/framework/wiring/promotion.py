"""Promotion mechanism: SHADOW → ACTIVE level transitions.

A promotion is an immutable record in CAS that documents:
- Which probe was promoted
- From which level to which level
- The evidence (run_ids) that justified the promotion
- The gate decision that approved it
- Timestamp and operator

Promotions can be rolled back (creating a new "demotion" record), but the
original promotion record is never deleted (R15 immutability).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from ..gate import GateAggregator, GateDecision, PromotionDecision
from ..snapshot import CASStore, RunLog, default_paths
from ..snapshot.cas import canonical_dumps, sha256_bytes
from ..wiring import WiringLevel


@dataclass(frozen=True)
class PromotionRecord:
    """Immutable record of a level promotion."""
    promotion_id: str
    probe_id: str
    from_level: str
    to_level: str
    gate_decision: str  # "approve"
    evidence_run_ids: list[str]
    gate_summary: dict[str, Any]
    created_at: float
    sha: str  # content-addressed sha of this record

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DemotionRecord:
    """Immutable record of a level demotion (rollback)."""
    demotion_id: str
    probe_id: str
    from_level: str
    to_level: str
    original_promotion_sha: str
    reason: str
    created_at: float
    sha: str


def _generate_promotion_id(probe_id: str, to_level: str) -> str:
    """Generate a unique promotion ID."""
    ts = time.strftime("%Y%m%dT%H%M%S")
    import hashlib
    h = hashlib.sha256(f"{probe_id}:{to_level}:{time.time()}".encode()).hexdigest()[:8]
    return f"promo_{ts}_{probe_id}_{to_level}_{h}"


class PromotionManager:
    """Manages probe level promotions and demotions.

    Usage:
        mgr = PromotionManager()
        record = mgr.promote("refusal-direction-v1", decision, run_ids)
        mgr.demote("refusal-direction-v1", record.sha, reason="rollback drill")
    """

    def __init__(self, root: Optional[str] = None):
        self._paths = default_paths(root)
        self._store = CASStore(self._paths)
        self._log = RunLog(self._paths, self._store)
        self._promotions_dir = self._paths.root / ".labs" / "promotions"
        self._promotions_dir.mkdir(parents=True, exist_ok=True)

    def promote(
        self,
        probe_id: str,
        decision: PromotionDecision,
        evidence_run_ids: Optional[list[str]] = None,
    ) -> PromotionRecord:
        """Record a promotion. Requires decision.decision == APPROVE.

        Raises ValueError if decision is not APPROVE.
        """
        if decision.decision != GateDecision.APPROVE:
            raise ValueError(
                f"Cannot promote: gate decision is {decision.decision.value}, "
                f"reason: {decision.reason}"
            )

        run_ids = evidence_run_ids or decision.evidence_run_ids
        promotion_id = _generate_promotion_id(probe_id, decision.to_level)

        record_data = {
            "promotion_id": promotion_id,
            "probe_id": probe_id,
            "from_level": decision.from_level,
            "to_level": decision.to_level,
            "gate_decision": decision.decision.value,
            "evidence_run_ids": run_ids,
            "gate_summary": {
                "capacity": asdict(decision.capacity) if decision.capacity else None,
                "margin": asdict(decision.margin) if decision.margin else None,
                "sgm": decision.sgm_summary,
                "hoeffding": decision.hoeffding_summary,
                "reason": decision.reason,
            },
            "created_at": time.time(),
        }

        # Content-address the record
        record_bytes = canonical_dumps(record_data)
        sha = sha256_bytes(record_bytes)
        record_data["sha"] = sha

        # Store in CAS
        self._store.put_bytes(record_bytes, kind="promotion", meta={
            "probe_id": probe_id,
            "promotion_id": promotion_id,
        })

        # Also write to promotions directory for easy lookup
        record_path = self._promotions_dir / f"{promotion_id}.json"
        record_path.write_text(
            json.dumps(record_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        record = PromotionRecord(
            promotion_id=promotion_id,
            probe_id=probe_id,
            from_level=decision.from_level,
            to_level=decision.to_level,
            gate_decision=decision.decision.value,
            evidence_run_ids=run_ids,
            gate_summary=record_data["gate_summary"],
            created_at=record_data["created_at"],
            sha=sha,
        )

        return record

    def demote(
        self,
        probe_id: str,
        original_promotion_sha: str,
        *,
        reason: str = "manual rollback",
        to_level: str = "shadow",
    ) -> DemotionRecord:
        """Record a demotion (rollback). The original promotion record is preserved."""
        demotion_id = _generate_promotion_id(probe_id, f"demote_{to_level}")

        record_data = {
            "demotion_id": demotion_id,
            "probe_id": probe_id,
            "from_level": "active",
            "to_level": to_level,
            "original_promotion_sha": original_promotion_sha,
            "reason": reason,
            "created_at": time.time(),
        }

        record_bytes = canonical_dumps(record_data)
        sha = sha256_bytes(record_bytes)
        record_data["sha"] = sha

        self._store.put_bytes(record_bytes, kind="demotion", meta={
            "probe_id": probe_id,
            "demotion_id": demotion_id,
        })

        record_path = self._promotions_dir / f"{demotion_id}.json"
        record_path.write_text(
            json.dumps(record_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return DemotionRecord(
            demotion_id=demotion_id,
            probe_id=probe_id,
            from_level="active",
            to_level=to_level,
            original_promotion_sha=original_promotion_sha,
            reason=reason,
            created_at=record_data["created_at"],
            sha=sha,
        )

    def list_promotions(self, probe_id: Optional[str] = None) -> list[PromotionRecord]:
        """List all promotions, optionally filtered by probe_id."""
        records = []
        for path in sorted(self._promotions_dir.glob("promo_*.json")):
            data = json.loads(path.read_text("utf-8"))
            # Skip demotion records
            if "demotion_id" in data:
                continue
            if probe_id and data.get("probe_id") != probe_id:
                continue
            records.append(PromotionRecord(**{
                k: data[k] for k in PromotionRecord.__dataclass_fields__
            }))
        return records

    def get_current_level(self, probe_id: str) -> str:
        """Get the current effective level for a probe.

        Looks at the latest promotion/demotion record.
        """
        all_records = sorted(
            self._promotions_dir.glob(f"*_{probe_id}_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not all_records:
            return "shadow"  # default

        latest = json.loads(all_records[0].read_text("utf-8"))
        return latest.get("to_level", "shadow")

    def close(self) -> None:
        self._log.close()
        self._store.close()
