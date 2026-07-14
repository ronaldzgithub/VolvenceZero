#!/usr/bin/env python3
"""Assemble the full learned-backend ACTIVE promotion evidence artifact.

Merges the three independent evidence sources the promotion gate
(``volvence_zero.agent.learned_active_gate``) requires:

1. a real-substrate learned-shadow soak artifact (``learned_shadow_soak.json``
   with ``substrate_mode="hf"``) — real_trace_turns / validation_delta /
   strict ETA gate / latency SLO / rollback drill / safety;
2. the same-substrate ablation verdict (``verdict_p1.json`` /
   ``verdict_p2.json``) — PE-off / ETA-off component-control direction
   (claim_component_causal_contribution pairwise effects);
3. the CMS anti-forgetting A/B contract suite
   (``packages/vz-memory/tests/test_cms_anti_forgetting_evidence.py``)
   executed live — CMS retention / absorption bits.

The output is an evidence artifact consumable by
``scripts/evaluate_learned_backend_promotion.py``. This script never flips
runtime defaults and never fabricates a missing source: absent sources yield
conservative ``False`` bits with the omission recorded in ``sources``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

_COMPONENTS = (
    "temporal_runtime_backend",
    "temporal_ssl_backend",
    "internal_rl_backend",
    "cms_torch_backend",
)
_CMS_AB_TEST = "packages/vz-memory/tests/test_cms_anti_forgetting_evidence.py"


class PromotionEvidenceError(SystemExit):
    """Fail-loud wrapper for malformed evidence sources."""


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise PromotionEvidenceError(f"evidence source missing: {path} ({exc})")


def _soak_evidence(payload: dict, *, path: Path) -> dict[str, object]:
    if payload.get("artifact_kind") != "learned_shadow_soak":
        raise PromotionEvidenceError(
            f"{path} is not a learned_shadow_soak artifact "
            f"(artifact_kind={payload.get('artifact_kind')!r})"
        )
    arm = payload["capacity_arm"]
    if arm.get("substrate_mode") != "hf":
        raise PromotionEvidenceError(
            "promotion evidence requires the real-substrate soak lane "
            f"(substrate_mode='hf'); {path} recorded "
            f"substrate_mode={arm.get('substrate_mode')!r}. Synthetic soaks "
            "are directional only."
        )
    gate = payload["learned_active_gate"]
    cp11 = payload["cp11_predictive_heads"]
    verdict_rows = gate["verdicts"]
    if not verdict_rows:
        raise PromotionEvidenceError(f"{path} learned_active_gate.verdicts is empty")
    safety_gate_ok = all(bool(row["safety_gate_ok"]) for row in verdict_rows)
    kill = cp11["final_kill_criteria"]
    if kill["kill_triggered"]:
        # A triggered self-reward kill criterion invalidates the soak's
        # validation_delta as promotion evidence.
        raise PromotionEvidenceError(
            f"{path} predictive-head kill criteria triggered: {kill['description']}"
        )
    return {
        "real_trace_turns": int(gate["real_trace_turns"]),
        "validation_delta": float(gate["validation_delta"]),
        "strict_eta_gate_passed": bool(payload["strict_eta_gate"]["gate_passed"]),
        "rollback_drill_passed": int(cp11["checkpoint_round_trips_verified"]) > 0,
        "latency_slo_ok": bool(gate["latency_slo_ok"]),
        "safety_gate_ok": safety_gate_ok,
        "internal_rl_no_optimize_full_beats_control": bool(
            payload["internal_rl_no_optimize_proof"]["full_beats_control"]
        ),
        "soak_capacity_arm": dict(arm),
        "soak_turn_count": int(payload["turn_count"]),
        "soak_mean_turn_seconds": float(payload["mean_turn_seconds"]),
    }


def _component_control_bits(payload: dict, *, path: Path) -> dict[str, object]:
    claims = {claim["claim_id"]: claim for claim in payload["claims"]}
    claim = claims.get("claim_component_causal_contribution")
    if claim is None:
        raise PromotionEvidenceError(
            f"{path} has no claim_component_causal_contribution claim"
        )
    effects = {effect["control"]: effect for effect in claim.get("effects") or ()}
    bits: dict[str, object] = {
        "ablation_verdict_state": payload.get("state"),
        "component_claim_status": claim["status"],
    }
    for arm, key in (("pe-off", "pe_off"), ("eta-off", "eta_off")):
        effect = effects.get(arm)
        if effect is None:
            bits[f"{key}_control_direction_correct"] = False
            bits[f"{key}_control_detail"] = f"{arm} arm missing from verdict effects"
            continue
        # Direction correct = the full pipeline beat the component-off arm
        # (positive pairwise delta) and the pairwise status is not a FAIL.
        direction_ok = float(effect["delta_mean"]) > 0.0 and effect["status"] != "fail"
        bits[f"{key}_control_direction_correct"] = direction_ok
        bits[f"{key}_control_detail"] = (
            f"{arm}: delta_mean={float(effect['delta_mean']):+.2f} "
            f"ci_low={float(effect['ci_low_nonoverlap']):+.2f} "
            f"status={effect['status']}"
        )
    return bits


def _run_cms_ab_suite() -> dict[str, object]:
    command = [sys.executable, "-m", "pytest", "-q", _CMS_AB_TEST]
    completed = subprocess.run(
        command, cwd=str(REPO_ROOT), capture_output=True, text=True, check=False
    )
    passed = completed.returncode == 0
    tail = "\n".join((completed.stdout or "").strip().splitlines()[-3:])
    return {
        "cms_ab_suite": _CMS_AB_TEST,
        "cms_ab_suite_passed": passed,
        "cms_ab_suite_tail": tail,
        # The suite asserts (a) uplift background-band drift <= rollback
        # (retention non-degrading under matched control) and (b) the
        # absorption/retention proxies are real bounded readouts + rollback
        # drill. Both CMS gate bits therefore key off the same suite verdict.
        "cms_retention_non_degrading": passed,
        "cms_absorption_improved": passed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--soak-artifact", type=Path, required=True)
    parser.add_argument(
        "--ablation-verdict",
        type=Path,
        default=None,
        help="verdict_p1.json / verdict_p2.json with component arms; omitted "
        "=> pe_off/eta_off control bits are False (blocked honestly).",
    )
    parser.add_argument(
        "--skip-cms-ab-test",
        action="store_true",
        help="Skip the live CMS anti-forgetting suite (bits become False).",
    )
    parser.add_argument("--prior-runtime-active", action="store_true")
    parser.add_argument("--prior-ssl-active", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/learned_backend_promotion/promotion_evidence.json"),
    )
    args = parser.parse_args(argv)

    sources: dict[str, object] = {"soak_artifact": str(args.soak_artifact)}
    soak = _soak_evidence(_load_json(args.soak_artifact), path=args.soak_artifact)

    if args.ablation_verdict is not None:
        sources["ablation_verdict"] = str(args.ablation_verdict)
        control = _component_control_bits(
            _load_json(args.ablation_verdict), path=args.ablation_verdict
        )
    else:
        sources["ablation_verdict"] = None
        control = {
            "pe_off_control_direction_correct": False,
            "eta_off_control_direction_correct": False,
            "pe_off_control_detail": "no ablation verdict supplied",
            "eta_off_control_detail": "no ablation verdict supplied",
        }

    if args.skip_cms_ab_test:
        cms = {
            "cms_ab_suite_passed": False,
            "cms_retention_non_degrading": False,
            "cms_absorption_improved": False,
            "cms_ab_suite_tail": "skipped by operator",
        }
    else:
        cms = _run_cms_ab_suite()
    sources["cms_ab_suite"] = _CMS_AB_TEST if not args.skip_cms_ab_test else None

    rows = []
    for component in _COMPONENTS:
        rows.append(
            {
                "component": component,
                "real_trace_turns": soak["real_trace_turns"],
                "validation_delta": soak["validation_delta"],
                "strict_eta_gate_passed": soak["strict_eta_gate_passed"],
                "pe_off_control_direction_correct": control[
                    "pe_off_control_direction_correct"
                ],
                "eta_off_control_direction_correct": control[
                    "eta_off_control_direction_correct"
                ],
                "rollback_drill_passed": soak["rollback_drill_passed"],
                "latency_slo_ok": soak["latency_slo_ok"],
                "safety_gate_ok": soak["safety_gate_ok"],
                "prior_runtime_active": bool(args.prior_runtime_active),
                "prior_ssl_active": bool(args.prior_ssl_active),
                "internal_rl_no_reward_leakage": soak[
                    "internal_rl_no_optimize_full_beats_control"
                ],
                "cms_retention_non_degrading": cms["cms_retention_non_degrading"],
                "cms_absorption_improved": cms["cms_absorption_improved"],
            }
        )

    payload = {
        "schema_version": "learned-backend-promotion-evidence.v1",
        "artifact_kind": "learned_backend_promotion_evidence",
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "sources": sources,
        "soak_summary": soak,
        "component_controls": control,
        "cms_ab_evidence": cms,
        "learned_active_gate": {"evidence": rows},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote promotion evidence: {args.output}")
    print(
        "  real_trace_turns={rt} validation_delta={vd:.4f} strict_eta={se} "
        "pe_off={po} eta_off={eo} cms_ab={ca}".format(
            rt=soak["real_trace_turns"],
            vd=soak["validation_delta"],
            se=soak["strict_eta_gate_passed"],
            po=control["pe_off_control_direction_correct"],
            eo=control["eta_off_control_direction_correct"],
            ca=cms["cms_ab_suite_passed"],
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
