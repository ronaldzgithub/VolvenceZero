#!/usr/bin/env python3
# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Same-substrate Companion Bench ablation -> thesis retain verdict.

Reads the per-track ``summary.json`` files emitted by ``run_real_submission.py``
(via ``score_reference_systems.py``) for the tracks of the same-substrate
ablation and produces the retain verdicts that debt #87 / the frozen claim
registry (docs/specs/human-world-model-ablation.md) require before the
human-world-model thesis first stage can be called "closed":

  1. claim_pipeline_gt_raw                  volvence > raw substrate
  2. claim_gt_standard_layers               volvence > ref-harness AND > camel
  3. claim_training_adds_value              volvence > volvence-cold
  4. claim_heldout_cohort_stable            volvence's score is tight + on enough arcs
  5. claim_component_causal_contribution    volvence > each component-off arm
                                            (pe-off / eta-off /
                                            active-learning-off / lora-adapter)

Each claim resolves to ``retain`` / ``weak`` / ``fail`` / ``insufficient_data``
and the whole run rolls up to one of the four #87 states:

  * ``kill-criteria-triggered``   any of claims 1-3 is ``fail`` (volvence not
                                  better than a control) -> thesis should shrink
                                  to a product-memory/companion claim.
  * ``wiring-ready``              flow works but inputs are missing to assess.
  * ``weak-positive``            claims 1-3 all positive (>0) but not all retain.
  * ``first-stage-retained``     claims 1-4 all retain.

``world-model-extension-ready`` is intentionally NOT emittable here: it requires
an independent physical/embodied benchmark, never the human-side ablation alone.

Guards (fail-loud, like compare_ablation.py for EQ-Bench):

* Every track's manifest must affirm all four CompanionBench attestation
  red lines, else no verdict is emitted.
* If ``--substrate-fingerprint``/``--fingerprint-file`` pairs are supplied they
  must all match (delegates to assert_same_substrate). Strongly recommended.

Significance uses the bootstrap CI already in each summary:
``ci_low(volvence - control) = volvence.ci95_lo - control.ci95_hi`` (a
conservative non-overlap lower bound). ``retain`` requires this > 0.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import pathlib
import sys
from typing import Any

# Track names (CLI keys).
TRACK_RAW = "raw"
TRACK_REF_HARNESS = "ref-harness"
TRACK_CAMEL = "camel"
TRACK_VOLVENCE_COLD = "volvence-cold"
TRACK_VOLVENCE = "volvence"

# Independent standard-layer arms (frozen claim registry claim 2,
# human-world-model-ablation.md: retain requires ALL THREE pairwise —
# memory-only, RAG, agent-framework). ``ref-harness`` (the combined
# 4-component wrapper) remains a supplementary control but can no longer
# stand in for the memory-only and RAG arms it used to collapse.
TRACK_MEMORY_ONLY = "memory-only"
TRACK_RAG = "rag"

STANDARD_LAYER_TRACKS: tuple[str, ...] = (
    TRACK_MEMORY_ONLY,
    TRACK_RAG,
    TRACK_CAMEL,
)

# Independent standard-layer arms (GAP-11 / frozen claim registry): the
# registry's claim_gt_standard_layers requires THREE pairwise controls —
# memory-only / RAG / agent-framework — not a single combined wrapper.
# ``ref-harness`` (all four components) remains accepted as an additional
# combined-wrapper control, but it cannot substitute for the independent
# memory-only and RAG arms.
TRACK_MEMORY_ONLY = "memory-only"
TRACK_RAG = "rag"

STANDARD_LAYER_TRACKS: tuple[str, ...] = (
    TRACK_MEMORY_ONLY,
    TRACK_RAG,
    TRACK_CAMEL,
)

# Component-causal arms (frozen claim registry, human-world-model-ablation.md
# claim 3 / verdict claim_component_causal_contribution): the full pipeline
# with exactly one component disabled, plus the "finetune-without-controller"
# LoRA control. Each must be same-substrate like every other track.
TRACK_PE_OFF = "pe-off"
TRACK_ETA_OFF = "eta-off"
TRACK_ACTIVE_LEARNING_OFF = "active-learning-off"
TRACK_LORA_ADAPTER = "lora-adapter"

COMPONENT_TRACKS: tuple[str, ...] = (
    TRACK_PE_OFF,
    TRACK_ETA_OFF,
    TRACK_ACTIVE_LEARNING_OFF,
    TRACK_LORA_ADAPTER,
)

_VALID_TRACKS: frozenset[str] = frozenset(
    {TRACK_RAW, TRACK_REF_HARNESS, TRACK_CAMEL, TRACK_VOLVENCE_COLD, TRACK_VOLVENCE}
    | set(STANDARD_LAYER_TRACKS)
    | set(COMPONENT_TRACKS)
)

_RED_LINE_KEYS: tuple[str, ...] = (
    "no_companionbench_derivative_in_training",
    "no_scenario_specific_prompt",
    "no_public_test_set_tuning",
    "cross_user_memory_isolation",
)

# Claim / verdict vocabularies.
RETAIN = "retain"
WEAK = "weak"
FAIL = "fail"
INSUFFICIENT = "insufficient_data"

STATE_KILL = "kill-criteria-triggered"
STATE_WIRING = "wiring-ready"
STATE_WEAK = "weak-positive"
STATE_FIRST_STAGE = "first-stage-retained"
STATE_WORLD_MODEL = "world-model-extension-ready"  # documented; never auto-emitted

# Stability proxy defaults for claim 4.
DEFAULT_STABILITY_REL_HALFWIDTH = 0.15
DEFAULT_MIN_ARCS_FOR_STABILITY = 120


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TrackResult:
    track: str
    system_name: str
    final_mean: float
    ci95_lo: float
    ci95_hi: float
    arc_count: int
    axis_means: dict[str, float]
    summary_path: str

    @property
    def ci_halfwidth(self) -> float:
        return (self.ci95_hi - self.ci95_lo) / 2.0

    @property
    def rel_ci_halfwidth(self) -> float:
        if self.final_mean <= 0:
            return float("inf")
        return self.ci_halfwidth / self.final_mean


def load_track(track: str, path: pathlib.Path) -> TrackResult:
    if track not in _VALID_TRACKS:
        raise ValueError(
            f"unknown track {track!r}; allowed: {sorted(_VALID_TRACKS)}"
        )
    if not path.exists():
        raise FileNotFoundError(f"summary not found for {track!r}: {path}")
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))

    manifest = data.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError(f"{path}: missing 'manifest' block")
    attestation = manifest.get("attestation")
    if not isinstance(attestation, dict):
        raise ValueError(f"{path}: missing manifest.attestation block")
    for key in _RED_LINE_KEYS:
        if attestation.get(key) is not True:
            raise ValueError(
                f"{path}: attestation['{key}'] is not True -- a CompanionBench "
                "red line is unaffirmed; verdict cannot be emitted."
            )

    aggregate = data.get("aggregate")
    if not isinstance(aggregate, dict):
        raise ValueError(f"{path}: missing 'aggregate' block")
    final_mean = _require_number(aggregate, "final_mean", path)
    ci = aggregate.get("final_ci95")
    if not isinstance(ci, list) or len(ci) != 2:
        raise ValueError(f"{path}: aggregate.final_ci95 must be a [lo, hi] list")
    arc_count = aggregate.get("arc_count")
    if not isinstance(arc_count, int):
        raise ValueError(f"{path}: aggregate.arc_count must be an int")
    axis_means_raw = aggregate.get("axis_means") or {}
    axis_means = {
        str(k): float(v)
        for k, v in axis_means_raw.items()
        if isinstance(v, (int, float))
    }
    return TrackResult(
        track=track,
        system_name=str(manifest.get("system_name", track)),
        final_mean=final_mean,
        ci95_lo=float(ci[0]),
        ci95_hi=float(ci[1]),
        arc_count=arc_count,
        axis_means=axis_means,
        summary_path=str(path),
    )


def _require_number(d: dict[str, Any], key: str, path: pathlib.Path) -> float:
    value = d.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{path}: '{key}' must be a number; got {type(value).__name__}")
    return float(value)


# ---------------------------------------------------------------------------
# Pairwise effect
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PairwiseEffect:
    control: str
    delta_mean: float
    ci_low_nonoverlap: float  # volvence.lo - control.hi
    status: str  # retain / weak / fail

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def pairwise(volvence: TrackResult, control: TrackResult) -> PairwiseEffect:
    delta = round(volvence.final_mean - control.final_mean, 4)
    ci_low = round(volvence.ci95_lo - control.ci95_hi, 4)
    if delta <= 0:
        status = FAIL
    elif ci_low > 0:
        status = RETAIN
    else:
        status = WEAK
    return PairwiseEffect(
        control=control.track,
        delta_mean=delta,
        ci_low_nonoverlap=ci_low,
        status=status,
    )


# ---------------------------------------------------------------------------
# Claims + verdict
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Claim:
    claim_id: str
    status: str  # retain / weak / fail / insufficient_data
    detail: str
    effects: tuple[PairwiseEffect, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "status": self.status,
            "detail": self.detail,
            "effects": [e.to_json() for e in self.effects],
        }


@dataclasses.dataclass(frozen=True)
class AblationVerdict:
    state: str
    claims: tuple[Claim, ...]
    tracks: dict[str, float | None]
    recommendations: tuple[str, ...]
    substrate_note: str
    timestamp_iso: str

    def to_json(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "claims": [c.to_json() for c in self.claims],
            "tracks": self.tracks,
            "recommendations": list(self.recommendations),
            "substrate_note": self.substrate_note,
            "timestamp_iso": self.timestamp_iso,
        }


def _combine_required(statuses: list[str]) -> str:
    """Combine sub-effect statuses for an AND-claim (e.g. > ref-harness AND > camel)."""

    if not statuses:
        return INSUFFICIENT
    if any(s == FAIL for s in statuses):
        return FAIL
    if all(s == RETAIN for s in statuses):
        return RETAIN
    return WEAK


def build_verdict(
    *,
    tracks: dict[str, TrackResult],
    substrate_note: str,
    stability_rel_halfwidth: float = DEFAULT_STABILITY_REL_HALFWIDTH,
    min_arcs_for_stability: int = DEFAULT_MIN_ARCS_FOR_STABILITY,
) -> AblationVerdict:
    volvence = tracks.get(TRACK_VOLVENCE)
    claims: list[Claim] = []

    # Claim 1: pipeline > raw.
    raw = tracks.get(TRACK_RAW)
    if volvence is None or raw is None:
        claims.append(Claim(
            "claim_pipeline_gt_raw", INSUFFICIENT,
            "need both 'volvence' and 'raw' tracks",
        ))
    else:
        eff = pairwise(volvence, raw)
        claims.append(Claim(
            "claim_pipeline_gt_raw", eff.status,
            f"volvence-raw delta={eff.delta_mean:+.2f}, ci_low={eff.ci_low_nonoverlap:+.2f}",
            (eff,),
        ))

    # Claim 2: > standard layers. Frozen registry requires ALL THREE
    # independent pairwise arms (memory-only / rag / camel) for retain;
    # ref-harness (combined wrapper) is a supplementary control only.
    # A missing registry arm caps the claim at 'weak' — the old behaviour
    # where ref-harness stood in for memory-only AND rag over-credited a
    # single collapsed arm (GAP-11).
    standard_results = {name: tracks.get(name) for name in STANDARD_LAYER_TRACKS}
    rh = tracks.get(TRACK_REF_HARNESS)
    supplied_standard = {
        name: t for name, t in standard_results.items() if t is not None
    }
    if volvence is None or (not supplied_standard and rh is None):
        claims.append(Claim(
            "claim_gt_standard_layers", INSUFFICIENT,
            "need 'volvence' and at least one standard-layer arm "
            f"({', '.join(STANDARD_LAYER_TRACKS)}; 'ref-harness' is "
            "supplementary only)",
        ))
    else:
        controls = list(supplied_standard.values())
        if rh is not None:
            controls.append(rh)
        effs = tuple(pairwise(volvence, c) for c in controls)
        status = _combine_required([e.status for e in effs])
        missing = [name for name, t in standard_results.items() if t is None]
        detail = "; ".join(
            f"{e.control}: delta={e.delta_mean:+.2f} ci_low={e.ci_low_nonoverlap:+.2f}"
            for e in effs
        )
        if missing:
            detail += (
                f" (missing registry arms: {', '.join(missing)} -> at most "
                "'weak'; frozen registry requires all three pairwise)"
            )
            if status == RETAIN:
                status = WEAK
        claims.append(Claim("claim_gt_standard_layers", status, detail, effs))

    # Claim 3: training adds value (> cold).
    cold = tracks.get(TRACK_VOLVENCE_COLD)
    if volvence is None or cold is None:
        claims.append(Claim(
            "claim_training_adds_value", INSUFFICIENT,
            "need both 'volvence' and 'volvence-cold' tracks",
        ))
    else:
        eff = pairwise(volvence, cold)
        claims.append(Claim(
            "claim_training_adds_value", eff.status,
            f"volvence-cold delta={eff.delta_mean:+.2f}, ci_low={eff.ci_low_nonoverlap:+.2f}",
            (eff,),
        ))

    # Claim 5 (registry claim 3): component causal contribution. Each supplied
    # component-off arm must be significantly WORSE than the full pipeline
    # (pairwise volvence > arm retains). No arms supplied -> insufficient_data
    # (the registered gap in human-world-model-ablation.md); a partial arm set
    # caps at 'weak' like claim 2 so a missing arm can never inflate the claim.
    component_results = {name: tracks.get(name) for name in COMPONENT_TRACKS}
    supplied = {name: t for name, t in component_results.items() if t is not None}
    if volvence is None or not supplied:
        claims.append(Claim(
            "claim_component_causal_contribution", INSUFFICIENT,
            "need 'volvence' and at least one component arm "
            f"({', '.join(COMPONENT_TRACKS)}); none supplied",
        ))
    else:
        effs = tuple(pairwise(volvence, t) for t in supplied.values())
        status = _combine_required([e.status for e in effs])
        missing = [name for name, t in component_results.items() if t is None]
        detail = "; ".join(
            f"{e.control}: delta={e.delta_mean:+.2f} ci_low={e.ci_low_nonoverlap:+.2f}"
            for e in effs
        )
        if missing:
            detail += f" (missing arms: {', '.join(missing)} -> at most 'weak')"
            if status == RETAIN:
                status = WEAK
        claims.append(Claim("claim_component_causal_contribution", status, detail, effs))

    # Claim 4: held-out / cohort stability proxy.
    if volvence is None:
        claims.append(Claim(
            "claim_heldout_cohort_stable", INSUFFICIENT, "need 'volvence' track",
        ))
    else:
        rel = volvence.rel_ci_halfwidth
        arcs = volvence.arc_count
        if arcs < min_arcs_for_stability:
            status = INSUFFICIENT
            detail = (
                f"arc_count={arcs} < {min_arcs_for_stability} "
                "(run held-out + multi-seed before claiming stability)"
            )
        elif rel <= stability_rel_halfwidth:
            status = RETAIN
            detail = f"rel_ci_halfwidth={rel:.3f} <= {stability_rel_halfwidth} on {arcs} arcs"
        elif rel <= 2 * stability_rel_halfwidth:
            status = WEAK
            detail = f"rel_ci_halfwidth={rel:.3f} <= {2 * stability_rel_halfwidth} on {arcs} arcs"
        else:
            status = FAIL
            detail = f"rel_ci_halfwidth={rel:.3f} too wide on {arcs} arcs"
        claims.append(Claim("claim_heldout_cohort_stable", status, detail))

    state, recs = _roll_up(claims)
    track_means = {name: (t.final_mean if t else None) for name, t in tracks.items()}
    return AblationVerdict(
        state=state,
        claims=tuple(claims),
        tracks=track_means,
        recommendations=tuple(recs),
        substrate_note=substrate_note,
        timestamp_iso=_dt.datetime.now(_dt.timezone.utc).isoformat(),
    )


def _roll_up(claims: list[Claim]) -> tuple[str, list[str]]:
    by_id = {c.claim_id: c for c in claims}
    core_ids = ("claim_pipeline_gt_raw", "claim_gt_standard_layers", "claim_training_adds_value")
    core = [by_id[i] for i in core_ids if i in by_id]
    core_status = [c.status for c in core]
    stability = by_id.get("claim_heldout_cohort_stable")
    component = by_id.get("claim_component_causal_contribution")

    recs: list[str] = []

    component_fail = component is not None and component.status == FAIL
    if any(s == FAIL for s in core_status) or component_fail:
        failed = [c.claim_id for c in core if c.status == FAIL]
        if component_fail:
            failed.append("claim_component_causal_contribution")
        recs.append(
            "KILL CRITERIA: volvence is not better than a control on "
            f"{', '.join(failed)}. Per debt #87, shrink the thesis to a "
            "product-memory / companion claim and downgrade "
            "human-world-model-thesis-2026-06.md; do NOT claim world-model enablement."
        )
        return STATE_KILL, recs

    if any(s == INSUFFICIENT for s in core_status):
        missing = [c.claim_id for c in core if c.status == INSUFFICIENT]
        recs.append(
            "wiring-ready: the flow runs but core claims are not assessable yet "
            f"({', '.join(missing)}). Supply all five tracks' summaries to assess."
        )
        return STATE_WIRING, recs

    all_core_retain = all(s == RETAIN for s in core_status)
    stability_retain = stability is not None and stability.status == RETAIN
    component_retain = component is not None and component.status == RETAIN

    if all_core_retain and stability_retain and component_retain:
        recs.append(
            "first-stage-retained: volvence beats raw, the standard memory "
            "wrapper, the CAMEL agent baseline, the no-bootstrap cold "
            "pipeline, and every component-off arm (PE / ETA / active-learning "
            "/ LoRA-only), with a tight held-out/multi-seed interval. The human "
            "world-model thesis FIRST STAGE may be called retained. World-model "
            "enablement still requires an independent physical-side benchmark."
        )
        return STATE_FIRST_STAGE, recs

    if all_core_retain and not component_retain:
        component_status = component.status if component is not None else INSUFFICIENT
        component_detail = component.detail if component is not None else "claim missing"
        recs.append(
            "core claims retained but claim_component_causal_contribution is "
            f"'{component_status}' ({component_detail}). Per the frozen claim "
            "registry (human-world-model-ablation.md), first-stage-retained "
            "requires all four component arms (pe-off / eta-off / "
            "active-learning-off / lora-adapter) to retain. Run the component "
            "arms on the same substrate to upgrade."
        )
        return STATE_WEAK, recs

    if all_core_retain and stability is not None and stability.status in (WEAK, INSUFFICIENT):
        recs.append(
            "core claims retained but stability is "
            f"'{stability.status}' ({stability.detail}). Run held-out + "
            "multi-seed (P2) to upgrade to first-stage-retained."
        )
        return STATE_WEAK, recs

    recs.append(
        "weak-positive: every core comparison is positive but at least one CI "
        "is not clear of zero. Add seeds / arcs to tighten intervals, then "
        "re-run the verdict."
    )
    return STATE_WEAK, recs


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_table(verdict: AblationVerdict) -> str:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("Companion Bench — same-substrate ablation verdict")
    lines.append("=" * 78)
    lines.append(f"timestamp: {verdict.timestamp_iso}")
    lines.append(f"substrate: {verdict.substrate_note}")
    lines.append("")
    lines.append(f"{'track':<18}{'final_mean':>12}")
    lines.append("-" * 30)
    for name, mean in verdict.tracks.items():
        mean_str = f"{mean:.2f}" if mean is not None else "n/a"
        lines.append(f"{name:<18}{mean_str:>12}")
    lines.append("")
    lines.append("Claims:")
    for c in verdict.claims:
        lines.append(f"  [{c.status.upper():<16}] {c.claim_id}")
        lines.append(f"       {c.detail}")
    lines.append("")
    lines.append("Recommendations:")
    for rec in verdict.recommendations:
        lines.append(f"  - {rec}")
    lines.append("")
    lines.append(f"STATE: {verdict.state.upper()}")
    lines.append("=" * 78)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _parse_track_pair(spec: str) -> tuple[str, pathlib.Path]:
    if "=" not in spec:
        raise ValueError(f"expected track=path, got {spec!r}")
    name, path = spec.split("=", 1)
    return name.strip(), pathlib.Path(path.strip())


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="compare_companion_ablation",
        description=(
            "Compare same-substrate ablation track summaries and emit the "
            "debt #87 thesis retain verdict."
        ),
    )
    p.add_argument(
        "--track",
        action="append",
        default=[],
        metavar="NAME=SUMMARY_JSON",
        help=(
            "per-track summary.json. NAME in {raw, ref-harness, camel, "
            "volvence-cold, volvence, pe-off, eta-off, active-learning-off, "
            "lora-adapter}. Repeatable; supply as many as available."
        ),
    )
    p.add_argument(
        "--substrate-fingerprint",
        action="append",
        default=[],
        metavar="TRACK=MODEL_ID",
        help="optional inline substrate fingerprint for the same-substrate guard.",
    )
    p.add_argument(
        "--fingerprint-file",
        action="append",
        default=[],
        metavar="TRACK=PATH",
        help="optional substrate_fingerprint.json paths for the same-substrate guard.",
    )
    p.add_argument(
        "--stability-rel-halfwidth", type=float, default=DEFAULT_STABILITY_REL_HALFWIDTH,
    )
    p.add_argument(
        "--min-arcs-for-stability", type=int, default=DEFAULT_MIN_ARCS_FOR_STABILITY,
    )
    p.add_argument("--output", type=pathlib.Path, default=None)
    return p


def _maybe_check_substrate(args: argparse.Namespace) -> str:
    """Run the same-substrate guard if fingerprints were supplied. Returns a note."""

    pairs_inline = list(args.substrate_fingerprint)
    pairs_file = list(args.fingerprint_file)
    if not pairs_inline and not pairs_file:
        return "NOT VERIFIED (no fingerprints supplied; rely on serve-time guard)"

    import importlib.util

    guard_path = pathlib.Path(__file__).resolve().parent / "assert_same_substrate.py"
    spec = importlib.util.spec_from_file_location("assert_same_substrate", guard_path)
    assert spec is not None and spec.loader is not None
    guard = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = guard
    spec.loader.exec_module(guard)

    fps = []
    for s in pairs_inline:
        name, model_id = guard.parse_pair(s)
        fps.append(guard.fingerprint_from_inline(name, model_id))
    for s in pairs_file:
        name, path_str = guard.parse_pair(s)
        fps.append(guard.fingerprint_from_file(name, pathlib.Path(path_str)))
    canonical = guard.assert_consistent(fps)  # raises on mismatch
    return f"VERIFIED {canonical.substrate_model_id}"


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.track:
        print("error: at least one --track NAME=summary.json is required", file=sys.stderr)
        return 2

    try:
        substrate_note = _maybe_check_substrate(args)
    except Exception as exc:  # noqa: BLE001 -- guard raises typed errors; surface them
        print(f"error: same-substrate guard failed: {exc}", file=sys.stderr)
        return 1

    tracks: dict[str, TrackResult] = {}
    try:
        for spec in args.track:
            name, path = _parse_track_pair(spec)
            tracks[name] = load_track(name, path)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    verdict = build_verdict(
        tracks=tracks,
        substrate_note=substrate_note,
        stability_rel_halfwidth=args.stability_rel_halfwidth,
        min_arcs_for_stability=args.min_arcs_for_stability,
    )
    print(render_table(verdict))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(verdict.to_json(), indent=2, ensure_ascii=False), encoding="utf-8",
        )
        print(f"verdict JSON -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
