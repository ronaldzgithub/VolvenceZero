"""Zero-compute audit of the ETA rate-distortion observation protocol.

The 2026-08-02 reduced Gate-1 sweep never fired the switch gate
(``boundary_f1 == 0`` across every cell) even though distortion dropped a lot.
The hypothesis is that the task does not *require* temporal segmentation: if the
expert action is already a deterministic function of a single observation text,
then a constant control code suffices and there is no pressure to switch.

This script tests that hypothesis without touching the model. It replays the
frozen seeded corpus through the environment, rebuilds the exact
``observation_text`` the rate-distortion bundle would emit (see
``_rate_distortion_observation_bundle`` in
``eta_rate_distortion_evidence.py``), and measures, for several observation
"views", whether the mapping ``observation_text -> expert_action`` is a
function. A view with near-zero ambiguity means memory/segmentation is
mathematically redundant for that view.

Views:

- ``full``: the exact protocol v1 text, including the per-route ``source_text``
  fingerprint.
- ``no_source``: the same text with the ``source_text`` fingerprint replaced by
  a constant, isolating how much route identity the fingerprint leaks.
- ``local_only``: current location + available transitions + completed
  objectives, with neither ``source_text`` nor phase counter.

It also reports, per route, whether any single observation text recurs with a
*different* expert action (the concrete situation that would force a switch).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from volvence_zero.agent.eta_proof_benchmark import generate_eta_proof_corpus
from volvence_zero.internal_rl.proof_environment import (
    HierarchicalRouteSpec,
    MiniHierarchicalEnvironment,
)


@dataclass(frozen=True)
class Row:
    case_id: str
    split: str
    source_text: str
    current_location: str
    available_targets: tuple[str, ...]
    completed: str
    phase_label: str
    action_id: str


def _replay_case(
    *,
    environment: MiniHierarchicalEnvironment,
    case_id: str,
    split: str,
    source_text: str,
    waypoints: tuple[str, ...],
    action_vocabulary: tuple[str, ...],
) -> list[Row]:
    """Rebuild one route's per-phase observation fields and expert actions.

    Mirrors ``_rate_distortion_observation_bundle`` exactly so the audit reads
    the same surface the sweep trains on.
    """

    route = HierarchicalRouteSpec(
        case_id=case_id,
        split=split,
        source_text=source_text,
        waypoints=waypoints,
    )
    state = environment.reset(route)
    rows: list[Row] = []
    for target_id in route.waypoints[1:]:
        observation = environment.observe(state)
        target_location = environment.location(target_id)
        phase_count = (
            max(target_location.min_persistence, 1)
            if target_location.is_objective
            else 1
        )
        if target_id not in action_vocabulary:
            raise ValueError(
                f"target {target_id!r} missing from action vocabulary."
            )
        completed = (
            ", ".join(observation.completed_objective_ids)
            if observation.completed_objective_ids
            else "none"
        )
        for phase_index in range(phase_count):
            rows.append(
                Row(
                    case_id=case_id,
                    split=split,
                    source_text=source_text,
                    current_location=observation.current_location_id,
                    available_targets=tuple(observation.available_targets),
                    completed=completed,
                    phase_label=f"Phase {phase_index + 1} of {phase_count}",
                    action_id=f"move:{target_id}",
                )
            )
        state = environment.step(state, target_id=target_id).next_state
    return rows


def _view_text(row: Row, *, view: str) -> str:
    transitions = ", ".join(row.available_targets)
    if view == "full":
        return (
            f"Task context: {row.source_text}. Current location: "
            f"{row.current_location}. Available transitions: {transitions}. "
            f"Completed objectives: {row.completed}. {row.phase_label}."
        )
    if view == "no_source":
        return (
            "Task context: <route-identity-removed>. Current location: "
            f"{row.current_location}. Available transitions: {transitions}. "
            f"Completed objectives: {row.completed}. {row.phase_label}."
        )
    if view == "local_only":
        return (
            f"Current location: {row.current_location}. Available "
            f"transitions: {transitions}. Completed objectives: "
            f"{row.completed}."
        )
    if view == "local_no_completed":
        # Neither route identity nor progress: only the current node and its
        # out-edges. If a single route revisits one such view with a different
        # next action, then progress can only be carried by an evolving latent
        # code -- i.e. switching becomes necessary, not just per-route memory.
        return (
            f"Current location: {row.current_location}. Available "
            f"transitions: {transitions}."
        )
    raise ValueError(f"unknown view {view!r}")


def _assess_view(rows: list[Row], *, view: str) -> dict[str, object]:
    text_to_actions: dict[str, set[str]] = defaultdict(set)
    text_to_steps: dict[str, int] = defaultdict(int)
    for row in rows:
        text = _view_text(row, view=view)
        text_to_actions[text].add(row.action_id)
        text_to_steps[text] += 1

    total_steps = len(rows)
    distinct_texts = len(text_to_actions)
    ambiguous_texts = {
        text for text, actions in text_to_actions.items() if len(actions) > 1
    }
    ambiguous_steps = sum(
        text_to_steps[text] for text in ambiguous_texts
    )
    determinism_rate = (
        1.0 - ambiguous_steps / total_steps if total_steps else 0.0
    )
    return {
        "view": view,
        "total_steps": total_steps,
        "distinct_texts": distinct_texts,
        "ambiguous_texts": len(ambiguous_texts),
        "ambiguous_steps": ambiguous_steps,
        "determinism_rate": round(determinism_rate, 6),
        "interpretation": (
            "expert action is a deterministic function of a single "
            "observation (segmentation redundant)"
            if ambiguous_steps == 0
            else "some observations map to multiple expert actions "
            "(latent memory could help)"
        ),
    }


def _within_route_ambiguity(rows: list[Row], *, view: str) -> dict[str, object]:
    """How often a single route revisits one observation with a new action.

    This is the concrete situation that would *force* a switch: the same
    observation text recurs inside one route but the correct action changed.
    """

    per_route: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for row in rows:
        per_route[row.case_id][_view_text(row, view=view)].add(row.action_id)
    routes_with_conflict = 0
    conflicting_pairs = 0
    for _case_id, text_map in per_route.items():
        conflicts = [
            actions for actions in text_map.values() if len(actions) > 1
        ]
        if conflicts:
            routes_with_conflict += 1
            conflicting_pairs += len(conflicts)
    return {
        "view": view,
        "routes_total": len(per_route),
        "routes_with_intra_route_conflict": routes_with_conflict,
        "conflicting_observation_slots": conflicting_pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether the ETA rate-distortion observation protocol makes "
            "the expert action a deterministic function of one observation."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/eta_stage1_protocol_audit_20260802"),
    )
    parser.add_argument("--corpus-seed", type=int, default=20260802)
    parser.add_argument("--objective-count", type=int, default=8)
    parser.add_argument("--corridor-count", type=int, default=2)
    parser.add_argument("--extra-edge-probability", type=float, default=0.35)
    parser.add_argument("--train-routes", type=int, default=64)
    parser.add_argument("--heldout-routes", type=int, default=24)
    parser.add_argument(
        "--train-lengths", type=int, nargs="+", default=[2, 3]
    )
    parser.add_argument(
        "--heldout-lengths", type=int, nargs="+", default=[3, 4]
    )
    args = parser.parse_args()

    corpus = generate_eta_proof_corpus(
        seed=args.corpus_seed,
        objective_count=args.objective_count,
        corridor_count=args.corridor_count,
        extra_edge_probability=args.extra_edge_probability,
        train_route_count=args.train_routes,
        heldout_route_count=args.heldout_routes,
        train_lengths=tuple(args.train_lengths),
        heldout_lengths=tuple(args.heldout_lengths),
    )
    environment = corpus.environment
    target_ids = {
        transition.target_id for transition in environment.transitions
    }
    action_vocabulary = tuple(
        location.location_id
        for location in environment.locations
        if location.location_id in target_ids
    )

    rows: list[Row] = []
    for case in (*corpus.train_cases, *corpus.heldout_cases):
        rows.extend(
            _replay_case(
                environment=environment,
                case_id=case.case_id,
                split=case.split,
                source_text=case.source_text,
                waypoints=case.route_signature,
                action_vocabulary=action_vocabulary,
            )
        )

    views = ("full", "no_source", "local_only", "local_no_completed")
    view_assessments = [_assess_view(rows, view=view) for view in views]
    intra_route = [
        _within_route_ambiguity(rows, view=view) for view in views
    ]

    full = next(a for a in view_assessments if a["view"] == "full")
    no_source = next(a for a in view_assessments if a["view"] == "no_source")
    segmentation_redundant = full["ambiguous_steps"] == 0
    leak_delta = round(
        float(no_source["determinism_rate"])
        - float(full["determinism_rate"]),
        6,
    )
    intra_by_view = {a["view"]: a for a in intra_route}
    # A view forces switching (evolving code) only if a single route revisits
    # one observation with a different action. Per-step route/progress leaks
    # (source_text, completed objectives) suppress this, so a constant code
    # would suffice under any view whose intra-route conflict is zero.
    switching_forced_views = [
        view
        for view, a in intra_by_view.items()
        if a["routes_with_intra_route_conflict"] > 0
    ]
    completed_leak_matters = (
        intra_by_view["local_only"]["routes_with_intra_route_conflict"] == 0
        and intra_by_view["local_no_completed"][
            "routes_with_intra_route_conflict"
        ]
        > 0
    )

    report = {
        "schema_version": "eta-protocol-determinism-audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus": {
            "seed": args.corpus_seed,
            "objective_count": args.objective_count,
            "corridor_count": args.corridor_count,
            "extra_edge_probability": args.extra_edge_probability,
            "train_routes": args.train_routes,
            "heldout_routes": args.heldout_routes,
            "train_lengths": args.train_lengths,
            "heldout_lengths": args.heldout_lengths,
        },
        "action_vocabulary": list(action_vocabulary),
        "total_steps": len(rows),
        "view_assessments": view_assessments,
        "intra_route_ambiguity": intra_route,
        "verdict": {
            "segmentation_redundant_under_v1": segmentation_redundant,
            "source_text_leak_determinism_delta": leak_delta,
            "views_that_force_switching": switching_forced_views,
            "completed_objectives_leak_blocks_switching": completed_leak_matters,
            "note": (
                "segmentation_redundant_under_v1=True confirms problem B: with "
                "the v1 observation surface the expert action is a function of "
                "one observation, so a constant control code suffices and the "
                "switch gate has no reason to fire. A large negative "
                "source_text_leak_determinism_delta means removing the "
                "per-route source_text fingerprint is what breaks determinism. "
                "But dropping source_text alone leaves intra-route conflict at "
                "zero, so a CONSTANT per-route code still solves the task -- no "
                "switching pressure. Only when BOTH source_text and the "
                "completed-objectives progress field are removed from recurring "
                "steps does a single route revisit a node with a different "
                "action, which can only be resolved by an EVOLVING latent code "
                "(genuine switching). Therefore protocol v2 must give the route "
                "plan once at step 0 and drop both source_text and completed "
                "objectives from every later step."
            ),
        },
    }

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "protocol_determinism.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# ETA rate-distortion observation protocol determinism audit",
        "",
        "Zero-compute test of whether the expert action is a deterministic "
        "function of a single observation (i.e. whether temporal segmentation "
        "is required at all).",
        "",
        f"- corpus seed {args.corpus_seed}, "
        f"{args.train_routes} train / {args.heldout_routes} heldout routes, "
        f"{len(rows)} total phases",
        f"- action vocabulary ({len(action_vocabulary)}): "
        f"{', '.join(action_vocabulary)}",
        "",
        "## Determinism by observation view",
        "",
        "| view | steps | distinct texts | ambiguous texts | ambiguous steps "
        "| determinism |",
        "|---|---|---|---|---|---|",
    ]
    for a in view_assessments:
        lines.append(
            f"| {a['view']} | {a['total_steps']} | {a['distinct_texts']} | "
            f"{a['ambiguous_texts']} | {a['ambiguous_steps']} | "
            f"{a['determinism_rate']:.4f} |"
        )
    lines += [
        "",
        "## Intra-route recurrence (would force a switch)",
        "",
        "| view | routes | routes with intra-route conflict | conflicting "
        "slots |",
        "|---|---|---|---|",
    ]
    for a in intra_route:
        lines.append(
            f"| {a['view']} | {a['routes_total']} | "
            f"{a['routes_with_intra_route_conflict']} | "
            f"{a['conflicting_observation_slots']} |"
        )
    lines += [
        "",
        "## Verdict",
        "",
        f"- segmentation redundant under v1: "
        f"**{segmentation_redundant}**",
        f"- source_text leak determinism delta (no_source - full): "
        f"**{leak_delta:+.4f}**",
        f"- views that force switching (intra-route conflict > 0): "
        f"**{switching_forced_views or 'none'}**",
        f"- completed-objectives leak blocks switching under no_source: "
        f"**{completed_leak_matters}**",
        "",
        "Dropping source_text alone leaves intra-route conflict at zero, so a "
        "constant per-route code still solves v1 minus the fingerprint. Only "
        "removing BOTH source_text and completed objectives from recurring "
        "steps forces an evolving latent code. Protocol v2 must therefore give "
        "the route plan once at step 0 and drop both fields from every later "
        "step.",
    ]
    (out / "protocol_determinism.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    payload = report_path.read_bytes()
    sha = hashlib.sha256(payload).hexdigest()
    (out / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "eta-protocol-audit-manifest.v1",
                "created_at": report["created_at"],
                "report": "protocol_determinism.json",
                "report_sha256": sha,
                "verdict": report["verdict"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"wrote {report_path}")
    print(f"segmentation_redundant_under_v1: {segmentation_redundant}")
    for a in view_assessments:
        print(
            f"  view={a['view']:<11} determinism={a['determinism_rate']:.4f} "
            f"ambiguous_steps={a['ambiguous_steps']}"
        )
    print(f"source_text_leak_determinism_delta: {leak_delta:+.4f}")


if __name__ == "__main__":
    main()
