#!/usr/bin/env python3
"""Three-track ablation diff + go/no-go verdict for EQ-Bench 3 runs.

Reads the per-track ``*.summary.json`` files emitted by
``run_eqbench3.py`` and produces:

* A delta table comparing the three tracks (rubric scores).
* Two structured verdicts:
    1. **Pipeline contribution**: companion - raw. Positive means the
       lifeform pipeline added EQ score over the bare substrate;
       negative means the pipeline subtracted score.
    2. **Bootstrap contribution**: companion - companion-cold.
       Positive means the trained companion bootstraps add score on
       top of the cold lifeform pipeline.
* A go/no-go gate for whether to spend the additional ELO judging
  budget and proceed to public submission (Packet 10).

Reproducibility checks:

* Refuses to emit a verdict if any track is missing the attestation
  block declaring the four debt #29 red lines (``frozen_substrate``,
  ``no_kernel_modification``, ``no_benchmark_text_in_system_prompt``,
  ``no_internal_architecture_terms_in_model_card``). Without these
  declarations we cannot confirm the run is publishable.
* Refuses if the substrate model id is inconsistent across tracks —
  three-track ablation requires the SAME Qwen weights served three
  ways.

Output:

* Printed table for terminal review.
* Optional ``--output`` writes a single JSON file with the same
  content for downstream tooling (Packet 9 will consume it for the
  cross-walk doc).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import logging
import pathlib
import sys
from typing import Any

_LOG = logging.getLogger("compare_ablation")

# Industry-mean rubric threshold below which we recommend NOT
# submitting publicly (per debt #29 修法 (2) "若分数 < 预期 → 不公开").
# 65 is the rough Qwen-2.5-1.5B band; higher-substrate runs should
# raise this threshold.
_DEFAULT_PUBLISH_THRESHOLD: float = 65.0

# Required keys in every track summary's attestation block.
_REQUIRED_ATTESTATION_KEYS: tuple[str, ...] = (
    "frozen_substrate",
    "no_kernel_modification",
    "no_benchmark_text_in_system_prompt",
    "no_internal_architecture_terms_in_model_card",
    "track",
    "request_mode",
    "substrate_model_id",
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TrackResult:
    name: str
    rubric_average: float | None
    request_mode: str
    substrate_model_id: str
    summary_path: pathlib.Path
    attestation: dict[str, Any]


def _load_summary(path: pathlib.Path) -> TrackResult:
    if not path.exists():
        raise FileNotFoundError(f"summary file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    track = data.get("track")
    if not isinstance(track, str) or not track.strip():
        raise ValueError(
            f"{path}: missing or invalid 'track' field — was this written by run_eqbench3?"
        )
    attestation = data.get("attestation")
    if not isinstance(attestation, dict):
        raise ValueError(
            f"{path}: missing 'attestation' block (debt #29 red lines)"
        )
    for key in _REQUIRED_ATTESTATION_KEYS:
        if key not in attestation:
            raise ValueError(
                f"{path}: attestation missing required key {key!r} "
                "(debt #29 red lines must be affirmed for verdict emission)"
            )
        # The four red-line bools must explicitly be ``True``.
        if key in {
            "frozen_substrate",
            "no_kernel_modification",
            "no_benchmark_text_in_system_prompt",
            "no_internal_architecture_terms_in_model_card",
        } and attestation[key] is not True:
            raise ValueError(
                f"{path}: attestation['{key}'] is not True — runner did not "
                "affirm a debt #29 red line. Verdict cannot be emitted."
            )
    rubric = data.get("rubric_average")
    rubric_value: float | None
    if rubric is None:
        rubric_value = None
    elif isinstance(rubric, (int, float)):
        rubric_value = float(rubric)
    else:
        raise ValueError(
            f"{path}: 'rubric_average' must be number or null; got {type(rubric).__name__}"
        )
    return TrackResult(
        name=track,
        rubric_average=rubric_value,
        request_mode=str(attestation["request_mode"]),
        substrate_model_id=str(attestation["substrate_model_id"]),
        summary_path=path,
        attestation=attestation,
    )


def _verify_consistent_substrate(results: list[TrackResult]) -> None:
    if not results:
        return
    base = results[0].substrate_model_id
    for r in results[1:]:
        if r.substrate_model_id != base:
            raise ValueError(
                f"ablation tracks have inconsistent substrate models: "
                f"{r.name}={r.substrate_model_id!r} vs {results[0].name}={base!r}. "
                "Three-track ablation requires the SAME substrate served three ways."
            )


# ---------------------------------------------------------------------------
# Delta + verdict
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class AblationVerdict:
    """Structured verdict for downstream tooling."""

    substrate_model_id: str
    tracks: dict[str, float | None]
    pipeline_delta: float | None
    bootstrap_delta: float | None
    publish_threshold: float
    recommendations: tuple[str, ...]
    go_no_go: str  # "go", "hold", or "insufficient_data"
    timestamp_iso: str

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _compute_delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return round(a - b, 4)


def _build_verdict(
    *,
    results: list[TrackResult],
    publish_threshold: float,
) -> AblationVerdict:
    by_name = {r.name: r for r in results}
    rubric_by_name = {r.name: r.rubric_average for r in results}

    companion = by_name.get("companion")
    companion_cold = by_name.get("companion-cold")
    raw = by_name.get("raw")

    pipeline_delta = (
        _compute_delta(
            companion.rubric_average if companion else None,
            raw.rubric_average if raw else None,
        )
        if companion and raw
        else None
    )
    bootstrap_delta = (
        _compute_delta(
            companion.rubric_average if companion else None,
            companion_cold.rubric_average if companion_cold else None,
        )
        if companion and companion_cold
        else None
    )

    recommendations: list[str] = []

    # Pipeline contribution recommendation.
    if pipeline_delta is None:
        recommendations.append(
            "pipeline-vs-raw delta unknown (need both companion and raw tracks "
            "with valid rubric scores)"
        )
    elif pipeline_delta > 1.0:
        recommendations.append(
            f"pipeline contribution +{pipeline_delta:.2f} is meaningful — "
            "the lifeform stack adds EQ score over the bare substrate."
        )
    elif pipeline_delta > -1.0:
        recommendations.append(
            f"pipeline contribution {pipeline_delta:+.2f} is within noise. "
            "Worth diagnosing whether expression-layer rewriting is "
            "swallowing model expressivity; not a publication blocker by itself."
        )
    else:
        recommendations.append(
            f"pipeline contribution {pipeline_delta:.2f} is NEGATIVE — the "
            "lifeform stack subtracts EQ score on this benchmark. Open a "
            "follow-up debt for diagnosis (likely PromptPlanner constraints "
            "or refusal layer over-firing) before public submission."
        )

    # Bootstrap contribution recommendation.
    if bootstrap_delta is None:
        recommendations.append(
            "bootstrap-vs-cold delta unknown (need both companion and "
            "companion-cold tracks)"
        )
    elif bootstrap_delta > 1.0:
        recommendations.append(
            f"trained bootstraps add +{bootstrap_delta:.2f} on cold lifeform — "
            "the offline calibration is paying out on EQ."
        )
    elif bootstrap_delta > -1.0:
        recommendations.append(
            f"trained bootstraps {bootstrap_delta:+.2f} ≈ no effect on EQ. "
            "Not a regression but worth noting in LSCB cross-walk."
        )
    else:
        recommendations.append(
            f"trained bootstraps {bootstrap_delta:.2f} HURT EQ relative to "
            "cold lifeform. Worth diagnosing whether regime priors over-bias "
            "in the benchmark scenarios."
        )

    # Publish threshold gate.
    primary = companion if companion else (raw if raw else None)
    if primary is None or primary.rubric_average is None:
        recommendations.append(
            "no primary rubric score available for publish-threshold gate"
        )
        go_no_go = "insufficient_data"
    elif primary.rubric_average >= publish_threshold:
        recommendations.append(
            f"{primary.name} rubric {primary.rubric_average:.2f} >= "
            f"threshold {publish_threshold:.2f}: GO for Packet 8 (EmpathyBench), "
            "Packet 9 (cross-walk doc), and ELO pass via --with-elo."
        )
        go_no_go = "go"
    else:
        recommendations.append(
            f"{primary.name} rubric {primary.rubric_average:.2f} < "
            f"threshold {publish_threshold:.2f}: HOLD on public submission. "
            "Keep results internal as baseline; consider larger substrate "
            "(7B/14B) per debt #29 修法 (2) decision tree."
        )
        go_no_go = "hold"

    substrate = results[0].substrate_model_id if results else "unknown"
    return AblationVerdict(
        substrate_model_id=substrate,
        tracks=rubric_by_name,
        pipeline_delta=pipeline_delta,
        bootstrap_delta=bootstrap_delta,
        publish_threshold=publish_threshold,
        recommendations=tuple(recommendations),
        go_no_go=go_no_go,
        timestamp_iso=_dt.datetime.now(_dt.timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_table(verdict: AblationVerdict) -> str:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("EQ-Bench 3 — three-track ablation summary")
    lines.append("=" * 78)
    lines.append(f"substrate: {verdict.substrate_model_id}")
    lines.append(f"timestamp: {verdict.timestamp_iso}")
    lines.append(f"publish_threshold: {verdict.publish_threshold:.2f} (rubric)")
    lines.append("")
    lines.append(f"{'track':<20}{'rubric_avg':>12}")
    lines.append("-" * 32)
    for name, rubric in verdict.tracks.items():
        rubric_str = f"{rubric:.2f}" if rubric is not None else "n/a"
        lines.append(f"{name:<20}{rubric_str:>12}")
    lines.append("")
    lines.append("Deltas:")
    if verdict.pipeline_delta is not None:
        lines.append(f"  companion - raw          = {verdict.pipeline_delta:+.2f}  (pipeline contribution)")
    else:
        lines.append("  companion - raw          = n/a  (pipeline contribution)")
    if verdict.bootstrap_delta is not None:
        lines.append(f"  companion - cold         = {verdict.bootstrap_delta:+.2f}  (bootstrap contribution)")
    else:
        lines.append("  companion - cold         = n/a  (bootstrap contribution)")
    lines.append("")
    lines.append("Recommendations:")
    for rec in verdict.recommendations:
        lines.append(f"  - {rec}")
    lines.append("")
    lines.append(f"go/no-go: {verdict.go_no_go.upper()}")
    lines.append("=" * 78)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="compare_ablation",
        description=(
            "Compare per-track EQ-Bench summary JSONs and emit go/no-go "
            "verdict for debt #29 packet 7."
        ),
    )
    p.add_argument(
        "--summaries",
        nargs="+",
        type=pathlib.Path,
        required=True,
        help=(
            "List of *.summary.json files emitted by run_eqbench3.py. "
            "Pass at least two for delta computation."
        ),
    )
    p.add_argument(
        "--publish-threshold",
        type=float,
        default=_DEFAULT_PUBLISH_THRESHOLD,
        help=(
            f"Minimum rubric score for go-verdict (default {_DEFAULT_PUBLISH_THRESHOLD}). "
            "Raise this when running on larger substrates (7B/14B should clear ~70+)."
        ),
    )
    p.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Optional path to dump the structured verdict as JSON.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    if not args.summaries:
        print("error: --summaries is required and must list at least one path", file=sys.stderr)
        return 2

    results: list[TrackResult] = []
    for path in args.summaries:
        try:
            results.append(_load_summary(path))
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    try:
        _verify_consistent_substrate(results)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    verdict = _build_verdict(results=results, publish_threshold=args.publish_threshold)
    print(_render_table(verdict))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(verdict.to_json(), f, indent=2, ensure_ascii=False)
        print(f"verdict JSON → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
