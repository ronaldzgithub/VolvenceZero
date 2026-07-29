#!/usr/bin/env python3
"""Build the direct multi-seed, multi-judge G-vs-B-prime quality gate."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _src in sorted((REPO_ROOT / "packages").glob("*/src")):
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from volvence_zero.state_kv_quality_noninferiority import (  # noqa: E402
    QualityPair,
    build_quality_noninferiority_verdict,
)

_PATH_PATTERN = re.compile(
    r"p2-state-strategy-routed-(?P<scenario>.+)-rollout-seed-"
    r"(?P<seed>\d+)-full-max16(?P<judge_suffix>-m3e)?$"
)


def _default_inputs() -> tuple[Path, ...]:
    paths = []
    root = REPO_ROOT / "artifacts" / "state_kv"
    for path in sorted(root.glob("p2-state-strategy-routed-*/verdict_identification.json")):
        match = _PATH_PATTERN.fullmatch(path.parent.name)
        if match is not None:
            paths.append(path)
    return tuple(paths)


def _pair_from_verdict(path: Path) -> QualityPair:
    match = _PATH_PATTERN.fullmatch(path.parent.name)
    if match is None:
        raise ValueError(f"cannot derive scenario/seed from {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("candidate_arm") != "state-kv-arm-g-prefix-pure":
        raise ValueError(f"{path} is not a G-prefix identification verdict")
    matching = payload.get("matching")
    if not isinstance(matching, list):
        raise TypeError(f"{path} matching must be an array")
    rows = {
        str(row["arm"]): row
        for row in matching
        if isinstance(row, dict) and "arm" in row
    }
    candidate = rows.get("state-kv-arm-g-prefix-pure")
    baseline = rows.get("state-kv-arm-bprime")
    if candidate is None or baseline is None:
        raise ValueError(f"{path} lacks G or B-prime matching rows")
    candidate_judge = str(candidate["judge_model_id"])
    baseline_judge = str(baseline["judge_model_id"])
    if candidate_judge != baseline_judge:
        raise ValueError(f"{path} G and B-prime use different judges")
    return QualityPair(
        experiment_id=path.parent.name,
        scenario_id=match.group("scenario"),
        sampling_seed=int(match.group("seed")),
        judge_model_id=candidate_judge,
        substrate_fingerprint=str(payload["substrate_fingerprint"]),
        candidate_accuracy=float(candidate["accuracy"]),
        bprime_accuracy=float(baseline["accuracy"]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument(
        "--output",
        default=(
            "artifacts/state_kv/quality-noninferiority/"
            "verdict_quality_noninferiority.json"
        ),
    )
    parser.add_argument("--noninferiority-margin", type=float, default=0.0)
    args = parser.parse_args(argv)

    inputs = (
        tuple(Path(value).expanduser().resolve() for value in args.input)
        if args.input
        else _default_inputs()
    )
    verdict = build_quality_noninferiority_verdict(
        pairs=tuple(_pair_from_verdict(path) for path in inputs),
        noninferiority_margin=args.noninferiority_margin,
    )
    output = (REPO_ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(verdict.to_json() + "\n", encoding="utf-8")
    print(f"gate_state = {verdict.gate_state}")
    print(f"pairs = {len(verdict.pairs)}")
    print(f"delta_ci = {verdict.delta_ci}")
    print(f"output = {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
