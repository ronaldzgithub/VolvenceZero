"""Qwen-only judge robustness replay (debt #48 weak proxy).

Replays the arc-level judge on already-collected bundle JSONs using
N different Qwen model sizes (qwen3-max / qwen-plus / qwen-flash),
without rerunning the SUT. This is a WEAK PROXY for true cross-family
robustness (#48 ACTIVE) — three Qwen sizes share training data
lineage and likely share family bias. Useful for catching
size-sensitivity outliers (e.g. an axis where qwen-flash collapses)
before paying for a full cross-family sweep.

Cost: ~3 judge × N bundles × ~2K input + ~500 output tokens per arc.
For 8 bundles (4 Qwen + 4 VZ) ≈ 24 arc-judge calls ≈ ¥5-15.

Output:
    artifacts/companion_bench_smoke/judge_robustness_qwen_proxy.json
    {
      "judges": ["qwen3-max", "qwen-plus", "qwen-flash"],
      "per_axis_sigma": {"A1": ..., "A2": ..., ...},
      "spearman_pairwise": {("qwen3-max","qwen-plus"): ..., ...},
      "per_arc_axis_scores": {<arc_id>: {<judge>: {<axis>: score}}},
      "ranking_flip_count": int,
      "ranking_per_judge": {"qwen3-max": [(submission_id, mean_final), ...], ...}
    }

Refs:
    docs/external/companion-bench-openrouter-setup.md §"Judge 合格度档级"
    docs/specs/companion-bench.md §5
    docs/known-debts.md #48
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import pathlib
import sys
import time
from typing import Iterable

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".local" / "llm.env"

_LOG = logging.getLogger("qwen_judge_robustness_replay")


def _source_env(env_file: pathlib.Path) -> None:
    if not env_file.exists():
        raise SystemExit(f"ERROR: {env_file} not found")
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value


def _reconstruct_arc(arc_data: dict):
    from companion_bench.arc_runner import ArcRecord, ArcSession, ArcTurn

    sessions = []
    for s in arc_data.get("sessions", []):
        turns = tuple(
            ArcTurn(
                session_index=t["session_index"],
                turn_index=t["turn_index"],
                inter_session_gap_days=t.get("inter_session_gap_days", 0),
                user_text=t.get("user_text", ""),
                assistant_text=t.get("assistant_text", ""),
                fsm_action=t.get("fsm_action"),
                fsm_payload=t.get("fsm_payload"),
                sut_model_id=t.get("sut_model_id", "?"),
                sut_prompt_tokens=t.get("sut_prompt_tokens"),
                sut_completion_tokens=t.get("sut_completion_tokens"),
                sut_telemetry=dict(t.get("sut_telemetry") or {}),
            )
            for t in s.get("turns", [])
        )
        sessions.append(
            ArcSession(
                session_index=s["session_index"],
                session_id=s["session_id"],
                inter_session_gap_days=s.get("inter_session_gap_days", 0),
                turns=turns,
            )
        )
    return ArcRecord(
        arc_id=arc_data["arc_id"],
        scenario_id=arc_data["scenario_id"],
        scenario_hash=arc_data.get("scenario_hash", ""),
        family=arc_data.get("family", "?"),
        paraphrase_seed=arc_data.get("paraphrase_seed", 0),
        submission_id=arc_data.get("submission_id", "?"),
        sut_model_id=arc_data.get("sut_model_id", "?"),
        started_at=arc_data.get("started_at", ""),
        finished_at=arc_data.get("finished_at", ""),
        sessions=tuple(sessions),
        user_simulator_model=arc_data.get("user_simulator_model", "?"),
        summary_extra=dict(arc_data.get("summary_extra") or {}),
    )


def _reconstruct_ledger(ledger_data: dict):
    from companion_bench.callback_ledger import (
        CallbackClaim,
        CallbackLedger,
        CallbackLedgerEntry,
    )

    entries = []
    for e in ledger_data.get("entries", []):
        entries.append(
            CallbackLedgerEntry(
                claim=CallbackClaim(
                    session_index=e.get("session_index", 0),
                    turn_index=e.get("turn_index", 0),
                    claim_text=e.get("claim_text", ""),
                    claimed_when=e.get("claimed_when", "unknown"),
                ),
                evidence_session=e.get("evidence_session"),
                evidence_turn=e.get("evidence_turn"),
                evidence_text=e.get("evidence_text"),
                matched=bool(e.get("matched", False)),
                similarity_score=float(e.get("similarity_score", 0.0)),
                fabricated=bool(e.get("fabricated", False)),
            )
        )
    return CallbackLedger(arc_id=ledger_data.get("arc_id", "?"), entries=tuple(entries))


def _make_arc_judge(model: str, base_url: str, api_key: str):
    from companion_bench.judge_arc import LLMArcJudge
    from companion_bench.user_simulator import OpenAIUtteranceClient

    client = OpenAIUtteranceClient(
        base_url=base_url, api_key=api_key, model=model, max_tokens=1024
    )

    def complete(prompt: str, *, seed: int, system: str = "") -> str:
        return client.complete(
            system_prompt=system,
            user_prompt=prompt,
            temperature=0.0,
            seed=seed,
        )

    return LLMArcJudge(client_complete=complete, model=model)


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation; numpy-free."""
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    n = len(xs)

    def rank(values: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: values[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx = rank(xs)
    ry = rank(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den_x = (sum((r - mx) ** 2 for r in rx)) ** 0.5
    den_y = (sum((r - my) ** 2 for r in ry)) ** 0.5
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return var ** 0.5


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qwen_judge_robustness_replay")
    p.add_argument(
        "--artifact-dir",
        type=pathlib.Path,
        default=REPO_ROOT / "artifacts" / "companion_bench_smoke",
    )
    p.add_argument(
        "--judges",
        type=lambda s: tuple(x.strip() for x in s.split(",") if x.strip()),
        default=("qwen3-max", "qwen-plus", "qwen-flash"),
    )
    p.add_argument(
        "--base-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    p.add_argument("--key-env", default="PROTOCOL_LLM_API_KEY")
    p.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="JSON output path; default <artifact-dir>/judge_robustness_qwen_proxy.json",
    )
    return p


def _iter_bundle_files(artifact_dir: pathlib.Path) -> Iterable[pathlib.Path]:
    for path in sorted(artifact_dir.rglob("*.bundle.json")):
        yield path


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    _source_env(ENV_FILE)
    api_key = os.environ.get(args.key_env, "").strip()
    if not api_key:
        _LOG.error("env var %s is empty/missing; add to %s", args.key_env, ENV_FILE)
        return 1

    bundles = list(_iter_bundle_files(args.artifact_dir))
    if not bundles:
        _LOG.error("no *.bundle.json found under %s", args.artifact_dir)
        return 2
    _LOG.info("found %d bundles under %s", len(bundles), args.artifact_dir)

    # Build judges (one per Qwen size).
    judges = {model: _make_arc_judge(model, args.base_url, api_key) for model in args.judges}
    _LOG.info("judges: %s", list(judges.keys()))

    # Per-arc per-judge axis scores.
    per_arc: dict[str, dict] = {}
    for bundle_path in bundles:
        data = json.loads(bundle_path.read_text(encoding="utf-8"))
        arc = _reconstruct_arc(data["arc"])
        ledger = _reconstruct_ledger(data["callback_ledger"])
        per_arc[arc.arc_id] = {
            "scenario_id": arc.scenario_id,
            "family": arc.family,
            "submission_id": arc.submission_id,
            "judges": {},
        }
        for model, judge in judges.items():
            t0 = time.time()
            try:
                scores = judge.score(arc=arc, ledger=ledger, family=arc.family)
                axis_dict = {a.value: float(scores[a]) for a in scores}
            except Exception as exc:  # noqa: BLE001 - replay is best-effort
                _LOG.error("[%s @ %s] judge call failed: %s", arc.arc_id, model, exc)
                axis_dict = None
            elapsed = time.time() - t0
            per_arc[arc.arc_id]["judges"][model] = axis_dict
            _LOG.info(
                "[%s] judge=%s axes=%s wallclock=%.1fs",
                arc.arc_id,
                model,
                "ok" if axis_dict else "FAIL",
                elapsed,
            )

    # Compute per-axis sigma across judges (per arc, then averaged).
    axes = ["A1", "A2", "A3", "A4", "A5", "A6"]
    per_axis_sigma_per_arc: dict[str, dict[str, float]] = {}
    for arc_id, arc_block in per_arc.items():
        sigma_block: dict[str, float] = {}
        for axis in axes:
            values = [
                arc_block["judges"][m][axis]
                for m in args.judges
                if arc_block["judges"].get(m) and axis in arc_block["judges"][m]
            ]
            sigma_block[axis] = _stdev(values) if len(values) >= 2 else 0.0
        per_axis_sigma_per_arc[arc_id] = sigma_block

    per_axis_sigma_avg = {
        axis: (
            sum(v[axis] for v in per_axis_sigma_per_arc.values())
            / max(1, len(per_axis_sigma_per_arc))
        )
        for axis in axes
    }

    # Spearman pairwise across judges (per axis -> across arcs flattened).
    spearman_pairwise: dict[str, dict[str, float]] = {}
    judge_pairs: list[tuple[str, str]] = [
        (args.judges[i], args.judges[j])
        for i in range(len(args.judges))
        for j in range(i + 1, len(args.judges))
    ]
    for axis in axes:
        spearman_pairwise[axis] = {}
        for a, b in judge_pairs:
            xs = [
                blk["judges"][a][axis]
                for blk in per_arc.values()
                if blk["judges"].get(a) and blk["judges"].get(b)
            ]
            ys = [
                blk["judges"][b][axis]
                for blk in per_arc.values()
                if blk["judges"].get(a) and blk["judges"].get(b)
            ]
            spearman_pairwise[axis][f"{a}__vs__{b}"] = _spearman(xs, ys)

    # Ranking per judge (mean A1..A6 → submission rank).
    submission_means_per_judge: dict[str, dict[str, float]] = {m: {} for m in args.judges}
    submission_to_arcs: dict[str, list[str]] = {}
    for arc_id, blk in per_arc.items():
        sid = blk["submission_id"]
        submission_to_arcs.setdefault(sid, []).append(arc_id)
    for sid, arc_ids in submission_to_arcs.items():
        for model in args.judges:
            collected: list[float] = []
            for aid in arc_ids:
                axes_block = per_arc[aid]["judges"].get(model)
                if not axes_block:
                    continue
                # Mean of all 6 axes as a proxy for "companionbench_final" rank ordering.
                # Use simple arithmetic mean since this is a robustness replay,
                # not a re-aggregation; the absolute number isn't compared
                # against the production aggregator.
                collected.append(sum(axes_block.get(a, 0.0) for a in axes) / 6)
            if collected:
                submission_means_per_judge[model][sid] = sum(collected) / len(collected)

    ranking_per_judge: dict[str, list[tuple[str, float]]] = {}
    for model, sid_means in submission_means_per_judge.items():
        ranking_per_judge[model] = sorted(sid_means.items(), key=lambda x: -x[1])

    # Ranking flip count: pairwise across judges, count how many SUT pairs
    # are ordered differently.
    sids = sorted({sid for blk in per_arc.values() for sid in [blk["submission_id"]]})
    flip_count = 0
    flip_total_pairs = 0
    for i in range(len(sids)):
        for j in range(i + 1, len(sids)):
            sid_a, sid_b = sids[i], sids[j]
            flip_total_pairs += 1
            ordering_set: set[bool] = set()
            for model in args.judges:
                m = submission_means_per_judge[model]
                if sid_a in m and sid_b in m:
                    ordering_set.add(m[sid_a] > m[sid_b])
            if len(ordering_set) > 1:
                flip_count += 1

    output_path = args.output or args.artifact_dir / "judge_robustness_qwen_proxy.json"
    payload = {
        "judges": list(args.judges),
        "bundle_count": len(bundles),
        "submissions": list(submission_to_arcs.keys()),
        "per_axis_sigma_avg": per_axis_sigma_avg,
        "per_axis_sigma_per_arc": per_axis_sigma_per_arc,
        "spearman_pairwise_per_axis": spearman_pairwise,
        "submission_means_per_judge": submission_means_per_judge,
        "ranking_per_judge": {
            m: [{"submission_id": sid, "mean": mean} for sid, mean in r]
            for m, r in ranking_per_judge.items()
        },
        "ranking_flip_count": flip_count,
        "ranking_total_pairs": flip_total_pairs,
        "per_arc": per_arc,
        "notes": (
            "WEAK PROXY: 3 Qwen sizes share training-data lineage and family bias. "
            "Use #48 cross-family sweep for true robustness ρ. "
            "If sigma > 8 on any axis or ranking_flip_count > 0, even the weak proxy "
            "shows judge instability; cross-family is a hard requirement before launch."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _LOG.info("wrote %s", output_path)

    print()
    print("=== Per-axis sigma (across 3 Qwen judges, avg over arcs) ===")
    for axis in axes:
        print(f"  {axis}: {per_axis_sigma_avg[axis]:6.2f}")
    print()
    print(f"Ranking flip count: {flip_count} / {flip_total_pairs} pairs")
    for model, ranking in ranking_per_judge.items():
        print(f"\n  {model} ranking:")
        for entry in ranking:
            print(f"    {entry[0]:42s}  mean={entry[1]:6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
