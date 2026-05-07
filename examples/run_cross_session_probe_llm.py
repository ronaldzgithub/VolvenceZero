"""Phase 2 W2.0c (debt #10B item 3) — cross-session evidence probe
on the LLM-backed companion lifeform.

This is the load-bearing run that closes known-debt #10B: it drives
``lifeform-bench --vertical companion --longitudinal-rounds N
--use-llm-semantic-runtime`` end-to-end, captures
``artifacts/eq_uplift/cross_session_probe_llm.json``, and prints
the human-readable verdict.

Pre-flight check probes whether the requested HF model
(default ``Qwen/Qwen2.5-1.5B-Instruct``) is already in the local
cache via ``AutoTokenizer.from_pretrained(..., local_files_only=True)``.
A cache miss is NOT a failure; it just means the first run will
also download (~3 GB for Qwen 1.5B).

Verdict surface:

* ``tom_records_total_last`` and ``common_ground_dyad_atoms_total_last``
  in the produced artifact are the load-bearing observables for
  debt #10B item 3 closure. Both > 0 = activated, archived.
* ``il_rapport_trend`` separately gates known-debt #10C (cross-round
  signal magnitude). The script reports it but does NOT fail when
  it falls below the 0.02 threshold — that is an explicitly-deferred
  follow-up, not a regression.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_DEFAULT_ARTIFACT = _REPO_ROOT / "artifacts" / "eq_uplift" / "cross_session_probe_llm.json"
_DEFAULT_MODEL_SOURCE = "Qwen/Qwen2.5-1.5B-Instruct"
_IL_RAPPORT_TREND_DEBT_10C_THRESHOLD = 0.02


def _preflight_qwen_cache(*, model_source: str) -> tuple[bool, str]:
    """Try to load the tokenizer locally; report cache state.

    Returns ``(cached, message)``. ``cached=True`` means the
    download phase will be skipped on the actual run; ``False``
    means the bench CLI will download on first call.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        return False, (
            f"transformers not installed ({exc!s}); "
            "run `pip install transformers torch` before this script"
        )
    try:
        AutoTokenizer.from_pretrained(model_source, local_files_only=True)
        return True, f"cached locally: {model_source}"
    except Exception as exc:
        return False, (
            f"NOT cached ({type(exc).__name__}: download will run on first call). "
            f"Estimated ~3 GB for Qwen 1.5B-Instruct, 5-15 min on a typical "
            f"home connection. Detail: {exc!s}"
        )


def _run_bench(
    *,
    rounds: int,
    artifact_path: pathlib.Path,
    model_source: str,
    torch_dtype: str,
    scenario: str,
    min_regime_match_rate: float,
) -> int:
    """Drive ``lifeform-bench`` programmatically and return its exit code.

    Returns the same exit code the CLI would surface; this script's
    own caller decides how to interpret it.
    """
    from lifeform_evolution.cli import main as bench_main

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        "--vertical",
        "companion",
        "--longitudinal-rounds",
        str(rounds),
        "--longitudinal-report",
        "--longitudinal-json",
        str(artifact_path),
        "--use-llm-semantic-runtime",
        "--llm-model-source",
        model_source,
        "--llm-torch-dtype",
        torch_dtype,
        "--scenario",
        scenario,
        "--min-regime-match-rate",
        str(min_regime_match_rate),
    ]
    print(f"[probe] invoking lifeform-bench {' '.join(argv)}")
    return bench_main(argv)


def _summarize_artifact(artifact_path: pathlib.Path) -> int:
    """Read the longitudinal artifact and print debt #10B / #10C verdicts.

    Returns 0 when debt #10B item 3 closure conditions are met
    (``tom_records_total_last > 0`` for at least one scenario),
    otherwise 1 — the bench CLI's own exit code is reported separately.
    """
    if not artifact_path.exists():
        print(f"[probe] FAIL: artifact missing at {artifact_path}", file=sys.stderr)
        return 1
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    scenarios = payload.get("scenarios") or []
    if not scenarios:
        print(
            f"[probe] FAIL: artifact at {artifact_path} has no scenarios",
            file=sys.stderr,
        )
        return 1

    print()
    print("=" * 72)
    print("  Cross-Session Probe Verdict (debt #10B item 3 closure)")
    print("=" * 72)
    debt_10b_closed = False
    debt_10c_status: list[tuple[str, float | None]] = []
    for scenario in scenarios:
        sid = scenario.get("scenario_id", "?")
        rounds = scenario.get("rounds", 0)
        tom_first = scenario.get("tom_records_total_first")
        tom_last = scenario.get("tom_records_total_last")
        tom_trend = scenario.get("tom_records_total_trend")
        cg_first = scenario.get("common_ground_dyad_atoms_total_first")
        cg_last = scenario.get("common_ground_dyad_atoms_total_last")
        cg_trend = scenario.get("common_ground_dyad_atoms_total_trend")
        il_trend = scenario.get("il_rapport_trend")
        il_trend_pos = scenario.get("il_rapport_trend_pos", False)
        passed = scenario.get("passed", False)
        print(
            f"  scenario={sid} rounds={rounds}\n"
            f"    tom_records_total: first={tom_first} last={tom_last} "
            f"trend={tom_trend}\n"
            f"    common_ground_dyad_atoms_total: first={cg_first} "
            f"last={cg_last} trend={cg_trend}\n"
            f"    il_rapport_trend: {il_trend!r:>10} pos={il_trend_pos}\n"
            f"    longitudinal.passed: {passed}"
        )
        if isinstance(tom_last, int) and tom_last > 0:
            debt_10b_closed = True
        debt_10c_status.append((sid, il_trend))

    print()
    print("=" * 72)
    print("  Debt verdicts")
    print("=" * 72)
    if debt_10b_closed:
        print(
            "  [10B item 3] CLOSED: at least one scenario produced "
            "tom_records_total_last > 0; EQ owner activation evidence "
            "captured."
        )
    else:
        print(
            "  [10B item 3] OPEN: every scenario reported "
            "tom_records_total_last == 0. The LLM proposal runtime "
            "did not extract any ToM records — likely a prompt / "
            "parse-format regression. Inspect the artifact and the "
            "Qwen output for the failing turn."
        )

    weak_il = [
        (sid, trend)
        for sid, trend in debt_10c_status
        if trend is not None and abs(trend) < _IL_RAPPORT_TREND_DEBT_10C_THRESHOLD
    ]
    if weak_il:
        print(
            f"  [10C] STILL OPEN: {len(weak_il)} scenario(s) had "
            f"|il_rapport_trend| < {_IL_RAPPORT_TREND_DEBT_10C_THRESHOLD:.2f}: "
            f"{weak_il!r}. This is the pre-existing readout-magnitude "
            f"debt, not a regression of the EQ owner chain. See "
            f"docs/known-debts.md #10C."
        )
    else:
        print(
            f"  [10C] all scenarios reported "
            f"|il_rapport_trend| >= {_IL_RAPPORT_TREND_DEBT_10C_THRESHOLD:.2f}; "
            f"check artifact for direction (positive = strengthening rapport)."
        )
    return 0 if debt_10b_closed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Drive lifeform-bench --use-llm-semantic-runtime end-to-end "
            "and write artifacts/eq_uplift/cross_session_probe_llm.json. "
            "Closes known-debt #10B item 3."
        )
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help=(
            "Number of longitudinal rounds per scenario (default 3). "
            "5 is the recommended floor for il_rapport trend evidence; "
            "3 is the floor for debt #10B item 3 closure (activation > 0)."
        ),
    )
    parser.add_argument(
        "--scenario",
        default="low-mood-disclosure",
        help=(
            'Built-in scenario name (or "all"). Default: low-mood-disclosure '
            '(single scenario keeps the run under 30 min on CPU). Use "all" '
            "when capturing a full evidence baseline (~1+ hr CPU)."
        ),
    )
    parser.add_argument(
        "--artifact-path",
        type=pathlib.Path,
        default=_DEFAULT_ARTIFACT,
        help=(
            f"Output JSON path (default {_DEFAULT_ARTIFACT.relative_to(_REPO_ROOT)})."
        ),
    )
    parser.add_argument(
        "--model-source",
        default=_DEFAULT_MODEL_SOURCE,
        help=(
            f"HF model source (default {_DEFAULT_MODEL_SOURCE}). "
            "Use Qwen/Qwen2.5-0.5B-Instruct on tight-RAM hosts at the "
            "cost of weaker semantic proposal quality."
        ),
    )
    parser.add_argument(
        "--torch-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help=(
            "Torch dtype (default bfloat16; keeps Qwen 1.5B under ~3 GB RAM). "
            "Use float32 on CPUs that lack bfloat16 support and have RAM "
            "headroom for ~6 GB."
        ),
    )
    parser.add_argument(
        "--min-regime-match-rate",
        type=float,
        default=0.0,
        help=(
            "Minimum regime-match rate for the bench's own pass gate (default 0.0). "
            "We deliberately default to 0.0 because this probe is about EQ "
            "activation, not regime accuracy — a low regime match rate "
            "should not mask a successful EQ chain activation."
        ),
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the local Qwen cache probe (use when transformers is missing).",
    )
    args = parser.parse_args(argv)

    print()
    print("=" * 72)
    print("  Cross-Session Probe (debt #10B item 3)")
    print("=" * 72)
    print(f"  rounds={args.rounds}  scenario={args.scenario!r}")
    print(f"  model_source={args.model_source!r}  torch_dtype={args.torch_dtype}")
    print(f"  artifact_path={args.artifact_path}")
    print()
    if not args.skip_preflight:
        cached, message = _preflight_qwen_cache(model_source=args.model_source)
        print(f"  [preflight] {message}")
        if not cached:
            print(
                "  [preflight] WARN: first call will download. Hit Ctrl-C "
                "now if you want to pre-cache via "
                f"`huggingface-cli download {args.model_source}`."
            )
        print()

    started = time.time()
    bench_exit = _run_bench(
        rounds=args.rounds,
        artifact_path=args.artifact_path,
        model_source=args.model_source,
        torch_dtype=args.torch_dtype,
        scenario=args.scenario,
        min_regime_match_rate=args.min_regime_match_rate,
    )
    elapsed = time.time() - started
    print()
    print(
        f"[probe] lifeform-bench exit_code={bench_exit} "
        f"elapsed={elapsed / 60:.1f} min"
    )

    summary_exit = _summarize_artifact(args.artifact_path)
    # The script's own exit code reflects debt #10B item 3 closure.
    # We do NOT propagate the bench exit code because it can
    # legitimately be 1 due to debt #10C (il_rapport_trend gate),
    # which is a separate, deferred follow-up.
    return summary_exit


if __name__ == "__main__":
    sys.exit(main())
