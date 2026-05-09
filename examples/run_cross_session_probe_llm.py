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
    scenarios_path: pathlib.Path | None,
    scenario: str,
    min_regime_match_rate: float,
) -> int:
    """Drive ``lifeform-bench`` programmatically and return its exit code.

    Returns the same exit code the CLI would surface; this script's
    own caller decides how to interpret it.

    When ``scenarios_path`` is supplied it routes through the cli's
    ``--scenarios PATH`` flag (single JSON file or directory of JSON
    files). That flag takes precedence over ``--vertical`` /
    ``--scenario`` for scenario selection but the longitudinal-pass
    builder still keys off ``--vertical companion`` so we get the
    LLM-backed lifeform with shared memory.

    When ``scenarios_path`` is None the cli falls through to its
    own scenario-resolution logic: ``--vertical companion`` always
    wins and runs all 13 vertical scenarios (slow on Qwen). The
    ``--scenario`` flag passed here is a no-op in that path; we
    keep it for forwards-compat with future cli changes.
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
        "--min-regime-match-rate",
        str(min_regime_match_rate),
    ]
    if scenarios_path is not None:
        argv.extend(["--scenarios", str(scenarios_path)])
    else:
        argv.extend(["--scenario", scenario])
    print(f"[probe] invoking lifeform-bench {' '.join(argv)}")
    return bench_main(argv)


def _summarize_artifact(artifact_path: pathlib.Path) -> int:
    """Read the longitudinal artifact and print debt #10B / #10C verdicts.

    Returns 0 when debt #10B item 3 closure conditions are met,
    otherwise 1 — the bench CLI's own exit code is reported separately.

    Closure semantics (option B / per-session runtimes follow-up):
    debt #10B item 3 closes when at least one scenario records a
    cumulative ``per_round_tom_proposal_parsed_ok_total`` or
    ``per_round_common_ground_proposal_parsed_ok_total`` entry > 0,
    i.e. the LLM proposal runtime emitted at least one schema-valid
    typed proposal during the session. Cumulative parsed-ok is the
    correct definition of "EQ chain activated": the legacy
    ``tom_records_total_last`` / ``common_ground_dyad_atoms_total_last``
    fields read only the LAST turn's owner snapshot, which is per-turn
    by design and is not a reliable activation signal on its own.
    Those fields stay in the printed summary as secondary readouts.
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
        # Wave E1 (debt #10B item 3) diagnostic counters: when
        # tom_last == 0, surface the per-round LLM proposal counters
        # so the operator can localise the root cause without
        # re-running.
        if not (isinstance(tom_last, int) and tom_last > 0):
            tom_attempts = scenario.get(
                "per_round_tom_proposal_attempts_total", []
            )
            tom_parsed_ok = scenario.get(
                "per_round_tom_proposal_parsed_ok_total", []
            )
            tom_parse_errors = scenario.get(
                "per_round_tom_proposal_parse_errors_total", []
            )
            tom_schema_mismatches = scenario.get(
                "per_round_tom_proposal_schema_mismatches_total", []
            )
            cg_attempts = scenario.get(
                "per_round_common_ground_proposal_attempts_total", []
            )
            cg_parse_errors = scenario.get(
                "per_round_common_ground_proposal_parse_errors_total", []
            )
            cg_schema_mismatches = scenario.get(
                "per_round_common_ground_proposal_schema_mismatches_total",
                [],
            )
            print(
                "    [diagnostic counters | tom_records_total == 0]\n"
                f"      tom_proposal_attempts        per-round = {tom_attempts}\n"
                f"      tom_proposal_parsed_ok       per-round = {tom_parsed_ok}\n"
                f"      tom_proposal_parse_errors    per-round = {tom_parse_errors}\n"
                f"      tom_proposal_schema_mismatch per-round = {tom_schema_mismatches}\n"
                f"      cg_proposal_attempts         per-round = {cg_attempts}\n"
                f"      cg_proposal_parse_errors     per-round = {cg_parse_errors}\n"
                f"      cg_proposal_schema_mismatch  per-round = {cg_schema_mismatches}"
            )
            if tom_attempts and not any(a or 0 for a in tom_attempts):
                print(
                    "      hint: 0 ToM attempts across all rounds -> the "
                    "LLM proposal runtime was never called. Check that "
                    "`--use-llm-semantic-runtime` is set AND that "
                    "`semantic_proposal_runtime` is a direct "
                    "`LLMSemanticProposalRuntime` instance (not a wrapper)."
                )
            elif tom_parse_errors and any((e or 0) > 0 for e in tom_parse_errors):
                print(
                    "      hint: > 0 parse errors -> model output failed "
                    "JSON parsing. Common cause: model wraps payload in "
                    "a markdown fence variant the parser does not strip "
                    "(only ```json / ```JSON / ``` with alphanumeric info "
                    "string are stripped today). Inspect raw outputs via "
                    "VZ_LLM_PROPOSAL_DEBUG_LOG=<path>."
                )
            elif tom_schema_mismatches and any(
                (m or 0) > 0 for m in tom_schema_mismatches
            ):
                print(
                    "      hint: > 0 schema mismatches -> JSON parsed but "
                    "no item passed schema/confidence floor. Inspect raw "
                    "outputs via VZ_LLM_PROPOSAL_DEBUG_LOG=<path>."
                )
        # Option B (per-session runtimes) closure rule: cumulative
        # parsed-ok across any round is the load-bearing signal. The
        # legacy last-turn snapshot fields (``tom_last`` /
        # ``cg_dyad_atoms_total_last``) remain in the printed summary
        # as secondary readouts but do not gate closure -- per-turn
        # owner snapshots emit only this-turn records, so a 5-turn
        # session whose LAST turn happens to produce no records would
        # falsely look "inactive" even when earlier turns produced
        # valid typed proposals.
        tom_parsed_ok_session = scenario.get(
            "per_round_tom_proposal_parsed_ok_total", []
        )
        cg_parsed_ok_session = scenario.get(
            "per_round_common_ground_proposal_parsed_ok_total", []
        )
        cumulative_parsed_ok = any(
            (n or 0) > 0 for n in tom_parsed_ok_session
        ) or any((n or 0) > 0 for n in cg_parsed_ok_session)
        if cumulative_parsed_ok or (
            isinstance(tom_last, int) and tom_last > 0
        ):
            debt_10b_closed = True
        debt_10c_status.append((sid, il_trend))

    print()
    print("=" * 72)
    print("  Cross-Scenario Summary (Wave E2)")
    print("=" * 72)
    cross = payload.get("cross_scenario_summary") or {}
    if cross:
        print(
            f"  scenarios run: {cross.get('scenario_count', 0)}\n"
            f"  pe_window_filled_scenario_ratio = "
            f"{cross.get('pe_window_filled_scenario_ratio', 0.0):.2f} "
            f"({cross.get('pe_window_filled_scenario_count', 0)} of "
            f"{cross.get('scenario_count', 0)} scenarios)\n"
            f"  il_rapport_trend_snr_mean       = "
            f"{cross.get('il_rapport_trend_snr_mean', 0.0):.3f}"
        )
    else:
        print(
            "  (cross-scenario summary not present in artifact - "
            "this is a pre-Wave-E2 longitudinal artifact)"
        )

    print()
    print("=" * 72)
    print("  Debt verdicts")
    print("=" * 72)
    if debt_10b_closed:
        print(
            "  [10B item 3] CLOSED: at least one scenario produced "
            "cumulative per_round_tom_proposal_parsed_ok_total > 0 "
            "or per_round_common_ground_proposal_parsed_ok_total > 0; "
            "the LLM proposal runtime emitted schema-valid typed "
            "proposals during the session, so the EQ owner activation "
            "chain is wired and live."
        )
    else:
        print(
            "  [10B item 3] OPEN: every scenario reported zero "
            "cumulative parsed_ok proposals across all rounds for both "
            "ToM and common-ground runtimes. Likely causes (in order "
            "of likelihood): (a) the LLM proposal runtime is wired but "
            "every payload failed the strict JSON schema (inspect raw "
            "outputs via VZ_LLM_PROPOSAL_DEBUG_LOG=<path>); (b) the "
            "scenario is too short or stylistically unlikely to elicit "
            "ToM/CG observations (try a long-form scenario from "
            "packages/lifeform-domain-emogpt/.../scenarios/long-form-*); "
            "(c) the runtime was never called at all (per_round_*_"
            "proposal_attempts_total is empty or all-zero -- check "
            "--use-llm-semantic-runtime is set and that "
            "semantic_proposal_runtime is a direct "
            "LLMSemanticProposalRuntime instance, not a wrapper)."
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
            'Built-in scenario name (only used when --scenarios-path is unset '
            'AND --vertical is omitted). Default: low-mood-disclosure. '
            "Note: when --vertical companion is set (which the LLM-runtime "
            "path requires), the cli ignores --scenario and runs all 13 "
            "vertical scenarios. Use --scenarios-path to restrict scope."
        ),
    )
    parser.add_argument(
        "--scenarios-path",
        type=pathlib.Path,
        default=None,
        help=(
            "Path to a single scenario JSON file (or directory of them). "
            "When set, takes precedence over --scenario and over the "
            "vertical's full scenario set, so the longitudinal pass runs "
            "ONLY this scenario across N rounds. Recommended for the "
            "evidence baseline run: point at "
            "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/"
            "scenarios/cross-session-emotional-followup.json (the "
            "scenario shipped explicitly for cross-session evidence)."
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
    print(f"  scenarios_path={args.scenarios_path}")
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
        scenarios_path=args.scenarios_path,
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
