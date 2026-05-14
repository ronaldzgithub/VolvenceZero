"""Contract test: D3 boundary baseline + drive ablation pipeline (#64 / #68).

Validates:

1. ``growth_advisor_boundary_eval.py --mode fake-judge --fake-judge oracle``
   produces a typed report with all required fields per
   ``docs/specs/growth-advisor-boundary-baseline.md`` §6.
2. The oracle judge yields perfect SLA (precision/recall = 1 on
   the GT subset where the boundary is the expected one).
3. ``--fake-judge silent`` deliberately fails SLA so the pipeline
   surfaces the negative case.
4. ``growth_advisor_drive_ablation.py --mode fake-judge`` produces
   the per-condition × per-boundary matrix; ``no-restraint`` cuts
   ``bp-no-hard-sell`` rate enough to clear the 30% decrease SLA.
5. Both scripts' typed schemas match the spec field set (drift-fail).

Refs:

* docs/known-debts.md #64 / #68
* docs/specs/growth-advisor-boundary-baseline.md §6 / §7
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from types import ModuleType


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"


def _load_script(filename: str) -> ModuleType:
    path = _SCRIPTS_DIR / filename
    spec = importlib.util.spec_from_file_location(
        f"_growth_advisor_d3_{path.stem}", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# #64 boundary baseline pipeline
# ---------------------------------------------------------------------------


def test_boundary_eval_oracle_judge_perfect_sla(tmp_path: pathlib.Path) -> None:
    boundary = _load_script("growth_advisor_boundary_eval.py")
    rc = boundary.main(
        [
            "--mode", "fake-judge",
            "--fake-judge", "oracle",
            "--output-dir", str(tmp_path),
        ]
    )
    assert rc == 0
    files = list(tmp_path.glob("boundary_baseline_*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    for field in (
        "report_kind",
        "report_mode",
        "judge_label",
        "n_scenarios",
        "per_boundary",
        "per_archetype",
        "kill_criteria_band",
        "sla_thresholds",
        "sla_pass_per_boundary",
        "sla_pass_overall",
    ):
        assert field in payload, f"boundary baseline missing field {field!r}"
    assert payload["report_kind"] == "boundary_baseline"
    assert payload["judge_label"] == "fake-judge:oracle"
    # Oracle: every expected boundary fires correctly → precision/recall
    # are 1.0 on boundaries that have at least one GT entry.
    for boundary_id, stats in payload["per_boundary"].items():
        if stats["tp"] + stats["fn"] > 0:
            assert stats["recall"] == 1.0
        if stats["tp"] + stats["fp"] > 0:
            assert stats["precision"] == 1.0


def test_boundary_eval_silent_judge_fails_sla(tmp_path: pathlib.Path) -> None:
    boundary = _load_script("growth_advisor_boundary_eval.py")
    rc = boundary.main(
        [
            "--mode", "fake-judge",
            "--fake-judge", "silent",
            "--output-dir", str(tmp_path),
        ]
    )
    assert rc == 0
    files = list(tmp_path.glob("boundary_baseline_*.json"))
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    # Silent judge fires nothing → trigger_rate=0 falls below kill band
    # for every boundary (kill band starts at 0.05).
    for stats in payload["per_boundary"].values():
        assert stats["trigger_rate"] == 0.0
        assert stats["in_kill_band"] is False
    assert payload["sla_pass_overall"] is False


# ---------------------------------------------------------------------------
# #68 drives ablation pipeline
# ---------------------------------------------------------------------------


def test_drive_ablation_no_restraint_clears_sla(tmp_path: pathlib.Path) -> None:
    ablation = _load_script("growth_advisor_drive_ablation.py")
    rc = ablation.main(
        [
            "--mode", "fake-judge",
            "--output-dir", str(tmp_path),
        ]
    )
    assert rc == 0
    files = list(tmp_path.glob("drive_ablation_*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    # All 4 conditions must be present.
    for condition in ("full", "no-restraint", "no-empathy", "no-trust"):
        assert condition in payload["per_condition"]
        rates = payload["per_condition"][condition]["boundary_trigger_rates"]
        for boundary in (
            "bp-no-hard-sell",
            "bp-no-overclaim",
            "bp-no-flooding",
            "bp-no-judgmental",
        ):
            assert boundary in rates
    # no-restraint must drop bp-no-hard-sell vs full by ≥ 30% relative.
    check = payload["ablation_check_no_restraint_vs_full"]
    assert check["sla_min_relative_decrease"] == 0.30
    assert check["bp-no-hard-sell_relative_decrease"] >= 0.30
    assert check["sla_pass"] is True


def test_drive_ablation_dry_run_emits_placeholder(tmp_path: pathlib.Path) -> None:
    ablation = _load_script("growth_advisor_drive_ablation.py")
    rc = ablation.main(
        [
            "--mode", "dry-run",
            "--output-dir", str(tmp_path),
        ]
    )
    assert rc == 0
    files = list(tmp_path.glob("drive_ablation_*.json"))
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["report_mode"] == "dry-run"
    # Real-run fields stay None until ACTIVE writes real measurements.
    for condition in payload["per_condition"].values():
        for value in condition["boundary_trigger_rates"].values():
            assert value is None
