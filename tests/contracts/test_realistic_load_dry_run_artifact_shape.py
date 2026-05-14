"""Contract test: realistic_load_*.py dry-run artifact shape (debt #45).

The three ``scripts/realistic_load_{companion,figure,growth_advisor}.py``
SHADOW scaffolds emit a placeholder JSON artifact when invoked with
``--dry-run``. This contract pins the field set + ensures
``expected_slo`` / ``sample_shape`` blocks stay aligned with the
SLO baseline spec so customer / ops review of the artifact matches
the SLO doc verbatim.

Refs:

* docs/specs/perf-baseline.md
* docs/known-debts.md #45
* docs/moving forward/cross-cutting-foundation-packet.md §2.1
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"


_REQUIRED_TOP_LEVEL_FIELDS = frozenset(
    {
        "scaffold_status",
        "vertical",
        "dry_run",
        "expected_slo",
        "sample_shape",
        "errors",
        "notes",
    }
)


def _load_script(filename: str) -> ModuleType:
    """Import ``scripts/<filename>`` as a module without ``conftest`` magic."""

    path = _SCRIPTS_DIR / filename
    spec = importlib.util.spec_from_file_location(
        f"_realistic_load_{path.stem}", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_dry(script_filename: str, output_dir: Path) -> dict:
    module = _load_script(script_filename)
    rc = module.main(["--dry-run", "--output-dir", str(output_dir)])
    assert rc == 0, f"{script_filename} --dry-run exited rc={rc}"
    files = list(output_dir.glob("*.json"))
    assert len(files) == 1, (
        f"{script_filename} should emit exactly one artifact under "
        f"{output_dir!r}; got {[p.name for p in files]}"
    )
    return json.loads(files[0].read_text(encoding="utf-8"))


def test_companion_dry_run_artifact_shape(tmp_path: Path) -> None:
    payload = _run_dry("realistic_load_companion.py", tmp_path)
    assert payload["scaffold_status"] == "SHADOW"
    assert payload["vertical"] == "companion"
    assert payload["dry_run"] is True
    missing = _REQUIRED_TOP_LEVEL_FIELDS - payload.keys()
    assert not missing, f"companion missing required fields: {sorted(missing)}"
    # Sample shape is review-aid only; must self-identify.
    assert payload["sample_shape"]["is_sample"] is True
    # SLO targets pinned to perf-baseline.md §2.
    assert payload["expected_slo"]["p50_turn_latency_s_max"] == 1.5
    assert payload["expected_slo"]["p99_turn_latency_s_max"] == 3.0
    assert payload["expected_slo"]["concurrent_ai_id_min"] == 50
    # Real-run fields stay None until ACTIVE writes real measurements.
    assert payload["p50_turn_latency_s"] is None
    assert payload["p99_turn_latency_s"] is None


def test_figure_dry_run_artifact_shape(tmp_path: Path) -> None:
    payload = _run_dry("realistic_load_figure.py", tmp_path)
    assert payload["scaffold_status"] == "SHADOW"
    assert payload["vertical"] == "figure"
    missing = _REQUIRED_TOP_LEVEL_FIELDS - payload.keys()
    assert not missing, f"figure missing required fields: {sorted(missing)}"
    # Figure-specific SLO fields.
    assert payload["expected_slo"]["l3_citation_rate_min"] == 0.95
    assert payload["expected_slo"]["l4_refusal_rate_min"] == 0.85
    # Real-run fields stay None.
    assert payload["l3_citation_rate"] is None
    assert payload["l4_refusal_rate"] is None
    assert payload["lora_swap_overhead_p50_ms"] is None


def test_growth_advisor_dry_run_artifact_shape(tmp_path: Path) -> None:
    payload = _run_dry("realistic_load_growth_advisor.py", tmp_path)
    assert payload["scaffold_status"] == "SHADOW"
    assert payload["vertical"] == "growth_advisor"
    missing = _REQUIRED_TOP_LEVEL_FIELDS - payload.keys()
    assert not missing, (
        f"growth_advisor missing required fields: {sorted(missing)}"
    )
    # Boundary trigger band (G-A kill criteria) pinned per spec.
    band = payload["expected_slo"]["boundary_trigger_rate_band"]
    assert band == [0.05, 0.50]
    # 4 anti-sales boundaries enumerated; real values still None.
    assert set(payload["boundary_trigger_rates"]) == {
        "bp-no-hard-sell",
        "bp-no-overclaim",
        "bp-no-flooding",
        "bp-no-judgmental",
    }
    for value in payload["boundary_trigger_rates"].values():
        assert value is None


def test_dry_run_real_metric_fields_remain_none_until_active() -> None:
    """SLO real-run fields must be ``None`` in dry-run artifacts.

    Catches the common regression of "I added a sample value into the
    p99 field for testing" — sample data must live under
    ``sample_shape``, never alongside the real-run None placeholders.
    """

    import tempfile

    for script in (
        "realistic_load_companion.py",
        "realistic_load_figure.py",
        "realistic_load_growth_advisor.py",
    ):
        with tempfile.TemporaryDirectory() as out:
            payload = _run_dry(script, Path(out))
        assert payload["dry_run"] is True
        assert payload["scaffold_status"] == "SHADOW"
        assert payload["p50_turn_latency_s"] is None
        assert payload["p99_turn_latency_s"] is None
