"""CursorRunner: wraps Cursor best-of-n-runner subagents as a Runner backend.

Each (cell, seed) unit is dispatched to an independent Cursor subagent running
in its own git worktree. The subagent executes:

    python -m volvence_labs.cli run --unit --probe <id> --cell <cell> --seed <seed> \
        --wiring <level> --root <worktree_root>

and returns the run_id + readouts sha. The parent CursorRunner aggregates all
unit results into an ExperimentReport.

Fallback: if Cursor SDK is unavailable or subagent launch fails, CursorRunner
degrades to local subprocess execution (same CLI command, no worktree isolation).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..probe import BaseProbe, GateReport, RunOutcome, get_registry
from ..probe.types import ReadoutBundle
from ..scheduler.runner import (
    ExperimentReport,
    UnitReport,
    _build_report,
    _ensure_probes_imported,
    _unit_from_jsonable,
)
from ..snapshot import CASStore, RunLog, default_paths, LabsPaths
from ..wiring import AblationCell, WiringLevel, WiringProfile


@dataclass
class CursorRunnerConfig:
    """Configuration for CursorRunner."""
    max_concurrent: int = 8
    timeout_per_unit: int = 3600
    use_worktrees: bool = False  # stage 1: start without worktrees, add later
    fallback_to_subprocess: bool = True


class CursorRunner:
    """Runner backend that dispatches units to Cursor subagents or subprocesses.

    In stage 1, this primarily uses subprocess-based execution of the CLI
    --unit command. Full Cursor SDK integration (best-of-n-runner with
    worktrees) is layered on top when the SDK is available.
    """

    def __init__(
        self,
        root: Optional[str] = None,
        config: Optional[CursorRunnerConfig] = None,
    ):
        self.paths: LabsPaths = default_paths(root)
        self.config = config or CursorRunnerConfig()
        self._cursor_sdk_available: Optional[bool] = None

    def run(self, probe_id: str, profile: WiringProfile) -> ExperimentReport:
        """Run all units for a probe under the given profile.

        Dispatches each (cell, seed) as an independent subprocess (or Cursor
        subagent if SDK is available and worktrees are enabled).
        """
        _ensure_probes_imported()
        level = profile.level_for(probe_id)
        units_args = [
            (probe_id, cell, seed, level)
            for cell in profile.cells
            for seed in profile.seeds
        ]

        if self._should_use_cursor_sdk():
            results = self._run_via_cursor_sdk(units_args)
        else:
            results = self._run_via_subprocess(units_args)

        units = [_unit_from_jsonable(r) for r in results]
        return _build_report(probe_id, profile, units)

    def _should_use_cursor_sdk(self) -> bool:
        """Check if Cursor SDK is available."""
        if self._cursor_sdk_available is None:
            try:
                import cursor_sdk  # noqa: F401
                self._cursor_sdk_available = True
            except ImportError:
                self._cursor_sdk_available = False
        return self._cursor_sdk_available and self.config.use_worktrees

    def _run_via_subprocess(
        self,
        units_args: list[tuple[str, AblationCell, int, WiringLevel]],
    ) -> list[dict]:
        """Run units as parallel subprocesses using CLI --unit mode."""
        import concurrent.futures

        results: list[dict] = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.config.max_concurrent
        ) as pool:
            futures = {
                pool.submit(
                    _subprocess_unit,
                    probe_id,
                    cell.value,
                    seed,
                    level.value,
                    str(self.paths.root),
                ): (probe_id, cell, seed)
                for probe_id, cell, seed, level in units_args
            }
            for future in concurrent.futures.as_completed(futures):
                probe_id, cell, seed = futures[future]
                try:
                    result = future.result(timeout=self.config.timeout_per_unit)
                    results.append(result)
                except Exception as e:
                    results.append({
                        "run_id": "",
                        "probe_id": probe_id,
                        "cell": cell.value,
                        "seed": seed,
                        "wiring": "shadow",
                        "ok": False,
                        "error": str(e),
                        "readouts": {},
                        "metrics": {},
                        "manifest_sha": "",
                        "output_sha": "",
                        "readouts_sha": "",
                        "input_sha": "",
                        "knobs_sha": "",
                        "duration_s": 0.0,
                    })
        return results

    def _run_via_cursor_sdk(
        self,
        units_args: list[tuple[str, AblationCell, int, WiringLevel]],
    ) -> list[dict]:
        """Placeholder for full Cursor SDK integration.

        When implemented, this will:
        1. Create a git worktree per unit.
        2. Launch a best-of-n-runner subagent per unit.
        3. Collect results from subagent output files.
        4. Freeze worktree branches.

        For now, falls back to subprocess execution.
        """
        return self._run_via_subprocess(units_args)


def _subprocess_unit(
    probe_id: str,
    cell_value: str,
    seed: int,
    level_value: str,
    root_str: str,
) -> dict:
    """Execute a single unit via subprocess CLI --unit command.

    Returns the parsed JSON output (UnitReport-compatible dict).
    """
    cmd = [
        sys.executable, "-m", "volvence_labs.cli",
        "--root", root_str,
        "run", "--unit",
        "--probe", probe_id,
        "--cell", cell_value,
        "--seed", str(seed),
        "--wiring", level_value,
        "--json",
    ]

    env = os.environ.copy()
    # Source root: where volvence_labs package lives (one level up from framework/).
    _pkg_dir = Path(__file__).resolve().parent.parent.parent  # src/volvence_labs/framework/parallel -> src/
    _src_root = str(_pkg_dir.parent)  # -> src/
    env["PYTHONPATH"] = _src_root + os.pathsep + env.get("PYTHONPATH", "")
    env["VOLVENCE_LABS_ROOT"] = root_str

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
            env=env,
            cwd=root_str,
        )
        duration = time.time() - start

        if result.returncode != 0:
            err_msg = result.stderr[-2000:] if result.stderr else ""
            if not err_msg and result.stdout:
                err_msg = result.stdout[-2000:]
            if not err_msg:
                err_msg = f"exit code {result.returncode}"
            return {
                "run_id": "",
                "probe_id": probe_id,
                "cell": cell_value,
                "seed": seed,
                "wiring": level_value,
                "ok": False,
                "error": err_msg,
                "readouts": {},
                "metrics": {},
                "manifest_sha": "",
                "output_sha": "",
                "readouts_sha": "",
                "input_sha": "",
                "knobs_sha": "",
                "duration_s": duration,
            }

        output = json.loads(result.stdout)
        return output

    except subprocess.TimeoutExpired:
        return {
            "run_id": "",
            "probe_id": probe_id,
            "cell": cell_value,
            "seed": seed,
            "wiring": level_value,
            "ok": False,
            "error": "subprocess timeout",
            "readouts": {},
            "metrics": {},
            "manifest_sha": "",
            "output_sha": "",
            "readouts_sha": "",
            "input_sha": "",
            "knobs_sha": "",
            "duration_s": time.time() - start,
        }
    except Exception as e:
        return {
            "run_id": "",
            "probe_id": probe_id,
            "cell": cell_value,
            "seed": seed,
            "wiring": level_value,
            "ok": False,
            "error": str(e),
            "readouts": {},
            "metrics": {},
            "manifest_sha": "",
            "output_sha": "",
            "readouts_sha": "",
            "input_sha": "",
            "knobs_sha": "",
            "duration_s": time.time() - start,
        }
