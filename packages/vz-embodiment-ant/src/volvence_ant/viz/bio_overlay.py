"""Workstream G3 — overlay simulated curves on published biological data.

Reads the Phase-0 result JSONs (homing precision + route familiarity) and
overlays them against the biological references they were benchmarked to:

- Homing: normalized endpoint error vs journey length, against the AntBot /
  desert-ant scale line (error stays a small fraction of the journey length).
- Familiarity: reducible novelty (``epistemic_magnitude``) vs exposure count,
  against the Ardin 2016 shape (surprise falls monotonically over a few dozen
  route exposures as the mushroom-body familiarity signal saturates).

This does not re-run any experiment; it is a pure read-and-plot over the SSOT
result artifacts, so it is cheap and deterministic.
"""

from __future__ import annotations

import json
import csv
from dataclasses import dataclass
from pathlib import Path

from volvence_ant.viz.render import save_line_overlay


@dataclass(frozen=True)
class BioOverlayReport:
    homing_figure: str | None
    familiarity_figure: str | None
    antbot_reference_ratio: float
    passes_antbot_scale: bool
    familiarity_improved: bool
    first_exposure_novelty: float
    last_exposure_novelty: float
    reference_data_real: bool
    ardin_metric_comparable: bool
    rmse: float | None
    shape_correlation: float | None
    description: str


def _load(results_dir: Path, name: str) -> dict:
    path = results_dir / name
    if not path.exists():
        raise FileNotFoundError(
            f"missing Phase-0 result {path}; run scripts/run_ant_phase0.py first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def build_bio_overlays(
    *,
    results_dir: Path,
    figures_dir: Path,
    reference_dir: Path = Path("research/ant/reference_data"),
) -> BioOverlayReport:
    homing = _load(results_dir, "phase0_homing.json")
    route = _load(results_dir, "phase0_route_learning.json")

    # --- homing overlay ---
    lengths = [pt["journey_length"] for pt in homing["curve"]]
    norm_err = [pt["mean_normalized_error"] for pt in homing["curve"]]
    with (reference_dir / "antbot_homing_2019.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        antbot_rows = list(csv.DictReader(handle))
    point_14m = next(row for row in antbot_rows if row["journey_m"] == "14.0")
    ref_ratio = float(point_14m["mean_error_percent"]) / 100.0
    homing_fig = save_line_overlay(
        series=[
            {"x": lengths, "y": norm_err, "label": "digital ant (sim)", "style": "-o"},
            {"x": lengths, "y": [ref_ratio] * len(lengths),
             "label": f"AntBot scale ({ref_ratio:.3f})", "style": "--"},
        ],
        x_label="journey length",
        y_label="normalized endpoint error",
        title="G3 homing precision vs AntBot scale",
        out_path=figures_dir / "g3_homing_overlay.png",
    )

    # Ardin reports route errors after one-shot storage, not exposure-wise
    # novelty decay. Plot the real Figure-4D summary without inventing a
    # commensurate simulation curve.
    with (reference_dir / "ardin_route_memory_2016.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        ardin_rows = list(csv.DictReader(handle))
    algorithms = [row["algorithm"] for row in ardin_rows]
    errors = [float(row["mean_errors_per_route"]) for row in ardin_rows]
    fam_fig = save_line_overlay(
        series=[
            {
                "x": list(range(len(algorithms))),
                "y": errors,
                "label": "Ardin 2016 Fig 4D reported means",
                "style": "-o",
            },
        ],
        x_label="algorithm index: random, perfect, MB, Infomax",
        y_label="mean errors per ~8m route",
        title="G3 real Ardin 2016 route-memory reference",
        out_path=figures_dir / "g3_familiarity_overlay.png",
    )

    return BioOverlayReport(
        homing_figure=str(homing_fig) if homing_fig else None,
        familiarity_figure=str(fam_fig) if fam_fig else None,
        antbot_reference_ratio=ref_ratio,
        passes_antbot_scale=bool(homing["passes_antbot_scale"]),
        familiarity_improved=bool(route["familiarity_improved"]),
        first_exposure_novelty=float(route["first_exposure_novelty"]),
        last_exposure_novelty=float(route["last_exposure_novelty"]),
        reference_data_real=True,
        ardin_metric_comparable=False,
        rmse=None,
        shape_correlation=None,
        description=(
            f"homing passes_antbot_scale={homing['passes_antbot_scale']} "
            f"(14m AntBot ratio {ref_ratio:.4f}); Ardin reference uses route "
            "errors after one-shot image storage, so exposure-novelty RMSE/"
            "correlation are intentionally not computed."
        ),
    )
