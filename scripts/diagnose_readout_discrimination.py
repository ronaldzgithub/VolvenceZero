"""Instrument-discrimination pre-check for formal primary-criterion readouts.

Read-only diagnostic (方案 §0 不变量 7): before any readout may be frozen as a
formal primary criterion, this script measures whether the readout can
distinguish the clusters the criterion is supposed to distinguish. It reports,
for the v1 raw readout and the v2 centered readout side by side:

- common-mode energy share (fraction of per-sample energy along the corpus
  mean direction);
- same-cluster vs different-cluster cosine gap with Cohen's d;
- 1-NN cluster retrieval accuracy against chance.

The v2 transform is never re-implemented here: fitting and application go
through the `vz-substrate` owner (`fit_forward_readout_reference_statistics`,
`SubstrateReadoutReferenceStatistics.apply`). This file is intentionally NOT
part of any formal runner's frozen SOURCE_FILES; it reads collected context
checkpoints and writes only its own report.

Prereg discipline: run this on a frozen train-split corpus only. Heldout /
validation data must not feed the fit or the go/no-go reading.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
for wheel in ("vz-contracts", "vz-substrate"):
    candidate = REPOSITORY_ROOT / "packages" / wheel / "src"
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from volvence_zero.substrate import (  # noqa: E402
    SUBSTRATE_FORWARD_CENTERED_READOUT_KIND,
    SUBSTRATE_FORWARD_READOUT_KIND,
    fit_forward_readout_reference_statistics,
    layer_normalized_readout_vector,
)

REPORT_SCHEMA_VERSION = "readout-discrimination-diagnostic.v1"


def _load_clustered_contexts(
    contexts_dir: Path,
) -> tuple[tuple[int, ...], tuple[int, ...], list[tuple[str, tuple[float, ...]]]]:
    """Load (cluster_id, v1 context values) rows from collected checkpoints."""

    files = sorted(contexts_dir.glob("*.json.gz"))
    if not files:
        raise FileNotFoundError(
            f"no collected context checkpoints (*.json.gz) under {contexts_dir}"
        )
    layer_indices: tuple[int, ...] = ()
    activation_widths: tuple[int, ...] = ()
    rows: list[tuple[str, tuple[float, ...]]] = []
    for path in files:
        payload = json.loads(gzip.open(path, "rt", encoding="utf-8").read())
        samples = payload["samples"]
        if not samples:
            raise ValueError(f"context checkpoint {path.name} contains no samples")
        for sample in samples:
            context = sample["context"]
            geometry = (
                tuple(int(v) for v in context["layer_indices"]),
                tuple(int(v) for v in context["activation_widths"]),
            )
            if not layer_indices:
                layer_indices, activation_widths = geometry
            elif geometry != (layer_indices, activation_widths):
                raise ValueError(
                    f"residual geometry drift inside {path.name}: "
                    f"{geometry} != {(layer_indices, activation_widths)}"
                )
            rows.append((path.stem, tuple(float(v) for v in context["values"])))
    return layer_indices, activation_widths, rows


def _discrimination_metrics(
    matrix: "list[list[float]]", clusters: list[str]
) -> dict[str, float]:
    import numpy as np

    data = np.asarray(matrix, dtype=np.float64)
    labels = np.asarray(clusters)
    unit = data / np.linalg.norm(data, axis=1, keepdims=True)

    mean_vector = unit.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_vector))
    if mean_norm > 1e-12:
        mean_direction = mean_vector / mean_norm
        common_mode_energy = float(np.mean((unit @ mean_direction) ** 2))
    else:
        common_mode_energy = 0.0

    gram = unit @ unit.T
    off_diagonal = ~np.eye(len(unit), dtype=bool)
    same = off_diagonal & (labels[:, None] == labels[None, :])
    different = labels[:, None] != labels[None, :]
    if not same.any() or not different.any():
        raise ValueError(
            "discrimination metrics need at least two clusters with two "
            "samples each"
        )
    same_values = gram[same]
    different_values = gram[different]
    pooled = math.sqrt(0.5 * (same_values.var() + different_values.var()))
    gap = float(same_values.mean() - different_values.mean())
    cohens_d = gap / pooled if pooled > 1e-12 else 0.0

    masked = gram.copy()
    np.fill_diagonal(masked, -np.inf)
    nearest = labels[masked.argmax(axis=1)]
    retrieval_accuracy = float((nearest == labels).mean())
    chance = float(
        np.mean(
            [
                (labels == label).sum() / len(labels)
                for label in np.unique(labels)
            ]
        )
    )

    return {
        "sample_count": int(len(unit)),
        "cluster_count": int(len(np.unique(labels))),
        "common_mode_energy_share": common_mode_energy,
        "pairwise_cosine_mean": float(gram[off_diagonal].mean()),
        "pairwise_cosine_std": float(gram[off_diagonal].std()),
        "same_cluster_cosine_mean": float(same_values.mean()),
        "different_cluster_cosine_mean": float(different_values.mean()),
        "same_vs_different_gap": gap,
        "cohens_d": float(cohens_d),
        "nn_retrieval_accuracy": retrieval_accuracy,
        "nn_retrieval_chance": chance,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--contexts-dir",
        type=Path,
        required=True,
        help=(
            "Directory of collected *.json.gz context checkpoints (one file "
            "per cluster/dyad); must be a frozen train-split corpus"
        ),
    )
    parser.add_argument(
        "--corpus-id",
        required=True,
        help="Frozen reference corpus identifier recorded in the fitted statistics",
    )
    parser.add_argument(
        "--principal-components",
        type=int,
        default=1,
        help="Frozen principal directions to remove in the v2 fit (default 1)",
    )
    parser.add_argument(
        "--power-iterations",
        type=int,
        default=60,
        help="Deterministic power-iteration count for the v2 fit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the JSON report",
    )
    parser.add_argument(
        "--statistics-output",
        type=Path,
        default=None,
        help="Optional path to persist the fitted reference statistics payload",
    )
    parser.add_argument(
        "--min-cohens-d",
        type=float,
        default=None,
        help=(
            "Optional gate: exit 2 unless the v2 Cohen's d reaches this "
            "preregistered floor"
        ),
    )
    args = parser.parse_args()

    layer_indices, activation_widths, rows = _load_clustered_contexts(
        args.contexts_dir.resolve()
    )
    clusters = [cluster for cluster, _ in rows]
    v1_vectors = [list(values) for _, values in rows]

    print(
        f"[diagnose] {len(rows)} samples / {len(set(clusters))} clusters, "
        f"layers {layer_indices} widths {activation_widths}",
        flush=True,
    )
    print("[diagnose] fitting v2 reference statistics (substrate owner)...", flush=True)
    statistics = fit_forward_readout_reference_statistics(
        corpus_id=args.corpus_id,
        layer_indices=layer_indices,
        activation_widths=activation_widths,
        vectors=tuple(tuple(vector) for vector in v1_vectors),
        principal_component_count=args.principal_components,
        power_iterations=args.power_iterations,
    )
    v2_vectors = [
        list(
            statistics.apply(
                layer_normalized_readout_vector(
                    tuple(vector), activation_widths=activation_widths
                )
            )
        )
        for vector in v1_vectors
    ]

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "contexts_dir": str(args.contexts_dir.resolve()),
        "corpus_id": args.corpus_id,
        "layer_indices": list(layer_indices),
        "activation_widths": list(activation_widths),
        "reference_statistics_sha256": statistics.statistics_sha256,
        "principal_components_removed": args.principal_components,
        "readouts": {
            SUBSTRATE_FORWARD_READOUT_KIND: _discrimination_metrics(
                v1_vectors, clusters
            ),
            SUBSTRATE_FORWARD_CENTERED_READOUT_KIND: _discrimination_metrics(
                v2_vectors, clusters
            ),
        },
        "prereg_note": (
            "train-split diagnostic only; heldout data must not feed the fit "
            "or the go/no-go reading, and thresholds are frozen at prereg time"
        ),
    }

    for kind, metrics in report["readouts"].items():
        print(f"--- {kind}")
        for key, value in metrics.items():
            print(f"    {key}: {value:.6f}" if isinstance(value, float) else f"    {key}: {value}")

    if args.statistics_output is not None:
        args.statistics_output.write_text(
            json.dumps(statistics.to_payload(), indent=2) + "\n", encoding="utf-8"
        )
        print(f"[diagnose] statistics -> {args.statistics_output}")
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[diagnose] report -> {args.output}")

    if args.min_cohens_d is not None:
        v2_d = report["readouts"][SUBSTRATE_FORWARD_CENTERED_READOUT_KIND]["cohens_d"]
        if v2_d < args.min_cohens_d:
            print(
                f"[diagnose] GATE FAIL: v2 cohens_d {v2_d:.4f} < "
                f"preregistered floor {args.min_cohens_d:.4f}"
            )
            return 2
        print(
            f"[diagnose] gate pass: v2 cohens_d {v2_d:.4f} >= "
            f"{args.min_cohens_d:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
