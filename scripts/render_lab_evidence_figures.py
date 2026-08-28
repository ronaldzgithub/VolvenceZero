"""Render the Coding Lab / Relationship Lab evidence figures.

Every number on every figure is read from a frozen artifact at render time. Each
figure emits an SVG (documents, print), a PNG (decks, chat) and a sidecar JSON
recording the plotted values plus the sha256 of each source artifact, so a
reviewer can recompute any figure directly from evidence.

    python scripts/render_lab_evidence_figures.py
    python scripts/render_lab_evidence_figures.py --only 01 03
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from matplotlib import pyplot as plt

from lab_figures import coding, honest, html_page, relationship, sources, style

OUTPUT_DIR = sources.REPO_ROOT / "docs" / "business" / "BP" / "figures"

FIGURES = (
    ("01", "coding-convention-learning", coding.figure_convention_learning),
    ("02", "coding-memory-pareto", coding.figure_memory_pareto),
    ("03", "coding-when-to-steer", coding.figure_when_to_steer),
    ("04", "relationship-mirror-pair", relationship.figure_mirror_pair),
    ("05", "relationship-outcome-composition", relationship.figure_outcome_composition),
    ("06", "relationship-longitudinal-timeline", relationship.figure_longitudinal_timeline),
    ("07", "honest-scoreboard", honest.figure_honest_scoreboard),
    ("08", "honest-learnable-degeneracy", honest.figure_learnable_degeneracy),
    ("09", "coding-cost-vs-session-length", coding.figure_cost_vs_session_length),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="INDEX",
        help="render only these figure indices (e.g. --only 01 04)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PNG resolution (default: 200)",
    )
    return parser.parse_args(argv)


def render(indices: set[str] | None, dpi: int) -> list[dict]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    family = style.apply()
    rendered: list[dict] = []

    for index, slug, builder in FIGURES:
        if indices is not None and index not in indices:
            continue
        stem = f"fig{index}-{slug}"
        figure, payload = builder()
        svg_path = OUTPUT_DIR / f"{stem}.svg"
        png_path = OUTPUT_DIR / f"{stem}.png"
        figure.savefig(svg_path, format="svg")
        figure.savefig(png_path, format="png", dpi=dpi)
        plt.close(figure)

        sidecar = {
            "figure_index": index,
            "slug": slug,
            "rendered_by": "scripts/render_lab_evidence_figures.py",
            "font_family": family,
            **payload,
        }
        sidecar_path = OUTPUT_DIR / f"{stem}.json"
        sidecar_path.write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=1, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        rendered.append(
            {
                "figure_index": index,
                "slug": slug,
                "svg": svg_path.name,
                "png": png_path.name,
                "sidecar": sidecar_path.name,
                "claim": payload["claim"],
                "evidence_tier": payload["evidence_tier"],
                "provenance": payload["provenance"],
            }
        )
        print(f"  [{index}] {stem}  ->  svg + png + json")

    return rendered


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    indices = set(args.only) if args.only else None
    print(f"rendering evidence figures into {OUTPUT_DIR.relative_to(sources.REPO_ROOT)}")
    rendered = render(indices, args.dpi)
    if not rendered:
        raise SystemExit("no figures matched --only; nothing rendered")

    if indices is not None:
        print("  (partial render: manifest and HTML page left untouched)")
        return 0

    rendered_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = {
        "schema_version": "lab-evidence-figures-manifest.v1",
        "rendered_at_utc": rendered_at,
        "discipline": (
            "Every plotted number is read from a frozen artifact at render time. "
            "Figures must never be hand-edited; re-run this script instead."
        ),
        "figures": rendered,
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"  manifest.json  ({len(rendered)} figures)")

    page = html_page.build(OUTPUT_DIR, rendered_at)
    page_path = OUTPUT_DIR / "index.html"
    page_path.write_text(page, encoding="utf-8")
    print(f"  index.html  ({len(page) / 1024:.0f} KB, self-contained)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
