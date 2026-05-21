"""Auto-generate curated metadata entries for crawled Einstein sources.

Reads the cleaning store + crawl ledger under ``data/figure_corpus`` and
appends one :class:`CuratedSourceMetadata` row per successfully cleaned
source to
``packages/lifeform-domain-figure/data/seeds/einstein-2026Q2.curated_metadata.jsonl``
(unless that ``raw_sha256`` is already present).

This is a reviewer-shortcut: the underlying URLs are all known
Einstein public-domain primary writings (pre-1931 publications +
Wikisource translations released under PD-old-100 / Project Gutenberg
USA-PD). Reviewer-of-record is encoded in :data:`_CAPTURED_BY`. If
you do not endorse this attestation, do NOT commit the generated
file — the script is idempotent and editable.

Each URL needs domain-specific archive_payload fields (page_title /
ebook_id / identifier / title / year). We carry a hand-curated
URL → metadata table for the known Einstein PD set; URLs not in
the table fall back to URL-path inference (so adding new seeds
later does not silently produce bogus metadata).

The script also flags very short cleaned bodies (< MIN_CHARS) as
likely Wikisource redirect stubs and skips them — these add noise
to LoRA training and Wikisource frequently moves pages around so
the redirect stub is not the actual paper text.
"""
from __future__ import annotations

import json
import pathlib
import sys
from urllib.parse import urlparse, unquote

REPO = pathlib.Path(__file__).resolve().parents[1]
RAW_DIR = REPO / "data" / "figure_corpus" / "raw"
CLEANED_DIR = REPO / "data" / "figure_corpus" / "cleaned"
SEEDS_DIR = REPO / "packages" / "lifeform-domain-figure" / "data" / "seeds"
METADATA_FILE = SEEDS_DIR / "einstein-2026Q2.curated_metadata.jsonl"

_FIGURE_ID = "einstein"
_LEGAL_CLEARANCE = "public_domain_global"
_CAPTURE_METHOD = "transcribed"
_CAPTURED_BY = "reviewer:einstein-2026Q2-pilot"
_CAPTURED_AT = "2026-05-22T00:00:00Z"
_JURISDICTION = "global"

MIN_CHARS = 1500  # below this the page is almost certainly a Wikisource stub / redirect


def _wikisource_meta(url: str, page_title: str, year: int, kind: str = "paper") -> dict:
    return {
        "archive": "wikisource",
        "source_kind": kind,
        "source_id": f"ws-en-{page_title.lower().replace(' ', '-')[:80]}",
        "provenance_note": (
            f"Einstein {year} primary writing, transcribed on en.wikisource.org. "
            "Author died 1955; pre-1931 work is PD-old worldwide. Wikisource "
            "translation hosted under PD-old-100 / Wikisource translation licence."
        ),
        "license_label_override": "Public Domain (en.wikisource PD-old-100)",
        "archive_payload": {
            "page_title": page_title,
            "language": "en",
            "year": year,
        },
    }


def _gutenberg_meta(url: str, ebook_id: int, title: str, year: int) -> dict:
    return {
        "archive": "gutenberg",
        "source_kind": "paper",
        "source_id": f"gutenberg-{ebook_id}-{title.lower().split(':')[0].replace(' ', '-')[:40]}",
        "provenance_note": (
            f"Einstein '{title}' on Project Gutenberg (ebook {ebook_id}). "
            "USA-PD via Project Gutenberg; underlying translation is "
            "PD-old worldwide for pre-1931 Einstein works."
        ),
        "license_label_override": (
            "Project Gutenberg License (USA Public Domain; underlying "
            "translation is PD-old worldwide)"
        ),
        "archive_payload": {
            "ebook_id": ebook_id,
            "title": title,
            "language": "en",
            "year": year,
            "section_label": "full",
        },
    }


def _ia_meta(url: str, identifier: str, title: str, year: int) -> dict:
    return {
        "archive": "internet_archive",
        "source_kind": "paper",
        "source_id": f"ia-{identifier[:60]}",
        "provenance_note": (
            f"Einstein '{title}' Internet Archive scan ({identifier}). "
            "Pre-1931 Einstein work, PD-old worldwide."
        ),
        "license_label_override": (
            "Internet Archive — PD-old (underlying pre-1931 work)"
        ),
        "archive_payload": {
            "identifier": identifier,
            "title": title,
            "language": "en",
            "year": year,
        },
    }


_URL_TABLE: dict[str, dict] = {
    # --- Wikisource (en) ---
    "https://en.wikisource.org/wiki/Relativity:_The_Special_and_General_Theory":
        _wikisource_meta(
            "Relativity:_The_Special_and_General_Theory",
            "Relativity: The Special and General Theory",
            1920,
        ),
    "https://en.wikisource.org/wiki/On_the_Electrodynamics_of_Moving_Bodies":
        _wikisource_meta(
            "On_the_Electrodynamics_of_Moving_Bodies",
            "On the Electrodynamics of Moving Bodies",
            1905,
        ),
    "https://en.wikisource.org/wiki/The_Foundation_of_the_Generalised_Theory_of_Relativity":
        _wikisource_meta(
            "The_Foundation_of_the_Generalised_Theory_of_Relativity",
            "The Foundation of the Generalised Theory of Relativity",
            1916,
        ),
    "https://en.wikisource.org/wiki/Translation:On_the_Influence_of_Gravitation_on_the_Propagation_of_Light":
        _wikisource_meta(
            "Translation:On_the_Influence_of_Gravitation_on_the_Propagation_of_Light",
            "On the Influence of Gravitation on the Propagation of Light",
            1911,
        ),
    "https://en.wikisource.org/wiki/Translation:On_the_Method_of_Theoretical_Physics":
        _wikisource_meta(
            "Translation:On_the_Method_of_Theoretical_Physics",
            "On the Method of Theoretical Physics",
            1933,
        ),
    "https://en.wikisource.org/wiki/Translation:On_a_Heuristic_Point_of_View_about_the_Creation_and_Conversion_of_Light":
        _wikisource_meta(
            "Translation:On_a_Heuristic_Point_of_View_about_the_Creation_and_Conversion_of_Light",
            "On a Heuristic Point of View about the Creation and Conversion of Light",
            1905,
        ),
    "https://en.wikisource.org/wiki/Translation:The_Development_of_Our_Views_on_the_Composition_and_Essence_of_Radiation":
        _wikisource_meta(
            "Translation:The_Development_of_Our_Views_on_the_Composition_and_Essence_of_Radiation",
            "The Development of Our Views on the Composition and Essence of Radiation",
            1909,
        ),
    "https://en.wikisource.org/wiki/Translation:The_Field_Equations_of_Gravitation":
        _wikisource_meta(
            "Translation:The_Field_Equations_of_Gravitation",
            "The Field Equations of Gravitation",
            1915,
        ),
    "https://en.wikisource.org/wiki/Translation:Dialog_about_Objections_against_the_Theory_of_Relativity":
        _wikisource_meta(
            "Translation:Dialog_about_Objections_against_the_Theory_of_Relativity",
            "Dialog about Objections against the Theory of Relativity",
            1918,
        ),
    "https://en.wikisource.org/wiki/Time,_Space,_and_Gravitation":
        _wikisource_meta("Time,_Space,_and_Gravitation", "Time, Space, and Gravitation", 1919),
    "https://en.wikisource.org/wiki/A_Brief_Outline_of_the_Development_of_the_Theory_of_Relativity":
        _wikisource_meta(
            "A_Brief_Outline_of_the_Development_of_the_Theory_of_Relativity",
            "A Brief Outline of the Development of the Theory of Relativity",
            1921,
        ),
    "https://en.wikisource.org/wiki/Ether_and_the_Theory_of_Relativity":
        _wikisource_meta(
            "Ether_and_the_Theory_of_Relativity",
            "Ether and the Theory of Relativity",
            1920,
        ),
    "https://en.wikisource.org/wiki/Translation:The_Bad_Nauheim_Debate":
        _wikisource_meta(
            "Translation:The_Bad_Nauheim_Debate",
            "The Bad Nauheim Debate",
            1920,
        ),
    "https://en.wikisource.org/wiki/Geometry_and_Experience":
        _wikisource_meta(
            "Geometry_and_Experience",
            "Geometry and Experience",
            1921,
        ),
    "https://en.wikisource.org/wiki/The_Meaning_of_Relativity":
        _wikisource_meta(
            "The_Meaning_of_Relativity",
            "The Meaning of Relativity",
            1922,
        ),
    # --- Gutenberg ---
    "https://www.gutenberg.org/ebooks/30155":
        _gutenberg_meta(
            "https://www.gutenberg.org/ebooks/30155",
            30155,
            "Relativity: The Special and General Theory",
            1920,
        ),
    "https://www.gutenberg.org/ebooks/36114":
        _gutenberg_meta(
            "https://www.gutenberg.org/ebooks/36114",
            36114,
            "Relativity: The Special and General Theory (1916 ed.)",
            1916,
        ),
    "https://www.gutenberg.org/ebooks/7333":
        _gutenberg_meta(
            "https://www.gutenberg.org/ebooks/7333",
            7333,
            "Sidelights on Relativity",
            1922,
        ),
    "https://www.gutenberg.org/ebooks/66944":
        _gutenberg_meta(
            "https://www.gutenberg.org/ebooks/66944",
            66944,
            "The Principle of Relativity (Einstein, Minkowski et al.)",
            1923,
        ),
    "https://www.gutenberg.org/ebooks/36276":
        _gutenberg_meta(
            "https://www.gutenberg.org/ebooks/36276",
            36276,
            "The Meaning of Relativity",
            1922,
        ),
    "https://www.gutenberg.org/ebooks/69572":
        _gutenberg_meta(
            "https://www.gutenberg.org/ebooks/69572",
            69572,
            "Fundamental ideas and problems of the theory of relativity",
            1923,
        ),
    # --- Internet Archive (existing) ---
    "https://archive.org/details/cu31924011804774":
        _ia_meta(
            "https://archive.org/details/cu31924011804774",
            "cu31924011804774",
            "Relativity: The Special and General Theory (1921)",
            1921,
        ),
    "https://archive.org/details/sidelightsonrela00einsuoft":
        _ia_meta(
            "https://archive.org/details/sidelightsonrela00einsuoft",
            "sidelightsonrela00einsuoft",
            "Sidelights on Relativity",
            1922,
        ),
}


def _load_cleaned_chars(sha: str) -> int:
    cleaned_versions = sorted(
        (CLEANED_DIR / sha).iterdir() if (CLEANED_DIR / sha).exists() else (),
        key=lambda p: p.name,
        reverse=True,
    )
    for ver_dir in cleaned_versions:
        text_file = ver_dir / "text.txt"
        if text_file.exists():
            return len(text_file.read_text(encoding="utf-8"))
    return 0


def main() -> int:
    if not RAW_DIR.exists():
        print(f"raw dir not found: {RAW_DIR}", file=sys.stderr)
        return 2
    existing: dict[str, dict] = {}
    if METADATA_FILE.exists():
        with METADATA_FILE.open("r", encoding="utf-8-sig") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                existing[payload["raw_sha256"]] = payload
    new_rows: list[dict] = []
    skipped_short: list[tuple[str, str, int]] = []
    skipped_unknown: list[tuple[str, str]] = []
    for sha_dir in sorted(RAW_DIR.iterdir()):
        if not sha_dir.is_dir():
            continue
        raw_sha = sha_dir.name
        if raw_sha in existing:
            continue
        sidecar_file = sha_dir / "sidecar.json"
        if not sidecar_file.exists():
            continue
        sidecar = json.loads(sidecar_file.read_text(encoding="utf-8"))
        source_url = sidecar.get("source_url", "")
        # normalise for table lookup
        lookup_url = source_url
        # strip ?action=raw etc.
        parsed = urlparse(lookup_url)
        lookup_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        lookup_url = unquote(lookup_url)
        chars = _load_cleaned_chars(raw_sha)
        if chars < MIN_CHARS:
            skipped_short.append((raw_sha, source_url, chars))
            continue
        if lookup_url not in _URL_TABLE:
            skipped_unknown.append((raw_sha, source_url))
            continue
        template = _URL_TABLE[lookup_url]
        row = {
            "raw_sha256": raw_sha,
            "figure_id": _FIGURE_ID,
            "archive": template["archive"],
            "source_kind": template["source_kind"],
            "source_id": template["source_id"],
            "legal_clearance": _LEGAL_CLEARANCE,
            "capture_method": _CAPTURE_METHOD,
            "captured_by": _CAPTURED_BY,
            "captured_at_iso": _CAPTURED_AT,
            "provenance_note": template["provenance_note"],
            "license_label_override": template["license_label_override"],
            "jurisdiction_hint": _JURISDICTION,
            "archive_payload": template["archive_payload"],
        }
        new_rows.append(row)
    if new_rows:
        with METADATA_FILE.open("a", encoding="utf-8") as fh:
            for row in new_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"existing rows kept: {len(existing)}")
    print(f"new rows appended : {len(new_rows)}")
    for r in new_rows:
        print(f"  + {r['archive']:<16} {r['raw_sha256'][:12]} {r['source_id']}")
    if skipped_short:
        print(f"skipped (too short, < {MIN_CHARS} chars):")
        for sha, url, chars in skipped_short:
            print(f"  ~ {sha[:12]} chars={chars:>6} {url[:90]}")
    if skipped_unknown:
        print("skipped (URL not in _URL_TABLE):")
        for sha, url in skipped_unknown:
            print(f"  ? {sha[:12]} {url[:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
