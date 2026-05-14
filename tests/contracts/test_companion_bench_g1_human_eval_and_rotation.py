"""Contract test: G1 Companion Bench launch — human-eval + rotation (#33 / #35).

Validates:

1. Human-eval site scaffolding exists at ``site/human-eval/``
   with the four documented pages (``index`` / ``apply`` /
   ``training`` / ``submit``).
2. Each page references the v0.1 protocol so reviewers can find
   the rubric without browsing the repo.
3. Apply form has the required NDA + COI checkboxes (debt #33
   protocol §2.1).
4. ``quarterly_rotation.py`` produces the three documented
   artifacts (paraphrase proposals jsonl, lexicon txt, judge
   rotation log row) when given a quarter + judge family.
5. Judge rotation log file exists after the script run with the
   expected markdown table header.

Refs:

* docs/known-debts.md #33 / #35
* docs/external/companion-bench-human-eval-protocol.md
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import ModuleType


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SITE_DIR = _REPO_ROOT / "site" / "human-eval"
_SCRIPTS_DIR = _REPO_ROOT / "scripts" / "companion_bench"


def _load_script(filename: str) -> ModuleType:
    path = _SCRIPTS_DIR / filename
    spec = importlib.util.spec_from_file_location(
        f"_g1_{path.stem}", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# #33 human-eval UI scaffolding
# ---------------------------------------------------------------------------


def test_human_eval_site_pages_exist() -> None:
    for page in ("index.html", "apply.html", "training.html", "submit.html"):
        path = _SITE_DIR / page
        assert path.exists(), f"site/human-eval/{page} missing (debt #33)"


def test_apply_form_has_nda_and_coi_checkboxes() -> None:
    apply_html = (_SITE_DIR / "apply.html").read_text(encoding="utf-8")
    # Required checkbox names from protocol §2.1.
    assert 'name="not_lab_employed"' in apply_html
    assert 'name="agree_nda"' in apply_html
    assert 'name="agree_payment"' in apply_html


def test_human_eval_pages_link_back_to_protocol() -> None:
    """Every page must reference the v0.1 protocol so reviewers can
    locate the rubric without browsing the repo by hand."""
    for page in ("index.html", "apply.html", "training.html"):
        text = (_SITE_DIR / page).read_text(encoding="utf-8")
        assert "companion-bench-human-eval-protocol" in text, (
            f"{page} must link to the human-eval protocol doc"
        )


# ---------------------------------------------------------------------------
# #35 quarterly_rotation.py
# ---------------------------------------------------------------------------


def test_quarterly_rotation_emits_three_artifacts(tmp_path: pathlib.Path) -> None:
    rotation = _load_script("quarterly_rotation.py")
    rc = rotation.main(
        [
            "--quarter", "2026Q3",
            "--judge-family", "deepseek",
            "--judge-rationale", "Test run for contract.",
            "--output-dir", str(tmp_path / "gov"),
        ]
    )
    assert rc == 0
    paraphrase = tmp_path / "gov" / "heldout_rotation_2026Q3.jsonl"
    lexicon = tmp_path / "gov" / "lexicon_rotation_2026Q3.txt"
    assert paraphrase.exists()
    assert lexicon.exists()
    # Judge rotation log lives in repo (intentional — the log is the
    # public audit trail). Verify a row was appended.
    log_path = (
        _REPO_ROOT / "docs" / "external"
        / "companion-bench-judge-rotation-log.md"
    )
    assert log_path.exists()
    text = log_path.read_text(encoding="utf-8")
    assert "2026Q3" in text
    assert "deepseek" in text
    assert "Test run for contract" in text


def test_quarterly_rotation_paraphrases_per_scenario(tmp_path: pathlib.Path) -> None:
    rotation = _load_script("quarterly_rotation.py")
    rotation.main(
        [
            "--quarter", "2026Q4",
            "--judge-family", "qwen",
            "--judge-rationale", "Q4 quota allocation.",
            "--output-dir", str(tmp_path / "gov"),
            "--paraphrases-per-scenario", "4",
        ]
    )
    paraphrase = tmp_path / "gov" / "heldout_rotation_2026Q4.jsonl"
    lines = paraphrase.read_text(encoding="utf-8").splitlines()
    # 6 families × 4 scenarios × 4 paraphrases = 96 rows.
    assert len(lines) == 96


def test_quarterly_rotation_dry_run_writes_nothing(tmp_path: pathlib.Path) -> None:
    rotation = _load_script("quarterly_rotation.py")
    rc = rotation.main(
        [
            "--quarter", "2026Q5",
            "--judge-family", "openai",
            "--judge-rationale", "Dry run.",
            "--output-dir", str(tmp_path / "gov"),
            "--dry-run",
        ]
    )
    assert rc == 0
    # output_dir should not be created in dry-run mode.
    assert not (tmp_path / "gov").exists()
