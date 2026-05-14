"""Contract test: G2 site cleanup (debt #38).

Validates:

1. ``docs/external/lscb-*.md`` redirect stubs have been removed
   (LSCB rebrand cleanup).
2. ``scripts/companion_bench/verify_site.py`` exits 0 on the
   shipped site (page existence + internal links + data field
   completeness + demo banner).
3. ``scripts/companion_bench/build_site.py --incremental`` skips
   per-submission rebuild when an existing detail file is newer
   than the source summary.json (debt #38 incrementalisation).

Refs:

* docs/known-debts.md #38
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import shutil
import sys
import time
from types import ModuleType


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / "scripts" / "companion_bench"


def _load_script(filename: str) -> ModuleType:
    path = _SCRIPTS_DIR / filename
    spec = importlib.util.spec_from_file_location(
        f"_g2_{path.stem}", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# LSCB rebrand cleanup
# ---------------------------------------------------------------------------


def test_lscb_redirect_stubs_removed() -> None:
    """All seven historical LSCB stub files must be gone."""
    for stub in (
        "docs/external/lscb-governance-charter-draft.md",
        "docs/external/lscb-rfc-v0.md",
        "docs/external/lscb-submission-protocol.md",
        "docs/external/lscb-eqbench-crosswalk.md",
        "docs/external/lscb-heldout-bootstrap.md",
        "docs/external/lscb-public-scenario-hashes.txt",
        "docs/specs/lscb-bench.md",
    ):
        path = _REPO_ROOT / stub
        assert not path.exists(), (
            f"LSCB redirect stub still present: {stub} (debt #38 cleanup)"
        )


# ---------------------------------------------------------------------------
# verify_site.py
# ---------------------------------------------------------------------------


def test_verify_site_passes_on_shipped_site() -> None:
    """The verifier should pass on the as-shipped ``site/`` tree."""
    verify = _load_script("verify_site.py")
    rc = verify.main(["--site-dir", str(_REPO_ROOT / "site")])
    assert rc == 0


def test_verify_site_detects_missing_page(tmp_path: pathlib.Path) -> None:
    """Removing a required page must surface as a verifier failure."""
    verify = _load_script("verify_site.py")
    fake_site = tmp_path / "site"
    fake_site.mkdir()
    # Copy the data dir + a couple of pages but deliberately omit
    # leaderboard.html so the verifier flags it.
    (fake_site / "data").mkdir()
    for page in ("index.html", "scenarios.html"):
        (fake_site / page).write_text(
            "<!DOCTYPE html><html></html>", encoding="utf-8"
        )
    rc = verify.main(["--site-dir", str(fake_site)])
    assert rc == 2


# ---------------------------------------------------------------------------
# build_site.py --incremental
# ---------------------------------------------------------------------------


def test_build_site_incremental_skips_when_detail_newer(tmp_path: pathlib.Path) -> None:
    """Per-submission rebuild is skipped when detail file is newer than
    summary.json (debt #38 incrementalisation)."""
    build = _load_script("build_site.py")

    # Set up a minimal artifact dir with one submission summary.
    artifact_dir = tmp_path / "artifacts"
    sub_dir = artifact_dir / "smoke-1"
    sub_dir.mkdir(parents=True)
    summary = {
        "manifest": {"submission_id": "smoke-1"},
        "aggregate": {"submission_id": "smoke-1", "axis_means": {}},
        "arc_count": 0,
    }
    summary_path = sub_dir / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    # Set up an existing site/data/submissions/smoke-1.json that's
    # newer than summary.json (cached payload).
    site_dir = tmp_path / "site"
    submissions_dir = site_dir / "data" / "submissions"
    submissions_dir.mkdir(parents=True)
    cached_payload = {
        "submission_id": "smoke-1",
        "aggregate": {"submission_id": "smoke-1"},
        "arcs": [],
        "family_means": {},
        "_marker": "from-incremental-cache",
    }
    cached_path = submissions_dir / "smoke-1.json"
    cached_path.write_text(json.dumps(cached_payload), encoding="utf-8")
    # Force the cache mtime to be newer than the summary.
    later = summary_path.stat().st_mtime + 60
    import os as _os
    _os.utime(cached_path, (later, later))

    rc = build.main(
        [
            "--artifact-dir", str(artifact_dir),
            "--site-dir", str(site_dir),
            "--incremental",
        ]
    )
    # build_site exit code may be 0 even if pairwise inputs are
    # empty — we just verify the cached marker survived.
    assert rc == 0
    after = json.loads(cached_path.read_text(encoding="utf-8"))
    assert after["_marker"] == "from-incremental-cache", (
        "incremental cache file was rebuilt despite being newer than summary.json"
    )
