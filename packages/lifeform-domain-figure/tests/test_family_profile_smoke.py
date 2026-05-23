"""Contract test for U4 — family memorial dynamic profile.

The family-memorial product (in VolvenceDeploy ``apps/family-memorial/``)
bakes one ``FigureArtifactBundle`` per memorial via the figure CLI:

```
python -m lifeform_domain_figure.cli bake-bundle \
  --figure family_<memorialId> \
  --profile-file <path>/profile.json \
  --corpus-mode curated ...
```

Before U4, the CLI's ``--figure`` accepted only built-in ids
(``einstein`` / ``lu_xun``) and there was no way to inject a
family-attested ``HistoricalFigureProfile``. This test locks the new
contract: the loader builds a minimal valid profile from a small JSON
dict, the CLI argparse accepts ``family_*`` ids, and the resolver
returns a profile factory that matches the JSON's profile_id.
"""

from __future__ import annotations

import json

import pytest

from lifeform_domain_figure.cli import _validate_figure_arg
from lifeform_domain_figure.cli._commands import _resolve_figure_factories
from lifeform_domain_figure.profiles.family import (
    build_family_profile_from_json,
    load_family_profile_file,
)


_VALID_PAYLOAD = {
    "figure_id": "family_ck1abcd2345xyz",
    "display_name": "外公",
    "relation": "外公",
    "birth_year": 1925,
    "death_year": 2008,
    "bio": "童年在江南；80 年代讲述过抗战经历。",
    "time_window_id": "family-ck1abcd2345xyz-1925-2008",
}


def test_validator_accepts_builtins_and_family_ids() -> None:
    assert _validate_figure_arg("einstein") == "einstein"
    assert _validate_figure_arg("lu_xun") == "lu_xun"
    assert (
        _validate_figure_arg("family_ck1abcd2345xyz") == "family_ck1abcd2345xyz"
    )


def test_validator_rejects_unknown() -> None:
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        _validate_figure_arg("napoleon")
    with pytest.raises(argparse.ArgumentTypeError):
        _validate_figure_arg("family_")  # too short
    with pytest.raises(argparse.ArgumentTypeError):
        _validate_figure_arg("family_with whitespace")


def test_build_family_profile_from_json_minimal_valid() -> None:
    profile = build_family_profile_from_json(_VALID_PAYLOAD)
    assert profile.profile_id == _VALID_PAYLOAD["figure_id"]
    assert profile.figure_name == _VALID_PAYLOAD["display_name"]
    assert profile.figure_lifespan == (1925, 2008)
    # L4 contract: domain_coverage_seed is narrow (single category) so
    # the coverage map cannot grant accidental breadth.
    assert profile.domain_coverage_seed == ("family_personal_narrative",)
    # Exactly one boundary prior declaring the out-of-scope list.
    assert len(profile.boundary_priors) == 1
    boundary = profile.boundary_priors[0]
    assert boundary.out_of_scope_topics
    assert "impersonating-living-people" in boundary.out_of_scope_topics
    assert "answer-only-from-family-uploaded-materials" in (
        boundary.required_disclaimers
    )
    # Single time window matching the lifespan.
    assert len(profile.time_windows) == 1
    win = profile.time_windows[0]
    assert win.year_start == 1925
    assert win.year_end == 2008


def test_build_family_profile_from_json_birth_year_optional() -> None:
    payload = dict(_VALID_PAYLOAD)
    payload.pop("birth_year")
    payload["time_window_id"] = ""
    profile = build_family_profile_from_json(payload)
    # When birth_year is unknown, lifespan collapses to (death_year, death_year)
    # so the (died >= born) invariant holds without claiming a birth year.
    assert profile.figure_lifespan == (2008, 2008)
    assert profile.time_windows == ()


def test_build_family_profile_from_json_missing_required_raises() -> None:
    payload = dict(_VALID_PAYLOAD)
    payload.pop("figure_id")
    with pytest.raises(ValueError, match="figure_id"):
        build_family_profile_from_json(payload)


def test_build_family_profile_from_json_death_before_birth_raises() -> None:
    payload = dict(_VALID_PAYLOAD)
    payload["birth_year"] = 2010
    payload["death_year"] = 2008
    with pytest.raises(ValueError, match="death_year"):
        build_family_profile_from_json(payload)


def test_load_family_profile_file_roundtrip(tmp_path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(_VALID_PAYLOAD), encoding="utf-8")
    profile = load_family_profile_file(path)
    assert profile.profile_id == _VALID_PAYLOAD["figure_id"]


def test_load_family_profile_file_accepts_single_entry_list(tmp_path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(json.dumps([_VALID_PAYLOAD]), encoding="utf-8")
    profile = load_family_profile_file(path)
    assert profile.profile_id == _VALID_PAYLOAD["figure_id"]


def test_load_family_profile_file_rejects_multi_entry(tmp_path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps([_VALID_PAYLOAD, _VALID_PAYLOAD]), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="exactly one"):
        load_family_profile_file(path)


def test_resolver_returns_family_factory_with_profile_file(tmp_path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(_VALID_PAYLOAD), encoding="utf-8")
    profile_factory, corpus_factory = _resolve_figure_factories(
        "family_ck1abcd2345xyz", profile_file=str(path)
    )
    assert profile_factory is not None
    profile = profile_factory()
    assert profile.profile_id == "family_ck1abcd2345xyz"
    # Family memorials are CURATED-CORPUS ONLY; the synthetic-corpus
    # factory must remain unset so the CLI fails loud on the wrong path.
    assert corpus_factory is None


def test_resolver_without_profile_file_returns_none_for_family() -> None:
    """``--figure family_*`` without ``--profile-file`` is a usage
    error, but the resolver itself must just return (None, None) and
    let the CLI handler emit the typed error message (so cmd_bake_bundle
    can decide the exit code)."""

    profile_factory, corpus_factory = _resolve_figure_factories(
        "family_ck1abcd2345xyz", profile_file=None
    )
    assert profile_factory is None
    assert corpus_factory is None


def test_resolver_mismatched_figure_and_profile_id_raises(tmp_path) -> None:
    path = tmp_path / "profile.json"
    payload = dict(_VALID_PAYLOAD)
    payload["figure_id"] = "family_someoneelse"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        _resolve_figure_factories(
            "family_ck1abcd2345xyz", profile_file=str(path)
        )
