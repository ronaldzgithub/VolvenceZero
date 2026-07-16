"""A2 (#81) contract tests: DomainHintCatalog replaces if-domain branches.

The reviewer-curated per-domain hint summary + topic tags are typed
owner data (``DomainHintCatalog`` next to ``ApplicationBrief`` in
``vz-cognition.regime.contracts``), not ``if domain == "X":`` chains in
``vz-application/runtime_helpers.py``. These tests pin:

* zero ``if domain ==`` branches left in ``runtime_helpers.py``
  (the #81 debt-recipe gate),
* summary/topic-tag consistency inside the catalog (a reviewer adding
  a domain cannot update one side and forget the other),
* the runtime_helpers wrappers read the catalog verbatim (byte-level
  ownership move, no copy drift), and
* unknown domains fall back to the documented defaults.
"""

from __future__ import annotations

import pathlib

import volvence_zero.application.runtime_helpers as runtime_helpers
from volvence_zero.application.runtime_helpers import (
    _domain_summary,
    _domain_topic_tags,
)
from volvence_zero.regime import DEFAULT_DOMAIN_HINT_CATALOG, DomainHintCatalog


def test_runtime_helpers_has_zero_if_domain_branches() -> None:
    source = pathlib.Path(runtime_helpers.__file__).read_text(encoding="utf-8")
    assert "if domain ==" not in source


def test_catalog_summary_domains_all_have_topic_tags() -> None:
    summary_domains = {d for d, _ in DEFAULT_DOMAIN_HINT_CATALOG.summary_per_domain}
    tag_domains = {d for d, _ in DEFAULT_DOMAIN_HINT_CATALOG.topic_tags_per_domain}
    missing = summary_domains - tag_domains
    assert not missing, f"domains with summary but no topic tags: {missing}"


def test_catalog_entries_are_non_empty_and_unique() -> None:
    summaries = [d for d, _ in DEFAULT_DOMAIN_HINT_CATALOG.summary_per_domain]
    tags = [d for d, _ in DEFAULT_DOMAIN_HINT_CATALOG.topic_tags_per_domain]
    assert len(summaries) == len(set(summaries))
    assert len(tags) == len(set(tags))
    assert all(text.strip() for _, text in DEFAULT_DOMAIN_HINT_CATALOG.summary_per_domain)
    assert all(entry for _, entry in DEFAULT_DOMAIN_HINT_CATALOG.topic_tags_per_domain)
    assert DEFAULT_DOMAIN_HINT_CATALOG.default_summary.strip()
    assert DEFAULT_DOMAIN_HINT_CATALOG.default_topic_tags
    assert DEFAULT_DOMAIN_HINT_CATALOG.language == "en"


def test_runtime_helpers_wrappers_read_catalog_verbatim() -> None:
    for domain, summary in DEFAULT_DOMAIN_HINT_CATALOG.summary_per_domain:
        assert _domain_summary(domain, regime_id=None) == summary
    for domain, tags in DEFAULT_DOMAIN_HINT_CATALOG.topic_tags_per_domain:
        assert _domain_topic_tags(domain) == tags


def test_unknown_domain_falls_back_to_defaults() -> None:
    assert (
        _domain_summary("never-registered", regime_id="task_execution")
        == DEFAULT_DOMAIN_HINT_CATALOG.default_summary
    )
    assert (
        _domain_topic_tags("never-registered")
        == DEFAULT_DOMAIN_HINT_CATALOG.default_topic_tags
    )


def test_i18n_seam_second_language_catalog() -> None:
    zh = DomainHintCatalog(
        language="zh",
        summary_per_domain=(("relational_repair", "先降级冲突再解释。"),),
        topic_tags_per_domain=(("relational_repair", ("修复",)),),
        default_summary="保持有界。",
    )
    assert zh.summary_for("relational_repair") == "先降级冲突再解释。"
    assert zh.topic_tags_for("unknown") == ("general",)
