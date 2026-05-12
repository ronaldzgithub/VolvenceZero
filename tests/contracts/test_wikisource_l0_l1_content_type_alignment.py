"""L0/L1 content-type alignment contract.

When the wikisource fetcher successfully follows the ``action=raw``
path it persists ``content_type=text/x-wiki`` (see
[`packages/lifeform-domain-figure/src/lifeform_domain_figure/crawl/fetchers/wikisource.py`](packages/lifeform-domain-figure/src/lifeform_domain_figure/crawl/fetchers/wikisource.py)).
The L1 cleaning dispatcher
[`packages/lifeform-domain-figure/src/lifeform_domain_figure/cleaning/parsers/__init__.py`](packages/lifeform-domain-figure/src/lifeform_domain_figure/cleaning/parsers/__init__.py)
must accept that label or ``re-clean-all`` fails loudly on real
wikisource bytes (Wave H closure).

This test enforces the invariant **statically**, i.e. without
requiring a live wikisource fetch in CI: every content_type
constant exported by an L0 fetcher's parser-side companion
module MUST also appear in the L1 dispatcher's
``parse_by_content_type`` body. The check is deliberately
content-type-symmetric so a future fetcher cannot drift
content_type without also wiring the cleaning side.
"""

from __future__ import annotations

import inspect

from lifeform_domain_figure.cleaning import parsers as parser_dispatch
from lifeform_domain_figure.cleaning.parsers import (
    ARCHIVE_ORG_OCR_CONTENT_TYPE,
    CPAE_PDF_CONTENT_TYPE,
    GUTENBERG_HTML_CONTENT_TYPE,
    GUTENBERG_TEXT_CONTENT_TYPE,
    WIKISOURCE_HTML_CONTENT_TYPE,
    WIKISOURCE_WIKITEXT_CONTENT_TYPE,
)


def _dispatcher_source() -> str:
    """Return the source code of ``parse_by_content_type``."""

    return inspect.getsource(parser_dispatch.parse_by_content_type)


def test_wikisource_wikitext_content_type_is_dispatched() -> None:
    """``text/x-wiki`` must route to the wikisource parser branch.

    We assert the dispatcher source references the **constant
    names** rather than the literal label strings — the constant
    is the SSOT and the dispatcher imports it by name. A future
    rename of the constant value would still satisfy the test as
    long as the dispatcher and the parser module agree.
    """

    src = _dispatcher_source()
    assert "WIKISOURCE_WIKITEXT_CONTENT_TYPE" in src, (
        "parse_by_content_type must reference WIKISOURCE_WIKITEXT_CONTENT_TYPE "
        "(text/x-wiki) so wikisource fetcher's action=raw path can be cleaned"
    )
    assert "WIKISOURCE_HTML_CONTENT_TYPE" in src, (
        "parse_by_content_type must still reference WIKISOURCE_HTML_CONTENT_TYPE"
    )
    # Concrete invariant: dispatcher calls parse_wikisource_html on the
    # wikitext branch (no separate parser).
    assert "parse_wikisource_html" in src
    # Also verify the runtime values agree (sanity check on imports).
    assert WIKISOURCE_WIKITEXT_CONTENT_TYPE == "text/x-wiki"
    assert WIKISOURCE_HTML_CONTENT_TYPE == "text/html; profile=wikisource"


def test_every_l0_fetcher_content_type_constant_dispatched() -> None:
    """Static guarantee: every parser-companion content_type constant
    is referenced by name in the L1 dispatcher body."""

    src = _dispatcher_source()
    expected = {
        "CPAE_PDF_CONTENT_TYPE": CPAE_PDF_CONTENT_TYPE,
        "WIKISOURCE_HTML_CONTENT_TYPE": WIKISOURCE_HTML_CONTENT_TYPE,
        "WIKISOURCE_WIKITEXT_CONTENT_TYPE": WIKISOURCE_WIKITEXT_CONTENT_TYPE,
        "GUTENBERG_HTML_CONTENT_TYPE": GUTENBERG_HTML_CONTENT_TYPE,
        "GUTENBERG_TEXT_CONTENT_TYPE": GUTENBERG_TEXT_CONTENT_TYPE,
        "ARCHIVE_ORG_OCR_CONTENT_TYPE": ARCHIVE_ORG_OCR_CONTENT_TYPE,
    }
    missing = sorted(name for name in expected if name not in src)
    assert not missing, (
        f"L1 parse_by_content_type does not reference these L0 content_type "
        f"constants: {missing!r}. Adding a fetcher? Wire the dispatcher in "
        f"the same packet."
    )


def test_dispatcher_rejects_unknown_content_type() -> None:
    """Dispatcher fails loud on unknown content_type (no silent fallback)."""

    import pytest

    with pytest.raises(ValueError, match="no parser registered"):
        parser_dispatch.parse_by_content_type(
            b"x", source_url="https://example.invalid/", content_type="text/foo"
        )
