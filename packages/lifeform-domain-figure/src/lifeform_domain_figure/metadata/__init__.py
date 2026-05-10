"""Reviewer-facing metadata adapters (OpenAlex / Wikidata / Crossref / SEP).

Metadata feeds the **coverage map** + **time window** layers of the
figure vertical. By design it does NOT feed the retrieval index, the
LoRA training data, or the steering bake — the
T3-严禁污染 invariant from
[`docs/moving forward/persona-figure-data-pipeline plan`](.cursor/plans/persona-figure-data-pipeline_100ddb18.plan.md):

    "T3 严禁污染原则：二手综述只用于 coverage_map / time_window /
     reviewed_by 引证；绝对不能进 retrieval index 或 LoRA 训练数据。"

That separation is the contract that distinguishes a "real Einstein"
from "Qwen wearing an Einstein costume": only the figure's own
primary corpus reaches the model artifacts; metadata only adjusts
which queries we accept and which time-window view we surface.

This subpackage gives reviewers four offline-first translators:

* :mod:`lifeform_domain_figure.metadata.openalex` — works metadata
  (publication year, venue, language) for the in-domain breadth
  signal.
* :mod:`lifeform_domain_figure.metadata.wikidata` — biographical
  metadata (lifespan, gender, occupation, named-as labels) used by
  the lifespan boundary check and the time-window split.
* :mod:`lifeform_domain_figure.metadata.crossref` — DOI-resolved
  publication metadata for citation locators.
* :mod:`lifeform_domain_figure.metadata.sep` — Stanford Encyclopedia
  of Philosophy outline / topic structure for in-domain centroid
  hints.

Each module follows the D3 pattern:

* A typed payload dataclass (the JSON-shaped record the curator
  pre-downloads).
* One or more ``parse_*`` translators producing typed metadata
  records from the payload.
* A Protocol-shaped client (``OpenAlexClient`` etc.) and an offline
  V1 stub that raises on ``.fetch_*``.

Plus the cross-source aggregator + time-window builder live here:

* :mod:`lifeform_domain_figure.metadata.records` — neutral typed
  records used by all four sources (``FigureLifespan``,
  ``AuthoredWorkSummary``, ``DomainCoverageHint``, ``TimeWindowHint``,
  ``MetadataDigest``).
* :mod:`lifeform_domain_figure.metadata.time_window_builder` — pure
  helper that turns a tuple of ``TimeWindowHint`` records into the
  bundle's ``time_windows`` payload.
* :mod:`lifeform_domain_figure.metadata.coverage_enrichment` —
  bridge from a ``MetadataDigest`` to the existing
  :func:`lifeform_domain_figure.build_figure_coverage_map`.
"""

from __future__ import annotations

from lifeform_domain_figure.metadata.coverage_enrichment import (
    enrich_profile_with_metadata,
)
from lifeform_domain_figure.metadata.crossref import (
    CrossrefClient,
    CrossrefWorkPayload,
    crossref_to_authored_work,
    offline_crossref_client,
)
from lifeform_domain_figure.metadata.openalex import (
    OpenAlexClient,
    OpenAlexWorkPayload,
    offline_openalex_client,
    openalex_to_authored_work,
    openalex_to_domain_hints,
)
from lifeform_domain_figure.metadata.records import (
    AuthoredWorkSummary,
    DomainCoverageHint,
    FigureLifespan,
    MetadataDigest,
    MetadataSource,
    TimeWindowHint,
    aggregate_metadata,
)
from lifeform_domain_figure.metadata.sep import (
    SEPClient,
    SEPEntryPayload,
    offline_sep_client,
    sep_to_domain_hints,
)
from lifeform_domain_figure.metadata.time_window_builder import (
    build_time_window_hints_from_lifespan,
)
from lifeform_domain_figure.metadata.wikidata import (
    WikidataClient,
    WikidataPersonPayload,
    offline_wikidata_client,
    wikidata_to_lifespan,
    wikidata_to_time_window_hints,
)


__all__ = [
    # Records
    "AuthoredWorkSummary",
    "DomainCoverageHint",
    "FigureLifespan",
    "MetadataDigest",
    "MetadataSource",
    "TimeWindowHint",
    "aggregate_metadata",
    # OpenAlex
    "OpenAlexClient",
    "OpenAlexWorkPayload",
    "offline_openalex_client",
    "openalex_to_authored_work",
    "openalex_to_domain_hints",
    # Wikidata
    "WikidataClient",
    "WikidataPersonPayload",
    "offline_wikidata_client",
    "wikidata_to_lifespan",
    "wikidata_to_time_window_hints",
    # Crossref
    "CrossrefClient",
    "CrossrefWorkPayload",
    "crossref_to_authored_work",
    "offline_crossref_client",
    # SEP
    "SEPClient",
    "SEPEntryPayload",
    "offline_sep_client",
    "sep_to_domain_hints",
    # Builders
    "build_time_window_hints_from_lifespan",
    "enrich_profile_with_metadata",
]
