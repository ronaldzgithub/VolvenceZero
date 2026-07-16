"""LLM audience analysis for the DLaaS control plane (debt #14 ACTIVE).

Replaces the placeholder ``audience/analyze`` behaviour ("persist the
caller's form fields verbatim") with real corpus analysis:

1. :func:`load_asset_corpus` resolves each linked asset's actual text
   content (``source_meta['inline_text']``, local file path, or
   ``file://`` URI — the registry stores metadata only; bytes live at
   ``uri``). Unreadable assets raise :class:`AssetCorpusError` — the
   route surfaces a typed 422, never a silently-empty analysis.
2. :class:`LLMAudienceAnalyzer` sends the corpus through the same
   env-configured OpenAI-compatible endpoint the rubric judge uses
   (``EVAL_LLM_*`` falling back to ``PROTOCOL_LLM_*``) with the
   centralized audience prompt, and validates the strict-JSON cohort
   profile (common questions / communication style / emotion triggers
   / decision patterns + evidence notes).

Design constraints (mirror :mod:`dlaas_platform_eval.llm_grader`):

* **Fail loudly.** Malformed LLM output raises
  :class:`AudienceAnalysisError`; there is no partial/guessed profile.
  When the LLM env is not configured the route keeps the SHADOW
  passthrough (caller-supplied fields verbatim) and stamps
  ``evidence_stats['analyzer'] = 'none'`` so the record is honest
  about not being backed by analysis.
* **R12 / R8.** Analysis is a platform readout: profile fields are
  persisted to the ``audience_profiles`` table only; nothing here
  reads or writes any kernel owner, reward, or Face path.
* **Sync transport.** ``analyze`` is sync (blocking stdlib HTTP via
  ``chat_completion_text``); route handlers run it off the event loop
  with ``asyncio.to_thread``.
"""

from __future__ import annotations

import json
import logging
import pathlib
import urllib.parse
from collections.abc import Mapping
from typing import Any

from dlaas_platform_contracts import AssetSpec

from dlaas_platform_eval.llm_grader import (
    EvalLLMConfig,
    EvalLLMError,
    LLMTransport,
    chat_completion_text,
    parse_strict_json_object,
    resolve_eval_llm_config,
)
from dlaas_platform_eval.prompts import (
    AUDIENCE_ANALYSIS_SYSTEM_PROMPT,
    AUDIENCE_ANALYSIS_USER_TEMPLATE,
)

_LOG = logging.getLogger(__name__)

# Bounded prompt budget: per-asset and total corpus text caps. Assets
# beyond the total budget are truncated, and the truncation is recorded
# in evidence_stats so operators see exactly what was analysed.
MAX_CHARS_PER_ASSET = 20_000
MAX_TOTAL_CORPUS_CHARS = 60_000
_MAX_LIST_FIELD_ITEMS = 16


class AudienceAnalysisError(EvalLLMError):
    """The audience-analysis LLM call failed or violated the contract."""


class AssetCorpusError(ValueError):
    """An asset's content could not be resolved into analysable text."""


def _read_local_asset_text(asset: AssetSpec) -> str:
    """Resolve local-file content for one asset (path or ``file://``)."""

    uri = asset.uri.strip()
    if uri.startswith("file://"):
        path = pathlib.Path(urllib.parse.urlparse(uri).path)
    else:
        path = pathlib.Path(uri)
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, IsADirectoryError, PermissionError) as exc:
        raise AssetCorpusError(
            f"asset {asset.asset_id!r} content at {uri!r} is unreadable: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    except UnicodeDecodeError as exc:
        raise AssetCorpusError(
            f"asset {asset.asset_id!r} at {uri!r} is not utf-8 text "
            f"(mime_type={asset.mime_type!r}); convert it to text before "
            f"audience analysis."
        ) from exc


def load_asset_corpus(
    assets: tuple[AssetSpec, ...],
    *,
    max_chars_per_asset: int = MAX_CHARS_PER_ASSET,
) -> tuple[str, ...]:
    """Resolve asset metadata into analysable text chunks.

    Resolution order per asset:

    1. ``source_meta['inline_text']`` — content uploaded inline with
       the asset record (the create-asset API accepts arbitrary
       ``source_meta``).
    2. Local file path or ``file://`` URI.

    Remote schemes (``http``, ``s3``, ``test:`` fixtures, …) are not
    fetched here — the platform's asset bytes are expected to be
    staged locally (or inline) before analysis; anything else raises
    :class:`AssetCorpusError` naming the asset, so the operator sees
    which link is not analysable instead of getting a hollow profile.
    """

    chunks: list[str] = []
    for asset in assets:
        inline = str(asset.source_meta.get("inline_text", "") or "")
        if inline.strip():
            text = inline
        elif asset.uri.strip():
            scheme = urllib.parse.urlparse(asset.uri).scheme
            if scheme in ("", "file"):
                text = _read_local_asset_text(asset)
            else:
                raise AssetCorpusError(
                    f"asset {asset.asset_id!r} uri {asset.uri!r} uses "
                    f"scheme {scheme!r}; audience analysis reads inline "
                    f"source_meta['inline_text'], local paths, or file:// "
                    f"URIs. Stage the content locally first."
                )
        else:
            raise AssetCorpusError(
                f"asset {asset.asset_id!r} has neither "
                f"source_meta['inline_text'] nor a uri; nothing to analyse."
            )
        text = text.strip()
        if not text:
            raise AssetCorpusError(
                f"asset {asset.asset_id!r} resolved to empty text."
            )
        chunks.append(text[:max_chars_per_asset])
    return tuple(chunks)


def _string_tuple(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise AudienceAnalysisError(
            f"audience LLM output field {field!r} must be a list, "
            f"got {value!r}"
        )
    items = tuple(str(item).strip() for item in value if str(item).strip())
    return items[:_MAX_LIST_FIELD_ITEMS]


class LLMAudienceAnalyzer:
    """Cohort-profile extractor backed by an OpenAI-compatible endpoint.

    R12 guard: pure readout — (cohort_name, corpus chunks) in, profile
    fields out. No kernel owner is read or written.
    """

    def __init__(
        self,
        config: EvalLLMConfig,
        *,
        transport: LLMTransport | None = None,
    ) -> None:
        self._config = config
        # ``None`` means "resolve chat_completion_text at call time" so
        # tests can monkeypatch the module-level function.
        self._transport = transport
        _LOG.info(
            "LLMAudienceAnalyzer configured (model=%s, base_url=%s)",
            config.model,
            config.base_url,
        )

    @property
    def analyzer_label(self) -> str:
        return f"llm:{self._config.model}"

    def analyze(
        self,
        *,
        cohort_name: str,
        corpus_chunks: tuple[str, ...],
    ) -> Mapping[str, Any]:
        """Extract the cohort profile from the corpus.

        Returns the enrichment mapping consumed by
        ``EvalStore.upsert_audience_profile``'s analyzer seam
        (``common_questions`` / ``communication_style`` /
        ``emotion_triggers`` / ``decision_patterns`` /
        ``evidence_stats``).

        Raises:
            AudienceAnalysisError: on transport failure or output that
                violates the strict-JSON profile contract. No partial
                profile is ever returned.
        """

        if not corpus_chunks:
            raise AudienceAnalysisError(
                "audience analysis requires a non-empty corpus; resolve "
                "asset content before calling analyze()."
            )
        corpus_text = "\n\n---\n\n".join(corpus_chunks)
        truncated = len(corpus_text) > MAX_TOTAL_CORPUS_CHARS
        if truncated:
            corpus_text = corpus_text[:MAX_TOTAL_CORPUS_CHARS]
        user_prompt = AUDIENCE_ANALYSIS_USER_TEMPLATE.format(
            cohort_name=cohort_name,
            corpus_text=corpus_text,
        )
        transport = self._transport or chat_completion_text
        try:
            content = transport(
                self._config,
                system_prompt=AUDIENCE_ANALYSIS_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except AudienceAnalysisError:
            raise
        except EvalLLMError as exc:
            raise AudienceAnalysisError(str(exc)) from exc
        return self._parse_profile(
            content,
            chunk_count=len(corpus_chunks),
            corpus_chars=len(corpus_text),
            truncated=truncated,
        )

    def _parse_profile(
        self,
        content: str,
        *,
        chunk_count: int,
        corpus_chars: int,
        truncated: bool,
    ) -> Mapping[str, Any]:
        try:
            data = parse_strict_json_object(content)
        except json.JSONDecodeError as exc:
            raise AudienceAnalysisError(
                f"audience LLM output is not valid JSON: {exc}; "
                f"content={content[:300]!r}"
            ) from exc
        for field in (
            "common_questions",
            "communication_style",
            "emotion_triggers",
            "decision_patterns",
        ):
            if field not in data:
                raise AudienceAnalysisError(
                    f"audience LLM output missing field {field!r}: "
                    f"{content[:300]!r}"
                )
        return {
            "common_questions": _string_tuple(
                data["common_questions"], field="common_questions"
            ),
            "communication_style": str(
                data["communication_style"] or ""
            ).strip(),
            "emotion_triggers": _string_tuple(
                data["emotion_triggers"], field="emotion_triggers"
            ),
            "decision_patterns": _string_tuple(
                data["decision_patterns"], field="decision_patterns"
            ),
            "evidence_stats": {
                "analyzer": self.analyzer_label,
                "chunk_count": chunk_count,
                "corpus_chars": corpus_chars,
                "corpus_truncated": truncated,
                "evidence_notes": str(
                    data.get("evidence_notes", "") or ""
                ).strip(),
            },
        }


def build_audience_analyzer_from_env() -> LLMAudienceAnalyzer | None:
    """Analyzer for this deployment, or ``None`` when the env is unset.

    Shares the ``EVAL_LLM_*`` / ``PROTOCOL_LLM_*`` env contract with
    the rubric judge (debt #14 trigger (b): one LLM provider config
    for the whole eval gate). ``None`` keeps the audience route on the
    honest SHADOW passthrough.
    """

    config = resolve_eval_llm_config()
    if config is None:
        return None
    return LLMAudienceAnalyzer(config)


__all__ = [
    "MAX_CHARS_PER_ASSET",
    "MAX_TOTAL_CORPUS_CHARS",
    "AssetCorpusError",
    "AudienceAnalysisError",
    "LLMAudienceAnalyzer",
    "build_audience_analyzer_from_env",
    "load_asset_corpus",
]
