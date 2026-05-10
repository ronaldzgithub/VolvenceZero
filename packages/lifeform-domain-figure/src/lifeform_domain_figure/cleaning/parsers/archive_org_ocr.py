"""Internet Archive DjVu OCR JSON parser.

Internet Archive distributes scanned items with a ``_djvu.txt`` and a
matching ``_djvu.json`` (or ``_chocr.html.json``) artefact. We accept
the canonical JSON shape::

    {
      "metadata": {
        "language": "eng",
        "licenseurl": "https://...",
        ...
      },
      "ocr": [
        {"page": 1, "text": "...", "confidence": 0.92},
        {"page": 2, "text": "...", "confidence": 0.87}
      ]
    }

The parser concatenates page text with form-feed separators, averages
per-page ``confidence`` into ``ocr_confidence``, and lifts the
``metadata.language`` / ``metadata.licenseurl`` (or licence string)
into the ``RawDocument``.
"""

from __future__ import annotations

import hashlib
import json

from lifeform_domain_figure.cleaning.raw_document import RawDocument

ARCHIVE_ORG_OCR_CONTENT_TYPE = "application/json; profile=archive-org-ocr"
PARSER_VERSION = "archive-org-ocr:1"


def _coerce_confidence(value: object) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    return 0.0


def _coerce_language(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().split("-", 1)[0][:3].lower()


def parse_archive_org_ocr_json(
    data: bytes,
    *,
    source_url: str,
    content_type: str = ARCHIVE_ORG_OCR_CONTENT_TYPE,
) -> RawDocument:
    """Parse Internet Archive OCR JSON bytes into a :class:`RawDocument`."""

    if content_type != ARCHIVE_ORG_OCR_CONTENT_TYPE:
        raise ValueError(
            f"parse_archive_org_ocr_json: refusing content_type={content_type!r}; "
            f"expected {ARCHIVE_ORG_OCR_CONTENT_TYPE!r}"
        )
    if not data:
        raise ValueError(
            f"parse_archive_org_ocr_json: empty bytes for source_url={source_url!r}"
        )
    try:
        decoded = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"parse_archive_org_ocr_json: bytes for source_url={source_url!r} "
            f"are not valid UTF-8 ({exc})"
        ) from exc
    payload = json.loads(decoded)
    if not isinstance(payload, dict):
        raise ValueError(
            f"parse_archive_org_ocr_json: top-level JSON must be an object; "
            f"got {type(payload).__name__} for source_url={source_url!r}"
        )
    pages = payload.get("ocr")
    if not isinstance(pages, list) or not pages:
        raise ValueError(
            f"parse_archive_org_ocr_json: 'ocr' must be a non-empty list of "
            f"page records for source_url={source_url!r}"
        )
    page_texts: list[str] = []
    confidences: list[float] = []
    for index, page in enumerate(pages):
        if not isinstance(page, dict):
            raise ValueError(
                f"parse_archive_org_ocr_json: ocr[{index}] must be an object "
                f"for source_url={source_url!r}; got {type(page).__name__}"
            )
        page_text = page.get("text")
        if not isinstance(page_text, str):
            raise ValueError(
                f"parse_archive_org_ocr_json: ocr[{index}].text must be a string "
                f"for source_url={source_url!r}"
            )
        page_texts.append(page_text)
        confidences.append(_coerce_confidence(page.get("confidence")))
    extracted = "\f".join(page_texts).strip()
    if not extracted:
        raise ValueError(
            f"parse_archive_org_ocr_json: concatenated page text is empty for "
            f"source_url={source_url!r}"
        )
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    language_detected = _coerce_language(metadata.get("language", ""))
    license_value = metadata.get("licenseurl", "") or metadata.get("license", "")
    license_notice = license_value.strip() if isinstance(license_value, str) else ""
    raw_sha256 = hashlib.sha256(data).hexdigest()
    return RawDocument(
        text=extracted,
        parser_version=PARSER_VERSION,
        layout_quality=1.0,
        ocr_confidence=avg_confidence,
        encoding_detected="utf-8",
        language_detected=language_detected,
        license_notice=license_notice,
        raw_sha256=raw_sha256,
    )


__all__ = [
    "ARCHIVE_ORG_OCR_CONTENT_TYPE",
    "PARSER_VERSION",
    "parse_archive_org_ocr_json",
]
