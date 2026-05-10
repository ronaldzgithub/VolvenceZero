"""Test fixture builders for the L1 cleaning pipeline.

These helpers return the raw bytes for one minimal sample of each
supported source format. They are kept in a normal ``.py`` module
(rather than committing 4-5 binary fixture files) so the fixtures
are reproducible from source review.
"""

from __future__ import annotations

import json


EINSTEIN_QUOTE = "Einstein wrote about relativity in 1905."


def build_minimal_cpae_pdf_bytes() -> bytes:
    """Build a minimal valid PDF carrying ``EINSTEIN_QUOTE``.

    This is the canonical 5-object skeleton (Catalog + Pages + Page
    + Contents + Font); xref byte-offsets are computed dynamically so
    the body stays human-editable.
    """

    content_stream = (
        b"BT\n/F1 12 Tf\n72 720 Td\n("
        + EINSTEIN_QUOTE.encode("ascii")
        + b") Tj\nET\n"
    )
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length "
        + str(len(content_stream)).encode("ascii")
        + b" >>\nstream\n"
        + content_stream
        + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    body_chunks: list[bytes] = []
    offsets: list[int] = []
    cursor = len(header)
    for index, obj_body in enumerate(objects, start=1):
        offsets.append(cursor)
        chunk = (
            str(index).encode("ascii") + b" 0 obj\n" + obj_body + b"\nendobj\n"
        )
        body_chunks.append(chunk)
        cursor += len(chunk)
    xref_offset = cursor
    xref_lines = [b"xref\n", f"0 {len(objects) + 1}\n".encode("ascii")]
    xref_lines.append(b"0000000000 65535 f \n")
    for offset in offsets:
        xref_lines.append(f"{offset:010d} 00000 n \n".encode("ascii"))
    trailer = (
        b"trailer\n<< /Size "
        + str(len(objects) + 1).encode("ascii")
        + b" /Root 1 0 R >>\nstartxref\n"
        + str(xref_offset).encode("ascii")
        + b"\n%%EOF\n"
    )
    return header + b"".join(body_chunks) + b"".join(xref_lines) + trailer


def build_wikisource_html_bytes() -> bytes:
    """Wikisource-style HTML page with a license template + body."""

    html = """<!DOCTYPE html>
<html lang="en">
<head><title>Sample Wikisource Page</title></head>
<body>
<div id="mw-content-text">
{{header
| title    = Annus Mirabilis Letter
| author   = Albert Einstein
| year     = 1905
| language = en
}}

This is the body of a sample Wikisource transcription. Albert Einstein
wrote about the photoelectric effect in 1905 in a famous paper.

''Italic emphasis here'' and a [[link to another page|link]] inline.

{{PD-old-100}}
</div>
</body>
</html>
"""
    return html.encode("utf-8")


def build_gutenberg_text_bytes() -> bytes:
    """Plain-text Gutenberg-shaped fixture with START/END markers."""

    text = """The Project Gutenberg eBook of Sample Letters by Albert Einstein

This eBook is for the use of anyone anywhere in the United States and
most other parts of the world at no cost and with almost no
restrictions whatsoever.

Title: Sample Letters
Author: Albert Einstein

*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE LETTERS ***

Dear colleague,

I have read your manuscript on the photoelectric effect with great
interest. The proposal seems sound; let us discuss the implications
in person.

Yours sincerely,
A. Einstein

*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE LETTERS ***

Updated editions will replace the previous one--the old editions will
be renamed.
"""
    return text.encode("utf-8")


def build_gutenberg_html_bytes() -> bytes:
    """HTML-rendered Gutenberg-shaped fixture with START/END markers."""

    html = """<!DOCTYPE html>
<html>
<body>
<p>This eBook is for the use of anyone anywhere in the United States.</p>
<p>*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE NOTES ***</p>
<p>Notes on the special theory of relativity.</p>
<p>The principle states that physical laws hold in every inertial
reference frame.</p>
<p>*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE NOTES ***</p>
<p>Trailer text not part of the work.</p>
</body>
</html>
"""
    return html.encode("utf-8")


def build_archive_org_ocr_json_bytes() -> bytes:
    """Internet Archive DjVu OCR JSON fixture, two pages."""

    payload = {
        "metadata": {
            "language": "eng",
            "licenseurl": "https://creativecommons.org/publicdomain/mark/1.0/",
        },
        "ocr": [
            {
                "page": 1,
                "text": "Page one of the lecture transcript.",
                "confidence": 0.91,
            },
            {
                "page": 2,
                "text": "Page two continues the discussion of relativity.",
                "confidence": 0.88,
            },
        ],
    }
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


__all__ = [
    "EINSTEIN_QUOTE",
    "build_archive_org_ocr_json_bytes",
    "build_gutenberg_html_bytes",
    "build_gutenberg_text_bytes",
    "build_minimal_cpae_pdf_bytes",
    "build_wikisource_html_bytes",
]
