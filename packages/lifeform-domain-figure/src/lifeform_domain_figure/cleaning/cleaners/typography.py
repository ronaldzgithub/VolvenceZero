"""Typography normalisation cleaner.

* Curly quotes (Unicode 2018-201F) → straight ASCII equivalents.
* En / em dashes → ASCII hyphen-minus.
* Hyphenated line breaks (``"experi-\nment"``) re-joined into one
  word (``"experiment"``).
* Non-breaking spaces (U+00A0) → regular space.

Length must not increase, so the hyphen-rejoin is a length reduction
(strips the hyphen + newline) and dash / quote remaps preserve length
exactly. NBSP-to-space is also length-preserving.
"""

from __future__ import annotations

import re

OP_VERSION = "1"

_QUOTE_MAP = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201A": "'",
        "\u201B": "'",
        "\u201C": '"',
        "\u201D": '"',
        "\u201E": '"',
        "\u201F": '"',
        "\u00A0": " ",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
)

_HYPHEN_LINEBREAK_RE = re.compile(r"([A-Za-z])-\n([a-z])")


def normalise_typography(text: str) -> str:
    text = text.translate(_QUOTE_MAP)
    text = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)
    return text


__all__ = ["OP_VERSION", "normalise_typography"]
