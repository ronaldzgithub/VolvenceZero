"""Minimal PII redaction cleaner.

Replaces obvious modern PII patterns with the literal token
``[PII]``. The cleaner is intentionally conservative: it covers the
patterns most likely to leak into a primary-source corpus by mistake
(a curator pasting a scrape that included a contact form footer,
etc.) and explicitly does NOT attempt name / address / SSN detection
— those need a real PII service and are out of scope for the L1
packet.

The cleaner is monotonically non-expanding: every pattern the regex
matches is at least as long as ``[PII]`` (5 chars). Email minimum is
7 chars (``ab@cd.ef``-style 2+1+1+1+2), phone minimum 11 chars,
credit card minimum 13 chars — all comfortably exceed the redaction
token length.

Patterns covered:

* Email addresses (``user@host.tld``)
* International phone numbers with 10+ digits, optional ``+`` /
  ``-`` / ``space`` separators
* 13–19 digit credit-card-style sequences (groups of 4 separated by
  spaces or hyphens, or unbroken)
"""

from __future__ import annotations

import re

OP_VERSION = "1"

REDACTED_TOKEN = "[PII]"

_EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+\-]{2,}@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
)
_PHONE_RE = re.compile(
    r"\+?\d[\d\-\s]{9,}\d",
)
_CREDIT_CARD_RE = re.compile(
    r"\b(?:\d{4}[ -]?){3,4}\d{1,4}\b",
)


def redact_pii(text: str) -> str:
    text = _EMAIL_RE.sub(REDACTED_TOKEN, text)
    text = _CREDIT_CARD_RE.sub(REDACTED_TOKEN, text)
    text = _PHONE_RE.sub(REDACTED_TOKEN, text)
    return text


__all__ = ["OP_VERSION", "REDACTED_TOKEN", "redact_pii"]
