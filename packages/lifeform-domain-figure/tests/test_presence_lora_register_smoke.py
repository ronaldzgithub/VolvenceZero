"""R3-5 smoke tests for :mod:`presence_lora_register`.

The HTTP layer is mocked via ``unittest.mock.patch``; we verify the
signed payload shape, the HMAC signature (which must match the
deploy-side ``requireInternalWriter`` contract), and the
fire-and-forget failure semantics.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import urllib.error
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from lifeform_domain_figure.presence_lora_register import (
    PresenceLoraRegistration,
    register_lora_into_presence,
    revoke_lora_from_presence,
)


_SECRET = "test-internal-secret-of-sufficient-length"


def _mock_response(status: int = 201) -> MagicMock:
    resp = MagicMock()
    resp.status = status
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def _capture_request(mock_urlopen: MagicMock) -> tuple[Any, dict[str, str]]:
    assert mock_urlopen.call_count == 1
    request = mock_urlopen.call_args.args[0]
    headers = {k.lower(): v for k, v in request.header_items()}
    return request, headers


def test_register_signs_request_and_returns_true_on_2xx() -> None:
    registration = PresenceLoraRegistration(
        persona_identifier="einstein:einstein-default",
        lora_fingerprint="sha256:abcdef0123456789",
        bundle_id="bundle_v1",
        license_label="public-domain",
        layer="persona",
    )

    with patch(
        "lifeform_domain_figure.presence_lora_register.urllib.request.urlopen",
        return_value=_mock_response(201),
    ) as mock_urlopen:
        ok = register_lora_into_presence(
            registration=registration,
            presence_base_url="https://presence.example.com/",
            internal_secret=_SECRET,
        )

    assert ok is True
    request, headers = _capture_request(mock_urlopen)

    assert (
        request.full_url
        == "https://presence.example.com/api/v1/internal/personas/einstein%3Aeinstein-default/lora"
    )
    assert request.method == "POST"

    body = request.data
    parsed = json.loads(body.decode("utf-8"))
    assert parsed == {
        "bundle_id": "bundle_v1",
        "layer": "persona",
        "license_label": "public-domain",
        "lora_fingerprint": "sha256:abcdef0123456789",
    }

    header_value = headers["x-presence-internal"]
    assert header_value.startswith("t=")
    parts = dict(p.split("=", 1) for p in header_value.split(";"))
    ts = parts["t"]
    sig = parts["sig"]

    body_hash = hashlib.sha256(body).hexdigest()
    path = "/api/v1/internal/personas/einstein%3Aeinstein-default/lora"
    expected_message = f"{ts}:POST:{path}:{body_hash}".encode("utf-8")
    expected_sig = hmac.new(
        _SECRET.encode("utf-8"), expected_message, hashlib.sha256
    ).hexdigest()
    assert sig == expected_sig


def test_register_returns_false_on_http_error() -> None:
    registration = PresenceLoraRegistration(
        persona_identifier="p1",
        lora_fingerprint="sha256:abcdef0123456789",
        bundle_id="bundle_v1",
        license_label="public-domain",
        layer="style",
    )
    with patch(
        "lifeform_domain_figure.presence_lora_register.urllib.request.urlopen",
        side_effect=urllib.error.HTTPError(
            "https://x", 500, "boom", hdrs=None, fp=None  # type: ignore[arg-type]
        ),
    ):
        ok = register_lora_into_presence(
            registration=registration,
            presence_base_url="https://presence.example.com",
            internal_secret=_SECRET,
        )
    assert ok is False


def test_register_returns_false_on_transport_error() -> None:
    registration = PresenceLoraRegistration(
        persona_identifier="p1",
        lora_fingerprint="sha256:abcdef0123456789",
        bundle_id="bundle_v1",
        license_label="public-domain",
        layer="style",
    )
    with patch(
        "lifeform_domain_figure.presence_lora_register.urllib.request.urlopen",
        side_effect=urllib.error.URLError("no route"),
    ):
        ok = register_lora_into_presence(
            registration=registration,
            presence_base_url="https://presence.example.com",
            internal_secret=_SECRET,
        )
    assert ok is False


def test_revoke_treats_404_as_success() -> None:
    with patch(
        "lifeform_domain_figure.presence_lora_register.urllib.request.urlopen",
        side_effect=urllib.error.HTTPError(
            "https://x", 404, "not found", hdrs=None, fp=None  # type: ignore[arg-type]
        ),
    ):
        ok = revoke_lora_from_presence(
            persona_identifier="p1",
            lora_fingerprint="sha256:abcdef0123456789",
            presence_base_url="https://presence.example.com",
            internal_secret=_SECRET,
        )
    assert ok is True
