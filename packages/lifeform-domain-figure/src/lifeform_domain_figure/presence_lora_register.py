"""R3-5 — register a baked :class:`FigureLoRAArtifact` fingerprint with
the deploy-side presence-service registry.

Why this lives here:

* Weights (real or synthetic) stay inside the VolvenceZero figure
  bundle (``FigureLoRAArtifact.adapter_layers`` /
  ``peft_checkpoint_dir``). The presence-service NEVER stores LoRA
  bytes; it only needs to know **which** LoRA fingerprint a persona is
  currently bound to so dashboards can audit the binding and so a
  consent revoke can fast-path "what bytes are in play".
* This helper does an HMAC-signed POST to
  ``/api/v1/internal/personas/<id>/lora`` using ``PRESENCE_INTERNAL_SECRET``
  (shared with the visual-service bridge). Failure is non-fatal: a
  bake that successfully produced :class:`FigureLoRAArtifact` is the
  authoritative event; presence-side registration is a downstream
  notification, not a precondition.

R15: rollback / revoke is a separate ``revoke_lora_from_presence``
DELETE call so a bake that turns out to be revoked does not require
another bake to roll back.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Literal, Optional
from urllib.parse import quote

from lifeform_domain_figure.lora_artifact import FigureLoRAArtifact


LoRALayerTag = Literal["style", "persona", "scope", "voice", "motion"]


@dataclass(frozen=True)
class PresenceLoraRegistration:
    """Resolved payload sent to the deploy-side presence-service."""

    persona_identifier: str
    """Either a presence-service DB id, or ``"<app-slug>:<external-ref>"``."""

    lora_fingerprint: str
    bundle_id: str
    license_label: str
    layer: LoRALayerTag


_HEADER_NAME = "X-Presence-Internal"
_HEADER_NAME_LOWER = "x-presence-internal"
_DEFAULT_TIMEOUT_S = 5.0


def _sign_request(
    *,
    secret: str,
    timestamp_ms: int,
    verb: str,
    path: str,
    body_bytes: bytes,
) -> str:
    body_hash = hashlib.sha256(body_bytes).hexdigest()
    message = f"{timestamp_ms}:{verb}:{path}:{body_hash}".encode("utf-8")
    return hmac.new(
        secret.encode("utf-8"), message, hashlib.sha256
    ).hexdigest()


def build_registration_from_artifact(
    *,
    artifact: FigureLoRAArtifact,
    persona_identifier: str,
    license_label: str,
    layer: LoRALayerTag = "persona",
    bundle_id: Optional[str] = None,
) -> PresenceLoraRegistration:
    """Project a :class:`FigureLoRAArtifact` into the presence wire payload.

    ``bundle_id`` defaults to ``artifact.figure_id``; callers that
    attached the artifact to a richer :class:`FigureArtifactBundle`
    should pass the bundle id explicitly so presence can join across
    the figure timeline.
    """

    return PresenceLoraRegistration(
        persona_identifier=persona_identifier,
        lora_fingerprint=artifact.integrity_hash,
        bundle_id=bundle_id or artifact.figure_id,
        license_label=license_label,
        layer=layer,
    )


def register_lora_into_presence(
    *,
    registration: PresenceLoraRegistration,
    presence_base_url: str,
    internal_secret: str,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """POST the LoRA fingerprint to presence-service. Returns ``True`` on 2xx.

    Never raises; logs at warning level on failure. Callers that want
    a hard-fail can check the return value, but the bake CLI invokes
    this fire-and-forget after a successful write so a temporary
    presence outage does not block model promotion.
    """

    log = logger or logging.getLogger(__name__)
    encoded_id = quote(registration.persona_identifier, safe="")
    path = f"/api/v1/internal/personas/{encoded_id}/lora"
    body = json.dumps(
        {
            "lora_fingerprint": registration.lora_fingerprint,
            "bundle_id": registration.bundle_id,
            "license_label": registration.license_label,
            "layer": registration.layer,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    timestamp_ms = int(time.time() * 1000)
    signature = _sign_request(
        secret=internal_secret,
        timestamp_ms=timestamp_ms,
        verb="POST",
        path=path,
        body_bytes=body,
    )
    headers = {
        "content-type": "application/json",
        _HEADER_NAME: f"t={timestamp_ms};sig={signature}",
    }
    url = presence_base_url.rstrip("/") + path
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status = resp.status
            if 200 <= status < 300:
                log.info(
                    "presence-lora-register: ok persona=%s status=%s",
                    registration.persona_identifier,
                    status,
                )
                return True
            log.warning(
                "presence-lora-register: non-2xx persona=%s status=%s",
                registration.persona_identifier,
                status,
            )
            return False
    except urllib.error.HTTPError as err:
        log.warning(
            "presence-lora-register: http error persona=%s status=%s",
            registration.persona_identifier,
            err.code,
        )
        return False
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        log.warning(
            "presence-lora-register: transport error persona=%s err=%s",
            registration.persona_identifier,
            err,
        )
        return False


def revoke_lora_from_presence(
    *,
    persona_identifier: str,
    lora_fingerprint: str,
    presence_base_url: str,
    internal_secret: str,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Best-effort DELETE counterpart to :func:`register_lora_into_presence`."""

    log = logger or logging.getLogger(__name__)
    encoded_id = quote(persona_identifier, safe="")
    path = f"/api/v1/internal/personas/{encoded_id}/lora"
    body = json.dumps(
        {"lora_fingerprint": lora_fingerprint},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    timestamp_ms = int(time.time() * 1000)
    signature = _sign_request(
        secret=internal_secret,
        timestamp_ms=timestamp_ms,
        verb="DELETE",
        path=path,
        body_bytes=body,
    )
    headers = {
        "content-type": "application/json",
        _HEADER_NAME: f"t={timestamp_ms};sig={signature}",
    }
    url = presence_base_url.rstrip("/") + path
    req = urllib.request.Request(url, data=body, headers=headers, method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as err:
        # 404 is "nothing to revoke" — treat as success.
        if err.code == 404:
            return True
        log.warning(
            "presence-lora-revoke: http error persona=%s status=%s",
            persona_identifier,
            err.code,
        )
        return False
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        log.warning(
            "presence-lora-revoke: transport error persona=%s err=%s",
            persona_identifier,
            err,
        )
        return False


__all__ = [
    "PresenceLoraRegistration",
    "build_registration_from_artifact",
    "register_lora_into_presence",
    "revoke_lora_from_presence",
]
