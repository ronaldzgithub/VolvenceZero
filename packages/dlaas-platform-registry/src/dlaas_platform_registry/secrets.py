"""Secret generation + hashing for tenant API credentials.

Tenant credentials follow the DLaaS public layout (see
``DLAAS_README.md`` §"Authentication"):

* ``api_key`` — a public identifier prefixed ``tk_``. Returned from
  ``GET`` calls. Compared in clear text on auth.
* ``api_secret`` — a high-entropy secret prefixed ``ts_``. Returned
  ONLY from the create response and never persisted in plaintext;
  ``api_secret_hash`` (SHA-256 of plaintext) is stored instead.

The platform also accepts a control-plane secret and a service
secret, both passed in via environment / configuration. Those are
NOT stored in the registry — they are configured on the api wheel.

We deliberately avoid bcrypt / argon2 / scrypt: the secret has 192
bits of entropy from :func:`secrets.token_urlsafe` so a SHA-256 hash
has no collision risk under any realistic threat model. Keeping the
hash function stdlib-only (``hashlib.sha256``) avoids a native build
dep on every CI environment.
"""

from __future__ import annotations

import hashlib
import secrets


_API_KEY_PREFIX = "tk_"
_API_SECRET_PREFIX = "ts_"


def fresh_tenant_id() -> str:
    """Return a fresh ``ten_<hex>`` tenant identifier."""
    return f"ten_{secrets.token_hex(6)}"


def fresh_api_key() -> str:
    """Return a fresh public API key with the ``tk_`` prefix."""
    return _API_KEY_PREFIX + secrets.token_urlsafe(16)


def fresh_api_secret() -> str:
    """Return a fresh secret with the ``ts_`` prefix.

    Length: 32 bytes of entropy → 43 url-safe chars; comfortably above
    the 192-bit floor needed for a SHA-256 verifier.
    """
    return _API_SECRET_PREFIX + secrets.token_urlsafe(32)


def hash_api_secret(secret: str) -> str:
    """Return the lowercase hex SHA-256 of an API secret."""
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def verify_api_secret(secret: str, expected_hash: str) -> bool:
    """Constant-time compare of a candidate secret against its stored hash."""
    return secrets.compare_digest(hash_api_secret(secret), expected_hash)


__all__ = [
    "fresh_api_key",
    "fresh_api_secret",
    "fresh_tenant_id",
    "hash_api_secret",
    "verify_api_secret",
]
