"""Shared immutable contracts and infrastructure for the Forge package."""

from __future__ import annotations

import hashlib
import json
import os
import string
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence

import jsonschema
import numpy as np


class ForgeError(RuntimeError):
    """Base error for an explicit Forge contract failure."""


class SchemaContractError(ForgeError):
    """Raised when an artifact or LLM result does not match its schema."""


class BackendError(ForgeError):
    """Raised when a configured semantic backend cannot produce a valid result."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ForgeError(f"Missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ForgeError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ForgeError(f"Expected JSON object in {path}")
    return raw


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    except OSError as exc:
        if temporary.exists():
            temporary.unlink()
        raise ForgeError(f"Failed to write {path}: {exc}") from exc


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


class SchemaStore:
    def __init__(self, schema_root: Path) -> None:
        self._root = schema_root

    def load(self, name: str) -> dict[str, Any]:
        path = self._root / name
        return read_json(path)

    def validate(self, value: Any, name: str) -> None:
        schema = self.load(name)
        try:
            jsonschema.Draft202012Validator(schema).validate(value)
        except jsonschema.ValidationError as exc:
            location = ".".join(str(part) for part in exc.absolute_path) or "<root>"
            raise SchemaContractError(f"{name} validation failed at {location}: {exc.message}") from exc


class PromptStore:
    def __init__(self, prompt_root: Path) -> None:
        self._root = prompt_root

    def render(self, name: str, **values: str) -> str:
        path = self._root / name
        try:
            template = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ForgeError(f"Missing prompt template: {path}") from exc
        fields = {field for _, field, _, _ in string.Formatter().parse(template) if field}
        missing = fields - values.keys()
        extra = values.keys() - fields
        if missing or extra:
            raise ForgeError(f"Prompt variables mismatch for {name}: missing={sorted(missing)}, extra={sorted(extra)}")
        return template.format(**values)


class StructuredBackend(Protocol):
    @property
    def backend_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    def complete_json(self, *, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]: ...


class EmbeddingBackend(Protocol):
    @property
    def model_name(self) -> str: ...

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass
class ReplayStructuredBackend:
    """Deterministic backend for tests and reproducible offline evidence drills."""

    responses: list[dict[str, Any]]
    model_name: str = "replay-fixture"
    backend_name: str = "replay"
    _cursor: int = 0

    @classmethod
    def from_path(cls, path: Path) -> ReplayStructuredBackend:
        raw = read_json(path)
        responses = raw.get("responses")
        if not isinstance(responses, list) or not all(isinstance(item, dict) for item in responses):
            raise BackendError(f"Replay file {path} must contain an object-list field named responses")
        model = raw.get("model", "replay-fixture")
        if not isinstance(model, str) or not model:
            raise BackendError(f"Replay file {path} has invalid model")
        return cls(responses=list(responses), model_name=model)

    def complete_json(self, *, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        del system, user
        if self._cursor >= len(self.responses):
            raise BackendError("Replay backend exhausted before workflow completion")
        response = self.responses[self._cursor]
        self._cursor += 1
        try:
            jsonschema.Draft202012Validator(schema).validate(response)
        except jsonschema.ValidationError as exc:
            raise BackendError(f"Replay response {self._cursor - 1} violates schema: {exc.message}") from exc
        return response


@dataclass(frozen=True)
class OpenAICompatibleBackend:
    base_url: str
    api_key: str
    model_name: str
    timeout_seconds: int = 120
    backend_name: str = "openai-compatible"

    @classmethod
    def from_env(cls) -> OpenAICompatibleBackend:
        base_url = os.environ.get("FORGE_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        api_key = os.environ.get("FORGE_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        model = os.environ.get("FORGE_LLM_MODEL")
        if not api_key:
            raise BackendError("FORGE_LLM_API_KEY or OPENAI_API_KEY is required for the OpenAI-compatible backend")
        if not model:
            raise BackendError("FORGE_LLM_MODEL is required; Forge does not guess a mutable default model")
        return cls(base_url=base_url, api_key=api_key, model_name=model)

    def complete_json(self, *, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "forge_response", "strict": True, "schema": schema},
            },
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=canonical_json(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise BackendError(f"OpenAI-compatible backend HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise BackendError(f"OpenAI-compatible backend connection failed: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise BackendError(f"OpenAI-compatible backend returned invalid JSON: {exc}") from exc
        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError("OpenAI-compatible response is missing choices[0].message.content") from exc
        if not isinstance(content, str):
            raise BackendError("OpenAI-compatible response content must be a JSON string")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise BackendError(f"Structured response content is not JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise BackendError("Structured response must be a JSON object")
        try:
            jsonschema.Draft202012Validator(schema).validate(parsed)
        except jsonschema.ValidationError as exc:
            raise BackendError(f"Structured response violates schema: {exc.message}") from exc
        return parsed


class SentenceTransformerEmbeddingBackend:
    def __init__(self, model_name: str, *, device: str = "cpu", local_files_only: bool = True) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise BackendError("Install volvence-forge[embed] to use semantic clustering") from exc
        try:
            self._model = SentenceTransformer(
                model_name,
                device=device,
                local_files_only=local_files_only,
            )
        except (OSError, ValueError) as exc:
            raise BackendError(f"Cannot load embedding model {model_name!r}: {exc}") from exc
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float64)
        vectors = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        array = np.asarray(vectors, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] != len(texts):
            raise BackendError(f"Embedding backend returned invalid shape {array.shape} for {len(texts)} texts")
        if not np.isfinite(array).all():
            raise BackendError("Embedding backend returned non-finite values")
        return array


def normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise BackendError("Semantic embedding has zero norm")
    return vector / norm


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(normalized(left), normalized(right)))
