"""OpenAI-compatible JSON rendering with typed failures and hard budgeting."""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Protocol


class LLMRenderError(RuntimeError):
    """Base class for external renderer failures."""


class LLMAuthenticationError(LLMRenderError):
    """Authentication or authorization is definitively denied."""


class LLMQuotaError(LLMRenderError):
    """Account quota or billing entitlement is definitively denied."""


class LLMTransientError(LLMRenderError):
    """Retryable network or service failure exhausted its attempts."""


class LLMResponseError(LLMRenderError):
    """The endpoint returned a malformed response contract."""


class CostLimitExceeded(LLMRenderError):
    """A planned or observed call would exceed the hard budget."""


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def __post_init__(self) -> None:
        if min(self.prompt_tokens, self.completion_tokens, self.total_tokens) < 0:
            raise ValueError("token counts must be non-negative")
        if self.total_tokens < self.prompt_tokens + self.completion_tokens:
            raise ValueError("total_tokens cannot be below prompt + completion")


@dataclass(frozen=True)
class RateCard:
    input_usd_per_million: float
    output_usd_per_million: float
    currency: str = "USD"

    def __post_init__(self) -> None:
        if self.input_usd_per_million < 0 or self.output_usd_per_million < 0:
            raise ValueError("rate-card prices must be non-negative")
        if self.currency != "USD":
            raise ValueError("v1 cost gate supports USD rate cards only")

    def cost(self, usage: TokenUsage) -> float:
        return (
            usage.prompt_tokens * self.input_usd_per_million + usage.completion_tokens * self.output_usd_per_million
        ) / 1_000_000.0


@dataclass(frozen=True)
class JsonCompletion:
    model_id: str
    request_id: str
    payload_json: str
    usage: TokenUsage
    cost_usd: float

    def payload(self) -> dict[str, object]:
        try:
            decoded = json.loads(self.payload_json)
        except json.JSONDecodeError as error:
            raise LLMResponseError("completion payload_json is invalid") from error
        if not isinstance(decoded, dict):
            raise LLMResponseError("completion JSON root must be an object")
        return decoded


class JsonCompletionClient(Protocol):
    @property
    def model_id(self) -> str: ...

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> JsonCompletion: ...


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    base_url: str
    api_key: str
    model_id: str
    rate_card: RateCard
    timeout_seconds: float = 90.0
    max_output_tokens: int = 4096
    max_attempts: int = 4
    initial_backoff_seconds: float = 1.0

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("base_url must be non-empty")
        if not self.api_key.strip():
            raise ValueError("api_key must be non-empty")
        if not self.model_id.strip():
            raise ValueError("model_id must be non-empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if self.initial_backoff_seconds < 0:
            raise ValueError("initial_backoff_seconds must be non-negative")


class OpenAICompatibleJsonClient:
    """Synchronous JSON-mode client with bounded, status-aware retries."""

    def __init__(self, config: OpenAICompatibleConfig) -> None:
        self._config = config

    @property
    def model_id(self) -> str:
        return self._config.model_id

    @property
    def max_output_tokens(self) -> int:
        return self._config.max_output_tokens

    @property
    def rate_card(self) -> RateCard:
        return self._config.rate_card

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> JsonCompletion:
        body = json.dumps(
            {
                "model": self._config.model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.0,
                "max_tokens": self._config.max_output_tokens,
            },
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        url = self._config.base_url.rstrip("/") + "/chat/completions"
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._config.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        last_transient: LLMTransientError | None = None
        for attempt in range(self._config.max_attempts):
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self._config.timeout_seconds,
                ) as response:
                    response_body = response.read().decode("utf-8")
                    request_id = response.headers.get("x-request-id") or response.headers.get("request-id") or ""
                return self._parse_completion(response_body, request_id=request_id)
            except urllib.error.HTTPError as error:
                body_text = error.read().decode("utf-8", errors="replace")
                status = int(error.code)
                if status in {401, 403}:
                    raise LLMAuthenticationError(f"LLM endpoint denied credentials with HTTP {status}") from error
                if status in {402} or _is_quota_denial(body_text):
                    raise LLMQuotaError(f"LLM endpoint denied quota/billing with HTTP {status}") from error
                if status not in {408, 409, 429, 500, 502, 503, 504}:
                    raise LLMRenderError(
                        f"LLM endpoint returned non-retryable HTTP {status}: {body_text[:500]}"
                    ) from error
                last_transient = LLMTransientError(f"retryable HTTP {status}: {body_text[:500]}")
            except (urllib.error.URLError, TimeoutError, socket.timeout) as error:
                last_transient = LLMTransientError(f"transient LLM network failure: {error}")
            if attempt + 1 < self._config.max_attempts:
                time.sleep(self._config.initial_backoff_seconds * (2**attempt))
        if last_transient is None:
            raise LLMTransientError("LLM request failed without diagnostic context")
        raise last_transient

    def _parse_completion(
        self,
        response_body: str,
        *,
        request_id: str,
    ) -> JsonCompletion:
        try:
            envelope = json.loads(response_body)
        except json.JSONDecodeError as error:
            raise LLMResponseError(f"LLM endpoint returned non-JSON envelope: {response_body[:500]}") from error
        if not isinstance(envelope, dict):
            raise LLMResponseError("LLM response envelope must be an object")
        choices = envelope.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMResponseError("LLM response missing non-empty choices")
        first = choices[0]
        if not isinstance(first, dict):
            raise LLMResponseError("LLM choices[0] must be an object")
        message = first.get("message")
        if not isinstance(message, dict):
            raise LLMResponseError("LLM choices[0].message must be an object")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMResponseError("LLM message.content must be a non-empty string")
        payload = _decode_json_content(content)
        usage = _parse_usage(envelope.get("usage"))
        resolved_model = envelope.get("model", self._config.model_id)
        if not isinstance(resolved_model, str) or not resolved_model.strip():
            raise LLMResponseError("LLM response model must be a non-empty string")
        resolved_request_id = request_id or _optional_response_id(envelope)
        return JsonCompletion(
            model_id=resolved_model,
            request_id=resolved_request_id,
            payload_json=json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            usage=usage,
            cost_usd=self._config.rate_card.cost(usage),
        )


@dataclass(frozen=True)
class BudgetReservation:
    reservation_id: str
    prompt_token_bound: int
    completion_token_bound: int
    reserved_usd: float


@dataclass(frozen=True)
class BudgetSnapshot:
    max_cost_usd: float
    settled_cost_usd: float
    reserved_cost_usd: float
    calls_settled: int
    prompt_tokens: int
    completion_tokens: int


class BudgetLedger:
    """Thread-safe pre-call reservation ledger enforcing a hard USD cap."""

    def __init__(self, *, max_cost_usd: float, rate_card: RateCard) -> None:
        if max_cost_usd < 0:
            raise ValueError("max_cost_usd must be non-negative")
        self._max_cost_usd = max_cost_usd
        self._rate_card = rate_card
        self._settled_cost_usd = 0.0
        self._reserved_cost_usd = 0.0
        self._calls_settled = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._reservations: dict[str, BudgetReservation] = {}
        self._lock = threading.Lock()

    def reserve(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
    ) -> BudgetReservation:
        prompt_bound = _conservative_token_bound(system_prompt, user_prompt)
        usage_bound = TokenUsage(
            prompt_tokens=prompt_bound,
            completion_tokens=max_output_tokens,
            total_tokens=prompt_bound + max_output_tokens,
        )
        reserved_usd = self._rate_card.cost(usage_bound)
        reservation = BudgetReservation(
            reservation_id=f"budget:{uuid.uuid4().hex}",
            prompt_token_bound=prompt_bound,
            completion_token_bound=max_output_tokens,
            reserved_usd=reserved_usd,
        )
        with self._lock:
            projected = self._settled_cost_usd + self._reserved_cost_usd + reservation.reserved_usd
            if projected > self._max_cost_usd + 1e-12:
                raise CostLimitExceeded(
                    f"planned call would raise cost to ${projected:.6f}, above --max-cost-usd ${self._max_cost_usd:.6f}"
                )
            self._reservations[reservation.reservation_id] = reservation
            self._reserved_cost_usd += reservation.reserved_usd
        return reservation

    def restore(self, *, usage: TokenUsage, calls: int) -> None:
        """Restore already-paid resume records before reserving new calls."""

        if calls < 0:
            raise ValueError("calls must be non-negative")
        restored_cost = self._rate_card.cost(usage)
        with self._lock:
            if self._reservations or self._calls_settled or self._settled_cost_usd:
                raise ValueError("budget ledger can only be restored once while empty")
            if restored_cost > self._max_cost_usd + 1e-12:
                raise CostLimitExceeded("resume journal already exceeds --max-cost-usd")
            self._settled_cost_usd = restored_cost
            self._calls_settled = calls
            self._prompt_tokens = usage.prompt_tokens
            self._completion_tokens = usage.completion_tokens

    def settle(
        self,
        reservation: BudgetReservation,
        completion: JsonCompletion,
    ) -> None:
        with self._lock:
            stored = self._reservations.pop(reservation.reservation_id, None)
            if stored is None:
                raise ValueError("unknown or already settled budget reservation")
            self._reserved_cost_usd -= stored.reserved_usd
            usage = completion.usage
            if (
                usage.prompt_tokens > stored.prompt_token_bound
                or usage.completion_tokens > stored.completion_token_bound
            ):
                raise CostLimitExceeded("endpoint-reported token usage exceeded the pre-call hard bound")
            actual_cost = self._rate_card.cost(usage)
            projected = self._settled_cost_usd + actual_cost
            if projected > self._max_cost_usd + 1e-12:
                raise CostLimitExceeded("endpoint-reported cost exceeded the hard budget reservation")
            self._settled_cost_usd = projected
            self._calls_settled += 1
            self._prompt_tokens += usage.prompt_tokens
            self._completion_tokens += usage.completion_tokens

    def release(self, reservation: BudgetReservation) -> None:
        with self._lock:
            stored = self._reservations.pop(reservation.reservation_id, None)
            if stored is None:
                raise ValueError("unknown or already released budget reservation")
            self._reserved_cost_usd -= stored.reserved_usd

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return BudgetSnapshot(
                max_cost_usd=self._max_cost_usd,
                settled_cost_usd=self._settled_cost_usd,
                reserved_cost_usd=self._reserved_cost_usd,
                calls_settled=self._calls_settled,
                prompt_tokens=self._prompt_tokens,
                completion_tokens=self._completion_tokens,
            )


def estimate_upper_bound_usd(
    *,
    system_prompt: str,
    user_prompts: tuple[str, ...],
    max_output_tokens: int,
    rate_card: RateCard,
) -> float:
    total = 0.0
    for user_prompt in user_prompts:
        prompt_bound = _conservative_token_bound(system_prompt, user_prompt)
        total += rate_card.cost(
            TokenUsage(
                prompt_tokens=prompt_bound,
                completion_tokens=max_output_tokens,
                total_tokens=prompt_bound + max_output_tokens,
            )
        )
    return total


def _decode_json_content(content: str) -> dict[str, object]:
    candidate = content.strip()
    if candidate.startswith("```"):
        newline = candidate.find("\n")
        if newline < 0:
            raise LLMResponseError("markdown-fenced JSON has no body")
        candidate = candidate[newline + 1 :]
        if candidate.endswith("```"):
            candidate = candidate[:-3].rstrip()
    try:
        decoded = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise LLMResponseError(f"LLM message content is not valid JSON: {candidate[:500]}") from error
    if not isinstance(decoded, dict):
        raise LLMResponseError("LLM message JSON root must be an object")
    return decoded


def _parse_usage(value: object) -> TokenUsage:
    if not isinstance(value, dict):
        raise LLMResponseError("LLM response missing usage object")
    prompt = value.get("prompt_tokens")
    completion = value.get("completion_tokens")
    total = value.get("total_tokens")
    if type(prompt) is not int or type(completion) is not int or type(total) is not int:
        raise LLMResponseError("LLM usage token counts must be integers")
    try:
        return TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
        )
    except ValueError as error:
        raise LLMResponseError("LLM usage token counts are invalid") from error


def _optional_response_id(envelope: dict[str, object]) -> str:
    response_id = envelope.get("id")
    if isinstance(response_id, str) and response_id.strip():
        return response_id
    return "request-id-unavailable"


def _is_quota_denial(body_text: str) -> bool:
    try:
        decoded = json.loads(body_text)
    except json.JSONDecodeError:
        return False
    if not isinstance(decoded, dict):
        return False
    error = decoded.get("error")
    if not isinstance(error, dict):
        return False
    error_type = error.get("type")
    error_code = error.get("code")
    return error_type == "insufficient_quota" or error_code == "insufficient_quota"


def _conservative_token_bound(system_prompt: str, user_prompt: str) -> int:
    # A tokenizer cannot emit more symbols than the UTF-8 bytes supplied.
    # Add fixed chat-envelope overhead so budget reservations are conservative.
    return len(system_prompt.encode("utf-8")) + len(user_prompt.encode("utf-8")) + 512


__all__ = [
    "BudgetLedger",
    "BudgetReservation",
    "BudgetSnapshot",
    "CostLimitExceeded",
    "JsonCompletion",
    "JsonCompletionClient",
    "LLMAuthenticationError",
    "LLMQuotaError",
    "LLMRenderError",
    "LLMResponseError",
    "LLMTransientError",
    "OpenAICompatibleConfig",
    "OpenAICompatibleJsonClient",
    "RateCard",
    "TokenUsage",
    "estimate_upper_bound_usd",
]
