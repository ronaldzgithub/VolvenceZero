"""Frozen real-substrate stateless baseline for Relationship Lab Gate 0.

The policy receives only the current user message and the closed action
surface.  Pair identity, user history, scene id, split, latent dynamics, and
future outcomes stay in the evaluator.  Each model call is recorded before
the evaluator attaches the expected action, producing a content-addressed
decision ledger and baseline attestation.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Protocol

from lifeform_domain_emogpt.lab import (
    RelationshipAction,
    RelationshipDatasetSplit,
    RelationshipTransferDataset,
    canonical_json,
    load_relationship_transfer_dataset,
    relationship_transfer_package_dir,
    sha256_json,
)
from lifeform_evolution.relationship_lab_gate0 import FrozenBaselineAttestation


if TYPE_CHECKING:
    from lifeform_evolution.relationship_lab_product_baselines import FrozenProductChatMessage


STATELESS_BASELINE_RUN_SCHEMA_VERSION = "relationship-stateless-baseline-run.v1"
STATELESS_BASELINE_DECISION_SCHEMA_VERSION = "relationship-stateless-baseline-decision.v1"
DEFAULT_STATELESS_MODEL_SOURCE = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_STATELESS_MODEL_ID = "qwen2.5-1.5b-instruct"
SCHEMA_CONSTRAINED_DECODING_MODE = "canonical_action_json_token_trie_v1"
_CONSTRAINED_ACTION_ORDER = (
    RelationshipAction.STAY_PRESENT_WITHOUT_PROBE,
    RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION,
    RelationshipAction.NEUTRAL_NOOP,
)


def _asset_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent


def stateless_prompt_path() -> pathlib.Path:
    return _asset_dir() / "prompts" / "relationship_lab_stateless_v1.txt"


def action_choice_schema_path() -> pathlib.Path:
    return _asset_dir() / "schemas" / "relationship_action_choice.schema.json"


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_model_weights_sha256(snapshot_root: pathlib.Path) -> str:
    """Hash only model weight shards using the baseline attestation contract."""

    files = tuple(
        sorted(
            (
                *snapshot_root.glob("*.safetensors"),
                *snapshot_root.glob("pytorch_model*.bin"),
            ),
            key=lambda path: path.name,
        )
    )
    if not files:
        raise FileNotFoundError(f"no model weight files found under frozen snapshot {snapshot_root}")
    manifest = tuple((path.name, path.stat().st_size, _sha256_file(path)) for path in files)
    return sha256_json(manifest)


def _frozen_tokenizer_assets_sha256(snapshot_root: pathlib.Path) -> str:
    """Bind the public tokenizer id to every tokenizer/chat-template asset."""

    fixed_names = {
        "added_tokens.json",
        "chat_template.jinja",
        "merges.txt",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "spiece.model",
        "vocab.json",
        "vocab.txt",
    }
    files = tuple(
        sorted(
            (
                path
                for path in snapshot_root.iterdir()
                if path.is_file() and (path.name.startswith("tokenizer") or path.name in fixed_names)
            ),
            key=lambda path: path.name,
        )
    )
    if not files:
        raise FileNotFoundError(f"no tokenizer assets found under frozen snapshot {snapshot_root}")
    manifest = tuple((path.name, path.stat().st_size, _sha256_file(path)) for path in files)
    return sha256_json(manifest)


def _parse_action_choice(raw_output: str) -> RelationshipAction | None:
    """Parse an exact protocol enum; malformed output remains visibly invalid."""

    try:
        payload = json.loads(raw_output.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or set(payload) != {"action_id"}:
        return None
    action_id = payload["action_id"]
    if not isinstance(action_id, str):
        return None
    try:
        return RelationshipAction(action_id)
    except ValueError:
        return None


@dataclass(frozen=True)
class CanonicalActionJsonTokenConstraint:
    """Exact three-candidate JSON token trie for Hugging Face generation."""

    candidates: tuple[str, ...]
    candidate_token_ids: tuple[tuple[int, ...], ...]
    eos_token_id: int
    mode: str = SCHEMA_CONSTRAINED_DECODING_MODE

    def __post_init__(self) -> None:
        expected = tuple(canonical_json({"action_id": action.value}) for action in _CONSTRAINED_ACTION_ORDER)
        if self.mode != SCHEMA_CONSTRAINED_DECODING_MODE:
            raise ValueError("schema constraint mode mismatch")
        if self.candidates != expected:
            raise ValueError("schema constraint candidates must be the three canonical action JSON values")
        if len(self.candidate_token_ids) != len(self.candidates):
            raise ValueError("schema constraint must bind one token sequence per candidate")
        if isinstance(self.eos_token_id, bool) or not isinstance(self.eos_token_id, int) or self.eos_token_id < 0:
            raise ValueError("schema constraint eos_token_id must be a non-negative integer")
        for sequence in self.candidate_token_ids:
            if not isinstance(sequence, tuple) or not sequence:
                raise ValueError("schema constraint token sequences must be non-empty tuples")
            if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in sequence):
                raise ValueError("schema constraint token ids must be non-negative integers")
            if self.eos_token_id in sequence:
                raise ValueError("schema constraint candidate tokens must not contain EOS")
        if len(set(self.candidate_token_ids)) != len(self.candidate_token_ids):
            raise ValueError("schema constraint candidate token sequences must be unique")
        for left_index, left in enumerate(self.candidate_token_ids):
            for right_index, right in enumerate(self.candidate_token_ids):
                if left_index != right_index and len(left) <= len(right) and right[: len(left)] == left:
                    raise ValueError("schema constraint candidate token sequences must be prefix-free")

    @property
    def maximum_completion_tokens(self) -> int:
        """Maximum candidate tokens including the required terminal EOS."""

        return max(len(sequence) for sequence in self.candidate_token_ids) + 1

    def generation_config_payload(self) -> dict[str, object]:
        """Hashable payload binding the mode, strings, token ids, and terminal."""

        return {
            "mode": self.mode,
            "canonical_candidates": list(self.candidates),
            "candidate_token_ids": [list(sequence) for sequence in self.candidate_token_ids],
            "terminal_eos_token_id": self.eos_token_id,
        }

    def prefix_allowed_tokens_fn(
        self,
        *,
        prompt_token_count: int,
    ) -> Callable[[int, Any], list[int]]:
        """Return a fail-loudly callback over the exact generated-token prefix."""

        if isinstance(prompt_token_count, bool) or not isinstance(prompt_token_count, int) or prompt_token_count <= 0:
            raise ValueError("prompt_token_count must be a positive integer")
        transitions: dict[tuple[int, ...], set[int]] = {}
        for sequence in self.candidate_token_ids:
            for index, token in enumerate(sequence):
                transitions.setdefault(sequence[:index], set()).add(token)
            transitions.setdefault(sequence, set()).add(self.eos_token_id)
        frozen_transitions = {prefix: tuple(sorted(allowed)) for prefix, allowed in transitions.items()}

        def _allowed_tokens(batch_id: int, input_ids: Any) -> list[int]:
            if isinstance(batch_id, bool) or not isinstance(batch_id, int) or batch_id < 0:
                raise ValueError("schema constraint batch_id must be a non-negative integer")
            raw_ids = input_ids.tolist()
            if not isinstance(raw_ids, list) or any(
                isinstance(token, bool) or not isinstance(token, int) for token in raw_ids
            ):
                raise TypeError("schema constraint input_ids must be a one-dimensional integer tensor")
            if len(raw_ids) < prompt_token_count:
                raise ValueError("schema constraint input_ids are shorter than the frozen prompt")
            generated_prefix = tuple(raw_ids[prompt_token_count:])
            allowed = frozen_transitions.get(generated_prefix)
            if allowed is None:
                raise ValueError("generation escaped the canonical action JSON token trie")
            return list(allowed)

        return _allowed_tokens


def build_canonical_action_json_token_constraint(tokenizer: Any) -> CanonicalActionJsonTokenConstraint:
    """Tokenize and verify the exact canonical action JSON candidates."""

    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int) or eos_token_id < 0:
        raise ValueError("tokenizer.eos_token_id must be a non-negative integer")
    candidates = tuple(canonical_json({"action_id": action.value}) for action in _CONSTRAINED_ACTION_ORDER)
    token_sequences: list[tuple[int, ...]] = []
    for candidate in candidates:
        encoded = tokenizer(candidate, add_special_tokens=False)
        input_ids = encoded.get("input_ids")
        if not isinstance(input_ids, list) or not input_ids:
            raise TypeError("canonical action tokenizer output must contain a non-empty input_ids list")
        sequence = tuple(input_ids)
        decoded = tokenizer.decode(input_ids, skip_special_tokens=True)
        if decoded != candidate:
            raise ValueError("canonical action token sequence does not decode exactly")
        token_sequences.append(sequence)
    return CanonicalActionJsonTokenConstraint(
        candidates=candidates,
        candidate_token_ids=tuple(token_sequences),
        eos_token_id=eos_token_id,
    )


@dataclass(frozen=True)
class StatelessActionCompletion:
    raw_output: str
    chosen_action_id: RelationshipAction | None
    prompt_tokens: int
    completion_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.raw_output, str):
            raise ValueError("raw_output must be a string")
        for field_name, value in (
            ("prompt_tokens", self.prompt_tokens),
            ("completion_tokens", self.completion_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")


class StatelessRelationshipActionPolicy(Protocol):
    model_id: str
    weights_sha256: str
    prompt_sha256: str
    generation_config_sha256: str

    def choose(self, *, current_input: str, seed: int) -> StatelessActionCompletion:
        """Choose from current input only; no user/history identifiers are passed."""


class HFStatelessRelationshipActionPolicy:
    """Local frozen Hugging Face policy with strict JSON enum output."""

    def __init__(
        self,
        *,
        model_source: str = DEFAULT_STATELESS_MODEL_SOURCE,
        model_id: str = DEFAULT_STATELESS_MODEL_ID,
        model_revision: str | None = None,
        device: str = "auto",
        torch_dtype: str = "auto",
        local_files_only: bool = True,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_new_tokens: int = 48,
        prefill_chunk_size: int | None = None,
        generation_use_cache: bool | None = None,
        schema_constrained_decoding: bool = False,
    ) -> None:
        if not model_source.strip() or not model_id.strip():
            raise ValueError("model_source and model_id must be non-empty")
        if model_revision is not None and not model_revision.strip():
            raise ValueError("model_revision must be non-empty when provided")
        if temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if max_new_tokens < 4:
            raise ValueError("max_new_tokens must be >= 4")
        if prefill_chunk_size is not None and (
            isinstance(prefill_chunk_size, bool) or not isinstance(prefill_chunk_size, int) or prefill_chunk_size <= 0
        ):
            raise ValueError("prefill_chunk_size must be a positive integer when provided")
        if generation_use_cache is not None and not isinstance(
            generation_use_cache,
            bool,
        ):
            raise TypeError("generation_use_cache must be bool or None")
        if prefill_chunk_size is not None and generation_use_cache is not True:
            raise ValueError("chunked prefill requires generation_use_cache=True")
        if not isinstance(schema_constrained_decoding, bool):
            raise TypeError("schema_constrained_decoding must be bool")

        import torch
        import transformers
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer

        resolved_device = device
        if device == "auto":
            if torch.cuda.is_available():
                resolved_device = "cuda"
            elif torch.backends.mps.is_available():
                resolved_device = "mps"
            else:
                resolved_device = "cpu"
        dtype_name = torch_dtype
        if torch_dtype == "auto":
            dtype_name = "bfloat16" if resolved_device == "cpu" else "float16"
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if dtype_name not in dtype_map:
            raise ValueError(f"torch_dtype must be auto or one of {sorted(dtype_map)}, got {torch_dtype!r}")
        snapshot = pathlib.Path(
            snapshot_download(
                repo_id=model_source,
                revision=model_revision,
                local_files_only=local_files_only,
            )
        )
        prompt_path = stateless_prompt_path()
        schema_path = action_choice_schema_path()
        if not prompt_path.is_file() or not schema_path.is_file():
            raise FileNotFoundError("Relationship Lab prompt/schema assets are missing")

        self.model_id = model_id
        self.model_revision = model_revision
        self.weights_sha256 = frozen_model_weights_sha256(snapshot)
        self.tokenizer_id = f"hf-chat-template:{model_id}@sha256:{_frozen_tokenizer_assets_sha256(snapshot)}"
        self.prompt_sha256 = _sha256_file(prompt_path)
        self._prompt = prompt_path.read_text(encoding="utf-8").strip()
        self._schema_sha256 = _sha256_file(schema_path)
        self._device = resolved_device
        self._temperature = temperature
        self._top_p = top_p
        self._max_new_tokens = max_new_tokens
        self._prefill_chunk_size = prefill_chunk_size
        self._generation_use_cache = generation_use_cache
        self.schema_constrained_decoding = schema_constrained_decoding
        self.max_new_tokens = max_new_tokens
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            snapshot,
            local_files_only=True,
        )
        self._schema_constraint = (
            build_canonical_action_json_token_constraint(self._tokenizer) if schema_constrained_decoding else None
        )
        if self._schema_constraint is not None and max_new_tokens < self._schema_constraint.maximum_completion_tokens:
            raise ValueError("max_new_tokens cannot fit the longest canonical action JSON candidate plus EOS")
        generation_config: dict[str, object] = {
            "device": resolved_device,
            "model_revision": model_revision,
            "torch_dtype": dtype_name,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "schema_sha256": self._schema_sha256,
            "do_sample": temperature > 0.0,
        }
        if generation_use_cache is not None:
            generation_config["generation_use_cache"] = generation_use_cache
        if prefill_chunk_size is not None:
            generation_config.update(
                prefill_chunk_size=prefill_chunk_size,
                torch_version=str(torch.__version__),
                transformers_version=str(transformers.__version__),
            )
        if self._schema_constraint is not None:
            generation_config["schema_constrained_decoding"] = self._schema_constraint.generation_config_payload()
        self.generation_config_sha256 = sha256_json(generation_config)
        self._model = AutoModelForCausalLM.from_pretrained(
            snapshot,
            local_files_only=True,
            dtype=dtype_map[dtype_name],
            low_cpu_mem_usage=True,
        ).to(resolved_device)
        self._model.eval()

    def choose(self, *, current_input: str, seed: int) -> StatelessActionCompletion:
        if not current_input.strip():
            raise ValueError("current_input must be non-empty")
        return self.choose_from_messages(
            messages=(
                {"role": "system", "content": self._prompt},
                {"role": "user", "content": current_input},
            ),
            seed=seed,
        )

    def count_tokens(self, text: str) -> int:
        """Count context tokens with the exact frozen substrate tokenizer."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        encoded = self._tokenizer(text, add_special_tokens=False)
        input_ids = encoded["input_ids"]
        if not isinstance(input_ids, list):
            raise TypeError("tokenizer input_ids must be a list")
        return len(input_ids)

    def count_message_tokens(
        self,
        *,
        messages: tuple[FrozenProductChatMessage, ...],
    ) -> int:
        """Count the exact chat-template ids consumed by ``choose_from_messages``.

        The public product-baseline message type is projected to the policy's
        existing strict payload and then sent through the same rendering and
        tensor-tokenization path used immediately before generation.  No
        external adapter needs access to the tokenizer or other private policy
        state.
        """

        if not isinstance(messages, tuple):
            raise TypeError("messages must be a tuple of FrozenProductChatMessage values")
        from lifeform_evolution.relationship_lab_product_baselines import (
            FrozenProductChatMessage as RuntimeFrozenProductChatMessage,
        )

        if not all(isinstance(message, RuntimeFrozenProductChatMessage) for message in messages):
            raise TypeError("messages must contain only FrozenProductChatMessage values")
        payload = tuple({"role": message.role, "content": message.content} for message in messages)
        encoded = self._encode_contextual_messages(messages=payload)
        return self._encoded_prompt_token_count(encoded)

    @staticmethod
    def _validate_contextual_messages(messages: tuple[dict[str, str], ...]) -> None:
        if not messages:
            raise ValueError("messages must be non-empty")
        allowed_roles = {"system", "user", "assistant"}
        for index, message in enumerate(messages):
            if set(message) != {"role", "content"}:
                raise ValueError(f"messages[{index}] must contain exactly role and content")
            if message["role"] not in allowed_roles:
                raise ValueError(f"messages[{index}].role is unsupported")
            if not message["content"].strip():
                raise ValueError(f"messages[{index}].content must be non-empty")
        if messages[-1]["role"] != "user":
            raise ValueError("the final contextual message must have role=user")

    def _encode_contextual_messages(
        self,
        *,
        messages: tuple[dict[str, str], ...],
    ) -> dict[str, Any]:
        self._validate_contextual_messages(messages)
        rendered = self._tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
        )
        return self._tokenizer(rendered, return_tensors="pt")

    @staticmethod
    def _encoded_prompt_token_count(encoded: dict[str, Any]) -> int:
        if "input_ids" not in encoded:
            raise ValueError("tokenizer output must contain input_ids")
        input_ids = encoded["input_ids"]
        shape = input_ids.shape
        if len(shape) != 2 or int(shape[0]) != 1 or int(shape[-1]) <= 0:
            raise ValueError("tokenizer input_ids must have shape [1, positive_tokens]")
        return int(shape[-1])

    def choose_from_messages(
        self,
        *,
        messages: tuple[dict[str, str], ...],
        seed: int,
    ) -> StatelessActionCompletion:
        """Run a frozen contextual arm through the same model instance."""

        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        encoded = self._encode_contextual_messages(messages=messages)
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        prompt_tokens = self._encoded_prompt_token_count(encoded)
        self._torch.manual_seed(seed)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self._max_new_tokens,
            "do_sample": self._temperature > 0.0,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if self._generation_use_cache is not None:
            generation_kwargs["use_cache"] = self._generation_use_cache
        if self._prefill_chunk_size is not None:
            generation_kwargs["prefill_chunk_size"] = self._prefill_chunk_size
        if self._schema_constraint is not None:
            generation_kwargs.update(
                eos_token_id=self._schema_constraint.eos_token_id,
                prefix_allowed_tokens_fn=self._schema_constraint.prefix_allowed_tokens_fn(
                    prompt_token_count=prompt_tokens
                ),
            )
        if self._temperature > 0.0:
            generation_kwargs.update(
                temperature=self._temperature,
                top_p=self._top_p,
            )
        with self._torch.inference_mode():
            generated = self._model.generate(**encoded, **generation_kwargs)
        completion_ids = generated[0, prompt_tokens:]
        raw_output = self._tokenizer.decode(
            completion_ids,
            skip_special_tokens=True,
        ).strip()
        chosen_action = _parse_action_choice(raw_output)
        if self._schema_constraint is not None and raw_output not in self._schema_constraint.candidates:
            raise RuntimeError("schema-constrained generation did not return a canonical candidate")
        return StatelessActionCompletion(
            raw_output=raw_output[:2000],
            chosen_action_id=chosen_action,
            prompt_tokens=prompt_tokens,
            completion_tokens=int(completion_ids.shape[-1]),
        )


@dataclass(frozen=True)
class StatelessBaselineDecision:
    decision_id: str
    scene_id: str
    pair_id: str
    split: RelationshipDatasetSplit
    seed: int
    current_input_sha256: str
    raw_output: str
    chosen_action_id: RelationshipAction | None
    expected_action_id: RelationshipAction
    valid: bool
    correct: bool
    prompt_tokens: int
    completion_tokens: int
    schema_version: str = STATELESS_BASELINE_DECISION_SCHEMA_VERSION

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "decision_id": self.decision_id,
            "scene_id": self.scene_id,
            "pair_id": self.pair_id,
            "split": self.split.value,
            "seed": self.seed,
            "current_input_sha256": self.current_input_sha256,
            "raw_output": self.raw_output,
            "chosen_action_id": (self.chosen_action_id.value if self.chosen_action_id is not None else None),
            "expected_action_id": self.expected_action_id.value,
            "valid": self.valid,
            "correct": self.correct,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


@dataclass(frozen=True)
class StatelessBaselineRun:
    dataset_fingerprint: str
    model_id: str
    weights_sha256: str
    prompt_sha256: str
    generation_config_sha256: str
    seed_schedule: tuple[int, ...]
    decisions: tuple[StatelessBaselineDecision, ...]
    schema_version: str = STATELESS_BASELINE_RUN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.seed_schedule or len(set(self.seed_schedule)) != len(self.seed_schedule):
            raise ValueError("seed_schedule must be non-empty and unique")
        if not self.decisions:
            raise ValueError("stateless baseline run requires decisions")
        decision_ids = tuple(item.decision_id for item in self.decisions)
        if len(set(decision_ids)) != len(decision_ids):
            raise ValueError("baseline decision ids must be unique")

    @property
    def seed_schedule_sha256(self) -> str:
        return sha256_json(self.seed_schedule)

    def decision_ledger_jsonl(self) -> str:
        return "".join(canonical_json(decision.to_payload()) + "\n" for decision in self.decisions)

    @property
    def decision_ledger_sha256(self) -> str:
        return hashlib.sha256(self.decision_ledger_jsonl().encode("utf-8")).hexdigest()

    @property
    def valid_decisions(self) -> int:
        return sum(int(item.valid) for item in self.decisions)

    @property
    def correct_decisions(self) -> int:
        return sum(int(item.correct) for item in self.decisions)

    @property
    def context_tokens_total(self) -> int:
        return sum(item.prompt_tokens for item in self.decisions)

    def to_summary_payload(self) -> dict[str, object]:
        total = len(self.decisions)
        return {
            "schema_version": self.schema_version,
            "dataset_fingerprint": self.dataset_fingerprint,
            "model_id": self.model_id,
            "weights_sha256": self.weights_sha256,
            "prompt_sha256": self.prompt_sha256,
            "generation_config_sha256": self.generation_config_sha256,
            "seed_schedule": list(self.seed_schedule),
            "seed_schedule_sha256": self.seed_schedule_sha256,
            "decision_ledger_sha256": self.decision_ledger_sha256,
            "valid_decisions": self.valid_decisions,
            "correct_decisions": self.correct_decisions,
            "evaluated_decisions": total,
            "accuracy": self.correct_decisions / total,
            "context_tokens_total": self.context_tokens_total,
        }


def run_stateless_baseline(
    policy: StatelessRelationshipActionPolicy,
    *,
    dataset: RelationshipTransferDataset | None = None,
    seed_schedule: tuple[int, ...] = (101, 211, 307),
) -> StatelessBaselineRun:
    """Run matched current-turn-only calls on train+validation mirrored pairs."""

    effective_dataset = dataset or load_relationship_transfer_dataset()
    if not seed_schedule or len(set(seed_schedule)) != len(seed_schedule):
        raise ValueError("seed_schedule must be non-empty and unique")
    decisions: list[StatelessBaselineDecision] = []
    allowed_splits = {
        RelationshipDatasetSplit.TRAIN,
        RelationshipDatasetSplit.VALIDATION,
    }
    for pair_id, members in effective_dataset.mirrored_pairs():
        split = members[0][1].split
        if split not in allowed_splits:
            continue
        current_input = members[0][0].current_input
        current_input_sha256 = hashlib.sha256(current_input.encode("utf-8")).hexdigest()
        for seed in seed_schedule:
            # One call is shared by the matched pair so identical current bytes
            # and seed cannot accidentally receive different model randomness.
            completion = policy.choose(current_input=current_input, seed=seed)
            for observation, dynamic in members:
                decision_id = sha256_json(
                    {
                        "dataset_fingerprint": effective_dataset.dataset_fingerprint,
                        "model_id": policy.model_id,
                        "pair_id": pair_id,
                        "scene_id": observation.scene_id,
                        "seed": seed,
                    }
                )
                valid = completion.chosen_action_id is not None
                correct = completion.chosen_action_id is dynamic.preferred_action
                decisions.append(
                    StatelessBaselineDecision(
                        decision_id=decision_id,
                        scene_id=observation.scene_id,
                        pair_id=pair_id,
                        split=split,
                        seed=seed,
                        current_input_sha256=current_input_sha256,
                        raw_output=completion.raw_output,
                        chosen_action_id=completion.chosen_action_id,
                        expected_action_id=dynamic.preferred_action,
                        valid=valid,
                        correct=correct,
                        prompt_tokens=completion.prompt_tokens,
                        completion_tokens=completion.completion_tokens,
                    )
                )
    return StatelessBaselineRun(
        dataset_fingerprint=effective_dataset.dataset_fingerprint,
        model_id=policy.model_id,
        weights_sha256=policy.weights_sha256,
        prompt_sha256=policy.prompt_sha256,
        generation_config_sha256=policy.generation_config_sha256,
        seed_schedule=seed_schedule,
        decisions=tuple(decisions),
    )


def freeze_stateless_baseline_attestation(
    run: StatelessBaselineRun,
    *,
    frozen_at_iso: str,
) -> FrozenBaselineAttestation:
    try:
        parsed = datetime.fromisoformat(frozen_at_iso.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("frozen_at_iso must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("frozen_at_iso must include a timezone")
    return FrozenBaselineAttestation(
        arm_id="stateless",
        dataset_fingerprint=run.dataset_fingerprint,
        model_id=run.model_id,
        weights_sha256=run.weights_sha256,
        prompt_sha256=run.prompt_sha256,
        generation_config_sha256=run.generation_config_sha256,
        seed_schedule_sha256=run.seed_schedule_sha256,
        decision_ledger_sha256=run.decision_ledger_sha256,
        evaluated_split="calibration",
        valid_decisions=run.valid_decisions,
        correct_decisions=run.correct_decisions,
        evaluated_decisions=len(run.decisions),
        context_tokens_total=run.context_tokens_total,
        hidden_test_opened=False,
        frozen_at_iso=frozen_at_iso,
    )


def write_stateless_baseline_run(
    run: StatelessBaselineRun,
    *,
    output_dir: pathlib.Path,
    frozen_at_iso: str,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    target = pathlib.Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    ledger_path = target / "decisions.jsonl"
    summary_path = target / "run.json"
    attestation_path = target / "baseline_attestation.json"
    existing = tuple(path for path in (ledger_path, summary_path, attestation_path) if path.exists())
    if existing:
        raise FileExistsError(f"baseline output files already exist: {existing}")
    attestation = freeze_stateless_baseline_attestation(
        run,
        frozen_at_iso=frozen_at_iso,
    )
    with ledger_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(run.decision_ledger_jsonl())
    if _sha256_file(ledger_path) != attestation.decision_ledger_sha256:
        raise RuntimeError("written baseline ledger hash does not match attestation")
    summary_payload = run.to_summary_payload()
    summary_payload["attestation_id"] = attestation.artifact_id
    with summary_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(summary_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    with attestation_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(attestation.to_json())
    return ledger_path, summary_path, attestation_path


def relationship_transfer_prereg_template_path() -> pathlib.Path:
    return relationship_transfer_package_dir() / "prereg_template.json"


__all__ = [
    "DEFAULT_STATELESS_MODEL_ID",
    "DEFAULT_STATELESS_MODEL_SOURCE",
    "HFStatelessRelationshipActionPolicy",
    "STATELESS_BASELINE_DECISION_SCHEMA_VERSION",
    "STATELESS_BASELINE_RUN_SCHEMA_VERSION",
    "StatelessActionCompletion",
    "StatelessBaselineDecision",
    "StatelessBaselineRun",
    "StatelessRelationshipActionPolicy",
    "action_choice_schema_path",
    "frozen_model_weights_sha256",
    "freeze_stateless_baseline_attestation",
    "relationship_transfer_prereg_template_path",
    "run_stateless_baseline",
    "stateless_prompt_path",
    "write_stateless_baseline_run",
]
