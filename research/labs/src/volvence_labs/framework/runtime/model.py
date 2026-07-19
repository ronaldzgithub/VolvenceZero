"""ModelRuntime: centralized model loading with MPS/CPU selection.

All HF model access goes through this class. Handles:
- Device selection (MPS > CPU, or explicit override)
- dtype management (fp16 on MPS, fp32 on CPU)
- Weight sha registration in CAS
- Lazy loading (model only loaded on first forward call)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from ..env import EnvConfig, load_env
from ..snapshot.weights import ensure_model_downloaded, model_sha_for_revision, register_model_in_cas


def _resolve_device(env: Optional[EnvConfig] = None) -> str:
    """Resolve the best available device."""
    if env is None:
        env = load_env()
    return env.resolve_device()


def _resolve_dtype(device: str, requested: str = "auto") -> torch.dtype:
    """Resolve dtype based on device and request."""
    if requested == "fp32" or requested == "float32":
        return torch.float32
    if requested == "fp16" or requested == "float16":
        return torch.float16
    if requested == "bf16" or requested == "bfloat16":
        return torch.bfloat16
    # auto: fp16 on MPS/CUDA, fp32 on CPU
    if device in ("mps", "cuda"):
        return torch.float16
    return torch.float32


class ModelRuntime:
    """Centralized model runtime for all probes.

    Usage:
        rt = ModelRuntime("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        logits, hidden = rt.forward_lm(input_ids, return_hidden_states=True)
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        *,
        device: str = "auto",
        dtype: str = "auto",
        revision: str = "main",
        env: Optional[EnvConfig] = None,
    ):
        if env is None:
            env = load_env()
        self._env = env
        self._model_id = model_id or env.models.default_llm
        self._revision = revision
        self._device_str = _resolve_device(env) if device == "auto" else device
        self._dtype = _resolve_dtype(self._device_str, dtype)

        # Lazy-loaded
        self._model: Any = None
        self._tokenizer: Any = None
        self._model_sha: Optional[str] = None
        self._local_path: Optional[Path] = None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def device(self) -> str:
        return self._device_str

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def model_sha(self) -> str:
        if self._model_sha is None:
            self._model_sha = model_sha_for_revision(self._model_id, self._revision)
        return self._model_sha

    def ensure_downloaded(self) -> Path:
        """Download model weights if not present. Returns local path."""
        if self._local_path is None:
            self._local_path = ensure_model_downloaded(
                self._model_id, revision=self._revision, env=self._env
            )
        return self._local_path

    def load_model(self) -> None:
        """Load model and tokenizer into memory."""
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        local_path = self.ensure_downloaded()

        self._tokenizer = AutoTokenizer.from_pretrained(
            str(local_path),
            local_files_only=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            str(local_path),
            local_files_only=True,
            dtype=self._dtype,
            device_map=None,  # manual placement
        )
        self._model = self._model.to(self._device_str)
        self._model.eval()

    @property
    def model(self):
        self.load_model()
        return self._model

    @property
    def tokenizer(self):
        self.load_model()
        return self._tokenizer

    @torch.no_grad()
    def forward_lm(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Run LM forward pass.

        Args:
            input_ids: (batch, seq_len) token ids
            attention_mask: optional attention mask
            return_hidden_states: if True, include all hidden states

        Returns:
            dict with keys: "logits", optionally "hidden_states"
        """
        self.load_model()
        input_ids = input_ids.to(self._device_str)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device_str)

        outputs = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=return_hidden_states,
        )

        result = {"logits": outputs.logits.cpu()}
        if return_hidden_states and outputs.hidden_states:
            result["hidden_states"] = tuple(h.cpu() for h in outputs.hidden_states)
        return result

    @torch.no_grad()
    def encode_text(self, texts: list[str], max_length: int = 512) -> dict[str, torch.Tensor]:
        """Tokenize and get last hidden state for a batch of texts.

        Returns dict with "embeddings" (batch, hidden_dim) and "input_ids".
        """
        self.load_model()
        encoded = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoded["input_ids"].to(self._device_str)
        attention_mask = encoded["attention_mask"].to(self._device_str)

        outputs = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Use last hidden state, last token (like sentence embedding)
        last_hidden = outputs.hidden_states[-1]  # (batch, seq, hidden)
        # Gather last non-pad token per sequence
        seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
        embeddings = last_hidden[torch.arange(len(texts)), seq_lengths]  # (batch, hidden)

        return {
            "embeddings": embeddings.cpu(),
            "input_ids": input_ids.cpu(),
            "attention_mask": attention_mask.cpu(),
        }

    @torch.no_grad()
    def get_logits_for_text(self, text: str, max_length: int = 512) -> dict[str, Any]:
        """Get per-token logits for a single text.

        Returns dict with "logits" (seq_len, vocab), "input_ids" (seq_len,), "tokens" (list[str]).
        """
        self.load_model()
        encoded = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = encoded["input_ids"].to(self._device_str)

        outputs = self._model(input_ids=input_ids, output_hidden_states=True)
        logits = outputs.logits[0].cpu()  # (seq_len, vocab)
        hidden = outputs.hidden_states[-1][0].cpu()  # (seq_len, hidden)
        tokens = self._tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())

        return {
            "logits": logits,
            "input_ids": input_ids[0].cpu(),
            "hidden_states": hidden,
            "tokens": tokens,
        }

    def register_sha(self, store) -> str:
        """Register model sha in CAS store."""
        return register_model_in_cas(
            store, self._model_id, self._revision, self._local_path
        )

    def unload(self) -> None:
        """Free model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if self._device_str == "mps":
            torch.mps.empty_cache()
        elif self._device_str == "cuda":
            torch.cuda.empty_cache()
