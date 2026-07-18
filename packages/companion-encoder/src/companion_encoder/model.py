# Copyright 2026 Companion Standard Contributors
# Licensed under the Apache License, Version 2.0.

"""Two-head relationship encoder (requires the ``[train]`` extra).

Architecture::

    text prefix -> backbone -> pooled hidden state
                                 ├─ phase head       (8-way logits)
                                 ├─ regression head  (trust/continuity/repair, sigmoid)
                                 └─ embedding head   (L2-normalized vector)

Backbones:

* ``tiny`` — from-scratch byte-level transformer. Exists so the full
  chain (data -> train -> G2 eval) dry-runs on CPU/MPS with zero
  downloads. Explicitly not a release candidate.
* ``hf:<model_id>`` — a Hugging Face LM as backbone (mean-pooled last
  hidden state). The intended M2 path; requires the ``[hf]`` extra.

Truncation policy: inputs longer than ``max_input_bytes`` keep the
*most recent* bytes (left truncation). Relationship state at an anchor
is dominated by recent sessions plus gap structure, and the serialized
session headers survive truncation for all but the earliest sessions.
"""

from __future__ import annotations

import dataclasses
import math
import pathlib

import torch
from torch import nn

from companion_encoder.dataset import PHASE_VOCAB, REGRESSION_TARGETS

PAD_ID = 256  # byte values occupy 0..255
VOCAB_SIZE = 257


@dataclasses.dataclass(frozen=True)
class EncoderConfig:
    backbone: str = "tiny"           # "tiny" | "hf:<model_id>"
    hidden_dim: int = 128            # tiny backbone width
    num_layers: int = 2              # tiny backbone depth
    num_attention_heads: int = 4     # tiny backbone heads
    max_input_bytes: int = 4096
    embedding_dim: int = 64          # output embedding head width
    dropout: float = 0.1

    def to_jsonable(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_jsonable(data: dict) -> "EncoderConfig":
        return EncoderConfig(**data)


def encode_bytes(text: str, *, max_input_bytes: int) -> list[int]:
    """UTF-8 bytes, left-truncated to keep the most recent context."""
    raw = text.encode("utf-8")
    if len(raw) > max_input_bytes:
        raw = raw[-max_input_bytes:]
    return list(raw)


def collate_byte_batch(
    texts: list[str], *, max_input_bytes: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-padded id tensor + boolean padding mask (True = padding)."""
    encoded = [encode_bytes(text, max_input_bytes=max_input_bytes) for text in texts]
    width = max(len(ids) for ids in encoded)
    batch = torch.full((len(encoded), width), PAD_ID, dtype=torch.long)
    for row, ids in enumerate(encoded):
        batch[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    padding_mask = batch.eq(PAD_ID)
    return batch.to(device), padding_mask.to(device)


class TinyByteBackbone(nn.Module):
    """Byte-level transformer encoder with sinusoidal positions + mean pool."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.byte_embedding = nn.Embedding(VOCAB_SIZE, config.hidden_dim, padding_idx=PAD_ID)
        position = torch.arange(config.max_input_bytes).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, config.hidden_dim, 2) * (-math.log(10000.0) / config.hidden_dim)
        )
        positional = torch.zeros(config.max_input_bytes, config.hidden_dim)
        positional[:, 0::2] = torch.sin(position * div_term)
        positional[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("positional", positional, persistent=False)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=config.num_layers, enable_nested_tensor=False
        )
        self.output_dim = config.hidden_dim

    def forward(self, texts: list[str]) -> torch.Tensor:
        device = self.byte_embedding.weight.device
        ids, padding_mask = collate_byte_batch(
            texts, max_input_bytes=self.config.max_input_bytes, device=device
        )
        hidden = self.byte_embedding(ids) + self.positional[: ids.shape[1]].unsqueeze(0)
        hidden = self.transformer(hidden, src_key_padding_mask=padding_mask)
        keep = (~padding_mask).unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * keep).sum(dim=1)
        counts = keep.sum(dim=1).clamp(min=1.0)
        return summed / counts


class HFBackbone(nn.Module):
    """Hugging Face LM backbone (mean-pooled last hidden state).

    Lazy-imports ``transformers`` so the wheel stays importable without
    the ``[hf]`` extra; instantiating without it fails loudly.
    """

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        model_id = config.backbone.removeprefix("hf:")
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "hf:* backbones require the [hf] extra "
                "(pip install 'companion-encoder[hf]')"
            ) from error
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(model_id)
        self.output_dim = int(self.model.config.hidden_size)

    def forward(self, texts: list[str]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_input_bytes // 4,
        ).to(device)
        hidden = self.model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def build_backbone(config: EncoderConfig) -> nn.Module:
    if config.backbone == "tiny":
        return TinyByteBackbone(config)
    if config.backbone.startswith("hf:"):
        return HFBackbone(config)
    raise ValueError(
        f"unknown backbone {config.backbone!r}: expected 'tiny' or 'hf:<model_id>'"
    )


class RelationshipEncoder(nn.Module):
    """Backbone + phase / regression / embedding heads."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = build_backbone(config)
        width = self.backbone.output_dim
        self.phase_head = nn.Linear(width, len(PHASE_VOCAB))
        self.regression_head = nn.Linear(width, len(REGRESSION_TARGETS))
        self.embedding_head = nn.Linear(width, config.embedding_dim)

    def forward(self, texts: list[str]) -> dict[str, torch.Tensor]:
        pooled = self.backbone(texts)
        embedding = nn.functional.normalize(self.embedding_head(pooled), dim=-1)
        return {
            "phase_logits": self.phase_head(pooled),
            "regressions": torch.sigmoid(self.regression_head(pooled)),
            "embedding": embedding,
        }


def save_checkpoint(
    model: RelationshipEncoder, path: pathlib.Path | str
) -> pathlib.Path:
    out_path = pathlib.Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"config": model.config.to_jsonable(), "state_dict": model.state_dict()},
        out_path,
    )
    return out_path


def load_checkpoint(
    path: pathlib.Path | str, *, device: str = "cpu"
) -> RelationshipEncoder:
    payload = torch.load(pathlib.Path(path), map_location=device, weights_only=True)
    model = RelationshipEncoder(EncoderConfig.from_jsonable(payload["config"]))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model


__all__ = [
    "PAD_ID",
    "VOCAB_SIZE",
    "EncoderConfig",
    "HFBackbone",
    "RelationshipEncoder",
    "TinyByteBackbone",
    "build_backbone",
    "collate_byte_batch",
    "encode_bytes",
    "load_checkpoint",
    "save_checkpoint",
]
