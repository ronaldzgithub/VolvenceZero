"""Injectable rare-heavy adapter training backend (S1, real PEFT path).

The brain's rare-heavy artifact pipeline asks the substrate runtime for a
:class:`SubstrateRareHeavyCheckpoint`. Until S1 the transformers runtime
trained adapter deltas with a small built-in autograd loop
(``_train_adapter_deltas``, ``training_mode="adapter-delta-v2"``) and the
synthetic runtime derived a purely heuristic checkpoint. This module adds
the missing first-class seam:

- :class:`RareHeavyAdapterTrainingBackend` — protocol an offline training
  backend implements. The runtime calls it from ``train_rare_heavy`` when
  one is injected via ``set_rare_heavy_training_backend``; otherwise the
  built-in loop remains the fallback (and the synthetic runtime keeps the
  heuristic path).
- :class:`PeftLoraRareHeavyBackend` — the real PEFT LoRA training loop,
  generalized from ``lifeform_domain_figure.lora_bake_peft`` (which stays
  the persona-bake owner) into a substrate-facing backend that emits
  :class:`SubstrateDeltaAdapterLayer` tuples sized to the runtime's hidden
  width, so the resulting checkpoint imports through the normal
  ``import_rare_heavy_state`` / ``ModificationGate`` orchestration in
  ``session_training_phase.py`` unchanged.

Boundary notes (R2 / R9 / R10):

- ``peft`` / ``transformers`` / ``torch`` are imported lazily inside
  ``train`` and missing deps fail loudly with an install hint; the
  substrate wheel's import surface stays dependency-free.
- The PEFT loop freezes every base parameter and trains only the LoRA
  matrices; the loaded base model weights stay byte-identical (frozen
  substrate). The trained deltas only reach the live runtime through the
  rare-heavy artifact -> pre-import replay -> gate path.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from volvence_zero.substrate.residual_contracts import (
    SubstrateDeltaAdapterLayer,
    TrainingTrace,
)

RARE_HEAVY_PEFT_TRAINING_MODE = "peft-lora-v1"

_DEFAULT_TARGET_MODULES: tuple[str, ...] = ("c_attn",)
_DEFAULT_MAX_STEPS = 20


@dataclass(frozen=True)
class RareHeavyTrainingRequest:
    """Everything a rare-heavy adapter backend needs from the runtime.

    ``hidden_size`` and ``layer_indices`` pin the output contract: the
    backend must emit one delta vector of width ``hidden_size`` per layer
    index so ``import_rare_heavy_state`` can install them verbatim.
    """

    model_id: str
    hidden_size: int
    layer_indices: tuple[int, ...]
    device: str
    traces: tuple[TrainingTrace, ...]

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError(
                f"RareHeavyTrainingRequest.hidden_size must be > 0, got {self.hidden_size!r}"
            )
        if not self.layer_indices:
            raise ValueError("RareHeavyTrainingRequest.layer_indices must be non-empty")


@dataclass(frozen=True)
class RareHeavyAdapterTrainingResult:
    """Trained adapter payload handed back to the runtime for packaging."""

    training_mode: str
    adapter_layers: tuple[SubstrateDeltaAdapterLayer, ...]
    training_loss: float
    initial_loss: float
    steps_taken: int
    description: str

    def __post_init__(self) -> None:
        if not self.training_mode.strip():
            raise ValueError("RareHeavyAdapterTrainingResult.training_mode must be non-empty")
        if not self.adapter_layers:
            raise ValueError(
                "RareHeavyAdapterTrainingResult.adapter_layers must be non-empty; "
                "a backend that produced nothing must raise instead of emitting "
                "an empty artifact."
            )


@runtime_checkable
class RareHeavyAdapterTrainingBackend(Protocol):
    """Offline rare-heavy adapter training backend seam.

    ``train`` runs the (possibly heavy) offline loop and returns the
    trained adapter layers. It must fail loudly on missing dependencies
    or empty training data — the runtime does not silently fall back to
    the built-in loop when an explicit backend was injected.
    """

    @property
    def backend_id(self) -> str: ...

    def train(self, request: RareHeavyTrainingRequest) -> RareHeavyAdapterTrainingResult: ...


def _peft_stack_available() -> bool:
    return all(
        importlib.util.find_spec(name) is not None
        for name in ("peft", "transformers", "torch")
    )


@dataclass(frozen=True)
class PeftLoraRareHeavyBackend:
    """Real PEFT LoRA rare-heavy training backend.

    Runs a short parameter-efficient fine-tuning loop on a frozen base
    model over the bundle's trace texts (causal-LM objective), then
    projects each trained LoRA ``B @ A`` product onto a
    ``hidden_size``-wide, sign-preserving delta vector per requested
    layer index.

    * ``base_model_source`` — optional override for where to load the
      frozen base from (defaults to the request's ``model_id``); lets
      offline hosts point at a local snapshot.
    * ``max_steps`` bounds wall clock (CPU smoke completes in seconds on
      tiny models); production runs raise it from the artifact plan.
    """

    target_modules: tuple[str, ...] = _DEFAULT_TARGET_MODULES
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    learning_rate: float = 5e-4
    max_steps: int = _DEFAULT_MAX_STEPS
    base_model_source: str = ""
    extra_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.target_modules:
            raise ValueError("PeftLoraRareHeavyBackend.target_modules must be non-empty")
        if self.rank <= 0 or self.alpha <= 0:
            raise ValueError("PeftLoraRareHeavyBackend rank/alpha must be > 0")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("PeftLoraRareHeavyBackend.dropout must be in [0,1)")
        if self.max_steps <= 0:
            raise ValueError("PeftLoraRareHeavyBackend.max_steps must be > 0")

    @property
    def backend_id(self) -> str:
        return RARE_HEAVY_PEFT_TRAINING_MODE

    def train(self, request: RareHeavyTrainingRequest) -> RareHeavyAdapterTrainingResult:
        if not _peft_stack_available():
            raise ImportError(
                "PeftLoraRareHeavyBackend.train requires peft + transformers + torch. "
                "Install them via ``pip install vz-runtime[torch]``. If you intended "
                "the built-in adapter-delta loop, do not inject a rare-heavy "
                "training backend."
            )
        texts = tuple(
            trace.source_text for trace in request.traces if trace.source_text.strip()
        )
        if not texts:
            raise ValueError(
                "PeftLoraRareHeavyBackend.train received no non-empty trace texts; "
                "refusing to emit an artifact from an empty training pass."
            )
        torch = importlib.import_module("torch")
        peft_mod = importlib.import_module("peft")
        transformers = importlib.import_module("transformers")

        source = self.base_model_source or request.model_id
        tokenizer = transformers.AutoTokenizer.from_pretrained(source)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
        base_model = transformers.AutoModelForCausalLM.from_pretrained(source)
        device = torch.device(request.device if request.device else "cpu")
        base_model.to(device)
        for param in base_model.parameters():
            param.requires_grad = False

        lora_config = peft_mod.LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=list(self.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        peft_model = peft_mod.get_peft_model(base_model, lora_config)
        peft_model.train()
        peft_model.to(device)
        trainable = [p for p in peft_model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError(
                "PeftLoraRareHeavyBackend.train: peft marked no parameters trainable; "
                f"target_modules={self.target_modules!r} does not match any module "
                f"in {source!r}."
            )
        optimizer = torch.optim.AdamW(trainable, lr=self.learning_rate)

        rows = []
        for text in texts:
            encoded = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=128
            )
            input_ids = encoded["input_ids"].to(device)
            if input_ids.shape[1] < 2:
                continue
            attention_mask = encoded.get("attention_mask")
            attention_mask = (
                attention_mask.to(device)
                if attention_mask is not None
                else torch.ones_like(input_ids)
            )
            rows.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": input_ids.clone(),
                }
            )
        if not rows:
            raise ValueError(
                "PeftLoraRareHeavyBackend.train: no trace text survived tokenization "
                "(all shorter than the two tokens a causal-LM loss needs)."
            )

        def _mean_loss() -> float:
            with torch.no_grad():
                losses = [
                    float(
                        peft_model(
                            input_ids=row["input_ids"],
                            attention_mask=row["attention_mask"],
                            labels=row["labels"],
                        ).loss.detach().cpu().item()
                    )
                    for row in rows
                ]
            return sum(losses) / len(losses)

        initial_loss = _mean_loss()
        steps_taken = 0
        while steps_taken < self.max_steps:
            for row in rows:
                if steps_taken >= self.max_steps:
                    break
                optimizer.zero_grad()
                loss = peft_model(
                    input_ids=row["input_ids"],
                    attention_mask=row["attention_mask"],
                    labels=row["labels"],
                ).loss
                loss.backward()
                optimizer.step()
                steps_taken += 1
        peft_model.eval()
        final_loss = _mean_loss()

        delta_matrices = _collect_lora_products(
            peft_model=peft_model,
            target_modules=self.target_modules,
            torch=torch,
        )
        if not delta_matrices:
            raise RuntimeError(
                "PeftLoraRareHeavyBackend.train: no LoRA adapter products extracted; "
                f"target_modules={self.target_modules!r} produced no state-dict "
                "entries."
            )
        adapter_layers = _project_products_to_layers(
            delta_matrices=delta_matrices,
            hidden_size=request.hidden_size,
            layer_indices=request.layer_indices,
            torch=torch,
            backend_id=self.backend_id,
        )
        return RareHeavyAdapterTrainingResult(
            training_mode=RARE_HEAVY_PEFT_TRAINING_MODE,
            adapter_layers=adapter_layers,
            training_loss=final_loss,
            initial_loss=initial_loss,
            steps_taken=steps_taken,
            description=(
                f"PEFT LoRA rare-heavy training on {source}: rank={self.rank} "
                f"steps={steps_taken} init_loss={initial_loss:.4f} "
                f"final_loss={final_loss:.4f} layers={len(adapter_layers)}."
            ),
        )


def _collect_lora_products(
    *, peft_model: Any, target_modules: tuple[str, ...], torch: Any
) -> list[Any]:
    """Return the trained ``B @ A`` product per LoRA-adapted module."""

    grouped: dict[str, dict[str, Any]] = {}
    for key, tensor in peft_model.state_dict().items():
        if "lora_A" in key:
            slot = "A"
        elif "lora_B" in key:
            slot = "B"
        else:
            continue
        if not any(target in key for target in target_modules):
            continue
        head, _, _ = key.partition(f".lora_{slot}")
        grouped.setdefault(head, {})[slot] = tensor.detach().cpu()
    products: list[Any] = []
    for path in sorted(grouped.keys()):
        slots = grouped[path]
        a_tensor = slots.get("A")
        b_tensor = slots.get("B")
        if a_tensor is None or b_tensor is None:
            continue
        if a_tensor.dim() == 2 and b_tensor.dim() == 2:
            if b_tensor.shape[1] == a_tensor.shape[0]:
                products.append(b_tensor @ a_tensor)
                continue
            if a_tensor.shape[1] == b_tensor.shape[0]:
                products.append(a_tensor @ b_tensor)
                continue
        products.append(torch.cat([a_tensor.flatten(), b_tensor.flatten()]))
    return products


def _project_products_to_layers(
    *,
    delta_matrices: list[Any],
    hidden_size: int,
    layer_indices: tuple[int, ...],
    torch: Any,
    backend_id: str,
) -> tuple[SubstrateDeltaAdapterLayer, ...]:
    """Pool the LoRA products into one ``hidden_size``-wide vector per layer.

    Layer ``i`` consumes product ``i % len(products)`` so every requested
    layer index gets a delta vector even when fewer modules were adapted
    than layers requested (bucket pooling is deterministic; re-runs on the
    same trained weights are byte-identical).
    """

    layers: list[SubstrateDeltaAdapterLayer] = []
    for offset, layer_index in enumerate(layer_indices):
        flat = delta_matrices[offset % len(delta_matrices)].flatten()
        vector = _bucket_pool(flat, hidden_size, torch=torch)
        layers.append(
            SubstrateDeltaAdapterLayer(
                layer_index=layer_index,
                delta_vector=tuple(float(v) for v in vector.tolist()),
                mean_abs_delta=float(torch.mean(torch.abs(vector)).item()),
                description=(
                    f"rare-heavy:{backend_id}:layer={layer_index} "
                    f"hidden={hidden_size}."
                ),
            )
        )
    return tuple(layers)


def _bucket_pool(values: Any, dim: int, *, torch: Any) -> Any:
    """Sum-pool a 1-D tensor into ``dim`` buckets, sign-preserving."""

    flat = values.detach().cpu().to(torch.float32)
    n = int(flat.numel())
    if n == 0:
        return torch.zeros(dim)
    if n <= dim:
        out = torch.zeros(dim)
        out[:n] = flat
        return out
    bucket_size = math.ceil(n / dim)
    padded_len = bucket_size * dim
    if padded_len > n:
        padded = torch.zeros(padded_len)
        padded[:n] = flat
    else:
        padded = flat
    return padded.view(dim, bucket_size).sum(dim=1)


__all__ = [
    "RARE_HEAVY_PEFT_TRAINING_MODE",
    "PeftLoraRareHeavyBackend",
    "RareHeavyAdapterTrainingBackend",
    "RareHeavyAdapterTrainingResult",
    "RareHeavyTrainingRequest",
]
