"""F6 / P6.2 — PEFT-backed LoRA bake backend (real training loop).

Concrete :class:`LoRABakeBackend` implementation that runs a short
parameter-efficient fine-tuning loop on a frozen Hugging Face base
model and exports the trained LoRA adapters as
:class:`SubstrateDeltaAdapterLayer` tuples — the same shape the
synthetic backend emits, so the persona-LoRA pool / activate path
consume both backends through one surface (debt #18 closure).

Why this lives in the figure-vertical wheel and not the substrate:

* The substrate owns the frozen base model + the
  :class:`SubstrateDeltaAdapterLayer` shape; it must not depend on
  ``peft`` (which would force the substrate's optional dep cone to
  carry HuggingFace).
* The figure vertical owns persona LoRA artifact provenance + the
  reviewed corpus the training loop reads from. The PEFT call is a
  bake operation, not a runtime forward — keeping it in the
  vertical keeps the substrate forward path untouched.

Activation discipline:

* The backend is **opt-in**. ``peft`` + ``transformers`` + ``torch``
  must all be importable; if any is missing, instantiating the
  backend raises ``ImportError`` with the precise install hint
  pointing at ``vz-runtime[torch]``.
* The training loop is short by default (``max_steps=20`` per
  ``epoch``, with ``epochs`` taken from the
  :class:`LoRATrainingPlan`) so a CPU-only smoke test can complete
  in seconds. Real production runs override these from the CLI.
* Output is byte-stable in the **shape** dimension (number of
  layers, vector_dim) but not in the **content** dimension across
  hardware / library versions — that's expected because the
  underlying optimizer / cuBLAS / CPU kernels are not bit-exact.
  The artifact's ``training_plan_hash`` still pins which corpus +
  hyperparams produced it, so audit / rollback discipline holds.

R10: a baked LoRA — synthetic OR PEFT — must still travel through
:class:`ModificationGate.OFFLINE` via
:func:`apply_persona_lora_through_gate`. The CLI surface enforces
this; calling :meth:`bake` directly does not bypass the gate
because ``bake`` only produces the artifact — applying it requires
the gate path.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import pathlib
from dataclasses import dataclass, field
from typing import Any, Literal

from volvence_zero.substrate import SubstrateDeltaAdapterLayer

from lifeform_domain_figure.lora_artifact import (
    SCHEMA_VERSION,
    FigureLoRAArtifact,
    LoRABakeBackend,
    compute_lora_integrity_hash,
)
from lifeform_domain_figure.lora_data_prep import LoRATrainingPlan


_BACKEND_ID = "peft-v1"
_DEFAULT_MODEL_ID = "sshleifer/tiny-gpt2"
_DEFAULT_TARGET_MODULES: tuple[str, ...] = ("c_attn",)
_DEFAULT_MAX_STEPS = 20
_DEFAULT_LORA_ALPHA = 16
_DEFAULT_LORA_DROPOUT = 0.0
# Per-layer delta_vector capacity cap. PEFT typically produces tens
# of thousands of trainable params per target_module on small
# models; we sample / pool them down to a fixed-width vector so the
# resulting SubstrateDeltaAdapterLayer stays compact for storage,
# audit, and pool registration. The value is large enough to retain
# the dominant signal but bounded so artifacts don't balloon the
# bundle pickle.
_DEFAULT_DELTA_VECTOR_DIM = 256

# Default on-disk location for peft.save_pretrained snapshots. The
# real-LoRA inference path (debt #40 closure) loads checkpoints from
# here via `peft.PeftModel.from_pretrained`. Layout:
#   PEFT_CHECKPOINT_CACHE_ROOT / <figure_id> / <plan_hash_prefix> /
# Keyed by training_plan_hash so two bakes of the same plan share
# one on-disk snapshot (cheap re-bake; portable across processes).
PEFT_CHECKPOINT_CACHE_ROOT = pathlib.Path(".local") / "peft-checkpoints"
_DISABLED_CHECKPOINT_DIR = pathlib.Path("")


@dataclass(frozen=True)
class PEFTLoRAConfig:
    """Typed PEFT LoRA hyper-parameters consumed by the backend.

    Mirrors the subset of :class:`peft.LoraConfig` fields that a
    figure-vertical curator actually controls; the rest are pinned
    to peft defaults so the audit log can re-construct the exact
    config from these typed fields alone.
    """

    target_modules: tuple[str, ...] = _DEFAULT_TARGET_MODULES
    rank: int = 8
    alpha: int = _DEFAULT_LORA_ALPHA
    dropout: float = _DEFAULT_LORA_DROPOUT

    def __post_init__(self) -> None:
        if not self.target_modules:
            raise ValueError(
                "PEFTLoRAConfig.target_modules must be a non-empty tuple"
            )
        if self.rank <= 0:
            raise ValueError(
                f"PEFTLoRAConfig.rank must be > 0, got {self.rank!r}"
            )
        if self.alpha <= 0:
            raise ValueError(
                f"PEFTLoRAConfig.alpha must be > 0, got {self.alpha!r}"
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(
                f"PEFTLoRAConfig.dropout must be in [0,1), got {self.dropout!r}"
            )


def _peft_available() -> bool:
    return all(
        importlib.util.find_spec(name) is not None
        for name in ("peft", "transformers", "torch")
    )


@dataclass(frozen=True)
class PEFTLoRABakeBackend(LoRABakeBackend):
    """Real PEFT-backed LoRA bake backend.

    Required fields:

    * ``model_id`` — HuggingFace model id of the frozen base. Defaults
      to ``sshleifer/tiny-gpt2`` so the CPU smoke test path is fast.
    * ``peft_config`` — typed :class:`PEFTLoRAConfig` (target_modules
      / rank / alpha / dropout).
    * ``runtime_device`` — ``"cpu"`` or ``"cuda"``. CPU works for
      tiny models; production runs use ``"cuda"``.
    * ``max_steps`` — hard cap on optimizer steps per epoch. The
      :class:`LoRATrainingPlan`'s ``epochs`` controls outer loop;
      ``max_steps`` keeps wall clock bounded even if the corpus
      is large.
    * ``checkpoint_dir`` — out-of-process artifact staging dir.
      ``None`` (default) auto-derives a path under
      :data:`PEFT_CHECKPOINT_CACHE_ROOT`/``<figure_id>/<plan_hash>``
      so the trained PEFT adapter is **always** persisted to disk.
      The path is recorded on the returned
      :attr:`FigureLoRAArtifact.peft_checkpoint_dir` so the runtime
      can load the real A/B matrices through
      :meth:`LoRAAwareResidualRuntime.activate_peft_adapter` at
      inference time (debt #40 closure path). Pass a Path to override
      the location (e.g. share a checkpoint across machines via a
      network mount); pass an empty :class:`pathlib.Path` to opt out
      (the artifact ships with ``peft_checkpoint_dir=""`` and the
      runtime falls back to the projected ``adapter_layers`` summary
      hook — degraded mode, kept for back-compat with older bundles).
    * ``delta_vector_dim`` — per-layer delta_vector width (see
      module docstring). Compresses the LoRA ``A @ B`` matrix to a
      bounded-width vector for storage parity with the synthetic
      backend.
    """

    model_id: str = _DEFAULT_MODEL_ID
    peft_config: PEFTLoRAConfig = field(default_factory=PEFTLoRAConfig)
    runtime_device: Literal["cpu", "cuda"] = "cpu"
    max_steps: int = _DEFAULT_MAX_STEPS
    checkpoint_dir: pathlib.Path | None = None
    delta_vector_dim: int = _DEFAULT_DELTA_VECTOR_DIM

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("PEFTLoRABakeBackend.model_id must be non-empty")
        if self.runtime_device not in ("cpu", "cuda"):
            raise ValueError(
                "PEFTLoRABakeBackend.runtime_device must be 'cpu' or 'cuda', "
                f"got {self.runtime_device!r}"
            )
        if self.max_steps <= 0:
            raise ValueError(
                f"PEFTLoRABakeBackend.max_steps must be > 0, got {self.max_steps!r}"
            )
        if self.delta_vector_dim <= 0:
            raise ValueError(
                f"PEFTLoRABakeBackend.delta_vector_dim must be > 0, "
                f"got {self.delta_vector_dim!r}"
            )

    @property
    def backend_id(self) -> str:
        return _BACKEND_ID

    def bake(self, plan: LoRATrainingPlan) -> FigureLoRAArtifact:
        """Run a short PEFT LoRA training loop and emit the artifact.

        Steps:

        1. Resolve peft / transformers / torch (raise ImportError
           with install hint if missing — fail loud per
           ``no-swallow-errors-no-hasattr-abuse.mdc``).
        2. Load the frozen base model + tokenizer; wrap with
           :class:`peft.LoraConfig`-derived adapter layers on the
           configured ``target_modules`` at the configured rank.
        3. Tokenize ``plan.examples`` (figure rows + replay rows
           interleaved); build a tiny in-memory dataset.
        4. Run the training loop: AdamW step over causal-LM loss,
           with per-row ``weight`` from the plan applied to the
           per-token loss.
        5. Extract the trained LoRA A / B matrices from
           ``model.state_dict()`` keyed by target_module.
        6. For each ``target_module`` produce one
           :class:`SubstrateDeltaAdapterLayer` whose ``delta_vector``
           is a deterministic projection of ``A @ B`` to a
           ``delta_vector_dim``-dimensional summary (sign-preserving
           bucket pooling — same idea the substrate uses elsewhere).
        7. Compute the :class:`FigureLoRAArtifact` integrity hash
           and return.

        The bake DOES NOT mutate the base model file on disk. PEFT
        operates on a deep-copied module; the original model's
        weights stay byte-identical (R2: base frozen, adaptation
        in controller layer).
        """

        if not _peft_available():
            raise ImportError(
                "PEFTLoRABakeBackend.bake requires peft + transformers + torch. "
                "Install them via ``pip install vz-runtime[torch]`` (which "
                "transitively installs peft>=0.11). Current import probe failed; "
                "if you intended to run a SHADOW deployment, use "
                "SyntheticLoRABakeBackend from "
                "lifeform_domain_figure.lora_bake_synthetic instead."
            )
        # Imports are local so the module loads fine when peft is absent
        # (the backend instance just can't bake until peft is reachable).
        torch = importlib.import_module("torch")
        peft_mod = importlib.import_module("peft")
        transformers = importlib.import_module("transformers")

        device_str = self.runtime_device
        if device_str == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "PEFTLoRABakeBackend.bake: runtime_device='cuda' but "
                "torch.cuda.is_available() is False; refusing to silently "
                "fall back to CPU. Set runtime_device='cpu' explicitly to "
                "force CPU bake."
            )
        device = torch.device(device_str)

        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
        base_model = transformers.AutoModelForCausalLM.from_pretrained(self.model_id)
        base_model.to(device)
        for param in base_model.parameters():
            param.requires_grad = False

        lora_config = peft_mod.LoraConfig(
            r=self.peft_config.rank,
            lora_alpha=self.peft_config.alpha,
            lora_dropout=self.peft_config.dropout,
            target_modules=list(self.peft_config.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        peft_model = peft_mod.get_peft_model(base_model, lora_config)
        peft_model.train()
        peft_model.to(device)

        trainable_params = [
            param for param in peft_model.parameters() if param.requires_grad
        ]
        if not trainable_params:
            raise RuntimeError(
                "PEFTLoRABakeBackend.bake: peft did not mark any parameters "
                "trainable. Check target_modules="
                f"{self.peft_config.target_modules!r} matches a module name "
                f"in {self.model_id!r}."
            )
        total_trainable = sum(int(p.numel()) for p in trainable_params)
        total_params = sum(int(p.numel()) for p in peft_model.parameters())
        optimizer = torch.optim.AdamW(
            trainable_params, lr=plan.learning_rate
        )

        encoded_rows = _encode_rows(
            plan=plan,
            tokenizer=tokenizer,
            device=device,
        )
        if not encoded_rows:
            raise RuntimeError(
                "PEFTLoRABakeBackend.bake: no training rows survived "
                "tokenization; refusing to emit an artifact derived from "
                "an empty training pass."
            )

        init_loss = _evaluate_loss(peft_model, encoded_rows)
        steps_taken = 0
        for _epoch in range(plan.epochs):
            for row in encoded_rows:
                if steps_taken >= self.max_steps:
                    break
                optimizer.zero_grad()
                outputs = peft_model(
                    input_ids=row["input_ids"],
                    attention_mask=row["attention_mask"],
                    labels=row["labels"],
                )
                loss = outputs.loss * row["weight"]
                loss.backward()
                optimizer.step()
                steps_taken += 1
            if steps_taken >= self.max_steps:
                break
        peft_model.eval()
        final_loss = _evaluate_loss(peft_model, encoded_rows)

        adapter_layers = _extract_adapter_layers(
            peft_model=peft_model,
            target_modules=self.peft_config.target_modules,
            target_layer_index=plan.target_layer_index,
            delta_vector_dim=self.delta_vector_dim,
            figure_id=plan.figure_id,
            plan_hash_prefix=plan.integrity_hash[:8],
        )
        if not adapter_layers:
            raise RuntimeError(
                "PEFTLoRABakeBackend.bake: no LoRA adapter layers produced. "
                f"Check that target_modules={self.peft_config.target_modules!r} "
                f"actually exists in {self.model_id!r}; the configured "
                f"modules did not match any base-model submodule names."
            )
        resolved_checkpoint_dir = self._resolve_checkpoint_dir(plan=plan)
        if resolved_checkpoint_dir is not None:
            resolved_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            peft_model.save_pretrained(str(resolved_checkpoint_dir))
            checkpoint_dir_str = str(resolved_checkpoint_dir.resolve())
        else:
            checkpoint_dir_str = ""

        integrity_hash = compute_lora_integrity_hash(
            figure_id=plan.figure_id,
            backend_id=_BACKEND_ID,
            training_plan_hash=plan.integrity_hash,
            adapter_layers=adapter_layers,
        )
        validation_delta = _compute_validation_delta(
            init_loss=init_loss, final_loss=final_loss
        )
        capacity_cost = _compute_capacity_cost(
            trainable=total_trainable, total=total_params
        )
        return FigureLoRAArtifact(
            schema_version=SCHEMA_VERSION,
            figure_id=plan.figure_id,
            backend_id=_BACKEND_ID,
            rank=plan.rank,
            target_layer_index=plan.target_layer_index,
            adapter_layers=adapter_layers,
            training_plan_hash=plan.integrity_hash,
            integrity_hash=integrity_hash,
            parameter_count=int(total_trainable),
            description=(
                f"PEFT persona LoRA for {plan.figure_id} on base "
                f"{self.model_id} (rank={plan.rank}, layers="
                f"{len(adapter_layers)}, steps={steps_taken}, "
                f"init_loss={init_loss:.4f}, final_loss={final_loss:.4f}, "
                f"validation_delta={validation_delta:.4f}, capacity_cost="
                f"{capacity_cost:.4f}). Plan {plan.integrity_hash[:8]}."
            ),
            peft_checkpoint_dir=checkpoint_dir_str,
        )

    def _resolve_checkpoint_dir(
        self, *, plan: LoRATrainingPlan
    ) -> pathlib.Path | None:
        """Decide where to persist the peft adapter for this bake.

        Three cases:

        * ``checkpoint_dir is None`` (default) — auto-derive to
          :data:`PEFT_CHECKPOINT_CACHE_ROOT` /
          ``<figure_id>/<plan_hash_prefix>`` under the current working
          directory. Always returns a real path: the trained adapter
          is persisted so the runtime can load it through
          ``peft.PeftModel.from_pretrained``.
        * ``checkpoint_dir`` set to an empty path
          (``pathlib.Path("")``) — explicit opt-out. Returns ``None``;
          the bake skips ``save_pretrained`` and the artifact ships
          ``peft_checkpoint_dir=""``. Kept for callers (e.g. unit
          tests) that genuinely don't want any disk I/O.
        * ``checkpoint_dir`` set to a concrete path — caller-supplied
          override. Returns it verbatim.
        """

        if self.checkpoint_dir is None:
            plan_prefix = plan.integrity_hash[:16] if plan.integrity_hash else "unhashed"
            return PEFT_CHECKPOINT_CACHE_ROOT / plan.figure_id / plan_prefix
        if self.checkpoint_dir == _DISABLED_CHECKPOINT_DIR:
            return None
        return self.checkpoint_dir


def _encode_rows(
    *,
    plan: LoRATrainingPlan,
    tokenizer: Any,
    device: Any,
) -> list[dict[str, Any]]:
    """Tokenize plan rows into per-row tensors with explicit weights."""

    torch = importlib.import_module("torch")
    rows: list[dict[str, Any]] = []
    for example in plan.examples:
        encoded = tokenizer(
            example.text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=False,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = attention_mask.to(device)
        if input_ids.shape[1] < 2:
            # Causal LM needs at least one shifted-pair to compute a loss.
            continue
        labels = input_ids.clone()
        rows.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "weight": float(example.weight),
            }
        )
    return rows


def _evaluate_loss(peft_model: Any, encoded_rows: list[dict[str, Any]]) -> float:
    """Compute the unweighted mean loss over ``encoded_rows``."""

    torch = importlib.import_module("torch")
    if not encoded_rows:
        return 0.0
    losses: list[float] = []
    with torch.no_grad():
        for row in encoded_rows:
            outputs = peft_model(
                input_ids=row["input_ids"],
                attention_mask=row["attention_mask"],
                labels=row["labels"],
            )
            losses.append(float(outputs.loss.detach().cpu().item()))
    if not losses:
        return 0.0
    return sum(losses) / len(losses)


def _extract_adapter_layers(
    *,
    peft_model: Any,
    target_modules: tuple[str, ...],
    target_layer_index: int,
    delta_vector_dim: int,
    figure_id: str,
    plan_hash_prefix: str,
) -> tuple[SubstrateDeltaAdapterLayer, ...]:
    """Extract trained LoRA matrices and project them to delta vectors.

    For every (parent_module_name, target_module_name) pair the
    LoRA adapter touched, compute ``A @ B`` and pool the result
    into a ``delta_vector_dim``-wide tuple. The pooling is
    sign-preserving bucket sum (same shape contract as the
    synthetic backend's hash-derived vectors).
    """

    torch = importlib.import_module("torch")
    state = peft_model.state_dict()
    # Group A / B matrices by their LoRA module path.
    grouped: dict[str, dict[str, Any]] = {}
    for key, tensor in state.items():
        if "lora_A" not in key and "lora_B" not in key:
            continue
        if not _matches_any_target(key, target_modules):
            continue
        path, slot = _split_lora_key(key)
        if path is None or slot is None:
            continue
        grouped.setdefault(path, {})[slot] = tensor.detach().cpu()
    layers: list[SubstrateDeltaAdapterLayer] = []
    for offset, path in enumerate(sorted(grouped.keys())):
        slots = grouped[path]
        a_tensor = slots.get("A")
        b_tensor = slots.get("B")
        if a_tensor is None or b_tensor is None:
            continue
        # peft stores A as (rank, in_features) and B as (out_features, rank).
        # The combined LoRA delta on the linear layer's weight is B @ A,
        # an (out_features, in_features) matrix. Some peft versions
        # store the transposes; we accept either ordering as long as
        # the contraction yields a 2-D tensor.
        delta_matrix = _safe_lora_product(a_tensor, b_tensor, torch=torch)
        delta_vector = _project_to_fixed_width(
            delta_matrix.flatten(), delta_vector_dim, torch=torch
        )
        mean_abs = float(torch.mean(torch.abs(delta_matrix)).item())
        layers.append(
            SubstrateDeltaAdapterLayer(
                layer_index=target_layer_index + offset,
                delta_vector=tuple(float(v) for v in delta_vector.tolist()),
                mean_abs_delta=mean_abs,
                description=(
                    f"figure-persona-lora:{figure_id}:backend={_BACKEND_ID}:"
                    f"plan={plan_hash_prefix}:module={path}"
                ),
            )
        )
    return tuple(layers)


def _matches_any_target(key: str, target_modules: tuple[str, ...]) -> bool:
    return any(target in key for target in target_modules)


def _split_lora_key(key: str) -> tuple[str | None, str | None]:
    """Split a peft state-dict key into (path, slot) where slot is 'A'|'B'."""

    if "lora_A" in key:
        slot = "A"
    elif "lora_B" in key:
        slot = "B"
    else:
        return (None, None)
    # peft keys look like
    #   base_model.model.transformer.h.0.attn.c_attn.lora_A.default.weight
    # We use everything up to the .lora_<X> token as a stable per-target
    # path.
    needle = f".lora_{slot}"
    head, _, _ = key.partition(needle)
    return (head, slot)


def _safe_lora_product(a_tensor: Any, b_tensor: Any, *, torch: Any) -> Any:
    """Return ``B @ A`` (or its transposed shape) as an (M, N) tensor."""

    a = a_tensor
    b = b_tensor
    if a.dim() != 2 or b.dim() != 2:
        # Unexpected shape; flatten what we have so we still get a deterministic vector.
        return torch.cat([a.flatten(), b.flatten()])
    # Standard peft layout: A=(rank, in), B=(out, rank).  B @ A -> (out, in).
    if b.shape[1] == a.shape[0]:
        return b @ a
    # Transposed layout: A=(in, rank), B=(rank, out).  A @ B -> (in, out).
    if a.shape[1] == b.shape[0]:
        return a @ b
    # Fallback: just concat.
    return torch.cat([a.flatten(), b.flatten()])


def _project_to_fixed_width(values: Any, dim: int, *, torch: Any) -> Any:
    """Sum-pool a 1-D tensor into ``dim`` buckets, sign-preserving."""

    flat = values.detach().cpu()
    n = int(flat.numel())
    if n == 0:
        return torch.zeros(dim)
    if n <= dim:
        out = torch.zeros(dim)
        out[:n] = flat
        return out
    # Bucket-sum: each output bucket sums a contiguous slice of input.
    # Deterministic (no randomness) so re-runs with the same trained
    # weights project to byte-identical vectors.
    bucket_size = math.ceil(n / dim)
    padded_len = bucket_size * dim
    if padded_len > n:
        padded = torch.zeros(padded_len)
        padded[:n] = flat
    else:
        padded = flat
    return padded.view(dim, bucket_size).sum(dim=1)


def _compute_validation_delta(*, init_loss: float, final_loss: float) -> float:
    """Relative loss reduction, clamped to [0.0, 1.0].

    Used as the OFFLINE gate's ``validation_delta`` evidence — a
    real PEFT bake that did not reduce loss is not eligible for
    the gate's ALLOW path, so this value drives the audit.
    """

    if init_loss <= 1e-9:
        return 0.0
    delta = (init_loss - final_loss) / init_loss
    if delta != delta:  # NaN guard
        return 0.0
    if delta < 0.0:
        return 0.0
    if delta > 1.0:
        return 1.0
    return float(delta)


def _compute_capacity_cost(*, trainable: int, total: int) -> float:
    """Trainable / total parameter ratio, clamped to [0.0, 1.0]."""

    if total <= 0:
        return 0.0
    ratio = trainable / total
    if ratio < 0.0:
        return 0.0
    if ratio > 1.0:
        return 1.0
    return float(ratio)


__all__ = [
    "PEFTLoRABakeBackend",
    "PEFTLoRAConfig",
]
