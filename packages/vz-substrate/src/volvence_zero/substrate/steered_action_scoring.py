"""Differentiable steered action scoring on the frozen transformers substrate.

ETA Eq.3 requires the metacontroller distortion to be the action
log-likelihood of the *controlled frozen model*:

    -log p_{theta,phi}(a_t | o_{1:t}, z_{1:t})

Every other forward surface in vz-substrate runs under ``torch.no_grad()``,
so nothing could train a controller *through* the frozen model. This module
adds the missing surface: a scorer that injects a per-step residual-stream
control delta at one hooked block via forward hook, keeps the upper blocks
inside the autograd graph, and returns the differentiable negative
log-likelihood of expert actions over a fixed action vocabulary read from
the frozen LM head.

Ownership and arms:

- Frozen arm (default): base parameters stay ``requires_grad=False``;
  gradient reaches only the injected control delta. This is the Eq.3 setup.
- Joint arm (``joint_training=True``): the blocks strictly above the
  injection layer plus the final norm become trainable. This is an explicit
  validity control replicating the paper's degenerate joint-training
  baseline. Pristine weights are snapshotted at construction;
  ``reset_joint_parameters()`` restores them and ``restore_and_freeze()``
  returns the shared model to its frozen state. The LM head / tied
  embeddings stay frozen in both arms (documented deviation: the paper
  joint-trains the full base model).

This is an offline evidence surface, not a runtime snapshot publisher; it
adds no snapshot slot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class SteeredActionOption:
    """One discrete action choice scored against the frozen LM head."""

    action_id: str
    surface_text: str


class TransformersSteeredActionScorer:
    """Differentiable action NLL through a hook-steered frozen model."""

    def __init__(
        self,
        *,
        torch_module: Any,
        model: Any,
        tokenizer: Any,
        block_modules: Sequence[Any],
        final_norm_module: Any,
        injection_layer_index: int,
        hidden_size: int,
        device: Any,
        model_id: str,
        action_options: tuple[SteeredActionOption, ...],
        prompt_suffix: str = "\nNext move:",
        max_length: int = 96,
        control_norm_ratio: float = 0.25,
        probe_texts: tuple[str, ...] = (),
        joint_training: bool = False,
    ) -> None:
        if len(action_options) < 2:
            raise ValueError(
                "Steered action scoring requires at least two action options."
            )
        if not 0 <= injection_layer_index < len(block_modules):
            raise ValueError(
                f"injection_layer_index {injection_layer_index} out of range "
                f"for {len(block_modules)} blocks."
            )
        if not 0.0 < control_norm_ratio <= 2.0:
            raise ValueError(
                "control_norm_ratio must be in (0, 2]; got "
                f"{control_norm_ratio!r}."
            )
        self._torch = torch_module
        self._model = model
        self._tokenizer = tokenizer
        self._blocks = tuple(block_modules)
        self._final_norm = final_norm_module
        self._injection_layer_index = injection_layer_index
        self._hidden_size = hidden_size
        self._device = device
        self.model_id = model_id
        self._options = action_options
        self._prompt_suffix = prompt_suffix
        self._max_length = max(8, max_length)
        self._joint_training = joint_training
        self._model_dtype = next(iter(model.parameters())).dtype

        if self._tokenizer.pad_token_id is None:
            if self._tokenizer.eos_token_id is None:
                raise ValueError(
                    f"Tokenizer for {model_id!r} has neither pad nor eos "
                    "token; batched steered scoring cannot pad."
                )
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._action_index_by_id = {
            option.action_id: index
            for index, option in enumerate(action_options)
        }
        if len(self._action_index_by_id) != len(action_options):
            raise ValueError("action_options carry duplicate action_id.")
        candidate_ids: list[int] = []
        for option in action_options:
            token_ids = self._tokenizer(
                " " + option.surface_text, add_special_tokens=False
            )["input_ids"]
            if not token_ids:
                raise ValueError(
                    f"Action option {option.surface_text!r} tokenizes to "
                    "nothing."
                )
            candidate_ids.append(int(token_ids[0]))
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError(
                "Action option first tokens collide; the restricted action "
                f"softmax would be ambiguous: {candidate_ids}."
            )
        self._candidate_token_ids = torch_module.tensor(
            candidate_ids, dtype=torch_module.long, device=device
        )

        probe_norm = self._probe_hidden_norm(
            probe_texts
            or tuple(
                f"Probe context {index}: available transitions "
                + ", ".join(
                    option.surface_text for option in action_options
                )
                + self._prompt_suffix
                for index in range(2)
            )
        )
        self._control_norm_cap = float(control_norm_ratio * probe_norm)
        self._probe_hidden_norm_value = float(probe_norm)

        self._joint_modules: tuple[Any, ...] = ()
        self._joint_pristine_state: tuple[dict[str, Any], ...] = ()
        if joint_training:
            joint_modules = [
                self._blocks[index]
                for index in range(
                    injection_layer_index + 1, len(self._blocks)
                )
            ]
            joint_modules.append(self._final_norm)
            self._joint_modules = tuple(joint_modules)
            self._joint_pristine_state = tuple(
                {
                    key: value.detach().to("cpu").clone()
                    for key, value in module.state_dict().items()
                }
                for module in self._joint_modules
            )
            for module in self._joint_modules:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        else:
            self._assert_fully_frozen()

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def control_norm_cap(self) -> float:
        return self._control_norm_cap

    @property
    def probe_hidden_norm(self) -> float:
        return self._probe_hidden_norm_value

    @property
    def injection_layer_index(self) -> int:
        return self._injection_layer_index

    @property
    def joint_training(self) -> bool:
        return self._joint_training

    @property
    def action_option_ids(self) -> tuple[str, ...]:
        return tuple(option.action_id for option in self._options)

    def action_index(self, action_id: str) -> int:
        index = self._action_index_by_id.get(action_id)
        if index is None:
            raise KeyError(
                f"Unknown action_id {action_id!r}; scorer options are "
                f"{tuple(self._action_index_by_id)}."
            )
        return index

    def trainable_parameters(self) -> tuple[Any, ...]:
        """Substrate-side parameters trained in the joint arm (empty when frozen)."""

        if not self._joint_training:
            return ()
        return tuple(
            parameter
            for module in self._joint_modules
            for parameter in module.parameters()
        )

    def reset_joint_parameters(self) -> None:
        """Restore pristine upper-block weights before a fresh joint run."""

        if not self._joint_training:
            raise RuntimeError(
                "reset_joint_parameters is only valid on a joint-training "
                "scorer."
            )
        torch = self._torch
        with torch.no_grad():
            for module, pristine in zip(
                self._joint_modules, self._joint_pristine_state, strict=True
            ):
                module.load_state_dict(
                    {
                        key: value.to(self._device)
                        for key, value in pristine.items()
                    }
                )

    def restore_and_freeze(self) -> None:
        """Restore pristine weights and re-freeze; call after the joint arm ends."""

        if not self._joint_training:
            raise RuntimeError(
                "restore_and_freeze is only valid on a joint-training scorer."
            )
        self.reset_joint_parameters()
        for module in self._joint_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self._joint_training = False
        self._joint_modules = ()
        self._joint_pristine_state = ()
        self._assert_fully_frozen()

    def action_nll(
        self,
        *,
        source_texts: tuple[str, ...],
        control_deltas: Any,
        action_indices: tuple[int, ...],
    ) -> Any:
        """Differentiable per-step NLL of expert actions under steering.

        ``control_deltas`` is a torch tensor of shape ``[T, hidden_size]``
        (any float dtype/device; it is cast inside the graph). Returns a
        CPU float64 tensor of shape ``[T]`` connected to the autograd graph
        so temporal-owner objectives can consume it directly.
        """

        return self._score(
            source_texts=source_texts,
            control_deltas=control_deltas,
            action_indices=action_indices,
            enable_grad=True,
        )

    def baseline_action_nll(
        self,
        *,
        source_texts: tuple[str, ...],
        action_indices: tuple[int, ...],
    ) -> tuple[float, ...]:
        """Unsteered (zero-delta) NLL readout; anchors the distortion axis."""

        torch = self._torch
        with torch.no_grad():
            values = self._score(
                source_texts=source_texts,
                control_deltas=torch.zeros(
                    (len(source_texts), self._hidden_size),
                    dtype=torch.float32,
                ),
                action_indices=action_indices,
                enable_grad=False,
            )
        return tuple(float(value) for value in values)

    def _score(
        self,
        *,
        source_texts: tuple[str, ...],
        control_deltas: Any,
        action_indices: tuple[int, ...],
        enable_grad: bool,
    ) -> Any:
        torch = self._torch
        if not source_texts:
            raise ValueError("source_texts must be non-empty.")
        if len(source_texts) != len(action_indices):
            raise ValueError(
                f"Got {len(source_texts)} texts for {len(action_indices)} "
                "action indices."
            )
        if int(control_deltas.shape[0]) != len(source_texts) or int(
            control_deltas.shape[1]
        ) != self._hidden_size:
            raise ValueError(
                "control_deltas must be [steps, hidden_size]="
                f"[{len(source_texts)}, {self._hidden_size}], got "
                f"{tuple(control_deltas.shape)}."
            )
        for index in action_indices:
            if not 0 <= index < len(self._options):
                raise ValueError(
                    f"action index {index} outside the {len(self._options)}"
                    "-way action vocabulary."
                )
        if not self._joint_training:
            self._assert_fully_frozen()

        batch = len(source_texts)
        encoded = self._tokenizer(
            [text + self._prompt_suffix for text in source_texts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length,
        )
        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded["attention_mask"].to(self._device)

        # Differentiable norm cap keeps the steered forward numerically
        # sane without detaching the graph.
        deltas = control_deltas.to(dtype=torch.float32)
        norms = deltas.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        scale = torch.clamp(self._control_norm_cap / norms, max=1.0)
        deltas = (deltas * scale).to(self._device, dtype=self._model_dtype)

        def hook(module: Any, args: Any, output: Any) -> Any:
            del module, args
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor):
                raise TypeError(
                    f"Steered scorer hook on {self.model_id!r} saw a "
                    "non-tensor block output."
                )
            adjusted = hidden + deltas.view(batch, 1, self._hidden_size).to(
                dtype=hidden.dtype
            )
            if isinstance(output, tuple):
                return (adjusted, *output[1:])
            return adjusted

        handle = self._blocks[
            self._injection_layer_index
        ].register_forward_hook(hook)
        try:
            grad_context = (
                torch.enable_grad() if enable_grad else torch.no_grad()
            )
            with grad_context:
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
        finally:
            handle.remove()
        logits = outputs.logits
        if not isinstance(logits, torch.Tensor):
            raise TypeError(
                f"Steered scorer on {self.model_id!r} did not receive tensor "
                "logits."
            )
        lengths = attention_mask.sum(dim=-1) - 1
        row_index = torch.arange(batch, device=self._device)
        last_logits = logits[row_index, lengths]
        candidate_logits = last_logits[:, self._candidate_token_ids]
        log_probs = torch.log_softmax(
            candidate_logits.to(dtype=torch.float32), dim=-1
        )
        target = torch.tensor(
            list(action_indices), dtype=torch.long, device=self._device
        )
        nll = -log_probs[row_index, target]
        # MPS cannot represent float64; hop to CPU first, then widen so the
        # temporal owner's float64 objective can consume the graph directly.
        return nll.to(device="cpu").to(dtype=torch.float64)

    def _probe_hidden_norm(self, probe_texts: tuple[str, ...]) -> float:
        torch = self._torch
        norms: list[float] = []

        def norm_hook(module: Any, args: Any, output: Any) -> None:
            del module, args
            hidden = output[0] if isinstance(output, tuple) else output
            norms.append(float(hidden.norm(dim=-1).mean()))
            return None

        encoded = self._tokenizer(
            list(probe_texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length,
        )
        handle = self._blocks[
            self._injection_layer_index
        ].register_forward_hook(norm_hook)
        try:
            with torch.no_grad():
                self._model(
                    input_ids=encoded["input_ids"].to(self._device),
                    attention_mask=encoded["attention_mask"].to(self._device),
                    use_cache=False,
                )
        finally:
            handle.remove()
        if not norms:
            raise RuntimeError(
                f"Steered scorer probe on {self.model_id!r} captured no "
                "hidden norm at the injection layer."
            )
        return norms[0]

    def _assert_fully_frozen(self) -> None:
        for parameter in self._model.parameters():
            if parameter.requires_grad:
                raise RuntimeError(
                    f"Frozen-arm steered scorer on {self.model_id!r} found a "
                    "trainable base parameter; the substrate contract "
                    "(R2 frozen basis) is violated."
                )
