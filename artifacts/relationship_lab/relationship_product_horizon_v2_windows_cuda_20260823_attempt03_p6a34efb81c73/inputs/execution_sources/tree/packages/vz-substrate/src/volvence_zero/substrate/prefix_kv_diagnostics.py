"""Mechanism-level measurements for the State-KV prefix carrier.

The identification lane asks a behavioural question -- do two users get
different answers. This module asks the two questions underneath it, which are
cheaper and fail for different reasons:

* **Is the prefix read at all?** Attention weights on the state slots, measured
  against zero and random-content controls.
* **Does the state reach the real tokens?** A linear probe fitted on the
  hidden state of the *prompt* positions after prefill -- not on the prefix
  tensors themselves, which are a deterministic function of the readout and
  would make the probe circular.

One control finding shapes the whole design and is worth stating up front:
**slot attention mass is an attention-sink artifact, not evidence.** On
Qwen2.5-0.5B a zero-content prefix draws 0.347 of the final query's attention
and a random prefix 0.339, against a uniform expectation of 0.16. Near-zero
attention logits absorb mass that real tokens, whose logits are mostly
negative, do not compete for. Any gate defined on total slot mass is passed by
a zero tensor, so mass is reported here and never asserted on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

__all__ = [
    "RIDGE_ALPHA_GRID",
    "ClassificationProbeFit",
    "PrefixAttentionProfile",
    "ProbeFit",
    "capture_prefix_diagnostics",
    "fit_linear_classification_probe",
    "fit_ridge_probe",
    "profile_spread",
    "select_ridge_alpha",
]

# Fixed grid, searched by grouped cross-validation on the *training* states
# only. An alpha picked against held-out performance would make the probe's
# own controls meaningless, which is how the first run of this gate produced a
# held-out R^2 of 0.89 alongside a shuffled-label control of +0.12: with 896
# features, 384 samples and alpha=1 the fit interpolates, and the evaluation
# states -- which reuse the same probe sentences -- sit close enough to the
# training features for that interpolation to score.
RIDGE_ALPHA_GRID: tuple[float, ...] = (
    1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8,
)


@dataclass(frozen=True)
class PrefixAttentionProfile:
    """Per-layer attention statistics for one (state, probe) forward pass.

    All statistics are taken at the **final prompt position** -- the query that
    emits the first response token. Earlier positions attend to the prefix too,
    but that one is the position whose output the identification lane reads.
    """

    slot_mass: tuple[float, ...]
    """Fraction of the final query's attention landing on the prefix slots."""

    slot_nonuniformity: tuple[float, ...]
    """Within-prefix max/min attention ratio; 1.0 means slots are interchangeable."""

    per_slot_mass: tuple[tuple[float, ...], ...]
    """Per-layer, per-slot attention mass (heads averaged)."""

    final_hidden: tuple[tuple[float, ...], ...] = field(default=())
    """Per-layer hidden state at the final prompt position."""

    @property
    def num_layers(self) -> int:
        return len(self.slot_mass)


def capture_prefix_diagnostics(
    *,
    torch_module: Any,
    model: Any,
    input_ids: Any,
    prefix_pairs: Sequence[tuple[Any, Any]] | None,
    cache_factory: Any,
    capture_hidden: bool = True,
) -> PrefixAttentionProfile:
    """One forward pass over a prefilled cache, recording attention and hidden state.

    ``prefix_pairs=None`` records the no-prefix control; slot statistics are
    then empty and only the hidden state is meaningful. Positions are pinned to
    ``0..n-1`` exactly as the generation path does, so the measurement describes
    the arm that actually runs rather than a differently-positioned variant.
    """

    torch = torch_module
    slots = 0
    cache = cache_factory()
    if prefix_pairs:
        slots = int(prefix_pairs[0][0].shape[-2])
        cache = cache_factory(prefix_pairs)
    length = int(input_ids.shape[-1])
    attention_mask = torch.ones(
        (1, slots + length), dtype=torch.long, device=input_ids.device
    )
    position_ids = torch.arange(length, device=input_ids.device).unsqueeze(0)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            output_attentions=slots > 0,
            output_hidden_states=capture_hidden,
        )

    mass: list[float] = []
    nonuniformity: list[float] = []
    per_slot: list[tuple[float, ...]] = []
    if slots > 0:
        for layer in outputs.attentions:
            # (heads, kv) at the final query position, averaged over heads.
            head_view = layer[0, :, -1, :].to(torch.float32)
            slot_view = head_view[:, :slots].mean(0)
            total = float(slot_view.sum())
            smallest = float(slot_view.min())
            largest = float(slot_view.max())
            mass.append(total)
            # A zero-content prefix scores exactly 1.0 here by construction:
            # identical slots cannot be told apart. That is the null this
            # statistic is designed to sit against.
            nonuniformity.append(
                largest / smallest if smallest > 1e-12 else 0.0
            )
            per_slot.append(
                tuple(float(value) for value in slot_view.detach().cpu().tolist())
            )

    hidden: list[tuple[float, ...]] = []
    if capture_hidden:
        # hidden_states[0] is the embedding output; the transformer blocks are
        # 1..L, which is the indexing the attention list already uses.
        for state in outputs.hidden_states[1:]:
            final_prompt_hidden = state[0, -1].to(torch.float32)
            hidden.append(
                tuple(
                    float(value)
                    for value in final_prompt_hidden.detach().cpu().tolist()
                )
            )

    return PrefixAttentionProfile(
        slot_mass=tuple(mass),
        slot_nonuniformity=tuple(nonuniformity),
        per_slot_mass=tuple(per_slot),
        final_hidden=tuple(hidden),
    )


def profile_spread(profiles: Sequence[Sequence[float]]) -> float:
    """Mean per-layer standard deviation across a set of profiles.

    Used for the state-modulation check: the spread of slot attention across
    *states* (same sentence) is compared against its spread across *sentences*
    (same state). A carrier that tracks who the user is must vary more with the
    former than the latter; a constant bias varies with neither.
    """

    if len(profiles) < 2:
        return 0.0
    layers = len(profiles[0])
    if any(len(profile) != layers for profile in profiles):
        raise ValueError("all profiles must have the same layer count.")
    total = 0.0
    for index in range(layers):
        column = [float(profile[index]) for profile in profiles]
        mean = sum(column) / len(column)
        variance = sum((value - mean) ** 2 for value in column) / len(column)
        total += variance ** 0.5
    return total / layers


@dataclass(frozen=True)
class ProbeFit:
    """Held-out ridge-probe result for one layer."""

    layer_index: int
    mean_r2: float
    per_coordinate_r2: tuple[float, ...]

    def as_json_dict(self) -> dict[str, object]:
        return {
            "layer": self.layer_index,
            "mean_r2": round(self.mean_r2, 6),
            "per_coordinate_r2": [
                round(value, 6) for value in self.per_coordinate_r2
            ],
        }


def fit_ridge_probe(
    *,
    torch_module: Any,
    train_features: Any,
    train_targets: Any,
    eval_features: Any,
    eval_targets: Any,
    layer_index: int,
    alpha: float = 1.0,
) -> ProbeFit:
    """Closed-form ridge from hidden state to the frozen 16-dim readout.

    Scored on **held-out states**, not held-out samples: the same state seen
    with a different probe sentence is not an independent test point, and
    splitting by sample would let the probe memorise states it was fit on.

    ``R^2 <= 0`` means the probe does no better than predicting the evaluation
    mean, which is the honest reading of "the state is not linearly present".
    """

    fitted = _ridge_solve(torch_module, train_features, train_targets, alpha)
    predictions = _predict(eval_features, fitted)
    residual = ((eval_targets - predictions) ** 2).sum(dim=0)
    total = (
        (eval_targets - eval_targets.mean(dim=0, keepdim=True)) ** 2
    ).sum(dim=0)
    scores: list[float] = []
    for index in range(int(eval_targets.shape[1])):
        denominator = float(total[index])
        if denominator <= 1e-12:
            # A coordinate with no variance across the evaluation states
            # carries no signal to recover; scoring it would inflate the mean.
            continue
        scores.append(1.0 - float(residual[index]) / denominator)
    mean_r2 = sum(scores) / len(scores) if scores else 0.0
    return ProbeFit(
        layer_index=layer_index,
        mean_r2=mean_r2,
        per_coordinate_r2=tuple(scores),
    )


@dataclass(frozen=True)
class ClassificationProbeFit:
    """Held-out linear classification probe result for one layer.

    Decodes a discrete label (e.g. the active subgoal) from a hidden state by
    least-squares classification: closed-form ridge onto one-hot targets, then
    argmax. Scored on held-out states. ``chance_accuracy`` is the uniform
    1/class baseline; ``majority_accuracy`` is the majority-class baseline. Gate
    checks that read "N x random" compare against ``chance_accuracy``.
    """

    layer_index: int
    accuracy: float
    chance_accuracy: float
    majority_accuracy: float
    class_count: int
    support: int

    def as_json_dict(self) -> dict[str, object]:
        return {
            "layer": self.layer_index,
            "accuracy": round(self.accuracy, 6),
            "chance_accuracy": round(self.chance_accuracy, 6),
            "majority_accuracy": round(self.majority_accuracy, 6),
            "class_count": self.class_count,
            "support": self.support,
        }


def fit_linear_classification_probe(
    *,
    torch_module: Any,
    train_features: Any,
    train_labels: Any,
    eval_features: Any,
    eval_labels: Any,
    layer_index: int,
    class_count: int,
    alpha: float = 1.0,
) -> ClassificationProbeFit:
    """Least-squares linear classifier from hidden state to a discrete label.

    ``*_labels`` are 1-D integer tensors of class ids in ``[0, class_count)``.
    The fit is closed-form ridge onto one-hot targets (deterministic, no
    iteration), and accuracy is argmax agreement on the held-out states. This
    is the classification analogue of :func:`fit_ridge_probe`; it exists so a
    gate can ask "is the active subgoal linearly decodable" rather than "is the
    16-dim readout linearly recoverable".
    """

    torch = torch_module
    if class_count < 2:
        raise ValueError("class_count must be at least 2 for a classification probe.")
    train_labels_long = train_labels.to(torch.long)
    eval_labels_long = eval_labels.to(torch.long)
    if alpha <= 0.0:
        raise ValueError("classification probe alpha must be > 0.")
    if train_labels_long.ndim != 1 or eval_labels_long.ndim != 1:
        raise ValueError("classification probe labels must be one-dimensional.")
    if int(train_labels_long.shape[0]) == 0 or int(eval_labels_long.shape[0]) == 0:
        raise ValueError("classification probe requires non-empty train and eval labels.")
    if train_features.ndim != 2 or eval_features.ndim != 2:
        raise ValueError("classification probe features must be two-dimensional.")
    if int(train_features.shape[0]) != int(train_labels_long.shape[0]):
        raise ValueError("train feature and label counts must match.")
    if int(eval_features.shape[0]) != int(eval_labels_long.shape[0]):
        raise ValueError("eval feature and label counts must match.")
    if int(train_features.shape[1]) != int(eval_features.shape[1]):
        raise ValueError("train and eval feature widths must match.")
    if int(train_labels_long.min()) < 0 or int(train_labels_long.max()) >= class_count:
        raise ValueError("train labels fall outside [0, class_count).")
    if int(eval_labels_long.min()) < 0 or int(eval_labels_long.max()) >= class_count:
        raise ValueError("eval labels fall outside [0, class_count).")
    one_hot = torch.zeros(
        (int(train_labels_long.shape[0]), class_count),
        dtype=train_features.dtype,
        device=train_features.device,
    )
    one_hot.scatter_(1, train_labels_long.unsqueeze(1), 1.0)
    fitted = _ridge_solve(torch, train_features, one_hot, alpha)
    scores = _predict(eval_features, fitted)
    predictions = scores.argmax(dim=1)
    support = int(eval_labels_long.shape[0])
    correct = int((predictions == eval_labels_long).sum())
    accuracy = correct / support if support else 0.0
    counts = torch.bincount(eval_labels_long, minlength=class_count)
    majority_accuracy = float(counts.max()) / support if support else 0.0
    return ClassificationProbeFit(
        layer_index=layer_index,
        accuracy=accuracy,
        chance_accuracy=1.0 / class_count,
        majority_accuracy=majority_accuracy,
        class_count=class_count,
        support=support,
    )


def _ridge_solve(torch_module: Any, features: Any, targets: Any, alpha: float):
    """Centred, standardised ridge. Returns the pieces needed to predict."""

    torch = torch_module
    x_mean = features.mean(dim=0, keepdim=True)
    x_scale = features.std(dim=0, keepdim=True).clamp_min(1e-6)
    y_mean = targets.mean(dim=0, keepdim=True)
    centred_x = (features - x_mean) / x_scale
    centred_y = targets - y_mean
    width = centred_x.shape[1]
    gram = centred_x.T @ centred_x + alpha * torch.eye(
        width, dtype=centred_x.dtype, device=centred_x.device
    )
    weights = torch.linalg.solve(gram, centred_x.T @ centred_y)
    return x_mean, x_scale, y_mean, weights


def _predict(features: Any, fitted) -> Any:
    x_mean, x_scale, y_mean, weights = fitted
    return ((features - x_mean) / x_scale) @ weights + y_mean


def _mean_r2(targets: Any, predictions: Any) -> float:
    residual = ((targets - predictions) ** 2).sum(dim=0)
    total = ((targets - targets.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
    scores = [
        1.0 - float(residual[index]) / float(total[index])
        for index in range(int(targets.shape[1]))
        if float(total[index]) > 1e-12
    ]
    return sum(scores) / len(scores) if scores else 0.0


def select_ridge_alpha(
    *,
    torch_module: Any,
    features: Any,
    targets: Any,
    groups: Sequence[int],
    alphas: Sequence[float] = RIDGE_ALPHA_GRID,
    folds: int = 4,
) -> float:
    """Pick ridge alpha by grouped cross-validation on the training set only.

    Folds split on ``groups`` (the state each row came from), never on rows:
    the same state seen under a different probe sentence is not an independent
    validation point, and splitting by row would select an alpha that has
    already seen every state.

    The held-out evaluation states are not consulted here, which is what keeps
    the shuffled-label and no-prefix controls meaningful at the selected alpha.
    """

    unique = sorted(set(int(g) for g in groups))
    if len(unique) < folds or not len(alphas):
        return float(alphas[len(alphas) // 2]) if len(alphas) else 1.0
    assignment = {group: index % folds for index, group in enumerate(unique)}
    row_fold = [assignment[int(g)] for g in groups]
    best_alpha = float(alphas[0])
    best_score = float("-inf")
    for alpha in alphas:
        scores = []
        for fold in range(folds):
            train_idx = [i for i, f in enumerate(row_fold) if f != fold]
            valid_idx = [i for i, f in enumerate(row_fold) if f == fold]
            if not train_idx or not valid_idx:
                continue
            train_index = torch_module.tensor(train_idx, device=features.device)
            valid_index = torch_module.tensor(valid_idx, device=features.device)
            fitted = _ridge_solve(
                torch_module,
                features.index_select(0, train_index),
                targets.index_select(0, train_index),
                float(alpha),
            )
            scores.append(
                _mean_r2(
                    targets.index_select(0, valid_index),
                    _predict(features.index_select(0, valid_index), fitted),
                )
            )
        mean_score = sum(scores) / len(scores) if scores else float("-inf")
        if mean_score > best_score:
            best_score, best_alpha = mean_score, float(alpha)
    return best_alpha
