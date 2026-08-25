"""Real-gradient Local Surprise Signal (NL/ETA migration, Phase 5).

NL defines the Local Surprise Signal (LSS) as the gradient of the loss with
respect to the model's output: "training a layer with backpropagation is
equivalent to building an associative memory that maps each input to its
prediction error", and that gradient *is the content being memorized*.

The live runtime keeps the semantic, turn-level :class:`PredictionError` as a
bounded proxy (it does not require autograd and runs every turn). This module
adds the **true** gradient-based LSS as a first-class, auditable OFFLINE
artifact computed via ``torch.autograd``, and shows the exact relationship
between the two so the proxy is grounded rather than asserted:

For an MSE loss ``L = 0.5 * ||o - t||^2`` the LSS is ``dL/do = (o - t)``. The
runtime PredictionError reports signed error as ``(actual - predicted) = -(o - t)``,
i.e. the runtime PE is exactly the negative true LSS (sign-correct, magnitude-
proportional). This module proves that identity and exports the LSS artifact.

``torch`` is imported lazily; this module is NOT re-exported from the
``volvence_zero.prediction`` facade, keeping it torch-free by default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - guarded at call sites
        raise ImportError(
            "Gradient LSS requires torch. Install the vz-cognition '[torch]' extra."
        ) from exc
    return torch


# The runtime PredictionError 4-axis order, kept here so the LSS artifact maps
# 1:1 onto the semantic PE axes (task / relationship / regime / action).
LSS_AXES: tuple[str, ...] = ("task", "relationship", "regime", "action")


@dataclass(frozen=True)
class LSSArtifact:
    """Gradient-based local surprise signal (floats only; rare-heavy path)."""

    axes: tuple[str, ...]
    local_surprise: tuple[float, ...]   # dL/do per output dim
    loss: float
    magnitude: float                    # L2 norm of the LSS
    predicted: tuple[float, ...]
    actual: tuple[float, ...]
    loss_kind: str
    description: str = ""


def compute_gradient_lss(
    predicted: Sequence[float],
    actual: Sequence[float],
    *,
    axes: Sequence[str] = LSS_AXES,
    loss_kind: str = "mse",
) -> LSSArtifact:
    """Compute LSS = dL/d(output) via real autograd.

    ``predicted`` is the model output ``o``; ``actual`` is the target ``t``.
    The returned ``local_surprise`` is the gradient of the loss w.r.t. ``o`` —
    NL's local surprise signal, the thing an associative memory memorizes.
    """

    torch = _require_torch()
    o = torch.tensor([float(v) for v in predicted], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([float(v) for v in actual], dtype=torch.float64)
    if loss_kind == "mse":
        loss = 0.5 * torch.sum((o - t).pow(2))
    elif loss_kind == "huber":
        loss = torch.nn.functional.smooth_l1_loss(o, t, reduction="sum")
    else:
        raise ValueError(f"unknown loss_kind {loss_kind!r}; expected 'mse' or 'huber'")
    loss.backward()
    lss = tuple(float(v) for v in o.grad.tolist())
    magnitude = math.sqrt(sum(v * v for v in lss))
    return LSSArtifact(
        axes=tuple(axes[: len(lss)]) if len(axes) >= len(lss) else tuple(
            list(axes) + [f"dim{i}" for i in range(len(axes), len(lss))]
        ),
        local_surprise=lss,
        loss=float(loss.detach()),
        magnitude=magnitude,
        predicted=tuple(float(v) for v in predicted),
        actual=tuple(float(v) for v in actual),
        loss_kind=loss_kind,
        description=(
            f"gradient LSS ({loss_kind}): |LSS|={magnitude:.4f} loss={float(loss.detach()):.4f}"
        ),
    )


@dataclass(frozen=True)
class LSSPredictorArtifact:
    """LSS computed through a 1-layer predictor: shows it flows to params too."""

    output_lss: tuple[float, ...]       # dL/do (memorized content at the output)
    weight_grad_norm: float             # ||dL/dW||: the LSS propagated into params
    loss: float
    parameters_changed: int
    description: str = ""


def compute_lss_through_predictor(
    inputs: Sequence[float],
    target: Sequence[float],
    *,
    out_dim: int | None = None,
    seed: int = 1234,
    learning_rate: float = 0.05,
) -> LSSPredictorArtifact:
    """Train a 1-layer predictor one step and read the LSS at its output.

    Demonstrates NL's claim that backprop builds an associative memory mapping
    input -> prediction-error: the gradient at the output (LSS) is propagated
    into the weights, and a step on it reduces loss.
    """

    torch = _require_torch()
    in_dim = len(inputs)
    out_dim = out_dim or len(target)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    W = torch.empty(out_dim, in_dim, dtype=torch.float64)
    W.normal_(0.0, 1.0 / math.sqrt(max(in_dim, 1)), generator=g)
    W.requires_grad_(True)
    b = torch.zeros(out_dim, dtype=torch.float64, requires_grad=True)
    x = torch.tensor([float(v) for v in inputs], dtype=torch.float64)
    t = torch.tensor([float(v) for v in target], dtype=torch.float64)

    o = torch.matmul(W, x) + b
    o.retain_grad()
    loss = 0.5 * torch.sum((o - t).pow(2))
    loss.backward()

    output_lss = tuple(float(v) for v in o.grad.tolist())
    weight_grad_norm = math.sqrt(float(W.grad.pow(2).sum()))

    before = W.detach().clone()
    with torch.no_grad():
        W -= learning_rate * W.grad
        b -= learning_rate * b.grad
    changed = int(((W.detach() - before).abs() > 1e-12).sum())

    return LSSPredictorArtifact(
        output_lss=output_lss,
        weight_grad_norm=weight_grad_norm,
        loss=float(loss.detach()),
        parameters_changed=changed,
        description=(
            f"predictor LSS: |dL/dW|={weight_grad_norm:.4f}, "
            f"params_changed={changed}"
        ),
    )


@dataclass(frozen=True)
class LSSProxyBridgeReport:
    lss: tuple[float, ...]
    runtime_signed_pe: tuple[float, ...]
    proxy_is_negative_lss: bool
    magnitude_correlates: bool
    description: str = ""


@dataclass(frozen=True)
class TypedLSSBridgeReport:
    """One registered Gate 1 PE surface compared with its true autograd LSS.

    Components are intentionally kept per-surface.  The link functions have
    different parameterizations and units, so this artifact must never be
    reduced across surface kinds into a new scalar learning signal.
    """

    surface_kind: str
    link_function: str
    predicted: tuple[float, ...]
    actual: tuple[float, ...]
    runtime_signed_pe: tuple[float, ...]
    lss: tuple[float, ...]
    component_decomposition: tuple[
        tuple[str, float, float, float], ...
    ]
    loss: float
    max_abs_bridge_error: float
    bridge_passed: bool
    target_index: int | None = None
    description: str = ""


_GATE1_SURFACE_KINDS = frozenset(
    {"numeric", "probability", "enum", "vector", "distribution"}
)


def _finite_tuple(
    values: Sequence[float],
    *,
    field_name: str,
) -> tuple[float, ...]:
    resolved = tuple(float(value) for value in values)
    if not resolved:
        raise ValueError(f"{field_name} must be non-empty")
    if not all(math.isfinite(value) for value in resolved):
        raise ValueError(f"{field_name} must contain only finite values")
    return resolved


def _probability_simplex(
    values: Sequence[float],
    *,
    field_name: str,
    strictly_positive: bool,
) -> tuple[float, ...]:
    resolved = _finite_tuple(values, field_name=field_name)
    if strictly_positive:
        valid_components = all(0.0 < value < 1.0 for value in resolved)
    else:
        valid_components = all(0.0 <= value <= 1.0 for value in resolved)
    if not valid_components or abs(sum(resolved) - 1.0) > 1e-9:
        qualifier = "strictly positive " if strictly_positive else ""
        raise ValueError(
            f"{field_name} must be a normalized {qualifier}probability simplex"
        )
    return resolved


def bridge_typed_runtime_pe_to_lss(
    *,
    surface_kind: str,
    predicted: Sequence[float],
    actual: Sequence[float] | float | int,
) -> TypedLSSBridgeReport:
    """Compare a registered typed runtime PE with its true link gradient.

    ``runtime_signed_pe`` is always target minus prediction in the link's
    observation space.  For each registered parameterization it must equal
    ``-dL/d(parameter)`` within the preregistered ``1e-9`` tolerance.
    """

    if surface_kind not in _GATE1_SURFACE_KINDS:
        raise ValueError(
            f"unsupported Gate 1 surface_kind {surface_kind!r}; expected one "
            f"of {sorted(_GATE1_SURFACE_KINDS)!r}"
        )
    torch = _require_torch()
    target_index: int | None = None
    if surface_kind in {"numeric", "vector"}:
        predicted_values = _finite_tuple(
            predicted,
            field_name="predicted",
        )
        if isinstance(actual, (float, int)):
            actual_values = (float(actual),)
        else:
            actual_values = _finite_tuple(actual, field_name="actual")
        if surface_kind == "numeric" and (
            len(predicted_values) != 1 or len(actual_values) != 1
        ):
            raise ValueError(
                "numeric Gate 1 surface requires scalar predicted and actual"
            )
        if len(predicted_values) != len(actual_values):
            raise ValueError(
                "Gate 1 MSE predicted/actual dimensions must match"
            )
        parameter = torch.tensor(
            predicted_values,
            dtype=torch.float64,
            requires_grad=True,
        )
        target = torch.tensor(actual_values, dtype=torch.float64)
        loss = 0.5 * torch.sum((parameter - target).pow(2))
        runtime_signed = tuple(
            target_value - predicted_value
            for predicted_value, target_value in zip(
                predicted_values,
                actual_values,
                strict=True,
            )
        )
        link_function = "scalar-mse" if surface_kind == "numeric" else "vector-mse"
    elif surface_kind == "probability":
        predicted_values = _finite_tuple(
            predicted,
            field_name="predicted",
        )
        if len(predicted_values) != 1 or not isinstance(actual, (float, int)):
            raise ValueError(
                "probability Gate 1 surface requires one probability and "
                "one scalar target"
            )
        probability = predicted_values[0]
        target_value = float(actual)
        if (
            not 0.0 < probability < 1.0
            or not math.isfinite(target_value)
            or target_value not in {0.0, 1.0}
        ):
            raise ValueError(
                "probability Gate 1 surface requires prediction in (0, 1) "
                "and binary target in {0, 1}"
            )
        parameter = torch.tensor(
            [math.log(probability / (1.0 - probability))],
            dtype=torch.float64,
            requires_grad=True,
        )
        target = torch.tensor([target_value], dtype=torch.float64)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            parameter,
            target,
            reduction="sum",
        )
        actual_values = (target_value,)
        runtime_signed = (target_value - probability,)
        link_function = "bernoulli-cross-entropy-logit"
    else:
        predicted_values = _probability_simplex(
            predicted,
            field_name="predicted",
            strictly_positive=True,
        )
        parameter = torch.tensor(
            [math.log(value) for value in predicted_values],
            dtype=torch.float64,
            requires_grad=True,
        )
        if surface_kind == "enum":
            if not isinstance(actual, int) or isinstance(actual, bool):
                raise ValueError(
                    "enum Gate 1 surface requires an integer target index"
                )
            target_index = actual
            if target_index < 0 or target_index >= len(predicted_values):
                raise ValueError(
                    "enum Gate 1 target index is outside the prediction "
                    "category range"
                )
            actual_values = tuple(
                1.0 if index == target_index else 0.0
                for index in range(len(predicted_values))
            )
            loss = torch.nn.functional.cross_entropy(
                parameter.unsqueeze(0),
                torch.tensor([target_index], dtype=torch.long),
                reduction="sum",
            )
            link_function = "categorical-cross-entropy-logit"
        else:
            if isinstance(actual, (float, int)):
                raise ValueError(
                    "distribution Gate 1 surface requires a target "
                    "probability simplex"
                )
            actual_values = _probability_simplex(
                actual,
                field_name="actual",
                strictly_positive=False,
            )
            if len(predicted_values) != len(actual_values):
                raise ValueError(
                    "distribution predicted/actual dimensions must match"
                )
            target = torch.tensor(actual_values, dtype=torch.float64)
            loss = -torch.sum(
                target
                * torch.nn.functional.log_softmax(parameter, dim=0)
            )
            link_function = "soft-categorical-cross-entropy-logit"
        runtime_signed = tuple(
            target_value - predicted_value
            for predicted_value, target_value in zip(
                predicted_values,
                actual_values,
                strict=True,
            )
        )
    loss.backward()
    lss = tuple(float(value) for value in parameter.grad.tolist())
    bridge_errors = tuple(
        abs(runtime_value + lss_value)
        for runtime_value, lss_value in zip(
            runtime_signed,
            lss,
            strict=True,
        )
    )
    max_abs_bridge_error = max(bridge_errors, default=0.0)
    component_decomposition = tuple(
        (
            f"component-{index}",
            predicted_value,
            actual_value,
            runtime_value,
        )
        for index, (
            predicted_value,
            actual_value,
            runtime_value,
        ) in enumerate(
            zip(
                predicted_values,
                actual_values,
                runtime_signed,
                strict=True,
            )
        )
    )
    return TypedLSSBridgeReport(
        surface_kind=surface_kind,
        link_function=link_function,
        predicted=predicted_values,
        actual=actual_values,
        runtime_signed_pe=runtime_signed,
        lss=lss,
        component_decomposition=component_decomposition,
        loss=float(loss.detach()),
        max_abs_bridge_error=max_abs_bridge_error,
        bridge_passed=max_abs_bridge_error <= 1e-9,
        target_index=target_index,
        description=(
            f"{surface_kind} runtime PE == -LSS via {link_function}: "
            f"{max_abs_bridge_error <= 1e-9}"
        ),
    )


def bridge_runtime_pe_to_lss(
    *,
    predicted: Sequence[float],
    actual: Sequence[float],
) -> LSSProxyBridgeReport:
    """Show the runtime semantic PE is exactly the negative true gradient LSS.

    Phase 5 grounding: the runtime PredictionError signed error is
    ``actual - predicted``; the true MSE LSS is ``predicted - actual``. So the
    runtime proxy equals ``-LSS`` exactly — sign-correct and magnitude-equal —
    which is why the bounded runtime signal is a faithful stand-in for the
    gradient surprise without needing per-turn autograd.
    """

    artifact = compute_gradient_lss(predicted, actual)
    runtime_signed = tuple(
        float(a) - float(p)
        for p, a in zip(predicted, actual, strict=True)
    )
    proxy_is_neg = all(
        abs(rs - (-lss)) <= 1e-9
        for rs, lss in zip(
            runtime_signed,
            artifact.local_surprise,
            strict=True,
        )
    )
    # larger |runtime PE| <=> larger |LSS|
    magnitude_ok = abs(
        math.sqrt(sum(v * v for v in runtime_signed)) - artifact.magnitude
    ) <= 1e-9
    return LSSProxyBridgeReport(
        lss=artifact.local_surprise,
        runtime_signed_pe=runtime_signed,
        proxy_is_negative_lss=proxy_is_neg,
        magnitude_correlates=magnitude_ok,
        description=(
            "runtime PE == -LSS (sign-correct, magnitude-equal): "
            f"{proxy_is_neg and magnitude_ok}"
        ),
    )
