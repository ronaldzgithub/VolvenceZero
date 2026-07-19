"""Runtime module: model loading, feature caching, dataset management."""

from .model import ModelRuntime
from .model_cache import get_model_runtime, clear_model_cache, cache_size
from .cache import FeatureCache
from .dataset import DatasetSnapshot
from .uncertainty import (
    epistemic_aleatoric_split,
    cross_entropy_per_token,
    softmax,
    mc_dropout_logits,
)

__all__ = [
    "ModelRuntime",
    "get_model_runtime",
    "clear_model_cache",
    "cache_size",
    "FeatureCache",
    "DatasetSnapshot",
    "epistemic_aleatoric_split",
    "cross_entropy_per_token",
    "softmax",
    "mc_dropout_logits",
]
