"""Readout module: SQL views + static HTML dashboard."""

from .views import ablation_diff, metrics_pivot, runs_view
from .dashboard import generate_dashboard

__all__ = ["ablation_diff", "metrics_pivot", "runs_view", "generate_dashboard"]
