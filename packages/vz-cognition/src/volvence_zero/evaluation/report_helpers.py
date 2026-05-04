"""Small :class:`EvaluationReport` read helpers.

``_report_trend`` and ``_report_metric_mean`` are private readout helpers
used by :class:`EvaluationBackbone` to pull trend values or mean metric
values out of an already-built :class:`EvaluationReport`.

Slice S.2 (2026-05-04): extracted from ``evaluation/backbone.py``.
"""

from __future__ import annotations

from volvence_zero.evaluation.types import EvaluationReport


def _report_trend(
    report: EvaluationReport,
    *,
    family: str,
    metric_name: str,
) -> float:
    for trend_family, trend_metric, value in report.trends:
        if trend_family == family and trend_metric == metric_name:
            return value
    return 0.0


def _report_metric_mean(
    report: EvaluationReport,
    *,
    family: str,
    metric_name: str,
) -> float:
    for score_family, records in report.scores_by_family:
        if score_family != family:
            continue
        values = [record.value for record in records if record.metric_name == metric_name]
        if values:
            return sum(values) / len(values)
    return 0.0


