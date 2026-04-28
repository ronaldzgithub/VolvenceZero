"""SSL training-loop demo — closes the trace \u2192 metacontroller loop.

This is the M1 of "downward growth": it takes the traces the
``TraceCollector`` produced, converts them into ``vz-substrate.TrainingTrace``
sequences, and runs ``vz-temporal.MetacontrollerSSLTrainer.optimize`` on each.
The aggregate report shows whether the metacontroller actually learnt
something: prediction loss should go DOWN, switch sparsity (\u03b2_t binarisation)
should drift away from random, and posterior drift should accumulate.

It does NOT yet:

* run the second-stage Internal RL loop (that is M2 of downward growth).
* swap the trained policy back into a live ``Brain`` and re-run lifeform-bench
  to observe behaviour change (that is M3 \u2014 needs Brain to accept a
  pre-trained ``FullLearnedTemporalPolicy``, which the kernel has not exposed
  on the public surface yet; see the closing follow-up below).

What it DOES:

* prove that traces collected on the synthetic substrate are semantically
  rich enough to drive non-trivial SSL gradients.
* expose machine-readable metrics so lifeform-evolution dashboards can
  watch training health over time (R12 / R13 \u2014 SSL phase visible).
* serve as the public entry point ``lifeform-ssl`` so ops / researchers can
  drive a training run from the command line.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterable

from volvence_zero.substrate import TrainingTrace, TrainingTraceDataset
from volvence_zero.temporal import (
    FullLearnedTemporalPolicy,
    MetacontrollerSSLTrainer,
    SSLTrainingReport,
    fit_policy_from_trace_dataset,
)

from lifeform_evolution.benchmark import all_built_in_scenarios
from lifeform_evolution.dataset_adapter import (
    trace_records_to_training_dataset,
    trace_records_from_ndjson,
)
from lifeform_evolution.trace_collector import TraceCollector, TraceTurnRecord


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SSLDemoReport:
    """Roll-up of the SSL training run.

    The per-trace ``SSLTrainingReport`` instances are also kept so dashboards
    can plot the trajectory across traces.

    ``switch_frequency`` is the ETA paper's \u03b2_t firing rate \u2014 fraction of
    steps where the switch gate fires. A trained metacontroller drives this
    toward a sparse, subgoal-aligned regime; for an under-trained one it
    sits near 0.5 (random). The first/last fields let dashboards see whether
    training actually moved this metric.
    """

    trace_count: int
    trained_step_count: int
    prediction_loss_first: float
    prediction_loss_last: float
    prediction_loss_delta: float
    kl_loss_mean: float
    switch_frequency_first: float | None
    switch_frequency_last: float | None
    mean_persistence_first: float | None
    mean_persistence_last: float | None
    posterior_drift_total: float
    per_trace_reports: tuple[SSLTrainingReport, ...] = field(default_factory=tuple)
    description: str = ""

    def loss_decreased(self) -> bool:
        return self.prediction_loss_delta < 0.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_ssl_demo(
    *,
    records: Iterable[TraceTurnRecord] | None = None,
    dataset: TrainingTraceDataset | None = None,
    policy: FullLearnedTemporalPolicy | None = None,
    trainer: MetacontrollerSSLTrainer | None = None,
    n_z: int = 3,
    alpha: float = 0.1,
) -> SSLDemoReport:
    """Run SSL training over a trace dataset and roll up the metrics.

    Args:
        records: ``TraceTurnRecord`` rows; converted via ``dataset_adapter``.
            Mutually exclusive with ``dataset``.
        dataset: pre-built ``TrainingTraceDataset``. Mutually exclusive with
            ``records``. If neither is given, the runner collects all built-in
            scenarios from a fresh lifeform on the synthetic substrate.
        policy: ``FullLearnedTemporalPolicy`` to train. Defaults to a fresh one.
        trainer: ``MetacontrollerSSLTrainer`` instance. Defaults to a fresh one.
        n_z, alpha: forwarded to the trainer constructor when ``trainer`` is None.
    """
    if records is not None and dataset is not None:
        raise ValueError("Pass either records or dataset, not both.")

    if dataset is None:
        if records is None:
            records = _collect_default_records()
        dataset = trace_records_to_training_dataset(records)

    if not dataset.traces:
        raise ValueError("SSL demo requires at least one trace; dataset is empty.")

    policy = policy or FullLearnedTemporalPolicy()
    trainer = trainer or MetacontrollerSSLTrainer(n_z=n_z, alpha=alpha)

    per_trace: list[SSLTrainingReport] = []
    posterior_drift_total = 0.0
    for trace in dataset.traces:
        report = trainer.optimize(policy=policy, trace=trace)
        per_trace.append(report)
        posterior_drift_total += report.posterior_drift

    # ``fit_from_signals`` propagates aggregate scalars into the parameter
    # store so the policy is also useful outside the SSL phase. This mirrors
    # ``fit_policy_from_trace_dataset`` for the Lite policy.
    fit_policy_from_trace_dataset(policy=_lite_view_of(policy), dataset=dataset)

    first_report = per_trace[0]
    last_report = per_trace[-1]
    kl_mean = (
        sum(r.kl_loss for r in per_trace) / len(per_trace)
        if per_trace
        else 0.0
    )
    switch_first_freq = (
        first_report.switch_gate_stats.switch_frequency
        if first_report.switch_gate_stats is not None
        else None
    )
    switch_last_freq = (
        last_report.switch_gate_stats.switch_frequency
        if last_report.switch_gate_stats is not None
        else None
    )
    persistence_first = (
        first_report.switch_gate_stats.mean_persistence_steps
        if first_report.switch_gate_stats is not None
        else None
    )
    persistence_last = (
        last_report.switch_gate_stats.mean_persistence_steps
        if last_report.switch_gate_stats is not None
        else None
    )
    description = (
        f"SSL demo: {len(per_trace)} traces, "
        f"{sum(r.trained_steps for r in per_trace)} steps, "
        f"loss \u0394 = {last_report.prediction_loss - first_report.prediction_loss:+.4f}, "
        f"posterior drift total = {posterior_drift_total:.4f}"
    )

    return SSLDemoReport(
        trace_count=len(per_trace),
        trained_step_count=sum(r.trained_steps for r in per_trace),
        prediction_loss_first=first_report.prediction_loss,
        prediction_loss_last=last_report.prediction_loss,
        prediction_loss_delta=last_report.prediction_loss - first_report.prediction_loss,
        kl_loss_mean=kl_mean,
        switch_frequency_first=switch_first_freq,
        switch_frequency_last=switch_last_freq,
        mean_persistence_first=persistence_first,
        mean_persistence_last=persistence_last,
        posterior_drift_total=posterior_drift_total,
        per_trace_reports=tuple(per_trace),
        description=description,
    )


def run_ssl_demo_from_ndjson(
    *,
    path: str,
    policy: FullLearnedTemporalPolicy | None = None,
    trainer: MetacontrollerSSLTrainer | None = None,
    n_z: int = 3,
    alpha: float = 0.1,
) -> SSLDemoReport:
    """Convenience: read collector output back from disk, then train."""
    records = trace_records_from_ndjson(path)
    return run_ssl_demo(records=records, policy=policy, trainer=trainer, n_z=n_z, alpha=alpha)


def format_ssl_demo_report(report: SSLDemoReport) -> str:
    lines: list[str] = []
    lines.append("== Lifeform SSL demo ==")
    lines.append(
        f"   traces: {report.trace_count}    "
        f"trained steps: {report.trained_step_count}    "
        f"posterior drift total: {report.posterior_drift_total:.4f}"
    )
    lines.append(
        f"   prediction loss: {report.prediction_loss_first:.4f} \u2192 "
        f"{report.prediction_loss_last:.4f}  "
        f"({report.prediction_loss_delta:+.4f})"
    )
    lines.append(f"   kl loss (mean): {report.kl_loss_mean:.4f}")
    if report.switch_frequency_first is not None and report.switch_frequency_last is not None:
        lines.append(
            f"   switch freq:    {report.switch_frequency_first:.3f} \u2192 "
            f"{report.switch_frequency_last:.3f}"
        )
    if report.mean_persistence_first is not None and report.mean_persistence_last is not None:
        lines.append(
            f"   persistence:    {report.mean_persistence_first:.3f} \u2192 "
            f"{report.mean_persistence_last:.3f}"
        )
    # \u03b2_t emergence verdict (ETA paper R3 / R4): training should drive the
    # switch frequency toward sparse, subgoal-aligned firings. We log the
    # delta explicitly so dashboards can pin it.
    if (
        report.switch_frequency_first is not None
        and report.switch_frequency_last is not None
    ):
        delta = report.switch_frequency_last - report.switch_frequency_first
        lines.append(
            f"   \u03b2_t verdict:   switch freq \u0394 = {delta:+.3f} "
            f"({'sparser' if delta < -0.1 else 'noisier' if delta > 0.1 else 'flat'})"
        )
    lines.append(
        f"   loss verdict:  prediction loss {'decreased' if report.loss_decreased() else 'flat-or-up'}"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _collect_default_records() -> tuple[TraceTurnRecord, ...]:
    collector = TraceCollector()
    try:
        for scenario in all_built_in_scenarios():
            collector.collect_scenario(scenario)
        return collector.records
    finally:
        collector.close()


def _lite_view_of(policy: FullLearnedTemporalPolicy):
    """Return a view that ``fit_policy_from_trace_dataset`` will accept.

    ``fit_policy_from_trace_dataset`` is typed for ``LearnedLiteTemporalPolicy``
    but only calls ``fit_from_signals`` \u2014 ``FullLearnedTemporalPolicy``
    implements that with the same signature, so passing the full policy is
    safe. We just adapt the type hint to keep static checkers quiet.
    """
    return policy
