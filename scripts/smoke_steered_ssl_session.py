"""Integration smoke: steered-action SSL session through the real frozen Qwen.

Builds the ETA open-weight runtime, the substrate steered action scorer, one
expert-action observation bundle, and runs two Eq.3 train_batch updates plus
an evaluate_batch readout. Verifies gradient reaches the controller and that
distortion is a genuine through-model action NLL.
"""

from __future__ import annotations

import time

from volvence_zero.agent.eta_proof_benchmark import (
    ETAOpenWeightRuntimeConfig,
    _build_eta_open_weight_runtime,
    _validate_eta_open_weight_runtime,
    build_default_eta_proof_environment,
    default_eta_proof_cases,
)
from volvence_zero.agent.eta_segment_credit_evidence import (
    _expert_action_observation_bundle,
)
from volvence_zero.substrate import SteeredActionOption
from volvence_zero.temporal import (
    MetacontrollerParameterStore,
    build_training_trace_from_substrate_snapshots,
)
from volvence_zero.temporal.torch_store_ssl import StoreSSLTrainingSession


def main() -> None:
    config = ETAOpenWeightRuntimeConfig(device="mps")
    runtime = _build_eta_open_weight_runtime(config)
    _validate_eta_open_weight_runtime(runtime=runtime, config=config)

    environment = build_default_eta_proof_environment()
    target_ids = {t.target_id for t in environment.transitions}
    options = tuple(
        SteeredActionOption(
            action_id=f"move:{location.location_id}",
            surface_text=location.location_id,
        )
        for location in environment.locations
        if location.location_id in target_ids
    )
    scorer = runtime.build_steered_action_scorer(action_options=options)
    print(
        f"scorer: injection_layer={scorer.injection_layer_index} "
        f"hidden={scorer.hidden_size} norm_cap={scorer.control_norm_cap:.2f} "
        f"probe_norm={scorer.probe_hidden_norm:.2f}"
    )

    cases = [c for c in default_eta_proof_cases() if c.split == "train"][:2]
    traces = []
    for case in cases:
        snapshots, texts, targets = _expert_action_observation_bundle(
            case, open_weight_runtime=runtime
        )
        traces.append(
            build_training_trace_from_substrate_snapshots(
                trace_id=f"smoke:{case.case_id}",
                source_text=case.source_text,
                snapshots=snapshots,
                expert_action_targets=targets,
                observation_texts=texts,
            )
        )
        print(f"case={case.case_id} steps={len(texts)}")

    baseline = scorer.baseline_action_nll(
        source_texts=tuple(
            step.observation_text for step in traces[0].steps
        ),
        action_indices=tuple(
            scorer.action_index(step.expert_action_target.action_id)
            for step in traces[0].steps
        ),
    )
    print(f"baseline NLL trace0: {[round(v, 3) for v in baseline]}")

    store = MetacontrollerParameterStore(n_z=16, initialization_seed=42)
    session = StoreSSLTrainingSession(
        n_z=16,
        alpha=0.1,
        proposal_prediction_weight=0.0,
        action_scorer=scorer,
        reparam_seed=1234,
    )
    for update in range(2):
        start = time.perf_counter()
        report = session.train_batch(
            store=store,
            traces=tuple(traces),
            batch_id=f"smoke:{update}",
            switch_threshold=0.55,
            write_back=False,
        )
        elapsed = time.perf_counter() - start
        print(
            f"update={update} pred={report.prediction_loss:.4f} "
            f"kl={report.kl_loss:.4f} total={report.total_loss:.4f} "
            f"grad_norm={report.grad_norm:.4f} "
            f"changed={report.parameters_changed} "
            f"switch_p={report.mean_switch_probability:.4f} "
            f"supervision={report.supervision_target} "
            f"distortion_target={report.distortion_target} "
            f"elapsed={elapsed:.1f}s"
        )
        if report.grad_norm <= 0.0:
            raise RuntimeError("no gradient reached the controller")
        if report.supervision_target != "steered-action-nll":
            raise RuntimeError("wrong supervision target")

    evaluation = session.evaluate_batch(
        store=store,
        traces=tuple(traces),
        batch_id="smoke:eval",
        switch_threshold=0.55,
    )
    print(
        f"eval: distortion={evaluation.distortion:.4f} "
        f"kl_rate={evaluation.kl_rate:.4f} "
        f"switch_p={evaluation.mean_switch_probability:.4f} "
        f"hard_freq={evaluation.hard_switch_frequency:.4f} "
        f"boundary_f1={evaluation.boundary_f1:.4f}"
    )
    print("smoke ok")


if __name__ == "__main__":
    main()
