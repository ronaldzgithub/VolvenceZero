"""End-to-end CLI smoke tests for ``python -m lifeform_domain_figure.cli``.

Each test drives :func:`main` directly (no subprocess) using a
``tmp_path``-rooted bundle and audit dir. The test layer exercises
the same code path the operator invokes from the shell so a green
suite proves the wheel + persistence + audit + gate + pool
registration chain is wired correctly.

The BLOCK exit-code test (``test_cmd_blocked_gate_*``) uses an
``EvaluationSnapshot`` with ``contract_integrity = 0.5`` (below the
0.95 OFFLINE-gate floor); :func:`evaluate_gate_reasons` rejects the
proposal, the CLI writes a BLOCK audit row, and ``main(...)``
returns :data:`EXIT_GATE_BLOCK`.
"""

from __future__ import annotations

import json

import pytest

from volvence_zero.substrate import default_persona_lora_pool

from lifeform_domain_figure import (
    list_figure_bundles,
    load_figure_bundle,
)
from lifeform_domain_figure.audit import (
    FigureBakeAction,
    FigureGateDecisionLabel,
    read_audit_records,
)
from lifeform_domain_figure.cli import main as cli_main
from lifeform_domain_figure.cli._commands import (
    EXIT_GATE_BLOCK,
    EXIT_OK,
)


@pytest.fixture
def cli_paths(tmp_path):
    bundle_root = tmp_path / "bundles"
    audit_root = tmp_path / "audit"
    return bundle_root, audit_root


def _common(bundle_root, audit_root) -> list[str]:
    return [
        "--bundle-root", str(bundle_root),
        "--audit-root", str(audit_root),
    ]


def _bake_bundle_einstein(bundle_root, audit_root) -> str:
    rc = cli_main(
        _common(bundle_root, audit_root)
        + ["bake-bundle", "--figure", "einstein"]
    )
    assert rc == EXIT_OK
    manifests = list_figure_bundles(root_dir=bundle_root, figure_id="einstein")
    assert manifests, "bake-bundle did not persist a manifest"
    return manifests[0].bundle_id


def test_cmd_bake_bundle_writes_bundle_and_audit(cli_paths):
    bundle_root, audit_root = cli_paths
    bundle_id = _bake_bundle_einstein(bundle_root, audit_root)

    bundle = load_figure_bundle(
        root_dir=bundle_root,
        bundle_id=bundle_id,
        figure_id="einstein",
    )
    assert bundle.figure_id == "einstein"
    assert bundle.bundle_id == bundle_id

    audits = read_audit_records(root_dir=audit_root)
    assert any(
        rec.action is FigureBakeAction.BAKE_BUNDLE
        and rec.bundle_id == bundle_id
        and rec.gate_decision is FigureGateDecisionLabel.NA
        for rec in audits
    )


def test_cmd_bake_steering_through_gate_writes_new_bundle_id(cli_paths):
    bundle_root, audit_root = cli_paths
    base_id = _bake_bundle_einstein(bundle_root, audit_root)

    rc = cli_main(
        _common(bundle_root, audit_root)
        + [
            "bake-steering",
            "--figure", "einstein",
            "--bundle", base_id,
            "--evaluation-snapshot", "default-clean",
            "--rollback-evidence", f"prev_steering=absent;base={base_id}",
        ]
    )
    assert rc == EXIT_OK

    manifests = list_figure_bundles(root_dir=bundle_root, figure_id="einstein")
    bundle_ids = {m.bundle_id for m in manifests}
    assert base_id in bundle_ids
    new_ids = bundle_ids - {base_id}
    assert len(new_ids) == 1
    new_id = next(iter(new_ids))

    new_bundle = load_figure_bundle(
        root_dir=bundle_root, bundle_id=new_id, figure_id="einstein"
    )
    assert new_bundle.steering is not None
    assert new_bundle.lora is None
    assert new_bundle.bundle_id != base_id

    audits = read_audit_records(root_dir=audit_root)
    assert any(
        rec.action is FigureBakeAction.BAKE_STEERING
        and rec.bundle_id == new_id
        and rec.previous_bundle_id == base_id
        and rec.gate_decision is FigureGateDecisionLabel.ALLOW
        for rec in audits
    )


def test_cmd_bake_lora_through_gate_registers_in_pool_and_writes_audit(
    cli_paths,
):
    bundle_root, audit_root = cli_paths
    base_id = _bake_bundle_einstein(bundle_root, audit_root)

    pool = default_persona_lora_pool()
    if pool.has("einstein"):
        # Tests share the process-wide pool; clear any leftover from
        # a previous test that registered einstein.
        pool.deregister("einstein")

    rc = cli_main(
        _common(bundle_root, audit_root)
        + [
            "bake-lora",
            "--figure", "einstein",
            "--bundle", base_id,
            "--backend", "synthetic",
            "--rank", "8",
            "--evaluation-snapshot", "default-clean",
            "--rollback-evidence", f"prev_lora=absent;base={base_id}",
        ]
    )
    assert rc == EXIT_OK

    assert pool.has("einstein")
    record = pool.lookup("einstein")
    assert record.figure_id == "einstein"

    audits = read_audit_records(root_dir=audit_root)
    lora_audits = [
        a for a in audits if a.action is FigureBakeAction.BAKE_LORA
    ]
    assert len(lora_audits) == 1
    assert lora_audits[0].record_id == record.record_id
    assert lora_audits[0].previous_record_id == "absent"
    assert lora_audits[0].gate_decision is FigureGateDecisionLabel.ALLOW

    pool.deregister("einstein")


def test_cmd_rollback_restores_previous_bundle_in_pool(cli_paths):
    bundle_root, audit_root = cli_paths
    base_id = _bake_bundle_einstein(bundle_root, audit_root)

    pool = default_persona_lora_pool()
    if pool.has("einstein"):
        pool.deregister("einstein")

    # First LoRA bake - establishes a baseline record we can roll back to.
    rc = cli_main(
        _common(bundle_root, audit_root)
        + [
            "bake-lora",
            "--figure", "einstein",
            "--bundle", base_id,
            "--backend", "synthetic",
            "--rank", "4",
            "--evaluation-snapshot", "default-clean",
            "--rollback-evidence", "first-bake",
        ]
    )
    assert rc == EXIT_OK
    audits = read_audit_records(root_dir=audit_root)
    first_lora = [a for a in audits if a.action is FigureBakeAction.BAKE_LORA][-1]
    first_bundle_id = first_lora.bundle_id
    first_record_id = first_lora.record_id

    # Second LoRA bake with different rank - replaces the pool record.
    rc = cli_main(
        _common(bundle_root, audit_root)
        + [
            "bake-lora",
            "--figure", "einstein",
            "--bundle", base_id,
            "--backend", "synthetic",
            "--rank", "8",
            "--evaluation-snapshot", "default-clean",
            "--rollback-evidence", "second-bake",
        ]
    )
    assert rc == EXIT_OK
    second_record_id = pool.lookup("einstein").record_id
    assert second_record_id != first_record_id

    # Roll back to the first bundle.
    rc = cli_main(
        _common(bundle_root, audit_root)
        + [
            "rollback",
            "--figure", "einstein",
            "--to-bundle", first_bundle_id,
            "--rollback-evidence", "test-rollback",
        ]
    )
    assert rc == EXIT_OK

    # The pool now resolves einstein to a record carrying the first
    # bundle's adapter layers (rollback re-registered with the same
    # training_plan_hash as the first bake).
    restored = pool.lookup("einstein")
    assert restored.training_plan_hash == first_lora.bundle_id.split(":")[-1] or (
        # The training plan hash is content-addressed; compare via the
        # bundle's lora artifact directly.
        load_figure_bundle(
            root_dir=bundle_root,
            bundle_id=first_bundle_id,
            figure_id="einstein",
        ).lora.training_plan_hash == restored.training_plan_hash
    )

    audits = read_audit_records(root_dir=audit_root)
    rollback_audits = [
        a for a in audits if a.action is FigureBakeAction.ROLLBACK
    ]
    assert len(rollback_audits) == 1
    assert rollback_audits[0].bundle_id == first_bundle_id
    assert rollback_audits[0].previous_record_id == second_record_id

    pool.deregister("einstein")


def test_cmd_blocked_gate_writes_audit_with_block_reasons_and_exits_2(
    cli_paths,
    tmp_path,
):
    bundle_root, audit_root = cli_paths
    base_id = _bake_bundle_einstein(bundle_root, audit_root)

    # contract_integrity below 0.95 triggers the OFFLINE gate's
    # fail-closed reasoning; capacity_cost / validation_delta defaults
    # would otherwise pass.
    snapshot_payload = {
        "description": "deliberately failing snapshot for #23 BLOCK test",
        "turn_scores": [
            {
                "family": "behavior",
                "metric_name": "contract_integrity",
                "value": 0.50,
                "confidence": 0.95,
                "evidence": "synthetic regression",
            },
            {
                "family": "behavior",
                "metric_name": "rollback_resilience",
                "value": 0.99,
                "confidence": 0.95,
                "evidence": "ok",
            },
            {
                "family": "behavior",
                "metric_name": "fallback_reliance",
                "value": 0.10,
                "confidence": 0.95,
                "evidence": "ok",
            },
        ],
        "session_scores": [],
        "alerts": [],
    }
    snapshot_path = tmp_path / "blocking_snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot_payload), encoding="utf-8")

    rc = cli_main(
        _common(bundle_root, audit_root)
        + [
            "bake-steering",
            "--figure", "einstein",
            "--bundle", base_id,
            "--evaluation-snapshot", str(snapshot_path),
            "--rollback-evidence", "block-test",
        ]
    )
    assert rc == EXIT_GATE_BLOCK

    audits = read_audit_records(root_dir=audit_root)
    block_audits = [
        a for a in audits if a.gate_decision is FigureGateDecisionLabel.BLOCK
    ]
    assert len(block_audits) == 1
    assert block_audits[0].action is FigureBakeAction.BAKE_STEERING
    assert block_audits[0].block_reasons
    assert any(
        "contract_integrity" in reason
        for reason in block_audits[0].block_reasons
    )

    # The persisted bundle set is unchanged: only the original base
    # bundle exists; the BLOCKed steering bundle was never saved.
    manifests = list_figure_bundles(
        root_dir=bundle_root, figure_id="einstein"
    )
    assert len(manifests) == 1
    assert manifests[0].bundle_id == base_id
