"""Smoke tests for :mod:`lifeform_domain_figure.audit`.

Validates the audit log invariants the rollback CLI relies on:

* ``write_audit`` writes a unique JSON file per record.
* ``find_previous_audit_for_bundle`` returns the most recent record
  for ``(figure_id, bundle_id)``.
* ``audit_id`` is deterministic over the record payload (same
  payload → same id), which is the property dedup logic relies on.
"""

from __future__ import annotations

import time

from lifeform_domain_figure.audit import (
    FigureBakeAction,
    FigureGateDecisionLabel,
    build_audit_record,
    find_previous_audit_for_bundle,
    read_audit_records,
    write_audit,
)


def test_write_audit_creates_unique_path_per_record(tmp_path):
    record = build_audit_record(
        action=FigureBakeAction.BAKE_BUNDLE,
        figure_id="einstein",
        bundle_id="figure-bundle:einstein:abc123",
        previous_bundle_id="absent",
        gate_decision=FigureGateDecisionLabel.NA,
        corpus_mode="synthetic",
    )
    path = write_audit(record, root_dir=tmp_path)
    assert path.exists()
    assert path.suffix == ".json"
    assert "BAKE_BUNDLE" in path.name
    assert "einstein" in path.name
    assert record.audit_id[:12] in path.name


def test_find_previous_audit_for_bundle_returns_latest_record(tmp_path):
    a = build_audit_record(
        action=FigureBakeAction.BAKE_BUNDLE,
        figure_id="einstein",
        bundle_id="figure-bundle:einstein:abc123",
        gate_decision=FigureGateDecisionLabel.NA,
    )
    write_audit(a, root_dir=tmp_path)
    # Sleep a tick so the second record's created_at_iso is strictly
    # later (second resolution).
    time.sleep(1.1)
    b = build_audit_record(
        action=FigureBakeAction.BAKE_STEERING,
        figure_id="einstein",
        bundle_id="figure-bundle:einstein:abc123",
        previous_bundle_id="figure-bundle:einstein:abc123",
        gate_decision=FigureGateDecisionLabel.ALLOW,
        rollback_evidence="rb",
        validation_delta=0.05,
        capacity_cost=0.20,
        backend_id="steering-cpu-contrastive-v1",
    )
    write_audit(b, root_dir=tmp_path)

    found = find_previous_audit_for_bundle(
        root_dir=tmp_path,
        figure_id="einstein",
        bundle_id="figure-bundle:einstein:abc123",
    )
    assert found is not None
    assert found.audit_id == b.audit_id
    assert found.action is FigureBakeAction.BAKE_STEERING


def test_find_previous_audit_returns_none_when_unmatched(tmp_path):
    record = build_audit_record(
        action=FigureBakeAction.BAKE_BUNDLE,
        figure_id="einstein",
        bundle_id="figure-bundle:einstein:abc",
        gate_decision=FigureGateDecisionLabel.NA,
    )
    write_audit(record, root_dir=tmp_path)
    assert find_previous_audit_for_bundle(
        root_dir=tmp_path,
        figure_id="lu_xun",
        bundle_id="figure-bundle:einstein:abc",
    ) is None
    assert find_previous_audit_for_bundle(
        root_dir=tmp_path,
        figure_id="einstein",
        bundle_id="nonexistent",
    ) is None


def test_audit_record_id_deterministic_over_payload():
    base_kwargs = dict(
        action=FigureBakeAction.BAKE_LORA,
        figure_id="einstein",
        bundle_id="figure-bundle:einstein:zzz",
        previous_bundle_id="figure-bundle:einstein:yyy",
        record_id="persona-lora:einstein:zzz",
        previous_record_id="persona-lora:einstein:yyy",
        gate_decision=FigureGateDecisionLabel.ALLOW,
        block_reasons=(),
        rollback_evidence="rb",
        validation_delta=0.05,
        capacity_cost=0.30,
        corpus_mode="synthetic",
        backend_id="synthetic-v1",
        created_at_iso="2026-05-10T07:00:00Z",
    )
    a = build_audit_record(**base_kwargs)
    b = build_audit_record(**base_kwargs)
    assert a.audit_id == b.audit_id


def test_audit_record_id_changes_when_payload_changes():
    a = build_audit_record(
        action=FigureBakeAction.BAKE_BUNDLE,
        figure_id="einstein",
        bundle_id="figure-bundle:einstein:abc",
        gate_decision=FigureGateDecisionLabel.NA,
        created_at_iso="2026-05-10T07:00:00Z",
    )
    b = build_audit_record(
        action=FigureBakeAction.BAKE_BUNDLE,
        figure_id="einstein",
        bundle_id="figure-bundle:einstein:def",
        gate_decision=FigureGateDecisionLabel.NA,
        created_at_iso="2026-05-10T07:00:00Z",
    )
    assert a.audit_id != b.audit_id


def test_read_audit_records_returns_empty_when_root_missing(tmp_path):
    nonexistent = tmp_path / "no-such-dir"
    records = read_audit_records(root_dir=nonexistent)
    assert records == ()


def test_read_audit_records_orders_by_created_at(tmp_path):
    early = build_audit_record(
        action=FigureBakeAction.BAKE_BUNDLE,
        figure_id="einstein",
        bundle_id="b1",
        gate_decision=FigureGateDecisionLabel.NA,
        created_at_iso="2026-01-01T00:00:00Z",
    )
    late = build_audit_record(
        action=FigureBakeAction.BAKE_STEERING,
        figure_id="einstein",
        bundle_id="b2",
        gate_decision=FigureGateDecisionLabel.ALLOW,
        rollback_evidence="rb",
        validation_delta=0.05,
        capacity_cost=0.20,
        backend_id="x",
        created_at_iso="2026-12-31T23:59:59Z",
    )
    write_audit(late, root_dir=tmp_path)
    write_audit(early, root_dir=tmp_path)

    records = read_audit_records(root_dir=tmp_path)
    assert len(records) == 2
    assert records[0].created_at_iso < records[1].created_at_iso
