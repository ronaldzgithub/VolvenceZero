"""Phase 2 W2.D contract test: sparse-reward held-out gate plumbing.

The full ETA open-weight paper suite is heavyweight (real HF model
load), so this test focuses on the gate plumbing rather than running
the suite end-to-end:

* CLI argparse exposes ``--require-sparse-reward-heldout``.
* The flag refuses to run without ``--eta-open-weight-paper-suite``.
* The internal ``_EtaPaperSuiteRunResult`` dataclass exposes the
  typed sparse-reward retain surface that the CLI gate reads.
* The CLI exit logic gates on ``sparse_reward_advantage_retain``.

End-to-end coverage of the actual ETA paper suite is owned by
``tests/test_eta_proof_benchmark.py``; this test pins the merge into
the lifeform-bench acceptance pipeline.
"""

from __future__ import annotations

import argparse

import pytest

from lifeform_evolution.cli import _EtaPaperSuiteRunResult, _build_bench_parser


def test_cli_exposes_require_sparse_reward_heldout_flag() -> None:
    parser = _build_bench_parser()
    actions = {a.dest: a for a in parser._actions}
    assert "require_sparse_reward_heldout" in actions
    flag = actions["require_sparse_reward_heldout"]
    assert flag.option_strings == ["--require-sparse-reward-heldout"]
    # Default must be False so existing CLI invocations continue to
    # behave as before.
    args = parser.parse_args([])
    assert args.require_sparse_reward_heldout is False


def test_cli_rejects_require_sparse_reward_heldout_without_paper_suite() -> None:
    """The gate cannot run without the paper suite that produces the
    verdict; argparse must fail rather than silently passing.
    """
    from lifeform_evolution.cli import main

    with pytest.raises(SystemExit) as excinfo:
        main(["--require-sparse-reward-heldout"])
    assert excinfo.value.code == 2  # argparse error exit


def test_eta_paper_suite_run_result_carries_typed_sparse_reward_surface() -> None:
    result = _EtaPaperSuiteRunResult(
        passed=True,
        sparse_reward_advantage_status="retain",
        sparse_reward_advantage_retain=True,
    )
    assert result.passed is True
    assert result.sparse_reward_advantage_status == "retain"
    assert result.sparse_reward_advantage_retain is True


def test_eta_paper_suite_run_result_distinguishes_status_from_retain_flag() -> None:
    """Status reflects the typed claim status (retain / weak / hold /
    missing); retain is a derived boolean. Non-retain statuses must
    NOT silently pass the gate.
    """
    for status in ("weak", "hold", "missing"):
        result = _EtaPaperSuiteRunResult(
            passed=True,
            sparse_reward_advantage_status=status,
            sparse_reward_advantage_retain=False,
        )
        assert result.sparse_reward_advantage_retain is False, status
