"""Production-grade performance / concurrency / multi-tenant test suite.

These tests cover commercialisation-evidence debt #45 (生产并发实测床).
All test modules are decorated with ``@pytest.mark.perf`` and are
**skipped by default** in CI so PR feedback latency is not affected.

Run locally with::

    pytest tests/perf/ -m perf

See :doc:`docs/specs/perf-baseline.md` for the SLO baselines that
these tests are expected to validate once they ship to ACTIVE.
"""
