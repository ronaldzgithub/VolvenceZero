"""Packet 7.4: CLI tool for human reviewers to inspect / approve /
reject pending protocol revision proposals.

This is the operator-facing surface for the L4-queued review path.
It does NOT have direct DB access — it operates on an injected
``RevisionQueue`` instance, which is how production code (a
service / batch job) would invoke it.

Usage (as a function called from a host script):

```python
from volvence_zero.cli.protocol_revision_review import run_review_session
from volvence_zero.protocol_runtime import RevisionQueue

queue = RevisionQueue()
# ... populate queue with submit() ...
run_review_session(queue=queue, reviewer_id="alice@team")
```

Or as a CLI:

```text
$ python -m volvence_zero.cli.protocol_revision_review --reviewer alice
```

The CLI variant uses an in-memory queue with no persistence —
useful for smoke-testing the review flow in dev. Production
deployments wire ``RevisionQueue`` into a backing store.
"""

from __future__ import annotations

import argparse
import sys

from volvence_zero.protocol_runtime import RevisionQueue


def _format_proposal(proposal, decision) -> str:
    return (
        f"\nProposal: {proposal.proposal_id}\n"
        f"  protocol: {proposal.target_protocol_id}\n"
        f"  field:    {proposal.target_field.value}\n"
        f"  entry:    {proposal.target_entry_id}\n"
        f"  kind:     {proposal.change_kind.value}\n"
        f"  required_review_level: {proposal.required_review_level.value}\n"
        f"  decision: {decision.outcome.value} ({decision.rationale})\n"
        f"  evidence: {proposal.evidence.summary}"
    )


def list_pending(queue: RevisionQueue) -> int:
    pending = queue.list_pending()
    if not pending:
        print("(no pending proposals)")
        return 0
    for proposal, decision in pending:
        print(_format_proposal(proposal, decision))
    return len(pending)


def run_review_session(
    *,
    queue: RevisionQueue,
    reviewer_id: str,
    auto_decision: str | None = None,
    input_stream=None,
    output_stream=None,
) -> dict[str, int]:
    """Interactively review pending proposals.

    For each pending proposal the reviewer can:

    * ``a`` — approve
    * ``r`` — reject (with one-line reason)
    * ``s`` — skip (leave in queue)

    Returns a summary dict with counts. ``auto_decision`` (used in
    tests) overrides the prompt and applies the same action to
    every proposal.
    """

    out = output_stream if output_stream is not None else sys.stdout
    inp = input_stream if input_stream is not None else sys.stdin

    summary = {"approved": 0, "rejected": 0, "skipped": 0}
    pending = queue.list_pending()
    for proposal, decision in pending:
        out.write(_format_proposal(proposal, decision) + "\n")
        if auto_decision is None:
            out.write("Decide [a]pprove / [r]eject / [s]kip: ")
            out.flush()
            choice = (inp.readline() or "").strip().lower()
        else:
            choice = auto_decision

        if choice == "a":
            queue.approve(proposal.proposal_id, reviewer_id=reviewer_id)
            summary["approved"] += 1
            out.write("APPROVED\n")
        elif choice == "r":
            reason = ""
            if auto_decision is None:
                out.write("Reason: ")
                out.flush()
                reason = (inp.readline() or "").strip()
            queue.reject(
                proposal.proposal_id,
                reviewer_id=reviewer_id,
                reason=reason or "rejected via CLI",
            )
            summary["rejected"] += 1
            out.write("REJECTED\n")
        else:
            summary["skipped"] += 1
            out.write("SKIPPED\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="volvence_zero.cli.protocol_revision_review",
        description="Inspect and review pending protocol revision proposals.",
    )
    parser.add_argument(
        "--reviewer",
        required=True,
        help="reviewer_id (audit-logged with each approval / rejection)",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="list pending proposals and exit (no review prompt)",
    )
    args = parser.parse_args(argv)

    queue = RevisionQueue()
    if args.list_only:
        list_pending(queue)
        return 0
    summary = run_review_session(queue=queue, reviewer_id=args.reviewer)
    print(
        f"\nReview session complete: "
        f"{summary['approved']} approved, "
        f"{summary['rejected']} rejected, "
        f"{summary['skipped']} skipped"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
