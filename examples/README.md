# Examples

End-to-end demos that exercise the VolvenceZero application stack.

These files live **outside** `packages/` on purpose:

* They are not wheels; nothing here is pip-installable.
* They are not part of the test suite proper; they can be run as
  scripts (`python examples/<name>.py`) and are also exercised by
  a thin integration test under `tests/lifeform_e2e/` to prevent
  rot.
* They only import from the public `lifeform-*` wheel APIs.
  **No `volvence_zero.*` (kernel) imports are allowed.** That
  invariant keeps the application / kernel boundary clean
  (`SPLIT.md`) and makes each demo portable across kernel
  refactors.

## Demos

### `coding_end_to_end_demo.py`

Puts three recent slices of infrastructure into one live pipeline:

1. **Coding vertical** (Gap 1 slices 2b + 2c): a `build_coding_lifeform()`
   session wired to the 5 filesystem affordances
   (`read_file` / `list_dir` / `grep` / `write_file` / `run_test`).
2. **Thinking adapter** (Gap 4 slice 2c): mid-frequency reflection
   in SHADOW mode, automatically submitting world- and self-lane
   tasks after every turn and collecting them at the start of the
   next turn.
3. **Affordance scorer** (Gap 1 slice 3): per-turn scoring of the
   5 affordances from the current `regime` / `dual_track` session
   snapshots; shows which tool the metacontroller-style readout
   would surface as the selected candidate.
4. (Bonus) **InterlocutorState readout** (Gap 9 slice 1): the
   12-axis view of the "who am I talking to" readout, emitted
   at key turns so the demo prints a human-readable signature
   of engagement / resistance / openness over the course of
   the session.

Run it:

```
python examples/coding_end_to_end_demo.py
```

The script creates a disposable sandbox directory, writes a tiny
Python project into it, drives the lifeform through a scripted
multi-turn session that exercises every affordance tier, and
prints a structured audit trail at the end. No permanent files
are written outside the temp dir; the sandbox is removed on exit.

## Adding a new demo

1. Create `examples/<name>.py`
2. Only import from `lifeform-*` wheels (+ stdlib)
3. Mirror the "run through a sandbox tmp dir" discipline used
   by `coding_end_to_end_demo.py` so the demo is idempotent
4. Add a thin integration test under `tests/lifeform_e2e/` that
   runs the demo's `main()` once and asserts it completes without
   raising
5. Link it from this README
