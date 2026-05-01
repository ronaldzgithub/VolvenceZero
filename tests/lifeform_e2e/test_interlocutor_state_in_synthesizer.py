"""InterlocutorState reaching the synthesizer (Gap 9 slice 2c).

The contract pinned here:

* ``Lifeform.create_session`` produces a per-session
  ``GroundedResponseSynthesizer`` clone whose
  ``interlocutor_state_provider`` is bound to a closure that
  reads ``LifeformSession.interlocutor_state`` (a 12-axis readout
  computed at access time from the latest kernel snapshots).
* The Brain-level synthesizer is left alone \u2014 no shared mutable
  state across sessions.
* The provider is bound *before* the brain session is built (the
  synthesizer is captured at brain-session construction); the
  back-fill happens via ``_LateBoundSessionHolder`` so the closure
  reads the live ``interlocutor_state`` after the session is
  constructed.
* On a fresh session ``interlocutor_state`` returns a low-confidence
  default; the planner treats it as "no modulation" so back-compat
  for callers that haven't run any turns yet is preserved.

These are the wire-up invariants. Per-axis modulation behaviour is
covered by ``tests/test_prompt_planner_interlocutor_state.py``.
"""

from __future__ import annotations

import pytest

from lifeform_core import Lifeform
from lifeform_domain_emogpt import build_companion_lifeform
from lifeform_expression import GroundedResponseSynthesizer
from lifeform_expression.response_synthesizer import (
    GroundedResponseSynthesizer as _G,
)


@pytest.fixture
def companion_with_grounded_synth() -> Lifeform:
    base = build_companion_lifeform()
    return Lifeform(base.config, response_synthesizer=GroundedResponseSynthesizer())


def test_brain_default_synthesizer_has_no_interlocutor_provider(
    companion_with_grounded_synth: Lifeform,
):
    """The brain-level synthesizer remains unbound. Sessions clone it."""
    life = companion_with_grounded_synth
    brain_synth = life._init_kwargs["response_synthesizer"]
    assert isinstance(brain_synth, _G)
    assert brain_synth.interlocutor_state_provider is None


def test_session_synthesizer_is_clone_with_interlocutor_provider(
    companion_with_grounded_synth: Lifeform,
):
    life = companion_with_grounded_synth
    brain_synth = life._init_kwargs["response_synthesizer"]
    session = life.create_session(session_id="il-clone")
    runner = session.brain_session.runner
    session_synth = runner._response_synthesizer
    assert isinstance(session_synth, _G)
    assert session_synth is not brain_synth, (
        "Lifeform.create_session must clone the brain-level synthesizer; "
        "binding an interlocutor provider on the shared instance would "
        "make all sessions race on each other's user-state readouts."
    )
    assert session_synth.interlocutor_state_provider is not None
    # Same planner instance \u2014 we only re-bind providers per session.
    assert session_synth.planner is brain_synth.planner


def test_interlocutor_provider_returns_session_state(
    companion_with_grounded_synth: Lifeform,
):
    """The closure reads through to ``LifeformSession.interlocutor_state``.

    Calling the provider returns the SAME object that
    ``session.interlocutor_state`` exposes \u2014 confirming the
    late-bound holder pattern correctly forwards.
    """
    life = companion_with_grounded_synth
    session = life.create_session(session_id="il-readback")
    runner = session.brain_session.runner
    provider = runner._response_synthesizer.interlocutor_state_provider
    assert provider is not None

    state_via_provider = provider()
    state_via_property = session.interlocutor_state
    # Two reads close in time return equivalent readouts \u2014 the
    # readout is pure-function so byte-equality is acceptable here.
    assert state_via_provider == state_via_property


def test_two_sessions_have_independent_interlocutor_providers(
    companion_with_grounded_synth: Lifeform,
):
    """Closures captured from one session must not leak into another.

    The factory makes two sessions back-to-back; their providers
    should resolve to different ``LifeformSession`` instances. If
    they didn't, multi-tenant lifeforms would mix user-state
    readouts across users.
    """
    life = companion_with_grounded_synth
    session_a = life.create_session(session_id="il-a")
    session_b = life.create_session(session_id="il-b")
    provider_a = session_a.brain_session.runner._response_synthesizer.interlocutor_state_provider
    provider_b = session_b.brain_session.runner._response_synthesizer.interlocutor_state_provider
    assert provider_a is not provider_b
    # Identity test on the closure: each captures its own holder.
    # We don't expose the holder directly, but we can check the
    # functions' ``__closure__`` cells differ.
    cells_a = {id(c.cell_contents) for c in (provider_a.__closure__ or ())}
    cells_b = {id(c.cell_contents) for c in (provider_b.__closure__ or ())}
    assert cells_a != cells_b


@pytest.mark.asyncio
async def test_synthesize_after_a_turn_sees_a_real_interlocutor_state(
    companion_with_grounded_synth: Lifeform,
):
    """End-to-end smoke: after one turn, the provider yields a non-cold readout."""
    life = companion_with_grounded_synth
    session = life.create_session(session_id="il-turn")
    await session.run_turn("Tell me something kind.")

    state = session.interlocutor_state
    assert state is not None
    # The fresh-session default has confidence very close to zero.
    # After a turn at least some signal should be present, even
    # against the synthetic substrate. We don't pin a specific
    # axis value because that would couple this test to the
    # readout's calibration; we only assert the readout has moved
    # off the cold-start floor.
    assert state.readout_confidence > 0.0
