from __future__ import annotations

from volvence_zero.learned_update import LearnedUpdateRule, LearnedUpdateRuleState


def _suppressed_state(*, feature_dim: int, hidden_dim: int) -> LearnedUpdateRuleState:
    return LearnedUpdateRuleState(
        rule_id="suppressed-rule",
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        update_count=0,
        last_feature_norm=0.0,
        last_improvement=0.0,
        last_guard_reason="",
        input_projection=tuple(tuple(0.0 for _ in range(feature_dim)) for _ in range(hidden_dim)),
        hidden_bias=tuple(0.0 for _ in range(hidden_dim)),
        output_projection=tuple(tuple(0.0 for _ in range(hidden_dim)) for _ in range(7)),
        output_bias=(-12.0, -12.0, -12.0, -12.0, -12.0, 0.0, -12.0),
        last_decisions=(),
        description="suppressed updater for attribution tests",
    )


def test_learned_update_rule_guard_applies_on_feature_overload() -> None:
    rule = LearnedUpdateRule(rule_id="guard-test", feature_dim=12, hidden_dim=8)

    decision = rule.decide(target_id="temporal-encoder", features=tuple(1.0 for _ in range(12)))
    state = rule.export_state()

    assert decision.guard_applied is True
    assert decision.guard_reason == "feature-overload"
    assert decision.write_gate < 0.8
    assert decision.step_scale < 0.8
    assert state.last_guard_reason == "feature-overload"
    assert state.last_decisions[-1].target_id == "temporal-encoder"


def test_learned_update_rule_restore_can_hold_write_gate_near_closed() -> None:
    rule = LearnedUpdateRule(rule_id="closed-gate", feature_dim=6, hidden_dim=4)
    rule.restore_state(_suppressed_state(feature_dim=6, hidden_dim=4))

    decision = rule.decide(target_id="cms-online", features=(0.2, 0.1, 0.0, 0.3, 0.2, 0.1))

    assert decision.write_gate < 0.01
    assert decision.step_scale < 0.01
    assert decision.slow_mix < 0.01
    assert decision.reset_mix < 0.01
    assert decision.confidence < 0.01


def test_learned_update_rule_export_restore_preserves_decisions_and_learning_continuity() -> None:
    rule = LearnedUpdateRule(rule_id="continuity", feature_dim=5, hidden_dim=4)
    first = rule.decide(target_id="temporal-switch", features=(0.1, 0.4, 0.2, 0.3, 0.5))
    rule.learn(
        features=(0.1, 0.4, 0.2, 0.3, 0.5),
        decision=first,
        improvement=0.25,
        stability=0.75,
    )
    saved = rule.export_state()

    restored = LearnedUpdateRule(rule_id="continuity", feature_dim=5, hidden_dim=4)
    restored.restore_state(saved)
    second = restored.decide(target_id="temporal-switch", features=(0.1, 0.4, 0.2, 0.3, 0.5))
    restored.learn(
        features=(0.1, 0.4, 0.2, 0.3, 0.5),
        decision=second,
        improvement=0.10,
        stability=0.60,
    )
    restored_state = restored.export_state()

    assert saved.last_decisions
    assert saved.last_decisions[-1].target_id == "temporal-switch"
    assert restored_state.update_count == saved.update_count + 1
    assert restored_state.last_decisions[-1].target_id == "temporal-switch"
    assert restored_state.last_feature_norm > 0.0
    assert restored_state.input_projection != tuple(tuple(0.0 for _ in range(5)) for _ in range(4))
