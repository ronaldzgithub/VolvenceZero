"""Volvence Zero runtime package."""

from importlib import import_module

_EXPORT_MAP = {
    "Brain": ("volvence_zero.brain", "Brain"),
    "BrainConfig": ("volvence_zero.brain", "BrainConfig"),
    "BrainSession": ("volvence_zero.brain", "BrainSession"),
    "AgentResponse": ("volvence_zero.agent", "AgentResponse"),
    "AgentSessionRunner": ("volvence_zero.agent", "AgentSessionRunner"),
    "AgentTurnResult": ("volvence_zero.agent", "AgentTurnResult"),
    "CMSMemoryCore": ("volvence_zero.memory", "CMSMemoryCore"),
    "ETANLJointLoop": ("volvence_zero.joint_loop", "ETANLJointLoop"),
    "InternalRLSandbox": ("volvence_zero.internal_rl", "InternalRLSandbox"),
    "JointCycleReport": ("volvence_zero.joint_loop", "JointCycleReport"),
    "ResponseSynthesizer": ("volvence_zero.agent", "ResponseSynthesizer"),
    "build_arg_parser": ("volvence_zero.agent", "build_arg_parser"),
    "default_active_runner": ("volvence_zero.agent", "default_active_runner"),
    "derive_abstract_action_credit": ("volvence_zero.internal_rl", "derive_abstract_action_credit"),
    "main": ("volvence_zero.agent", "main"),
    "render_turn_result": ("volvence_zero.agent", "render_turn_result"),
    "run_repl": ("volvence_zero.agent", "run_repl"),
}

__all__ = [
    "Brain",
    "BrainConfig",
    "BrainSession",
    "AgentResponse",
    "AgentSessionRunner",
    "AgentTurnResult",
    "CMSMemoryCore",
    "ETANLJointLoop",
    "InternalRLSandbox",
    "JointCycleReport",
    "ResponseSynthesizer",
    "build_arg_parser",
    "default_active_runner",
    "derive_abstract_action_credit",
    "main",
    "render_turn_result",
    "run_repl",
]


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORT_MAP[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
