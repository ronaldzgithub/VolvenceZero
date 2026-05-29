"""Cultivation seed + curriculum value objects and text builders.

These are *data* the engine feeds through the kernel's canonical intake
surfaces, NOT control-plane prompt hacks. The seed charter and the
per-topic study briefs are short, templated strings centralised here
(per ``llm-prompt-centralization``); the kernel's cognition forms the
expert's identity and school from the *experience* of ingesting and
studying this material, rather than from a hardcoded behavioural rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CultivationSeed:
    """The "rough persona" an operator hands to a new cultivation.

    Deliberately under-specified: the operator gives a sketch (who the
    expert roughly is, what domain, what it must NOT become) and the
    autonomous loop grows the full cognitive system from researched
    material. ``single_school_objective`` is the convergence intent —
    expressed as a cultivation goal, never as a runtime prompt rule.
    """

    display_name: str
    domain: str
    role_archetype: str
    focus: str = ""
    value_boundaries: tuple[str, ...] = ()
    single_school_objective: str = (
        "形成一个内部自洽的流派，而不是不同流派的混合体"
    )

    def to_json(self) -> dict[str, object]:
        return {
            "display_name": self.display_name,
            "domain": self.domain,
            "role_archetype": self.role_archetype,
            "focus": self.focus,
            "value_boundaries": list(self.value_boundaries),
            "single_school_objective": self.single_school_objective,
        }

    @classmethod
    def from_json(cls, raw: object) -> "CultivationSeed":
        if not isinstance(raw, dict):
            raise ValueError("CultivationSeed.from_json requires an object")
        display_name = str(raw.get("display_name", "")).strip()
        domain = str(raw.get("domain", "")).strip()
        role_archetype = str(raw.get("role_archetype", "")).strip()
        if not display_name or not domain or not role_archetype:
            raise ValueError(
                "CultivationSeed requires non-empty display_name, domain, "
                "and role_archetype"
            )
        boundaries_raw = raw.get("value_boundaries") or ()
        if isinstance(boundaries_raw, str):
            raise ValueError("value_boundaries must be a list, not a string")
        objective = str(
            raw.get("single_school_objective", "").strip()
        ) or "形成一个内部自洽的流派，而不是不同流派的混合体"
        return cls(
            display_name=display_name,
            domain=domain,
            role_archetype=role_archetype,
            focus=str(raw.get("focus", "")).strip(),
            value_boundaries=tuple(str(b) for b in boundaries_raw),
            single_school_objective=objective,
        )


@dataclass(frozen=True)
class CultivationCurriculum:
    """The autonomous study schedule for a cultivation.

    ``topics`` are the research themes the loop will work through (one
    research+study cycle each, cycling back to the start when exhausted).
    ``coherence_threshold`` is the school-concentration score above which
    the cultivation is considered to have converged onto one school.
    """

    topics: tuple[str, ...]
    source_hints: tuple[str, ...] = ()
    target_cycles: int = 24
    docs_per_topic: int = 3
    reflect_every: int = 4
    coherence_threshold: float = 0.7
    min_cycles_for_convergence: int = 8

    def to_json(self) -> dict[str, object]:
        return {
            "topics": list(self.topics),
            "source_hints": list(self.source_hints),
            "target_cycles": self.target_cycles,
            "docs_per_topic": self.docs_per_topic,
            "reflect_every": self.reflect_every,
            "coherence_threshold": self.coherence_threshold,
            "min_cycles_for_convergence": self.min_cycles_for_convergence,
        }

    @classmethod
    def from_json(cls, raw: object) -> "CultivationCurriculum":
        if not isinstance(raw, dict):
            raise ValueError("CultivationCurriculum.from_json requires an object")
        topics_raw = raw.get("topics") or ()
        if isinstance(topics_raw, str):
            raise ValueError("topics must be a list, not a string")
        topics = tuple(str(t).strip() for t in topics_raw if str(t).strip())
        if not topics:
            raise ValueError("CultivationCurriculum requires at least one topic")
        source_hints_raw = raw.get("source_hints") or ()
        if isinstance(source_hints_raw, str):
            raise ValueError("source_hints must be a list, not a string")
        return cls(
            topics=topics,
            source_hints=tuple(str(s) for s in source_hints_raw),
            target_cycles=int(raw.get("target_cycles", 24) or 24),
            docs_per_topic=int(raw.get("docs_per_topic", 3) or 3),
            reflect_every=int(raw.get("reflect_every", 4) or 4),
            coherence_threshold=float(raw.get("coherence_threshold", 0.7) or 0.7),
            min_cycles_for_convergence=int(
                raw.get("min_cycles_for_convergence", 8) or 8
            ),
        )

    def topic_for_cycle(self, cycle_index: int) -> str:
        return self.topics[cycle_index % len(self.topics)]

    def research_query(self, cycle_index: int) -> str:
        topic = self.topic_for_cycle(cycle_index)
        if self.source_hints:
            hint = self.source_hints[cycle_index % len(self.source_hints)]
            return f"{topic} {hint}".strip()
        return topic


# --- Centralised short text templates (per llm-prompt-centralization) ---
# These are <=5 line directives fed as ingestion/apprentice material,
# not behavioural control logic. They carry the cultivation *intent* as
# experience; the kernel decides how to integrate it.

_CHARTER_TEMPLATE = (
    "养成宪章 / Cultivation Charter\n"
    "身份：{display_name}（{role_archetype}）\n"
    "领域：{domain}\n"
    "关注：{focus}\n"
    "边界：{boundaries}\n"
    "养成目标：{objective}\n"
    "学习方式：自主搜集并比较该领域内的合理理论，评估其证据与适用条件，"
    "逐步整合成一个内部自洽、可解释的认知体系。"
)

_STUDY_BRIEF_TEMPLATE = (
    "学习主题：{topic}\n"
    "请基于刚刚摄入的材料，评估比较其中的不同观点，"
    "保留与你正在形成的体系一致的部分，明确指出冲突之处并给出取舍依据，"
    "把结论整合进你自己的认知体系（{domain}）。"
)


def build_charter_text(seed: CultivationSeed) -> str:
    """Render the seed into a one-shot charter corpus chunk."""

    boundaries = "、".join(seed.value_boundaries) if seed.value_boundaries else "无"
    return _CHARTER_TEMPLATE.format(
        display_name=seed.display_name,
        role_archetype=seed.role_archetype,
        domain=seed.domain,
        focus=seed.focus or "（未指定）",
        boundaries=boundaries,
        objective=seed.single_school_objective,
    )


def build_study_brief(*, topic: str, domain: str) -> str:
    """Render a short apprentice-study directive for one topic."""

    return _STUDY_BRIEF_TEMPLATE.format(topic=topic, domain=domain)


__all__ = [
    "CultivationCurriculum",
    "CultivationSeed",
    "build_charter_text",
    "build_study_brief",
]
