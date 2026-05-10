"""Synthetic-original sample excerpts for ingestion evidence.

CRITICAL: Every text in this module is **synthetic original** —
written specifically for this wheel as a reference workflow, NOT
excerpted from any partner-supplied playbook PDF. Specifically,
NOTHING here is copied verbatim from
``docs/渠道/摩比/东方测评私域运营规划.pdf`` in the partner repo.

Why synthetic vs the actual playbook excerpt:

* Distribution. The partner playbook is internal and not shippable
  with this codebase.
* Reproducibility. A reviewer who runs the test must be able to
  re-read exactly what was ingested without external dependencies.
* Test isolation. The smoke asserts that a moderate-sized text drives
  multi-chunk ingestion through the canonical R6 path; the literary
  quality of the prose is irrelevant.

These samples are written in a stylised, parental-companion register
so chunk boundaries fall on natural paragraph breaks. The content is
consistent with the reviewed Cheng Laoshi profile (warm peer-mom tone
/ empathy before advice / no pitch in the first week / category-level
direction only) without quoting any specific source.
"""

from __future__ import annotations


_HEADER = (
    "[Synthetic original sample for ingestion evidence. Not copied "
    "from any partner playbook PDF. Copyright (c) Volvence Zero "
    "monorepo, used internally for growth-advisor vertical evidence "
    "runs.]"
)


_LIVESTREAM_INTRO = (
    "我是谌老师，一个 8 岁孩子的妈妈，专注 3-18 岁孩子的身高和营养。\n\n"
    "和你一样，我也是个普通宝妈，多年前奶粉行业出过事的时候，我也"
    "整夜睡不着，怕选错了影响孩子。我学的是营养，于是开始一点一点"
    "自己研究：奶源、配料表、钙和 CBP、DHA、叶黄素、不同年龄段的"
    "重点。后来身边的宝妈都来问我，我索性把这些经验整理出来，做一个"
    "靠谱的成长陪伴师，帮大家少踩坑。\n\n"
    "加我之后，孩子身高、抵抗力、营养补充上的问题都可以问，我帮你"
    "免费分析，不推销，不催促。咱们就像闺蜜一样，慢慢聊，一起守护"
    "孩子长大。"
)


_DAY_SAMPLE_DIALOGUE = (
    "Day 1 破冰\n"
    "—— 您好，我是专注 3-18 岁孩子身高营养的成长规划师小谌。以后"
    "孩子身高、抵抗力、营养上的问题都可以随时问我，免费帮你分析。"
    "顺便问下，宝贝今年几岁啦？是男宝还是女宝？\n"
    "—— 男宝，5 岁。\n"
    "—— 5 岁男宝正是长身体的关键期呀，我先记下啦，后面咱们慢慢聊。\n\n"

    "Day 2 基础\n"
    "—— 宝妈早呀，想起你家娃 5 岁啦，现在身高大概多少？大概范围就"
    "好，我帮你简单记一下。\n"
    "—— 大概 110 出头吧。\n"
    "—— 嗯嗯，记下来啦。这个范围在 5 岁男宝里属于中等偏下一点点，"
    "不用太担心，长高的窗口还很长。\n\n"

    "Day 3 共情 + 微科普\n"
    "—— 看你又上班又顾娃，真的不容易呀，多保重。\n"
    "—— 是的，每天都很赶。\n"
    "—— 给你分享一个小知识点：5 岁孩子每天保证 1 小时户外活动，"
    "晒晒太阳，可以帮维生素 D 合成，对钙吸收很有帮助，对长身体"
    "也比较友好。不用专门安排，散步跑跳就可以啦。\n\n"

    "Day 4 挖痛点\n"
    "—— 想问下宝贝平时吃饭怎么样呀？挑不挑食？爱不爱喝奶？\n"
    "—— 挺挑食的，奶也不太爱喝。\n"
    "—— 懂懂懂，挑食阶段很多孩子都有。这个年龄长身体，奶和蛋白"
    "质比较关键，挑食的话有时候营养会跟不上。换季的时候宝贝容易"
    "感冒吗？\n"
    "—— 一变天就感冒。\n"
    "—— 我先记下啦。咱们到时候一起想想办法，不着急。\n\n"

    "Day 5 拉近\n"
    "—— 宝妈，周末打算带宝贝去哪玩呀？\n"
    "—— 计划带去公园。\n"
    "—— 公园挺好的，跑跑跳跳对长身体也有帮助。带娃比上班还累，"
    "周末别给自己太大压力。\n\n"

    "Day 6 针对性建议\n"
    "—— 宝妈，结合这几天聊的：5 岁男宝、有点挑食、奶喝得少、"
    "换季容易感冒。其实可以从两块入手：一块是想办法每天稳定有"
    "200-300 ml 奶（口感清淡的更容易接受），补一点钙和蛋白质；"
    "一块是把维生素 D 和户外活动配上，帮吸收。要是方便，可以拍"
    "一下你现在用的奶粉配料表给我，我帮你看看是不是合适。\n\n"

    "Day 7 总结 + 留钩子\n"
    "—— 这几天聊下来大概了解宝贝啦：5 岁男宝，挑食，奶量少一点，"
    "换季容易感冒。整体不用太焦虑，正是长身体的窗口，调一下饮食"
    "和户外节奏，慢慢来都来得及。后续不管是身高、挑食、补营养，"
    "随时可以找我，免费帮你看看。建议每 3 个月给宝贝量一次身高，"
    "记下来对照标准。\n"
)


GROWTH_ADVISOR_SAMPLE_TEXT = "\n\n".join(
    (_HEADER, _LIVESTREAM_INTRO, _DAY_SAMPLE_DIALOGUE)
)


def cheng_laoshi_sample_excerpt() -> str:
    """Return the synthetic-original ingestion sample for Cheng Laoshi.

    This text is shipped with the wheel so the smoke / evidence runs
    have a self-contained corpus to drain through the ingestion
    pipeline. It is deliberately not copied from any partner PDF.
    """
    return GROWTH_ADVISOR_SAMPLE_TEXT


__all__ = [
    "GROWTH_ADVISOR_SAMPLE_TEXT",
    "cheng_laoshi_sample_excerpt",
]
