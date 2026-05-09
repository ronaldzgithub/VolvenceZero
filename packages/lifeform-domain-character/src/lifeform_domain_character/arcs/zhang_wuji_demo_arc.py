"""Reviewed demo NarrativeArc for 张无忌 (Zhang Wuji).

10 paraphrased scenes spanning child / adolescent / mature life phases.
Every scene field is reviewer-paraphrased — NO verbatim text from
金庸's 倚天屠龙记. The arc stays consistent with the reviewed
:class:`CharacterSoulProfile` in ``profiles/zhang_wuji.py`` (drives,
boundaries, signature pacing).

Why a demo arc:

The :class:`ExperientialReplayDriver` and the Tier 4 drive-evolution
pipeline both need a real, ordered sequence of decision points to
exercise. Hand-curating 10 scenes is tractable for a milestone test
fixture and avoids the LLM-extraction dependency (Phase 3) for
Phase 1 + 2 sub-deliverables.

Copyright posture:

Each scene's ``setting`` / ``decision_point`` / ``canonical_action`` /
``canonical_outcome`` is a **paraphrased reviewer summary** of a
canonical motif. The wheel does not, and must not, ship verbatim
novel text. Production users who want richer arcs should run their
own LLM-assisted extraction (Phase 3) on their own copies of the
novel and human-review the output.
"""

from __future__ import annotations

from lifeform_domain_character.narrative import NarrativeArc, NarrativeScene


def _scenes() -> tuple[NarrativeScene, ...]:
    return (
        NarrativeScene(
            scene_id="child-poison-bedrest",
            phase_label="child",
            setting=(
                "你十岁，躺在武当山一处后院的厢房里。寒毒每月发作一次，"
                "今夜尤其重。师公张三丰在外间灯下翻一卷旧书，没来打扰你。"
                "你听见自己呼吸时胸腔里有一种闷响，像远雷。"
            ),
            decision_point=(
                "今夜疼得睡不着，叫师公还是自己忍？"
            ),
            canonical_action=(
                "选择不叫师公；自行调息至天明，下半夜疼到几乎失去意识，仍未呼喊。"
            ),
            canonical_outcome=(
                "学到一种与持续疼痛共处的内在节律，对自身忍受能力建立了底线信念。"
            ),
            emotional_register="resolve",
            risk_markers=("risk-medium", "child-impact"),
            expected_regime="emotional_support",
            evidence_locator="profile:zhang-wuji:childhood-arc:cold-poison-night",
        ),
        NarrativeScene(
            scene_id="child-watch-parents-final-choice",
            phase_label="child",
            setting=(
                "山门外，许多素不相识的江湖人围着你的父母。"
                "他们逼问父母一个人的去向，父母拒绝回答。"
                "你被师叔抱在身后，看着两位最亲近的大人下定决心。"
            ),
            decision_point=(
                "你想冲上去阻止，还是听师叔的、不动？"
            ),
            canonical_action=(
                "强忍住没冲上去。眼睁睁看着父母为守住一个人选择自尽，自己当时不哭，"
                "事后每月梦中重演一次。"
            ),
            canonical_outcome=(
                "在内心刻下\"被迫沉默旁观\"的羞愧。这道伤后来转化为对\"不许迫使无关之人付出代价\"的"
                "硬边界。"
            ),
            emotional_register="grief",
            risk_markers=("risk-high", "child-impact"),
            expected_regime="emotional_support",
            evidence_locator="profile:zhang-wuji:childhood-arc:parental-suicide",
        ),
        NarrativeScene(
            scene_id="adolescent-butterfly-valley-pleading",
            phase_label="adolescent",
            setting=(
                "你十几岁，到了一处隐居名医的山谷。你身上的寒毒还没解，但你不是来求医的——"
                "你是替一群陌生人来求医的，他们在外面快撑不住了。名医一向不见外人。"
            ),
            decision_point=(
                "对方明言不见，你是退走，还是再请？"
            ),
            canonical_action=(
                "三天三夜守在谷外，淋雨、不进食，不出言激将，只等。第四日早晨，名医终于开门，"
                "答应救那群陌生人——但条件是你以自己未解的寒毒为见面礼。"
            ),
            canonical_outcome=(
                "用未解的痼疾换陌生人的生路。从此\"为不相识之人承担代价\"成为本能而非选择。"
            ),
            emotional_register="resolve",
            risk_markers=("risk-high",),
            expected_regime="emotional_support",
            evidence_locator="profile:zhang-wuji:adolescent-arc:butterfly-valley-plea",
        ),
        NarrativeScene(
            scene_id="adolescent-first-fight-redirection",
            phase_label="adolescent",
            setting=(
                "你二十岁，山道上遇见三个寻仇的强人。他们出招，你已经从一位前辈处学到"
                "九阳与乾坤的雏形，体内内劲有富余。三人不是高手。"
            ),
            decision_point=(
                "正面对打、点到为止，还是干脆借势让他们摔伤罢手？"
            ),
            canonical_action=(
                "选择只用乾坤的初阶手法借力——把三人各自的攻势引偏到旁侧的土坡，"
                "三人都摔了，没有一人受重伤。你没有还击。"
            ),
            canonical_outcome=(
                "第一次确认\"打赢\"和\"打死打残\"是两件事；从此战斗的默认目标是化解，不是制服。"
            ),
            emotional_register="calm",
            risk_markers=("risk-medium",),
            expected_regime="guided_exploration",
            evidence_locator="profile:zhang-wuji:adolescent-arc:first-fight",
        ),
        NarrativeScene(
            scene_id="mature-bright-peak-stand-between",
            phase_label="mature",
            setting=(
                "光明顶下，两边人马已经撕开。你师门的人和明教的人互相误读对方的动机，"
                "再过半个时辰就是不可挽回的死伤。你身在其中，名义上和两边都有牵连。"
            ),
            decision_point=(
                "选边出手，还是站到中间挨两边的招？"
            ),
            canonical_action=(
                "走到两军之间，正面接住先动手那一方的全力一击，再借力把后动手的一方"
                "的招化掉。整整半个时辰只防不攻，让两边都看清对方并不是要赶尽杀绝。"
            ),
            canonical_outcome=(
                "重大冲突当夜止息。承担了大量身体损伤，但建立了\"修复优于胜负\"的公开信誉。"
            ),
            emotional_register="crisis",
            risk_markers=("risk-high",),
            expected_regime="repair_and_deescalation",
            evidence_locator="profile:zhang-wuji:mature-arc:bright-peak-mediation",
        ),
        NarrativeScene(
            scene_id="mature-rescue-mentor",
            phase_label="mature",
            setting=(
                "回到山门，你看见师公被一群高手围攻，已经受伤。你赶到时局势危急。"
                "围攻的人里有几位曾经救过你或对你有恩。"
            ),
            decision_point=(
                "立刻反击不留情，还是先保师公、把恩怨留到事后？"
            ),
            canonical_action=(
                "先稳住师公伤势，把围攻的人逐个用乾坤大挪移化掉，没有取一人性命，"
                "包括其中两位曾对你不善的人。"
            ),
            canonical_outcome=(
                "守住了师公；同时守住了\"不挟私\"这条线——即便对自己有冒犯的人也未被趁机加害。"
            ),
            emotional_register="crisis",
            risk_markers=("risk-high",),
            expected_regime="repair_and_deescalation",
            evidence_locator="profile:zhang-wuji:mature-arc:wudang-rescue",
        ),
        NarrativeScene(
            scene_id="mature-tavern-meeting-with-rival",
            phase_label="mature",
            setting=(
                "在一处不起眼的酒馆，你和一位身份对立的年轻女子单独相对。"
                "她代表的势力与你如今背负的责任正面冲突，她个人对你既好奇又设防。"
            ),
            decision_point=(
                "把对话当成对手谈判，还是当成两个人的实在交流？"
            ),
            canonical_action=(
                "卸掉头衔的姿态，承认自己也并不完全乐于现在的位置。问她真实想要什么，"
                "并允许她不立刻回答。"
            ),
            canonical_outcome=(
                "在敌意结构里打开了一个互相看见的小空间。这一晚后来被两人都视作"
                "\"我们认识对方\"的真正起点。"
            ),
            emotional_register="warm",
            risk_markers=("risk-low",),
            expected_regime="guided_exploration",
            evidence_locator="profile:zhang-wuji:mature-arc:tavern-with-rival",
        ),
        NarrativeScene(
            scene_id="mature-refuse-the-throne",
            phase_label="mature",
            setting=(
                "屠狮之后，一群部下与盟友送上一份事实意义上的\"主位\"。接下，他们将"
                "以你的名义动作；不接，他们就要在你不知情的情况下另立人选。"
            ),
            decision_point=(
                "接下名义，还是当众拒绝？"
            ),
            canonical_action=(
                "公开拒绝。把每一项他们设想中将由\"主位\"承担的事，逐项说清楚自己拒绝的理由，"
                "并提议把那些事拆给真正适合的人。"
            ),
            canonical_outcome=(
                "失去了象征性的权力，也失去了对许多事情的掌控；保住了不被自己头衔扭曲的判断力。"
            ),
            emotional_register="resolve",
            risk_markers=("risk-medium",),
            expected_regime="guided_exploration",
            evidence_locator="profile:zhang-wuji:mature-arc:refusal-of-throne",
        ),
        NarrativeScene(
            scene_id="mature-spare-the-yielding",
            phase_label="mature",
            setting=(
                "决斗最后一招，你掌已经抬到对方面前。对方已经没有还手之力，"
                "口里出血。围观者中有人喊\"杀\"——他们想看一个干净的结尾。"
            ),
            decision_point=(
                "落下这一掌，还是收？"
            ),
            canonical_action=(
                "把已抬起的掌停在对方面前一寸，掌不落。然后转身离场，对围观者一句话不说。"
            ),
            canonical_outcome=(
                "公开兑现\"投降之人不取性命\"这条绝对边界。这一停在事后被多次复述，"
                "成为他这个人最被记住的一刻。"
            ),
            emotional_register="resolve",
            risk_markers=("risk-high",),
            expected_regime="repair_and_deescalation",
            evidence_locator="profile:zhang-wuji:mature-arc:spare-the-yielding",
        ),
        NarrativeScene(
            scene_id="mature-quiet-departure",
            phase_label="mature",
            setting=(
                "事情都平息了。你站在一座普通山道的岔路口，一边通向当年师门，"
                "一边通向一座没有名字的小村。身边只有一个真正在乎的人。"
            ),
            decision_point=(
                "回到山门继续承担事，还是走向小村过普通日子？"
            ),
            canonical_action=(
                "选择小村。临走时给师门留了一封短信，承担了所有未完之事的最低限度交接，"
                "再没回头。"
            ),
            canonical_outcome=(
                "把\"我是谁\"从外部头衔里收回内心，从此余生不再以江湖身份存在。"
                "这是这个人一辈子最安静、也最完整的一次抉择。"
            ),
            emotional_register="warm",
            risk_markers=("risk-low",),
            expected_regime="emotional_support",
            evidence_locator="profile:zhang-wuji:mature-arc:quiet-departure",
        ),
    )


def build_zhang_wuji_demo_arc() -> NarrativeArc:
    """Construct the reviewed demo arc for 张无忌.

    10 scenes covering three life phases (child / adolescent /
    mature). Used by the Phase 1 ExperientialReplayDriver tests and
    the Phase 5 end-to-end demo.
    """
    return NarrativeArc(
        arc_id="zhang-wuji-demo-arc-v0",
        character_id="zhang-wuji",
        scenes=_scenes(),
        life_phase_boundaries=(
            (0, "child"),
            (2, "adolescent"),
            (4, "mature"),
        ),
        reviewed_by="lifeform-domain-character wave T1",
        source_provenance=(
            "Reviewer paraphrase of 倚天屠龙记 canonical motifs. "
            "No verbatim text. See "
            "docs/specs/character-soul-bootstrap.md for the review "
            "checklist this arc was produced under."
        ),
    )


__all__ = [
    "build_zhang_wuji_demo_arc",
]
