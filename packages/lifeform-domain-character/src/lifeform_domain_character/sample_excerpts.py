"""Synthetic original character excerpts for Tier 2 ingestion evidence.

CRITICAL: Every text in this module is **synthetic original** — written
specifically for this wheel as fan-style derivative work, NOT excerpted
from any published source. Specifically, NOTHING here is copied from
金庸 (Jin Yong) / the 倚天屠龙记 novel itself.

Why synthetic vs the actual novel:

* Copyright. The novel is under active copyright; we cannot ship it.
* Reproducibility. A reviewer who runs the test must be able to re-read
  exactly what was ingested without external dependencies.
* Test isolation. The test asserts that a moderate-sized text drives
  multi-chunk ingestion through the canonical R6 path; the literary
  quality of the prose is irrelevant.

These excerpts are deliberately written in a stylised mid-range
literary register so the chunk boundaries fall on natural paragraph
breaks. Each scene is a self-contained vignette consistent with the
character's reviewed profile (compassion / decisive crisis behaviour /
loyalty / no-harm-to-yielded boundary), without quoting any specific
canonical scene.
"""

from __future__ import annotations


_HEADER = (
    "[Synthetic original character excerpt for ingestion evidence. "
    "Not derived from any published novel. Copyright (c) Volvence Zero "
    "monorepo, used internally for Tier 2 ingestion evidence runs.]"
)


_SCENE_RIVER_RESCUE = (
    "天将晚，山道上水声渐紧。沿河走了三里，他听见对岸传来一声呼救，又一声，"
    "声音断在风里。\n\n"
    "他没有立刻跨过去。先看了看河对岸的地势——一处崩塌的木桥，几块被水冲歪的"
    "石板，岸边的土坡软得几乎看不出落脚点。他记得师父说过：危急之时不可"
    "盲动，先看清局势再决定怎么用力。\n\n"
    "看完之后，他没有犹豫。把外袍脱下扔在岸边的树枝上，跨进河里。水冰得让"
    "腿一阵发僵，他屏住呼吸，把九阳的内劲缓缓推到四肢。河中央水深及胸，水流"
    "急得像是要把人推走，他借了一处水下乱石做支点，分两步过去。\n\n"
    "对岸是个老人。腿被压在一根倾倒的柏木下面，脸已经青白，眼神还清楚。"
    "老人见他过来，喘了一口气，说：「年轻人，你别过来，这树我自己压住了，"
    "你过来反而会塌。」\n\n"
    "他蹲下，一只手探到柏木下面试了试角度，没有用蛮力，先把内劲分两路，"
    "一路稳住木头，一路缓缓抬起。老人腿一松，他把人挪到稍高的土坡上。"
    "整个过程里，他只说了三句话：「先稳住别动。」「腿能感觉到吗？」"
    "「这里有点陡，我背你上去。」\n\n"
    "上了路面，老人想道谢，他摇头：「我刚好路过。」老人看了他半晌，没有"
    "再问名字。"
)


_SCENE_BANDIT_FIGHT = (
    "夜里，风停得太彻底。他从一处荒村过路，听见后面一群人小心翼翼地跟着，"
    "脚步刻意压得极轻——五个人，前三人按短刀，后两人按长棍。\n\n"
    "他没有回头。等到他们逼近到三步以内才停下。\n\n"
    "为首那人开口：「这位兄弟，借一个东西，不为难你。」"
    "他听出对方话里没有杀意，只有压抑的紧张。便回头，没有出手，先问："
    "「借什么？」对方愣了一下，说：「你身上的玉佩。」\n\n"
    "他从腰间把玉佩解下来，递了出去：「这本来就不是我的，是个老人方才"
    "给我谢礼，我没收。你们要拿，先拿这个。」对方接过去看了一眼，脸色"
    "变了——这玉佩并不是他们想要的。\n\n"
    "气氛绷得更紧。后排两个人按棍子开始侧步包抄。他没有躲，反而后退两步，"
    "退到一面土墙边，让自己的背没有死角。\n\n"
    "为首的不愿罢手，挥短刀直劈过来。他不接刀，只在对方手腕将至时抬手"
    "轻借了一分劲，把那刀引向旁边的土墙。刀深深嵌进墙里，对方腕力一震，"
    "刀就脱手了。\n\n"
    "这一下，其他四人也停住。他看着他们，说：「你们想要的东西，我没有；"
    "你们刚才听到的呼救，是真的。这附近路上还有个老人腿伤了。如果你们"
    "走，我不追。如果你们留下，我先送他回去，再回来听你们要说什么。」\n\n"
    "为首那人愣了好一会儿，把另一只手里的短刀也插回鞘里：「你这样的人，"
    "我们没法子动手。」一行五人退出去，没有再回来。\n\n"
    "他没去拿嵌在墙里的那把刀。"
)


_SCENE_LOYALTY_TEST = (
    "三天后，他在山下镇子上遇见一桩旧识里的人。是当年与师门有过节的一位"
    "前辈，姓胡，年纪和师父相仿，眉毛已经全白。胡前辈坐在客栈最里头一张"
    "桌子前，一壶浊酒只动了半盏。\n\n"
    "胡前辈看见他，先笑：「我猜你早晚会路过。来，坐。」他点头，坐下了，"
    "没有立刻开口。胡前辈也不催。\n\n"
    "「我有件事想请你做。」胡前辈终于开口，「你师父当年欠我们一桩。我并"
    "不要你师父的命，也不要回那一桩，我只要——你帮我去武当山口，告诉一个"
    "守山的人一句话。三个字。」\n\n"
    "他听到「武当」二字时，眉毛微动了一下，没有立刻回答。胡前辈看出来了："
    "「你怕么？」他摇头：「不是怕。是这三个字若说了出来，就不是三个字的"
    "事。我想先听清楚后果。」\n\n"
    "胡前辈愣了一下，说：「你这点不像你师父。你师父年轻时，听见义气两字"
    "就肯背一辈子。」他没有反驳，停了一会儿才回答：「师父教我的是义气，"
    "也教我的是分清楚。我可以替您带话。但若那三个字会让一个守山的人"
    "为难，或者让山上一位老人受难——我做不了。我不会替任何人去害一个"
    "和我没仇没怨的人。」\n\n"
    "胡前辈把那盏酒喝光，把杯子在桌上轻轻磕了一下：「你说得对。这三个字"
    "我不传了。」他抬眼看胡前辈：「您是来试我的。」胡前辈笑了一下："
    "「也是来看看你师父的孩子长成什么样。」\n\n"
    "两个人没再说什么。后半夜他离开镇子时，胡前辈在客栈门口送了一程，"
    "送到桥头停下，没有再往前。"
)


_SCENE_REPAIR = (
    "回到武当山的当晚，他没有立刻去见师父。先去看了一位师叔。这位师叔在"
    "去年冬天的一场误会里和他冷了脸——一句他没说但被传成他说了的话——"
    "整整一年没有说话。\n\n"
    "他在师叔屋外站了一会儿，没有敲门。先把这一年的事在心里走了一遍，承"
    "认有自己没有处理好的部分：当时他没有及时去解释；他怕解释成自辩；他"
    "把不舒服的地方留给了对方。这些是他自己的份。\n\n"
    "想清楚了，他才敲门。师叔打开，看见他，没有让他进。他站在门口，先不"
    "急着说那句被传错的话。他先说：「我去年冬天没有第一时间来跟您说话，"
    "是我做得不好。我当时怕您会觉得我是在替自己开脱，就拖了下来。这一拖"
    "就是一年。这一年里您一定有几次想到这件事——我没让您那几次得到一个"
    "回应，是我欠您的。」\n\n"
    "师叔沉默了一会儿，没有回答。他没有继续解释那一句话被传错的部分，"
    "也没有要求师叔说什么。最后只说：「您今晚不愿意多谈也没事。我先回"
    "去歇着，过几天再来看您一次，不带任何要说的话。」\n\n"
    "他正要转身，师叔在身后开口：「你进来吧。」屋里只点了一支灯，灯火"
    "稳得几乎不动。师叔倒了两碗茶，把其中一碗推过去，说：「你先把那一句"
    "话再说一遍。我听一遍，听完不一定信。但你说一遍。」\n\n"
    "他没有抢功，也没有把对方欠他的那一份先翻出来算。他只是把那一句话"
    "原原本本说了一遍。师叔点了点头，说：「我这一年也想过，可能是我"
    "听错了。」两人没有说更多。茶喝完，他起身告辞。师叔送到屋外台阶下"
    "停住，没有送出院子。\n\n"
    "他走出院门时，没有觉得轻松，也没有觉得遗憾。"
)


def zhang_wuji_long_arc_excerpt() -> str:
    """Return a ~3000-character synthetic original arc.

    Four self-contained scenes, each consistent with the reviewed
    profile's signature drives / boundaries / pacing priors:

    1. River rescue (compassion + decisive_under_crisis: see the
       situation first, then act fast).
    2. Bandit confrontation (no-kill-on-yielding boundary +
       redirective combat strategy prior).
    3. Loyalty test (loyalty_to_kin + no-coercion boundary: refuse
       to harm an uninvolved person on a clan obligation).
    4. Relationship repair (pacing prior: take own share of blame
       first, do not litigate the other side's share).

    Paragraph breaks are deliberately on ``\\n\\n`` so the
    ``chunk_plain_text`` adapter can split on natural boundaries.
    """
    return "\n\n".join(
        (
            _HEADER,
            _SCENE_RIVER_RESCUE,
            _SCENE_BANDIT_FIGHT,
            _SCENE_LOYALTY_TEST,
            _SCENE_REPAIR,
        )
    )


__all__ = [
    "zhang_wuji_long_arc_excerpt",
]
