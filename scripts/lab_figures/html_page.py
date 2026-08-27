"""Build a single self-contained HTML page from the rendered figures.

SVGs are inlined so the page is one file with no external dependencies: it can be
emailed, opened offline, and zoomed without losing crispness. Every figure card
carries a collapsible panel with the plotted numbers and the sha256 of each source
artifact, read back from the sidecar JSON rather than retyped.
"""

from __future__ import annotations

import html
import json
import pathlib
import re

SECTIONS = (
    (
        "coding",
        "编程实验室",
        "判决权交给自动化测试，不交给人也不交给另一个模型。用来证明范式本身成不成立。",
        ("01", "02", "03"),
    ),
    (
        "relationship",
        "对话实验室",
        "同一套方法论搬到关系场景：把测试换成会反应的用户，把隐藏约定换成每个人不同的相处方式。",
        ("04", "05", "06"),
    ),
    (
        "honest",
        "我们还没拿到什么",
        "这一节是防守面。每一格状态都由冻结产物里的判定字段生成，不是手填的。",
        ("07", "08"),
    ),
)

FIGURE_COPY = {
    "01": (
        "没人告诉它的那条规矩，只有一组学会了",
        "三条线从同一点出发是关键——它证明差距来自学习，不是能力或提示词。"
        "注意全历史那一组：它看得见每一次失败的完整记录，却始终没学会。",
    ),
    "02": (
        "又便宜，又不更差",
        "左上角是最优象限。带入内容降到约十分之一，通过率反而更高——这是帕累托占优，"
        "看懂它不需要懂统计。",
    ),
    "03": (
        "会出手不值钱，知道何时出手才值钱",
        "先看最长那条：一直干预比从不干预还差一倍多。干预能力本身不但不值钱，用错了还会主动把系统弄坏。",
    ),
    "04": (
        "同一句话，两个相反的正确答案",
        "两个用户此刻说的话逐字节相同，但各自的四段过往要求相反的回应。"
        "而且两种动作在历史里各成功一次各失败一次——数不出答案，只能读懂。",
    ),
    "05": (
        "不只是正向变多，两种失败都在缩小",
        "看构成而不只看总数：正向从个位数涨到六成以上，同时「没被接住」和「越界」两种失败都在收缩。",
    ),
    "06": (
        "人会变，中间还会消失两周",
        "开场之后这个人的相处方式翻转了。系统刚学会的规律从此全是错的——"
        "这里「记得更多」帮不上忙，需要的是发现规律变了。",
    ),
    "07": (
        "我们知道自己在哪",
        "左列是已经拿到的，右边三列是尚未拿到、也不声称拿到的。",
    ),
    "08": (
        "我们自己作废了最好看的那个数字",
        "曾经报出过 +36 个百分点。复查发现对照组行为与实验组完全相同，于是判词写成「对比无效」。"
        "一台会否决自己的仪器，是前面那些数字值得信的理由。",
    ),
}

_SVG_PROLOGUE = re.compile(r"^.*?(?=<svg)", re.DOTALL)


def _inline_svg(path: pathlib.Path) -> str:
    """Strip the XML prologue/DOCTYPE so the SVG can be embedded in HTML."""

    markup = path.read_text(encoding="utf-8")
    markup = _SVG_PROLOGUE.sub("", markup, count=1)
    # Let CSS control the rendered size rather than the baked-in pt dimensions.
    return re.sub(
        r'<svg width="[^"]*" height="[^"]*"',
        "<svg",
        markup,
        count=1,
    )


_STYLE = """
:root {
  --paper: #F8F6F1;
  --card: #FFFFFF;
  --ink: #0B0E13;
  --ink-soft: #5C5952;
  --rule: #D6D2C8;
  --sage: #6FA08C;
  --sage-deep: #41705E;
  --amber: #C8985A;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: "Microsoft YaHei", "PingFang SC", "Hiragino Sans GB", system-ui, sans-serif;
  line-height: 1.75;
  -webkit-font-smoothing: antialiased;
}
header.masthead {
  max-width: 1180px;
  margin: 0 auto;
  padding: 64px 32px 28px;
}
header.masthead h1 {
  font-size: 34px;
  line-height: 1.3;
  margin: 0 0 14px;
  letter-spacing: -0.01em;
}
header.masthead p.lede {
  font-size: 16px;
  color: var(--ink-soft);
  max-width: 62ch;
  margin: 0 0 22px;
}
.discipline {
  border-left: 3px solid var(--sage);
  padding: 10px 0 10px 18px;
  font-size: 14.5px;
  color: var(--ink-soft);
  max-width: 66ch;
}
nav.toc {
  position: sticky;
  top: 0;
  z-index: 20;
  background: rgba(248, 246, 241, 0.94);
  backdrop-filter: blur(6px);
  border-bottom: 1px solid var(--rule);
}
nav.toc .inner {
  max-width: 1180px;
  margin: 0 auto;
  padding: 13px 32px;
  display: flex;
  gap: 26px;
  align-items: center;
  flex-wrap: wrap;
  font-size: 14.5px;
}
nav.toc a { color: var(--ink-soft); text-decoration: none; }
nav.toc a:hover { color: var(--sage-deep); }
nav.toc label {
  margin-left: auto;
  color: var(--ink-soft);
  cursor: pointer;
  user-select: none;
  display: inline-flex;
  gap: 8px;
  align-items: center;
}
main { max-width: 1180px; margin: 0 auto; padding: 8px 32px 96px; }
section.band { padding-top: 54px; }
section.band > h2 {
  font-size: 24px;
  margin: 0 0 8px;
}
section.band > p.band-note {
  color: var(--ink-soft);
  font-size: 15px;
  max-width: 66ch;
  margin: 0 0 8px;
}
figure.card {
  background: var(--card);
  border: 1px solid var(--rule);
  border-radius: 4px;
  margin: 30px 0 0;
  padding: 26px 28px 20px;
}
figure.card .eyebrow {
  font-size: 12.5px;
  letter-spacing: 0.10em;
  color: var(--sage-deep);
  text-transform: uppercase;
  margin-bottom: 7px;
}
figure.card h3 { font-size: 20px; margin: 0 0 10px; }
figure.card p.howto {
  color: var(--ink-soft);
  font-size: 14.5px;
  max-width: 74ch;
  margin: 0 0 20px;
}
figure.card .plot { margin: 0 -8px; }
figure.card .plot svg { width: 100%; height: auto; display: block; }
details.prov {
  margin-top: 18px;
  border-top: 1px solid var(--rule);
  padding-top: 14px;
  font-size: 13.5px;
}
details.prov summary {
  cursor: pointer;
  color: var(--sage-deep);
  font-size: 14px;
}
details.prov .tier {
  display: inline-block;
  margin: 12px 0 6px;
  padding: 3px 10px;
  border: 1px solid var(--rule);
  border-radius: 999px;
  color: var(--ink-soft);
  font-size: 12.5px;
}
details.prov pre {
  background: var(--paper);
  border: 1px solid var(--rule);
  border-radius: 3px;
  padding: 14px 16px;
  overflow-x: auto;
  font-family: "Cascadia Mono", "Consolas", ui-monospace, monospace;
  font-size: 12.5px;
  line-height: 1.65;
  color: var(--ink-soft);
  margin: 8px 0 0;
}
details.prov .src { margin-top: 12px; color: var(--ink-soft); }
details.prov .src code {
  font-family: "Cascadia Mono", "Consolas", ui-monospace, monospace;
  font-size: 12px;
  color: var(--ink);
  word-break: break-all;
}
body.hide-honest section.band#honest { display: none; }
footer.colophon {
  border-top: 1px solid var(--rule);
  margin-top: 64px;
  padding-top: 22px;
  font-size: 13.5px;
  color: var(--ink-soft);
}
"""

_SCRIPT = """
const toggle = document.getElementById('honest-toggle');
toggle.addEventListener('change', () => {
  document.body.classList.toggle('hide-honest', !toggle.checked);
});
"""


def _provenance_block(sidecar: dict) -> str:
    numbers = {
        key: value
        for key, value in sidecar.items()
        if key
        not in {
            "figure_index",
            "slug",
            "rendered_by",
            "font_family",
            "claim",
            "provenance",
            "evidence_tier",
        }
    }
    rows = "".join(
        f'<div class="src">· <code>{html.escape(item["path"])}</code><br>'
        f'&nbsp;&nbsp;sha256 <code>{html.escape(item["sha256"])}</code></div>'
        for item in sidecar["provenance"]
    )
    return (
        "<details class=\"prov\"><summary>展开：图上的数字与来源产物指纹</summary>"
        f'<div class="tier">证据档位：{html.escape(sidecar["evidence_tier"])}</div>'
        f"<pre>{html.escape(json.dumps(numbers, ensure_ascii=False, indent=1))}</pre>"
        f"<div class=\"src\"><strong>来源产物</strong></div>{rows}"
        "</details>"
    )


def build(figures_dir: pathlib.Path, rendered_at: str) -> str:
    """Assemble the page from the manifest and the per-figure sidecars."""

    manifest = json.loads((figures_dir / "manifest.json").read_text(encoding="utf-8"))
    by_index = {item["figure_index"]: item for item in manifest["figures"]}

    bands = []
    for anchor, heading, note, indices in SECTIONS:
        cards = []
        for index in indices:
            entry = by_index[index]
            sidecar = json.loads(
                (figures_dir / entry["sidecar"]).read_text(encoding="utf-8")
            )
            title, howto = FIGURE_COPY[index]
            svg = _inline_svg(figures_dir / entry["svg"])
            cards.append(
                f'<figure class="card" id="fig{index}">'
                f'<div class="eyebrow">图 {index}</div>'
                f"<h3>{html.escape(title)}</h3>"
                f'<p class="howto">{html.escape(howto)}</p>'
                f'<div class="plot">{svg}</div>'
                f"{_provenance_block(sidecar)}"
                "</figure>"
            )
        bands.append(
            f'<section class="band" id="{anchor}">'
            f"<h2>{html.escape(heading)}</h2>"
            f'<p class="band-note">{html.escape(note)}</p>'
            + "".join(cards)
            + "</section>"
        )

    nav_links = "".join(
        f'<a href="#{anchor}">{html.escape(heading)}</a>'
        for anchor, heading, _, _ in SECTIONS
    )
    source_count = len(
        {item["path"] for figure in manifest["figures"] for item in figure["provenance"]}
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Volvence 实验结果图集</title>
<style>{_STYLE}</style>
</head>
<body>
<header class="masthead">
<h1>两个实验室，八张图</h1>
<p class="lede">编程实验室回答的是科学问题：在冻结的大模型之上，一个小控制器能不能只靠「做完之后结果好不好」
学会何时该出手。对话实验室回答的是迁移问题：同一套方法搬到关系场景，能不能同样立住。</p>
<div class="discipline">图上每一个数字都是渲染时从冻结产物里读出来的，没有一处手工填写。
每张图下面可以展开它用到的数字，以及来源产物的 sha256——现场重算即可核对。
本页共引用 {source_count} 份冻结产物。</div>
</header>
<nav class="toc"><div class="inner">{nav_links}
<label><input type="checkbox" id="honest-toggle" checked> 显示「我们还没拿到什么」</label>
</div></nav>
<main>
{"".join(bands)}
<footer class="colophon">
生成时间 {html.escape(rendered_at)}　·　由 <code>scripts/render_lab_evidence_figures.py</code> 生成，请勿手工编辑<br>
诚实边界：编程域证据全程处于影子模式，未改动线上接线；关系域为开发档、合成个体，
正式档证据、真人验证与生产接线均未授权。
</footer>
</main>
<script>{_SCRIPT}</script>
</body>
</html>
"""


__all__ = ["FIGURE_COPY", "SECTIONS", "build"]
