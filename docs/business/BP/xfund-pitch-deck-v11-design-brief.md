# VolvenceZero — Xfund Pitch Deck v11 — Designer Brief

> Status: **v1.0 (2026-05-17)**
> Audience: graphic designer / freelancer briefed to produce the on-screen and PDF version of the v11 pitch deck.
> Companion document: `xfund-pitch-deck-v11-zhaojiangbo.md` (the deck content / scripts / production-tone notes). This brief is the *visual* layer; the deck markdown is the *content* layer.
>
> **One-line creative direction:**
>
> > *Make this deck look like it could be an article in The Information or a Stripe Press book. Editorial-grade typographic discipline. Dark-mode reading experience. One sage-green accent. Hairline rules. Generous margins. No icons. No gradients. No flourishes. The visual restraint is the message — this team knows the difference between elegance and decoration.*

---

## 0. Mood reference (give the designer all of these)


| Should look like                                                                 | Should not look like                   |
| -------------------------------------------------------------------------------- | -------------------------------------- |
| Stripe Press book interiors (*High Growth Handbook*, *The Revolt of the Public*) | YC pitch deck templates                |
| The Information long-form article layouts                                        | TechCrunch article hero                |
| Anthropic research-paper PDFs                                                    | OpenAI marketing site                  |
| Bloomberg Businessweek data features                                             | McKinsey blue-corporate decks          |
| Tufte *Beautiful Evidence* internal pages                                        | Any deck with SmartArt or stock photos |
| Linear product launch pages (the dark-mode editorial ones)                       | Notion marketing site (too friendly)   |
| Patagonia annual reports (slow, considered)                                      | A startup landing page                 |


**The single most important sentence in this brief:** *every visual decision should reduce the chance that Patrick Chung perceives this deck as "a startup pitch" and increase the chance that he perceives it as "a thesis paper that happens to be on screen."*

---

## 1. Color tokens

Build these as design tokens (Figma variables / CSS custom properties) so global theme swap (dark → light) is one switch.

### 1.1 Dark theme (primary — meeting / on-screen / projection)

```
─── Surfaces ──────────────────────────────
--bg-primary         #0B0E13   near-black, NOT pure black (pure black is harsh on projector)
--bg-secondary       #14181F   one shade up; for cards / table headers / page-frame
--bg-tertiary        #1B2028   for nested cards (rarely used)

─── Text ──────────────────────────────────
--text-primary       #E8E6E1   warm off-white (NOT pure white, less clinical)
--text-secondary     #B6B2AB   muted warm
--text-tertiary      #8C8A86   for captions, page numbers, secondary labels
--text-disabled      #4A4D52   for ghost/placeholder

─── Lines ─────────────────────────────────
--rule-hairline      #2A2D33   1px hairline rules
--rule-medium        #3A3E45   2px section rules
--rule-strong        #5A5F68   used very rarely (only for column dividers in Slide 15)

─── Accent (sage / moss — the only non-neutral color) ─
--accent-base        #6FA08C   "moss"; default accent
--accent-strong      #8FC7AC   "sage"; punchline boxes / kill criterion / key numbers
--accent-tint        #6FA08C @ 8%  background wash for highlighted rows / Phase 1 column

─── Warning (used only for risk / falsifier rows) ──
--warn-base          #C8985A   muted amber
--warn-tint          #C8985A @ 6%  background wash for risk rows
```

### 1.2 Light theme (leave-behind PDF — for investor reading on flights / desks)

```
--bg-primary         #F8F6F1   warm cream, NOT pure white
--bg-secondary       #EFECE5
--bg-tertiary        #E6E2D9

--text-primary       #1A1D22   near-black warm, NOT pure black
--text-secondary     #4A4D52
--text-tertiary      #6E7178
--text-disabled      #A8AAAE

--rule-hairline      #D6D2C9
--rule-medium        #BFBAAF
--rule-strong        #8C8A86

--accent-base        #5A8F7B   moss runs slightly darker on light to maintain contrast
--accent-strong      #3F7560
--accent-tint        #5A8F7B @ 10%

--warn-base          #A47B40
--warn-tint          #A47B40 @ 8%
```

### 1.3 Color rules

- **Maximum 3 non-neutral colors on a page.** Sage + amber + body text. That is it.
- **No gradients. Ever.** Solid fills only. The one exception: arrow heads on the Slide 4 flywheel may have a soft fade-out at the tail (5% opacity) to suggest momentum — that is the only "gradient" allowed in the entire deck.
- **No drop shadows. No glow. No bevel.** Depth is communicated through hairline rules and tint backgrounds, not 3D effects.
- **No icons unless explicitly listed in §6.** No emoji. No clip-art.

---

## 2. Typography

### 2.1 Font selection (two routes)

**Route A — premium (recommended; ~$400-800 one-time)**


| Role                           | Font                          | Weights used                      |
| ------------------------------ | ----------------------------- | --------------------------------- |
| Headlines, section titles      | **GT Sectra** (display serif) | Light 300 · Regular 400           |
| Body, quotes, "Observed" notes | **GT Sectra** (text serif)    | Regular 400 · Italic 400i         |
| UI / table labels / small caps | **Söhne**                     | Buch 400 · Kräftig 600 · Mono 400 |
| Numbers / code / arXiv IDs     | **Söhne Mono**                | Buch 400                          |


**Route B — free (acceptable; ~80 pts of the premium effect)**


| Role                           | Font               | Weights used                            |
| ------------------------------ | ------------------ | --------------------------------------- |
| Headlines, section titles      | **Source Serif 4** | Light 300 · Regular 400                 |
| Body, quotes, "Observed" notes | **Source Serif 4** | Regular 400 · Italic 400i               |
| UI / table labels / small caps | **Inter**          | Regular 400 · Medium 500 · Semibold 600 |
| Numbers / code                 | **IBM Plex Mono**  | Regular 400                             |


**Forbidden fonts**: Helvetica, Arial, Roboto, SF Pro Display, Calibri, Times New Roman, anything Google-default. Each of those carries a context smell that breaks the editorial mood.

### 2.2 Type scale (16:9 slide, 1920 × 1080 export)


| Token            | Use                         | Family          | Size | Weight | Tracking         | Leading |
| ---------------- | --------------------------- | --------------- | ---- | ------ | ---------------- | ------- |
| `display-xl`     | Cover tagline (line 1 only) | Serif           | 64pt | 300    | -10              | 1.10    |
| `display-l`      | Section divider title       | Serif italic    | 44pt | 400i   | 0                | 1.15    |
| `h1`             | Slide title                 | Serif           | 32pt | 400    | -5               | 1.20    |
| `h2`             | Claim / sub-headline        | Serif italic    | 22pt | 400i   | 0                | 1.30    |
| `h3`             | In-slide block label        | Sans            | 14pt | 600    | +60 (small caps) | 1.30    |
| `body`           | Body paragraphs             | Serif           | 16pt | 400    | 0                | 1.55    |
| `body-s`         | Table cells                 | Sans            | 13pt | 400    | 0                | 1.40    |
| `mono`           | Numbers in tables           | Mono            | 14pt | 400    | 0                | 1.40    |
| `caption`        | Page number, source notes   | Sans            | 10pt | 400    | +40              | 1.30    |
| `observed-label` | "Observed" footer label     | Sans small caps | 11pt | 600    | +120             | 1.30    |
| `observed-body`  | "Observed" footer body      | Serif italic    | 13pt | 400i   | 0                | 1.55    |


### 2.3 Type rules

- **Body is serif. Always.** This is the single decision that makes the deck read as "essay, not pitch."
- **All-caps only in `h3` (small caps with positive tracking).** Never set body in all-caps.
- **Tables: labels in sans, numbers in mono.** Mono numbers align right edge automatically; this is what makes the financial tables look like FT/Bloomberg.
- **No underlines for emphasis. No bold for emphasis in body.** Use italic in body, sage color for punchlines.
- **Headlines never use bold.** Light or Regular only. Bold serif headlines look like NYT op-eds from 2008.

---

## 3. Spacing and grid

### 3.1 Slide canvas

- **Aspect:** 16:9
- **Export size:** 1920 × 1080 (PDF), 1280 × 720 (Figma working size — scale up at export)
- **Outer margin:** 9% on each side (≈ 173px on a 1920 canvas)
- **Top margin:** 8% (≈ 86px)
- **Bottom margin:** 7% (≈ 76px)

### 3.2 Grid

- **12 columns**, 24px gutter (at 1920 canvas)
- Most page content sits inside columns 2-11 (i.e. 10 columns)
- Tables can use full 12 columns when dense
- Cover and section dividers use only columns 2-7 (left-aligned, 50% width)

### 3.3 Vertical rhythm

- All vertical spacing is a multiple of **8px**
- Common spacings: 8 / 16 / 24 / 32 / 48 / 64 / 96 / 128
- Between H1 and body: **48px**
- Between body block and "Observed" footer: **64px** (with 1px hairline at midpoint)
- Inside a table: row height **40px** (12px padding top/bottom on `body-s`)

---

## 4. Page-frame elements (every slide gets these)

```
┌───────────────────────────────────────────────────────────────────┐
│  [Slide title — h1]                              VOLVENCE · 04/18 │  ← caption, top-right
│  [Optional h2 claim — italic serif]                               │
│                                                                   │
│  [body content area — columns 2-11]                               │
│                                                                   │
│                                                                   │
│                                                                   │
│  ─────────────────────────────────────────────────                │  ← optional hairline if Observed
│                                                                   │
│  OBSERVED                                                         │
│    [observed body — italic serif, indented]                       │
│                                                                   │
│  ┌─                                                              ─┘
│  ┃ moss accent mark                              part B · slide 04│  ← caption, bottom
└───────────────────────────────────────────────────────────────────┘
```

### 4.1 Page-number system

- Top-right: `VOLVENCE · 04 / 18` in `caption` style, `text-tertiary`
- Bottom-right: `part B · slide 04` in `caption`, `text-tertiary` — this provides parts navigation without needing a TOC slide

### 4.2 Continuity mark (single brand glyph)

Bottom-left of every slide: a **single 4px sage rectangle, 24px tall**, vertical, flush against left margin. That is the entire branding element. No logo. No wordmark. No tagline. The bar is `--accent-base` and is the only place the accent color appears on slides without sage content.

If the cover or section dividers have explicit "VOLVENCE" wordmark, they use only `text-primary` color and `display-l` style — never the sage bar. The bar is for body slides only.

---

## 5. Page templates

The deck has 18 body slides + 4 section dividers = 22 frames. Each maps to one of these 6 templates:


| Template      | Used on                            | Description                                                 |
| ------------- | ---------------------------------- | ----------------------------------------------------------- |
| `T-cover`     | Slide 1                            | Cover. Two staggered taglines. Heavy whitespace.            |
| `T-section`   | 4 dividers (between Parts A/B/C/D) | Part name + section title. Left 1/3 only.                   |
| `T-thesis`    | Slides 3, 4, 7, 12, 15             | Dense table or diagram-led page. The "must-land" pages.     |
| `T-evidence`  | Slides 8, 9, 11, 13, 14            | Multi-block data page (financials, unit economics, JV map). |
| `T-narrative` | Slides 5, 6, 10, 16                | Mixed text + table + Observed footer.                       |
| `T-list`      | Slides 2, 17, 18                   | List-led (team, ask, close). Lower density.                 |


§6 details the four must-land pages explicitly. The other templates inherit from this base spec.

---

## 6. The four must-land slides — wireframes & specs

These are the four pages where Patrick should pause, ask the most informative question, and walk out of the room able to repeat the punchline. Spend disproportionate visual attention here.

### 6.1 Slide 4 — The Trajectory Flywheel (`T-thesis` · diagram-led)

**Hierarchy:**

1. The flywheel diagram dominates — occupies columns 2-8, vertically centered
2. The metric/economic/irreversibility table runs columns 9-12, right side
3. The "parenting maximizes all six" green tagline below the diagram, full width
4. The "Observed" footer at the very bottom

**Wireframe:**

```
┌───────────────────────────────────────────────────────────────────┐
│ The Trajectory Flywheel                          VOLVENCE · 04/18 │
│ [h2: The compounding mechanism that makes Generation 2 an asset]  │
│                                                                   │
│  ┌─────────────────────────┐   ┌──────────────────────────────┐  │
│  │                         │   │ Stage  Metric    Economic    │  │
│  │   ① Continuity          │   │ ─────  ──────    ────────    │  │
│  │        ↓                │   │ ①      D30/90    CAC ↓       │  │
│  │   ② State accumulates   │   │ ②      bytes/wk  asset ↑     │  │
│  │        ↓                │   │ ③      Δ/wk      cost ↓      │  │
│  │   ③ Runtime tunes       │   │ ④      relevance rev/turn    │  │
│  │        ↓                │   │ ⑤      DA turns  daily-rel   │  │
│  │   ④ Quality rises       │   │ ⑥      session   LTV ×       │  │
│  │        ↓                │   │ ⑦      cum bytes growth ↑    │  │
│  │   ⑤ Dependence rises    │   │ ⑧      reonbrd   churn floor │  │
│  │        ↓                │   │                              │  │
│  │   ⑥ Density rises       │   │ Each stage's irreversibility │  │
│  │        ↓                │   │ property = the moat.         │  │
│  │   ⑦ Data compounds      │   │                              │  │
│  │        ↓                │   └──────────────────────────────┘  │
│  │   ⑧ Switching cost ↑    │                                     │
│  │        ↓                │                                     │
│  │   └────→ back to ①      │                                     │
│  └─────────────────────────┘                                     │
│                                                                   │
│  [sage tagline] Parenting maximizes all six conditions            │
│  simultaneously — the proof environment, not the destination.     │
│                                                                   │
│  ─────────────────────────────────────────────────                │
│  OBSERVED — early closed-alpha pilot                              │
│    High-frequency intervention raised short-term engagement       │
│    but visibly weakened week-3 trust signals on the same cohort.  │
│    The flywheel optimizes for relationship health, not session-   │
│    level activity.                                                │
│                                                                   │
│  ▌                                              part A · slide 04 │
└───────────────────────────────────────────────────────────────────┘
```

**Diagram specifics:**

- Eight stage nodes set in `body` serif, **NOT** in boxes — boxes corporatize. Use a 1.5px sage hairline circle (28px diameter) around each numeral, with stage label set right of the circle in serif.
- Arrows between stages: 1.5px stroke, sage `--accent-base`, with a 12px arrowhead. Tail end fades from 100% to 70% opacity over the last 30% of arrow length (the only "gradient" allowed in the deck).
- The wraparound arrow ("back to ①") uses a curved path on the right side of the column, again 1.5px sage, with a slight 60% opacity to suggest cyclical motion without dominating.
- The right-side table uses `body-s` (sans) for "Stage / Metric / Economic" header row and `mono` for the column-1 stage symbol.

**Interaction:** when the speaker reaches "back to ①" on the slide, the wraparound arrow can fade in last (after the linear arrows are visible). Keep this subtle — 400ms fade.

---

### 6.2 Slide 7 — Memory is substrate. Relationship governance is the product. (`T-thesis` · table-led)

**Hierarchy:**

1. The big claim runs full-width across the top, set in `h2` serif italic
2. The page is split horizontally by a hairline rule
3. Above the rule: 4-bullet list "what foundation memory does (and will increasingly do well)" — set in `text-tertiary` (deliberately faded — visual concession)
4. Below the rule: 7-row governance dimensions table — full saturation (this is the product)
5. Punchline green block at the bottom, before "Observed"
6. "Observed" footer

**Wireframe:**

```
┌───────────────────────────────────────────────────────────────────┐
│ Memory is substrate. Relationship governance is the product.     │
│ [h2: The substitution-risk reframe — the deck's center of gravity│
│  for Q1.]                                                        │
│                                                                   │
│  WHAT FOUNDATION MEMORY DOES                            [muted]  │
│   · recall what was said                                          │
│   · maintain long context                                         │
│   · personalize tone                                              │
│   · persist preferences                                           │
│                                                                   │
│  ─────────────────────────────────────────────────                │
│                          ↓                                        │
│  WHAT IT DOES NOT DO  (and will not, because labs optimize for   │
│  generality across one billion users)                            │
│                                                                   │
│  ┌──────────────────────────┬───────────────────────────────┐    │
│  │ What persists            │ which states should compound… │    │
│  │ What adapts              │ which signals change behavior │    │
│  │ What decays              │ which states age out vs never │    │
│  │ What never transfers     │ Alice ≠ Bob; audit evidence   │    │
│  │ When monetization stops  │ post-rupture suppression gate │    │
│  │ How rupture is repaired  │ typed-enum durable feedback   │    │
│  │ What consent means       │ day 1 vs day 365; revocation  │    │
│  │   longitudinally         │                               │    │
│  └──────────────────────────┴───────────────────────────────┘    │
│                                                                   │
│  ┌─ sage 1px border ────────────────────────────────────────┐    │
│  │ The moat is not remembering more.                        │    │
│  │ The moat is deciding what persists, what adapts,         │    │
│  │ what repairs trust, and what becomes economically        │    │
│  │ actionable over years.                                   │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ─────────────────────────────────────────────────                │
│  OBSERVED — governance in action                                  │
│    System inferred disengagement from silence and over-          │
│    intervened on a parent who had simply stepped away.            │
│    Re-engagement only occurred after the agent acknowledged      │
│    its own uncertainty. Repair, not avoidance.                   │
│                                                                   │
│  ▌                                              part A · slide 07 │
└───────────────────────────────────────────────────────────────────┘
```

**Specifics:**

- The 4-bullet list above the rule is set 100% in `text-tertiary` `#8C8A86`. Visual signal: "this is real, but it is not where we live." The fading is the message.
- The 7-row table below the rule uses `text-primary` for the left column (governance dimension) and `text-secondary` for the right column (description). Hairline `--rule-hairline` between rows.
- The punchline block is `body` serif on `--bg-secondary` background with a 1px sage `--accent-base` border. Padding 24px all sides.
- The down-arrow (↓) between the two halves is centered, sage `--accent-base`, 24pt — the visual hinge of the page.

---

### 6.3 Slide 12 — Phase 1 → 2 → 3 → 4 sequencing (`T-thesis` · timeline-but-not-gantt)

**Hierarchy:**

1. Big claim across the top
2. Four-column phase timeline filling columns 2-11
3. Phase 1 column has 8% sage tint background (current focus)
4. Bottom punchline: "each phase's evidence is the entry ticket for the next"

**Wireframe:**

```
┌───────────────────────────────────────────────────────────────────┐
│ Phase 1 → 2 → 3 → 4: why this wedge precedes the category        │
│                                                                   │
│  ┌─────────────────┬─────────────┬─────────────┬───────────────┐ │
│  │ PHASE 1   ✦     │  PHASE 2    │  PHASE 3    │  PHASE 4      │ │
│  │ M0–M18          │ M0–M12 ‖    │ M12–M36     │ M36+          │ │
│  │ [sage tint bg]  │             │             │               │ │
│  │                 │             │             │               │ │
│  │ Parenting       │  Mobi       │  Enterprise │  Cross-       │ │
│  │ 高盖伦 + plat-  │  B2B2C      │  / regulated│  vertical     │ │
│  │ form + hardware │  private-   │  workflows  │  relationship │ │
│  │                 │  traffic    │             │  graph        │ │
│  │                 │             │             │               │ │
│  │ Asset proof     │  Unit-econ  │  Governance │  Platform     │ │
│  │                 │  proof      │  defensibil │  option       │ │
│  │                 │             │  -ity proof │               │ │
│  │ Trajectory      │  RMB 5.6 →  │  Audit /    │  5–10 yr      │ │
│  │ compounds.      │  16.8 →     │  deletion / │  story.       │ │
│  │ Switching cost  │  33.6M ramp │  consent    │  NOT 12 mo.   │ │
│  │ is biographical │             │             │               │ │
│  └─────────────────┴─────────────┴─────────────┴───────────────┘ │
│       ↓                ↓                ↓                ↓        │
│   asset →        unit econ →      governance →       platform     │
│                                                                   │
│  ┌─ sage 1px border ────────────────────────────────────────┐    │
│  │ Each phase's evidence is the entry ticket for the next.  │    │
│  │ Parenting precedes the category because all six flywheel │    │
│  │ conditions peak there simultaneously.                     │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ▌                                              part C · slide 12 │
└───────────────────────────────────────────────────────────────────┘
```

**Specifics:**

- **Column widths are intentionally unequal**: Phase 1 ≈ 30%, Phase 2 ≈ 25%, Phase 3 ≈ 25%, Phase 4 ≈ 20%. This visually communicates "current focus is here."
- **Phase 1 column** has `--accent-tint` (sage @ 8%) background fill plus a small ✦ glyph next to the title. This is the only place a non-text decoration appears in the entire deck — and it is a single character.
- **Top row (PHASE N)** uses `h3` small caps sans, sage `--accent-base`.
- **Time row (M0-M18)** uses `mono` `--text-tertiary`.
- **Body** uses `body` serif `--text-primary`.
- **The "asset → unit econ → governance → platform" tag row** below the columns uses `caption` sans, `--text-tertiary`, with the arrow-glyph in sage. This is the "entry ticket" metaphor visualized.
- The "‖" symbol next to Phase 2 dates indicates *parallel with Phase 1* — visual shorthand. Use a small footnote or tooltip equivalent: "‖ runs in parallel with Phase 1."

---

### 6.4 Slide 15 — The uncertainty boundary (`T-thesis` · three-temperature columns)

**Hierarchy:**

1. Title across the top
2. Three columns of equal width, each with its own background tint
3. The "discipline" green block below

**Wireframe:**

```
┌───────────────────────────────────────────────────────────────────┐
│ The uncertainty boundary: proven · suspected · believed           │
│ [h2 italic: A controlled-humility surface — we name it ourselves.]│
│                                                                   │
│  ┌────────────────┬────────────────┬───────────────────┐         │
│  │ PROVEN         │ STRONGLY       │ LONG-TERM         │         │
│  │ today          │ SUSPECTED      │ BELIEF            │         │
│  │ DD will verify │ (evidence;     │ (5–10 yr thesis)  │         │
│  │                │  not yet at    │                   │         │
│  │ [cool tint]    │  scale)        │ [neutral grey]    │         │
│  │                │ [sage tint]    │                   │         │
│  ├────────────────┼────────────────┼───────────────────┤         │
│  │ 1063+ contract │ Trajectory     │ Generation 2 data │         │
│  │ tests PASS     │ flywheel will  │ becomes a primary │         │
│  │                │ compound on    │ AI asset class    │         │
│  │ 96 new tests   │ parenting in   │                   │         │
│  │ PASS, 0 reg    │ 12-18 months   │ Governance        │         │
│  │                │                │ becomes a regula- │         │
│  │ 5 verticals    │ Mobi 0.6-1.0%  │ tory standard     │         │
│  │ co-loaded      │ in 90 days     │                   │         │
│  │                │                │ Cross-vertical    │         │
│  │ Closed-alpha   │ Foundation mem │ relationship      │         │
│  │ live           │ helps our      │ graph forms a     │         │
│  │                │ substrate cost │ platform          │         │
│  │ Memory probes  │                │                   │         │
│  │ 4/4 PASS;      │ Phase 1 →      │ Foundation labs   │         │
│  │ RAG fails 3/4  │ Phase 3        │ do not vertical-  │         │
│  │                │ generalizes    │ ize relationship  │         │
│  │ 6 signed JVs   │                │ governance over   │         │
│  │                │ Governance     │ 5-10 years        │         │
│  │ RMB 5M self-   │ opens 50K      │                   │         │
│  │ funded         │ enterprise     │                   │         │
│  └────────────────┴────────────────┴───────────────────┘         │
│                                                                   │
│  ┌─ sage 1px border ────────────────────────────────────────┐    │
│  │ Column 1 is what diligence verifies.                      │    │
│  │ Column 2 has a kill criterion or falsifier in this deck.  │    │
│  │ Column 3 does not need to be proven for this round.       │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ▌                                              part C · slide 15 │
└───────────────────────────────────────────────────────────────────┘
```

**Specifics — column-temperature tints:**


| Column                | Header label                                                          | Background tint                      | Body text saturation  |
| --------------------- | --------------------------------------------------------------------- | ------------------------------------ | --------------------- |
| 1. Proven             | `PROVEN` (sans small caps, sage `--accent-base`)                      | `#1A2229` (cool blue-gray @ 50%)     | `--text-primary` 100% |
| 2. Strongly suspected | `STRONGLY SUSPECTED` (sans small caps, sage strong `--accent-strong`) | `#1F2620` (sage @ 8%)                | `--text-secondary`    |
| 3. Long-term belief   | `LONG-TERM BELIEF` (sans small caps, neutral `--text-tertiary`)       | `#1A1D22` (warm-gray ≈ secondary bg) | `--text-tertiary`     |


The cooling temperature from left to right is the message: as confidence drops, color saturation drops. **Do not use prefix icons (no ✓ ? ★).** The temperature tint *is* the icon.

Internal row separators inside each column: `--rule-hairline` 1px.

The three columns are separated by `--rule-strong` 2px vertical rules — this is the only place in the deck `--rule-strong` is used.

---

## 7. The "Observed" footer — the deck's signature element

This is the v10.1 / v11 visual signature. It must be unmistakable and consistent across the four slides where it appears (4, 5, 6, 7).

**Visual spec:**

```
─────────────────────────────────────  ← --rule-hairline 1px, full body width
                                         32px above the label
OBSERVED — early closed-alpha pilot
  ↑                ↑
  observed-label   observed-label, regular weight, --text-tertiary

  Several parents resumed interaction after weeks of inactivity
  not to ask for advice, but to preserve continuity of observation.
  
  The earliest visible form of switching cost is behavioral,
  not commercial.

  ↑
  observed-body — italic serif, --text-secondary, 13pt
  indented 24px from the label
```

**Spacing:**

- 64px above the hairline
- 16px between hairline and `OBSERVED` label
- 12px between label and observed body
- 48px below observed body before page footer

**Voice rules (these are the content — designer should not modify; included so designer understands the intended texture):**

- Always begins with `OBSERVED — [scope]` where scope is one of: `early closed-alpha pilot`, `early pilot, governance in action`, `early pilot, onboarding friction`, `early pilot behavior`
- Body is past-tense, qualitative, NEVER contains numbers or percentages
- Body is 2-4 sentences, max 60 words
- Italic serif on principle: it reads as marginal commentary, not as a competing claim

**Critical:** the Observed block must look like a **footnote in a published essay**, NOT like a callout box on a marketing page. If the designer wants to draw a box around it: the answer is no.

---

## 8. Section dividers

Four section dividers separate the body deck into:

- **Part A** — Thesis & flywheel (Slides 3-4)
- **Part B** — Parenting (Slides 5-6)
- **Part C** — Memory→Governance + Mobi + Phases + Financials (Slides 7-15)
- **Part D** — Risks, ask, close (Slides 16-18)

Each divider is a full slide showing only the part name + section title:

```
┌───────────────────────────────────────────────────────────────────┐
│                                                                   │
│                                                                   │
│  PART B                                                           │
│                                                                   │
│  Parenting as the                                                 │
│  proof environment                                                │
│                                                                   │
│  ────                                                             │
│                                                                   │
│  Six wedge-quality criteria peak in one vertical.                 │
│  The flywheel locks in fastest where it can be observed first.    │
│                                                                   │
│                                                                   │
│  ▌                                                                │
└───────────────────────────────────────────────────────────────────┘
```

**Specs:**

- "PART B" uses `caption` (mono `Söhne Mono` / `IBM Plex Mono` 10pt, +120 tracking, sage `--accent-base`)
- Section title uses `display-l` (serif italic 44pt) — split across 2 lines to break the symmetry
- A short hairline rule (4 chars wide) below the title
- One italic descriptive line in `body` `--text-secondary`
- Whole composition aligned to **column 2 left edge** (NOT centered) and vertically centered in the upper-third zone
- 90% of the slide is empty — the silence IS the section break

---

## 9. Diagram-level rules

### 9.1 Flywheel (Slide 4)

Already covered in §6.1. Key rule reiterated: **no boxes around the eight stages**. Numerals in 28px sage hairline circles only.

### 9.2 Three-generation diagram (Slide 3)

Use the same code-block monospace style as in the markdown — the deck markdown `Generation 0 / Generation 1 / Generation 2 / Adjacent` block is meant to render as quasi-ASCII typography. Set in `mono` 16pt with a tasteful left-aligned indent. **No boxes. No icons. No arrows.** Just typography. The fact that this is *literally code-block-style* in a deck is part of the visual signal — "this team thinks in primitives."

### 9.3 Day-30 conversation diff (Slide 9)

Two parallel columns:

- **Left (SCRM baseline):** `--text-tertiary` muted; sample dialog set in `body` serif italic; the column has a subtle 1px hairline frame in `--rule-hairline`
- **Right (Volvence runtime):** `--text-primary` full saturation; same sample format; **plus** a "runtime state" sub-block above the dialogue, set in `mono` 12pt, sage colored, fixed-width, looking like a code log:
  ```
  user_state.lifecycle_stage = "newborn_4-6_weeks"
  user_state.no_bone_broth   = true
  relationship_state         = "early_trust_building"
  recommend_now              = false
  ```
  This is the visual lede of the page — runtime state literally rendered as data, on the same slide as the conversation. Strong asymmetry between left (vague broadcast) and right (typed runtime + governed conversation) is the entire pedagogical point.

### 9.4 12-week parenting arc (Slide 6)

Four-row table where each row is one week (Week 1 / 4 / 8 / 12). Each row has three vertical zones:


| Conversation surface | Runtime state accumulated | System behavior change |
| -------------------- | ------------------------- | ---------------------- |


The middle column ("Runtime state accumulated") uses `mono` 12pt, sage colored — same code-log treatment as Slide 9. Visual rhyme between Slides 6 and 9 is intentional: in both cases, the runtime state is rendered as data, not described in prose.

---

## 10. Animation / motion

**Permitted:**

- `fade-in`: 200-400ms, ease-out
- `row-fade`: a single table row from 30% → 100% opacity when speaker arrives at it (use sparingly, only on Slides 4, 7, 12, 15)
- Cover-slide staggered fade: line 1 at 0ms, line 2 at +2500ms

**Forbidden** (cause for rejection):

- `fly-in`, `dissolve`, `spin`, `bounce`, `pop`, `reveal`, `curtain`, `blinds`, `3D rotate`, `cube`, `morph`
- Any animation longer than 600ms
- Any animation that includes ease-in-out-bounce or any easing curve named after a physical metaphor
- Cursor-tracking effects, hover-state animations on PDF, video embeds, GIF embeds

**Interaction with the deck cadence:**

- Speaker spends 2-4 minutes per slide on must-land pages. Avoid animations that demand attention for more than the first 2 seconds — by second 3 the page should be static.
- The *one* deliberate animation moment per slide is acceptable (e.g., the wraparound arrow on Slide 4 fading in last, or the Observed footer fading in after the speaker reaches it). More than one is too much.

---

## 11. Light theme variant (leave-behind PDF)

A separate but visually equivalent theme, used to produce the take-away PDF that Patrick will read on a flight or forward to partners.

**Production rule:** build the dark theme first; the light theme is a single global token swap (§1.2). Verify visually after swap — typography sometimes needs +1pt body weight in light mode to compensate for ink-on-cream contrast.

**Output for light theme:**

- Single PDF, 16:9, 1920×1080 export
- All page-frame elements identical in position
- Sage accent darkens to `#5A8F7B` (already specified in §1.2)
- The four "Observed" italic blocks remain italic, but lose the muted background they have in dark theme
- Section dividers: cream background, sage accent, otherwise identical

**Don't ship two visually divergent decks. They are the same deck, two themes.**

---

## 12. File structure & deliverables

### 12.1 Working files (Figma)

- `volvence-deck-v11.fig` — single Figma file containing both themes
- Pages structured as:
  - `Cover system` (cover, section dividers, end card)
  - `Body slides — dark theme` (Slides 1-18)
  - `Body slides — light theme` (Slides 1-18, theme-swapped)
  - `Components` (page frame, Observed footer, table styles, type styles)
  - `Reference` (mood board, accepted refusals, anti-examples)

### 12.2 Final deliverables


| Asset                        | Format                 | Use                                                      |
| ---------------------------- | ---------------------- | -------------------------------------------------------- |
| Meeting deck (dark)          | PDF, 1920×1080, vector | Patrick's screen / projector                             |
| Leave-behind (light)         | PDF, 1920×1080, vector | Email after meeting                                      |
| PowerPoint (only on request) | .pptx, 1920×1080       | Only if Patrick asks for editable                        |
| Source                       | .fig                   | Volvence retains; future updates                         |
| Asset library                | tokens.json            | Color + typography tokens for future Volvence collateral |


**File naming:** `volvence-pitch-v11-{dark|light}-{YYYYMMDD}.pdf`

**Total page count target:** 22 frames (1 cover + 4 dividers + 17 body + closing). If page count drifts above 24, the deck has lost discipline — flag for review.

---

## 13. The negative spec (what to refuse)

If anyone (founder, investor, advisor, designer, friend) asks for any of the following — push back. The integrity of the deck depends on it.


| Forbidden                                         | Why                                                             |
| ------------------------------------------------- | --------------------------------------------------------------- |
| Bigger logo                                       | There is no logo. The 4px sage bar is the entire mark.          |
| Photos / illustrations                            | Breaks editorial-thesis register.                               |
| Icons in tables                                   | Temperature tint is the icon.                                   |
| Emoji on slides                                   | Breaks editorial-thesis register; carries 2017 SaaS-deck smell. |
| Colored backgrounds beyond §1 tokens              | Already 3 non-neutral colors at maximum on a page. No more.     |
| Gradient-fill arrows                              | One exception only — see §6.1.                                  |
| Drop shadows / glow / bevel                       | Breaks the editorial flatness.                                  |
| Blue accent                                       | Reads as SaaS / fintech. Sage is intentional.                   |
| Stock photography                                 | Never.                                                          |
| QR codes                                          | Never (anywhere).                                               |
| 3D charts                                         | Never.                                                          |
| "Animated infographics"                           | Already covered. The answer is no.                              |
| Multiple typefaces beyond the two-family system   | Two families: serif body + sans/mono UI. That is it.            |
| Marketing tagline beneath the deck title on cover | The two phrases ARE the marketing.                              |
| Centered text on body slides                      | Body is left-aligned. Centered = corporate.                     |
| Heavy/Black/Bold weight in body                   | Use italic + sage color for emphasis.                           |
| Pure black `#000000`                              | Use `#0B0E13`.                                                  |
| Pure white `#FFFFFF`                              | Use `#E8E6E1`.                                                  |


---

## 14. Acceptance criteria

The designer's work is approved when, holding the printed PDF or projected slide at arms' length:

1. **The cover slide produces a 3-second pause.** A reader's first instinct is to read both lines slowly, not skim.
2. **Slide 4 (flywheel) reads correctly without any verbal explanation.** A diligent reader can parse the eight stages, the irreversibility table, and the parenting tagline from typography alone.
3. **Slide 7 (memory→governance) makes the upper half feel deliberately faded.** The visual concession in the top is unmistakable.
4. **Slide 12 (Phase 1→4) makes Phase 1 visually heavier than Phase 4 without verbal explanation.** Column-width asymmetry + sage tint succeeds.
5. **Slide 15 (uncertainty boundary) lets the reader feel temperature drop left-to-right.** Cool → warm → neutral, no icons needed.
6. **The four "Observed" footers all read as marginalia.** None of them read as competing punchlines.
7. **No reader, on first pass, would describe the deck as "a startup pitch."** They would describe it as "a thesis paper" or "an essay" or "a research brief."
8. **Light-mode and dark-mode versions are visually identical in structure.** Theme swap only.
9. **The page count is between 18 and 22.** Anything above 24 is a discipline failure.
10. **A Bloomberg Businessweek art director would not be embarrassed to have made it.**

---

## 15. Designer onboarding sequence (1 week)

**Day 1**: Read this brief + read the v11 deck markdown. Pull all reference images (Stripe Press, Bloomberg, Anthropic PDF, Tufte). Build mood board in Figma reference page.

**Day 2**: Build the design token system (§1) and typography styles (§2) as Figma variables and shared text styles.

**Day 3**: Build the page-frame component, "Observed" component, section divider component, table component. Confirm with founder.

**Day 4-5**: Build Slides 1, 4, 7, 12, 15 (the must-land set) in dark theme. First review with founder.

**Day 6**: Build the remaining 13 body slides + 3 section dividers. Second review.

**Day 7**: Theme-swap to light. Final review, export PDFs, deliver.

If a step takes longer than the estimate, the bottleneck is almost always Slide 4 (flywheel diagram) or Slide 15 (three-temperature columns). Spend the time. Those are the pages that make the meeting.

---

## 16. Brief summary for the designer

> *Build this in Figma in dark mode. Two type families: serif for body, sans/mono for UI and numbers. One sage-green accent, one muted amber for risk. Hairline rules. 12-column grid. Four must-land slides — the flywheel, memory-vs-governance, phase-1-to-4, and uncertainty boundary — get disproportionate attention. Every body slide ends with a small italic "Observed" footnote that should read as marginalia from a published essay. No icons, no photos, no gradients, no SmartArt. The visual restraint is the message. If it ever looks like a startup pitch, you have over-designed it.*

---

## Change log

- **2026-05-17 v1.0**: Initial designer brief. Companion to `xfund-pitch-deck-v11-zhaojiangbo.md` v11 deck content. Covers color tokens, typography system, grid, six page templates, four must-land slide wireframes, "Observed" footer spec, section dividers, animation rules, light-theme variant, deliverables, negative spec, acceptance criteria, and onboarding sequence.

