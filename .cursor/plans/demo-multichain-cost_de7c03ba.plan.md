---
name: demo-multichain-cost
overview: 把并排演示器从单链升级为多链（有界并发 + 断点续跑），用现有正式产物免费产出诚实的成本曲线图并修正 MaaS 稿口径，最后跑一次 8 链录制。
todos:
  - id: fig09
    content: 在 scripts/lab_figures/coding.py 加 figure_cost_vs_session_length()：聚合累计比值主线 + 8 条链散布细线 + 交叉点/中位数/5-of-8 标注；在 render_lab_evidence_figures.py 注册为图 09 并渲染检查
    status: completed
  - id: cost-caveat
    content: MaaS 定位稿 P3/P4/P11/P13 各加方差警告句（保留 53.7%），business 索引「成本口径统一规定」同步，并登记 fig09
    status: completed
  - id: multichain
    content: 演示器加 --chains N、asyncio.Semaphore 有界并发（--max-concurrency 默认 6）、每 (chain,arm) 断点续跑与 --resume
    status: completed
  - id: board-fractions
    content: 实时看板与汇总：add_helper 曲线按次序位聚合成分数；链数 < 3 时不打印通过率与总 token
    status: completed
  - id: demo-figure
    content: 从演示 report 生成同视觉语言的曲线图，输出到演示目录并标注 live demo / not evidence
    status: completed
  - id: smoke2
    content: 先跑 --chains 2 验证并发、续跑与看板分数显示
    status: completed
  - id: record8
    content: 跑 --hand api --chains 8 --episodes 10 --max-concurrency 6 录制素材（约 1,680 万 token，25–35 分钟）
    status: completed
isProject: false
---

和你原本设想的两处偏差，先说清楚：

- **成本曲线不花钱**：[coding_lab_packet2_formal_v2 报告](artifacts/coding_lab/coding_lab_packet2_formal_v2_qwen3codernext_20260813/report.json) 已有 8 链 × 10 回合逐回合 token，比新跑一条噪声链更有信息量。
- **"不用短会话"= 加链数，不是加回合数**：`_helper_variants` 只有 4 个变体且 `helper_pool.pop()`，所以**一条链最多 4 次可学机会**，20 回合只会多出一堆 `fix_bug`。

# 多链演示器 + 诚实成本曲线

## 已确认的关键事实

- [scripts/run_coding_lab_packet2.py](scripts/run_coding_lab_packet2.py) 正式路径是**全串行**（`for chain / for arm` 逐个 await）并带**断点续跑**（每 arm/chain 一份 `rows.json`）。240 回合串行约 160 分钟，所以多链必须并发 + 可续跑。
- 成本方差极大。8 链在 10 回合的 brain/steelman 比值：`0.17 0.42 0.49 0.60 0.95 1.07 1.21 2.67`，中位数 **0.776**，仅 **5/8** 条链 brain 更省。聚合 0.537 是被少数 steelman 爆掉的链拉出来的。**方差源自 agent 步数，不是记忆机制**——载荷比 882/9,027 才是低方差事实。
- 53.7% 只出现在 [MaaS 定位稿](docs/business/BP/volvence-maas-positioning-v0.1-cn.md) 与 [business 索引](docs/business/00_INDEX.md)；一页摘要、图说版、非技术说明引用的都是载荷比，无需改。

## 一 · 免费的先做：成本曲线图 + 口径修正

新增 `fig09-coding-cost-vs-session-length`，数据全部来自已有正式产物，**零 API 花费**：

- 主线：8 链聚合累计 brain/steelman 比值随回合数（第 1 回合 1.236 → 第 2 回合 0.987 交叉 → 第 10 回合 0.537）
- 底层：8 条细线画**每条链**的同一比值，把 0.17–2.67 的真实散布画出来
- 标注：交叉点、中位数 0.776、`5/8 条链 brain 更省`

在 [scripts/lab_figures/coding.py](scripts/lab_figures/coding.py) 加 `figure_cost_vs_session_length()`，在 [scripts/render_lab_evidence_figures.py](scripts/render_lab_evidence_figures.py) 的 `FIGURES` 注册 `09`，边车 JSON 记录每链比值与交叉点。

口径修正按你选的**最小改动**（保留 53.7%，加方差警告）：

- MaaS 稿 P3 主张一那格、P4 成本页、P11 单位经济、P13 尽调问答各加一句：`聚合口径 0.537 是我们的 COGS；单会话中位数 0.78，8 条链中 3 条更贵（最差 2.67 倍），方差来自 agent 步数而非记忆机制`
- 索引「成本口径统一规定」同步加该句
- 索引图集条目登记 fig09

## 二 · 演示器升级为多链

改 [scripts/run_investor_side_by_side_demo.py](scripts/run_investor_side_by_side_demo.py)：

- `--chains N`（跑链 0..N-1）替代单一 `--chain-index`
- **有界并发**：`--max-concurrency` 默认 6，用 `asyncio.Semaphore` 包住每个 (chain, arm) 单元。24 路并发未验证过，可能撞 429；手内已有 4 次指数退避重试
- **断点续跑**：照正式路径写每单元 `rows.json`，`--resume` 复用。30 分钟量级的跑必须能续
- 实时看板把 `add_helper` 曲线**按次序位聚合成分数**（`#2: 3/8 违反`），而不是单链的二元违反/遵守
- 汇总输出标注哪些指标在当前链数下才稳定：链数 < 3 时不打印通过率与总 token

## 三 · 演示产出自己的图

从演示 report 生成一张与 fig01 同视觉语言的曲线图，写进**演示输出目录**（不进 `docs/business/BP/figures`），文件名与标题都带 `live demo · not evidence`，避免与证据图混淆。

## 四 · 跑 8 链录制

`--hand api --chains 8 --episodes 10 --max-concurrency 6`

- 预计 **约 1,680 万 prompt token**（按正式数据每链 210 万）
- 预计墙上时间 **25–35 分钟**（240 个 arm-episode，并发 6）
- env_seed 保持 20260827（与正式运行的 20260812 区分，明确是新的演示运行）
- 先用 `--chains 2` 验证并发与续跑通路，再放到 8

## 边界

演示 report 的 `evidence_tier` 保持 `live_demo_api_hand`，`claim_boundary` 保持"不得当门结果引用"。8 链 10 回合虽与正式运行同规模，但**无预注册、单次运行**，正式结论仍只引 `coding_lab_packet2_formal_v2_qwen3codernext_20260813`。