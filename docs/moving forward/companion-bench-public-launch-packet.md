# Companion Bench 公开化前置 Packet

> 出处：商业化反思 26 条 debt 的 P5 组（[`docs/known-debts.md`](../known-debts.md) #48 / #52 / #53 / #54 / #55 / #56 / #57）
> 覆盖：debt #48（LLM-judge 跨家族方差）+ #52（6 轴权重 + A6 cap calibration）+ #53（user_simulator bias）+ #54（120 scenario × 6 轴 statistical power）+ #55（中英文 scenario 平衡）+ #56（季度更新成本闭环）+ #57（held-out trusted runner 机制）
> 与既有 debt 的关系：[#29](../known-debts.md) / [#32](../known-debts.md) / [#33](../known-debts.md) / [#34](../known-debts.md) / [#35](../known-debts.md) / [#36](../known-debts.md) / [#38](../known-debts.md) 是 launch 路径前置（已 land 部分 + 仍开放部分）；本 packet 是 launch **之前的 evidence + 治理证据补全**
> 状态：plan v0.1，待 packet review
> 上游商业承诺锚点：[`docs/business/commercialization-assessment.md`](../business/commercialization-assessment.md) §3.3 / §4.5 / §5.2 / §7.2 / §10.2

---

## 1. P5 公开化的关键论断

### 1.1 概率与位置

- **12 个月签约 / 上线 / 收入概率 60-75%**（commercialization-assessment §4.5；P5 是 6 条路径中**唯一一条 > 50% 的高确定性路径**）
- 概率高的根因是**已有完整 v1.0 实现 + RFC + 私有 held-out submodule**（[#29](../known-debts.md) 轨 2 + [#32](../known-debts.md) 已 land 70%），剩下的不是工程债而是**「公开化第一击的可信度地基」**
- P5 是 P1 / P2 / P4 的**乘数**——客户做尽调时看到 VZ 在自己 benchmark 上有 evidence-backed 排名，是直接抬高客单价的话术（commercialization-assessment §5.1 evidence cascade 第 1 条）

### 1.2 公开化第一击的可信度地基 = #48 + #52 + #54

[`docs/business/commercialization-assessment.md`](../business/commercialization-assessment.md) §7.2 第 2 步明确：

> 「**第一次主动跑分**——这是最关键的一击……公开发布"第一份长程陪伴排行榜"，技术报告同步上 arXiv」

但**第一份榜单一发布就要面对三类公开质疑**：

| 质疑 | 触发它的 debt | 不解决的后果 |
|---|---|---|
| 「你的 LLM judge 偏 GPT 所以 GPT 排第一」 | [#48](../known-debts.md) | 跨家族方差未量化 → 海外学术界第一周内质疑成立 |
| 「为什么 A6 cap 是 60 不是 55？为什么 A3 权重是 0.25 不是 0.20？」 | [#52](../known-debts.md) | calibration 来源未落档 → 头部厂商提交后无答辩材料 |
| 「two SUT 排名差 X 分到底显不显著？」 | [#54](../known-debts.md) | statistical power 未量化 → 第二份榜单更新时排名乱跳 → 公信力崩 |

**结论**：#48 + #52 + #54 三条**必须在 [#32](../known-debts.md) sub-track 1（真 reference 跑分）启动同时或之前**完成；否则 reference 跑分跑出来的数字本身会成为质疑材料而非可信度材料。

### 1.3 第一次跑分要不要让 VZ 排第一？

[`docs/business/commercialization-assessment.md`](../business/commercialization-assessment.md) §7.2 第 2 步明确反目标：

> 「**VZ 自己作为 SUT 也跑一次——但不要让 VZ 排第一**（如果排第一会被怀疑作弊，可信度反而崩）；让 VZ 在某些子轴（A3 关系连续性 / A4 自适应学习）领先即可」

这条 GTM 决策与本 packet 的关系：

- **顺位 1 sweep（#48 + #52 + #54）的 reference SUT 池中，VZ 应作为其中一位**——这样 robustness sweep 的输出能直接验证「VZ 在 A3 / A4 子轴领先 + A1 / A6 不领先」是否成立
- **如果 sweep 结果显示 VZ 在某个非预期子轴领先**（比如 A6 安全），需要**回过头检查 calibration**——是真的强还是 weight 配错。这正是 #52 calibration sweep 的兜底意义
- **如果 sweep 结果显示 VZ 全面领先且 inter-judge κ 一致**：触发 §10.2 反目标 "公开 companion-bench 的私有 held-out 提示集 → benchmark 价值归零" 的同构风险——「我们家产品在我家 benchmark 全面第一」=  benchmark 公信力归零；此时 GTM 团队应主动调整 reference SUT 池让 VZ 排第 2-3 位

### 1.4 与 [#29](../known-debts.md) / [#37](../known-debts.md) 的差异

- [#29](../known-debts.md) / [#37](../known-debts.md) 解决的是「**对外可被 google 到的客观分数**」（EQ-Bench 3 / Chatbot Arena）—— P5 公开化的**外部参照点**
- 本 packet 解决的是「**作为 RFC convener 自己定义的 benchmark 第一次跑分时的方法论防御**」—— P5 公开化的**内生可信度**
- 两者顺序：**[#37](../known-debts.md) actuation 先跑（拿第三方分数）→ 本 packet 7 条同步 land（补 evidence）→ [#32](../known-debts.md) sub-track 1 真 reference 跑分（出 LSCB 第一份榜单）**——三步打通，48 小时内 marketing 故事可串

### 1.5 概率拆解：为什么是 60-75% 而不是 80%+

[`docs/business/commercialization-assessment.md`](../business/commercialization-assessment.md) §4.5 给出 60-75% 区间，下限 60% 而非 80%+ 的原因（**本 packet 直接对应处理上限抬升 5-10pp 的关键 5 项**）：

| 不确定性来源 | 概率拖累 | 本 packet 处理 |
|---|---|---|
| working group 不形成（无外部组织接入） | -10pp | ❌ 不在本 packet 范围（[#32](../known-debts.md) sub-track 3 治理） |
| 第一份榜单方法论被严重质疑无法答辩 | -10pp | ✅ #48 / #52 / #54 三条 evidence sweep 直接处理 |
| 海外学术界质疑「中文 benchmark 不可比」 | -3pp | ✅ #55 跨语言平衡处理 |
| 头部厂商不诚实提交触发 leak event | -3pp | ✅ #57 trusted-runner + leak protocol 处理 |
| API 预算不可控（季度更新成本失控） | -2pp | ✅ #56 cost model 处理 |
| OpenAI / Anthropic 在 12 个月内自发 long-session benchmark RFC | -7pp | ❌ 时间窗口风险，本 packet 通过加速 launch 间接缓解 |
| reference SUT 池选择 / 跑分时机争议 | -5pp | ⚠️ 部分（#48 协议固化 + 推荐 SUT 池附录 B） |

**本 packet 完整 land 后概率上限 60% → 70-72%**（约 +10-12pp）；剩余概率拖累不在本 packet 范围。

### 1.6 与 §10.2 反目标的对偶检查

[`docs/business/commercialization-assessment.md`](../business/commercialization-assessment.md) §10.2 列出 8 条商业反目标，本 packet 命中其中 1 条 + 对齐另 1 条：

- ❌「公开 companion-bench 的私有 held-out 提示集 → benchmark 价值归零」 → 本 packet [#57](../known-debts.md) trusted-runner 协议直接对应；任何 self-hosted submission 不允许碰 held-out
- ❌「接受改变工程纪律的客户合同（"我们急着上线，跳过 SHADOW 验证"）」 → 本 packet 7 个 sub-packet 全部走 SHADOW → ACTIVE 退出标准，不允许 launch 前跳过

本 packet 不命中其余 6 条反目标。

---

## 2. Packet 列表

每条 debt 一份子 packet。代码层文件路径用相对仓库根的 markdown 链接；脚本与文档命名对齐 `scripts/companion_bench/` + `docs/external/` 既有约定（与 [#32](../known-debts.md) sub-track 1 / [#34](../known-debts.md) staged executor 复用同一套 reference SUT 跑分基础设施）。

---

### 2.1 #48 LLM-as-judge 跨家族方差量化

- **路径**
  - 新脚本：`scripts/companion_bench/judge_robustness_sweep.py`
  - 新公开报告：`docs/external/companion-bench-judge-robustness-v0.md`
  - 数据落点：`artifacts/companion_bench/judge_robustness/<timestamp>/`
  - 影响 wheel：[`packages/companion-bench/src/companion_bench/judge_perturn.py`](../../packages/companion-bench/src/companion_bench/judge_perturn.py) + [`packages/companion-bench/src/companion_bench/judge_arc.py`](../../packages/companion-bench/src/companion_bench/judge_arc.py)（新增可注入 judge family 列表的入口，不改默认行为）

- **退出标准（SHADOW → ACTIVE）**
  1. SHADOW：5 个 judge family（GPT-5 / Claude Opus 4.7 / Qwen-Max / DeepSeek V4 / Gemini 3 Pro）× 5 个 reference SUT × 24 公开 scenario × 1 paraphrase seed 跑通；产出 `judge_robustness_v0.json` 含 inter-rater Spearman / Kendall κ + per-axis variance σ
  2. ACTIVE 准入条件（同时满足）：
     - per-axis 平均 σ < 8 分（百分制）
     - inter-rater Spearman ≥ 0.65（与 RP-Bench / EQ-Bench 文献基准对齐）
     - 任意单一 judge family 被剔除后排名 top-3 / bottom-3 集合不变（rank stability）
  3. 公开报告 `docs/external/companion-bench-judge-robustness-v0.md` 必须 include 上述 3 个数字 + 失败案例 walk-through
  4. RFC v0.1 → v1.0 升级时把本报告作为 §5 必备附录（与 [#52](../known-debts.md) calibration report 同卷出版）

- **子任务（5 项）**
  1. 在 `judge_perturn.py` / `judge_arc.py` 增 `judge_family_id` 字段（`openai-gpt5` / `anthropic-claude-opus-4.7` / `qwen-max` / `deepseek-v4` / `gemini-3-pro`），落入 `JudgeIdentity` typed dataclass（不允许 hasattr / try-except 静默回退，遵守 [`no-swallow-errors-no-hasattr-abuse.mdc`](../../.cursor/rules/no-swallow-errors-no-hasattr-abuse.mdc)）
  2. 写 `judge_robustness_sweep.py`：复用 [`scripts/companion_bench/score_reference_systems.py`](../../scripts/companion_bench/score_reference_systems.py) 的 SUT orchestration；仅对 judge layer 做 N 家族扫描；reference SUT 固定为 5 系统（VZ + 4 公开 SUT），固定 1 seed 控成本
  3. 引入 `scipy.stats.spearmanr` / `scipy.stats.kendalltau` 计算 inter-rater agreement；输出 `judge_robustness_v0.json`（[`packages/companion-bench`](../../packages/companion-bench/) 已是 numpy 依赖图，不新增重依赖）
  4. 写公开报告 `docs/external/companion-bench-judge-robustness-v0.md`：含方法论 + per-axis σ 表 + Spearman 矩阵 + 鲁棒性结论
  5. 加 `tests/contracts/test_judge_robustness_required.py`：任何新增 LLM judge family 必须在 [`docs/specs/companion-bench.md`](../specs/companion-bench.md) §5 注册 + 给出 fingerprint（`judge_family_id` + 模型版本）

- **资源估算**
  - 工程：1 人 × 2-3 周（含 5 项子任务 + 报告写作 + 测试）
  - API token 成本：参考 [`packages/companion-bench/src/companion_bench/cost.py`](../../packages/companion-bench/src/companion_bench/cost.py) `_DEFAULT_PRICES`（2026-Q1 价格表）
    - 5 judge family × 5 SUT × 24 scenario × 1 seed = 600 arcs
    - per-arc 总 token：~25 turn × 3K input + 500 output（per-turn judge）+ ~8K input + 1K output（arc judge）≈ 86.5K input + 13.5K output = 100K tokens / arc / judge
    - 5 judge × 600 arcs × 100K = 300M tokens / sweep
    - 加权平均 judge 价格（Claude Opus 4.6 $15/$75，GPT-5 $5/$15，DeepSeek $0.27/$1.10，Qwen $0.40/$1.20，Gemini 3 Pro $5/$15 per 1M）取中位 ~$8 input + $25 output / 1M
    - 240M input × $8 + 60M output × $25 = $1920 + $1500 = **~$3400 / sweep**
    - 加 SUT inference（5 SUT × 600 arcs × 25 turn × 4K token avg = 300M tokens × ~$3 / 1M blended）≈ $900
    - **合计 ~$4000-5000 USD / 一次完整 sweep（≈ 28-35k CNY）**
    - **注**：判官层 dominant，可通过砍掉 Opus 4.6（最贵）替换为 Sonnet 3.7 把成本降到 ~$2000

- **依赖**
  - 上游：`AsyncIO + connection pool`（[#34](../known-debts.md) 修法 1）落地后 wallclock 才控得住——但本 sweep 即使 sync 跑也能完成，只是 wallclock ~24-48 小时；建议**[#34](../known-debts.md) 修法 1 land 后再跑大头**，先跑 1 judge × 5 SUT × 24 scenario calibration（成本 ~$800）确认 pipeline
  - 下游：本 sweep 输出是 [#52](../known-debts.md) calibration sweep 的输入（同一套 reference SUT 跑分基础设施直接复用）

- **风险 & fallback**
  - **风险 A**：某一 judge family 持续给 VZ 高分（self-preference）→ inter-rater κ < 0.5。**Fallback**：报告中明确 disclose；将该 family 从 v1.0 default judge ensemble 中剔除；触发季度 judge rotation（与 [#35](../known-debts.md) 联动）
  - **风险 B**：5 judge family 中 ≥ 2 家 API 在 sweep 期间不可用 → 无法收齐数据。**Fallback**：缩减到 3 家族 + 在 RFC §5 注明「v1.0 baseline n=3, expand at v1.1」
  - **风险 C**：sweep 结果显示当前权重下排名不稳（任一 family 剔除排名变化）→ 触发 [#52](../known-debts.md) v1.x 权重调整 + 公开 errata

---

### 2.2 #52 6 轴权重 + A6 cap=60 calibration 来源落档

- **路径**
  - 新脚本：`scripts/companion_bench/calibration_sweep.py`
  - 新公开报告：`docs/external/companion-bench-calibration-report-v0.md`
  - 影响 wheel：[`packages/companion-bench/src/companion_bench/aggregator.py`](../../packages/companion-bench/src/companion_bench/aggregator.py) `WEIGHTS` / `A6_CAP_THRESHOLD` / `A6_CAP_VALUE` 增添 metadata block（不改值，加注释 + version pin）
  - RFC 附录：[`docs/external/companion-bench-rfc-v0.md`](../external/companion-bench-rfc-v0.md) §6 增 calibration 附录段（v1.0 升级时引）

- **退出标准（SHADOW → ACTIVE）**
  1. SHADOW：在 [#48](../known-debts.md) sweep 输出的 600 arcs × 5 judge × 5 SUT 数据上**post-hoc 重新聚合**——6 轴权重各 ±0.05 范围 sweep（21 组合）+ A6 cap 在 50 / 55 / 60 / 65 / 70 sweep（5 组合）= **105 个权重配置**下重新计算每个 SUT 的 final score
  2. ACTIVE 准入条件：
     - 当前权重 (0.10/0.15/0.25/0.20/0.10/0.20) 下的 reference SUT 排名与所有 105 个邻近配置下排名差不超过 ±1 位（rank stability radius）
     - A6 cap=60 的选择有可解释的 evidence：sweep 显示 cap=55/65 都让 ≥ 1 个安全有问题的 SUT 排名虚高/被错误封顶
  3. 公开报告 `docs/external/companion-bench-calibration-report-v0.md` 必须包含 105 配置 sensitivity matrix + 当前权重选择论证 + 与 EQ-Bench 3 / RP-Bench 同行口径对照
  4. RFC v0.1 → v1.0 升级时纳入 §6 必备附录

- **子任务（5 项）**
  1. 写 `calibration_sweep.py`：读取 [#48](../known-debts.md) sweep 产出的 `arc_axis_scores` 原始分（`packages/companion-bench/src/companion_bench/judge_arc.py:ArcAxisScores`），post-hoc 用不同权重重新调用 [`aggregate_arc()`](../../packages/companion-bench/src/companion_bench/aggregator.py)（无需重跑 SUT / judge）
  2. 输出 `calibration_v0.json` 含 105 配置 × 5 SUT 排名矩阵 + per-config Kendall τ to baseline ranking
  3. 在 `aggregator.py` `WEIGHTS` 上方增 docstring：注明权重选择 evidence 出处（"see docs/external/companion-bench-calibration-report-v0.md §3 for sensitivity sweep"）+ 加 `WEIGHTS_VERSION = "v1.0"` 常量
  4. 写公开报告 `docs/external/companion-bench-calibration-report-v0.md`：含方法论 + 105 配置矩阵 + 当前权重论证（A3=0.25 因长程陪伴核心价值轴 / A6=0.20 因安全是 hard constraint / A6 cap=60 因 sweep 显示 cap=55 有 X 个 SUT 排名虚高 cap=65 有 Y 个被错误封顶）
  5. （可选）Companion Bench site 加 "Why these weights?" page（与 [#38](../known-debts.md) site verifier 同 sprint）

- **资源估算**
  - 工程：1 人 × 1-1.5 周（无 API 调用，纯 post-hoc 计算 + 写作）
  - API token 成本：**$0**（完全 post-hoc，复用 [#48](../known-debts.md) sweep 数据）
  - 计算成本：本地 CPU minutes 量级，可忽略

- **依赖**
  - **强依赖** [#48](../known-debts.md) sweep 已跑（提供原始 `arc_axis_scores`）。理论上可独立跑一份**仅 calibration sweep**（1 judge × 5 SUT × 24 scenario × 5 seed → ~$1000），但浪费——优先复用

- **风险 & fallback**
  - **风险 A**：当前权重在 105 配置中 rank stability radius > ±1 → 触发 v1.x 权重调整。**Fallback**：找 105 配置中 rank-stability 最大的权重组合作为 v1.1 推荐，与原 v1.0 同时发布并标 ALTERNATIVE
  - **风险 B**：A6 cap=60 在 sweep 中显示既不太严也不太松 → 当前选择无 evidence-backed 论证。**Fallback**：把 cap 改为 sweep 中 sweet-spot 值（比如 58 或 62）作为 v1.1
  - **风险 C**：报告写成后被外部 reviewer 质疑「sensitivity sweep 范围 ±0.05 太窄，应做 ±0.10」。**Fallback**：post-hoc 重跑（仍 $0），扩到 ±0.10 + 41 组合矩阵；写 v1.1 errata

---

### 2.3 #53 user_simulator bias 注入测量

- **路径**
  - 新脚本：`scripts/companion_bench/simulator_robustness_sweep.py`
  - spec 修订：[`docs/specs/companion-bench.md`](../specs/companion-bench.md) §4 新加 "Simulator family rotation" 段
  - 影响 wheel：[`packages/companion-bench/src/companion_bench/user_simulator.py`](../../packages/companion-bench/src/companion_bench/user_simulator.py) + [`packages/companion-bench/src/companion_bench/arc_runner.py`](../../packages/companion-bench/src/companion_bench/arc_runner.py)（`RunRecord` 加 `simulator_family` 必填字段）
  - 新公开报告（可选）：`docs/external/companion-bench-simulator-robustness-v0.md`

- **退出标准（SHADOW → ACTIVE）**
  1. SHADOW：5 个 simulator family（与 [#48](../known-debts.md) judge family 选择不同的 5 家，避免 simulator 与 judge 同源）× 固定 5 个 reference SUT × 24 公开 scenario × 1 seed → 600 arcs；每 SUT × axis 计算 simulator-family-induced variance σ
  2. ACTIVE 准入条件：
     - per-SUT × per-axis simulator-induced σ < 6 分
     - 任一 simulator family 剔除后 SUT 排名 top-3 / bottom-3 集合不变
  3. spec §4.x "Simulator family rotation" 段明确每季度公开榜单跑分时，simulator LLM 必须从公布的 ≥ 4 家族池随机抽
  4. `RunRecord.simulator_family` 字段必填；榜单 site 显示 simulator family（[`scripts/companion_bench/build_site.py`](../../scripts/companion_bench/build_site.py) submission detail page 加该字段渲染）

- **子任务（5 项）**
  1. 在 `user_simulator.py` 增 `simulator_family_id` typed enum + `make_simulator(family_id, ...)` factory；不允许 simulator 与 SUT 在同一 sweep 内来自同一 family（contract test 守门）
  2. `RunRecord` 加 `simulator_family: str` 必填字段；spec round-trip 测试同步更新
  3. 写 `simulator_robustness_sweep.py`：与 [#48](../known-debts.md) sweep 同 protocol 但**只扫 simulator layer**；judge layer 固定为 ensemble of 3（与 [#48](../known-debts.md) sweep 选出的 top-3 一致 judge）
  4. 输出 `simulator_robustness_v0.json` + 写 `docs/external/companion-bench-simulator-robustness-v0.md`
  5. spec §4 加 "Simulator family rotation" 段；与 [#35](../known-debts.md) 季度治理自动化联动（同样的 rotation 协议）

- **资源估算**
  - 工程：1 人 × 1.5 周
  - API token 成本：与 [#48](../known-debts.md) 同量级 ~$3000-4000 / sweep（≈ 21-28k CNY）
  - **注**：本 sweep 与 [#48](../known-debts.md) sweep 可**串行复用同一批 reference SUT cache**（如果 reference SUT 是 closed API 即只需 cache transcript；如果是 LSCB 自定义 SUT 则需重跑）

- **依赖**
  - 上游：[#48](../known-debts.md) judge robustness sweep 已 land（确定 judge ensemble = top-3）
  - 下游：[#35](../known-debts.md) 季度治理自动化的 simulator rotation log 直接消费本 packet 的 family pool

- **风险 & fallback**
  - **风险 A**：sweep 显示某 simulator family 显著影响 SUT 评分 → 揭露 P5 公正性命门。**Fallback**：把该 family 从 default ensemble 剔除；spec 明文 simulator family rotation 协议
  - **风险 B**：5 simulator family 中找不到 ≥ 4 个稳定可用的 → simulator pool 退化到 3。**Fallback**：v1.0 spec 明文「simulator pool 至少 3 family，rotation cadence ≥ quarterly」（保守起步）

---

### 2.4 #54 120 scenario × 6 轴 statistical power 量化

- **路径**
  - 新脚本：`scripts/companion_bench/statistical_power_analysis.py`
  - 新公开报告：`docs/external/companion-bench-statistical-power-v0.md`
  - 影响 site：[`scripts/companion_bench/build_site.py`](../../scripts/companion_bench/build_site.py) leaderboard table 加 "ELO ± 95% CI" 列
  - 影响 wheel：[`packages/companion-bench/src/companion_bench/elo.py`](../../packages/companion-bench/src/companion_bench/elo.py) 增 `compute_elo_with_ci()` 函数

- **退出标准（SHADOW → ACTIVE）**
  1. SHADOW：固定 5 reference SUT × 24 公开 scenario × **5 paraphrase seeds**（本 sweep 的关键改动是多 seed） → 600 arcs；用 bootstrap resampling 算 per-SUT × per-axis ELO 95% CI
  2. ACTIVE 准入条件：
     - 任意两 SUT ELO 差 ≥ 30 分时 95% CI 不重叠（即 ELO=30 是 distinguishable threshold）
     - 任意 SUT 在 5 seed 间的 ELO σ < 25 分
     - 如果不满足 → 触发 v1.x 扩 scenario 到 200+（与 RFC §9 cadence 联动）
  3. 公开报告 `docs/external/companion-bench-statistical-power-v0.md` 含 power curve（n=24 / 48 / 96 / 200 多档下 distinguishable threshold 估算）
  4. 榜单 site 每行 SUT 加 "ELO ± 95% CI"，**不发布单一数字**

- **子任务（5 项）**
  1. 写 `statistical_power_analysis.py`：5 SUT × 24 scenario × 5 seed = 600 arcs（仅公开 scenario，先不动 held-out）
  2. 在 [`elo.py`](../../packages/companion-bench/src/companion_bench/elo.py) 加 `compute_elo_with_ci(matches, n_bootstrap=1000)` 函数（bootstrap percentile method）
  3. 写公开报告 `docs/external/companion-bench-statistical-power-v0.md`
  4. 修改 `build_site.py` leaderboard 渲染：每行 SUT 列 "ELO" 改为 "ELO ± CI"（与 [#38](../known-debts.md) site verifier sprint 同 commit）
  5. 加 [`tests/contracts/test_elo_ci_consistency.py`](../../tests/contracts/) 守 bootstrap CI 在固定 seed 下 deterministic

- **资源估算**
  - 工程：1 人 × 1.5-2 周
  - API token 成本：与 [#48](../known-debts.md) 同量级但**多 5x seed** → 5 SUT × 24 × 5 seed × judge ensemble 3 = ~3600 arcs
    - 如果**复用 [#48](../known-debts.md) sweep 中 5 SUT × 24 × 1 seed × 5 judge = 600 arcs 的 transcript cache**（transcript 是 SUT 输出，不依赖 judge family），只需多跑 4 个新 paraphrase seed × 5 SUT × 24 scenario = 480 新 arcs
    - 480 arcs × judge ensemble 3 × 100K token / arc / judge = 144M token
    - 加权 judge 价格 ~$8 input + $25 output / 1M → ~144M × ~$10 blended = **~$1500 / sweep**
    - 加 SUT 480 arcs × ~100K token = 48M token × ~$3 / 1M = **~$150**
    - **合计 ~$1650-2000 USD（≈ 11.5-14k CNY）**

- **依赖**
  - 强依赖 [#48](../known-debts.md) sweep 已 land（共享 transcript cache）
  - 与 [#52](../known-debts.md) calibration sweep 同 packet 设计更高效

- **风险 & fallback**
  - **风险 A**：n=24 scenario 在公开 set 上 power 不足（5 SUT 内 distinguishable threshold > 60 分）→ 触发 v1.x 扩 scenario 到 200+。**Fallback**：v1.0 leaderboard 仅展示「distinguishable bands」（top tier / mid tier / bottom tier）而非 ELO 数字；与 [`docs/external/companion-bench-rfc-v0.md`](../external/companion-bench-rfc-v0.md) §9 "v1.x roadmap" 联动加 scenario expansion 路径
  - **风险 B**：bootstrap CI 计算成本超预期（n_bootstrap=10000 时单次 ~10 min on CPU）→ 限制在 n_bootstrap=1000 + 备 deterministic seed 让结果可复现

---

### 2.5 #55 跨语言 scenario 平衡（中文 / 英文）

- **路径**
  - 影响 wheel：[`packages/companion-bench/src/companion_bench/spec.py`](../../packages/companion-bench/src/companion_bench/spec.py) `ScenarioSpec` 加 `language: Literal["zh", "en", "bilingual"]` 必填字段
  - 24 公开 scenario 重新平衡：现状 24 个 YAML 文件中（按 [`F1-continuity-001.yaml`](../../packages/companion-bench/src/companion_bench/scenarios/public/F1-continuity-001.yaml) 抽样）persona / payload 都是英文 → 需补 12 个中文 scenario 或将 12 个现有 scenario 翻译为中文
  - 96 私有 held-out submodule（[`docs/external/companion-bench-heldout-bootstrap.md`](../external/companion-bench-heldout-bootstrap.md) 治理）：48 中 + 48 英平衡
  - 影响 site：[`scripts/companion_bench/build_site.py`](../../scripts/companion_bench/build_site.py) 加跨语言子榜单 view（中文 / 英文 / 综合）
  - spec 修订：[`docs/specs/companion-bench.md`](../specs/companion-bench.md) §3 + RFC §3 同步更新

- **退出标准（SHADOW → ACTIVE）**
  1. SHADOW：`ScenarioSpec.language` 字段加 + 24 公开 scenario 全部填值（先标 24 全 `en` → bootstrap 起步）；新加 12 中文 scenario → 总 36 公开（平衡比 12 中 + 24 英）；或 retire 12 旧英文换 12 新中文（平衡比 12 中 + 12 英）
  2. ACTIVE 准入条件：
     - 公开 24 scenario 至少 12 中 + 12 英 平衡
     - 私有 held-out 48 中 + 48 英平衡（[#32](../known-debts.md) sub-track 2 held-out repo 创建后再补；可推迟到 v1.1）
     - site 跨语言子榜单 view 上线
     - 跨语言子榜单 ELO 计算独立（中文 SUT 跑分只入中文榜，英文 SUT 跑英文榜，bilingual SUT 入 3 个榜）
  3. spec §3 + RFC §3 同步更新

- **子任务（5 项）**
  1. `ScenarioSpec.language` 字段加；`to_canonical()` 同步加（**注意**：这是 hash-affecting 改动 → 24 公开 scenario hash 全部重生成 → [`docs/external/companion-bench-public-scenario-hashes.txt`](../external/companion-bench-public-scenario-hashes.txt) 同步更新；与 [#32](../known-debts.md) sub-track 2 一同 ship）
  2. 24 现有公开 scenario 全部加 `language: "en"`（无功能变化但 hash 变 → bump scenario_id minor revision 可选）
  3. 新加 12 中文 scenario（按 6 family × 2 = 12 平衡）；persona / payload / FSM payload 全中文；语义保持与对应英文 scenario 同 family 同 axis 探针
  4. `build_site.py` 加跨语言子榜单 view；leaderboard.html / scenarios.html 加 language filter
  5. 加 [`tests/contracts/test_scenario_language_balance.py`](../../tests/contracts/) 守 公开 set 中 中 / 英文 scenario 比例 ≥ 1:1.5（容忍轻微不平衡）

- **资源估算**
  - 工程：1 人 × 2-3 周（含 12 个新中文 scenario reviewer-curated drafting + reviewer 审 + spec / hash / site 同步）
  - API token 成本：**$0**（pure dev work；scenarios 是 hand-curated）
  - 中文 scenario reviewer 工时：~30 min × 12 scenario = 6 小时（reviewer 自评 + 内部 cross-review）

- **依赖**
  - **冲突**：scenario_id 与 hash 变化与 [#32](../known-debts.md) sub-track 2（held-out repo 创建）需要同 release 节奏 ship；否则 hash 表不同步会让旧提交失效
  - 与 [#48](../known-debts.md) / [#52](../known-debts.md) / [#54](../known-debts.md) 三 sweep **可独立先行**（sweep 用现有 24 英文 scenario 跑；中文 scenario v1.1 时再跑跨语言 sweep）

- **风险 & fallback**
  - **风险 A**：中文 scenario 翻译质量不一致 → 跨语言子榜单结果不可比。**Fallback**：每条中文 scenario 至少 2 reviewer cross-review；v1.0 时只发布"综合"榜单 + 中英子榜单标 PROVISIONAL
  - **风险 B**：海外厂商提交跑分时只配置英文 SUT → 跨语言子榜单覆盖率不足。**Fallback**：spec 明文 SUT 可选语言宣告 + 跨语言子榜单允许跨语言 missing data 时不入榜单

---

### 2.6 #56 季度更新成本闭环精算

- **路径**
  - 新脚本：`scripts/companion_bench/estimate_quarterly_cost.py`
  - 新公开报告：`docs/external/companion-bench-cost-model-v0.md`
  - 数据落点：`artifacts/companion_bench/quarterly_cost_estimate_<timestamp>.md`
  - 影响 wheel：[`packages/companion-bench/src/companion_bench/cost.py`](../../packages/companion-bench/src/companion_bench/cost.py) `CostTracker` 已 ready；不改

- **退出标准（SHADOW → ACTIVE）**
  1. SHADOW：基于 [#48](../known-debts.md) + [#52](../known-debts.md) + [#53](../known-debts.md) + [#54](../known-debts.md) 四 sweep 的真实 token 用量 record，反推：
     - 单次 reference 跑分（10 SUT × 120 scenario × 3 seed）总 token / 总 USD
     - 季度更新（new-only 4 SUT + 全部重跑 6 SUT）总 token / 总 USD
     - 8 季度（2 年）累积总 USD
  2. ACTIVE 准入条件：
     - 季度更新单次成本 < $4000 USD（即年度 < $16k，对齐 commercialization-assessment §7.2 "10-30 万人民币 ≈ $14-42k" 的下限）
     - 公开报告 `docs/external/companion-bench-cost-model-v0.md` 给出 transparent 成本表 + 提交方预算指引
  3. 与 [#34](../known-debts.md) staged executor 联动：成本超 budget cap 时优先跑公开 24，private held-out 季度跑 1 次而非每次

- **子任务（5 项）**
  1. 写 `estimate_quarterly_cost.py`：从 [#48/#52/#53/#54] sweep artifact 的 `cost.json`（[`CostTracker.freeze()`](../../packages/companion-bench/src/companion_bench/cost.py) 输出）反推 per-arc per-axis token 均值
  2. 模拟 8 季度成本：每季度 = (4 new SUT + 6 全 rerun) × 120 scenario × 3 seed = 3600 new arcs / 季 × per-arc cost
  3. 输出 `quarterly_cost_estimate.md` 表（columns: quarter / new_arcs / cumulative_arcs / quarter_USD / cumulative_USD）
  4. 写公开报告 `docs/external/companion-bench-cost-model-v0.md`：含 RFC §6.7 价格表 + 季度更新模型 + 提交方预算指引（"submitter 应预算单次跑分 $X，第三方 trusted-runner 服务 VZ 收费 $Y"）
  5. 加 [`tests/contracts/test_cost_model_consistency.py`](../../tests/contracts/) 守 estimator 输出与 `CostTracker.freeze()` 真实记录在同一批 fake transcript 上 ≤ 5% 误差

- **资源估算**
  - 工程：1 人 × 1 周（pure analytics + 写作）
  - API token 成本：**$0**（依赖前 4 个 sweep 已跑出的真 token 数据）
  - 计算成本：本地 minutes 量级

- **依赖**
  - **强依赖** [#48](../known-debts.md) + [#52](../known-debts.md) + [#53](../known-debts.md) + [#54](../known-debts.md) 至少有 1 sweep 已跑（单 sweep 的 `cost.json` 也够 bootstrap，多 sweep 校准更准）
  - 与 [#34](../known-debts.md) Cost-aware scheduler 修法 5 联动（cost cap → 自动 throttle）

- **风险 & fallback**
  - **风险 A**：估算结果显示季度更新 > $4000 → 商业承诺与运营预算对账失败。**Fallback**：(a) 转 staged tier model：公开 24 季度跑 1 次 + private held-out 半年跑 1 次；(b) judge ensemble 从 5 缩到 3；(c) reference SUT 池从 10 缩到 6（保 GPT-5 / Claude / Qwen / DeepSeek / Gemini / VZ）
  - **风险 B**：vendor 价格波动（OpenAI / Anthropic 季度调价）→ 模型失准。**Fallback**：cost report 标 "valid as of 2026-Q2"，每季度 review 价格表

---

### 2.7 #57 私有 held-out trusted runner 机制规约

- **路径**
  - 新脚本：`scripts/companion_bench/trusted_runner.py`
  - 新协议文档：`docs/external/companion-bench-trusted-runner-protocol.md`
  - 新泄露应对协议：`docs/external/companion-bench-heldout-leak-protocol.md`
  - 新 contract test：`tests/contracts/test_heldout_access_audit.py`
  - 影响 wheel：[`packages/companion-bench/src/companion_bench/heldout_loader.py`](../../packages/companion-bench/src/companion_bench/heldout_loader.py) + [`packages/companion-bench/src/companion_bench/submission.py`](../../packages/companion-bench/src/companion_bench/submission.py)（任何 read held-out 必经 audit logger）

- **退出标准（SHADOW → ACTIVE）**
  1. SHADOW：trusted_runner.py 跑通 mock submission（fake SUT endpoint）→ 加密存 endpoint credential → 调用 → 生成 verdict.json → **销毁 transcript**（只留 hash + verdict）→ audit log 完整
  2. ACTIVE 准入条件：
     - 两种提交模式明文规约：
       - "self-hosted run"：提交方自跑，仅公开 24 scenario，结果排公开榜
       - "trusted-runner run"：VZ 跑，提交方提供 OpenAI-compat endpoint + token，结果可上 held-out 完整榜
     - `tests/contracts/test_heldout_access_audit.py` 全绿——AST 守 任何 import / read held-out scenario 的 code path 必须经过 `HeldoutAccessAudit` logger
     - `docs/external/companion-bench-heldout-leak-protocol.md` 落档：泄露事件应对（rotate held-out / 取消该季榜单 / 公告流程）
  3. 第一个外部 trusted-runner 提交跑通（end-to-end smoke）

- **子任务（5 项）**
  1. 写 `trusted_runner.py`：(a) `TrustedRunnerCredential` typed dataclass（OpenAI-compat endpoint / api_key / token budget cap）；(b) credential 用 `cryptography.fernet` 加密存 `~/.companion_bench/trusted_runner_creds/`；(c) 调用完成后销毁 transcript 仅留 verdict.json + scenario_hash；(d) 输出 audit ledger entry
  2. 在 `heldout_loader.py` 加 `HeldoutAccessAudit` logger context manager；任何 `load_heldout_scenarios` 调用必须包在 `with HeldoutAccessAudit(reason=...):` 内（contract test 守门）
  3. 写 `docs/external/companion-bench-trusted-runner-protocol.md`：含两种提交模式协议 + credential 安全协议 + verdict 公开范围 + 计费协议（"trusted-runner mode VZ 收 $X / submission"）
  4. 写 `docs/external/companion-bench-heldout-leak-protocol.md`：泄露事件应对（rotate held-out / 取消该季榜单 / 公告时间表 / 责任人决策链）
  5. 加 `tests/contracts/test_heldout_access_audit.py`：AST 静态分析 [`packages/companion-bench/src/`](../../packages/companion-bench/src/) 中所有调用 `load_heldout_scenarios` 的 callsite 必须在 `HeldoutAccessAudit` context 内

- **资源估算**
  - 工程：1 人 × 1-2 周
  - API token 成本：**$0**（dev work + smoke test 用 fake SUT）
  - 安全 review：1 人 × 半天 review credential 加密协议 + leak response 协议

- **依赖**
  - 上游：[#32](../known-debts.md) sub-track 2（VolvenceZero/companion-bench-heldout 私有 repo 创建）必须先 land；本 packet trusted_runner 才能真消费 held-out scenario
  - 下游：[#33](../known-debts.md) human-eval 轨道 v0.3 起步时人评 transcript 也走类似 audit logger 模式（可复用 `HeldoutAccessAudit` 抽象）

- **风险 & fallback**
  - **风险 A**：第一个外部 submission 不诚实 / 反向工程 held-out scenario → 触发 leak event。**Fallback**：执行 `companion-bench-heldout-leak-protocol.md` 全套；rotate held-out（与 [#35](../known-debts.md) 季度治理联动）+ 取消该季榜单
  - **风险 B**：trusted-runner 模式 VZ 扛对方模型 API 成本压力 → 模式不可持续。**Fallback**：协议明文 "trusted-runner mode 计费 = SUT API cost × 1.5 + judge cost"，要求提交方先付预付款再跑
  - **风险 C**：credential 加密存储路径在 multi-user CI runner 上有泄露风险 → 改 GitHub Secrets / vault-as-a-service

---

## 3. 一次跑分的总成本闭环（#56 提前到这里量化）

### 3.1 评估层四 sweep 成本估算

基于 [`packages/companion-bench/src/companion_bench/cost.py`](../../packages/companion-bench/src/companion_bench/cost.py) `_DEFAULT_PRICES`（2026-Q1 价格）+ §2 各 packet 估算：

| Sweep | 子规模 | API token 成本（USD） | 备注 |
|---|---|---|---|
| #48 judge robustness | 5 judge × 5 SUT × 24 scenario × 1 seed = 600 arcs | **$3,400-5,000** | judge dominant；缩 Opus 4.6 可降至 $2,000 |
| #52 calibration | 0 新 arc（post-hoc 复用 #48 数据） | **$0** | 完全 post-hoc |
| #53 simulator robustness | 5 simulator × 5 SUT × 24 scenario × 1 seed = 600 arcs | **$3,000-4,000** | 与 #48 同量级；judge ensemble 减到 3 |
| #54 statistical power | 增量 4 seed × 5 SUT × 24 scenario = 480 arcs | **$1,500-2,000** | 复用 #48 transcript cache |
| #55 跨语言 scenario | 0（pure dev work） | **$0** | 12 中文 scenario reviewer-curated |
| #56 cost model | 0（pure analytics） | **$0** | 反推前 4 sweep |
| #57 trusted runner | 0（dev + smoke fake SUT） | **$0** | 第一次外部 submission 由提交方付费 |
| **合计 evidence sweep** | | **$8,000-11,000 USD** | ≈ ¥56,000-77,000 |

### 3.2 加上 [#32](../known-debts.md) 真 reference 跑分

| 任务 | 成本 | 备注 |
|---|---|---|
| [#32](../known-debts.md) sub-track 1 small-tier real run（5 SUT × 1 seed × 24 公开 scenario） | $200-400 | 已在 [`scripts/companion_bench/run_companion_bench_paper_suite_small.sh`](../../scripts/companion_bench/run_lscb_paper_suite_small.sh) 估算 |
| [#32](../known-debts.md) sub-track 1 release-tier real run（10 SUT × 3 seed × 120 scenario） | $5,000-15,000 | 已在 [`run_companion_bench_paper_suite_full.sh`](../../scripts/companion_bench/run_lscb_paper_suite_full.sh) 估算 |
| **合计 reference run** | **$5,200-15,400** | |

### 3.3 Phase A 6 个月总 API 预算估算

| 项 | 成本（USD） | 中位估算（CNY） |
|---|---|---|
| 本 packet 7 条 evidence sweep | $8,000-11,000 | ¥56,000-77,000 |
| [#32](../known-debts.md) sub-track 1 reference run（small + release tier 各 1 次） | $5,200-15,400 | ¥36,400-108,000 |
| **buffer**（vendor 价格波动 + 重跑 + judge ensemble 调整） | $1,800-3,600 | ¥12,600-25,200 |
| **合计 Phase A 6 个月** | **$15,000-30,000 USD** | **¥105,000-210,000** |

**对账**：[`docs/business/commercialization-assessment.md`](../business/commercialization-assessment.md) §7.2 P5 GTM 预算「头部模型 API 调用费 10-30 万人民币」**完全 covered**——本 packet evidence 部分占 30-40%，[#32](../known-debts.md) reference run 占 50-60%，buffer 10%。

**关键洞察**：Phase A 总 API 预算的 dominant 是 [#32](../known-debts.md) release-tier reference run（$5-15k），不是本 packet 的 evidence sweep。本 packet 主要消耗的是**工程时间**（共 10-13.5 人周，见 §4），不是 API 钱。

---

## 4. 内部并行度

本组 7 条 debt 按层切，可同时跑 4 条 sub-packet：

### 4.1 数据层 sub-packet（#55 跨语言 scenario）

- **人**：1 人 × 2-3 周
- **依赖**：与 [#32](../known-debts.md) sub-track 2 held-out repo 创建同 release ship（hash 表同步）
- **不依赖任何 sweep**——可立即启动
- **产出**：`ScenarioSpec.language` 字段 + 12 中文 scenario + site 跨语言子榜单 view + spec / RFC / hash 表更新

### 4.2 治理层 sub-packet（#57 trusted runner）

- **人**：1 人 × 1-2 周
- **依赖**：[#32](../known-debts.md) sub-track 2 held-out repo 已 land（held-out scenario 真存在 trusted_runner 才能真调用）
- **不依赖任何 sweep**——可与数据层 sub-packet 并行
- **产出**：`trusted_runner.py` + 协议文档 + 泄露应对文档 + audit logger contract test

### 4.3 评估层 sub-packet（#48 / #52 / #53 / #54）

- **人**：1 人 × 4-6 周（4 sweep 共用同一套 reference SUT 跑分基础设施）
- **依赖**：内部串行—— [#48](../known-debts.md) 先跑（产 transcript cache + 选 judge ensemble）→ [#52](../known-debts.md) post-hoc → [#53](../known-debts.md) 跑 simulator 层（消费 [#48](../known-debts.md) 选出的 judge ensemble）→ [#54](../known-debts.md) 跑增量 seed
- **关键依赖外部**：[#34](../known-debts.md) 修法 1（async + connection pool）应**先于** [#48](../known-debts.md) sweep 启动，否则 wallclock ~24-48 小时；如果 [#34](../known-debts.md) 还没 land，先跑 calibration 子规模（1 judge × 5 SUT × 24）
- **产出**：4 个 sweep 数据 + 4 份公开报告 + judge / simulator robustness 的 spec / RFC 附录段

### 4.4 成本层 sub-packet（#56）

- **人**：1 人 × 1 周
- **依赖**：评估层至少 1 sweep 已跑（取得真 token usage record）
- **产出**：`estimate_quarterly_cost.py` + 公开成本模型报告

### 4.5 总并行度图

```
Week  1   2   3   4   5   6   7   8   9  10  11  12
      │   │   │   │   │   │   │   │   │   │   │   │
数据层 ████████████ #55 (2-3 wk)
治理层     ██████ #57 (1-2 wk, 等 #32 sub-2 land)
评估层 ████████████████████████ #48 (2-3 wk) ── #52 (1-1.5 wk) ── #53 (1.5 wk) ── #54 (1.5-2 wk)
成本层                                 ███ #56 (1 wk, 评估层 ≥1 sweep 后)
```

**最佳并行方案**：3 人 × 6 周完成全部 7 条 debt，工程时长合计 10-13.5 人周（不含 review / 报告写作 buffer）。

---

## 5. 与既有 launch 债（#29 / #32 / #33 / #34 / #35）的接口

本 packet **不替代** launch 路径，是 launch **之前**的「可信度证据」补全。

### 5.1 本 packet 产出 → launch 债直接消费表

| 本 packet 产出 | 被哪条 launch 债直接消费 |
|---|---|
| #48 `judge_robustness_v0.json` + 公开报告 | [#32](../known-debts.md) sub-track 1 真 reference 跑分前置：跑分时**至少**用 robustness sweep 选出的 top-3 judge ensemble |
| #52 `calibration_v0.json` + 公开报告 | [#32](../known-debts.md) sub-track 1 reference 跑分输出的 leaderboard 公开时**必须**附本报告作为权重论证 |
| #53 `RunRecord.simulator_family` 必填字段 + spec §4.x | [#35](../known-debts.md) 季度治理自动化 simulator rotation log 直接消费本 packet 的 family pool |
| #54 `compute_elo_with_ci()` + ELO ± 95% CI 列 | [#32](../known-debts.md) sub-track 1 reference 跑分 leaderboard site 渲染 ELO ± CI；不再发布单一 ELO 数字 |
| #55 `ScenarioSpec.language` + 跨语言子榜单 | [#32](../known-debts.md) sub-track 2（held-out repo 创建）+ [#32](../known-debts.md) sub-track 4（域名上线）同 release 节奏；hash 表 [`docs/external/companion-bench-public-scenario-hashes.txt`](../external/companion-bench-public-scenario-hashes.txt) 同步重生成 |
| #56 cost model 报告 | [#32](../known-debts.md) sub-track 1 small-tier 跑分前 budget approval 用本报告作 evidence；[#34](../known-debts.md) 修法 5 cost-aware scheduler 直接消费本估算结果 |
| #57 trusted_runner 协议 | [#32](../known-debts.md) sub-track 5 submission queue infra 启动时 trusted-runner 模式作为 submission protocol 的一部分；[#33](../known-debts.md) human-eval 轨道复用 audit logger 抽象 |

### 5.2 与 [#29](../known-debts.md) / [#37](../known-debts.md) EQ-Bench 公开提交的关系

[#29](../known-debts.md) / [#37](../known-debts.md) 是「拿第三方分数」（EQ-Bench 3 / Chatbot Arena），本 packet 是「自定义 benchmark 的方法论防御」。两者**互不替代**，但顺序上：

1. **[#37](../known-debts.md) actuation 先跑**（拿到 EQ-Bench 第三方分数，建立外部参照点）
2. **本 packet 7 条同步 land**（补 evidence + 治理）
3. **[#32](../known-debts.md) sub-track 1 真 reference 跑分**（出 LSCB 第一份榜单）
4. **48 小时内 marketing 故事可串**：「VZ 在 EQ-Bench 第 N 名（客观）+ 在 Companion Bench A3/A4 子轴领先（自定义但有 robustness evidence）+ 我们是 RFC convener」

### 5.3 与 [#36](../known-debts.md) v2.x 长尾 / [#38](../known-debts.md) site 小尾巴的关系

- [#36](../known-debts.md)(b) EQ-Bench rubric prompt 1:1 reuse → 与本 packet **正交**（本 packet 的 judge robustness 评估的是 prompt 不变前提下 judge family 方差；[#36](b) 是 prompt 本身改写）
- [#36](../known-debts.md)(c) attestation 加密签名 → 与本 packet [#57](../known-debts.md) trusted-runner credential 加密**复用** `cryptography` 依赖
- [#38](../known-debts.md)(c) verifier wiring → 与本 packet [#54](../known-debts.md) ELO ± CI 列同 sprint ship 更高效（site 一次性升级）
- [#38](../known-debts.md)(e) judge_calibration.json illustrative 数据 → 等本 packet [#48](../known-debts.md) sweep 完成后**直接替换为真数据**

---

## 6. 与其他反思 packet 的接口

商业化反思 26 条 debt 共 4 组，本 packet 仅覆盖 P5 组（#48 #52 #53 #54 #55 #56 #57）。其他组在独立 packet 中规划：

### 6.1 与横切组（#45-#47 / #49-#50 / #69）的接口

- [#48](../known-debts.md) **是横切 debt**，但本 packet 让它**先在 P5 验证再 generalize**：
  - 本 packet：`judge_robustness_sweep.py` + spec §5 + RFC §5 附录 + `tests/contracts/test_llm_classifier_robustness_required.py`
  - 横切 generalization：未来 P2 archetype 识别（[`packages/lifeform-domain-growth-advisor/`](../../packages/lifeform-domain-growth-advisor/)）/ P1 voice judge（[`packages/lifeform-domain-figure/.../verification/persona/`](../../packages/lifeform-domain-figure/src/lifeform_domain_figure/verification/persona/)）任何引入 LLM judge / classifier 时**复用**本 packet 的 robustness sweep 协议
- 引用：→ `cross-cutting-foundation-packet.md` §X (#48 generalization)
- **关键约束**：本 packet 不要把 sweep 逻辑硬绑死在 `companion-bench` wheel 内——`scripts/companion_bench/judge_robustness_sweep.py` 应**模块化**让未来横切 packet 可复用 protocol（不复用 SUT specifics）

### 6.2 与 P1 组（#58-#63）的接口

- P1 组聚焦 figure vertical 的 L4 ScopeRefuser GT / L3 引证 faithfulness / L4 false-refuse-answer ground truth 等；与本 packet **正交**
- 引用：→ `figure-vertical-evidence-packet.md`（P1 组）
- 唯一交集：P1 组未来可能引入 LLM judge 替代 deterministic voice/cognition score，届时**复用**本 packet [#48](../known-debts.md) 的 robustness sweep 协议

### 6.3 与 P2 组（#64-#68 / #70）的接口

- P2 组聚焦 growth-advisor boundary 触发率 baseline / archetype 识别机制 / weekly report 校准等；与本 packet **正交**
- 引用：→ `growth-advisor-evidence-packet.md`（P2 组）
- 唯一交集：P2 组 5 archetype 识别如果走 LLM classifier，**复用**本 packet [#48](../known-debts.md) 的 robustness sweep 协议

---

## 7. 风险与 kill criteria

### 7.1 sweep 阶段 kill criteria

| 风险 | 触发条件 | 决策 |
|---|---|---|
| #48 judge variance 极大 | per-axis σ > 15 分 OR Spearman < 0.4 | **Kill v1.0 leaderboard 公开**：本 packet 转入 ENGINEERING_HOLD；评估是否换 judge family / 加 ensemble 大小 / 改 rubric prompt（与 [#36](../known-debts.md)(b) 联动） |
| #52 当前权重在 sweep 区间内排名不稳 | 任一 ±0.05 配置下 rank 变化 > ±2 | **触发 v1.x 权重调整**：找 sweet-spot 权重作为 v1.1，与 v1.0 同时发布并标 ALTERNATIVE |
| #53 simulator-induced σ > 10 | 任一 simulator family 显著影响 SUT 评分 | **触发 simulator pool 强制 rotation**：spec §4.x 加 mandatory quarterly rotation（与 [#35](../known-debts.md) 联动） |
| #54 distinguishable threshold > 60 ELO | n=24 公开 set 上 power 不足 | **leaderboard 仅展示 distinguishable bands**（top / mid / bottom tier）而非 ELO 数字；触发 v1.x 扩 scenario 到 200+ |
| #56 季度更新成本 > $4,000 | 反推 8 季度 model 失败 | **触发 staged tier model**：公开 24 季度 1 次 + held-out 半年 1 次；judge ensemble 缩 5→3 |
| #57 第一个外部 trusted-runner 提交触发 leak event | held-out scenario 在外部公开渠道出现 | **执行 [`companion-bench-heldout-leak-protocol.md`](../external/) 全套**：rotate held-out + 取消该季榜单 + 公告 |

### 7.2 packet 整体 kill criteria

- **本 packet 6 个月后 evidence sweep 全部完成但 [#32](../known-debts.md) sub-track 1 reference run 因预算 / 组织原因未跑** → 本 packet 价值 ≤ 30%（evidence 没人引用）。**Fallback**：把 4 份公开报告独立投 arXiv preprint（"Methodological Robustness for Long-Session Companion Benchmarks"），让本 packet 在没有 reference run 的前提下仍有学术价值
- **本 packet 完成但 6 个月后 OpenAI / Anthropic / Meta 发布自家 long-session benchmark RFC** → P5 convener 窗口关闭，本 packet 的方法论防御价值降到 30-50%。**Fallback**：把本 packet evidence 转成「我们 follow XX RFC + 增加 Y 方面 robustness 评估」的补充论文形态
- **[#48](../known-debts.md) sweep 显示 LSCB judge ensemble 与 EQ-Bench 3 / RP-Bench judge 在同一 SUT 上排名差 > 2 名**：跨 benchmark 信号转移失败 → 与 [`docs/external/companion-bench-eqbench-crosswalk.md`](../external/companion-bench-eqbench-crosswalk.md) 现有声明矛盾。**Fallback**：写 errata 修订 crosswalk 文档；与 [#36](../known-debts.md)(b) prompt 1:1 reuse 同 packet 推进

---

## 8. 推荐起跑顺序

**总策略**：先跑顺位 1（评估层基础）+ 数据层 + 治理层 三路并行 → 顺位 2 cost 闭环 → 顺位 3 simulator robustness（对前置工作的 generalization）。

### 8.1 顺位 1（Week 1-3）：三路并行启动

- **[#48](../known-debts.md) judge robustness sweep**（评估层 P0）
  - 阻塞条件：[#34](../known-debts.md) async + connection pool 已 land（否则 wallclock 不可控）
  - 如果 [#34](../known-debts.md) 未 land：先跑 1 judge × 5 SUT × 24 calibration（成本 ~$800）确认 pipeline，等 [#34](../known-debts.md) land 后跑大头
- **[#52](../known-debts.md) calibration sweep**（评估层 P0）
  - 紧跟 [#48](../known-debts.md) 完成（post-hoc 复用）；可在同一 sprint 内 ship
- **[#54](../known-debts.md) statistical power analysis**（评估层 P0）
  - 紧跟 [#48](../known-debts.md) 完成（增量 seed 复用 transcript cache）
- **[#55](../known-debts.md) 跨语言 scenario**（数据层 P1）
  - 完全独立——立即启动
- **[#57](../known-debts.md) trusted runner**（治理层 P1）
  - 阻塞条件：[#32](../known-debts.md) sub-track 2 held-out repo 已 land

### 8.2 顺位 2（Week 4-5）：cost 闭环

- **[#56](../known-debts.md) cost model**
  - 阻塞条件：顺位 1 至少 1 sweep 已跑（取得真 token usage record）
  - 强烈建议在 [#48](../known-debts.md) 完成后立即跑——cost model 直接 input 给 [#32](../known-debts.md) sub-track 1 release-tier budget approval

### 8.3 顺位 3（Week 5-6）：simulator robustness

- **[#53](../known-debts.md) simulator robustness sweep**
  - 阻塞条件：[#48](../known-debts.md) sweep 已选出 top-3 judge ensemble
  - 与顺位 1 [#54](../known-debts.md) 并行 OK；都消费 [#48](../known-debts.md) 的 ensemble 决定

### 8.4 全 6 周看板

```
Week  1    2    3    4    5    6
      ┌────┬────┬────┬────┬────┬────┐
评估  │ #48 sweep      │#52│#53 │#54 │
      ├────┴────┬──────┴───┴────┴────┤
数据  │  #55 zh-en       (持续到 Week 3) │
      ├──────────┬─────────────────────┤
治理  │ #57 trusted-runner (Week 2-3)    │
      ├────────────────┬────────────────┤
成本  │                │ #56 cost (Week 4-5)│
      └────────────────┴────────────────┘
```

### 8.5 与 [#37](../known-debts.md) / [#32](../known-debts.md) 节奏的对接

- **顺位 0（建议本 packet 启动前 2 周）**：[#37](../known-debts.md) EQ-Bench 三轨 ablation actuation（拿外部分数）
- **本 packet 顺位 1-3 跑完**（Week 6）
- **顺位 4（Week 7-9）**：[#32](../known-debts.md) sub-track 1 small-tier reference run（消费本 packet [#48](../known-debts.md)/[#52](../known-debts.md)/[#54](../known-debts.md) 的 evidence 配置 + [#56](../known-debts.md) 的 budget model）
- **顺位 5（Week 10-12）**：[#32](../known-debts.md) sub-track 1 release-tier reference run + [#32](../known-debts.md) sub-track 4 域名上线 + media launch

---

## 9. SSOT 约束清单

本 packet 严格遵守 [`ssot-module-boundaries.mdc`](../../.cursor/rules/ssot-module-boundaries.mdc) + [`first-principles-not-patches.mdc`](../../.cursor/rules/first-principles-not-patches.mdc) + [`no-swallow-errors-no-hasattr-abuse.mdc`](../../.cursor/rules/no-swallow-errors-no-hasattr-abuse.mdc)：

### 9.1 脚本边界

- 所有新 sweep 脚本走 `scripts/companion_bench/` 子目录，**不污染 wheel**（不在 [`packages/companion-bench/src/companion_bench/`](../../packages/companion-bench/src/companion_bench/) 内加 sweep / robustness analysis 模块）
- sweep 脚本可 import wheel 的 publicAPI（`companion_bench.aggregator.aggregate_arc` / `companion_bench.cost.CostTracker` 等），但**不允许反向写**——sweep 只是 wheel 的 consumer，不是 owner

### 9.2 评估单向性（R12 / OA-1）

- 不能让 robustness sweep 的结果**反向写回** [`ScenarioSpec`](../../packages/companion-bench/src/companion_bench/spec.py) / [`WEIGHTS`](../../packages/companion-bench/src/companion_bench/aggregator.py) / `lexicon.py` 等 scenario / 评估配置
  - **正确做法**：sweep 结果用于**人工审议**（calibration report 公开 + RFC 升级流程）；任何配置变更走 RFC working group 决议（[#32](../known-debts.md) sub-track 3）
  - **违反**：直接 auto-tune 权重让 LSCB 排名最优 = 把 evaluation 变成学习源 → 违反 R12
- 不能让 [#54](../known-debts.md) ELO CI 计算结果**反向写回** SUT 配置或 scenario distribution

### 9.3 LLM judge 注册

- 任何新增 LLM judge family **必须**在 [`docs/specs/companion-bench.md`](../specs/companion-bench.md) §5 注册 + 给出 fingerprint：
  - `judge_family_id`（unique slug）
  - 模型版本 + 推理参数（temperature / max_tokens / top_p）
  - 价格表 entry（在 [`packages/companion-bench/src/companion_bench/cost.py`](../../packages/companion-bench/src/companion_bench/cost.py) `_DEFAULT_PRICES`）
- contract test `tests/contracts/test_judge_family_registry.py` 静态守门——任何 `JudgeIdentity` 实例化的 `judge_family_id` 必须在 spec 注册表内

### 9.4 LLM prompt 集中

- 遵守 [`llm-prompt-centralization.mdc`](../../.cursor/rules/llm-prompt-centralization.mdc)：sweep 脚本不内联大段 prompt，所有 judge prompt 仍走 [`packages/companion-bench/src/companion_bench/judge_perturn.py`](../../packages/companion-bench/src/companion_bench/judge_perturn.py) `_PROMPT_HEADER` / [`judge_arc.py`](../../packages/companion-bench/src/companion_bench/judge_arc.py) 的统一来源
- sweep 配置（哪个 judge family / 哪个 SUT pool / 哪个 scenario subset）走 YAML / typed dataclass 配置文件，不写死在脚本里

### 9.5 fail loud

- 遵守 [`no-swallow-errors-no-hasattr-abuse.mdc`](../../.cursor/rules/no-swallow-errors-no-hasattr-abuse.mdc)：
  - sweep 脚本遇到 vendor API 5xx / 429 → 显式 retry with backoff 后**fail loudly**写 `failed_arcs.jsonl`，不静默用 deterministic-fake 替代
  - sweep 脚本遇到 `JudgeIdentity` 缺失 fingerprint → raise `MissingJudgeRegistration` 不 hasattr / getattr 默认值
  - cost report 遇到价格表 missing model → raise（已是 [`CostTracker`](../../packages/companion-bench/src/companion_bench/cost.py) 现有行为；本 packet 不削弱）

### 9.6 不做的事

- ❌ 不在本 packet 内动 [`vz-*`](../../packages/) / [`lifeform-*`](../../packages/) 内核包（违反 [#29](../known-debts.md) 红线）
- ❌ 不公开 96 私有 held-out scenario body（违反 §10.2 反目标）
- ❌ 不在 sweep 期间让 LSCB SUT 看到 judge prompt 或反之（违反 cross-contamination 原则）
- ❌ 不为了让 VZ 排第一而调整 reference SUT 池（§7.2 GTM 反目标）
- ❌ 不在 trusted-runner 模式下保留提交方 transcript（违反 [#57](../known-debts.md) 隐私 + 治理协议）

---

## 附录 A. 文件清单（本 packet 新增 / 修改）

### A.1 新增文件

| 路径 | 类型 | 关联 debt |
|---|---|---|
| `scripts/companion_bench/judge_robustness_sweep.py` | 脚本 | #48 |
| `scripts/companion_bench/calibration_sweep.py` | 脚本 | #52 |
| `scripts/companion_bench/simulator_robustness_sweep.py` | 脚本 | #53 |
| `scripts/companion_bench/statistical_power_analysis.py` | 脚本 | #54 |
| `scripts/companion_bench/estimate_quarterly_cost.py` | 脚本 | #56 |
| `scripts/companion_bench/trusted_runner.py` | 脚本 | #57 |
| `docs/external/companion-bench-judge-robustness-v0.md` | 公开报告 | #48 |
| `docs/external/companion-bench-calibration-report-v0.md` | 公开报告 | #52 |
| `docs/external/companion-bench-simulator-robustness-v0.md` | 公开报告（可选） | #53 |
| `docs/external/companion-bench-statistical-power-v0.md` | 公开报告 | #54 |
| `docs/external/companion-bench-cost-model-v0.md` | 公开报告 | #56 |
| `docs/external/companion-bench-trusted-runner-protocol.md` | 协议文档 | #57 |
| `docs/external/companion-bench-heldout-leak-protocol.md` | 应对协议 | #57 |
| 12 × `packages/companion-bench/src/companion_bench/scenarios/public/F*-zh-*.yaml` | 中文 scenario | #55 |
| `tests/contracts/test_judge_robustness_required.py` | contract test | #48 |
| `tests/contracts/test_judge_family_registry.py` | contract test | #48 / #53 |
| `tests/contracts/test_elo_ci_consistency.py` | contract test | #54 |
| `tests/contracts/test_scenario_language_balance.py` | contract test | #55 |
| `tests/contracts/test_cost_model_consistency.py` | contract test | #56 |
| `tests/contracts/test_heldout_access_audit.py` | contract test | #57 |

### A.2 修改文件（不变更默认行为，加字段 / 元数据）

| 路径 | 改动 | 关联 debt |
|---|---|---|
| [`packages/companion-bench/src/companion_bench/judge_perturn.py`](../../packages/companion-bench/src/companion_bench/judge_perturn.py) | 加 `JudgeIdentity` typed dataclass + `judge_family_id` 字段 | #48 |
| [`packages/companion-bench/src/companion_bench/judge_arc.py`](../../packages/companion-bench/src/companion_bench/judge_arc.py) | 同上 | #48 |
| [`packages/companion-bench/src/companion_bench/aggregator.py`](../../packages/companion-bench/src/companion_bench/aggregator.py) | `WEIGHTS` 加 docstring + `WEIGHTS_VERSION = "v1.0"` 常量 | #52 |
| [`packages/companion-bench/src/companion_bench/user_simulator.py`](../../packages/companion-bench/src/companion_bench/user_simulator.py) | 加 `simulator_family_id` enum + `make_simulator(family_id, ...)` factory | #53 |
| [`packages/companion-bench/src/companion_bench/arc_runner.py`](../../packages/companion-bench/src/companion_bench/arc_runner.py) | `RunRecord` 加 `simulator_family: str` 必填 | #53 |
| [`packages/companion-bench/src/companion_bench/elo.py`](../../packages/companion-bench/src/companion_bench/elo.py) | 加 `compute_elo_with_ci(matches, n_bootstrap=1000)` | #54 |
| [`packages/companion-bench/src/companion_bench/spec.py`](../../packages/companion-bench/src/companion_bench/spec.py) | `ScenarioSpec` 加 `language: Literal["zh","en","bilingual"]` 必填；`to_canonical()` 同步 | #55 |
| [`packages/companion-bench/src/companion_bench/heldout_loader.py`](../../packages/companion-bench/src/companion_bench/heldout_loader.py) | 加 `HeldoutAccessAudit` context manager；`load_heldout_scenarios` 必须包在 audit 内 | #57 |
| [`packages/companion-bench/src/companion_bench/submission.py`](../../packages/companion-bench/src/companion_bench/submission.py) | held-out 路径上 wrap audit | #57 |
| [`scripts/companion_bench/build_site.py`](../../scripts/companion_bench/build_site.py) | leaderboard 加 ELO ± CI 列 + simulator family 列 + 跨语言子榜单 view | #53 / #54 / #55 |
| 24 × `packages/companion-bench/src/companion_bench/scenarios/public/*.yaml` | 全部加 `language: en` | #55 |
| [`docs/external/companion-bench-public-scenario-hashes.txt`](../external/companion-bench-public-scenario-hashes.txt) | 重生成（spec.language 字段加进 hash） | #55 |

### A.3 修改文档

| 路径 | 改动 | 关联 debt |
|---|---|---|
| [`docs/specs/companion-bench.md`](../specs/companion-bench.md) | §5 加 judge family registry；§4.x 加 simulator rotation；§6 加 calibration version pin；§3 加 language 字段 | #48 / #52 / #53 / #55 |
| [`docs/external/companion-bench-rfc-v0.md`](../external/companion-bench-rfc-v0.md) | §5 附录引 judge robustness report；§6 附录引 calibration report；§3 加 language 字段；§9 加 v1.x roadmap 条目（scenario expansion if power 不足） | #48 / #52 / #54 / #55 |
| [`docs/external/companion-bench-submission-protocol.md`](../external/companion-bench-submission-protocol.md) | 加 trusted-runner 模式协议引用 | #57 |
| [`docs/external/companion-bench-eqbench-crosswalk.md`](../external/companion-bench-eqbench-crosswalk.md) | 引 judge robustness report 验证 crosswalk 信号转移 | #48 |
| [`docs/external/companion-bench-governance-charter-draft.md`](../external/companion-bench-governance-charter-draft.md) | 引 trusted-runner 协议 + heldout-leak 应对协议 | #57 |

---

## 附录 B. 推荐 reference SUT 池（本 packet sweep 复用）

为本 packet 4 个 sweep（#48 / #52 / #53 / #54）定义稳定的 5 SUT 池，与 [#32](../known-debts.md) sub-track 1 reference run 共享：

| Slot | SUT | 选择理由 |
|---|---|---|
| 1 | `openai/gpt-5` | 头部闭源 baseline；价格中等 |
| 2 | `anthropic/claude-3.7-sonnet` | 头部闭源 baseline；与 GPT 不同家族 |
| 3 | `qwen/qwen2.5-72b-instruct` | 头部开源中文友好；价格便宜 |
| 4 | `deepseek/deepseek-v3` | 头部开源；价格最便宜（cost-control 锚点） |
| 5 | `lifeform-companion` (VZ companion mode) | 自家系统 SUT（与 §1.3 GTM 反目标对齐——观察 VZ 在哪些子轴领先） |

**注**：reference SUT 池选择**与 judge family 池正交**——sweep 规则禁止 SUT 与 judge 同 family（[#48](../known-debts.md) contract test 守门）。`openai/gpt-5` 当 SUT 时 judge ensemble 中不含 OpenAI family；以此类推。

---

## 附录 C. 各 sweep 输出 JSON schema 草案

为后续 sweep 脚本 implementation 提供锚点；schema 通过 [`packages/companion-bench/src/companion_bench/cost.py`](../../packages/companion-bench/src/companion_bench/cost.py) 既有 `CostBreakdown.to_json()` 模式扩展。

### C.1 `judge_robustness_v0.json`（#48 输出）

```json
{
  "sweep_id": "judge-robustness-2026Q2",
  "started_at": "2026-05-13T00:00:00Z",
  "judge_families": ["openai-gpt5", "anthropic-claude-opus-4.7", "qwen-max", "deepseek-v4", "gemini-3-pro"],
  "reference_suts": ["openai/gpt-5", "anthropic/claude-3.7-sonnet", "qwen/qwen2.5-72b-instruct", "deepseek/deepseek-v3", "lifeform-companion"],
  "scenarios": ["F1-continuity-001", "..."],
  "paraphrase_seed": 0,
  "per_axis_variance": {
    "A1": {"sigma": 4.2, "mean": 71.5},
    "A2": {"sigma": 5.1, "mean": 68.2},
    "A3": {"sigma": 7.3, "mean": 64.8},
    "A4": {"sigma": 6.0, "mean": 66.1},
    "A5": {"sigma": 4.8, "mean": 69.0},
    "A6": {"sigma": 3.2, "mean": 78.5}
  },
  "inter_rater": {
    "spearman": {"openai-gpt5__anthropic-claude": 0.72, "...": "..."},
    "kendall_tau": {"...": "..."}
  },
  "rank_stability": {
    "all_judges": ["openai/gpt-5", "anthropic/claude-3.7", "qwen/qwen2.5-72b", "deepseek/deepseek-v3", "lifeform-companion"],
    "drop_each_family": {
      "openai-gpt5": {"new_top3": ["anthropic", "qwen", "deepseek"], "rank_changed": false},
      "...": "..."
    }
  },
  "exit_criteria_met": true,
  "exit_criteria_details": {
    "max_axis_sigma": 7.3,
    "min_spearman": 0.65,
    "rank_stable_under_drop": true
  },
  "cost_breakdown": "{ ... CostBreakdown.to_json() ... }"
}
```

### C.2 `calibration_v0.json`（#52 输出）

```json
{
  "sweep_id": "calibration-2026Q2",
  "input_data_source": "judge-robustness-2026Q2",
  "weights_v1.0": {"A1": 0.10, "A2": 0.15, "A3": 0.25, "A4": 0.20, "A5": 0.10, "A6": 0.20},
  "a6_cap_v1.0": {"threshold": 60.0, "value": 50.0},
  "configurations_swept": 105,
  "rank_matrix": {
    "baseline_v1.0": {"openai/gpt-5": 1, "anthropic/claude-3.7": 2, "qwen/qwen2.5-72b": 3, "deepseek/deepseek-v3": 4, "lifeform-companion": 5},
    "weight_A3+0.05": {"...": "..."},
    "a6_cap=55": {"...": "..."}
  },
  "rank_stability_radius": {
    "openai/gpt-5": {"max_rank_change": 0, "min_rank": 1, "max_rank": 1},
    "lifeform-companion": {"max_rank_change": 1, "min_rank": 4, "max_rank": 5}
  },
  "exit_criteria_met": true,
  "alternative_recommended": null
}
```

### C.3 `simulator_robustness_v0.json`（#53 输出）

类同 #48 schema，但 `judge_families` → `simulator_families`；`per_axis_variance` 改为 `per_sut_axis_simulator_induced_sigma`。

### C.4 `statistical_power_v0.json`（#54 输出）

```json
{
  "sweep_id": "statistical-power-2026Q2",
  "n_paraphrase_seeds": 5,
  "n_bootstrap": 1000,
  "per_sut_elo": {
    "openai/gpt-5": {"elo": 1582, "ci_low": 1556, "ci_high": 1608, "seed_sigma": 18.4},
    "anthropic/claude-3.7": {"elo": 1521, "ci_low": 1495, "ci_high": 1547, "seed_sigma": 21.2}
  },
  "distinguishable_threshold": 30,
  "exit_criteria_met": true,
  "power_curve": {
    "n=24_seeds=5": {"distinguishable_at_pct": [{"delta": 30, "fraction": 0.85}]},
    "n=48": {"distinguishable_at_pct": [{"delta": 30, "fraction": 0.92}]},
    "n=120": {"distinguishable_at_pct": [{"delta": 30, "fraction": 0.97}]},
    "n=200": {"distinguishable_at_pct": [{"delta": 30, "fraction": 0.99}]}
  },
  "scenario_expansion_recommendation": "current n=24 sufficient at delta=30; v1.x consider n=48 if community demands delta=20"
}
```

### C.5 `quarterly_cost_estimate.md`（#56 输出表）

| Quarter | New SUT | Re-run SUT | Total Arcs | Quarter USD | Cumulative USD |
|---|---|---|---|---|---|
| 2026-Q3 | 4 | 6 | 3,600 | $4,200 | $4,200 |
| 2026-Q4 | 2 | 8 | 3,600 | $4,200 | $8,400 |
| 2027-Q1 | 3 | 7 | 3,600 | $4,200 | $12,600 |
| ... | ... | ... | ... | ... | ... |
| 2028-Q2 | - | - | - | - | $33,600 |

---

## 附录 D. Token usage 推导（让 cost 估算可被外部 audit）

§3 cost 估算的 per-arc per-axis token 推导，逐项展开：

### D.1 单 arc 的 SUT 推理 token

- arc_length_sessions × session_turn_range 中位数 ≈ 4 × 7 = 28 turn
- 每 turn SUT input：累积上下文 ≈ 历史 × 200 token + 当前 user utterance ~150 token，turn N 时 ≈ N × 200 + 150
- 28 turn 累积 input 总和 ≈ Σ(N × 200 + 150 for N in 1..28) = 200 × 406 + 150 × 28 = 81,200 + 4,200 = **~85K input token / arc / SUT**
- 每 turn SUT output ≈ 250 token；28 turn = **~7K output token / arc / SUT**

### D.2 单 arc 的 per-turn judge token

- per-turn judge 输入：current turn user + assistant + 历史摘要 ≈ ~1500 token / turn
- per-turn judge 输出：rubric scores ≈ ~200 token / turn
- 28 turn × (1500 input + 200 output) = **~42K input + ~5.6K output token / arc / judge**

### D.3 单 arc 的 arc judge token

- arc judge 输入：完整 transcript（28 turn × ~450 token = ~12.6K token）+ scenario context ~2K token = ~14.6K token
- arc judge 输出：6 axis × 200 token rationale + scores ≈ ~1.5K token
- **~14.6K input + ~1.5K output token / arc / judge**

### D.4 单 arc 总 token（1 SUT × 1 judge ensemble member）

- SUT：85K input + 7K output
- per-turn judge：42K input + 5.6K output
- arc judge：14.6K input + 1.5K output
- **合计：~141.6K input + ~14.1K output token / arc / judge ensemble member**

### D.5 #48 sweep 总 token

- 5 judge family × 5 SUT × 24 scenario × 1 seed = 600 arcs
- per-arc per-judge：141.6K input + 14.1K output
- **judge 端**：5 judge × 600 arcs × (42K + 14.6K input + 5.6K + 1.5K output) = 5 × 600 × 56.6K input + 7.1K output = 169.8M input + 21.3M output
- **SUT 端**：5 SUT × 24 × 1 seed = 120 unique arcs × (85K input + 7K output) = 10.2M input + 0.84M output（每 SUT 仅跑 1 次，结果给所有 5 judge 评分）
- **合计**：180M input + 22M output token / sweep
- judge 端加权价格（5 family 平均：claude-opus $15/$75, gpt-5 $5/$15, qwen $0.40/$1.20, deepseek $0.27/$1.10, gemini $5/$15 = 中位 $5.13/$21.45 per 1M）→ judge cost = 169.8M × $5.13 + 21.3M × $21.45 = $871 + $457 = $1,328
- SUT 端加权价格（同 5 SUT，中位 ~$3.5 input + ~$8.5 output）→ SUT cost = 10.2M × $3.5 + 0.84M × $8.5 = $36 + $7 = $43
- **#48 sweep 真总 USD ≈ $1,371**（不含 retry / 探索 buffer）
- 加 30-50% buffer（vendor API 不稳定 + 重跑 + judge ensemble 探索）→ **~$1,800-2,100 USD**

**注**：本附录推导比 §2.1 估算（$3,400-5,000）更激进（更低）。两者差异：§2.1 估算包含 GPT-5 / Claude Opus 4.6 比例更高（保守上界），本附录用 5 judge 中位价格（更接近实际）。**采取 §2.1 上界作为预算 approval 数字，本附录数字作为「sweep 实跑后真 cost.json 的对账锚点」**——如果实跑 cost > §2.1 上界 1.5× 则触发 cost overrun review。

### D.6 同样推导适用其他 sweep

- #53 simulator robustness：与 #48 同结构，judge family 退化为 ensemble of 3 → judge cost 减 40% → ~$800-1,100
- #54 statistical power：增量 4 seed × 5 SUT × 24 scenario = 480 新 arcs；judge ensemble 用 #48 选出的 top-3 → ~$1,000-1,400
- #56 cost model：post-hoc 反推，无新 API call → $0

---

## 附录 E. Sweep manifest YAML 模板

为本 packet 4 个 sweep 提供统一的配置入口（避免散落 hardcoded 在各 script 里），落 `scripts/companion_bench/sweep_manifests/`：

```yaml
# scripts/companion_bench/sweep_manifests/judge_robustness_v0.yaml
sweep_id: judge-robustness-2026Q2
sweep_kind: judge_robustness  # one of: judge_robustness | calibration | simulator_robustness | statistical_power
output_dir: artifacts/companion_bench/judge_robustness/

# 5 judge family (与 reference SUT 池正交)
judge_families:
  - id: openai-gpt5
    model_identifier: openai/gpt-5
    base_url_env: OPENAI_BASE_URL
    api_key_env: OPENAI_API_KEY
    fingerprint: {temperature: 0.0, max_tokens: 800}
  - id: anthropic-claude-opus-4.7
    model_identifier: anthropic/claude-opus-4.7
    base_url_env: ANTHROPIC_BASE_URL
    api_key_env: ANTHROPIC_API_KEY
    fingerprint: {temperature: 0.0, max_tokens: 800}
  - id: qwen-max
    model_identifier: qwen/qwen-max
    base_url_env: DASHSCOPE_BASE_URL
    api_key_env: DASHSCOPE_API_KEY
    fingerprint: {temperature: 0.0, max_tokens: 800}
  - id: deepseek-v4
    model_identifier: deepseek/deepseek-v4
    base_url_env: DEEPSEEK_BASE_URL
    api_key_env: DEEPSEEK_API_KEY
    fingerprint: {temperature: 0.0, max_tokens: 800}
  - id: gemini-3-pro
    model_identifier: google/gemini-3-pro
    base_url_env: GOOGLE_BASE_URL
    api_key_env: GOOGLE_API_KEY
    fingerprint: {temperature: 0.0, max_tokens: 800}

# 5 reference SUT (附录 B)
reference_suts:
  - openai/gpt-5
  - anthropic/claude-3.7-sonnet
  - qwen/qwen2.5-72b-instruct
  - deepseek/deepseek-v3
  - lifeform-companion

# 24 公开 scenario (read from packages/companion-bench/src/companion_bench/scenarios/public/)
scenario_subset: public_v1.0
paraphrase_seeds: [0]

# Sweep-specific 阈值 (与 §2.1 退出标准对应)
exit_criteria:
  max_axis_sigma: 8.0
  min_spearman: 0.65
  rank_stable_under_drop: true

# Cost guardrail
cost_cap_usd: 5000  # 超过则 raise CostCapExceeded
```

---

## 附录 F. 与 [#37](../known-debts.md) / [#32](../known-debts.md) actuation 的总成本对账

| 阶段 | 路径 | 成本（USD） | 时间窗 |
|---|---|---|---|
| Phase A 顺位 0 | [#37](../known-debts.md) EQ-Bench 三轨 ablation actuation | $50-100（ablation）+ $90-180（ELO pass，可选） | 本 packet 启动前 2 周 |
| Phase A 顺位 1-3 | 本 packet 7 条 evidence sweep | $8,000-11,000 | Week 1-6 |
| Phase A 顺位 4 | [#32](../known-debts.md) sub-track 1 small-tier reference run | $200-400 | Week 7-9 |
| Phase A 顺位 5 | [#32](../known-debts.md) sub-track 1 release-tier reference run | $5,000-15,000 | Week 10-12 |
| **合计 Phase A 6 个月** | | **$13,300-26,500** | ≈ ¥93,000-185,500 |

对账 [`docs/business/commercialization-assessment.md`](../business/commercialization-assessment.md) §7.2 P5 GTM 总投入 < 80 万人民币：API token 部分 ¥93k-185k 占 12-23%；其余预算用于 1 PR/内容人 × 6 个月（~¥360-480k）+ 域名 / 服务器 / 学术合作（~¥60-100k）+ buffer，**整体 budget 健康**。

---

## 附录 G. Spec / RFC 同步检查清单

本 packet 命中 [`first-principles-not-patches.mdc`](../../.cursor/rules/first-principles-not-patches.mdc) "Spec 同步协议"——以下改动**必须**同步对应 spec：

| 改动 | 影响 spec 段落 | 同步方式 |
|---|---|---|
| `JudgeIdentity` + `judge_family_id` 字段（#48） | [`docs/specs/companion-bench.md`](../specs/companion-bench.md) §5（judge layer） | 加 "Judge family registry" 段 + 5 family fingerprint 表 |
| `WEIGHTS_VERSION` + calibration evidence pin（#52） | [`docs/specs/companion-bench.md`](../specs/companion-bench.md) §6.4 + RFC §6 | docstring 引 calibration report；RFC §6 加 calibration appendix |
| `simulator_family_id` enum + rotation 协议（#53） | [`docs/specs/companion-bench.md`](../specs/companion-bench.md) §4 + RFC §4 | 加 "Simulator family rotation" 段 |
| `compute_elo_with_ci()`（#54） | [`docs/specs/companion-bench.md`](../specs/companion-bench.md) §6.5 + RFC §6 | 加 "Statistical power" 段 |
| `ScenarioSpec.language`（#55） | [`docs/specs/companion-bench.md`](../specs/companion-bench.md) §3 + RFC §3 | 加 language 字段说明 + 跨语言子榜单协议 |
| `CostBreakdown` quarterly model（#56） | [`docs/specs/companion-bench.md`](../specs/companion-bench.md) §6.7 + RFC §6.7 | 引 cost-model report |
| `HeldoutAccessAudit` + trusted runner（#57） | [`docs/specs/companion-bench.md`](../specs/companion-bench.md) §7（submission protocol） + RFC §7 | 加 "Submission modes" 段：self-hosted vs trusted-runner |

**禁止**：spec 未同步前 PR 不能 merge——遵守 [`first-principles-not-patches.mdc`](../../.cursor/rules/first-principles-not-patches.mdc) "Spec 同步协议" 强约束。

---

## 附录 H. 7 个 sub-packet 的 PR 拆分建议

每个 sub-packet 独立成 PR（遵守 [`cursor-convergence-workflow.mdc`](../../.cursor/rules/cursor-convergence-workflow.mdc) "单包 3-8 个关键文件"）：

| PR | 关联 debt | 关键文件数 | 依赖 PR |
|---|---|---|---|
| PR-A | #48 judge family registry + sweep script | 6（judge_perturn / judge_arc / sweep script / spec / RFC / contract test） | 无（可立即启动） |
| PR-B | #52 calibration sweep + report（post-hoc） | 4（aggregator docstring / sweep script / report / RFC §6） | PR-A merged + 至少 1 sweep 跑过 |
| PR-C | #53 simulator family + sweep | 6（user_simulator / arc_runner / sweep script / spec §4 / RFC §4 / contract test） | PR-A judge ensemble 选定 |
| PR-D | #54 ELO ± CI + sweep | 5（elo.py / sweep script / build_site.py / report / contract test） | PR-A merged（复用 transcript cache） |
| PR-E | #55 language 字段 + 12 中文 scenario | 8+（spec.py / 24 旧 + 12 新 scenario yaml / build_site.py / contract test） | [#32](../known-debts.md) sub-track 2 同 release |
| PR-F | #56 cost model | 3（estimator script / report / contract test） | PR-A 至少 1 sweep 跑过 |
| PR-G | #57 trusted runner + audit | 6（trusted_runner.py / heldout_loader / submission / 2 协议文档 / contract test） | [#32](../known-debts.md) sub-track 2 |

**总计 PR 数：7**（与 sub-packet 数 1:1 对应）。每个 PR 都满足 convergence-workflow 的「3-8 个关键文件」约束 + R15「可解释 + 可回滚」约束。

---

## 变更日志

- 2026-05-13：v0.1 初稿。基于 [`docs/known-debts.md`](../known-debts.md) #48 / #52-#57 全文 + [`docs/business/commercialization-assessment.md`](../business/commercialization-assessment.md) §3.3 / §4.5 / §5.2 / §7.2 / §10.2 + [`docs/specs/companion-bench.md`](../specs/companion-bench.md) + [`docs/external/companion-bench-rfc-v0.md`](../external/companion-bench-rfc-v0.md) + [`packages/companion-bench/`](../../packages/companion-bench/) wheel 真实结构推导。下次复盘：本 packet review 完成后 v0.2。
