# Companion Bench: i18n / 跨语言 Scenario Roadmap (v0)

> Status: SHADOW + 6 zh demo scenarios shipped
> Driving debt: [`docs/known-debts.md`](../known-debts.md) #55
> Driving packet: [`docs/moving forward/companion-bench-public-launch-packet.md`](../moving%20forward/companion-bench-public-launch-packet.md) §2.4

## 1. 目的

回答 RFC §6.5 / reviewer 第五问：**"Companion Bench 是否对中英母语用户的关系连续性保真都验证？"**

24 个初始公开 scenarios 全为 reviewer-curated 英文 simulator 内容；中文用户人格 / 文化语境的覆盖缺位会让中文母语 SUT 的真实差异被掩盖。

## 2. 架构

`ScenarioSpec.language` 字段定义 scenario 的 simulator content 主语言：

| 取值 | 含义 |
|---|---|
| `en` | simulator 全英文（initial 24 reviewer-curated 默认） |
| `zh` | simulator 全中文 |
| `bilingual` | simulator arc 中混语切换（reserved，v1.x 才启用） |

**Hash 稳定性**：`language` 字段**不**进入 `ScenarioSpec.to_canonical()` —— 跨语言"等同语义" scenario 共享同一 `scenario_hash`。这让 leaderboard 可按 language 分两个 sub-view 同时不破坏 RFC §3 P3 reproducibility。

## 3. 当前状态 (2026-05-14)

| 语言 | scenario 数 | 来源 |
|---|---|---|
| `en` | 24 | initial reviewer-curated (Wave A; F1-F6 各 4 个) |
| `zh` | 6 | i18n demo scaffold (此次 land；F1-F6 各 1 个) |
| `bilingual` | 0 | reserved |
| **合计** | **30** | |

6 个 zh demo scenarios（命名 `F*-{family-suffix}-zh-001.yaml`）已加入 `packages/companion-bench/src/companion_bench/scenarios/public/`：

- `F1-continuity-zh-001` —— 上海设计师持续话题（连续性）
- `F2-repair-zh-001` —— 互联网新人修复（rupture/repair）
- `F3-personalization-zh-001` —— 高校教师 register 偏好（个性化）
- `F4-long-absence-zh-001` —— 全职妈妈长间隔回归（长缺席）
- `F5-boundary-zh-001` —— 失眠管理者边界（boundary）
- `F6-goal-drift-zh-001` —— 应届生目标漂移（goal drift）

每个 zh demo 与对应 family 的 en variant 是**不同 scenario**（不同人格 / 不同 callback 探针 / 不同 disqualifier 情景），不是翻译 — i18n roadmap 的核心是覆盖中文母语用户的真实关系连续性挑战，不是英文文本翻译。

## 4. 12 zh + 12 en 增补 Roadmap

每个 family 计划补足到 4 zh + 4 en（总 48 公开 scenarios），分两批：

### Batch 1（W4-W6，本计划已 land 6 zh demo）
- F1-F6 各 1 个 zh ✅
- ScenarioSpec.language 字段 ACTIVE ✅
- contract test ([`tests/contracts/test_companion_bench_i18n_scenarios.py`](../../tests/contracts/test_companion_bench_i18n_scenarios.py)) ✅

### Batch 2（W4-W6，reviewer-curated 后续）
- F1-F6 各 +1 zh = +6 zh（总 zh: 12）
- F1-F6 各保留 4 en（总 en: 24）→ 但仅前 12 进 v1.0 公开 leaderboard，剩 12 进 v1.1 缓冲
- 触发条件：reviewer 标注预算批准 + 中文 reviewer 招募完成（运营动作，不在本 plan）

### 公开 hash 表更新
- `emit_scenario_hashes.py` 已支持遍历公开目录；新加的 zh 6 个 scenario 自动出现在 `docs/external/companion-bench-public-scenario-hashes.txt` 下次重生时
- v1.0 launch 前要求公开 hash 表中 zh / en 都有 ≥ 12

## 5. 风险 & 不变量

| 项 | 不变量 |
|---|---|
| `language` 不进 `to_canonical` | hash 稳定性；语义等同跨语言 scenario 同 hash |
| zh / en 评估指标可加可分 | `BenchmarkReport` 默认 axis means 跨语言聚合，可选按 language 切分 |
| 无 leakage：zh demo 不出现在 held-out | held-out 同样要求覆盖 zh，但走 private repo（`companion-bench-heldout`） |
| 翻译 + reviewer-curated 工艺一致 | 中文 reviewer 也要标 disqualifiers + persona 完整字段，质量门槛与英文一致 |

## 6. 变更日志

- 2026-05-14: v0 i18n roadmap + 6 zh demo scenarios + contract test land；ScenarioSpec.language 默认值从 "zh" 改为 "en"（更准确反映 initial 24 yaml 现状）。剩余 6 zh + 12 en 增补留 reviewer-curated 后续。
