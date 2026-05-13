# Growth-Advisor Archetype Detection Spec

> Status: scaffold v0.1 (SHADOW)
> Owner: growth-advisor-pilot-packet G-C (debt #66)
> Implementation: [`packages/lifeform-domain-growth-advisor/src/lifeform_domain_growth_advisor/archetype_classifier.py`](../../packages/lifeform-domain-growth-advisor/src/lifeform_domain_growth_advisor/archetype_classifier.py)

## 1. 问题

`profile.py` 的 5 archetype seeds 没有识别机制，导致下游 boundary policy / playbook day routing / 月报 archetype distribution 失去信号源。本 spec 锁定 SHADOW 用 (a) LLMArchetypeClassifier、长期过渡到 (c) metacontroller β_t。

## 2. 三路径决策表

| 路径 | 是否采用 | 优势 | 劣势 | 与 R 铁律的关系 |
|---|---|---|---|---|
| **(a) LLMArchetypeClassifier** | ✅ **采用** (短期) | 短期可用；与 cheng_laoshi typed payload 对齐；robustness sweep 可量化 | LLM 调用成本；跨 family bias（与 #48 同坑）| 不违反；LLM 是表达层兼语义判断层 |
| **(b) keyword / regex** | ❌ **永久排除** | 零成本 | 字符串脆弱；不能泛化 | **违反 [`no-keyword-matching-hacks.mdc`](../../.cursor/rules/no-keyword-matching-hacks.mdc)** |
| **(c) learned metacontroller β_t** | ⏳ **长期** (Phase B+) | 涌现式；零调用成本（控制器内联）| 依赖 [`#44`](../known-debts.md) SYS-1 ACTIVE | 符合 R3/R4；但当前不可用 |

## 3. (a) LLMArchetypeClassifier 设计

### 3.1 调用频率

每 N=3 turn 调一次 LLM（不每 turn）：
- N=1 成本翻 3 倍，且 archetype 短期内不会大幅波动
- N=3 平衡成本 vs 实时性；首次调用在第 2 turn（min_turns_to_classify）

### 3.2 缓存策略

per (tenant_id, end_user_id) 缓存最近 N=10 classification 历史。计算 stability = 同 primary 的 fraction，给下游一个"transitioning vs settled"信号。

### 3.3 LLM 选型

默认 **DeepSeek V4**：
- 价格便宜（~$0.0014 per 1M input tokens 同档下最低）
- 中文理解好（cheng_laoshi 主要是中文场景）
- JSON 输出 schema 一致性好

可切换到 Qwen3-Max / Claude Opus 4.7（对接客户偏好）。

### 3.4 单位经济

| 项 | 估算 |
|---|---|
| Per call tokens | ~500 input + ~100 output |
| Per call cost (DeepSeek V4) | ~$0.0014 |
| Per end user per month | ~150 turn / 3 ≈ 50 calls × $0.0014 = ~$0.07 |
| Per tenant per month (10 席位 × 50 end_user/席位) | ~500 end_user × $0.07 ≈ **$35 ≈ ¥250** |
| 客户单位经济影响 (§6.3 表 ¥5万/月 客单价) | **¥250-500/月 → 0.5-1% 占比，与 88% 毛利完全兼容** |

### 3.5 robustness sweep

复用 [`#48`](../known-debts.md) sweep 协议（[`scripts/companion_bench/judge_robustness_sweep.py`](../../scripts/companion_bench/judge_robustness_sweep.py) 模板）：

- 5 LLM family × 50 reviewed transcript × archetype-recall (vs reviewer GT)
- ACTIVE 准入：cross-family variance σ ≤ 30%
- 失败 fallback：锁定单一 family (DeepSeek V4) + 降级 ACTIVE 准入到 σ ≤ 40%

## 4. (c) 长期过渡路径

依赖 [`#44`](../known-debts.md) SYS-1 CPD β_t emerge。过渡 4 阶段：

1. **SHADOW (now)**: metacontroller β_t 与 LLMArchetypeClassifier 双轨并行；记录两者一致度
2. **ACTIVE (Phase B+)**: SYS-1 在 archetype 域上 β_t 涌现可观察 (一致度 > 70%)
3. **Cutover**: LLMArchetypeClassifier 降级为 audit / 兜底；β_t 作为运行时 archetype 信号
4. **Sunset**: LLMArchetypeClassifier 关闭，月成本降至 0；archetype detection 内联到控制器层

## 5. SSOT 约束

| 不变量 | 守门 |
|---|---|
| Archetype 状态由 `LLMArchetypeClassifier` **唯一**所有 | downstream 只读 `ArchetypeStateSnapshot`，不重建 |
| Archetype 识别**不**走 keyword 路径 | [`tests/contracts/test_no_keyword_archetype_detection.py`](../../tests/contracts/test_no_keyword_archetype_detection.py) AST 守门 |
| Classifier prompt 集中管理 | [`packages/lifeform-expression/src/lifeform_expression/prompts/growth_advisor_archetype_classify.txt`](../../packages/lifeform-expression/src/lifeform_expression/prompts/growth_advisor_archetype_classify.txt) |
| Classifier 输出 JSON 走 schema | [`packages/lifeform-expression/src/lifeform_expression/schemas/archetype_classification.json`](../../packages/lifeform-expression/src/lifeform_expression/schemas/archetype_classification.json) |

## 6. 退出标准

| 阶段 | 标准 |
|---|---|
| **SHADOW**（W2-W5） | 本 spec v0.1 + `archetype_classifier.py` (Protocol + LLM impl SHADOW) + prompt + schema + AST contract test 通过 |
| **ACTIVE**（W6） | DeepSeek V4 真接入；50 transcript robustness sweep 通过；archetype-recall ≥ 0.7；下游 boundary / playbook / 月报 真消费 ArchetypeStateSnapshot |

## 变更日志

- 2026-05-13: v0.1 SHADOW spec + 决策表。
