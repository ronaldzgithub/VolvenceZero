# Reliable Apprenticeship Alignment（可靠学徒对齐）

**Slot**: `apprenticeship_alignment`
**Owner**: `ApprenticeshipAlignmentModule`（`packages/vz-cognition/src/volvence_zero/apprenticeship/`）
**默认接线**: SHADOW
**对应需求**: R-PE、R7、R8、R11、R15

---

## 1. 目的

在学徒模式（apprentice / ingestion turn）里，把 operator 给出的指导/教材与 AI **当前公共认知**实时对比，回答两个问题：

1. **指导 vs 认知（实时对比）**：这条指导是否已被当前认知覆盖（agreement region → AI 可可靠自主），还是落在未覆盖区（disagreement region → AI 应"知道自己没把握"、延迟/求证）？分**事实级**与**抽象级**。
2. **材料/指导内部矛盾**：当多条指导无法被任何单一"operator 意图假设"同时满足时，版本空间塌空（`INCONSISTENT`）——这是"材料自相矛盾"的严格判据，并定位**最小互斥约束集**。

## 2. 理论主干

Hanneke, Yang, Wang & Song, *Reliable Active Apprenticeship Learning*, ALT 2025, PMLR 272:512-538。

| 论文概念 | 本 owner 实例化 |
|---|---|
| 专家查询/最优动作 | 学徒 turn 里 operator 的 teach/纠错（一条意图约束） |
| 版本空间 V | 与历史指导一致的 operator 意图约束集（owner 内部 bounded window） |
| 可靠性区/不一致区 | 当前认知是否被指导锁定；不一致区 = 该求证而非静默自主 |
| eluder 信息量 | 指导对认知的未覆盖度 = `guidance_surprise` |
| 带噪专家（Massart/Tsybakov） | operator 可能自相矛盾；矛盾需 reliability margin 或 recurrence 确认 |

支撑件：CLAIRE（WikiCollide 失配类型分类法）做 `mismatch_type`；AGM-BENCH/Belief-R 的最小改动（Inclusion）/不相关信念保留（Preservation）约束信念修正。

## 3. 契约（frozen dataclass）

见 `packages/vz-cognition/src/volvence_zero/apprenticeship/contracts.py`。

- `IntentConstraint`：`constraint_id, statement, level(factual|abstract), polarity(+1/-1/0), target_key, confidence, source_turn, embedding`
- `MismatchRef`：`guidance_constraint_id, level, mismatch_type, belief_ref, severity, description`
- `ContradictionFinding`：`finding_id, constraint_ids(最小互斥集), level, severity, description`
- `ApprenticeshipAlignmentSnapshot`：`version_space_status(idle|consistent|shrinking|inconsistent), consistency_margin, reliability(idle|reliable|deferring), in_agreement_region, guidance_surprise, active_constraint_count, mismatch_refs, contradiction_findings, revision_proposal_refs, description`

## 4. 依赖与数据流

依赖（只读公共快照）：`belief_assumption`、`goal_value`、`user_model`、`boundary_consent`、`regime`。

```
operator 指导文本
  → GuidanceConstraintExtractor（默认 holistic / 生产 LLM 结构化 / 测试 mapping）
  → reconcile_guidance(new vs prior constraints, cognition records)
       ├─ 覆盖度 = 字符 bigram token Jaccard（CJK/拉丁均可分离；stub embedding 对 CJK 不可分）
       ├─ surprise = 1 - 覆盖度；agreement = surprise < 阈值
       ├─ mismatch = surprise ≥ 阈值（novelty 类型，按 level 分级）
       └─ contradiction = 同 topic + 相反 polarity + (margin 或 recurrence 确认)
  → ApprenticeshipAlignmentSnapshot（发布）
```

相似度用 **字符 bigram token Jaccard**（`stub_semantic_tokens`），因为当前 stub embedding 对中文不可分离；`embedding` 字段保留给未来真实 embedding head。token 重叠是通用相似度度量（非关键词→行为映射）。

## 5. 与 PE 集成（R-PE，Phase 2）

`PredictionErrorModule` 声明 `apprenticeship_alignment` 为依赖，在 `_advance` 内以 `_compute_apprenticeship_contribution` 叠加一个**离散事件 PE 源**（类比 AAC alignment overlay）：

- `guidance_surprise` → 抬高 `magnitude`（指导总是"惊奇"，eluder 信息量）
- 确认的 mismatch/contradiction severity → 把 `regime_error` 与 `signed_reward` 推向负向
- owner SHADOW 时 PE 收到 placeholder → overlay 为 **no-op**（R15 可回滚）

PE 仍是唯一所有者；本 owner 不重建 PE 内部状态。

## 6. 与信念修正集成（AGM，Phase 3）

`revision_enabled=True` 时，owner 生成 `SemanticProposal`（不自行写库）：

- 可靠（confidence ≥ reliability margin）的 novel mismatch → `CREATE`/`REVISE`，仅触及该记录（AGM Inclusion 最小改动，保留不相关信念 Preservation），factual→`belief_assumption`，abstract→`goal_value`
- 确认的 contradiction → `BLOCK` + `requires_confirmation`：并存竞争假设，不静默覆盖（带噪专家安全）

提案经 `drain_revision_proposals()` 取出，由 session-post writeback 路径用 `apply_apprenticeship_revisions`（包装 `apply_semantic_writeback_result`）经**单写者** store 落地，与反思 writeback 同一路径（R8 SSOT）。

## 7. 关键不变量

1. owner 只消费/发布快照，零跨 owner 直接调用（R8）
2. 失配/惊奇经 PE owner 进主链；owner 不重建 PE 状态（R-PE）
3. 信念修正只发 proposal，经单写者 store；不绕过 belief/goal owner
4. 矛盾需 margin 或 recurrence 确认；单次低置信对立 = 噪声，不判矛盾
5. 默认 SHADOW；PE overlay 在 SHADOW 为 no-op；可回滚（R15）
6. 相似度/分级走语义/重叠度量，不用关键词→行为硬映射

## 8. 迁移（WiringLevel 三态）

- **SHADOW**（默认）：发布快照、与现有 PE/belief 并跑比对；不发 PE overlay（SHADOW 不进 active 链）、默认不开 revision
- **ACTIVE**：PE 消费 overlay；可开 `revision_enabled` 让信念修正提案进 session-post writeback
- **DISABLED**：发布 placeholder
- 回滚：`FinalRolloutConfig.apprenticeship_alignment` 改回 SHADOW/DISABLED

## 9. 测试

`tests/test_apprenticeship_alignment.py`：confirming（agreement/consistent）、novel（deferring/shrinking/mismatch）、high-confidence 对立（版本空间塌空 + 定位互斥集）、低置信单次（噪声不判矛盾）、recurrence 确认（Massart）、AGM 最小改动提案 + 单写者落地、PE overlay no-op（absent/idle）与矛盾抬高 magnitude。

## 10. 参考

- `docs/next_gen_emogpt.md` — R-PE / R7 / R8 / R11 / R15
- `docs/specs/prediction-error-loop.md` — PE 主链与离散事件源叠加模式（AAC alignment）
- `docs/specs/semantic-state-owners.md` — belief_assumption / goal_value 单写者
- `docs/specs/runtime-ingestion.md` — 学徒/ingestion turn 入口
- `docs/DATA_CONTRACT.md` §6 — slot 注册表
