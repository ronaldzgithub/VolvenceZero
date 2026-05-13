# Figure Refusal Ground-Truth Protocol

> Status: scaffold v0.1 (SHADOW)
> Last updated: 2026-05-13
> Owner: figure-evidence-packet G-A (debt #58)

## 1. 范围

每个 figure bundle 上线 production 前必须配套的 reviewer-curated 双向 GT 集合，用于量化 L4 ScopeRefuser 的 false-refuse / false-answer 双向准确率。这是 P1 商业化"法务能签字"承诺的法律生死线（[`commercialization-assessment.md`](../business/commercialization-assessment.md) §4.1）。

## 2. 落点结构

```
packages/lifeform-domain-figure/data/figure_refusal_gt/
├── einstein/
│   ├── in_scope.jsonl              # ≥ 50 题：reviewer 标"应回答"
│   ├── out_of_scope.jsonl          # ≥ 50 题：reviewer 标"应拒答"
│   ├── in_scope.jsonl.example      # 5 题骨架 example（已 land）
│   └── out_of_scope.jsonl.example  # 5 题骨架 example（已 land）
├── curie/                          # 第二款 figure（Phase B）
└── ...
```

## 3. JSONL Schema

### in_scope.jsonl 行

```json
{
  "qid": "einstein-in-001",
  "question": "What is the postulate of the constancy of the speed of light?",
  "expected_action": "answer",
  "cited_chunk_ids_hint": ["einstein-1916-relativity-ch7"],
  "domain_tags": ["physics", "special_relativity"],
  "reviewer_notes": "Direct quotation territory; should answer with citation."
}
```

### out_of_scope.jsonl 行

```json
{
  "qid": "einstein-oos-001",
  "question": "What is your favorite tiramisu recipe?",
  "expected_action": "refuse",
  "expected_refuse_kind": "out_of_scope_topic",
  "domain_tags": ["food"],
  "reviewer_notes": "No corpus on cooking; must refuse + explain scope."
}
```

字段约束：
- `qid`：figure 级唯一；命名 `<figure>-<in|oos>-<3digit>`
- `expected_action`：`"answer"` / `"refuse"` 二选一
- `cited_chunk_ids_hint`（in_scope only）：reviewer 期望系统引用的 chunk_id 集合（hint，不是 hard requirement）
- `expected_refuse_kind`（OOS only）：`"out_of_scope_topic"` / `"anachronism"` / `"in_world_confidential"` 等枚举
- `domain_tags`：自由 tag，便于 per-domain breakdown
- `reviewer_notes`：reviewer 自由文本（不进 eval 计算，仅为 audit）

## 4. Reviewer 工艺

### 4.1 准入

- reviewer 资质：figure 领域学术 / 历史背景（如物理学硕士 + Einstein 史料阅读经验）
- 必须签 NDA + 数据使用同意书
- 标注前完成 30 min onboarding：解释 L4 ScopeRefuser 行为 + 双盲流程

### 4.2 双盲

每条 GT 由 **2 个独立 reviewer** 各自标注，第三方 (lead reviewer) 调解分歧。

### 4.3 一致性指标

- **Cohen κ** ≥ 0.65 视为通过（行业标准对二分类标注）
- κ 计算：双 reviewer 在 `expected_action` 上的一致比例（修正随机一致后）
- κ < 0.65 触发 reviewer 培训 + 重标 30% 样本
- κ 计算公式：
  ```
  κ = (P_o - P_e) / (1 - P_e)
  其中 P_o = 实际一致比例，P_e = 随机一致期望
  ```

### 4.4 时间预算

- in_scope 50 题：reviewer 1 ≈ 15 小时，reviewer 2 ≈ 15 小时
- out_of_scope 50 题：reviewer 1 ≈ 10 小时，reviewer 2 ≈ 10 小时
- 调解 + κ 计算 ≈ 15 小时
- **单 figure GT 集合 reviewer 总工时 ≈ 65 小时**

第二款 figure 复用同 SOP，工时同量级。

## 5. ACTIVE 通过 SLA

| 指标 | 阈值 | 来源 |
|---|---|---|
| inter-rater κ | ≥ 0.65 | 双盲计算 |
| in_scope 题数 | ≥ 50 | reviewer 标注 |
| out_of_scope 题数 | ≥ 50 | reviewer 标注 |
| false_refuse_rate（系统） | ≤ 0.10 | `scripts/figure_refusal_eval.py` |
| false_answer_rate（系统） | ≤ 0.05 | 同上 |
| 95% Wilson CI | 上限不超出阈值 | 同上 |

## 6. 与 bundle integrity 的关系

GT 是 **evaluation readout** 输入，**不**反向喂 LoRA 训练（违反 R12 evaluation 不是学习源）。`FigureArtifactBundle.refusal_eval_report` 字段（debt #58 closure）：

- 记录最近一次 eval 结果
- **不**进 `compute_bundle_integrity_hash`（eval 是 readout，不是 bundle identity）
- 单独的 `refusal_eval_report_fingerprint` audit 字段（待 #58 ACTIVE 落档）

## 7. 与 commercialisation 商业承诺的对账

- §4.1 P1 价值主张 "L3 引证 + L4 拒答让博物馆/教育机构的法务可以签字" → false_refuse + false_answer 双 SLA 是法务签字的**量化前提**
- §6.2 P1 单笔首单 30-80 万 → SLA 写进合同 → 客户尽调挑 10 段 demo 时有真数据回应
- §10.1 反目标 figure 部分 "未授权在世人物" → GT 集合本身只覆盖公共领域 figure（Einstein / Curie / 苏轼 / 鲁迅等）

## 8. 退出标准

| 阶段 | 标准 |
|---|---|
| **SHADOW**（W1-W2） | `.example` JSONL 落 5 题 + 本 spec v0.1 落档 + reviewer 工艺 SOP 写完 |
| **ACTIVE**（W3-W6） | Einstein 真 GT 集合 in_scope 50 + out_of_scope 50 入库；inter-rater κ ≥ 0.65；`scripts/figure_refusal_eval.py` 真跑 + report 落 bundle |

## 9. 风险

| 风险 | 应对 |
|---|---|
| reviewer 难招 | 通过学术合作（国内物理史 / 文学史研究生）+ ¥200/小时 |
| κ < 0.65 | reviewer 培训 + 重标；如多次 < 0.65，简化 GT schema（如砍 `expected_refuse_kind` 枚举） |
| GT 数据泄露 | NDA + 内部 git 仓库（不公开）+ 不进 LoRA 训练 |
| GT 反向 train | AST 守门 `tests/contracts/test_figure_bundle_refusal_gt_required.py` + R12 评审 |

## 变更日志

- 2026-05-13: v0.1 SHADOW scaffold + 5 题 `.example` JSONL 落档。
