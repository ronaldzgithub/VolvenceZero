# Figure Grounding Ground-Truth Protocol

> Status: scaffold v0.1 (SHADOW)
> Last updated: 2026-05-13
> Owner: figure-evidence-packet G-B (debt #59)

## 1. 范围

每个 figure bundle 上线 production 前必须配套的 reviewer-curated grounding GT 集合，用于量化 L3 GroundedDecoder 的 evidence faithfulness（pointer 真支持断言 vs 引证 hallucination）。

## 2. 落点结构

```
packages/lifeform-domain-figure/data/figure_grounding_gt/
├── einstein/
│   ├── assertions.jsonl            # ≥ 100 题
│   └── assertions.jsonl.example    # 5 题骨架 example（已 land）
└── ...
```

## 3. JSONL Schema

```json
{
  "qid": "einstein-ground-001",
  "question": "What is general covariance?",
  "expected_assertion": "The laws of physics should take the same form in all coordinate systems.",
  "ground_truth_chunk_ids": ["einstein-1916-gr-section-3"],
  "reviewer_notes": "Core GR principle; assertion must be supported by GR paper §3."
}
```

字段约束：
- `qid`：唯一；命名 `<figure>-ground-<3digit>`
- `expected_assertion`：reviewer 标的"系统应该说的实质性断言"（不需要逐字匹配，语义对齐即可）
- `ground_truth_chunk_ids`：必须 ≥ 1 个，且必须真存在于 bundle.retrieval_index
- `reviewer_notes`：自由文本

## 4. Reviewer 工艺

### 4.1 准入 + 双盲

同 [`figure-refusal-gt-protocol.md`](figure-refusal-gt-protocol.md) §4.1-4.2。

### 4.2 一致性指标

- **Jaccard similarity** on `ground_truth_chunk_ids` ≥ 0.6（双 reviewer 选的支持 chunk 集合重叠度）
- **expected_assertion 相似度** ≥ 0.7（手工或 LLM judge 评估"两 reviewer 标的断言是否说同一件事"）

### 4.3 时间预算

- 100 题 × 双 reviewer × ~7 min/题 = ~23 小时 × 2 reviewer = 46 小时
- 调解 + Jaccard 计算 ≈ 6 小时
- **单 figure grounding GT 集合 reviewer 总工时 ≈ 52 小时**

## 5. ACTIVE 通过 SLA

| 指标 | 阈值 |
|---|---|
| Jaccard κ on chunk_ids | ≥ 0.6 |
| 题数 | ≥ 100 |
| evidence_faithfulness | ≥ 0.95 |
| unsupported_assertion_rate | ≤ 0.05 |

## 6. 与 bundle / R12 的关系

同 refusal GT 工艺（§6 of refusal protocol）。`FigureArtifactBundle.grounding_eval_report` 字段不进 integrity_hash。

## 7. 与 #41 真 Qwen 跑分的耦合

grounding eval 在 synthetic / tiny-gpt2 substrate 上几乎无意义（生成质量太低）。**真有效跑分依赖 #41 真 Qwen-1.5B PEFT bake**完成；本协议 ACTIVE 阶段同步 #41 ACTIVE。

## 8. 退出标准

| 阶段 | 标准 |
|---|---|
| **SHADOW**（W1-W2） | `.example` JSONL + 本 spec v0.1 |
| **ACTIVE**（W4-W6） | Einstein 100 题 GT 入库；Jaccard κ ≥ 0.6；`scripts/figure_grounding_eval.py` 真跑 + report 落 bundle |

## 变更日志

- 2026-05-13: v0.1 SHADOW scaffold + 5 题 `.example` JSONL 落档。
