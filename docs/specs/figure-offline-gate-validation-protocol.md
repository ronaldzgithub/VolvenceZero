# Figure OFFLINE Gate Validation Protocol

> Status: scaffold v0.2 (SHADOW；audit ledger 接通)
> Last updated: 2026-05-14
> Owner: figure-evidence-packet G-E (debt #62)

## 1. 范围

`apply_steering_through_gate(...)` / `apply_persona_lora_through_gate(...)` 走 ModificationGate.OFFLINE 时，gate proposal 必须带 `validation_delta`。本 spec 锁定 `validation_delta` 的**测量方法**——避免不同人测出不同数 → gate 形同虚设。

## 2. 当前问题

PEFT bake 的 `validation_delta = (init_loss − final_loss) / init_loss` 是**训练 loss 改善**，与下游行为改善（refusal accuracy / grounding faithfulness / voice perceptibility）只是相关，不是因果。

## 3. 双重 validation_delta

每次 OFFLINE gate proposal 必须同时带：

| 字段 | 含义 | 计算 |
|---|---|---|
| `train_loss_delta` | 训练 loss 相对改善 | `(init_loss − final_loss) / init_loss`（已有） |
| `downstream_score_delta` | 下游 eval 改善 | refusal/grounding/voice eval 跑分对比 baseline |

Gate 决策：

```python
gate_passes = (
    train_loss_delta >= 0.05      # 当前阈值
    and downstream_score_delta >= 0.05  # 新增；空时降级 warn
)
```

## 4. downstream_score_delta 测量

bake CLI 必须传 `--downstream-eval-gt-root` 参数（指向 `data/figure_refusal_gt/<figure>/` 与 `data/figure_grounding_gt/<figure>/`）：

1. bake 完成后，临时 attach 候选 LoRA / steering 到 bundle
2. 跑 `scripts/figure_refusal_eval.py` + `scripts/figure_grounding_eval.py` 在 GT 集合上
3. 与 baseline（不带候选）的 eval report 比对
4. `downstream_score_delta = (candidate_score − baseline_score) / baseline_score`（每个 SLA 指标分别算）
5. 取最差的指标 delta 作为 gate 输入

## 5. 默认行为（兼容性）

如果 `--downstream-eval-gt-root` 未传：
- gate 仍用旧 `train_loss_delta` 单一判定（向后兼容）
- audit 字段 `downstream_score_delta_method = "absent"`，warn

如果传了：
- 双重判定生效
- audit 字段 `downstream_score_delta_method = "refusal+grounding"`

## 6. 与 #58 / #59 / #41 的耦合

- #58 refusal GT 必须 ACTIVE → 才能算 refusal score
- #59 grounding GT 必须 ACTIVE → 才能算 grounding score
- #41 真 Qwen 必须可跑 → 才能让 score 不是噪声

ACTIVE 节奏：W6（#58/#59 ACTIVE）+ #41（Phase B 早期）→ 本协议 ACTIVE。

## 7. 退出标准

| 阶段 | 标准 |
|---|---|
| **SHADOW v0.1**（W5） | 本 spec 落档；CLI `cmd_bake_lora` 加 `--downstream-eval-gt-root` 占位（fail-loud "scaffolded"） |
| **SHADOW v0.2**（W5+） | `apply_*_through_gate(audit_log_dir=...)` 真写 `OfflineGateAuditEntry`（schema v0.2）+ contract test 守门 schema 字段稳定；`downstream_score_delta` / `downstream_score_delta_method` 字段已 wire 但默认 `None` / `"absent"` 直到 v0.3 ACTIVE |
| **ACTIVE v0.3**（W6+） | 双重 gate 真生效（`downstream_score_delta` 必填非 None）；audit 字段记录 `downstream_score_delta_method`；旧 `train_loss_delta` 单一判定降级为兼容路径 |

## 8. Audit Ledger 字段（v0.2 真接通）

[`packages/lifeform-domain-figure/src/lifeform_domain_figure/gate_apply.py`](../../packages/lifeform-domain-figure/src/lifeform_domain_figure/gate_apply.py) `OfflineGateAuditEntry`：

| 字段 | 类型 | 含义 |
|---|---|---|
| `audit_id` | str | `audit-{figure_id}-{epoch_ms}-{integrity[:8]}` 唯一审计 id |
| `audit_log_schema_version` | str | 当前 `"v0.2"`（schema 升版必动） |
| `timestamp_iso` | str | UTC ISO 8601 |
| `figure_id` | str | 适用 figure |
| `artifact_kind` | str | `"persona_lora"` / `"steering"` |
| `artifact_integrity_hash` | str | 候选 artifact 的完整 SHA-256 |
| `train_loss_delta` | float | gate 的训练 loss 改善（已有路径） |
| `downstream_score_delta` | float \| None | 下游 eval 改善（v0.2 默认 None；v0.3 ACTIVE 必填） |
| `downstream_score_delta_method` | str | `"absent"` / `"refusal+grounding"` / `"refusal"` / `"grounding"` |
| `capacity_cost` | float | gate proposal 的 capacity_cost |
| `decision` | str | `"ALLOW"` / `"BLOCK"` |
| `block_reasons` | tuple[str] | BLOCK 时的原因列表（ALLOW 时为 `()`） |
| `base_bundle_id` | str | 基线 bundle id（rollback target） |
| `candidate_bundle_id` | str \| None | ALLOW 时为新 bundle id；BLOCK 时为 None |
| `previous_record_id` | str | 之前 LoRA pool record id（`"absent"` 表示首次） |
| `record_id` | str \| None | ALLOW 时为新 pool record id；BLOCK 时为 None |
| `rollback_evidence` | str | 调用方传入的 rollback evidence string |

文件路径：`{audit_log_dir}/offline-gate-audit-{figure_id}-{YYYYMMDD}.jsonl`（按日 rotate；append-only）。

## 变更日志

- 2026-05-13: v0.1 SHADOW scaffold。
- 2026-05-14: v0.2 audit ledger 接通——`apply_persona_lora_through_gate` 加 `audit_log_dir` / `downstream_score_delta` / `downstream_score_delta_method` kwargs + `OfflineGateAuditEntry` typed dataclass + per-day jsonl rotation；contract test ([`tests/contracts/test_figure_offline_gate_audit_ledger.py`](../../tests/contracts/test_figure_offline_gate_audit_ledger.py)) 守门 schema 字段稳定 + ALLOW/BLOCK 都有 audit row + rollback evidence 必填。
