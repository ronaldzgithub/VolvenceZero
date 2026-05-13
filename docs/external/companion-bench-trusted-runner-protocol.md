# Companion Bench: Trusted Runner Protocol v0

> Status: SHADOW (协议设计 land；运行时 wire 在 #57 ACTIVE)
> Driver: [`scripts/companion_bench/trusted_runner.py`](../../scripts/companion_bench/trusted_runner.py)
> Driving debt: [`docs/known-debts.md`](../known-debts.md) #57
> Driving packet: [`docs/moving forward/companion-bench-public-launch-packet.md`](../moving%20forward/companion-bench-public-launch-packet.md) §2.7

## 1. 问题

96 held-out scenario 对榜单可信度至关重要——但任何 submitter 自跑就会看到 prompt，**一旦泄露**，held-out 价值清零（[`docs/known-debts.md`](../known-debts.md) #57）。

## 2. 两种提交模式

### 2.1 Self-hosted（开放）

- submitter 在自己的 infra 上跑
- 只能用 `public-24` scenario（hash 公开）
- 结果落 **公开榜单**
- 成本：submitter 自付 API tokens
- 适合：想快速跑分 + 接受公开 24 限制的 submitter

### 2.2 Trusted-runner（held-out）

- submitter 提供 OpenAI-compat `base_url` + `api_key`（加密落 VZ 仓库）
- VZ 在自己 infra 上跑全 120 scenario（24 公开 + 96 held-out）
- 跑完只返回 verdict + 6 轴 scores；transcript **不返回**，跑完即从 VZ 文件系统删除
- 结果落 **held-out 榜单**
- 成本：见 [`companion-bench-cost-model-v0.md`](companion-bench-cost-model-v0.md) § 5
- 适合：想公开比较自家模型 + 愿意付费 + 接受 transcript 不可见的 submitter

## 3. Trusted-runner 凭证协议

### 3.1 凭证收集

submitter 上传 `submission.encrypted.json`：

```
{
  "submission_id": "<id>",
  "system_name": "<name>",
  "model_identifier": "<id>",
  "base_url": "<url>",
  "api_key_ciphertext": "<base64 of API key encrypted with VZ public key>",
  "system_prompt": "<text>",
  "generation_config": {...},
  "leaderboard_category": "open-weight | closed-api | bespoke"
}
```

VZ 公钥发布在 [`docs/external/companion-bench-trusted-runner-pubkey.asc`](companion-bench-trusted-runner-pubkey.asc)（待 #57 ACTIVE 时生成）。

### 3.2 凭证生命周期

| 阶段 | 操作 |
|---|---|
| 收到 | 解密 → 存入 VZ secrets vault（透明审计可读，但只读管理员可写） |
| 跑分中 | 调用 submitter 的 endpoint；每次调用记录到 cost ledger |
| 完成 | 从 secrets vault 删除（30 天后自动 purge） |
| 提交方下次跑 | 必须重新上传（VZ 不存长期凭证） |

## 4. Transcript 处置

- 跑分中：transcript 临时落 `artifacts/companion_bench_runs/<submission_id>/`
- 跑分完成：计算完 verdict 后立即删除（cron 每小时跑一次清理）
- ledger 记录：删除时写 sha256 + 删除时间到 ledger，证明已删

## 5. 计费

- VZ 收 $X / submission（见 cost model § 5）
- 收费包括：API tokens cost + VZ infra + 管理费
- submitter 收到 invoice 后付款，付款确认后结果上线 held-out 榜单

## 6. 失败 / 退款

- 跑分中 submitter endpoint 失败 → 全额退款
- VZ infra 失败 → 全额重跑 / 退款（submitter 选）
- 凭证泄露事件 → 见 [`companion-bench-heldout-leak-protocol.md`](companion-bench-heldout-leak-protocol.md)

## 7. 法律 & 合规

- VZ 与 submitter 签 NDA（凭证 + transcript）
- submitter 可任意时刻请求"跑分撤回"，从 held-out 榜单删除其结果（保留 ledger 记录）
- VZ 不公布任何 submitter 的 prompt 内容（即使在 sweep 报告里）

## 8. 退出标准

| 阶段 | 标准 |
|---|---|
| **SHADOW**（W4） | 本协议 + heldout-leak protocol 落档；`trusted_runner.py` --dry-run 可用 |
| **ACTIVE**（W6+） | VZ 公钥生成 + 加密上传链路 + secrets vault + 删 transcript cron + 计费链路全部 wire |

## 变更日志

- 2026-05-13: v0 SHADOW protocol 落档。
