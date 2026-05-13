# Companion Bench: Held-Out Leak Response Protocol v0

> Status: SHADOW
> Driving debt: [`docs/known-debts.md`](../known-debts.md) #57
> Driving packet: [`docs/moving forward/companion-bench-public-launch-packet.md`](../moving%20forward/companion-bench-public-launch-packet.md) §2.7

## 1. 范围

private held-out submodule（[`docs/external/companion-bench-heldout-bootstrap.md`](companion-bench-heldout-bootstrap.md)）一旦泄露，benchmark 价值归零（[`commercialization-assessment.md`](../business/commercialization-assessment.md) §10.2 反目标已写）。本 protocol 锁定泄露事件应对。

## 2. 泄露场景分类

| 场景 | 严重度 | 行动 |
|---|---|---|
| 单 submitter trusted-runner 凭证泄露（不影响 held-out scenario 本身） | 中 | 撤销该 submitter 凭证；强制重发；不影响榜单 |
| 单个 held-out scenario 文件泄露（如代码仓库 fork 误公开） | 高 | rotate 该 scenario；本季度榜单暂停；下季度复工 |
| 多个 held-out scenario / 整 submodule 泄露 | 极高 | 整个 held-out 轨道下线；rotate 全部 scenarios（约 6 个月工作量）；公开公告 |

## 3. 检测机制

| 检测 | 实施层 |
|---|---|
| GitHub secret scanning（held-out scenario 文本进 public PR） | GitHub 默认开启 |
| `tests/contracts/test_heldout_access_audit.py` AST 守门：任何 `heldout_loader` 调用必须有 audit logger | contract test |
| 每季度跑一次 web crawl 抓 google / github 看 held-out scenario hash 出现 | ops 季度运维 |

## 4. 事件响应流程

### Step 1: 确认（≤ 4h）

- 确认泄露事实 + 影响范围（哪些 scenario / 多少 submitter / 是否影响历史榜单）
- 关闭 held-out 仓库的 read 权限
- 通知 ops + 创始人

### Step 2: 公开声明（≤ 24h）

- 在 [`docs/external/companion-bench-rfc-v0.md`](companion-bench-rfc-v0.md) 顶部加 banner："held-out 第 N 季暂停，预计 X 月恢复"
- 邮件通知所有 trusted-runner submitter
- 不公开泄露细节（避免攻击者复用攻击模式）

### Step 3: rotate（≤ 6 周）

- reviewer 团队启动新 scenario 起草（参考 [#35](../known-debts.md) 季度治理 rotation 节奏）
- 加 paraphrase / 同义场景 / 新 family 增加多样性
- 重置 held-out submodule（旧 scenario 移到归档；新 scenario 进 active）

### Step 4: 复工（≤ 季度结束前）

- 新 held-out 上线
- 公开 changelog（不暴露具体 scenario）
- 邀请 trusted-runner submitter 重跑

## 5. 责任决策链

| 决策 | 责任人 |
|---|---|
| 是否触发 rotate | Tech Lead + 创始人 |
| 公开声明文案 | Tech Lead + 内容/PR |
| 撤销 submitter 凭证 | Ops |
| 退款 | Finance + 创始人 |

## 6. 复盘

每次事件完成后 30 天内 retrospective：
- 泄露根因
- 检测延迟
- response 节奏问题
- 改进 action items

复盘报告归档 `docs/external/companion-bench-incident-<id>.md`，公开（去敏感信息）。

## 7. 退出标准

本 protocol 一直处于 SHADOW（不希望被触发）。ACTIVE 触发时立即按 § 4 走。

## 变更日志

- 2026-05-13: v0 SHADOW protocol 落档。
