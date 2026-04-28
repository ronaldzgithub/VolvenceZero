# AAC Commitment Lifecycle Spec

> Status: draft
> Last updated: 2026-04-29
> 对应需求: R-PE, R8, R11, R14
> 来源: `docs/implementation/13_emogpt_prd_alignment_upgrade.md` Gap 7

## 要解决的问题

VolvenceZero 已有 `commitment` / `open_loop` / `plan_intent` 三个语义 owner，可以表达"已承诺要做什么"，但**缺一段显式的前向状态机**：

```
AI 倡导 (advocate) → 用户对齐 (align) → 进入承诺 (commit) → 跟进 (followup) → outcome 显式分类
```

EmoGPT v4 PRD §5.6 用 AAC（Advocacy-Alignment-Commitment）框架把这条路径明确化。本 spec 把这套需求映射到 VolvenceZero 第一性设计：

- **不**新建 AAC owner
- **不**用 LLM keyword 关键词判断 alignment 状态
- 把 advocacy / alignment / followup_policy / outcome_kind 全部加为 `commitment` owner 的内部 lifecycle 字段
- 让 alignment 由 `unknown` 跳到 `reject` 这种状态变化作为 R-PE 显式信号源

## 关键不变量

- `commitment` owner 仍是单写者；advocacy / alignment / outcome 的所有状态迁移**只能**通过 `SemanticProposal` typed path 进入 owner store
- `SemanticProposalOperation` 的 8 类 op 已足够覆盖状态转移；**禁止**新增 op
- 任何"用户表态" alignment 判定**只能**来自 LLM structured output 或 embedding similarity；**禁止**关键词匹配
- `commitment.last_outcome` 一旦写入必须带 `last_outcome_evidence != ""`，否则 contract test fail
- alignment 从 `agree` 转 `reject` 的状态变化是 PE 信号源（高 PE）；从 `unknown` 转 `agree` 不算 PE
- followup 调度优先级由 `commitment.followup_policy` 决定，**不**由关键词或 user_input 长度决定
- 不可绕过 `commitment` owner 直接写 `FollowupManager`；FollowupManager 是 lifeform 层"读 commitment + 排队 due tick"，不是 owner

## 工程挑战

- 让 LLM extraction 可靠输出 alignment_state（structured output schema 必须严格）
- 让 reject 状态真正流转到 `repair_and_deescalation` regime，而不是停在 commitment owner 内部
- 让 outcome enum 在 reflection writeback 路径完整闭合
- 不让"AI 想推进 commitment"和"用户实际同意"混淆——advocacy 是 AI 一端的状态，alignment 是 user 一端的状态

## 算法候选

来自 `docs/next_gen_emogpt.md`：

| 算法 | 用途 |
|---|---|
| ETA `β_t` 切换单元 | reject 触发 regime 从 problem_solving 切到 repair_and_deescalation |
| Internal RL z_t 学习 | metacontroller 学到"alignment_state 与 abstract action 选择的相关性" |
| R-PE 主链 | alignment 反向跳变作为高 PE 事件，进入 PE schedule |

## 接口契约

### `commitment` owner snapshot 字段扩展

`vz-cognition/semantic_state/__init__.py` 内 `CommitmentEntry` 增加：

```python
class AdvocacyState(str, Enum):
    NOT_READY = "not_ready"     # 证据/把握不足，AI 还没准备好提议
    READY = "ready"             # AI 内部已 ready 但还没说出口
    PROPOSED = "proposed"       # AI 已向用户提出

class AlignmentState(str, Enum):
    UNKNOWN = "unknown"         # 用户还没明确表态
    AGREE = "agree"             # 同意
    MODIFY = "modify"           # 同意但要改 / 加条件
    REJECT = "reject"           # 不同意

class FollowupPolicy(str, Enum):
    GENTLE_CHECKIN = "gentle_checkin"
    DEFER_ONLY = "defer_only"

class CommitmentOutcomeKind(str, Enum):
    PROGRESSED = "commitment_progressed"
    COMPLETED = "commitment_completed"
    STALLED = "commitment_stalled"
    REJECTED = "commitment_rejected"
    FOLLOWUP_NO_RESPONSE = "followup_no_response"

@dataclass(frozen=True)
class CommitmentEntry:                 # 已有 dataclass 扩展
    # ... 现有字段保留 ...
    advocacy_state: AdvocacyState = AdvocacyState.NOT_READY
    alignment_state: AlignmentState = AlignmentState.UNKNOWN
    followup_policy: FollowupPolicy = FollowupPolicy.GENTLE_CHECKIN
    last_outcome: CommitmentOutcomeKind | None = None
    last_outcome_evidence: str = ""
    last_outcome_at_turn: int = -1
```

向后兼容：所有字段都有默认值，旧持久化数据加载后默认 `NOT_READY / UNKNOWN / GENTLE_CHECKIN / None`。

### 合法状态转移

| 当前 advocacy | 合法迁移目标 | 触发 |
|---|---|---|
| `NOT_READY` | `READY` | 证据 + plan_intent + AI 把握度 ≥ 阈值 |
| `READY` | `PROPOSED` | AI 在 turn 中实际提议（response 含 commitment_signal） |
| `PROPOSED` | `READY` | 用户 modify 后 AI 重新内部 ready |

| 当前 alignment | 合法迁移目标 | 触发 |
|---|---|---|
| `UNKNOWN` | `AGREE` / `MODIFY` / `REJECT` | LLM extraction structured output |
| `AGREE` | `MODIFY` / `REJECT` | LLM extraction（用户改主意） |
| `MODIFY` | `AGREE` / `REJECT` | 同上 |
| `REJECT` | `MODIFY` / `AGREE` | 用户修复后回归 |

非法转移（如 `REJECT` → 直接 `commit_completed`）由 `commitment` owner 在 merge 阶段拒绝，写入 `SemanticProposal.rejected_reason`。

### LLM extraction 协议

`vz-cognition/semantic_state/prompts/extraction.md`（现有 prompt template）扩展，输出 schema 增加：

```json
{
  "commitment_id": "string",
  "advocacy_state": "not_ready|ready|proposed",
  "alignment_state": "unknown|agree|modify|reject",
  "alignment_evidence": "string (max 200 chars)",
  "alignment_confidence": "float [0, 1]"
}
```

`SemanticProposalRuntime` 在 LLM 输出后强制：

- `alignment_confidence < 0.6` 时 fallback 到 `OBSERVE` op（不写状态）
- `alignment_evidence` 必须 non-empty 才 accept

`AdapterSemanticProposalRuntime` 不允许产 alignment_state 字段——adapter 是结构化字段映射器（tool_result / profile / task），不该判定用户对话语义。

### Reflection writeback outcome 闭合

`vz-cognition/reflection/writeback.py:PolicyConsolidation` 增加：

```python
@dataclass(frozen=True)
class CommitmentOutcomeRecord:
    commitment_id: str
    outcome_kind: CommitmentOutcomeKind
    evidence: str
    detected_at_turn: int

@dataclass(frozen=True)
class PolicyConsolidation:
    # ... 现有字段保留 ...
    commitment_outcomes: tuple[CommitmentOutcomeRecord, ...] = ()
```

session-post slow loop 把 outcome 写回 commitment owner 的 `last_outcome`；同时把 outcome 作为 RewardRecord 喂给 ETA / regime（R9 跨层信用）。

### R-PE 信号接入

`vz-cognition/prediction/error.py` 在计算 prediction_error 时新增一项 contribution：

```python
def compute_alignment_pe_contribution(
    *,
    prev_alignment: AlignmentState,
    curr_alignment: AlignmentState,
) -> float:
    # AGREE → REJECT 是高 PE
    # AGREE → MODIFY 是中 PE
    # UNKNOWN → 任意 是低 PE（合法初次表态）
    transition_severity = {
        (AlignmentState.AGREE, AlignmentState.REJECT): 0.9,
        (AlignmentState.AGREE, AlignmentState.MODIFY): 0.45,
        (AlignmentState.MODIFY, AlignmentState.REJECT): 0.7,
        (AlignmentState.PROPOSED if hasattr(AlignmentState, "PROPOSED") else None,
         AlignmentState.REJECT): 0.6,
    }
    return transition_severity.get((prev_alignment, curr_alignment), 0.0)
```

注意：上面是示意；真实实现**禁止 hasattr 滥用**（红线 C），用 if/elif 显式枚举。

### Followup 集成

`lifeform-core/followup_manager.py:FollowupManager.ingest_commitments()` 改为按 `commitment.followup_policy` 分流：

- `GENTLE_CHECKIN`：默认 due delay
- `DEFER_ONLY`：延迟 ≥ 2x 默认 delay；用户主动提及前不主动 surface

## ETA / NL 集成

- **R-PE**：alignment reject 是慢于"per-turn substrate PE"但快于"reflection PE"的中频 PE 源。它和 vitals drive PE 是两种性质不同的 R-PE 输入：vitals 是连续标量、alignment 是离散状态跳变
- **R14 regime**：`alignment_state == REJECT` 在 metacontroller 输入向量中是显式特征；regime selection_weights 在反复观察后会学到"reject → repair_and_deescalation"
- **R11 内部状态显式化**：advocacy_state / alignment_state / followup_policy / last_outcome 都是 nameable + publishable，是 R11 的直接落地

## 当前 proof surface

引入后必须能证明的命题：

1. `alignment-reject-triggers-pe-spike`
   - reject 出现 turn 的 PE magnitude 显著高于 baseline turn（mean + 1σ 以上）
   - acceptance：`tests/test_prediction_error.py::test_alignment_reject_pe_spike`
2. `reject-leads-to-repair-regime`
   - 多轮 scenario 下 reject 后下一 turn `regime.identity.regime_id == repair_and_deescalation` 的转移率 ≥ 50%（calibrated regime weights 启用时）
   - acceptance：`tests/lifeform_e2e/test_aac_repair_path.py`
3. `outcome-evidence-required`
   - 任何 `last_outcome != None` 必须带 `last_outcome_evidence != ""`
   - acceptance：`tests/contracts/test_aac_lifecycle.py::test_outcome_requires_evidence`
4. `no-keyword-alignment-extraction`
   - grep 仓库找不到 `if "agree" in user_text` / `if "好的" in user_text` 等关键词判定
   - acceptance：`tests/contracts/test_no_keyword_routing.py`
5. `illegal-transition-rejected`
   - `REJECT → COMPLETED` / `NOT_READY → PROPOSED` 等非法转移在 owner merge 阶段被拒绝
   - acceptance：unit test 矩阵覆盖所有非法转移

## 接口契约（公开数据流向）

**消费的输入**（commitment owner）：

- `SemanticProposalBatch`（含 advocacy / alignment proposal）
- `prev CommitmentSnapshot` 自身

**产出的输出**：

- `CommitmentSnapshot.value`：含完整 lifecycle 字段
- 跨 owner 影响：
  - `prediction_error` 把 alignment 跳变作为信号源
  - `regime` 把 alignment_state 加入 metacontroller 输入特征
  - `FollowupManager` 按 followup_policy 调度
  - `case_memory` 按 outcome_kind 记录历史

## 与其他能力域的关系

| 关系 | 能力域 | 说明 |
|---|---|---|
| 依赖 | 语义状态一等 Owner | commitment 是 9 个语义 owner 之一，本 spec 是该 spec 的细化 |
| 依赖 | Prediction Error 主链 | alignment 跳变是 PE 信号源 |
| 协作 | 认知 Regime | reject → repair_and_deescalation 转移 |
| 协作 | 信用分配 | outcome enum 进入 RewardRecord，跨层信用 |
| 协作 | 评估体系 | F3（关系连续性）观察 reject → repair 转移率 |

## 回滚

字段全部带默认值，向后兼容。

回滚路径：

- 如果 LLM extraction 不稳定：`SemanticProposalRuntime.extract_alignment = False`，fallback 到 `UNKNOWN`，AAC 状态机仍存在但不主动迁移
- 如果 PE spike 太频繁：`compute_alignment_pe_contribution` 降级返回 0，PE 主链回到不含 alignment 信号
- 如果 followup_policy 行为有问题：`FollowupManager.use_commitment_policy = False`，回到旧均匀延迟

每条独立可回滚，互不影响。

## 变更日志

- 2026-04-29：初始版本，对应 `docs/implementation/13_emogpt_prd_alignment_upgrade.md` Gap 7 设计冻结。
