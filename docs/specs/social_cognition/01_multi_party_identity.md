# Multi-Party Identity Spec

> Status: draft
> Last updated: 2026-05-02
> 对应需求: R16, R7, R-PE, R1, R8, R15

## 要解决的问题

当前 kernel 隐含一个匿名单一对话对象：`user_model`、`relationship_state`、`interlocutor_state`、memory scope 都是 flat 的。这足以支持一对一 companion v1，但无法表达多人生活场景中的基本事实：Alice 的偏好不等于 Bob 的偏好，和 Alice 的关系连续性不等于和 Bob 的关系连续性，私人记忆不能自动暴露给同一房间里的其他人。

R16 把"谁"变成社会认知层的第一性索引。任何 other-agent 状态都必须带 `interlocutor_id`，任何 memory / semantic update 都必须声明 subject / audience scope。

## 关键不变量

- `interlocutor_id` 是 other-agent 状态的必备键，不是 optional metadata。
- `primary` 只是单人兼容键，不是 cognitive truth。
- `user_model`、`relationship_state`、`interlocutor_state` 迁移为 keyed aggregate 时，旧 flat snapshot 只能在 SHADOW 期作为兼容视图存在。
- `MemoryEntry` 必须区分"关于谁"与"谁可见"，分别由 `subject_ids` 和 `audience_ids` 表达。
- 身份推断错误必须产生 social prediction error，不能被 renderer 文案掩盖。

## Owner / Timescale / Prediction Error / ETA Consumption

### Owner

`MultiPartyIdentityModule` 是唯一 owner，发布 `multi_party_identity` snapshot，并在 Phase 1 期间发布 `interlocutor_models` / `relationship_states` / `interlocutor_states` 的 keyed views。旧 `UserModelModule` 和 `RelationshipStateModule` 在迁移期只拥有 flat 兼容输出，不再重新推断多方身份。

### Timescale

- `online-fast`: 为当前 turn 绑定 `active_speaker_id`、`addressee_ids`、`subject_ids`、`audience_ids`。
- `session-medium`: 在 scene 结束时稳定 person continuity、identity merge/split、relationship continuity。
- `background-slow`: 通过 reflection evidence 提升 durable identity boundaries 和 long-term relationship anchors。
- `rare-heavy`: 只用于 offline identity-resolution artifact refresh，不在 live runtime 直接改基底。

### Prediction Error

Owner 在 action 前发布 social predictions：

- 当前 turn 的状态应写到哪个 `interlocutor_id`
- 哪些 memory 可以被当前 audience 使用
- 哪些 relationship / preference / boundary 属于 active speaker

以下 outcome 产生 typed social PE：

- wrong-person memory application
- private memory leaked to wrong audience
- Alice 的 preference 被用于 Bob
- relationship repair 写到错误 dyad
- identity merge/split 后与后续 evidence 冲突

### ETA Consumption

`multi_party_identity` 是 compact social state。`regime`、`temporal_abstraction`、`response_assembly`、`PromptPlanner` 可读它，但不得重新从 text 构造身份逻辑。长期适应发生在 controller code `z_t` 和 switch unit `beta_t` 空间；renderer 只表达已选 plan。

## 工程挑战

- 现有 snapshot 和 tests 大量假设 flat `user_model` / `relationship_state`。
- MemoryStore 当前只有 `track`，没有 subject / audience scope。
- ResponseContext 当前没有 active speaker / audience。
- 需要可回滚迁移：SHADOW keyed views 与 flat snapshot 并跑，证据稳定后切 ACTIVE。

## 算法候选

- 结构化 event envelope：host / service 明确传入 speaker / audience / subject。
- LLM structured proposal：仅用于 identity proposal，不作为 owner。
- Embedding / memory similarity：用于 identity continuity candidate，不直接改状态。
- PE-backed merge/split：identity conflict 由 owner 依据 outcome evidence 决定。

## 接口契约

```python
@dataclass(frozen=True)
class InterlocutorIdentity:
    interlocutor_id: str
    display_name: str | None
    aliases: tuple[str, ...]
    confidence: float
    evidence: tuple[str, ...]

@dataclass(frozen=True)
class MultiPartyIdentitySnapshot:
    active_speaker_id: str
    addressee_ids: tuple[str, ...]
    subject_ids: tuple[str, ...]
    audience_ids: tuple[str, ...]
    interlocutors: tuple[InterlocutorIdentity, ...]
    identity_predictions: tuple[SocialPrediction, ...]
    description: str
```

Planned field extensions:

- `MemoryEntry.subject_ids: tuple[str, ...]`
- `MemoryEntry.audience_ids: tuple[str, ...]`
- `ResponseContext.active_speaker_id: str`
- `ResponseContext.audience_ids: tuple[str, ...]`

## 与其他能力域的关系

- R17 ToM owners require R16 keying.
- R18 conversational role uses R16 identity fields.
- R19 common ground uses dyad / group identity keys.
- R20 group identity uses R16 membership.
- R-PE receives `social_prediction_error` from identity misattribution.

## 变更日志

- 2026-05-02: 初始 draft，作为 Social Cognition Learning Layer Phase 1 的 contract freeze。
