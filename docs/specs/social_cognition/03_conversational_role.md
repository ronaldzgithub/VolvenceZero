# Conversational Role Spec

> Status: draft
> Last updated: 2026-05-02
> 对应需求: R18, R16, R17, R3, R4, R-PE, R8, R15

## 要解决的问题

多人互动中，同一句话可能来自 Alice、说给 Bob、谈论 Carol、被 Dave 旁听。当前 `ResponseContext` 和 semantic owner 都假设 turn 是"用户对我说"，因此无法区分 addressee、subject、witness、overhearer。结果是 memory write、relationship update、boundary 判断、response mode 都容易写错对象。

R18 把 conversational role 作为每 turn 的 learned state，由 owner 发布，供 controller / regime / planner 消费。

## 关键不变量

- role 是 typed snapshot，不是 renderer 根据文本临时判断。
- active speaker、addressee、subject、witness、overhearer 是不同字段。
- role owner 只发布角色状态，不拥有 ToM / relationship / common ground 的事实。
- role mistake 必须产生 social PE。
- role 状态影响 latent social action selection，不得变成 prompt 模板分支。

## Owner / Timescale / Prediction Error / ETA Consumption

### Owner

`ConversationalRoleModule` 是唯一 owner，发布 `conversational_role` snapshot。它消费 R16 identity state、host event envelope、scene context、prior common ground 和 ToM summaries，但不写这些 owner。

### Timescale

- `online-fast`: 每 turn 绑定 speaker / addressee / subject / audience。
- `session-medium`: 稳定 scene 内参与结构，如谁在旁听、谁退出、谁成为共同目标成员。
- `background-slow`: 反思 recurring role patterns，例如某人经常是委托者、某人经常是 witness。

### Prediction Error

Role owner 在 action 前预测：

- 谁应该收到回应
- 谁会被回应影响
- 哪些 owner 应接收 semantic / memory write
- 当前 turn 是否需要 group-addressed rather than individual-addressed response

错误 addressee、误把 subject 当 speaker、给 witness 暴露私密记忆、或遗漏 group audience 会产生 role PE。

### ETA Consumption

`conversational_role` 影响 latent control: direct answer to addressee, repair to subject, grounding to group, or silence / deferral when overhearer constraints dominate. The metacontroller learns these switches in `z_t` / `beta_t` space. Renderer only surfaces the selected stance.

## 工程挑战

- Host / CLI 当前没有 multi-party event envelope，需要 default role scaffold。
- Memory / semantic write APIs 需要 role scope。
- `ResponseContext` 与 `ResponseAssemblySnapshot` 需要带 role summary，但不能成为 role owner。
- 需要 3-party evidence scenarios，而不是继续只跑一对一 companion transcript。

## 算法候选

- Explicit host event fields for speaker / addressee / audience。
- LLM structured role proposals for ambiguous language, reconciled by owner。
- Common-ground repair feedback as role PE signal。
- Learned role-switch priors in regime / temporal controller space。

## 接口契约

```python
@dataclass(frozen=True)
class ConversationalRoleSnapshot:
    active_speaker_id: str
    addressee_ids: tuple[str, ...]
    subject_ids: tuple[str, ...]
    witness_ids: tuple[str, ...]
    overhearer_ids: tuple[str, ...]
    group_audience_ids: tuple[str, ...]
    role_confidence: float
    active_predictions: tuple[SocialPrediction, ...]
    description: str
```

Default single-user turn:

```python
active_speaker_id = "primary"
addressee_ids = ("self",)
subject_ids = ("primary",)
witness_ids = ()
overhearer_ids = ()
```

## 与其他能力域的关系

- R16 supplies `interlocutor_id` keys.
- R17 supplies ToM summaries for role ambiguity resolution.
- R19 common ground uses role scope to decide dyad / group target.
- R20 group entity consumes role state for group-addressed turns.
- R3/R4 metacontroller consumes role summary as latent control input.

## 变更日志

- 2026-05-02: 初始 draft，冻结 role snapshot as Social Cognition Learning Layer Phase 3。
