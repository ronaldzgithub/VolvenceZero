# Theory of Mind Spec

> Status: draft
> Last updated: 2026-05-02
> 对应需求: R17, R16, R11, R-PE, R1, R8, R15

## 要解决的问题

当前 `UserModelSnapshot` 把 stable preferences、working style hints、sensitive boundaries、durable goals 都压在同一类 `SemanticRecord` 上。它缺少认知科学中最基础的区分：对方相信什么、打算做什么、正在感受什么、长期偏好什么。这些不是同一类状态，不能共享一个 owner 或一个更新规则。

R17 把 Theory of Mind 拆成四个 learned owners，并要求每个 owner 都有自己的 prediction、PE、timescale 和 reconciliation 规则。

## 关键不变量

- `belief_about_other`、`intent_about_other`、`feeling_about_other`、`preference_about_other` 是不同 owner，不是 `UserModelSnapshot` 的四个字符串字段。
- 所有 ToM state 必须 keyed by `interlocutor_id`，依赖 R16。
- LLM structured output 只能产生 `SemanticProposal`，不能直接成为 ToM state。
- owner 必须使用 memory、confidence、time decay、conflict 和 prediction error reconciler 来维护稳定状态。
- belief / intent / feeling / preference 的 PE 不可混写。
- ToM owners 消费 Environment Event / role / identity 提供的 conversational frame；不得从 renderer 文案或 raw text 下游重建 speaker / audience / subject。

## Owner / Timescale / Prediction Error / ETA Consumption

### Owner

四个 owner 分别单写：

- `BeliefAboutOtherModule`
- `IntentAboutOtherModule`
- `FeelingAboutOtherModule`
- `PreferenceAboutOtherModule`

每个 owner 发布 keyed snapshot，`UserModelSnapshot` 在迁移后降级为兼容 aggregate / read model，不再拥有全部 ToM truth。

### Timescale

- `online-fast`: 接收当前 turn semantic proposals，更新 transient affect / intent hypotheses。
- `session-medium`: 在 scene 内根据 follow-through、repair、clarification、explicit feedback 校正 working model。
- `background-slow`: 通过 reflection writeback 将稳定 preference / belief / boundary 提升为 durable records。
- `rare-heavy`: 用于 offline ToM evaluator / calibrator refresh，不在 live runtime 直接更新 substrate。

### Prediction Error

每个 owner 发布不同 prediction：

- belief predicts how new information will be interpreted
- intent predicts likely next action or follow-through
- feeling predicts affective response / rapport movement
- preference predicts durable response style or boundary

outcome mismatch 生成 owner-specific social PE，例如 intent follow-through failure 不应污染 durable preference；affect prediction failure 不应覆盖 belief state。

### ETA Consumption

ToM snapshots 是 controller / regime / planner 的 compact advisories。它们可以影响 `z_t`、`beta_t`、regime priors 和 question budget，但长期策略更新仍在 latent controller space。Renderer 不读取用户文本判断 ToM；它只表达 planner 已选择的 social action.

## 工程挑战

- 旧 `UserModelModule` 既承担 profile summary 又承担 preference-ish state，需要拆职责。
- SemanticProposalAdapter 必须显式 target owner，不能把所有 profile event 都扔到 `user_model`。
- 需要 owner-specific evidence gates：false-belief、intent mismatch、affect misread、preference conflict。
- 迁移期必须避免 `response_assembly` 同时消费 old `user_model` 和 new owners 导致 double counting。

## 算法候选

- LLM structured proposal with typed target owner and confidence。
- Embedding similarity for belief / preference continuity candidates。
- PE-weighted record promotion: repeated low-error predictions increase stability。
- Conflict-aware decay: contradicted beliefs decay faster than durable preferences。
- Session-post reflection for durable ToM consolidation。

## 接口契约

```python
@dataclass(frozen=True)
class OtherMindRecord:
    record_id: str
    interlocutor_id: str
    summary: str
    detail: str
    confidence: float
    status: str
    source_turn: int
    prediction_error_refs: tuple[str, ...]
    evidence: str

@dataclass(frozen=True)
class BeliefAboutOtherSnapshot:
    records: tuple[OtherMindRecord, ...]
    active_predictions: tuple[SocialPrediction, ...]
    control_signal: float
    description: str
```

`IntentAboutOtherSnapshot`、`FeelingAboutOtherSnapshot`、`PreferenceAboutOtherSnapshot` mirror the same frozen contract but define owner-specific prediction / outcome vocabularies.

## 与其他能力域的关系

- R16 supplies `interlocutor_id` and audience / subject identity.
- R18 uses ToM state to distinguish addressee vs subject role consequences.
- R19 common ground uses belief owner outputs but does not own beliefs.
- R20 group state may aggregate ToM evidence but cannot rewrite individual ToM owners.
- R-PE / credit receive typed ToM prediction outcomes.
- Environment Interface supplies event provenance and conversational frame for ToM proposals; ToM owners reconcile proposals into learned state.

## 变更日志

- 2026-05-02: 补充 Environment Interface 依赖：ToM proposal 进入 owner 前必须带 canonical event / role context。
- 2026-05-02: 初始 draft，冻结 ToM owner decomposition as Social Cognition Learning Layer Phase 2。
