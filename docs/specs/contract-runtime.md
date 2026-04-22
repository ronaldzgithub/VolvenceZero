# 契约式运行时 Spec

> Status: draft
> Last updated: 2026-04-20
> 对应需求: R8, R11, R15

## 要解决的问题

如何让系统在持续适应的同时保持可调试、可检查、可回滚？这是实现其他能力域的**必要脚手架**。

## 关键不变量

- 每个运行时区域有唯一主要 owner
- 跨模块交换通过公共快照，消费者不重建生产者内部状态
- 快照不可变（frozen dataclass）
- 内部状态可命名、可发布、可检查
- 系统通过有界增量包演进，不一次性替换整个架构

## 工程挑战

- 设计模块间通信的快照/契约机制
- 确保内部状态可命名、可发布、可检查（R11）
- 设计增量演进机制：每个新自适应层有明确 owner，rollout 可逆（R15）
- 快照不可变性保证

## 算法候选

契约式运行时主要是工程架构设计，不直接对应 NL/ETA 算法。但其设计原则源自：
- R8 的"快照优先、契约优先"
- R11 的"可学习的内部状态表示"
- R15 的"迁移必须保持可解释性和可回滚"

## 接口契约

### 快照基类

```python
@dataclass(frozen=True)
class Snapshot:
    slot_name: str          # 全局唯一
    owner: str              # 唯一所有者
    version: int            # 单调递增
    timestamp_ms: int
    value: Any              # frozen dataclass
```

```python
class WiringLevel(str, Enum):
    DISABLED = "disabled"
    SHADOW = "shadow"
    ACTIVE = "active"
```

### 模块基类

```python
class Module(ABC):
    slot_name: str
    owner: str
    value_type: type[Any]
    dependencies: tuple[str, ...] = ()
    wiring_level: WiringLevel = WiringLevel.ACTIVE
    async def process(self, upstream: dict[str, Snapshot]) -> Snapshot
    async def process_standalone(self, **kwargs) -> Snapshot  # 独立调用模式
```

### 编排器

- Wave 级调度：协调一轮交互中各模块的执行顺序
- 快照传播：收集快照，构建 upstream dict
- 事件分发：异步通知
- 后台任务管理：慢反思等异步任务

### 编排器约束

- 编排器只调用模块公开的 `process()` 契约，不调用模块内部私有方法
- 不持有模块的内部状态

### 守卫与运行时视图

P00 运行时内核固定以下最小守卫和视图：

- `OwnershipGuard`：slot → owner 唯一性和版本单调递增
- `DependencyGuard`：模块只能消费声明的 upstream slot
- `SchemaGuard`：发布值必须符合声明的 frozen dataclass schema
- `ImmutabilityGuard`：发布后消费前校验 value hash 不变
- `UpstreamView`：对模块暴露带守卫的上游快照视图，缺失 slot 统一返回 runtime placeholder snapshot

### 当前实现补充

- substrate owner 现已区分三层：`OpenWeightResidualRuntime`（真实 runtime / hook 所有者）、`OpenWeightResidualStreamSubstrateAdapter`（对外发布稳定 `SubstrateSnapshot`）、`OpenWeightResidualInterventionBackend`（owner-side residual control 执行位点）
- 这意味着未来接 Hugging Face / 其他 open-weight backend 时，只需实现 runtime 契约，不需要改 temporal / internal RL / evaluation 的公共消费面
- 当前 `TransformersOpenWeightResidualRuntime` 已实现 Hugging Face causal LM 的中层 block forward hook capture / intervention；runtime owner 负责 hook 层选择、冻结边界和控制投影、模型家族 block 解析，以及稳定 feature summary 的发布，消费者继续只读取公共快照
- 当前 runtime owner 已显式支持 `SubstrateFallbackMode`：`allow-builtin` 允许回退到内置 tiny transformers runtime，`deny` 在首选 open-weight runtime 不可用时 fail closed；评估/production-like 路径应优先使用 `deny`
- 默认 `AgentSessionRunner` / CLI 已切换到真实 `TransformersOpenWeightResidualRuntime` 路径；当首选 HF 模型不可用且 fallback mode 允许时，回退到内置 tiny transformers runtime，而不是 synthetic runtime，保证默认主链仍消费真实 hookable residual substrate
- 当前 substrate 区域已新增正式 `substrate_self_mod` owner：它消费 `substrate + evaluation + prediction_error`，发布 machine-readable 的 online-fast substrate delta proposal / gate preview / parameter-change telemetry；真正的 apply / rollback 仍只通过 substrate runtime owner surface 执行，避免 `session` / `joint_loop` 直接成为 substrate 第二 owner。默认 frozen-substrate doctrine 下，该 owner 只发布 proposal / evidence，不触发 live substrate mutation
- `FinalRolloutConfig` 当前默认采用 widened application rollout：`case_memory`、`strategy_playbook`、`experience_fast_prior`、`experience_consolidation` 默认随主链开启；其中 `experience_consolidation` 继续是 session-owned post surface，而不是 final wiring DAG 中的第二 owner
- slow reflection 现通过 typed `TemporalPriorUpdate` 提案写回 temporal owner；编排层只负责 target-specific gate + audit + 调用 owner 的 apply surface，不重建 metacontroller 内部状态
- agent session 现允许通过 `substrate_adapter_factory(user_input, turn_index)` 注入 substrate adapter；表达层响应生成只消费 richer distilled context，不再持有完整 runtime snapshot dict，减少跨 event loop 的隐式耦合
- 当前主链已新增正式 `prediction_error` owner/slot，公共交换固定为 `evaluated_prediction -> actual_outcome -> next_prediction -> error`
- `memory` / `regime` / `credit` / `reflection` / `temporal` 已直接消费 `prediction_error`；`evaluation` 只在 final wiring 中追加 prediction-error evidence，保持 readout 定位
- 当前 temporal 区域已从单 owner 扩展为 staged multi-owner contract：`world_temporal`、`self_temporal`、`world_temporal_consolidation`、`self_temporal_consolidation` 各自拥有独立 slot；`temporal_abstraction` 由 `TemporalAggregateModule` 作为公共聚合 owner 对下游发布兼容快照
- 这类 aggregate slot 只允许发布 compact public state；不得反向成为底层 track owner 的第二所有者
- 当前 session-post slow loop 也已升级为正式运行时 surface：`session_post_slow_loop` 由独立 owner 发布 queue state 与 recent completion summaries；`AgentSessionRunner` 负责驱动该 owner 刷新。默认口径下它仍保持 report/shadow surface，而 `experience_consolidation` 作为其下游 session-owned public snapshot 对外暴露
- 当前应用层第一阶段也已新增正式 surface：`retrieval_policy`、`domain_knowledge`、`boundary_policy` 作为独立 owner 发布在线检索控制、专业事实证据与边界判断；response/evaluation 只能消费这些公共快照，不允许反向读取 owner 私有知识存储
- 当前应用层第二阶段已新增 `case_memory` surface：它作为 `memory` 的 sibling owner 发布 compact case hits、problem patterns 与 risk markers；该 surface 只服务 retrieval mix 和 evaluation evidence，不允许把案例经验重新折叠回 `memory` 主快照。当前默认口径下该 surface 进入 active application chain
- 当前应用层第三阶段已新增 `strategy_playbook`、`experience_consolidation` 与 `experience_fast_prior`：前者作为 turn-time 公共 slot 发布 problem-pattern-level strategy priors，`experience_consolidation` 作为 session-post 公共 surface 发布 machine-readable experience deltas、typed `ApplicationPriorUpdate` 与 writeback report，`experience_fast_prior` 负责把 delayed credit 压成 fast-path 可消费 bias。三者都不得越权成为 `temporal` / `regime` / `session_post_slow_loop` 的第二 owner；当前默认口径下 `strategy_playbook` 与 `experience_fast_prior` 进入 active chain，`experience_consolidation` 继续保持 session-owned active public surface
- application prior 的真正 apply 属于 owner-side post-processing：它必须先通过 `EvolutionJudgement` 与 target-specific credit gate，再由 session owner 驱动 writeback helper 调用 application owners 的 apply surface，不能进入 turn-time direct dependency DAG
- 当前应用层第四阶段已新增 application rare-heavy checkpoint/state：它不作为 turn-time slot 发布，而是沿现有 rare-heavy artifact/review/import/rollback 链由 session owner 管理，并把离线 domain bias、case cluster 与 distilled playbook 通过 `retrieval_policy` / `case_memory` / `strategy_playbook` 的公共快照间接体现在 fast path 中。底层 pipeline / joint-loop 不回收 application owner 身份
- 当前表达层已新增正式 `response_assembly` surface：它读取 `regime`、`temporal_abstraction`、`memory`、`reflection`、`domain_knowledge`、`case_memory`、`strategy_playbook`、`boundary_policy`，发布 compact prompt residue、generation constraints 与 numeric control。`session` / `response` / runtime 只能消费该公共 surface，不应继续在下游重建 knowledge/case/playbook/boundary 的表达语义
- 当前 `retrieval_policy` 还应被理解为 **compact retrieval control surface**：`ETA` / temporal 只通过该 surface 对 knowledge/experience owners 施加影响，而不吸收知识库/经验库本体。应用层后续若引入 learned readout，也应替换在 readout seam 内，而不是让 `RetrievalPolicyModule` 或 `temporal` owner 直接回收知识/经验所有权
- 当前 session-post 侧的 application prior proposal 也应保持为 **owner-side helper**：慢层经验可以生成 `ApplicationPriorUpdate` 提案，但它只能沿 `experience_consolidation -> experience_fast_prior -> owner-side apply` 的公共链回流，不能形成新的 `session -> temporal` 或 `evaluation -> temporal` 旁路

### 直接依赖 vs enrichment

当前运行时需要明确区分两层关系：

1. **direct dependency**：模块在 `process()` 中通过 upstream dict 直接声明并读取的 slot
2. **enrichment / post-processing**：编排层或 final wiring 在主传播完成后额外读取某些结果，生成附加 evidence / report / writeback

这两层关系不能混写，否则会误导 reader 以为：

- 某模块在 runtime DAG 中直接依赖了一个其实只在 post-processing 才读取的 slot
- 或某个 report/evidence 工件也是正式 slot owner

当前最典型的例子：

- `evaluation` 的 direct dependency 仍是其模块声明的 upstream slots
- 但 final wiring 会在 propagation 之后额外读取 `prediction_error`、joint-loop result、writeback result，为 `evaluation` 追加 evidence
- 这属于 **evaluation enrichment**，不是 `EvaluationModule.process()` 的 direct dependency
- 同理，`substrate_self_mod` 的 direct dependency 是 `substrate + evaluation + prediction_error`；而 online-fast substrate checkpoint 的真正应用、回滚和 credit audit 属于 session/joint-loop 在 propagation 之后执行的 owner-side apply/enrichment，不意味着 `SubstrateModule.process()` 或 `EvaluationModule.process()` 直接持有 substrate runtime 写权限。在默认 doctrine 下，这些 apply path 还必须额外通过 frozen-substrate guard

### 内部状态发布（R11）

每个模块必须能命名和发布其内部状态，包含：
- 活跃动机和张力
- 候选路径或策略
- 不确定性、模糊性和开放问题
- 用户状态估计、关系状态估计
- 当前抽象控制 regime
- 预期的下一个测试或信号

**详细 schema**：见 `docs/DATA_CONTRACT.md`

## 与其他能力域的关系

| 关系 | 能力域 | 说明 |
|------|--------|------|
| 被依赖 | **所有其他能力域** | 契约式运行时是所有能力域的基础设施 |
| 协作 | 调试体系 | 契约守卫（Layer 2）验证运行时不变量 |

## 变更日志

- 2026-04-20: 新增“direct dependency vs enrichment”边界说明，明确 `evaluation` 对 `prediction_error` 的 final-wiring evidence append 属于 post-processing，而非模块 direct dependency
- 2026-04-06: P18 Propagation Topo-Sort + Guard Closure: propagate() now auto-sorts modules by declared dependencies (topo_sort_modules). Cycle detection via detect_dependency_cycle; cycles fall back gracefully to input order. Post-propagation guard closure verifies immutability of all published snapshots. CyclicDependencyError added to runtime contract errors.
- 2026-04-08: 默认主链切到真实 transformers substrate；`reflection` / `temporal` 默认 ACTIVE；slow reflection 新增 typed `TemporalPriorUpdate` 写回 temporal owner，并带 target-specific gate / audit
- 2026-04-06: 补充 open-weight runtime / adapter / intervention backend 三层契约，以及 session 级 substrate 注入点
- 2026-03-25: 初始版本，从 SYSTEM_DESIGN.md 和 DATA_CONTRACT.md 提取
