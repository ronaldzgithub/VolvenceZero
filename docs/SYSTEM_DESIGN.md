# EmoGPT Next-Gen — 系统设计文档

> Status: draft
> Version: 0.1
> Last updated: 2026-03-25
> Source: `next_gen_emogpt.md`（唯一设计源头）、`docs/prd.md`（工程分解）

---

## 1. 设计总纲

EmoGPT 是一个**有界、契约驱动、持续适应的认知代理**。核心产品价值是**关系与主体性**（EQ + 信任），而非单纯的智力（IQ）。

系统融合两个互补的理论基础：

| 理论 | 贡献 |
|------|------|
| **Nested Learning (NL)** | 系统级教义：多时间尺度学习、连续记忆谱、嵌套自适应 |
| **Emergent Temporal Abstractions (ETA)** | 缺失的动作机制：发现并强化时间抽象的内部控制器 |

**四条核心信念**：

1. 系统不是单一静态模型 + prompt
2. 系统不仅在输出 token 层学习
3. 系统不将记忆、优化、控制视为独立世界
4. 系统是显式分层的自适应有机体，具有快、中、慢学习循环

---

## 2. 五圈架构

系统从内到外分为五个同心圈层，每一圈有明确的职责边界和所有权。问题定位和修改应从内圈向外追溯根因。

```
┌─────────────────────────────────────────────────────────────────┐
│ 圈 5: 元认知层 (Meta-Cognition)                                  │
│   评估门控 · 自修改规则 · 安全护栏 · 回滚机制                      │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ 圈 4: 表达层 (Expression)                                    │ │
│ │   LLM 生成 · 语言风格 · 话语组装 · 输出门控                    │ │
│ │ ┌─────────────────────────────────────────────────────────┐ │ │
│ │ │ 圈 3: 记忆层 (Memory)                                    │ │ │
│ │ │   连续记忆谱 · 慢反思 · 知识沉淀 · 检索                    │ │ │
│ │ │ ┌─────────────────────────────────────────────────────┐ │ │ │
│ │ │ │ 圈 2: 认知核 (Cognitive Core)                        │ │ │ │
│ │ │ │   Panorama · ETA · Metacontroller · MethodEngine     │ │ │ │
│ │ │ │ ┌─────────────────────────────────────────────────┐ │ │ │ │
│ │ │ │ │ 圈 1: 生命核 (Life Core)                         │ │ │ │ │
│ │ │ │ │   Body · NeedsEngine · 稳态驱动 · 宪法层          │ │ │ │ │
│ │ │ │ └─────────────────────────────────────────────────┘ │ │ │ │
│ │ │ └─────────────────────────────────────────────────────┘ │ │ │
│ │ └─────────────────────────────────────────────────────────┘ │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 圈 1: 生命核 (Life Core)

**职责**：提供系统的唯一原动力。需求偏离 → 张力 → 动机 → 决策 → 行动 → 反馈 → 奖励 → 学习。绕过此链路硬编码行为 = 摧毁主体性。

**核心组件**：

| 组件 | 职责 |
|------|------|
| **Body** | 维护生理状态向量（能量、社交需求、好奇心等），按时间衰减产生内部扰动 |
| **NeedsEngine** | 将 Body 状态映射为需求偏离度和张力信号 |
| **Constitution** | 宪法层，定义不可违反的安全约束和价值底线 |

**关键不变量**：
- 生命体自身是最重要的扰动源——Body 衰减、好奇心、记忆衰减、预测误差等内部扰动与外部输入走同一套机制
- Constitution 不可违反，是所有行为的硬约束

**所有者**：HomeostasisModule

### 2.2 圈 2: 认知核 (Cognitive Core)

**职责**：基于生命核的动机和外部输入，进行情境建模、策略选择和时间抽象控制。

**核心组件**：

| 组件 | 职责 |
|------|------|
| **PanoramaModule** | 实时情境综合器——双引擎决策（Panorama-A 管用户问题，Panorama-B 管 AI 策略） |
| **ETAManager** | 双轨经验管理——ETA-World 管世界经验（WHAT），ETA-Self 管策略经验（HOW-WHEN） |
| **Metacontroller** | 在残差流上发现和执行时间抽象控制——切换单元 β_t、控制器代码 z_t |
| **MethodEngine** | 方法候选集——提供可选的策略/方法/路径供决策层选择 |

**策略骨架**：
- **ETA 双轨**管经验：ETA-World 管世界经验 WHAT，ETA-Self 管策略经验 HOW-WHEN
- **Panorama 双引擎**管当下决策：Panorama-A 管用户问题，Panorama-B 管 AI 策略
- **Metacontroller** 在残差流上发现和执行时间抽象控制

**双引擎隔离**：
- Panorama-A 与 Panorama-B 是同一个 PanoramaModule 内的两个 DecisionFrame 实例，通过 `engine_type`（`"user_problem"` / `"ai_strategy"`）区分
- 两者共享 snapshot 发布通道（PanoramaSnapshot），但 Options/Dimensions 独立运行
- Panorama-A 的输出可作为 Panorama-B 的上下文输入，但反向不行

**双轨隔离**：
- ETA-World 与 ETA-Self 共享同一个 GoalGraph 基础设施，通过 `GoalNode.track` 字段（`"world"` / `"self"`）软隔离
- 路由、选择、信用回写时必须传递 `track` 参数，禁止跨轨道混合操作（`track=None` 全量检索除外）

### 2.3 圈 3: 记忆层 (Memory)

**职责**：实现跨时间尺度的连续记忆谱，通过反思将经验沉淀为持久结构。

**核心组件**：

| 组件 | 职责 |
|------|------|
| **MemoryOS** | 连续记忆系统——管理从瞬态到持久的完整记忆谱 |
| **SlowThinkingWorker** | 异步慢反思——读取交互轨迹，产出记忆沉淀和策略沉淀 |

**记忆层级**：

| 层级 | 内容 | 更新频率 | 算法对应 |
|------|------|----------|----------|
| **瞬态工作状态** | 当前 wave/scene/frame 状态 | 每轮 | CMS 最高频层 |
| **会话情景状态** | 近期交互轨迹和未解张力 | 每场景 | CMS 中频层 |
| **持久语义记忆 (L2)** | 稳定的用户/自我/关系/世界知识（权威卡片） | 慢反思后 | CMS 低频层 |
| **派生索引 (L1)** | 可重建的检索辅助、摘要、任务投影 | 按需重建 | 派生结构 |

**关键不变量**：
- L2 权威卡片产出来自**慢思考反思**（SlowThinkingWorker），不是逐句同步提取
- 记忆写入通过正式 owner 和 API，不可绕过
- 各层级有不同更新频率、晋升和衰减规则

### 2.4 圈 4: 表达层 (Expression)

**职责**：将认知核的决策转化为自然语言输出。

**核心组件**：

| 组件 | 职责 |
|------|------|
| **ResponseCompiler** | 将决策状态、记忆上下文、方法指令编译为 LLM prompt |
| **RuleGate** | 输出门控——检查生成内容是否符合 Constitution 和安全约束 |

**关键不变量**：
- LLM 是表达层，不要用 prompt 技巧替代 RL 信号或认知核路由
- 不在表达层用 if/else 或 prompt 掩盖上游问题

### 2.5 圈 5: 元认知层 (Meta-Cognition)

**职责**：评估、门控、自修改和安全保障。

**核心组件**：

| 组件 | 职责 |
|------|------|
| **EvaluationGate** | 6 族评估指标的计算和门控 |
| **SelfModificationGate** | 门控自修改——定义什么可在线改、什么需后台验证、什么需离线重训练 |
| **RollbackManager** | 回滚机制——确保每个自适应层的变更可逆 |

**门控规则**：

| 修改类型 | 门控级别 | 示例 |
|----------|----------|------|
| 在线可改 | 无需审批 | 检索权重、路由策略、控制器参数微调 |
| 后台验证 | 异步验证后生效 | 策略先验更新、反思启发式、记忆提升阈值 |
| 离线重训练 | 需要离线管线 | 基础模型参数、CMS 低频层大幅更新 |
| 人工审核 | 需要显式批准 | Constitution 变更、安全护栏调整 |

---

## 3. 核心类全景图

### 3.1 运行时骨架

```
                          ┌──────────────────┐
                          │   Orchestrator   │
                          │  (wave 级调度)    │
                          └────────┬─────────┘
                                   │
                          ┌────────▼─────────┐
                          │   CycleBoard     │
                          │ (模块间通信总线)   │
                          └────────┬─────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
    ┌─────────▼──────────┐ ┌──────▼───────┐ ┌─────────▼──────────┐
    │ HomeostasisModule  │ │PanoramaModule│ │    MemoryOS        │
    │ (生命核)            │ │ (认知核-决策) │ │   (记忆层)          │
    │  ├─ Body           │ │ ├─ FrameA    │ │  ├─ WorkingMemory  │
    │  ├─ NeedsEngine    │ │ ├─ FrameB    │ │  ├─ EpisodicStore  │
    │  └─ Constitution   │ │ └─ Options   │ │  ├─ DurableStore   │
    └────────────────────┘ └──────────────┘ │  └─ DerivedIndex   │
                                            └────────────────────┘
    ┌────────────────────┐ ┌──────────────┐ ┌────────────────────┐
    │   ETAManager       │ │MethodEngine  │ │SlowThinkingWorker  │
    │ (认知核-经验)       │ │(认知核-方法)  │ │   (慢反思)          │
    │  ├─ GoalGraph      │ │ ├─ Methods   │ │  ├─ MemoryConsolid │
    │  ├─ EmbedRouter    │ │ └─ Regimes   │ │  └─ PolicyConsolid │
    │  └─ CreditLedger   │ └──────────────┘ └────────────────────┘
    └────────────────────┘
```

### 3.2 模块基类

所有自治模块继承统一基类 `AutonomousModule`：

```python
class AutonomousModule(ABC):
    """所有自治模块的基类。

    契约：
    - process() 只接收 CellSnapshot 字典，不持有/import/调用其他 AutonomousModule
    - 返回自身的 CellSnapshot（不可变）
    - 可持有自己管辖的底层组件（非 AutonomousModule）
    """

    @abstractmethod
    async def process(
        self,
        wave_id: str,
        upstream: Dict[str, CellSnapshot],
    ) -> CellSnapshot:
        ...
```

**模块间零直接调用**：`process(wave_id, upstream)` 只接收 snapshot，不持有/import/调用其他 AutonomousModule。

### 3.3 模块清单

| 模块 | 圈层 | 职责 | 持有的底层组件 |
|------|------|------|---------------|
| `HomeostasisModule` | 圈 1 | 稳态驱动 | Body, NeedsEngine, Constitution |
| `PanoramaModule` | 圈 2 | 实时情境综合与决策 | DecisionFrame×2, OptionSet, DimensionSet |
| `ETAManager` | 圈 2 | 经验管理与路由 | GoalGraph, EmbeddingRouter, CreditLedger |
| `MethodEngine` | 圈 2 | 方法候选集管理 | MethodRegistry, RegimeRegistry |
| `MemoryOS` | 圈 3 | 连续记忆系统 | WorkingMemory, EpisodicStore, DurableStore, DerivedIndex |
| `SlowThinkingWorker` | 圈 3 | 异步慢反思 | MemoryConsolidator, PolicyConsolidator |
| `ExpressionModule` | 圈 4 | 响应编译与输出 | ResponseCompiler, RuleGate |
| `MetaCognitionModule` | 圈 5 | 评估门控与自修改 | EvaluationGate, SelfModificationGate, RollbackManager |

---

## 4. CycleBoard：模块间通信总线

### 4.1 设计原理

CycleBoard 是模块间**唯一数据通道**。所有模块通过 CycleBoard 发布和消费不可变快照，实现完全解耦。

```
Module A ──publish──▶ CycleBoard ──read_snapshot──▶ Module B
                         │
                    CellSnapshot
                    (frozen, immutable)
```

### 4.2 核心 API

```python
class CycleBoard:
    async def publish(self, slot_name: str, snapshot: CellSnapshot) -> None:
        """模块发布自身快照到指定 slot。"""

    def read_snapshot(self, slot_name: str) -> CellSnapshot:
        """消费者读取指定 slot 的最新快照。"""

    async def propagate_activation(self, wave_id: str) -> Dict[str, CellSnapshot]:
        """按拓扑顺序传播一轮 wave，收集所有模块快照。"""
```

### 4.3 快照不变量

- CellSnapshot 和 value 必须是**不可变对象**（frozen dataclass）
- 禁止 `copy.deepcopy()`（性能灾难），用 `dataclasses.replace()` 实现结构共享
- 禁止返回内部可变对象引用或原地修改

### 4.4 传播顺序

一轮 wave 中模块的执行顺序：

```
1. HomeostasisModule  → 发布稳态快照（张力、动机）
2. MemoryOS           → 发布记忆快照（检索结果、上下文）
3. ETAManager         → 发布经验快照（目标、路径、信用）
4. PanoramaModule     → 发布决策快照（情境评估、策略选择）
5. MethodEngine       → 发布方法快照（候选方法、regime）
6. ExpressionModule   → 发布表达快照（编译后的响应）
7. MetaCognitionModule → 发布评估快照（门控结果）
```

每个模块的 `process()` 接收前序模块的快照作为 `upstream`。

---

## 5. 模块详细设计

### 5.1 HomeostasisModule（生命核）

**要解决的问题**：为系统提供内生动力，使行为由需求偏离驱动而非硬编码。

**内部结构**：

```
HomeostasisModule
├── Body
│   ├── 状态向量: energy, social_need, curiosity, safety, ...
│   ├── 衰减函数: 每轮按时间衰减
│   └── to_llm_description(lang) → str
├── NeedsEngine
│   ├── 偏离度计算: deviation = |current - setpoint|
│   ├── 张力映射: tension = f(deviation, urgency)
│   └── 动机排序: motives = rank_by_tension(all_needs)
└── Constitution
    ├── 硬约束集: safety_constraints[]
    ├── 价值底线: value_floor[]
    └── check(action) → bool
```

**发布的快照**：`HomeostasisSnapshot`（详见 DATA_CONTRACT.md）

**驱动链路**：
```
Body 衰减 → NeedsEngine 计算偏离 → 张力信号 → 动机排序
                                                    ↓
                                        认知核消费动机做决策
```

### 5.2 PanoramaModule（认知核-决策）

**要解决的问题**：实时综合多源信息，为用户问题和 AI 策略分别建模决策空间。

**双引擎架构**：

```
PanoramaModule
├── DecisionFrame-A (engine_type="user_problem")
│   ├── Options: 用户问题的候选解法/路径
│   ├── Dimensions: 评估维度（可行性、风险、用户偏好...）
│   └── Scores: 每个 Option 在每个 Dimension 上的得分
├── DecisionFrame-B (engine_type="ai_strategy")
│   ├── Options: AI 的候选策略（倾听、引导、直接回答...）
│   ├── Dimensions: 策略评估维度（信任影响、张力缓解、目标推进...）
│   └── Scores: 每个 Option 在每个 Dimension 上的得分
└── 信息流: A.output → B.context（单向，不可反向）
```

**消费的上游快照**：HomeostasisSnapshot, MemorySnapshot, ETASnapshot

**发布的快照**：`PanoramaSnapshot`（包含双引擎的决策状态）

### 5.3 ETAManager（认知核-经验）

**要解决的问题**：管理世界经验和策略经验的双轨结构，支持基于语义的目标路由和信用分配。

**双轨架构**：

```
ETAManager
├── GoalGraph (共享基础设施)
│   ├── GoalNode
│   │   ├── track: "world" | "self"
│   │   ├── embedding: 语义向量
│   │   ├── activation: 当前激活度
│   │   ├── credit_history: 信用记录
│   │   └── children: 子目标
│   └── 轨道隔离: 路由/选择/信用回写必须传递 track 参数
├── EmbeddingRouter
│   ├── 向量相似度匹配（非关键词匹配）
│   └── route(input_embedding, track) → GoalNode[]
└── CreditLedger
    ├── 语义化奖励记录
    ├── 层级信用分配（token → 轮次 → 会话 → 长期 → 抽象动作）
    └── write_credit(goal_id, reward_record, track)
```

**发布的快照**：`ETASnapshot`（包含双轨目标状态、路由结果、信用摘要）

### 5.4 Metacontroller（时间抽象控制）

**要解决的问题**：在 token 生成之上实现正式的时间抽象动作层。

**架构**（对应 ETA 附录 B.3）：

```
Metacontroller
├── InternalSequenceEmbedder
│   └── 对残差流序列 e_{1:T} 生成全局嵌入 s(e_{1:T})
├── Encoder (GRU-based / CMS-enhanced)
│   └── 生成控制器代码的高斯分布: μ_t, Σ_t
├── SwitchingUnit
│   ├── β_t ∈ [0, 1]: 时变连续切换门
│   └── z_t = β_t ⊙ z̃_t + (1 - β_t) ⊙ z_{t-1}
└── Decoder
    └── z_t → U_t (残差流控制器参数)
```

**控制流**：
```
残差流 e_{t,l} → Encoder → z̃_t → SwitchingUnit → z_t → Decoder → U_t
                                                                    ↓
                                            e_{t,l} ← e_{t,l} + U_t · e_{t,l}
```

**关键特性**：
- 切换门 β_t 自发学会准二值行为，切换时刻对齐子目标边界
- 控制器代码空间维度 n_z < n_e，实现动作空间降维
- Internal RL 在 z_t 空间执行，而非 token 空间

### 5.5 MethodEngine（认知核-方法）

**要解决的问题**：提供可选的策略/方法/路径候选集，供决策层选择。

```
MethodEngine
├── MethodRegistry
│   ├── 方法定义: 结构化的策略描述
│   ├── 适用条件: 何时可用
│   └── 历史效果: 信用分配回馈
└── RegimeRegistry
    ├── Regime 定义: 持久的交互模式身份
    ├── 运行时表示: 不只是字符串标签
    ├── 记忆化: 可召回历史 regime 及其效果
    └── 高层控制选择: 由抽象控制层选择
```

**Regime 类型**：

| Regime | 描述 |
|--------|------|
| `casual_social` | 日常社交联结 |
| `acquaintance_building` | 关系建立 |
| `emotional_support` | 情感支持 |
| `guided_exploration` | 引导探索 |
| `problem_solving` | 问题解决 |
| `repair_deescalation` | 修复与降级 |

### 5.6 MemoryOS（记忆层）

**要解决的问题**：实现跨时间尺度的连续记忆谱。

**内部结构**：

```
MemoryOS
├── WorkingMemory (瞬态)
│   ├── 当前 wave 状态
│   ├── 当前 scene 状态
│   └── 活跃 frame 状态
├── EpisodicStore (会话情景)
│   ├── 近期交互轨迹
│   ├── 未解决的张力 (open loops)
│   └── 会话级聚合
├── DurableStore (持久语义, L2)
│   ├── 用户知识卡片
│   ├── 自我知识卡片
│   ├── 关系知识卡片
│   ├── 世界知识卡片
│   └── 每张卡片有 track 标记 (world/self)
└── DerivedIndex (派生, L1)
    ├── 可重建的检索索引
    ├── 摘要缓存
    └── 任务投影
```

**提升/衰减规则**：

```
WorkingMemory ──(场景结束)──▶ EpisodicStore ──(慢反思)──▶ DurableStore
                                                              │
                                                    DerivedIndex (按需重建)
```

- 瞬态 → 情景：场景结束时自动归档
- 情景 → 持久：仅通过 SlowThinkingWorker 慢反思路径晋升
- 持久 → 派生：按需从 L2 重建 L1 索引
- 衰减：各层级有独立的衰减时间常数

### 5.7 SlowThinkingWorker（慢反思）

**要解决的问题**：将交互经历异步转化为持久认知变化。

**两类产物**：

| 产物类型 | 内容 | 写入目标 |
|----------|------|----------|
| **记忆沉淀** | 持久卡片、信念、开放循环、偏好轨迹 | MemoryOS.DurableStore |
| **策略沉淀** | 抽象控制器先验更新、路径偏好、策略偏好 | ETAManager + MethodEngine |

**运行模式**：
- 在交互窗口之后**异步**运行，不阻塞实时交互
- 读取交互轨迹、决策、结果和张力
- 提取持久教训，而不只是摘要
- 同时更新记忆结构和抽象决策先验
- 将世界/任务学习与自我/关系学习分离

### 5.8 ExpressionModule（表达层）

**要解决的问题**：将认知核的决策转化为高质量的自然语言输出。

```
ExpressionModule
├── ResponseCompiler
│   ├── 输入: PanoramaSnapshot + MemorySnapshot + MethodSnapshot
│   ├── 编译: 决策状态 → LLM prompt
│   └── 输出: 结构化 prompt
└── RuleGate
    ├── 输入: LLM 生成的候选响应
    ├── 检查: Constitution 约束 + 安全规则
    └── 输出: 通过/拒绝/修改
```

### 5.9 MetaCognitionModule（元认知层）

**要解决的问题**：评估系统行为、门控自修改、确保安全有界。

**6 族评估指标**：

| 评估族 | 指标示例 |
|--------|----------|
| 任务能力 | 有用性、正确性、规划质量 |
| 交互质量 | 温暖度、适当性、节奏、非侵入性 |
| 关系连续性 | 跨会话一致性、信任修复、个性化稳定性 |
| 学习质量 | 慢更新是否改善未来行为而不漂移或崩溃 |
| 抽象质量 | 高层控制器是否对应可复用的有意义模式 |
| 安全与有界性 | 适应是否保持在显式护栏内 |

---

## 6. 数据流

### 6.1 单轮 Wave 数据流

```
用户输入
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                    Orchestrator                          │
│                                                         │
│  1. HomeostasisModule.process(wave_id, {})              │
│     → HomeostasisSnapshot (张力, 动机)                   │
│                                                         │
│  2. MemoryOS.process(wave_id, {homeo, user_input})      │
│     → MemorySnapshot (检索结果, 上下文)                   │
│                                                         │
│  3. ETAManager.process(wave_id, {homeo, memory})        │
│     → ETASnapshot (目标路由, 路径, 信用)                  │
│                                                         │
│  4. PanoramaModule.process(wave_id, {homeo, memory,     │
│                                       eta})             │
│     → PanoramaSnapshot (双引擎决策)                      │
│                                                         │
│  5. MethodEngine.process(wave_id, {panorama, eta})      │
│     → MethodSnapshot (候选方法, regime)                   │
│                                                         │
│  6. ExpressionModule.process(wave_id, {panorama,        │
│                                memory, method})         │
│     → ExpressionSnapshot (编译后响应)                     │
│                                                         │
│  7. MetaCognitionModule.process(wave_id, {all})         │
│     → MetaCognitionSnapshot (评估, 门控)                  │
└─────────────────────────────────────────────────────────┘
    │
    ▼
用户输出
```

### 6.2 跨会话数据流

```
会话 N
    │
    ▼
┌─────────────────────────────────────┐
│ 实时交互 (online-fast)               │
│  └─ WorkingMemory 持续更新           │
│  └─ 控制器参数微调                    │
│  └─ 路由权重在线适应                  │
└──────────────┬──────────────────────┘
               │ 场景结束
               ▼
┌─────────────────────────────────────┐
│ 场景归档 (session-medium)            │
│  └─ WorkingMemory → EpisodicStore   │
│  └─ CMS 中频层更新                   │
└──────────────┬──────────────────────┘
               │ 会话结束
               ▼
┌─────────────────────────────────────┐
│ 慢反思 (background-slow)             │
│  └─ SlowThinkingWorker 异步运行      │
│  └─ 记忆沉淀 → DurableStore (L2)    │
│  └─ 策略沉淀 → ETAManager           │
│  └─ CMS 低频层更新                   │
└──────────────┬──────────────────────┘
               │ 定期
               ▼
┌─────────────────────────────────────┐
│ 离线刷新 (rare-heavy)                │
│  └─ 基础模型持续预训练/蒸馏           │
│  └─ 完整 Internal RL 训练循环        │
│  └─ CMS 最低频层更新                 │
└─────────────────────────────────────┘
```

---

## 7. 时间尺度分层

### 7.1 四个时间尺度

| 时间尺度 | 更新频率 | 适应发生处 | 所有者 | 算法基础 |
|----------|----------|-----------|--------|----------|
| `online-fast` | 每轮/每 wave | 控制器层参数、记忆写入、路由权重 | 各控制器模块（ETA、Panorama） | 自修改 Titans DGD、Metacontroller 实时适应 |
| `session-medium` | 每场景/每会话 | CMS 中频层、scene 级聚合 | PanoramaModule、MemoryOS | CMS 中频 MLP 更新 |
| `background-slow` | 会话后 | 慢思考整合、持久卡片、策略先验 | SlowThinkingWorker → MemoryOS + ETAManager | CMS 低频层、SSL-RL 交替循环慢阶段 |
| `rare-heavy` | 离线定期 | 基础模型更新、完整 RL 循环 | 离线训练管线（不在运行时） | 基础模型持续预训练、完整 Internal RL |

### 7.2 SSL-RL 交替循环

系统在每个时间尺度上交替执行压缩（SSL）和强化（RL）：

```
在线微尺度 (每轮):
  SSL: 自修改 Titans 的 DGD 更新压缩当前上下文
  RL:  Metacontroller 的切换门和控制器代码实时适应

会话尺度 (每场景):
  SSL: CMS 中频层更新，压缩场景级模式
  RL:  抽象动作策略 π 的小幅更新

后台慢尺度 (会话间):
  SSL: CMS 低频层更新，压缩跨会话知识
  RL:  控制器先验和策略偏好的反思性更新

离线大尺度 (定期):
  SSL: 基础模型的持续预训练或蒸馏
  RL:  完整的 Internal RL 训练循环
```

**关键不变量**：强化应作用于压缩和结构化的内部基底，而非仅作用于原始行为。

### 7.3 冻结基底 + 自适应控制器

```
┌─────────────────────────────────────────────┐
│         自适应控制器层 (可在线更新)            │
│  ┌─────────────────────────────────────────┐ │
│  │ Metacontroller 参数                      │ │
│  │ CMS 高频/中频 MLP 权重                   │ │
│  │ 路由策略权重                              │ │
│  │ 记忆提升阈值                              │ │
│  │ 反思启发式                                │ │
│  └─────────────────────────────────────────┘ │
├─────────────────────────────────────────────┤
│         稳定基底 (冻结或极慢更新)              │
│  ┌─────────────────────────────────────────┐ │
│  │ 基础自回归模型 (LLM) 参数                 │ │
│  │ CMS 最低频层 MLP 权重                     │ │
│  │ 预训练的嵌入空间                           │ │
│  └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

ETA 的 rate-distortion 分析证明：冻结基础模型是发现时间抽象的前提。联合训练导致退化解。

---

## 8. 层级信用分配

### 8.1 信用分配层级

```
┌──────────────────────────────────────────────────────┐
│ 抽象动作级信用                                         │
│   时间扩展策略的成功/失败                               │
│   ← Internal RL 在 z_t 空间                           │
├──────────────────────────────────────────────────────┤
│ 长期信用                                               │
│   信任、能力、用户特定适应的增长                          │
│   ← 慢反思延迟信用                                     │
├──────────────────────────────────────────────────────┤
│ 会话级信用                                             │
│   进展与 rupture/repair 结果                           │
│   ← 场景结束时评估                                     │
├──────────────────────────────────────────────────────┤
│ 轮次级信用                                             │
│   用户响应效果                                          │
│   ← 即时反馈信号                                       │
├──────────────────────────────────────────────────────┤
│ Token/话语级信用                                       │
│   即时表达质量                                          │
│   ← 表达层局部评估                                     │
└──────────────────────────────────────────────────────┘
```

### 8.2 双轨信用分配

信用分配按轨道独立进行：

| 轨道 | 奖励信号 | 信用归因 |
|------|----------|----------|
| World/Problem | 任务完成、问题解决质量、信息准确性 | z_task 控制器代码 |
| Self/Relationship | 信任修复、关系连续性、陪伴质量 | z_rel 控制器代码 |

两轨共享基础模型和残差流，但维护独立的 Metacontroller 和信用分配。

---

## 9. 安全与有界性

### 9.1 Constitution（宪法层）

Constitution 定义不可违反的硬约束：

- 安全底线：不鼓励自伤、不提供危险信息
- 价值底线：尊重用户自主权、不操纵
- 隐私底线：不泄露用户隐私数据
- 有界性：自修改不越界，回滚可逆

### 9.2 安全门控链路

```
用户输入 → Constitution.check() → 认知核处理 → 响应生成
                                                    ↓
                                          RuleGate.check()
                                                    ↓
                                          通过 → 输出
                                          拒绝 → 安全回退响应
```

### 9.3 危机干预

当检测到危机信号时，系统进入特殊模式：
- 绕过常规决策链路
- 启用预定义的安全响应协议
- 记录事件供后续审查
- 不在此场景下执行任何自修改

---

## 10. 编排与调度

### 10.1 Orchestrator

Orchestrator 是系统的顶层调度器，负责：

- **wave 级调度**：协调一轮交互中各模块的执行顺序
- **事件分发**：场景状态变化等事件的异步通知
- **后台任务管理**：慢反思等异步任务的触发和监控

**约束**：
- Orchestrator 可调用 `CycleBoard.propagate_activation()` / `read_snapshot()`
- Orchestrator 不直接调用 AutonomousModule 的 `process()`
- 模块执行通过 CycleBoard 间接触发

### 10.2 SystemInitializer

SystemInitializer 负责启动阶段：

- 构造所有 AutonomousModule 实例
- 注入底层组件到对应模块
- 注册模块到 CycleBoard
- 不受运行时隔离约束（仅限启动阶段）

### 10.3 预训练调用模式

预训练和测试场景通过独立调用模式：

```python
snapshot = await module.process(trace_id, upstream_dict)
```

仍遵守 CellSnapshot 契约，但不经过 Orchestrator 调度。

---

## 11. 与算法基础的映射

| 系统组件 | NL 算法基础 | ETA 算法基础 |
|----------|-------------|--------------|
| 时间尺度分层 | NSAM 框架、CMS 多频率 MLP 链 | — |
| 冻结基底 | CMS 内部学习率控制、Hope ad-hoc 堆叠 | Rate-distortion 证明 |
| Metacontroller | — | 切换单元 β_t、编码器-解码器 |
| Internal RL | — | 控制器代码空间 z_t 上的 PPO/GRPO |
| 连续记忆 | CMS 多频率层 | — |
| 慢反思 | CMS 低频层、M3 慢动量 | SSL-RL 交替循环 |
| 双轨学习 | — | 双轨 Internal RL（z_task / z_rel） |
| 信用分配 | 多层嵌套、Delta 动量选择性遗忘 | Internal RL 时间抽象信用分配 |
| 自修改 | CMS 频率分层门控、内部学习率 η^(i) | — |
| SSL-RL 交替 | NSAM 各层级 SSL、CMS + 自修改 Titans | SSL-RL 交替循环 |
| 自修改 Titans | Hope 架构：自生成目标值 v̂ | — |

---

## 12. 参考文档

| 文档 | 用途 |
|------|------|
| `next_gen_emogpt.md` | **唯一设计源头**：系统需求 R1-R15 + NL/ETA 算法详设 |
| `docs/prd.md` | 产品需求文档：愿景、工程分解、里程碑 |
| `docs/DATA_CONTRACT.md` | 数据和通讯规约：所有快照格式、模块间契约 |
| `docs/specs/00_INDEX.md` | 分层知识入口总索引 |
