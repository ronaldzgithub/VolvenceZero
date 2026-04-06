# 契约式运行时 Spec

> Status: draft
> Last updated: 2026-03-25
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

- 2026-03-25: 初始版本，从 SYSTEM_DESIGN.md 和 DATA_CONTRACT.md 提取
