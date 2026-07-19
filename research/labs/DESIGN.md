# volvence-labs — Cognitive AGI 统一实验框架

> 目标：把 `volvence-research/probe/` 总结出的 **7 primitives + 5 frontier 未解前沿**（共 12 条主线）放在同一个 harness 下做可比实验，支持 **AB ablation × SHADOW / ACTIVE wiring × 多 seed 并行**，并且严格遵守 VZ 的 R 不变量（R8 快照、R10 ModificationGate、R12 评估只读、R15 可回滚）。
>
> 设计日期：2026-05-13。

---

## 0. 底线（先说清）

整个框架存在的唯一理由是 3 条底线。任何实现只要违反其中一条，必须推翻重来，不做妥协：

1. **一切都是不可变 snapshot**。输入、中间 readout、输出、决策都是 content-addressed artifact。
   rollback = 指针切换，**没有** 原地修改。
2. **评估通道与训练通道物理隔离**。SHADOW 读数只走 read-only bus。
   要把 SHADOW 读数变成训练源，**必须** 显式通过 ModificationGate（Two-Gate + SGM e-value 验收）。这是代码层的硬门，不是口头约定。
3. **AB / SHADOW / ACTIVE 是 wiring level 不是 flag**。同一份 probe 代码，通过 wiring 配置决定是否"通电"。
   禁止在业务代码里写 `if ab_mode:` 这种分支——它会让 SHADOW 和 ACTIVE 行为漂移。

---

## 1. 两个正交轴

| 轴 | 取值 | 含义 |
|---|---|---|
| **WiringLevel** | `DISABLED` / `SHADOW` / `ACTIVE` | probe 的"通电状态"。SHADOW = 只读 readout，不影响行为；ACTIVE = 参与决策 |
| **AblationCell** | `baseline` / `probe_on` / `probe_off` / `counterfactual` | probe 在对照中的位置 |

**一次实验** = 一组 AblationCell 在同一 WiringLevel 下的并行执行。**一次升级**（SHADOW → ACTIVE）= 通过 gate 的 wiring 切换。

---

## 2. 核心抽象：Probe

每个 probe（无论是 primitive 还是 frontier）都实现同一个接口：

```python
class Probe:
    id: str                     # 稳定名，比如 "pe-epistemic-split-v1"
    hypothesis: str             # 一句话：我在验证什么
    primitive: PrimitiveTag     # 对应 7+5 的哪一件事
    r_ids: list[str]           # 触及 VZ 哪几条 R 不变量

    def knobs(self) -> dict[str, list]: ...     # AB 维度
    def run_cell(
        self,
        cell: AblationCell,
        knobs: dict,
        inputs: SnapshotRef,
        wiring: WiringLevel,
    ) -> ReadoutBundle:
        """单元运行。只能返回 ReadoutBundle，不能写任何外部状态。"""

    def gate(self, readouts: list[ReadoutBundle]) -> GateReport:
        """SHADOW → ACTIVE 的统计验收。默认实现跑 SGM e-value。"""
```

关键约束：

- `run_cell` 是**纯函数**：输入 = (knobs, snapshot, wiring)，输出 = readouts。
- Probe 不能持有跨 cell 的可变状态。要累计信息，只能通过 snapshot。
- wiring 层统一注入依赖（frozen substrate、latent controller 等），probe 不自己决定挂不挂。

---

## 3. 目录结构

```
volvence-labs/
├── DESIGN.md                  # 本文档
├── README.md                  # 使用入口
├── pyproject.toml             # 包配置（阶段 0 用 stdlib only）
├── Makefile                   # 常用入口
├── .gitignore
├── src/volvence_labs/
│   ├── __init__.py
│   ├── framework/
│   │   ├── __init__.py
│   │   ├── snapshot/           # CAS 内容寻址存储 + SQLite 索引
│   │   ├── wiring/             # WiringLevel + AblationCell + Profile
│   │   ├── probe/              # Probe 基类 + 注册器 + Readout schema
│   │   ├── scheduler/          # 顺序 + 并行 runner
│   │   ├── gate/               # （阶段 1 再实）Two-Gate + SGM
│   │   ├── readout/            # 只读 bus（阶段 1）
│   │   └── parallel/           # Cursor best-of-n / worktree 封装（阶段 1+）
│   ├── probes/
│   │   ├── primitive_1_frozen_substrate/
│   │   ├── primitive_2_latent_controller/
│   │   ├── primitive_3_emergent_switching/
│   │   ├── primitive_4_multitimescale_memory/
│   │   ├── primitive_5_epistemic_pe/
│   │   ├── primitive_6_bounded_self_mod/
│   │   ├── primitive_7_readonly_monitoring/
│   │   ├── frontier_1_epistemic_pe_llm_scale/
│   │   ├── frontier_2_cross_modal_zt/
│   │   ├── frontier_3_mesa_objective_detect/
│   │   ├── frontier_4_pe_distributional_rlhf/
│   │   └── frontier_5_r15_formalization/   # meta-probe：框架自证
│   └── cli.py                  # CLI 入口：run / ls / rollback / gate
├── experiments/                # 运行产物（gitignore 整目录）
│   └── <run_id>/
│       ├── manifest.json       # wiring + knobs + input/output hashes
│       ├── readouts/
│       └── gate_report.json
├── configs/
│   └── wiring_profiles/
│       ├── dev.yaml
│       ├── shadow.yaml
│       ├── canary.yaml
│       └── active.yaml
├── tests/
│   ├── test_snapshot_roundtrip.py
│   ├── test_wiring_isolation.py
│   └── test_r15_rollback.py
└── .labs/                      # 本地运行时（gitignore 整目录）
    ├── cas/                    # content-addressed store：<sha256>.bin
    └── index.sqlite            # snapshot / run 索引
```

---

## 4. Snapshot：content-addressed store

**为什么**：R8 快照不可变、R15 可回滚，都要求所有状态都能被"hash → 还原"。

**存储**：
- 数据本体：`.labs/cas/<sha256[:2]>/<sha256>.bin`（去重、幂等）
- 索引：`.labs/index.sqlite` 表 `snapshots(sha, kind, created_at, meta_json)` + 表 `runs(run_id, probe_id, wiring, ablation_cell, knobs_sha, input_sha, output_sha, created_at)`

**API**：

```python
class CASStore:
    def put_bytes(self, data: bytes, *, kind: str, meta: dict) -> str:
        """返回 sha256；重复内容直接返回既有 sha。"""
    def put_obj(self, obj: Any, *, kind: str, meta: dict = {}) -> str:
        """canonical JSON 序列化后走 put_bytes。"""
    def get_bytes(self, sha: str) -> bytes: ...
    def get_obj(self, sha: str) -> Any: ...
    def exists(self, sha: str) -> bool: ...

class RunLog:
    def record_run(self, run: RunRecord) -> None: ...
    def list_runs(self, *, probe_id: str | None = None) -> list[RunRecord]: ...
    def get_run(self, run_id: str) -> RunRecord: ...
```

**不变量**：
- 写入幂等：同样字节进同一个 sha，索引 `INSERT OR IGNORE`。
- 消费者只通过 sha 引用，**不** 传原始对象。
- 任何 mutate 操作都必须产生新 sha，旧 sha 保留（这是 R15 的物理基础）。

---

## 5. Wiring：WiringLevel × AblationCell

```python
class WiringLevel(enum.Enum):
    DISABLED = "disabled"
    SHADOW   = "shadow"
    ACTIVE   = "active"

class AblationCell(enum.Enum):
    BASELINE        = "baseline"
    PROBE_ON        = "probe_on"
    PROBE_OFF       = "probe_off"
    COUNTERFACTUAL  = "counterfactual"

@dataclass(frozen=True)
class WiringProfile:
    name: str
    default_level: WiringLevel
    probe_overrides: dict[str, WiringLevel]  # per-probe override
    seeds: list[int]
    cells: list[AblationCell]
```

**profile 内置 4 份**：

| profile | 用途 | 默认 level |
|---|---|---|
| `dev` | 开发机上跑冒烟，单 seed | SHADOW |
| `shadow` | 批量 SHADOW 长跑 | SHADOW |
| `canary` | 个别 probe 升到 ACTIVE 的灰度 | 混合 |
| `active` | 全量 ACTIVE（实际生产几乎用不到） | ACTIVE |

---

## 6. Scheduler：单机并行

阶段 0 只做**本地多进程**并行，阶段 1+ 再对接 Cursor best-of-n / cloud agent。

- 每个 `(probe, cell, seed)` 是一个 unit。
- 用 `concurrent.futures.ProcessPoolExecutor` 并行跑 unit。
- 每个 unit 产出一个 `RunRecord` 和一批 readout artifact，都写入 CAS + index。
- 主进程最后聚合所有 RunRecord，生成实验 manifest。

核心调度伪码：

```python
def run_experiment(probe: Probe, profile: WiringProfile) -> ExperimentReport:
    units = [(cell, seed) for cell in profile.cells for seed in profile.seeds]
    with ProcessPoolExecutor(max_workers=N) as pool:
        records = list(pool.map(lambda u: _run_unit(probe, profile, *u), units))
    report = aggregate(records)
    # 阶段 1 才启用 gate
    # report.gate = probe.gate(records)
    return report
```

---

## 7. F5 R15 Meta-Probe：框架自证

> **F5 是整个体系的 keystone**。没有它，后面所有 probe 的 rollback claim 都是空头支票。

**Hypothesis**：任何运行完的实验，都能从 snapshot 完整还原其输入、输出、run 记录，且多次还原 bit-exact 一致。

**测试流程**：

1. 跑一个任意 probe，得到 `run_id`。
2. 记录 run 的所有 input/output sha。
3. 把 `experiments/<run_id>/` 目录整个删除。
4. 从 CAS + RunLog 重建所有产物，校验与原始 bit-exact。
5. 再跑一次（相同 seed、相同 snapshot 输入），校验 output sha 完全一致。
6. 任何一步失败 → F5 fail → 整个框架不能上 ACTIVE。

F5 本身也是 probe，走同一 Scheduler，这样它可以 catch 调度器自己的 bug。

### 7.1 F5 v1：ε-tolerance for real hooks（阶段 1+）

阶段 0 的 F5 v0 要求 **bit-exact**（纯 Python stdlib，确定性）。阶段 1 引入真实模型
（TinyLlama, V-JEPA）后，fp16/bf16 CUDA kernel 的非确定性使得 float 输出不再 bit-exact。

**v1 的区分**：

| 类别 | 要求 | 示例 |
|---|---|---|
| 逻辑状态 | bit-exact | `model_sha`, `dataset_sha`, `knobs_sha`, `input_sha`, token ids |
| 浮点 metrics | ε-tolerance (`abs(a-b) < 1e-5`) | `mean_pe`, `accuracy`, `cosine_sim` |
| 浮点 artifacts | ε-tolerance | `pe_head` 数组, `epistemic_head` 数组 |

**v0 保留**：`r15-rollback-v0` 继续存在，用于验证纯 Python probe（pe-baseline-v0）的 bit-exact 性。
v1 (`r15-rollback-v1`) 针对 real-hooks probe（pe-curiosity-critic-v1 等）。

---

## 8. Smoke Probe（`primitive_5_epistemic_pe` 的最小实现）

阶段 0 还需要一个非 meta 的 probe 验证端到端，选 **P5 baseline**：

- 输入：从 snapshot 读一段"预测序列 + 真值"。
- cell:
  - `baseline`：返回 raw PE（逐 token cross-entropy）
  - `probe_on`：返回 Curiosity-Critic 风格的 epistemic PE（stub：ensemble disagreement = std）
  - `probe_off`：probe_on 把 ensemble 设为 1
  - `counterfactual`：把预测序列 shuffle 后再算 PE（应退化到 noise 水平）
- readout：`mean_pe`、`std_pe`、`epistemic_share`

这个 probe 在阶段 0 只是 stub，主要用来 drive framework 走通。真正的 epistemic split 在阶段 1 实现。

---

## 9. 阶段 0 交付清单

本阶段要完成的具体工件（阶段 1 开始后就冻结接口，只允许扩展不允许破坏）：

- [x] `DESIGN.md`（本文件）
- [ ] 独立仓 `volvence-labs/` 初始化（git + pyproject + README + .gitignore）
- [ ] `framework/snapshot`：CAS + RunLog + Manifest
- [ ] `framework/wiring`：WiringLevel + AblationCell + WiringProfile + 4 份 profile
- [ ] `framework/probe`：Probe 基类 + Registry + Readout schema
- [ ] `framework/scheduler`：SequentialRunner + ParallelRunner
- [ ] CLI：`volvence-labs run` / `ls` / `rollback` / `snapshot-show`
- [ ] F5 meta-probe：`frontier_5_r15_formalization/r15_rollback.py`
- [ ] smoke probe：`primitive_5_epistemic_pe/baseline.py`
- [ ] `tests/`：snapshot 回环 / wiring 隔离 / R15 rollback 三组 pytest（用 stdlib `unittest` 写，阶段 1 再切 pytest 不迟）
- [ ] 冒烟：`make smoke` 能端到端跑绿

**阶段 0 显式不做** 的事：
- 不实现 gate（Two-Gate / SGM e-value），只占位。
- 不引入任何第三方依赖（阶段 1 才会加 pydantic / numpy）。
- 不对接 Cursor best-of-n，只跑本地多进程。
- 不做 dashboard / UI。

---

## 10. 阶段 1 的预告（不在本次实现）

为了让阶段 0 的接口选择有方向感，列出阶段 1 会加什么：

- `framework/gate`：Two-Gate capacity bound + SGM e-value + Hoeffding。probe 升 ACTIVE 必须过。
- `framework/readout`：只读 bus + 简单 SQLite dashboard。
- `framework/parallel`：Cursor best-of-n-runner 封装（每 seed 一 worktree）。
- 7 primitives 的真实实现（不再是 stub）。
- 跨 probe synthesis：主 agent 聚合 → gate 决策 → human-in-the-loop 审批（frontier 层）。

---

## 11. 为什么这个框架对 VZ 来说是必要的

一句话：**它是 VZ 所有 R 不变量在实验层的可执行映射**。

| R 不变量 | labs 中的对应 |
|---|---|
| R8 快照 + 契约 SSOT | CAS + RunLog |
| R10 ModificationGate | `framework/gate`（阶段 1）|
| R12 评估只读 | `ReadoutBundle` 在业务路径上不可写 |
| R15 可回滚 wiring level | WiringLevel + F5 meta-probe |
| R-PE | primitive 5 / frontier 1 + 4 的 probe 主题 |
| R3 / R4 latent 控制 | primitive 2 / 3 / frontier 2 + 3 |

如果某条 R 不变量在 labs 里没有直接对应——要么 labs 设计错了，要么 R 不变量可以废。两者都值得 BOSS 显式过一遍。
