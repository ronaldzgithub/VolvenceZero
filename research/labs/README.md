# volvence-labs

Cognitive AGI 统一实验框架。支持 **AB ablation × SHADOW / ACTIVE wiring × 多 seed 并行**，严格遵守 VZ R 不变量（R8 / R10 / R12 / R15）。

- 完整设计：[`DESIGN.md`](DESIGN.md)
- 统一研究控制台规范：[`../../docs/specs/research-lab.md`](../../docs/specs/research-lab.md)
- Research Lab Web：[`web/`](web/)（local-first；当前里程碑为只读界面壳）
- 调研源头：[`../volvence-research/probe/`](../volvence-research/probe/)（7 primitives + 5 frontier 的出处）

## 阶段 0 快速入口

```bash
# 跑所有单测
make test

# 跑冒烟（smoke probe + F5 meta-probe）
make smoke

# 列出所有 run
python -m volvence_labs.cli ls

# 跑指定 probe
python -m volvence_labs.cli run --probe pe-baseline-v0 --profile dev

# R15 回滚演练（从 snapshot 重建 run 产物）
python -m volvence_labs.cli rollback --run <run_id>
```

## 设计底线

1. 一切都是不可变 snapshot（CAS）。
2. 评估通道 ≠ 训练通道；SHADOW 读数不得反向训练，除非通过 ModificationGate。
3. AB / SHADOW / ACTIVE 是 wiring level 不是 if flag。

## 目录

```
src/volvence_labs/
├── framework/   snapshot / wiring / probe / scheduler
├── probes/      primitive_* + frontier_*
└── cli.py
tests/           单测（stdlib unittest）
experiments/     运行产物（gitignore）
.labs/           CAS + SQLite index（gitignore）
web/             Forge → Praxist → SHADOW → ACTIVE 统一操作面
```

`web/` 不改变本包既有 Probe、CAS 或实验 promotion 的权限边界。它只聚合正式 owner artifact；
生产 A0/A1/A2、ModificationGate 与 target wiring 仍由各自契约和 owner 执行。

## 阶段 0 完成状态

见 [`DESIGN.md`](DESIGN.md) §9。
