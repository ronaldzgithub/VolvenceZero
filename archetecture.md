# Volvence Architecture Boundary Charter

> Status: active architecture entry
> Last updated: 2026-08-30
> 文件名保留历史拼写，供现有链接兼容。

## First Principles

- Prediction Error 是原始学习信号；credit/evaluation 是下游聚合与只读 gate。
- owner 间只交换不可变 snapshot；consumer 不持有、调用或重建 producer。
- frozen substrate 与 bounded controller 分离；在线不端到端更新 base。
- `beta_t/z_t` 与 Internal RL 属于 temporal owner，不在 token 空间学习长期策略。
- World / Self、semantic owners、platform governance 各有唯一写者。
- 新能力以 `DISABLED/SHADOW/ACTIVE` 渐进迁移，并有证据、退出和回滚条件。

## Current Wheel Boundary（40）

### Kernel / research core（8）

| Wheel | Unique responsibility |
|---|---|
| `vz-contracts` | Snapshot、RuntimeModule、guards、propagate、跨 wheel frozen types |
| `vz-substrate` | frozen LLM/non-language adapter surface、residual capture、bounded carriers |
| `vz-temporal` | metacontroller、`beta_t/z_t`、Internal RL、SSL/RL loop |
| `vz-memory` | CMS continuum、retrieval、promotion/decay、checkpoint |
| `vz-cognition` | PE、credit、gate、dual track、semantic/social owners、regime、reflection、evaluation |
| `vz-application` | knowledge、case、playbook、boundary、retrieval/assembly、experience owners |
| `vz-runtime` | Brain facade 与唯一跨业务 wheel 编排 |
| `vz-embodiment-ant` | public-facade-only non-language embodiment testbed |

### Lifeform / vertical / product（20）

| Group | Wheels |
|---|---|
| Core product | `lifeform-core`, `lifeform-affordance`, `lifeform-thinking`, `lifeform-ingestion`, `lifeform-expression`, `lifeform-service` |
| Evolution / intake | `lifeform-evolution`, `lifeform-cultivation`, `lifeform-protocol-runtime`, `lifeform-mcp-bridge`, `lifeform-openai-compat`, `lifeform-synthetic-data` |
| Verticals | `lifeform-domain-emogpt`, `lifeform-domain-coding`, `lifeform-domain-venture`, `lifeform-domain-character`, `lifeform-domain-figure`, `lifeform-domain-growth-advisor`, `lifeform-domain-repair30`, `lifeform-domain-digital-employee` |

`lifeform-*` 只能经 Brain facade、contracts 与 ModificationGate 进入 brain core；禁止
`vz-*` 反向 import `lifeform-*`。Vertical content 编译进既有 application/cognition
owner，不建立第二 memory/regime/policy owner。

### DLaaS platform（6）

| Wheel | Boundary |
|---|---|
| `dlaas-platform-contracts` | zero-kernel-dependency governance DTO |
| `dlaas-platform-registry` | tenant/resource SQLite SSOT |
| `dlaas-platform-launcher` | instance/session lifecycle、shared substrate |
| `dlaas-platform-api` | typed HTTP dispatch |
| `dlaas-platform-ops` | pause/handoff/operator/SSE/ledger |
| `dlaas-platform-eval` | audience/exam/license readouts |

平台不拥有 cognitive state；只调用 lifeform facade、读取公共 snapshot/readout。

### Companion ecosystem（6）

| Wheel | Boundary |
|---|---|
| `companion-standard` | zero-dependency relationship representation SSOT |
| `companion-bench` | benchmark reference implementation |
| `companion-ref-harness` | minimal memory reference baseline |
| `companion-camel-baseline` | CAMEL same-substrate baseline |
| `companion-trajgen` | canonical synthetic trajectory generation |
| `companion-encoder` | relationship encoder training/eval scaffold |

这些 wheel 的 evaluator/encoder 是 readout 或 proposal source，不成为 runtime owner。

## Split Axes

- R2：substrate 与 online adaptation 分离。
- R3/R4：temporal 独占 latent control 与 abstract action。
- R5/R6：memory 独占记忆连续谱。
- R-PE/R7/R9–R12/R14：cognition 独占 PE、credit、semantic/social、regime、evaluation。
- R8/R15：contracts + runtime 使交换和迁移显式化。
- Product variability：lifeform verticals；governance variability：DLaaS；外部评测：Companion。

历史能力名 `vz-pe-credit`、`vz-self-model`、`vz-evaluation` 不是当前 wheel。未来物理
拆分必须同改 `DATA_CONTRACT.md`、本章程、import-boundary tests 和依赖声明。

## Migration Rules

1. 新 slot 先登记 owner、value type、dependencies、wiring、退出与回滚。
2. 公共 shape/owner/boundary 改变时，同步能力 spec。
3. 跨 wheel import 必须符合单向层级并有 contract test；不得用函数内 import 绕过。
4. 缺字段时丰富 publisher；不得在 consumer 建 shadow owner 或宽泛 fallback。
5. 多文件架构改动一次只收敛一个 owner、一个 snapshot、一个主要 consumer。
6. benchmark/evaluation 只读 structured artifact；不得反向写 PE/reward。
7. rare-heavy artifact 必须 content-addressed、gate-bound、可回滚。
8. 只有 [SPLIT.md](./SPLIT.md) 的触发条件满足时才移动仓库边界。

## Document Map

- [docs/specs/00_INDEX.md](./docs/specs/00_INDEX.md)：能力域默认入口。
- [docs/DATA_CONTRACT.md](./docs/DATA_CONTRACT.md)：slot/schema/依赖注册表。
- [docs/SYSTEM_DESIGN.md](./docs/SYSTEM_DESIGN.md)：当前系统数据流。
- [docs/package_usage.md](./docs/package_usage.md)：完整包地图与 API 用法。
- [docs/current.md](./docs/current.md)：实现、证据和生产边界。
- [docs/next_gen_emogpt.md](./docs/next_gen_emogpt.md)：R-ID 与 NL/ETA 设计依据。
- [SPLIT.md](./SPLIT.md)：monorepo 拆分 charter。
