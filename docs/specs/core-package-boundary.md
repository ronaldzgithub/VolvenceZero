# Core Package Boundary Spec

> Status: draft
> Last updated: 2026-04-25
> 对应需求: R2, R8, R11, R15

## 要解决的问题

如何把 Volvence Zero 的大脑内核包装成稳定 Python package，同时避免把模型权重、产品数据、HTTP 服务、研究 benchmark 和垂直场景内容混进核心包。

第一阶段目标是 **local editable package / package-first core**，不是外网发布：

- 不上传 PyPI
- 不 push / deploy
- 不打包 Qwen 或其他模型权重
- 不包含用户数据、产品运营数据或 secrets
- 不启动公网服务

## 关键不变量

- `volvence-zero` 是 Python distribution 名；`volvence_zero` 是 import package 名。
- 稳定公共入口优先走 `volvence_zero.brain`，而不是 `volvence_zero.agent` 的研究/benchmark 大出口。
- core package 默认不需要模型权重；默认 `BrainConfig` 使用 synthetic substrate。
- Hugging Face / Qwen runtime 必须显式选择 `substrate_mode="hf"`，并通过 optional extra 安装依赖。
- 模型 ID 是配置，不是 package data。
- HTTP 服务是后续薄适配层，不属于当前 core package 边界。

## 包内内容

- 稳定 session facade：`Brain`、`BrainConfig`、`BrainSession`
- 契约运行时：snapshot、module、wiring、guard
- 记忆、经验、domain experience package schema 与编译器
- application stores 与轻量 file persistence 接口
- substrate adapter 协议与 synthetic / trace 风格运行时
- 评估和内部 proof 工件代码可继续存在于 repo/package 中，但不作为稳定公共 API 承诺

## 包外或可选内容

| 内容 | 边界 |
|------|------|
| Qwen / HF 模型权重 | 外部 runtime cache / deployment volume，不进 package |
| `torch` / `transformers` | optional extra `volvence-zero[hf]` |
| CLI / REPL | convenience entry point，不是稳定 core API |
| Dialogue / ETA paper-suite benchmark | research / bench surface，不是默认 product API |
| HTTP / gRPC 服务 | 后续 service adapter |
| 用户记忆、产品数据、运营配置 | deployment / product storage |
| 垂直场景经验 | `DomainExperiencePackage` 数据，作为外部 package / 配置注入 |

## 当前实现口径

- 根目录 `pyproject.toml` 定义本地可安装包 `volvence-zero`，core dependencies 为空，`hf` / `bench` / `dev` extras 才包含 `torch` 和 `transformers`。
- `volvence_zero.brain` 提供稳定 facade：
  - `BrainConfig(substrate_mode="synthetic")` 默认不拉取外部模型
  - `BrainConfig(substrate_mode="hf")` 才显式构建 HF substrate
  - `BrainConfig(substrate_mode="injected")` 要求调用方注入 runtime
- HF runtime 的 `device="auto"` 在可用时选择 CUDA，其次选择 Apple MPS，最后回退 CPU；MPS 加载路径使用 `float16` 以适配本机 7B 级模型实验。
- `Brain.create_session()` 将 domain experience packages 注入 `AgentSessionRunner`，但仍沿现有 application stores 和 rare-heavy state 生效。
- `volvence_zero.__init__` 懒导出 `Brain` / `BrainConfig` / `BrainSession`，保持 root import 轻量。
- 面向其他项目的本机接入说明见 `docs/package_usage.md`。

## 与其他能力域的关系

| 关系 | 能力域 | 说明 |
|------|------|------|
| 依赖 | 契约式运行时 | package API 只封装运行时契约，不绕过 owner / snapshot 边界 |
| 依赖 | Domain Experience Layer | 垂直经验通过 package 注入，不进入 core hardcode |
| 依赖 | 稳定基底 + 自适应控制器 | 模型 substrate 是外部可替换基底，core 保持冻结/显式注入边界 |
| 协作 | 证据计划 | benchmark 可作为 optional / research surface，不等同稳定 product API |

## 变更日志

- 2026-04-25: 初始版本，建立 package-first core 边界、optional HF extra、stable Brain facade 和不发布外网原则。
- 2026-04-25: HF optional runtime 的 auto device 增加 Apple MPS 支持，用于本机 7B 交互实验。
