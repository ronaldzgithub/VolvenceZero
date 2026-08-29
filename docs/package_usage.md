# Volvence Package Usage

> Status: current 40-wheel package map and stable entry points
> Last updated: 2026-08-30

## 1. 安装与稳定入口

仓库根 `pyproject.toml` 只是 workspace meta-package。产品代码依赖公开 facade，不从
具体 owner 内部模块导入：

```bash
pip install -e packages/vz-runtime
pip install -e packages/lifeform-core
pip install -e packages/lifeform-service
```

需要真实 Hugging Face runtime 时安装对应 extra；默认 `BrainConfig()` 使用 synthetic
substrate，方便无模型权重的开发和契约验证。

```python
from volvence_zero.brain import Brain, BrainConfig

brain = Brain(BrainConfig(rare_heavy_enabled=False))
session = brain.create_session(session_id="example-session")
result = session.run_turn("Help me think this through.")

print(result.response.text)
print(result.active_snapshots["prediction_error"].value)
```

真实模型必须显式启用，并建议指定本地 snapshot：

```python
from volvence_zero.brain import Brain, BrainConfig

brain = Brain(
    BrainConfig(
        substrate_mode="hf",
        substrate_model_source="/models/Qwen2.5-1.5B-Instruct",
        substrate_local_files_only=True,
        substrate_fallback_mode="deny",
    )
)
```

## 2. `vz-*`：脑核与非语言 testbed（8）

| Wheel | 使用场景 | 边界 |
|---|---|---|
| `vz-contracts` | Snapshot、RuntimeModule、WiringLevel、guards、公共 frozen schema | 所有 wheel 可依赖；不承载业务 owner |
| `vz-substrate` | frozen open-weight runtime、residual/State-KV/Prefix-KV、adapter artifact | 不做策略、regime 或 prompt ownership |
| `vz-temporal` | `beta_t/z_t`、segment closure、SSL/Internal RL | 只拥有时间抽象与 controller state |
| `vz-memory` | continuum memory、CMS、checkpoint/persistence | memory 唯一 owner |
| `vz-cognition` | PE、credit、gate、dual-track、semantic/social owners、regime、reflection、evaluation | cognition plane；历史能力名不拆 wheel |
| `vz-application` | knowledge、case、playbook、boundary、retrieval/assembly、experience consolidation | vertical 经验编译目标 |
| `vz-runtime` | `Brain`、`BrainSession`、final wiring、session-post、evidence orchestration | 唯一业务组合层 |
| `vz-embodiment-ant` | non-language 2D sensorimotor research testbed | 只经 public runtime facade 接内核 |

应用通常只直接依赖 `vz-runtime`；需要共享 schema 时依赖 `vz-contracts`。不要让产品
代码拼装 `final_wiring` 内部模块。

## 3. `lifeform-*`：产品与生命体层（21）

| Wheel | 职责 |
|---|---|
| `lifeform-core` | tick、scene、follow-up、persona base、Lifeform facade |
| `lifeform-affordance` | tool/action descriptor registry、snapshot、renderers、invoker |
| `lifeform-thinking` | async thinking task/artifact/scheduler 与 SHADOW advisory |
| `lifeform-ingestion` | 文本/JSON/PDF/DOCX 等变为 canonical ingestion turn |
| `lifeform-expression` | prompt plan、speech plan、response synthesis、etiquette |
| `lifeform-service` | multi-session aiohttp service、alpha/product routes |
| `lifeform-evolution` | dialogue benchmark、replay、evidence dashboard/bundle |
| `lifeform-cultivation` | 行业专家自培养 intake/reflect orchestration |
| `lifeform-protocol-runtime` | document/task uptake → reviewed BehaviorProtocol candidate |
| `lifeform-mcp-bridge` | MCP tools/resources/prompts → affordance/ingestion/reviewed knowledge |
| `lifeform-openai-compat` | Chat Completions-compatible benchmark facade |
| `lifeform-synthetic-data` | deterministic truth、rendered text、canonical trajectory/task projections |
| `lifeform-domain-emogpt` | relationship companion vertical |
| `lifeform-domain-coding` | coding/pair-programmer vertical |
| `lifeform-domain-venture` | Foundry-facing stateful commercial cognition sidecar vertical |
| `lifeform-domain-operations` | AutoCompany-facing stateful operational cognition sidecar vertical |
| `lifeform-domain-character` | fictional character profile/package/manifest |
| `lifeform-domain-figure` | historical figure corpus/verification/artifact vertical |
| `lifeform-domain-growth-advisor` | long-term growth advisor vertical |
| `lifeform-domain-repair30` | field-service repair vertical |
| `lifeform-domain-digital-employee` | org-agent / employee-twin vertical |

Vertical 只能提供 reviewed data、bootstrap、domain package、boundary 与评测材料；不能
import owner internals 或新建平行 cognition state。

## 4. `dlaas-platform-*`：治理控制面（6）

| Wheel | 职责 |
|---|---|
| `dlaas-platform-contracts` | zero-kernel-dependency 的 InteractionEnvelope、OutputAct 与治理 DTO |
| `dlaas-platform-registry` | SQLite tenant/shell/asset/template/contract/focus-person/identity-link SSOT |
| `dlaas-platform-launcher` | `{ai_id -> SessionManager}`、shared substrate、instance lifecycle |
| `dlaas-platform-api` | `/dlaas/*` aiohttp router 与 auth middleware |
| `dlaas-platform-ops` | pause、operator message、handoff queue、ledger、SSE |
| `dlaas-platform-eval` | audience/exam/license readout gate |

平台调用 lifeform facade，不让 kernel 知道租户、license、handoff 或 API 的存在。

## 5. `companion-*`：标准、基准与开放工具（6）

| Wheel | 职责 |
|---|---|
| `companion-standard` | 零依赖关系表征、semantic records、canonical trajectories、hash/conformance |
| `companion-bench` | 长会话 companion benchmark reference implementation |
| `companion-ref-harness` | vendor-neutral cross-session memory baseline |
| `companion-camel-baseline` | CAMEL framework same-substrate baseline |
| `companion-trajgen` | FSM/LLM simulator → canonical synthetic trajectories |
| `companion-encoder` | open-weight relationship encoder training/evaluation scaffold |

这些包是标准、数据与 readout 面，不是 live relationship owner。encoder 输出进入商业
runtime 时必须走 typed proposal → owner 审核。

## 6. WiringLevel 与读取 turn

```python
from volvence_zero.integration import FinalRolloutConfig
from volvence_zero.runtime import WiringLevel

rollout = FinalRolloutConfig(
    decision_workspace=WiringLevel.SHADOW,
    evaluation_mid=WiringLevel.SHADOW,
    evaluation_expensive=WiringLevel.DISABLED,
)
brain = Brain(BrainConfig(final_rollout_config=rollout))
```

- `result.active_snapshots` 只含 authoritative/placeholder active surface；
- `result.shadow_snapshots` 用于双跑比较，产品行为不得依赖尚未晋升的 SHADOW slot；
- `RuntimePlaceholderValue` 表示 disabled 或缺依赖，不应被伪装成真实 value。

## 7. 身份、持久化与反馈

跨 session memory 需要配置 `memory_scope_root_dir` 并由 identity provider 解析 typed
`UserIdentity`。owner hydration 复用 owner 的 export/hydrate contract；外部 store 不
直接改 owner 私有字段。

产品反馈应走 typed API，例如 `submit_dialogue_outcome`、semantic event batch、tool
outcome 或 environment outcome。禁止直接写 PE、credit、memory internals，也禁止让
evaluation score 变成 reward。

Relationship Memory Console 已通过 `BrainSession` facade 提供 scoped durable
list/keep/delete/rewrite，并用 action ledger 处理幂等；语义/boundary 变更排入下一
canonical turn，不在 HTTP handler 内伪装已持久化。

## 8. Domain experience 与 character package

`DomainExperiencePackage` 只编译到 `vz-application` 既有 stores/priors。角色数值载体
使用 L1 `CommonAdapterBundle` + L2 `CharacterPackageManifest` + L3 tenant state：

- 启动默认 `CHARACTER_PACKAGE_MODE=shadow`；
- `CHARACTER_PACKAGE_WIRING=id=active,...` 可逐角色覆盖；
- ACTIVE 必须通过 manifest、artifact SHA、base weights 与 OFFLINE gate；
- unknown `character_id` 不注入；旧 Character Residual 只读 SHADOW。

## 9. Public 与 internal surface

优先使用：

- `volvence_zero.brain.Brain / BrainConfig / BrainSession`；
- `volvence_zero.integration.FinalRolloutConfig`；
- `volvence_zero.runtime.Snapshot / WiringLevel`；
- 各 wheel 顶层 facade 导出的 frozen contract types。

避免依赖 `agent.session`、`integration.final_wiring`、owner store 私有字段和具体
`*.backbone` 实现。需要新字段时丰富 publisher snapshot 并同步
[DATA_CONTRACT.md](./DATA_CONTRACT.md)，不要在 consumer 里 `getattr(..., default)`
隐藏 schema 漂移。

## 10. 快速验证

```bash
pytest -q tests/test_core_package_boundary.py
pytest -q tests/contracts/test_import_boundaries.py
```

共享 slot/schema 或 wheel dependency 变化时追加 `pytest tests/contracts`；真实模型、
GPU、外部 judge、长轨迹与多 seed evidence 只在相应机制或 promotion gate 改变时运行。
