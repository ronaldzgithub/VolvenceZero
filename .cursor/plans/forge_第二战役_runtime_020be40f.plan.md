---
name: Forge 第二战役 runtime
overview: 把 forge RSI 循环的可编辑面扩展到 volvence 产品 runtime 资产:挖掘 companion-bench 判官 bundle 失败流,提案编辑角色域运行时语义资产,晋升经 scripts 侧 ModificationGate.OFFLINE 裁决器 + 人审双闸。
todos:
  - id: p0-evidence-since
    content: 包0:evidence-since 修复(sources/mine/cli + 测试 + spec)
    status: pending
  - id: p1-bench-source
    content: 包1:BenchBundleSource 解析 bundle/arc_failure + schema v2 + 挖掘 prompt 增补
    status: pending
  - id: p2-surface-adjudicator
    content: 包2:扩 editable_surface + scripts/forge_gate_adjudicator.py + 契约测试
    status: pending
  - id: p3-dual-gate
    content: 包3:validate 按 component 执行验证 + apply 双闸校验
    status: pending
  - id: p4-e2e-drill
    content: 包4:端到端演练 mine→propose→validate→gate→人审→apply
    status: pending
  - id: spec-sync
    content: docs/specs/rsi-forge.md 同步第二战役章节
    status: pending
isProject: false
---

# Forge 第二战役:产品 runtime 可编辑面 + ModificationGate.OFFLINE

## 核心设计决定(基于调研)

- **失败流**:产品 live runtime 默认不落盘轨迹,第二战役以 **companion-bench 判官 bundle**(`artifacts/**/*.bundle.json`,约 783 个,含逐轮对话 + per-turn 8 准则分 + disqualifier + `arc_failure.jsonl`)为代理产品失败源,粒度到 turn。
- **可编辑面(第一批,刻意窄)**:角色域**运行时语义资产**——scenario package 的 `scenes.yaml` + `ssot_fragment.json`(场景检测语义,owner `lifeform-domain-character`)。**明确排除**:`test_suite.yaml`、companion-bench 场景 yaml、`evaluation/*.json` 保真矩阵——这些是评估器资产,进可编辑面即违反"evaluator 在循环外"。产品 prompts 因缺内容级回归测试暂缓扩面。
- **晋升双闸**:runtime 目标的 apply 需要 `ModificationGate.OFFLINE` 裁决 ALLOW **且** 人审通过。裁决器放 `scripts/`(允许 import vz-cognition),forge 本体保持不 import `vz-*` 的边界。
- **validation_delta 折算(预注册,仓库空白件)**:`delta = 候选资产在冻结 test_suite 上的通过率 − 基线通过率`,门槛沿用 OFFLINE 的 ≥0.05;`capacity_cost` 对文件编辑取常量 0.1;`contract_integrity` = 冻结套件与阈值未被触碰(1.0/0.0);`rollback_resilience` = 反向补丁演练通过(1.0/0.0)。折算规则先写进 spec 再实现,观察窗内不得改。

```mermaid
flowchart LR
  subgraph src [只读失败源]
    B["companion-bench bundle x783 (per-turn 判官分)"]
    AF[arc_failure.jsonl]
    PV[promotion_verdict.json]
  end
  subgraph loop [forge 循环]
    M[mine] --> P[propose] --> V["validate (冻结 test_suite 通过率)"]
  end
  src --> M
  V --> G["scripts/forge_gate_adjudicator.py\nModificationGate.OFFLINE"]
  G -->|ALLOW| H{人审}
  G -->|BLOCK| L1[ledger 记录]
  H -->|接受| A["apply scenes.yaml / ssot_fragment.json + ledger"]
  A --> M2[下轮 mine 预测兑现]
```

## 包 0:阶段一收尾——预测检查证据时间下界(前置,已设计好)

按 [rsi_元层_forge_7c864134.plan.md](.cursor/plans/rsi_元层_forge_7c864134.plan.md) 阻塞 A 的方案:

- [forge/src/volvence_forge/sources.py](forge/src/volvence_forge/sources.py):`load_source_bundle(..., since)` 按源文件 mtime 过滤。
- [forge/src/volvence_forge/mine.py](forge/src/volvence_forge/mine.py):`evidence_since` 记入 inventory;`_prediction_checks` 在证据窗无法排除 apply 前污染时输出 `inconclusive`;`prediction_checks.json` 升 v2。
- [forge/src/volvence_forge/cli.py](forge/src/volvence_forge/cli.py):`--evidence-since <ISO>` / `--evidence-since-ledger` 互斥参数。
- 测试:`forge/tests/test_sources_mine.py` 补 since 过滤与 inconclusive 用例。

## 包 1:runtime 失败源接入(mine 扩展)

- [forge/src/volvence_forge/sources.py](forge/src/volvence_forge/sources.py) 新增 `BenchBundleSource`:解析 `*.bundle.json` 的 `perturn_rubric.turn_scores`(低分轮 + 对应对话文本切片)、`disqualifier_report`、`arc_axis_scores`,以及 `arc_failure.jsonl`;证据引用定位到 `arc/session/turn`。
- `failure_pattern.schema.json` 的 `source_kind` 枚举增加 `bench_bundle`(schema 升 v2,mine/propose/validate 同步)。
- [forge/prompts/failure_mining.system.md](forge/prompts/failure_mining.system.md) 增补 runtime 语义段:判官低分是行为失败证据,须区分"场景检测错误 / 语义资产缺口 / 内核能力不足(out-of-surface)";延续既有"负向晋升证据"规则。
- CLI:`mine --bench-root <dir>`(默认扫 `artifacts/`,可限定)。

## 包 2:可编辑面扩面 + OFFLINE 裁决器

- [forge/editable_surface.yaml](forge/editable_surface.yaml) 新增 component(示例):

```yaml
- component: character_scenario_semantics
  paths:
    - packages/lifeform-domain-character/src/**/scenario_packages/*/scenes.yaml
    - packages/lifeform-domain-character/src/**/scenario_packages/*/ssot_fragment.json
  requires_offline_gate: true
  validation:
    held_in: pytest packages/lifeform-domain-character/tests/test_character_migration_package.py -q
    held_out: pytest packages/lifeform-domain-character/tests -q
```

  同时把 `test_suite.yaml`、`packages/companion-bench/**`、`packages/lifeform-domain-character/**/evaluation/**` 显式加入只读面。
- 新建 `scripts/forge_gate_adjudicator.py`(循环外,import vz-cognition):读 proposal bundle + validate 报告 → 按预注册折算规则计算 `validation_delta` → 仿 [scripts/adapter_promotion_evidence.py](scripts/adapter_promotion_evidence.py) 的 `decide_offline_promotion` 合成 `EvaluationSnapshot` → `evaluate_gate_reasons` → 写 `gate_decision.json`(decision + reasons + 输入哈希)到 proposal 目录。
- 新增 `tests/contracts` 用例:forge 提案 target 命中只读评估器路径必须 BLOCK;adjudicator 对 delta<0.05 必须 BLOCK。

## 包 3:validate/apply 双闸接线

- [forge/src/volvence_forge/validate.py](forge/src/volvence_forge/validate.py):按 component 执行 `editable_surface.yaml` 声明的 held-in/held-out 命令(现有机制推广);对 `requires_offline_gate` 的 component,validate 产出基线/候选通过率对比,供裁决器折算。
- [forge/src/volvence_forge/apply.py](forge/src/volvence_forge/apply.py):target 属于 `requires_offline_gate` component 时,apply 前置校验 proposal 目录存在 `gate_decision.json` 且 decision=ALLOW(哈希绑定 patch/manifesto),缺失或 BLOCK 则 fail-closed;ledger 事件记录 gate decision 引用。
- 开发环资产(`.cursor/rules` 等)维持原人审单闸,不受影响。

## 包 4:端到端战役演练(验收)

1. `mine --backend openai --bench-root artifacts/ --evidence-since-ledger` → 产出含 bench 失败模式的 `failure_patterns.jsonl`。
2. `propose` → 至少一份指向 `scenes.yaml`/`ssot_fragment.json` 的提案包。
3. `validate` → 冻结套件基线/候选通过率报告。
4. `python scripts/forge_gate_adjudicator.py <proposal_dir>` → `gate_decision.json`。
5. 人审 → `apply` → ledger 含 gate 引用。
6. 全程回归:`ruff check forge scripts/forge_gate_adjudicator.py` + `pytest forge/tests tests/contracts/test_forge_boundaries.py` + 相关 character 测试。

## Spec 同步与不变量

- [docs/specs/rsi-forge.md](docs/specs/rsi-forge.md):新增第二战役章节——runtime 可编辑面清单、评估器只读清单、validation_delta 预注册折算规则、双闸流程、`inconclusive` 语义。
- 不变量:evaluator/gate/bench/prereg 对循环只读;每个 runtime apply 必须同时有 gate ALLOW + 人审;折算规则观察窗内冻结;全部改动 git 可回滚,ledger append-only。

## 前置依赖(用户侧)

- **LLM 凭据**:live mine/propose 需要 `FORGE_LLM_API_KEY` + `FORGE_LLM_MODEL`(可选 `FORGE_LLM_BASE_URL`)。包 0–3 的实现与测试不依赖凭据(测试用 replay 后端),仅包 4 演练需要。

## 明确不做

- 产品 prompts / substrate anchors / CharacterPackageManifest+LoRA 扩面(验证太贵或缺内容级回归,留第三批)。
- lifeform-service 增加 live 失败流落盘(独立收敛包,归 lifeform-service owner)。
- 提案器 meta 优化、自动 apply(无人审)。