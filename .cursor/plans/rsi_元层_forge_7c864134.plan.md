---
name: RSI 元层 Forge
overview: 在仓库新建与 packages/ 平级的顶层元层 forge/,实现"失败挖掘 → 有界提案 → 验证晋升"的第一个 RSI 循环,目标是开发环 harness(rules/prompts/evidence 脚本),并把本次讨论的完整思考沉淀为 forge/README.md。
todos:
  - id: packet0-skeleton
    content: 包0:建 forge/ 骨架、README(思考沉淀)、editable_surface.yaml、pyproject
    status: completed
  - id: packet0-spec
    content: 包0:写 docs/specs/rsi-forge.md 并登记 00_INDEX
    status: completed
  - id: packet0-boundary
    content: 包0:新增 tests/contracts/test_forge_boundaries.py 双向边界测试
    status: completed
  - id: packet1-sources
    content: 包1:sources.py 解析 transcripts / verdict bundle / plans
    status: completed
  - id: packet1-mine
    content: 包1:mine.py 三层失败记录 + 语义聚类 + prompts/schemas
    status: completed
  - id: packet2-propose
    content: 包2:propose.py 有界提案(diff + manifesto)+ 去重
    status: completed
  - id: packet3-validate
    content: 包3:validate.py fail-closed 校验 + 回归检查
    status: completed
  - id: packet3-apply
    content: 包3:apply.py 人审落地 + ledger 预测兑现闭环
    status: completed
  - id: verify-e2e
    content: 端到端演练:对真实 transcripts+gate 失败跑一轮循环,产出 proposal bundle
    status: completed
  - id: close-apply
    content: 闭环:apply 两份已人审通过的提案(pr_8205 + pr_3166),ledger 记录
    status: completed
  - id: close-remine
    content: 闭环:跑新一轮 mine,检查 prediction_checks 中两提案预测兑现情况
    status: pending
  - id: close-verify
    content: 闭环:回归验证(ruff + forge/tests + 边界契约测试)并出验收结论
    status: completed
isProject: false
---

# RSI 元层 Forge 实施计划

## 定位与原则

Forge 是一个指向 volvence 的 RSI 元层:物理上独立于运行时 wheel,只通过公开产物(agent 轨迹、`artifacts/**/promotion_verdict.json`、rules/prompts 文件)读取系统,产出**有界编辑提案**,晋升走人审(第一阶段)/ ModificationGate(第二阶段)。三条硬边界:

- **循环外只读**:gate 代码、evaluation、verifier 脚本、tests、`vz-contracts` 对 forge 永远只读——对应 AHE 的"每分收益可归因"约束。
- **可编辑面白名单**:提案只能落在 `forge/editable_surface.yaml` 显式列出的文件上;初始仅 `.cursor/rules/*.mdc` 与 `forge/prompts/**`,扩面需人工修改白名单。
- **决策可观测**:每个提案带 manifesto(证据、根因、预测影响、风险回归、回滚方式),记入 ledger,下一轮挖掘时验证预测——可证伪。

第一个循环选**开发环 harness**(验证器硬:测试/lint/gate 过不过),跑通后再切产品 runtime 可编辑面(接 `ModificationGate.OFFLINE`,那是第二战役,本计划只留 roadmap)。

```mermaid
flowchart LR
  subgraph inputs [只读输入]
    T[agent-transcripts jsonl x93]
    V[artifacts promotion_verdict + report]
    P[.cursor/plans]
  end
  subgraph forgeLoop [forge 循环]
    M[mine 失败模式挖掘] --> Pr[propose 有界提案]
    Pr --> Va[validate 静态校验 + 回归]
  end
  inputs --> M
  Va --> H{人审}
  H -->|接受| A[apply 落到可编辑面 + ledger 记录预测]
  H -->|拒绝| L[ledger 记录原因]
  A --> M2[下轮 mine 验证预测是否兑现]
```

## 目录结构

```
forge/
  README.md                 # 本次讨论的完整思考(见包 0)
  pyproject.toml            # 独立包 volvence_forge,不进 uv workspace
  editable_surface.yaml     # 可编辑面白名单 + 只读面声明
  ledger.jsonl              # 提案决策台账(append-only)
  prompts/                  # forge 自己的挖掘/提案 prompt(遵守 prompt 集中化规则)
  schemas/                  # failure_pattern / proposal manifesto 的 JSON Schema
  src/volvence_forge/
    config.py               # transcripts 路径、artifacts 路径、LLM backend(env 配置)
    sources.py              # 三类输入的解析器(transcripts JSONL / verdict bundle / plans)
    mine.py                 # 失败挖掘与聚类
    propose.py              # 有界提案生成
    validate.py             # 提案静态校验 + 回归检查
    apply.py                # 人审通过后落盘 + ledger
    cli.py                  # forge mine / propose / validate / apply
  tests/                    # forge 自测(pytest forge/tests 显式运行)
```

运行产物写入 `artifacts/forge_<run>_<timestamp>/`,沿用仓库 artifact 命名规律。

## 包 0:骨架 + 思考沉淀 + 边界(先行)

- `forge/README.md`:把本次讨论完整写下——Weng harness-RSI 框架的要点与"优化对象阶梯"、与 AI Scientist(静态手工 harness)的区别、RSI 对 volvence 的价值判断(速度/经济杠杆而非能力杠杆)、为什么做成指向 volvence 的元层而非独立项目或主线 wheel、evaluator/gate 在循环外的安全原则、开发环→产品 runtime 两阶段路线、模糊评估器/多样性坍缩/reward hacking 三个域特有风险。这满足"把思考写到子目录根部"的要求。
- `forge/editable_surface.yaml`:初始白名单 `.cursor/rules/*.mdc` + `forge/prompts/**`;只读面显式列出(`packages/**`、`tests/**`、`scripts/run_gate*`、`docs/specs/**`、`docs/DATA_CONTRACT.md`、`forge/src/**`、`forge/editable_surface.yaml` 自身)。
- `docs/specs/rsi-forge.md`:按 spec 模板写职责边界、契约(failure_pattern / manifesto schema)、不变量(三条硬边界)、退出与回滚条件;登记进 [docs/specs/00_INDEX.md](docs/specs/00_INDEX.md)(§31 补充表)。
- 新建 `tests/contracts/test_forge_boundaries.py`(AST 扫描,复用 [tests/contracts/test_import_boundaries.py](tests/contracts/test_import_boundaries.py) 的模式):(a) `packages/**` 任何文件不得 import `volvence_forge`;(b) `forge/src/**` 顶层不得 import `volvence_zero.*` / `lifeform_*`。
- forge 不进根 `pyproject.toml` deps、不进 `install.sh`(刻意隔离);单独 `pip install -e forge` 使用。

## 包 1:失败挖掘(mine)

- `sources.py` 解析三类输入:
  - Cursor transcripts(`~/.cursor/projects/Users-mengfu-Documents-GitHub-volvence/agent-transcripts/**/*.jsonl`,路径走 config):提取 tool_use 序列、错误重试链、turn_ended 状态。
  - `artifacts/**/promotion_verdict.json` + `report.md`:gate 失败记录(如 gate2 的 `not-supported / single-seed-stoploss`)。
  - `.cursor/plans/*.plan.md`:战役叙事上下文(只读参考,不是编辑对象)。
- `mine.py`:LLM 结构化输出(prompt 在 `forge/prompts/`,schema 在 `forge/schemas/`)把原始轨迹压成三层失败记录(verifier 层原因 / agent 行为因果 / 暴露的机制),嵌入相似度聚类成 failure pattern,每个 pattern 必须映射到一个可编辑面资产或标记为 `out-of-surface`(只报告不提案)。禁止关键词路由,语义聚类走嵌入。
- 输出 `artifacts/forge_mine_<ts>/failure_patterns.jsonl` + `report.md`。
- LLM/嵌入后端:forge 自带薄 OpenAI-compatible 客户端(env 配置),不 import 任何 vz wheel。

## 包 2:有界提案(propose)

- `propose.py`:对每个 in-surface pattern 生成提案 = unified diff + manifesto JSON(字段对齐 `ModificationProposal` 精神:target、evidence 引用(pattern id + 轨迹片段)、root cause、targeted fix、predicted impact、at-risk regressions、rollback = git revert)。
- 偏好可复发、窄改动可解决的模式;跳过任务难度型失败;对同一 pattern 生成的候选做嵌入去重(防多样性坍缩)。
- 输出 `artifacts/forge_propose_<ts>/proposals/<id>/{patch.diff,manifesto.json}`,**不落盘到目标文件**。

## 包 3:验证 + 人审落地(validate / apply)

- `validate.py`(fail-closed):
  1. diff 目标全部在白名单内,否则 BLOCK;
  2. patch 可干净应用(git apply --check);
  3. 对 `.mdc` 编辑:不删除既有硬约束条目(结构 diff 检查)、有 LLM judge 对照 manifesto 判定"是否针对该失败模式";
  4. 回归:`ruff check` 通过、`pytest tests/contracts/test_forge_boundaries.py` 通过。
- `apply.py`:只接受 validate 通过 + 人工确认的提案;落盘、把 manifesto + 决策 + 预测写入 `forge/ledger.jsonl`;下一轮 `mine` 读 ledger,对已应用提案的 predicted impact 做兑现检查并写入报告(决策可观测闭环)。
- 第一阶段晋升门 = 人审(HUMAN_REVIEW 语义);接 `ModificationGate` 留待第二战役。

## 验证与完成条件

- `ruff check forge/ tests/contracts/test_forge_boundaries.py`
- `pytest forge/tests tests/contracts/test_forge_boundaries.py`
- 端到端演练:对现有 93 个 transcripts + gate2 失败 bundle 跑一轮 `mine → propose → validate`,产出至少一份完整 proposal bundle 供人审——这是本计划的验收证据。
- 回滚方式:forge 全部新增文件,删除目录即回滚;被 apply 的 rule 编辑经 git revert 回滚,ledger 保留记录。

## 闭环验收(2026-08-01 人审已通过,待执行)

人审决定(闸门:mengfu,经会话确认):**两份提案均接受**。

- `pr_8205178855dbbf79` → `forge/prompts/failure_mining.system.md`:挖掘 prompt 区分"负向因果证据"(gate 正确拒绝晋升)与"执行失败"。接受。
- `pr_3166d2cd166a039a` → `.cursor/rules/cursor-convergence-workflow.mdc`:结构化失败后的有界恢复与交接(保留证据包、单次有界重试、环境类前置失败显式交接)。接受,附注意事项:观察是否出现"过早判定外部阻塞"的回归。

执行步骤(按序):

1. **apply 两份提案**(bundle 位于 `artifacts/forge_propose_rsi_e2e_20260801T033000Z/proposals/`):

```bash
python -m volvence_forge apply \
  artifacts/forge_propose_rsi_e2e_20260801T033000Z/proposals/pr_8205178855dbbf79 \
  --validation-report artifacts/forge_propose_rsi_e2e_20260801T033000Z/proposals/pr_8205178855dbbf79/validation.json \
  --human-approved-by mengfu

python -m volvence_forge apply \
  artifacts/forge_propose_rsi_e2e_20260801T033000Z/proposals/pr_3166d2cd166a039a \
  --validation-report artifacts/forge_propose_rsi_e2e_20260801T033000Z/proposals/pr_3166d2cd166a039a/validation.json \
  --human-approved-by mengfu
```

   预期:目标文件被打补丁、`forge/ledger.jsonl` 追加两条 applied 记录(含 predicted_impact)。注意 apply 会重新校验 target preimage 哈希——若目标文件在人审后被改动过,会 fail-closed,需重跑 propose。

2. **跑新一轮 mine** 验证预测兑现(`--backend openai` 需要 env 中的 OpenAI-compatible 配置;嵌入模型本地加载):

```bash
python -m volvence_forge mine --backend openai
```

   预期:新 `artifacts/forge_mine_<ts>/prediction_checks.json` 中出现对两份已 apply 提案的兑现检查条目(指标 `pattern_occurrence_count`,pr_3166 预测 10→≤7,pr_8205 预测 1→0)。注意:兑现判定只统计 apply 时间戳**之后**产生的新轨迹/新 verdict;若新增证据太少,记录为 `inconclusive` 而非虚假兑现。

3. **回归验证**:`ruff check forge tests/contracts/test_forge_boundaries.py` + `pytest forge/tests tests/contracts/test_forge_boundaries.py -q`(注:`forge/.pytest_cache` 中 `test_mine_requires_semantic_alignment_and_marks_out_of_surface` 有历史失败缓存,validate 报告显示后续已 10 passed,执行时确认当前为绿)。

4. **验收结论**:第一个完整周期 = mine → propose → validate → 人审 → apply → 再 mine 预测检查全链路跑通。此后进入常态运转,第二战役(产品 runtime 可编辑面 + `ModificationGate.OFFLINE`)另立计划。

回滚:每份提案的 manifesto 内含 `rollback.command`(`git apply --reverse <patch>`);ledger 为 append-only,回滚事件同样记录。

## 执行记录(2026-08-01)

- **步骤 1 完成**:两份提案已 apply,ledger 共 3 条(initialized + 2 条 applied 决策,含 predicted_impact)。`forge/prompts/failure_mining.system.md` 与 `.cursor/rules/cursor-convergence-workflow.mdc` 补丁已落盘。
- **步骤 3 完成**:`ruff check forge tests/contracts/test_forge_boundaries.py` 通过;`pytest forge/tests tests/contracts/test_forge_boundaries.py` 1226 项全绿(历史缓存中的失败测试已确认为绿)。
- **步骤 2 受阻,两个阻塞项**:

### 阻塞 A:预测检查存在证据污染缺陷(需改代码)

`mine.py::_prediction_checks` 不按 apply 时间过滤证据:它直接拿新一轮 mine 对**全部历史源**的 `occurrence_count` 与 baseline 比较。老轨迹中的历史失败仍在,立即重跑 mine 必然把预测标为 `refuted`——假裁决。证据引用(`evidence_ref`)不携带时间戳,无法事后拆分。

修复设计(约 3 文件 + 测试 + spec 同步):

1. `sources.py`:`load_source_bundle(..., since: datetime | None)`,按源文件 mtime ≥ since 过滤 transcripts/verdicts/plans。
2. `mine.py`:`mine_failures(..., evidence_since)` 记入 inventory;`_prediction_checks` 增加参数——若 `evidence_since` 为 None 或早于该提案的 apply 时间戳,status 输出 `inconclusive`(附原因),否则按现逻辑 fulfilled/refuted;`prediction_checks.json` schema_version 升至 v2。
3. `cli.py`:mine 增加 `--evidence-since <ISO8601>` 与 `--evidence-since-ledger`(取 ledger 最近一条 applied 时间戳)互斥参数。
4. `forge/tests/test_sources_mine.py`:补 since 过滤与 inconclusive 判定用例。
5. `docs/specs/rsi-forge.md`:同步预测检查语义。

此缺陷本身就是 RSI 循环第一周期的 dogfood 产出:循环暴露了自己验证机器的缺陷,修复对象是循环外验证器代码(人/agent 常规开发,不走 forge 提案面)。

### 阻塞 B:缺少 LLM 凭据(需用户提供)

真实 mine 运行需要 `FORGE_LLM_API_KEY`(或 `OPENAI_API_KEY`)+ `FORGE_LLM_MODEL`(可选 `FORGE_LLM_BASE_URL`)。当前 shell 环境、`~/.zshrc`、仓库内均无可用配置;此前 e2e 用的是 replay 后端(预录响应),不能用于挖掘新证据。

### 恢复执行清单

1. 实施阻塞 A 的修复(需要允许编辑 forge 源码)。
2. 用户 export LLM 凭据。
3. 等待 apply 之后累积新的开发轨迹/verdict(当前会话轨迹即为首批 post-apply 证据)。
4. `python -m volvence_forge mine --backend openai --evidence-since-ledger` → 检查 `prediction_checks.json`。

## 明确不做(留给第二战役)

- 产品 runtime 可编辑面(playbook/角色包/记忆整合策略)与 `ModificationGate.OFFLINE` 对接。
- 提案器自身的 meta 优化(STOP 式)。
- 自动 apply(无人审)。
