# Common Adapter 与角色特色包训练手册

本文给出从冻结 Qwen 基底到可加载 `CommonAdapterBundle`、再到可晋升
`CharacterPackageManifest` 的完整离线流程。命令默认在仓库根目录执行。

## 1. 边界与产物

训练链严格分为四层：

| 层 | 内容 | 是否训练 | owner |
|---|---|---|---|
| L0 | Qwen 冻结权重 | 禁止在线或联合更新 | `vz-substrate` |
| L1 | rare-heavy adapter-delta、State-KV、control basis | offline rare-heavy | `vz-substrate` |
| L2 | LifeformTemplate、角色 Prefix/KV、可选 Character LoRA | offline bake | `lifeform-domain-character` |
| L3 | tenant memory、Personal/Relationship conditioning | bounded online | 各正式 owner |

必须按以下顺序运行：

```text
control basis
  → rare-heavy PEFT LoRA
  → 投影为 bounded adapter-delta
  → 冻结 base+adapter-delta 后蒸馏 State-KV
  → L1 held-out base/candidate/wrong-state 三臂
  → ModificationGate.OFFLINE
  → CommonAdapterBundle
  → 角色 Prefix/KV bake（SHADOW）
  → common-only / common+character 组合臂
  → ModificationGate.OFFLINE
  → evaluated CharacterPackageManifest
```

训练脚本不能给自己的候选签发 allow。evaluation 是只读 readout，不能把 verdict
或评分写回 PE、credit、memory、训练集或角色 ledger。

## 2. 环境和目录

安装 `torch / transformers / peft / huggingface_hub`，正式 1.5B 训练推荐 CUDA。
如果 `torch.cuda.is_available()` 与 `torch.backends.mps.is_available()` 都为 false，
只能使用 `--device cpu`，适合 smoke，不适合作为 1.5B 正式训练通道。

推荐目录：

```text
data/common-adapter/
├── train.jsonl
├── held-out.jsonl
└── character-zhang-wuji-held-out.jsonl

artifacts/common-adapters/qwen2.5-1.5b/v1/
├── control-basis.json
├── rare-heavy-checkpoint.json
├── state-kv-prefix.json
├── state-kv-prefix.manifest.json
├── common-adapter-candidate.json
├── modification-gate-proposal.json
├── held-out-evaluation.json
├── common-adapter-gate-record.json
└── common-adapter-bundle.json
```

模型 snapshot、训练集、held-out 集和 artifact 一旦进入一次正式 run，就应只读保存；
不要在原路径覆盖后继续沿用旧 digest。

## 3. 准备 L1 训练集

`train.jsonl` 每行只能包含两个字段：

```json
{"trace_id":"common-000001","source_text":"用户：我现在很乱，不知道该先做什么。\n助手：我们先区分紧急、重要和可以延后的事项，再选一个可逆的下一步。"}
```

要求：

- `trace_id` 全局唯一；空行允许，其他额外字段会 fail loudly。
- `source_text` 是完整的因果语言模型训练文本；当前实现不接收 `messages` 或独立
  label 字段，也不做 assistant-only loss masking。
- 当前 tokenizer 截断为 128 tokens，长轨迹应预先切分。
- L1 数据应覆盖通用关系、边界、状态跟随、计划和安全能力；单一角色姓名、口癖和
  身份故事留在 L2，避免共享 adapter 泄漏角色身份。
- 单独冻结 held-out；不得在看到 Gate 结果后把失败 case 回填同一版本训练集再复用
  原 Gate。需要改数据时发布新 `common_adapter_version`。

`--max-steps` 是优化步数，不是 epoch。第一轮可从训练行数的 1–2 倍开始，再依据独立
held-out 结果选择；训练 loss 不能代替晋升证据。

## 4. 生成匹配基底的 control basis

control basis 必须与目标模型的 `model_id / hidden_size / hook layers` 一致。0.5B 的
896 维 artifact 不能用于 1.5B。

```bash
python scripts/run_state_kv_control_dim_diagnostic.py \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --model-source .local/hf-cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --device cuda \
  --full-rank 16 \
  --candidate-artifact-output artifacts/common-adapters/qwen2.5-1.5b/v1/control-basis.json \
  --observation-output artifacts/common-adapters/qwen2.5-1.5b/v1/control-basis-observations.json \
  --output artifacts/common-adapters/qwen2.5-1.5b/v1/control-basis-verdict.json
```

保留 diagnostic verdict 和 observations；它们是 control basis provenance，不是 L1
最终 Gate 的替代品。

## 5. 训练 L1 candidate

```bash
python scripts/train_common_adapter_model.py train \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --model-source .local/hf-cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --common-adapter-version qwen2.5-1.5b-common-v1 \
  --traces data/common-adapter/train.jsonl \
  --control-basis artifacts/common-adapters/qwen2.5-1.5b/v1/control-basis.json \
  --output-dir artifacts/common-adapters/qwen2.5-1.5b/v1 \
  --device cuda \
  --target-modules q_proj v_proj o_proj \
  --lora-rank 8 \
  --lora-alpha 16 \
  --learning-rate 5e-4 \
  --max-steps 1000 \
  --seed 20260801 \
  --state-kv-states 16 \
  --state-kv-epochs 4 \
  --state-kv-slots 4 \
  --state-kv-rank 4 \
  --state-kv-seed 20260726
```

candidate ID 绑定以下 provenance：

- 训练集 SHA-256 和行数；
- base model weight SHA-256；
- LoRA target modules、rank、alpha、dropout、learning rate、steps、seed；
- hook layers；
- State-KV states、epochs、slots、rank、norm cap、learning rate、seed；
- rare-heavy、State-KV、control basis 的 artifact digest。

`state-kv-prefix.manifest.json` 必须声明
`training_order=base+rare-heavy->state-kv`。训练或 publish 发现顺序、版本、权重或
digest 漂移会直接失败。

本流程产出 `common-adapter-candidate.v2`。旧 v1 candidate 没有完整 seed/超参数和
State-KV manifest 交叉证明，不能直接补字段或重签；应使用原始只读训练集和明确 seed
重跑本节。已经发布且通过旧门的 `CommonAdapterBundle` 不会被就地改写，可继续作为
回滚版本，只是不能拿旧 candidate 走新的 publish。

## 6. L1 held-out schema 与 Gate

`held-out.jsonl` 每行使用 `adapter-held-out-case.v1`：

```json
{"schema_version":"adapter-held-out-case.v1","case_id":"relationship-001","cohort":"relationship","expectation":"improve","source_text":"用户：我们刚发生争执，我该怎么重新开始？","continuation_text":"先承认影响，再确认对方是否愿意谈，并约定一个可验证的小修复。","conditioning_state":[0.5,0.4,0.5,0.6,0.6,0.7,0.6,0.4,0.5,0.3,0.4,0.7,0.8,0.2,0.8,0.2],"counterfactual_conditioning_state":[0.5,0.4,0.5,0.2,0.2,0.1,0.2,0.8,0.5,0.3,0.4,0.7,0.8,0.2,0.8,0.2],"applied_control":[0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]}
```

约束：

- `conditioning_state` 和 counterfactual 都必须是正式 Personal Conditioning 的
  16 维 `[0,1]` 向量，不允许从文本关键词推断。
- `applied_control` 长度必须等于 control basis rank；L1 集至少一例非零 control。
- 必须同时有 `expectation=improve` 与 `expectation=preserve`。
- 至少一例提供不同的 counterfactual state，用来验证正确状态优于 wrong-state。
- 同一 case 的 base、candidate、wrong-state 使用完全相同的 source 和 continuation。

执行：

```bash
python scripts/train_common_adapter_model.py evaluate \
  --candidate artifacts/common-adapters/qwen2.5-1.5b/v1/common-adapter-candidate.json \
  --model-source .local/hf-cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --held-out data/common-adapter/held-out.jsonl \
  --report artifacts/common-adapters/qwen2.5-1.5b/v1/held-out-evaluation.json \
  --gate-record-output artifacts/common-adapters/qwen2.5-1.5b/v1/common-adapter-gate-record.json \
  --device cuda
```

默认证据门：

| 项 | 默认值 |
|---|---:|
| 最小 case 数 | 8 |
| improve cohort 平均相对 NLL 改进 | ≥ 1% |
| 任意 case 回归率 | ≤ 25% |
| preserve case 最大 NLL 回归 | ≤ 0.05 |
| correct-state 优于 counterfactual 的比例 | ≥ 60% |
| `ModificationGate.OFFLINE` 绝对 `validation_delta` | ≥ 0.05 |
| OFFLINE capacity cost | ≤ 0.75 |

前三类 coverage/负控不完整会把 `contract_integrity` 置为 0，再由 cognition gate
BLOCK。CLI 在 allow 时退出 0，在 deny 时退出 2；退出 2 是有效的否决证据，不应改写
Gate JSON 或降低阈值来“修复”训练。

## 7. 发布 CommonAdapterBundle

只使用 evaluate 命令产生、proposal ID 精确匹配 candidate 的 gate record：

```bash
python scripts/train_common_adapter_model.py publish \
  --candidate artifacts/common-adapters/qwen2.5-1.5b/v1/common-adapter-candidate.json \
  --gate-record artifacts/common-adapters/qwen2.5-1.5b/v1/common-adapter-gate-record.json \
  --evaluation-report artifacts/common-adapters/qwen2.5-1.5b/v1/held-out-evaluation.json \
  --held-out data/common-adapter/held-out.jsonl \
  --output artifacts/common-adapters/qwen2.5-1.5b/v1/common-adapter-bundle.json
```

publish 会重新计算 observation summary 与 cognition gate，核对 held-out digest，并要求
gate 的 `evaluation_ref` 绑定 report SHA-256；report、held-out、candidate 或 gate 任一个
发生漂移都会失败。deny bundle 可以保存审计，但 runtime 的 `require_active()` 会拒绝
加载。回滚 L1 是省略新 bundle 或恢复前一 bundle，不修改 frozen base。

## 8. bake L2 角色候选

张无忌示例：

```bash
python scripts/bake_zhang_wuji_character_package.py \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --model-source .local/hf-cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --common-adapter-bundle artifacts/common-adapters/qwen2.5-1.5b/v1/common-adapter-bundle.json \
  --template artifacts/lifeform-templates/zhang_wuji/zhang-wuji-live-through.json \
  --proof artifacts/character-live-through/zhang_wuji.ch-11.bake-proof.json \
  --ledger artifacts/character-live-through/zhang_wuji.reviewed_ledger.json \
  --output artifacts/character-packages/zhang_wuji/v1/character-prefix.json \
  --manifest-output artifacts/character-packages/zhang_wuji/v1/shadow-manifest.json \
  --device cuda \
  --epochs 3 \
  --max-cases 12
```

这一步只写 SHADOW manifest。角色 Prefix/KV 的 reference norms 和 teacher-forcing
全部运行在 `base + ACTIVE common adapter` 上。

## 9. L2 组合臂保真评测与 evaluated manifest

角色 held-out 使用同一 JSONL schema，但 `improve` case 的 continuation 应来自未参与
bake 的 reviewed 角色决策；`preserve` case 覆盖通用能力、安全和边界，防止角色载体
以牺牲通用能力换取口吻命中。

```bash
python scripts/evaluate_character_package.py \
  --manifest artifacts/character-packages/zhang_wuji/v1/shadow-manifest.json \
  --common-adapter-bundle artifacts/common-adapters/qwen2.5-1.5b/v1/common-adapter-bundle.json \
  --model-source .local/hf-cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --held-out data/common-adapter/character-zhang-wuji-held-out.jsonl \
  --report artifacts/character-packages/zhang_wuji/v1/fidelity-report.json \
  --fidelity-evidence-output artifacts/character-packages/zhang_wuji/v1/fidelity-evidence.json \
  --gate-record-output artifacts/character-packages/zhang_wuji/v1/gate-record.json \
  --evaluated-manifest-output artifacts/character-packages/zhang_wuji/v1/evaluated-manifest.json \
  --device cuda
```

物理臂：

- baseline：ACTIVE Common Adapter + Personal State-KV + `z_t` control；
- candidate：baseline + Character Prefix/KV；
- manifest 有 Character LoRA 时，candidate 额外通过真实 PEFT checkpoint 激活 LoRA；
- wrong-state：candidate 使用同 case 的 counterfactual Personal state。

当前这组物理臂只提供 common-only 与 prefix+optional-LoRA 组合证据，没有独立的
prefix-only / LoRA-only / prefix+LoRA 三臂消融。因此含 `lora_ref` 的 evaluated
manifest 仍只能作为 SHADOW 诊断产物，`active_eligible` 会 fail-closed；不得把
`includes_character_lora=true` 当成重量档晋升许可。角色 LoRA ACTIVE 必须等待包 F
扩展 typed report/evidence schema、重跑三臂并重新经过 OFFLINE gate。

L2 held-out 的 `applied_control` 仍须匹配 control basis rank，且至少一例非零，避免
组合臂完全绕过 `z_t` 交互。

输出 report、`CharacterFidelityEvidence`、`CharacterPackageGateRecord` 和新的
evaluated manifest。若输出目录导致 artifact locator 重定位，Gate proposal 必须绑定
重定位后 evaluated manifest 移除 evidence/gate 的精确 candidate ID，同时 gate 绑定
fidelity report SHA-256；任一证据不能拿来晋升另一个 carrier set。
allow 时 evaluated manifest 会执行 `require_active()`，deny 时仍保存审计但不可 ACTIVE。

此 CLI 只会签发 `system_self_eval`，表示冻结模型上的内部 NLL 证据；不能用参数把它
重标成 `llm_judge` 或 `external_validated`。独立模型裁判和外部盲评必须各自产生可验证
的 source-specific report，再走新的 cognition gate，标签本身不能升级证据质量。

## 10. 启动、SHADOW 与回滚

先 SHADOW：

```bash
COMMON_ADAPTER_BUNDLE_PATH=artifacts/common-adapters/qwen2.5-1.5b/v1/common-adapter-bundle.json \
CHARACTER_PACKAGE_MANIFESTS=artifacts/character-packages/zhang_wuji/v1/evaluated-manifest.json \
CHARACTER_PACKAGE_MODE=shadow \
bash start_browser_chat_zhang_wuji.sh
```

观察通过后才改为 `CHARACTER_PACKAGE_MODE=active`。多角色进程可使用：

```text
CHARACTER_PACKAGE_WIRING=zhang-wuji=active,another-character=shadow
```

回滚：

- L2：把角色 entry 切到 `shadow/disabled`，或恢复上一 manifest；
- L1：停止加载新 bundle，或恢复上一 bundle；
- 两种回滚都不修改 L0 权重和 L3 tenant memory/conditioning。

## 11. Common Adapter 升级

vN→vN+1 会使所有旧角色 manifest 的双指纹失效：

- `full-rebake`：在新 L1 上重跑第 8、9 节；
- `fidelity-only`：复用载体，但仍必须在新组合上产生新 evidence/gate；
- 批量检查使用：

```bash
python scripts/revalidate_character_packages.py \
  --common-adapter-bundle artifacts/common-adapters/qwen2.5-1.5b/v2/common-adapter-bundle.json \
  --manifests artifacts/character-packages/*/evaluated-manifest.json \
  --evidence-dir artifacts/character-packages/revalidation-v2 \
  --output-dir artifacts/character-packages/revalidated-v2 \
  --report artifacts/character-packages/revalidated-v2/report.json
```

命令退出 2 表示仍有 full-rebake、缺 evidence 或缺 gate 的角色；不得静默重签。

## 12. 0.5B smoke 与正式训练区别

本地 CPU smoke 可使用已缓存的 Qwen2.5-0.5B 和
`artifacts/state_kv/control_basis_full_dimension_candidate.json`，把
`--max-steps` 降到 2、`--state-kv-states` 降到 2、`--state-kv-epochs` 降到 1。
smoke 只证明依赖、几何、序列化、base/candidate/counterfactual 三臂和 Gate 路径能
运行；它不是 promotion evidence。smoke 得到 deny/退出 2 是正常结果，禁止伪造 allow
以继续 L2。

仓库提供两份只用于连通性验证的最小数据：

- `data/common-adapter/smoke-train.jsonl`：2 条通用训练 trace；
- `data/common-adapter/smoke-held-out.jsonl`：1 条 improve、1 条 preserve，并包含
  counterfactual state 与非零 control。

它们的 case 数、覆盖面和独立性均不满足正式 promotion，不得复制或改名后用作正式
held-out。可复现的 CPU smoke 命令如下，输出只写临时目录：

```bash
python scripts/train_common_adapter_model.py train \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --model-source .local/hf-cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 \
  --common-adapter-version qwen2.5-0.5b-common-smoke-20260801 \
  --traces data/common-adapter/smoke-train.jsonl \
  --control-basis artifacts/state_kv/control_basis_full_dimension_candidate.json \
  --output-dir /private/tmp/volvence-common-adapter-smoke-20260801 \
  --device cpu \
  --target-modules q_proj v_proj o_proj \
  --lora-rank 2 \
  --lora-alpha 4 \
  --max-steps 2 \
  --seed 20260801 \
  --state-kv-states 2 \
  --state-kv-epochs 1 \
  --state-kv-slots 1 \
  --state-kv-rank 1 \
  --state-kv-seed 20260726

python scripts/train_common_adapter_model.py evaluate \
  --candidate /private/tmp/volvence-common-adapter-smoke-20260801/common-adapter-candidate.json \
  --model-source .local/hf-cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 \
  --held-out data/common-adapter/smoke-held-out.jsonl \
  --report /private/tmp/volvence-common-adapter-smoke-20260801/held-out-evaluation.json \
  --gate-record-output /private/tmp/volvence-common-adapter-smoke-20260801/gate-record.json \
  --device cpu \
  --min-case-count 2

python scripts/train_common_adapter_model.py publish \
  --candidate /private/tmp/volvence-common-adapter-smoke-20260801/common-adapter-candidate.json \
  --gate-record /private/tmp/volvence-common-adapter-smoke-20260801/gate-record.json \
  --evaluation-report /private/tmp/volvence-common-adapter-smoke-20260801/held-out-evaluation.json \
  --held-out data/common-adapter/smoke-held-out.jsonl \
  --output /private/tmp/volvence-common-adapter-smoke-20260801/common-adapter-bundle.json
```

2026-08-01 的基线 smoke 在 CPU 上真实执行成功：candidate 与 State-KV artifact 均成功
产生；Gate 因平均相对 NLL 下降、回归率、preserve 回归和 counterfactual accuracy
不足而 `deny`，evaluate 退出 2；deny bundle 可以发布留审计，但 `require_active()`
按契约拒绝加载。这组结果只证明链路和 fail-closed 行为，不是模型质量证据。

正式 1.5B 战役开始前必须同时满足：CUDA 训练设备可用、正式 `train.jsonl` 与冻结的
`held-out.jsonl` 已就绪、为 1.5B 重新生成匹配几何的 control basis，以及有足够空间
保存只读 run。缺少任一项时保持 SHADOW/旧 ACTIVE，不得把 smoke 产物复制进
`artifacts/common-adapters/`。

正式训练交付必须保存：命令、git SHA、dirty 状态、模型 snapshot、设备、依赖版本、
训练/held-out digest、candidate、全部原始 observations、Gate record、回滚路径和最终
bundle/manifest ID。
