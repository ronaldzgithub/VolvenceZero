# lifeform-domain-emogpt

Vertical for the **relationship-aware companion** archetype — what was historically called EmoGPT.

A vertical is **data + light glue** that compiles into the kernel's owner snapshots:

| Asset | Compiles into kernel surface |
|---|---|
| `DomainExperiencePackage` (knowledge / cases / playbooks / boundary hints) | `vz-application.domain_knowledge / case_memory / strategy_playbook / boundary_policy` |
| Regime priors (built into the package) | `vz-cognition.regime` warm-start |
| Scenario packs (`scenarios/*.json`) | `vz-cognition.evaluation` benchmark inputs |
| **Pre-trained bootstraps** (`bootstraps/*.snap`, `*.bs`) | `vz-temporal.MetacontrollerParameterSnapshot` + `vz-cognition.regime.RegimeBootstrap` |

## Public API

```python
from lifeform_domain_emogpt import (
    build_companion_package,           # DomainExperiencePackage data
    scenarios_dir,                     # path to scripted scenarios
    bootstraps_dir,                    # path to pre-trained artifacts
    load_companion_temporal_bootstrap, # MetacontrollerParameterSnapshot
    load_companion_regime_bootstrap,   # RegimeBootstrap
    build_companion_lifeform,          # ready-to-run Lifeform with everything wired
)
```

## Relationship Lab（offline only）

`lifeform_domain_emogpt.lab` 与
`scenario_packages/relationship_transfer_v1/` 承载“同句镜像用户、相反正确
动作”的离线证据环境。它拥有公开经历、封存动力学真值、反应式 action→outcome
转移和内容寻址 decision sidecar；`lifeform-evolution` 只读这些工件发布 Gate 0
与 Gate 1 判词。P1 的 structured-state 只经既有 `MemoryStore` 正式 API，RAG
steelman 真实使用 `companion-ref-harness`，两者都不在本 wheel 新建 owner。

该子包不由 companion 产品路径导入，不新增内核 owner/slot，也不会自动把 outcome
提交给 PE。机械校准命令：

```bash
.venv/bin/python scripts/run_relationship_lab_gate0.py \
  --machinery-only \
  --output-dir /tmp/relationship-lab-gate0
```

没有冻结真实 substrate 的 stateless/raw baseline attestation 时，预期结果是
`machinery_ready=true`、`gate0_passed=false`。

本机已有 Qwen 权重时，可生成 current-turn-only 冻结账本并关闭 Gate 0：

```bash
.venv/bin/python scripts/run_relationship_lab_stateless_baseline.py \
  --model-source Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir artifacts/relationship_lab/<run-id>
```

2026-08-19 的开发期校准为 24/24 有效、4/24 正确，六项 Gate 0 检查全部
PASS；这只证明仪器可判别，不是四能力轴或 formal hidden-test 结论。详见
`docs/specs/relationship-lab.md`。

P1 protocol smoke 与完整 development calibration：

```bash
.venv/bin/python scripts/run_relationship_lab_packet1.py probe-protocol \
  --model-source Qwen/Qwen2.5-1.5B-Instruct

.venv/bin/python scripts/run_relationship_lab_packet1.py run \
  --model-source Qwen/Qwen2.5-1.5B-Instruct \
  --rag-model-source BAAI/bge-m3 \
  --output-dir artifacts/relationship_lab/<packet1-run-id>
```

2026-08-19 的 P1 v2 开发运行中，跨进程恢复、user scope、token scaling 和
console 纠删通过；但 structured-state mirrored pair flip 只有 0.25，prompt/RAG
steelman 未达资格且 RAG 有一条 strict JSON invalid，故
`machinery_ready=false / gate1_passed=false`。这份结果不能升级为 Appendable 能力
成立，也不授权进入 P2 formal。

P1b 用同一冻结 Qwen 把 contextual arm 改为“typed evidence readout → 通用分数
compiler”，并把完整 request template 纳入 lineage。开发协议固定 readout v3、
RAG `top_k=2`；P1 原始 top-k 与 artifact 不改写：

```bash
.venv/bin/python scripts/run_relationship_lab_packet1.py probe-p1b \
  --model-source Qwen/Qwen2.5-1.5B-Instruct \
  --background-depth 32

.venv/bin/python scripts/run_relationship_lab_packet1.py run-p1b \
  --model-source Qwen/Qwen2.5-1.5B-Instruct \
  --rag-model-source BAAI/bge-m3 \
  --output-dir artifacts/relationship_lab/<packet1b-run-id>
```

`expected_action` 只能在 readout observer 发布并落盘后由 evaluator 附着。P1b 仍是
train/validation development calibration，不训练 PE/credit/controller，也不授权
Readable、Learnable、Steerable 或 formal hidden-test 主张。

2026-08-19 的 lineage-complete v3 run 为 24/24 readout strict-valid；prompt、RAG、
structured-state 的 accuracy 分别为 0.25、0.50、0.50，mirrored pair flip 均为 0，
五项 machinery checks 通过，判词
`machinery_ready=true / baseline_underqualified / gate1_passed=false`。报告 artifact 为
`9cd149b1e8c3f74d54d0cbaf72c216edfdaeba2979829925bc50c8ac3d60c4e8`。格式协议已
关闭，但稳定历史抽象读出尚未成立；不得继续通过放宽 parser 或降低阈值晋升 P2。

P1c 不再调 prompt，而是把 stronger-substrate 资格分叉冻结成一个内容寻址协议：

```bash
# 只检查 frozen lineage、本地 cache 与空间，不运行模型
.venv/bin/python scripts/run_relationship_lab_packet1c.py --preflight-only

# 经明确允许 materialize 权重后，串联 candidate Gate 0 与 same-substrate P1b
.venv/bin/python scripts/run_relationship_lab_packet1c.py \
  --allow-download \
  --output-dir artifacts/relationship_lab/<packet1c-run-id>
```

当前 candidate 是 Qwen2.5-3B，权威 protocol v2 id 为
`f209cf49957e3fa22aef20e977d42bd1f76c970c39c97f57a0e47794e0efff87`。runner 只会发布
三种科学下一步：冻结 formal prereg 候选、版本化饱和场景、或重写公开 evidence/label
contract；Gate 0 与 machinery 失败单独报告。任何一路在 secret heldout 解封前都不是
Appendable / Readable / Learnable / Steerable PASS。

2026-08-20 的完整 v2 run 中，fresh Gate 0 为 24/24 valid、10/24 correct，随后
same-substrate P1b 的 prompt/RAG/structured-state 三臂均 8/8 correct、pair flip 1.0。
报告 artifact `599e7e94ac1a06a7b342f6024614c1489b6130e768c1d5db8fbd7b833bfba1d7`
据此判 `version_scenario_dataset_saturated`：下一包必须版本化 `relationship_transfer_v2`，
不能进入 P2，也不能把这个满分结果解释成 Volvence 四能力或相对基线优势。

## "Vertical-shipped calibration" — what it is and why

The kernel ships flat / uniform defaults so it stays vertical-agnostic. Every concrete vertical **encodes its product priors** by:

1. Defining its scripted scenarios (`scenarios/*.json`).
2. Running `lifeform-super-loop` over those scenarios, which jointly trains the metacontroller and the regime classifier.
3. Saving the best-round artifacts into `bootstraps/`.
4. Shipping all of the above as wheel package data.

A product that wants the relationship-companion archetype just calls:

```python
from lifeform_domain_emogpt import build_companion_lifeform

life = build_companion_lifeform()
session = life.create_session(session_id="my-product-session")
result = session.run_turn("I have been feeling really stuck lately.")
```

…and gets the calibrated lifeform without ever running training itself. The kernel layer is untouched; everything domain-specific lives in this wheel.

Adding a new vertical (coding assistant, customer-service bot, teacher) is a new `lifeform-domain-*` package with its own `scenarios/` and `bootstraps/`. The kernel never knows which vertical is loaded. This is what proves trigger ② of `SPLIT.md` ("second consumer of the brain kernel").

## Updating the bootstraps

When you change scenarios, prior data, or the calibration loop, regenerate the artifacts:

```bash
lifeform-super-loop \
  --scenarios packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/scenarios \
  --rounds 3 \
  --save-temporal packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/bootstraps/companion-temporal.snap \
  --save-regime   packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/bootstraps/companion-regime.bs
```

Both files use magic-byte-prefixed pickle envelopes (`VZ-METASNAP\0` and `VZ-REGIMEBS\0`); reading them via the typed loaders fail-fast on schema-version drift, so we know when to retrain rather than load a stale artifact.

## Ablation

`build_companion_lifeform(use_temporal_bootstrap=False, use_regime_bootstrap=False)` returns the lifeform with neither bootstrap applied. Useful for evaluation harnesses comparing baseline vs. each axis vs. both.

## Sharing one Qwen across many sessions

When `lifeform-service` is deployed on a single GPU, every session must share **one** in-memory open-weight runtime. Pass the pre-built runtime through:

```python
from volvence_zero.substrate import build_transformers_runtime_with_fallback
from lifeform_domain_emogpt import build_companion_lifeform

shared = build_transformers_runtime_with_fallback(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    allow_live_substrate_mutation=False,  # required when sharing
)
life_a = build_companion_lifeform(substrate_runtime=shared)
life_b = build_companion_lifeform(substrate_runtime=shared)
```

`build_companion_lifeform` forces the underlying `BrainConfig` into `substrate_mode="injected"` whenever `substrate_runtime` is supplied so the brain consumes the shared instance instead of building a fresh one per session. The runtime must be frozen (`allow_live_substrate_mutation=False`, the default) — sharing a mutation-capable runtime would let one session's adapter-delta updates leak into every other session. `lifeform-service.create_app` enforces this fail-loud at construction time.
