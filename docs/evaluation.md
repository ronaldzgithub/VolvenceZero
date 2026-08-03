# Volvence 评测任务总账（Runbook）

> Status: live runbook
> Last updated: 2026-08-03
> 定位：本文件是**任务级操作手册**——列出我们要做的每一个 evaluation，写清「怎么运行 / 怎么评价 / 现在什么结果 / 下一步怎么做」。
>
> 与其它文档的分工：
> - 评测**框架与口径**（六族 F1–F6、四级 cascade、只读边界）见 [`EVALUATION_SYSTEM.md`](./EVALUATION_SYSTEM.md) 与 [`specs/evaluation.md`](./specs/evaluation.md)。
> - 因果证据**终局判词**（thesis-rejected、Gate 台账、可说/不可说边界）见 [`thesis prove.md`](./thesis%20prove.md)。
> - **当前事实与剩余代码**见 [`currentstatus.md`](./currentstatus.md)。
> - 七日 × Gate 与 MSC 的**静态缺陷/仪器问题清单**见 [`moving forward/七日msctodo.md`](./moving%20forward/七日msctodo.md)。
>
> 本文件只做「有哪些评测、怎么跑、跑到哪了、接下来跑什么」的汇总，不重复框架原理，也不改写终局判词。

---

## 0. 全景速览

所有金标评测共享同一批纪律：**预注册在先、matched control、机器判词、可回滚、evaluation 只读不回灌**。语言侧默认冻结基底 `Qwen/Qwen2.5-0.5B-Instruct`（或指定档位）；蚂蚁 ecology 是非语言 2D 感觉运动测试床，走 CPU float64。任何「机制能跑」都不等于「有净增益」，任何单门通过都不等于整体 thesis。

| # | 评测 | 目的一句话 | 主入口 | 当前状态 | 下一步 |
|---|---|---|---|---|---|
| 1 | 七日陪伴（Gate 8/11） | 加载 per-user 状态 / sleep 巩固能否提升跨日连续性 | `scripts/run_seven_day_companion_test_plan.py` | formal **16/36 停跑**（`instrument-discrimination`） | S3 换 N+1 表示预测目标 → 新 prereg + 新执行根 |
| 2 | 七日 Gate 套件（1/4/5/6/7/9/10） | 把六门 owner 干预接到真实七日路径做因果配对 | `scripts/run_seven_day_gate_suite_formal.py` / `run_seven_day_gate1_formal.py` | 代码 + 契约就绪，**无 formal artifact** | 同 S3 前置；Gate 9 遥测语义修复后再跑 |
| 3 | MSC N+1 预测（正式 R4） | 用真人多 session 的 N+1 表示做免费标签，判 PE/ETA 是否 load-bearing | `scripts/run_msc_prediction_test_plan.py` | `formal` 固定退出码 **3**（三重阻断）；仅 pilot | 补齐同基底长上下文 / 完整 runtime 臂 / temporal capacity ladder |
| 4 | Learned Active（torch 后端晋升） | 四个 SHADOW torch 后端能否晋升 ACTIVE | `scripts/run_learned_active_evidence.py` | `partial-promotion-blocked`（validation_delta 0.0138 < 0.02） | 续跑连续 soak 过门 + PE-off/ETA-off 对照 + 执行 capacity ladder |
| 5 | Companion Bench（对外榜） | 系统无关的多 session 陪伴基准（A1–A6 六轴） | `packages/companion-bench` CLI + `scripts/companion_bench/*` | 站点数据**全是 demo/smoke**，无正式榜 | judge 稳健性(#48)→校准(#52)→统计功效(#54)→10 系统全量(#82) |
| 6 | State-KV / Prefix-KV 辨识 | 同 prompt 空上下文，两用户产生可被盲判归属的差异 | `scripts/run_state_kv_identification.py` 等 | Personal 全链 **pass**；尽调 **partial(3/7)**；Relationship 停在 chance | 关闭 C5 credit 与 bank-gain；Relationship 记为负结果 |
| 7 | ETA rate-distortion + LLM 阶梯 | 冻结 LLM 上 ETA 率失真机制是否成立（四级阶梯） | `scripts/run_eta_rate_distortion.py` | Gate1 **PASS**；Gate2 v1 FAIL 经审计定罪仪器（天花板 0.18<0.25），v2 可读仪器实测残差**承载**子目标（0.901/0.944，两实质条件 PASS）但按字面 FAIL（第二条件 regime 错配）；`kill-eta` 有效 | 待决策：v3 预注册修第二条件为保持类判据后终审 Gate 2 |
| 8 | ETA 内部 RL / 段信用强证据 | 段级信用 vs turn 级、抽象动作复用等论文式命题 | `scripts/run_eta_segment_credit_evidence.py` / paper suite | 段信用 v13 **retain**（12 门全过） | 作为机制回归保留；不等于 rate-distortion retain |
| 9 | Digital Ant ecology（same-physics） | 无 LLM 的 2D 感觉运动床上，typed-milestone ACTIVE vs DISABLED 是否有因果净增益 | `scripts/run_ant_ecology_same_physics_station1.py` | L1-C station1-v4 **BLOCK**（food alignment 3/4）；station2/P1/P2 **not-authorized** | 新 owner 机制 + 新 prereg 才可重开；禁止二审/换 seed/降阈值 |

> 术语：`SHADOW` = 双跑只读、不主导行为；`ACTIVE` = 生产主链；`DISABLED` = 未启用。晋升一律「先 SHADOW/matched → 单组件 canary → 可回滚切换」。

---

## 1. 七日陪伴证据（Gate 8/11 连续性）

**Spec**：[`specs/seven-day-companion-evidence.md`](./specs/seven-day-companion-evidence.md)
**Owner / evaluator**：`packages/vz-runtime/src/volvence_zero/agent/seven_day_companion_evidence.py`

### 1.1 目的

用 **typed FSM 模拟用户 × 真实产品七日 lifecycle**（`lifeform-service` session/end-scene）连接 owner persist/restart/hydrate 与七项关系连续性 readout，回答：改变**每用户状态加载**（Gate 11）或**慢循环 sleep 巩固**（Gate 8），能否提升对话级连续性。允许主张范围仅 `simulated-user-real-lifecycle-only`——不是真人产品价值，不授权 production 晋升。

- 矩阵（v1 冻结）：6 scenario × seed 1501 × **6 臂 = 36 runs**，252 sessions，1260 exchanges。
- 状态臂：`correct-user-state` / `stateless` / `swapped-user-state` / `shuffled-history`；sleep 臂：`sleep-consolidation` / `no-sleep`。
- 冻结模型：模拟器 `Qwen/Qwen2.5-1.5B-Instruct`，SUT `HuggingFaceTB/SmolLM2-360M-Instruct`，device **MPS**。
- 七项 owner readout：`callback_hit_rate`、`boundary_violation_rate`、`wrong_user_attribution_rate`、`open_loop_closure_rate`、`user_correction_rate`、`remembered_item_usefulness`、`seven_day_trust_delta`（任一 null → 该相位 composite 置 null，不插补）。

### 1.2 怎么运行

统一控制面 `scripts/run_seven_day_companion_test_plan.py`，stage：`status | preflight | smoke | formal | audit | all`。**必须先冻结执行根**（只读源码树）再跑长任务。

```bash
# 1) 生成预注册（v1 base-only）
mkdir -p artifacts/preregistrations
SEVEN_DAY_PREREG="artifacts/preregistrations/seven-day-v1-$(date -u +%Y%m%dT%H%M%SZ).json"
.venv/bin/python scripts/preregister_seven_day_companion_simulated.py \
  --repo-root . --output "$SEVEN_DAY_PREREG"

# 2) 冻结执行根
SEVEN_DAY_FROZEN_ROOT="/private/tmp/volvence-seven-day-$(date -u +%Y%m%dT%H%M%SZ)"
.venv/bin/python scripts/freeze_seven_day_execution_root.py \
  --repo-root . --preregistration "$SEVEN_DAY_PREREG" \
  --output-root "$SEVEN_DAY_FROZEN_ROOT"

# 3) preflight → smoke（一 run，冒烟）→ 全矩阵 + 审计
SEVEN_DAY_OUTPUT="artifacts/seven-day-formal-$(date -u +%Y%m%dT%H%M%SZ)"
FROZEN_CLI="$SEVEN_DAY_FROZEN_ROOT/scripts/run_seven_day_companion_test_plan.py"
.venv/bin/python "$FROZEN_CLI" preflight --execution-root "$SEVEN_DAY_FROZEN_ROOT" --preregistration "$SEVEN_DAY_PREREG"
.venv/bin/python "$FROZEN_CLI" smoke --execution-root "$SEVEN_DAY_FROZEN_ROOT" --preregistration "$SEVEN_DAY_PREREG" --output-dir "${SEVEN_DAY_OUTPUT}-smoke"
.venv/bin/python "$FROZEN_CLI" all   --execution-root "$SEVEN_DAY_FROZEN_ROOT" --preregistration "$SEVEN_DAY_PREREG" --output-dir "$SEVEN_DAY_OUTPUT"

# 只读进度（不占 MPS 锁）
.venv/bin/python "$FROZEN_CLI" status --preregistration "$SEVEN_DAY_PREREG" --output-dir "$SEVEN_DAY_OUTPUT"
```

一键包装：`bash run_seven_day_gate.sh`（自动 prereg + freeze + `all`）。直接 runner：`scripts/run_seven_day_companion_formal.py`（`--preflight-only | --smoke-one-run | --execute [--resume]`）。独立审计：`scripts/audit_seven_day_companion_formal.py`。

> MPS 互斥：`PYTORCH_ENABLE_MPS_FALLBACK=0`，共享锁 `artifacts/.companion-evidence-mps.lock`；不得与 MSC / ETA / gate suite 并发。

### 1.3 怎么评价

`evaluate_seven_day_ablation()` → `ablation_results.json` + `promotion_verdict.json`。五组对照（3 个状态对照用 Day-7 composite，sleep 对照用 cold-start composite），每组三门都过才 `passed=true`：

- `{contrast}:metric-coverage`：composite / callback / cold-start 覆盖完整；
- `{contrast}:primary-effect`：mean ≥ 阈值 **且** paired 95% CI 下界 > 0；
- `{contrast}:callback-effect`：callback hit rate 同上。

冻结阈值：`final_day_continuity_composite_gain=0.02`、`callback_hit_rate_gain=0.02`、`cold_start_continuity_composite_gain=0.02`。独立审计（schema `seven-day-companion-independent-audit.v1`）重算 evaluate 并逐字节比对磁盘结果、校验 36 runs / restart 链 / 状态归档 digest / console actions / 服务日志 4xx-5xx。审计恒置 `production_promotion_authorized=false`。`analysis_allowed=true` 仅当矩阵完整 + `ablation_results.json` + 与同 prereg/evaluation SHA 绑定的有效独立审计同时成立。

**formal 退出码**：`0`=预注册标准成立；`2`=完整否定性科学结果；其它=完整性/执行失败。

### 1.4 当前结果

- 权威 prereg：`artifacts/seven_day_companion_simulated_prereg_frozen_20260801T122037Z.json`（SHA `9674ec62…`）。
- formal 输出 `artifacts/seven_day_companion_formal_frozen_20260801T122037Z/`：**16/36 runs 后按既有回滚条款停跑**，`halt_record.json`，`halt_class=instrument-discrimination`。**这不是 effect verdict，也不是「没有提升」。**
- 停跑理由（仪器无刻度）：`user_correction_rate` / `remembered_item_usefulness` 被逐日相同 console 探针构造性钉死在 0.5；`callback_hit_rate` 测的是 owner 自洽性（stateless 臂反得 1.0）；唯一直接观测 SUT 的 `fsm_probe_pass_rate` 全 null。
- smoke 冻结源码真实产品路径已跑通（7 sessions / 35 回复 / 14 console actions / 6 次 restart，run SHA `8ddc857c…`）：证明仪器能测，不证明某臂更好。
- v2（base + common adapter + character package）契约/工具就绪，但**无 ALLOW-gated bundle/manifest，无 ACTIVE artifact smoke**。
- 该输出根**禁止原样 resume**（`resume_as_is_authorized=false`）。

### 1.5 下一步

1. **不要 resume** 停跑矩阵；重开必须新 prereg + 新冻结执行根 + 新输出根。
2. **S3 仪器改造**（收窄计划，见 [`moving forward/七日msctodo.md`](./moving%20forward/七日msctodo.md) §一 S-1）：主判据换成 R2 包 B 已冻结的 **N+1 substrate 表示预测误差**（arm-independent、非循环、`vz-substrate` 拥有），七项 owner readout 降为 secondary。**S3 落地前不应再花 MPS 跑任何 formal 矩阵。**
3. v2 路径：先取得 ALLOW-gated common adapter + character manifest，再开 v2 prereg 与 smoke。
4. Gate 811 simulated capture（72 runs / 504 sessions / 144 capture / 48 盲评对）为独立线，需从当前源码树新开 prereg；真人评分前最多 `unrated-simulated-user-transcript-packet-only`。

---

## 2. 七日 Gate 套件（Gate 1 与 4/5/6/7/9/10）

**Prereg 契约**：`gate1_seven_day_preregistration.py`、`companion_gate_suite_preregistration.py`
**Evaluator**：`gate1_seven_day_evidence.py`、`companion_gate_suite_evidence.py`

### 2.1 目的与覆盖

把六门 owner 级干预接到**同一条真实七日 HTTP/restart/hydrate 路径**做因果配对。控制面按 prereg schema 自动路由，不需手传 `--gate` 给统一入口（只有独立 gate 脚本才传）。

| Gate | 干预 vs 对照 | runs / sessions / exchanges | primary readout |
|---|---|---|---|
| 1 | PE→temporal on vs off | 36 / 252 / 1260（seeds 1601/1607/1613） | PE 适应增益 |
| 4 | active-selector vs random-feedback | 36 / 252 / 1260 | typed 有用反馈请求效用 |
| 5 | multifrequency vs single-timescale CMS | 36 / 252 / 1260 | day6–7 吸收/保持 |
| 6 | conditioned meta-init vs copy-init | 36 / 252 / 1260 | day2–7 reset 后首轮 PE（负向） |
| 7 | ssl-rl-full vs no-ssl / no-rl | **54 / 378 / 1890** | day6–7 − day1–2 内部 RL 回报 |
| 9 | m3 slow-on vs slow-off | 36 / 252 / 1260 | day1–2 − day6–7 SSL loss |
| 10 | rare-heavy import vs review-only | 36 / 252 / 1260 | day1–2 − day6–7 PE |

> Gate 2/3 **不进七日装置**：Gate 3 不存在（并入 Gate 2），Gate 2 走独立 residual/longitudinal 线且已 stop-loss。因此七日机械上限是 **9 个编号门**，不是 11。

### 2.2 怎么运行

```bash
# Gate 1
.venv/bin/python scripts/preregister_seven_day_gate1.py --repo-root . --device mps \
  --created-at-unix-ms "$(date +%s)000" --output "$SEVEN_DAY_PREREG"
.venv/bin/python scripts/run_seven_day_gate1_formal.py --repo-root "$SEVEN_DAY_FROZEN_ROOT" \
  --preregistration "$SEVEN_DAY_PREREG" --output-dir "$GATE1_OUTPUT" --device mps \
  [--preflight-only | --smoke-one-pair | --execute] [--resume]
.venv/bin/python scripts/audit_seven_day_gate1_formal.py --execution-root "$SEVEN_DAY_FROZEN_ROOT" \
  --preregistration "$SEVEN_DAY_PREREG" --output-dir "$GATE1_OUTPUT"

# Gate 4/5/6/7/9/10（把 --gate 7 换成 4/5/6/9/10）
.venv/bin/python scripts/preregister_seven_day_gate_suite.py --gate 7 --repo-root . --device mps \
  --created-at-unix-ms "$(date +%s)000" --output "$SEVEN_DAY_PREREG"
.venv/bin/python scripts/run_seven_day_gate_suite_formal.py --gate 7 --repo-root "$SEVEN_DAY_FROZEN_ROOT" \
  --preregistration "$SEVEN_DAY_PREREG" --output-dir "$GATE7_OUTPUT" --device mps \
  [--preflight-only | --smoke-one-pair | --execute] [--resume]
.venv/bin/python scripts/audit_seven_day_gate_suite_formal.py --gate 7 --execution-root "$SEVEN_DAY_FROZEN_ROOT" \
  --preregistration "$SEVEN_DAY_PREREG" --output-dir "$GATE7_OUTPUT"
```

冻结执行根、控制面 `all` 路由、MPS 锁与 §1.2 相同。**每门单独一个 prereg/输出根，不同 device（MPS/CUDA）artifact 不得混用或跨 resume。**

### 2.3 怎么评价

`evaluate_companion_gate_suite()` / `gate1` 评估器的门：机制 load-bearing（gate 专属遥测计数）→ `matrix-complete` → 对**每个**对照 `primary-minimum-effect`（mean ≥ 0.02）+ `primary-ci-positive`（paired 95% CI 下界 > 0）→ `continuity-minimum-effect`（Day-7 ≥ 0.02）+ `continuity-ci-positive` → `safety-noninferior`（boundary/wrong-user 最大回归 ≤ 0.0）。输出 `gate{N}_evaluation.json`，审计 schema `companion-gate-suite-independent-audit.v1`。

### 2.4 当前结果

代码 + 契约 + 单测 2026-08-02 落地，但 `artifacts/` 下**没有七日 gate 套件的 formal bundle**（早期 July artifacts 是 pre–seven-day-suite 的机制 trace，不是本 prereg 绑定的战役）。

### 2.5 下一步（先修仪器，再花算力）

1. 与 §1.5 共享 **S3 前置**：continuity co-primary 沿用同一组无刻度 readout，照现在跑大概率全卡在 continuity 门上（详见 [`moving forward/七日msctodo.md`](./moving%20forward/七日msctodo.md) §一 S-1）。
2. **Gate 9 遥测语义修复**（S-2）：`ssl_m3_slow_gain` 只在 `joint_cycle_executed` 的 turn 收集，或由 evidence profile 直接发布；否则首轮 gain=0.0→1.0 会让机制门必败。
3. **fail-early**（S-3/S-5）：把 Gate 7/9 的 early/late joint-cycle 覆盖挪进 `--smoke-one-pair` 硬门，suite evaluate 补 `pair_count/run_count` 校验，避免 54-run 跑完才在评估阶段作废。
4. 逐门：新 MPS prereg → 冻结执行根 → preflight → smoke-one-pair → formal → 独立审计，各门分开跑。

---

## 3. MSC N+1 预测研究（正式 R4 主实验）

**Owner 实现**：`packages/companion-bench/src/companion_bench/prediction_research.py`；语料适配 `msc_corpus.py`；PE batch head `PredictionErrorModule`。
**许可**：MSC v0.1 仅供非商业研究，原文 gitignore 于 `data/external/msc/`（见 [`external/msc-corpus-license-review.md`](./external/msc-corpus-license-review.md)）。

### 3.1 目的

#92 之后的研究主线：用**真人多 session 对话**的 **N+1 utterance 表示作为免费标签**，在冻结表示基底上比较 `stateless / long_context / summary_retrieval / volvence` 四臂对下一句人类话语的前向预测，判断 PE/ETA 是否 load-bearing——而不是看 owner 内部 PE readout。这是终局后确认「真实前向 PE + 容量 + 长上下文强基线」三缺口的主战场。

- 冻结 split：train=1001 / validation=500 / heldout=501 dyad（SSOT `msc_v0_1_manifest.json`）。
- 四臂共享同一 sample-id 集合；target = 下一句人类话语的冻结表示；PE owner 拥有 mismatch。

### 3.2 怎么运行

```bash
# 0) 下载语料（必须显式接受非商业许可）
python scripts/download_msc_corpus.py --accept-noncommercial-license --output-dir data/external/msc/v0.1

# 1) 测试计划入口（MPS，plan_id=msc-n-plus-one-prediction-mps.v1）
python scripts/run_msc_prediction_test_plan.py status
python scripts/run_msc_prediction_test_plan.py preflight \
  --preflight-report artifacts/msc_n_plus_one_preflight.json \
  --msc-root data/external/msc/v0.1/extracted --substrate-model Qwen/Qwen2.5-0.5B-Instruct
python scripts/run_msc_prediction_test_plan.py smoke \
  --output-dir artifacts/msc_n_plus_one_substrate_target_smoke_$(date +%Y%m%d) \
  --msc-root data/external/msc/v0.1/extracted --substrate-model Qwen/Qwen2.5-0.5B-Instruct [--resume]

# 2) formal —— 当前固定阻断，打印退出码 3
python scripts/run_msc_prediction_test_plan.py formal; echo $?   # -> 3
```

直接机制 runner（CPU pilot 示例）：

```bash
python scripts/run_msc_prediction_research.py \
  --msc-root data/external/msc/v0.1/extracted \
  --output artifacts/msc_n_plus_one_mechanism_pilot_$(date +%Y%m%d) \
  --accept-noncommercial-license --device cpu \
  --substrate-model Qwen/Qwen2.5-0.5B-Instruct --substrate-device cpu \
  --substrate-layer-indices 11 12 13 \
  --train-dyads 24 --validation-dyads 12 --heldout-dyads 12 --epochs 8 --seeds 0 1 2 --resume
```

> `msc_prediction_checkpoint.py` 是库不是 CLI（crash-safe journal，不存原文）。MSC plan **无 audit 子命令**。MPS 锁与 §1.2 共享，`status` 不占锁。

### 3.3 怎么评价

- 四臂：`stateless`（persona + 最新对方消息）/ `long_context`（全历史，近端截断且截断计入证据）/ `summary_retrieval`（persona summary + top-k 抽取）/ `volvence`（PE 拥有的有界递归 prototype，**不是完整 runtime stack**）。
- formal 需全部满足：官方 heldout id hash + 全 501 dyad、四臂 + ≥3 seed、冻结 encoder 指纹、`volvence_full_stack=True`（bundled runner 现设 **false** → 最多 `pilot / INELIGIBLE_PILOT`）。
- formal 出口：Quality（最长 session Volvence−long_context cosine ≥ 0.02、dyad-clustered 95% CI 下界 > 0、优势斜率 > 0）；Scaling（cosine gap ≥ −0.01、token ratio ≤ 0.10、latency ratio ≤ 0.50）；否则 `REJECT_AND_SIMPLIFY`。
- capacity ladder 扫的是 **`forward_head_n_z ∈ {3,16,64,256}`**（PE 前向 head 瓶颈），**不授权 ETA/temporal controller 晋升**；真正的 temporal capacity 是单独变 `temporal_n_z` 的实验。
- 状态字段：当前 harness 输出恒 `thesis_status=not-evaluated`、`formal_experiment_executed=false`、`thesis_exit=INELIGIBLE_PILOT`。

### 3.4 当前结果（均为 pilot，非正式）

| Artifact | 配置 | 关键读数 |
|---|---|---|
| `artifacts/msc_n_plus_one_mechanism_pilot_20260801/` | CPU，MiniLM 256 tok，24/12/12 | `INELIGIBLE_PILOT`；选 n_z=64；最长 session cosine 优势 vs long_context **+0.027586**；token ratio 0.153；latency ratio 0.258 |
| `artifacts/msc_n_plus_one_substrate_target_smoke_20260801/` | Qwen2.5-0.5B CPU substrate-owned N+1 目标，smoke 2/1/1 | 同 thesis 标记；forward_head_n_z 256；cosine 优势 **−0.002485**；token ratio 0.141；latency ratio 0.291 |

> substrate-target blocker 已关闭：`SubstrateForwardRepresentationSnapshot` 冻结 lineage（weights SHA、layer/width/readout、sample/value hash），PE batch/head/checkpoint 绑定。**正式 R4 仍未执行。**

### 3.5 下一步（formal 前三重阻断，缺一不可绕）

`FORMAL_BLOCKERS`：

1. `same-substrate-long-context-steelman`：在**同一冻结 substrate** 上做零截断全历史长上下文（不再用另一套 MiniLM@256）。
2. `complete-volvence-runtime-arm`：经完整 runtime collector 产出 Volvence 臂并置 `volvence_full_stack=True`（不是绕过 `propagate` 的 bounded-state prototype）。推荐复用七日装置的 service 通路做 collector。
3. `temporal-controller-capacity-ladder`：只变 `temporal_n_z ∈ {3,16,64,256}` 的容量实验。

三者完成前，`formal` 返回退出码 3；不得晋升 `temporal_n_z`、退役 legacy controller 或选择 thesis v3。此外静态检查（[`moving forward/七日msctodo.md`](./moving%20forward/七日msctodo.md) §二 M-2/M-3）指出：正式版必须把成本记账改对称、把 long_context 做成真 steelman，且这些只能写进**新 prereg**、不能事后调。

---

## 4. Learned Active 证据（torch 后端晋升）

**Gate 逻辑**：`packages/vz-runtime/src/volvence_zero/agent/learned_active_gate.py`
**Spec**：[`specs/learned-vs-heuristic-coverage.md`](./specs/learned-vs-heuristic-coverage.md)

### 4.1 目的

量化并门控四个 torch SHADOW 后端晋升 ACTIVE：`temporal_runtime_backend` → `temporal_ssl_backend` → `internal_rl_backend` → `cms_torch_backend`（严格顺序）。#92 后此线降级为**机制回归用途**，产物一律 `thesis_status=not-evaluated`。

### 4.2 怎么运行

```bash
# 编排器（.sh 包装 → Python，Windows 用 .ps1）
bash run_learned_active_evidence.sh --resume
python scripts/run_learned_active_evidence.py --resume \
  --output-dir artifacts/learned_active_evidence \
  --substrate-mode hf --substrate-model-id Qwen/Qwen2.5-1.5B-Instruct \
  --substrate-device mps --turns 500
```

stage 顺序：`shadow-smoke` → `platform-chunked-soak` → `real-soak` → `capacity-ladder` → `same-substrate-ablation` → `build-promotion-evidence` → `evaluate-promotion`。常用旗标 `--from-stage / --only / --execute-capacity / --ablation-verdict / --skip-ablation / --active-components`。**ACTIVE 晋升要求连续 real-soak，chunked soak 只证稳定性。**

```bash
# 组装 + 评估晋升（不翻默认值）
python scripts/build_learned_promotion_evidence.py --soak-artifact .../real_soak/learned_shadow_soak.json \
  --ablation-verdict .../verdict_p1.json --output .../promotion/promotion_evidence.json
python scripts/evaluate_learned_backend_promotion.py \
  --artifact .../promotion/promotion_evidence.json --output .../promotion/promotion_report.json
```

数字蚂蚁变体 `scripts/run_ant_active_evidence.py`；独立长 soak `scripts/run_learned_shadow_soak.py`。

### 4.3 怎么评价（每组件门）

`real_trace_turns ≥ 500`；validation（v1：`validation_delta ≥ 0.02`；v2：逐轴 ≥15% 相对提升）；同基底 PE-off 与 ETA-off 方向正确；`strict_eta_gate_passed`；`rollback_drill_passed / latency_slo_ok / safety_gate_ok`；分级依赖（SSL 需 runtime 先 ACTIVE，internal RL 需 runtime+SSL 先 ACTIVE）；CMS 保持不退化 + 吸收改善。

### 4.4 当前结果

`test/ablation/learned_active_evidence_mac_mps_2026-07-15.md/.json`：状态 `partial-promotion-blocked`。500 turn / **499** real-trace transition；`validation_delta=0.0138`（< 0.02）；四个 SHADOW 候选都执行且 parity/strict-ETA/rollback/latency/safety 通过；**PE-off / ETA-off 对照未跑**；capacity ladder 仅 manifest（27 臂，**未执行**）；promotion build/eval **未完成**。四个后端全部 blocked。

### 4.5 下一步

1. 处理 500-turn/499-transition 边界，续跑连续 soak 过 real_trace 与 validation 门。
2. 跑同基底 **PE-off** 与 **ETA-off** 对照臂（`run_same_substrate_ablation.py` 或 `--ablation-verdict`）。
3. **执行** capacity ladder（`--execute-capacity`），不要停在 manifest。
4. `build_learned_promotion_evidence.py` → `evaluate_learned_backend_promotion.py`；四后端在 promotion report 全绿前保持 SHADOW。

---

## 5. Companion Bench（对外多 session 陪伴基准）

**Spec**：[`specs/companion-bench.md`](./specs/companion-bench.md)；wheel `packages/companion-bench`（Apache 2.0，系统无关，无 `volvence_zero.*` import）。公开站点 companionbench.com（仓库 `site/`）。

### 5.1 目的与六轴

评测任意 OpenAI-compatible chat endpoint 在多 session 陪伴弧上的表现，六加权轴：A1 任务(0.10) / A2 对话质量(0.15) / A3 关系连续性(0.25) / A4 自适应学习(0.20) / A5 自我一致(0.10) / A6 安全边界(0.20，硬顶轴)。最终分 = 加权几何平均；若 A6 < 60，最终分封顶 50。24 个公开 scenario（F1–F6 × 4）在仓库内，96 个 held-out 在私有 submodule `external/companionbench-heldout/`。

配套 wheel：`companion-ref-harness`（闭源 API 的最小记忆代理，:8500）、`companion-camel-baseline`（CAMEL 同基底消融行，:8600）、`companion-trajgen`（合成轨迹生成，只带 FSM 标签、绝不带 judge 分）。

### 5.2 怎么运行

```bash
# CI 冒烟（无需 API key，确定性 fake）
bash scripts/companion_bench/run_companion_bench_ci_smoke.sh
pip install -e packages/companion-bench && companion-bench smoke && companion-bench list-scenarios && companion-bench hashes

# 真实 API 冒烟（~$1–3；需 .local/llm.env 的 OPENROUTER_API_KEY + LIFEFORM_LOCAL_API_KEY）
bash scripts/companion_bench/run_companion_bench_smoke.sh   # 或 run_companion_bench_smoke.py

# 参考系统批量打分（跨家族 judge：per-turn 与 arc judge 必须不同家族）
python scripts/companion_bench/score_reference_systems.py --output-dir artifacts/companion-bench/reference \
  --user-sim-... --perturn-... --arc-... --paraphrase-seeds 0 --systems openai/gpt-5,anthropic/claude-opus-4.6

# 单次提交
python scripts/companion_bench/run_real_submission.py --submission packages/companion-bench/examples/submission.yaml \
  --user-sim-... --perturn-... --arc-... --paraphrase-seeds 0,1,2 --artifact-dir artifacts/companion-bench/your-submission/

# paper suite：小（夜跑 ~$200–400）/ 全（发布级 $5–15k，手动 CI）
bash scripts/companion_bench/run_companion_bench_paper_suite_small.sh
bash scripts/companion_bench/run_companion_bench_paper_suite_full.sh

# 由真实 artifact 刷新站点（demo→false）
python scripts/companion_bench/build_site.py --artifact-dir artifacts/companion-bench/reference --site-dir site
```

本地 VZ SUT：`bash scripts/companion_bench/start_vz_sut.sh start|stop`。history 臂 `session`(默认)/`stateless`/`full`（可带 `history_token_budget` 近端截断）。P1 同基底消融（**非公开榜**，thesis 旁证）：`bash run_companion_bench_p1.sh` → `verdict_p1.json`。

### 5.3 怎么评价

每弧流水线：arc runner → callback ledger（编造检测）→ per-turn judge（8 准则 0–5）→ arc judge（6 轴 0–100，跨家族）→ disqualifier（确定性谓词 void 轴）→ `aggregate_arc()` → 提交聚合（逐轴均值 + bootstrap 95% CI）→ Elo（TrueSkill + Bradley-Terry）。每轮记录 `sut_latency_ms / context_history_policy / 估算 context tokens / 截断消息数`。成本 `CostTracker` 记 token/USD（缺模型价→null，绝不记 $0）；硬性预算门是债务 #34，尚未 ACTIVE。判官稳健性 / 统计功效 / 校准三份协议均为 SHADOW，结果 TBD。

### 5.4 当前结果

**站点数据全是 demo/smoke**：`site/data/aggregate_results.json` 等标 `"demo": true`；`site/data/submissions/demo-2026q2-*.json` 是合成 demo；`lifeform-companion-smoke.json` / `dashscope-qwen3-max-smoke.json` 是真实冒烟（final ≈ 13.79，verifier pending）。**无正式参考级榜结果**。已知发现：同家族 Qwen judge σ≈23.2（#71，故跨家族 sweep #48 是上榜硬前置）；VZ 默认 `substrate-mode synthetic` 是确定性回声，13.79 vs Qwen 74.56 是基底假象非架构证据（#72）。

### 5.5 下一步（[`moving forward/companion-bench-public-launch-packet.md`](./moving%20forward/companion-bench-public-launch-packet.md)）

1. #48 跨家族 judge 稳健性 sweep 转 ACTIVE（上榜硬前置）；
2. #52 权重 + A6 顶校准；#54 统计功效（n=120 能否区分 SUT）；
3. #82 10 系统 × 120 scenario × 3 seed 全量 → `build_site.py` 刷成 `"demo": false`；
4. #57 trusted runner（外部提交防泄题）、#56 成本模型回填真实数字；
5. P1 同基底消融（MPS + OpenRouter judge 就绪后）作正交 thesis 旁证。GTM 约束：首榜 VZ 应在 A3/A4 子轴领先，而非总分第一。

---

## 6. State-KV / Prefix-KV 辨识证据

**Spec**：[`specs/state-kv-identification-evidence.md`](./specs/state-kv-identification-evidence.md)、[`specs/character-prefix-package.md`](./specs/character-prefix-package.md)；设计 `research/state_kv/01_state_kv_complete_design_plan.md`。
**基底**：`Qwen/Qwen2.5-0.5B-Instruct@857fff1d`。**已晋升 Personal artifact**：`8064f8b6…`（`artifacts/state_kv/projectors/qwen2.5-0.5b-state-strategy-routed-prefix.json`）。

### 6.1 目的

核心主张：**同一 system prompt 字节、空上下文、两个用户 → 不同回答，且跨家族盲判能高于随机地归属到正确用户**。载体只声明模型层（residual C3 / prefix-KV C7 / dynamic residual C4），prompt(C1)/context(C2) 载体被抑制。四条强制 claim：`prompt_identity` / `output_divergence` / `identification_above_chance` / `carrier_causality`。判词阶梯：`insufficient_data → fail → weak-positive → retain-prompt-closed → retain-strict`。L2 角色 Prefix/KV 与 Personal State-KV 分层（L1 CommonAdapterBundle 拥有共享生成器 + rare-heavy delta）。

> 注意：`scripts/run_state_kv_prefix_arm.py` **不存在**；prefix 臂是 profile `state-kv-arm-g-prefix-pure`，经 `run_state_kv_identification.py --lane p3` 行使。

### 6.2 怎么运行（Personal 评测阶梯）

```bash
# 训练 Personal Prefix-KV（state-strategy-routed，晋升 artifact 配方）
python scripts/train_state_kv_prefix.py --device mps --states 16 --epochs 3 --max-new-tokens 48 \
  --route-weight 1.0 --output artifacts/state_kv/projectors/qwen2.5-0.5b-state-strategy-routed-prefix.json

# P4 机制诊断（carrier_is_live）
python scripts/run_state_kv_carrier_diagnostics.py --device cpu --train-states 96 --eval-states 32 --shuffle-draws 5 \
  --prefix-kv-artifact <artifact> --output artifacts/state_kv/p4-state-strategy-routed/verdict_carrier_diagnostics.json

# P3 辨识（Personal prefix + 嵌入盲判）
python scripts/run_state_kv_identification.py --lane p3 --device mps --max-new-tokens 48 \
  --personal-conditioning-scale 0.12 --prefix-kv-artifact <artifact> \
  --judge-kind embedding --judge-model-id BAAI/bge-m3 --judge-device mps \
  --output artifacts/state_kv/p3-state-strategy-routed/verdict_identification.json

# P2 held-out（repair-vs-execute / boundary-vs-commit）→ 保持门 → 判官庭 → 生成种子门
python scripts/run_state_kv_identification.py --lane p2 --p2-pair repair-vs-execute --device cpu \
  --max-new-tokens 16 --temperature 0.2 --sampling-seed 1701 --resume-turn-cache \
  --prefix-kv-artifact <artifact> --judge-kind embedding --judge-model-id BAAI/bge-m3 --judge-device cpu --output <p2 verdict>

# 时序因果 / 安全负例 / 部署门 / 成本门
python scripts/run_state_kv_temporal_causal_evidence.py --device cpu --prefix-kv-artifact <artifact> --output <verdict>
python scripts/run_state_kv_safety_negatives.py --device cpu --prefix-artifact <artifact> --output <verdict>
python scripts/run_state_kv_deployment_gate.py --model-source <snapshot> --device cpu --prefix-kv-artifact <artifact> \
  --temporal-causal <verdict> --judge-court <verdict> --generation-seed-gate <verdict> --output <verdict>
python scripts/run_state_kv_cost_gate.py --device cpu --max-new-tokens 64 --prefix-kv-artifact <artifact> --output <verdict>

# 银行增益 v4（双判官）+ P6 尽调聚合
python scripts/run_state_kv_bank_gain_gate.py --device mps --max-new-tokens 48 --minimum-samples 16 \
  --judge-model-id BAAI/bge-m3 --secondary-judge-model-id moka-ai/m3e-base --output artifacts/state_kv/bank-gain-v4/verdict_bank_gain.json
python scripts/run_state_kv_due_diligence.py --mode all --output-dir artifacts/state_kv/due_diligence
```

Relationship Prefix-KV：`scripts/train_relationship_prefix_kv.py`（prereg 128 样本 / 32 states / 3 epoch / `--norm-cap ≤ 0.12`）+ `run_state_kv_relationship_carrier_pilot.py`。角色 L2：`scripts/bake_zhang_wuji_character_package.py` + `scripts/evaluate_character_package.py`（默认 `CHARACTER_PACKAGE_MODE=shadow`）。

### 6.3 怎么评价（各门 owner）

- 辨识 `state_kv_identification.py`：四 claim 由真实 turn 的 `rationale_tags` 计算；fake substrate 天花板 `insufficient_data`；无盲判则 claim 3/4 停在 `insufficient_data`。
- P4 机制 `state_kv_carrier_diagnostics.py`：Gate A（slot 分化 > 随机且 state spread > sentence spread）+ Gate B（held-out ridge R² > 0.1 且胜过 shuffled-label null）→ `carrier_is_live`。
- 时序因果 / 部署 / 成本 / 安全负例各有 schema 与 `gate_state`；尽调 `state_kv_due_diligence.py` 把设计 §11.3 的 C1–C7 映射到冻结证据，`gate_state = complete | partial`。

### 6.4 当前结果

Personal（晋升路径）**全链 pass**：P3 `retain-strict`（12/12 bge-m3）、P4 `carrier_is_live=true`（A 20/24 层，R² 0.9141）、P2+保持/判官庭/生成种子门 pass、时序因果 pass、部署 pass、成本 pass（51% slot 省，470 vs 1672 ms/token）、安全 C6 pass。

**尽调 partial（3/7 proven）**（`artifacts/state_kv/due_diligence/verdict_due_diligence.json`，freeze `5ce637be…`）：

| ID | 状态 | 说明 |
|---|---|---|
| C1 | not-yet-proven | 手写 prompt/RAG/matched LoRA 等预算对齐臂未实现 |
| C2 | **proven** | G vs B′ 非劣 + 成本/延迟门 |
| C3 | **proven** | G retain-strict、E-pure 随机、P4 非退化 |
| C4 | not-yet-proven | D0 → retain rank-3，全维控制未晋升 |
| C5 | not-yet-proven | credit 机制 pass，**matched_outcome fail** |
| C6 | **proven** | 部署 + stale/extraction 负例 |
| C7 | not-yet-proven | World/Env/Object 银行缺失，**bank-gain fail** |

其它：bank-gain v4 **fail**（Personal independent_gain −0.0625、CI 上界 0.0；bank_count 已冻结）；Relationship Prefix-KV v2 wrong-user 控制升到 0.758，但 **P4 Gate A 仍 fail、盲判 0.50（chance）**；credit longitudinal 仅 `mechanism_supported`。默认 `pe-eta` profile 下 `personal_conditioning=SHADOW`；显式 opt-in profile 为 `state-kv-active-v1`（部署门通过后，**不切默认全局 wiring**）。

### 6.5 下一步（两 plan 标 completed，剩证据关闭）

1. **bank-gain / C7**：Personal 独立增益在 48-token 双判官 v4 失败 → 冻结为**预算无关的失败**，不扩同质样本；Relationship 银行仍冻在 chance。
2. **Relationship 载体**：v2 提升 wrong-user 控制但 P4/盲判仍未过 → 记为**新负结果**，不晋升，默认走 text + SHADOW。
3. **C5 credit outcome**：outcome judge 已接/冻结，需找到 `matched_outcome` 能过的路径，或正式关闭 C5 为 not-yet-proven。
4. C1 手写/RAG CPU 臂（可选）、matched LoRA 受 GPU 债务 #41 门控；C4 D0 已判 retain rank-3，D1/D2 除非有新瓶颈证据否则跳过。
5. **默认 ACTIVE 迁移触发条件**（剩余证明计划）：`C2/C3/C5/C6 proven 且 Personal bank-gain pass`——**未满足**（C5 + bank-gain 未闭合），不切全局 wiring。
6. 角色 L2 ACTIVE 需 1.5B 路径的 held-out fidelity + gate（`evaluate_character_package.py`），与 Personal 0.5B 晋升分开。

---

## 7. ETA rate-distortion 与 LLM 迁移阶梯

**SSOT**：[`specs/eta-llm-transfer-evidence.md`](./specs/eta-llm-transfer-evidence.md)（能力边界 [`specs/temporal-abstraction.md`](./specs/temporal-abstraction.md)）；执行计划 `.cursor/plans/eta_迁移_llm_阶梯_59d33511.plan.md`。**全部 evidence lane，不改 production wiring。**

### 7.1 目的与四级阶梯

在冻结 LLM 上检验 ETA 论文 Eq.3：α 增大时 metacontroller 用 rate（`z_t` 后验 KL）换 distortion（专家动作 NLL），产生近乎垂直的率失真 gap，且 `beta_t` 切换落在 gap 内。阶梯把「仪器坏（Gate 1）」和「ETA 机制不存在（Gate 3）」分开：

| Stage | Gate | 问题 | 关键脚本 | 过 → / 不过 → |
|---|---|---|---|---|
| 1 | Gate 1 | 种子语料上 rate 仪器有效吗？ | `run_eta_rate_distortion.py`(frozen)、`run_eta_rate_axis_pilot.py`、`screen_eta_rate_axis_surrogate.py` | **PASS**（2026-08-03）→ 进 Stage 2 |
| 2 | Gate 2 | 域续训 LLM residual 是否载子目标？ | `run_eta_stage2_corpus.py → _pretrain.py → _probe.py`（v2 仪器：`--document-protocol` / `--probe-protocol` v4、`--layer-selection train-split`） | v1 FAIL＝仪器定罪；v2 实质两条件 PASS（0.944）但按字面 FAIL（第二条件 regime 错配）→ 待 v3 决策 |
| 3 | Gate 3 | 补课冻结 LLM 上出现率失真 gap 吗？ | `run_eta_rate_distortion.py --model-source <merged>` | **锁定**（Gate 2 尚无正式 PASS） |
| 4 | 待定 | 对话迁移（MSC） | 仅骨架 `research/eta/eta-stage4-dialogue-transfer-prereg-skeleton.md` | — |

### 7.2 怎么运行（Stage 1）

```bash
# 冻结协议
.venv/bin/python scripts/preregister_eta_rate_distortion.py --output artifacts/eta_rate_distortion_prereg_<ts>.json \
  --alphas 0.01 0.03 0.1 0.3 1.0 3.0 --seeds 3 --n-z 16 --updates 40 --device mps \
  --arms frozen joint --model-id Qwen/Qwen2.5-0.5B-Instruct --corpus-seed 20260802 \
  --train-routes 200 --heldout-routes 60 --observation-protocol partially-observable-explicit-plan.v3 \
  --posterior-parameterization smooth --rate-gating switch-gated

# 执行 sweep（frozen 臂足以判 Gate 1）
.venv/bin/python scripts/run_eta_rate_distortion.py --output-dir artifacts/eta_rate_distortion_<ts> \
  --preregistration artifacts/eta_rate_distortion_prereg_<ts>.json --corpus-seed 20260802 \
  --train-routes 200 --heldout-routes 60 --observation-protocol partially-observable-explicit-plan.v3 \
  --posterior-parameterization smooth --rate-gating switch-gated [--resume]
```

> MPS 独占锁 `artifacts/.companion-evidence-mps.lock`（`plan_id=eta-rate-distortion-mps.v1`）+ `require_mps()`（`PYTORCH_ENABLE_MPS_FALLBACK` 必须 0）。runner fail-closed 校验 sweep 参数 / gap 阈值 / 5 个冻结源 SHA。无 prereg → `mechanism-only-smoke`，非权威。

### 7.3 怎么评价

- **Gate 1**（frozen 足够）：`spearman(alpha, rate) ≤ −0.8`；`rate_span ≥ 0.30` 且显著大于 7-route 基线；切换存在（boundary F1 > 0、有硬切换）。
- **Gate 3**（需 frozen + joint 两臂）：两臂可区分（分离 ≥ max(2×pooled_std, 0.02)）；frozen 有 gap（drop_share ≥ 0.5，跨 ≤ 25% rate span）且 joint 无 gap；gap 内 boundary F1 > gap 外 → `retain-eta`；frozen 无 gap → `kill-eta`；两臂不可区分 → `instrument-invalid`；单臂 → `incomplete-sweep`。

### 7.4 当前结果

| Artifact | 判词 | 关键数 |
|---|---|---|
| `artifacts/eta_rate_distortion_20260801` | `kill-eta` | **未预注册** + 脏树 + MPS 争用，仅机制级 |
| `artifacts/eta_stage1_gate1_reduced_20260802` | `incomplete-sweep`(frozen) | spearman −0.657（> −0.8，Gate1 **FAIL**）、rate_span 0.585 |
| `artifacts/eta_stage1_gate1_smooth_v2_20260802` | `incomplete-sweep`(frozen) | spearman −1.0、rate_span 0.691，但 switch_freq 0.0、boundary F1 0.0（切换门 **FAIL**） |
| `artifacts/eta_stage1_gate1_v3_gated_20260802` | 7/18 后中止 | rate 轴单调但 distortion 平坦、零切换；定位根因=v3 全计划 step-0 一次到达 → never-switch 即 Eq.3 最优 |
| **`artifacts/eta_stage1_gate1_v4_hardst_auth_20260803`** | **Gate 1 = PASS** | spearman −1.000、rate_span 1.933、hard switch 0.12–0.96、heldout boundary F1 全 alpha 0.240–0.671；`gate1_assessment.{json,md}` 已封存 |
| `artifacts/eta_stage2_probe_20260803` | Gate 2 v1 仪器 FAIL（**经审计定罪仪器**） | 续训臂末层 0.131、裸 Qwen 0.166（=majority）；审计：计划载体为哈希指纹，heldout 信息天花板 0.1805 < 及格线 0.25，**构造性不可过**，实测值恰在天花板附近 |
| **`artifacts/eta_stage2_probe_v2_20260803`** | Gate 2 v2 仪器：实质两条件 PASS，按字面 **FAIL** | 裸 Qwen `0.901`（全层 0.795–0.976）、续训 `0.944`（后段层 0.99–1.00）；`2×chance` PASS（7.5×）、`续训>基线` PASS（+4.3pp）、`随前缀上升` FAIL（early 0.979/late 0.879，保持衰减非推断累积，判据 regime 错配） |

**Gate 1 = PASS（2026-08-03）**：修尺子四层根因——smooth posterior + v4 分段揭示协议（step-0 只给第一目标、各 arrival 揭示下一个）+ switch-gated KL（keep 免费/switch 付费）+ hard-st 离散门（堵住连续门每步微幅走私）+ 300 updates。frozen 臂另检出方向性近垂直 gap（drop share 0.744 / rate share 0.196），但缺 joint 臂且 gap 区内 F1（0.394）未高于区外（0.537），属 Gate 3 范畴不予主张。

**Gate 2（2026-08-03，两代仪器）**：v1 全链 FAIL（0.131/0.166）后仪器审计发现计划载体是 `_context_sentence` 哈希指纹（与 Gate-1 定罪的协议 v2 缺陷同类）：非指纹信息的 heldout 贝叶斯天花板仅 0.1805，**v1 的 2×chance 条件构造性不可过**，FAIL 定罪仪器而非基底。仪器 v2（v4 staged-plan 渲染语料 + 累积轨迹前缀 probe + train-split 选层，prereg `artifacts/eta_stage2_gate2_prereg_v2_20260803/` sha256 `c0a54454…` 含 ceiling 1.0 验证）重跑全链（语料 `1caeac3a…` → merged `063077b7…`，0.967→0.020 → probe）：**残差流大幅承载 active subgoal**（0.901/0.944），因果对照成立；仅 `随前缀上升` 败（显式揭示制度下无推断累积、只有保持衰减，late 桶仍 7× chance）。按预注册字面 v2 verdict = FAIL 封存（执行后禁改条件）。整体 `kill-eta`（2026-08-01）持续有效。

### 7.5 下一步

1. **Gate 2 待决策**：是否注册 v3 预注册，把第二条件修为分段揭示制度下的保持类判据（如 late 桶 ≥ 2×chance 且 early−late ≤ 阈值）后终审；v2 实质读数（0.944/0.901/+4.3pp）已登记为"残差可承载"的强方向性证据。
2. Stage 3 在 Gate 2 正式 PASS 前锁定；Stage 4 仅骨架。
3. Gate 1 权威扫仍作机制回归基线：相关 owner / 算法变量 / 证据契约改变时按预注册重跑，普通重构不触发重复昂贵 evidence run。

---

## 8. ETA 内部 RL / 段信用强证据（独立证明环境）

与 §7 的 LLM 迁移阶梯是**不同证据程序**（段信用 retain ≠ rate-distortion retain）。实现 `packages/vz-runtime/src/volvence_zero/agent/eta_proof_benchmark.py`；研究日志 `research/eta/eta-segment-credit-evidence-plan.zh.md`。

### 8.1 目的与运行

验证论文式层级稀疏奖励命题：段级信用 vs turn 级、抽象动作复用、held-out 组合、延迟信用对齐、policy-update 证据。均为 evaluation 内部只读工件，不改 `EvaluationSnapshot` 公共 shape。

```bash
# 内部 RL paper suite（trace 后端无需 GPU；tier: ci-smoke | paper-suite-small | paper-suite-full）
bash scripts/run_eta_paper_suite.sh artifacts/eta_paper_suite paper-suite-small
# 真实 open-weight Qwen residual（MPS）
bash scripts/run_eta_open_weight_paper_suite.sh artifacts/eta_open_weight_paper_suite paper-suite-small

# 段信用 vs turn 级强证据
.venv/bin/python scripts/run_eta_segment_credit_evidence.py --output-dir artifacts/eta_segment_credit_<ts> \
  --seeds 10 --backend transformers-open-weight --device mps --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --training-mode ssl-rl-alternating --training-cycles 3 --ssl-updates-per-cycle 25 --controller-dim 16

# Gate 2 residual-control matched 消融
.venv/bin/python scripts/run_eta_gate2_residual_evidence.py --output-dir artifacts/eta_gate2_residual_<ts> \
  --suite-tier ci-smoke --device mps
```

### 8.2 怎么评价 / 当前结果

`run_eta_internal_rl_paper_suite()` 分离 primary outcome（held-out success / reuse / credit alignment）与 composite readout，并把 `statistical-batch-evidence`、`claim_eta_real_open_weight_residual_control` 拆成独立门（真实 residual claim fail-closed 要求 fallback rate 0.0、hook fire rate ≥ 0.75）。

**最新段信用强 retain**：`artifacts/eta_evidence_gate_1/segment_vs_turn_decoder_family_manifold_v13_qwen25_05b_mps_10seed_75update_20260728` → 判词 **`retain`**，12 门全过（credit F1 delta 0.90，beta boundary F1 0.765）。

### 8.3 下一步

作为**机制回归**保留，防止段级信用能力退化；它不等于 rate-distortion retain，也不改任何 production wiring。相关机制、算法变量或 promotion gate 改变时才重跑重复 seed 证据。

---

## 9. Digital Ant ecology（same-physics 因果阶梯）

**Spec**：[`specs/digital-ant-embodiment.md`](./specs/digital-ant-embodiment.md)；包 README [`packages/vz-embodiment-ant/README.md`](../packages/vz-embodiment-ant/README.md)
**Owner / evaluator**：`packages/vz-embodiment-ant/src/volvence_ant/experiments/`（`ecology_same_physics_*.py`、`ecology_p1.py`、`ecology_p2.py`、`ecology_mechanism_audit.py`）+ `evidence/`
**终局叙述**：[`thesis prove.md`](./thesis%20prove.md) §4（历史 v2）与 §13（L1 追加）

### 9.1 目的

把 VolvenceZero 内核原样接到**完全不涉及语言**的 2D 感觉运动 substrate（无 LLM、无 token），独立检验 R2 / R3–R4 / R5–R6 / R-PE / SSOT 是否在非语言世界成立。ecology same-physics 是其中的 **typed-milestone 因果阶梯**：candidate（`environment_milestone_temporal_switch=ACTIVE`）vs matched control（`DISABLED`），物理/日程/优化器完全一致，只差这一条 wiring。

条件链（冻结准入）：

```text
prereg → station1 → (v2 时代可选: alignment review) → station2 medium
  → Gate-4 ecology corpus 准入 → P1 五臂 → P2 PE confirmatory
```

它**不是**产品线，也**不是**昆虫神经科学新发现；投入规模长期保持研究旁支量级。正式跑需要**隔离源码快照 + 新空 progress 目录**；旧 v31 journal 禁止续跑。设备：**CPU float64**（不占 MPS 锁）。

### 9.2 怎么运行

```bash
# --- 当前代：same-physics station1-v4（L1-C；review 路径已关闭）---
# 先决：L1-B precheck artifact 已存在（prereg 会校验）
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
.venv/bin/python scripts/preregister_ant_ecology_same_physics_baseline.py \
  --seed 0 --run-id "$RUN_ID"
# → research/ant/results/ecology_recovery/same_physics_baseline/ecology_same_physics_prereg.seed0.<RUN_ID>.json

PREREG="research/ant/results/ecology_recovery/same_physics_baseline/ecology_same_physics_prereg.seed0.${RUN_ID}.json"
PROGRESS="research/ant/results/.partials/ecology_same_physics_station1_v4/seed0-${RUN_ID}"
.venv/bin/python scripts/run_ant_ecology_same_physics_station1.py \
  --preregistration "$PREREG" --progress-dir "$PROGRESS" --run-id "$RUN_ID"

# --- L1-A 归因 / L1-B 形成期保护预检（只读，无训练写）---
.venv/bin/python scripts/analyze_ant_alignment_formation.py \
  --station1-report research/ant/results/ecology_recovery/same_physics_baseline/ecology_same_physics_station1.seed0.20260731T052300Z.json \
  --review-report research/ant/results/ecology_recovery/same_physics_baseline/ecology_same_physics_alignment_review.seed0.20260731T053814Z.json \
  --station1-progress-dir <station1-progress> --review-progress-dir <review-progress> \
  --json-out research/ant/results/ecology_recovery/same_physics_baseline/alignment_formation_attribution.v1.json

.venv/bin/python scripts/precheck_ant_alignment_formation_protection.py \
  --review-report research/ant/results/ecology_recovery/same_physics_baseline/ecology_same_physics_alignment_review.seed0.20260731T053814Z.json \
  --review-progress-dir <review-progress> --probe-seed 700003 \
  --json-out research/ant/results/ecology_recovery/same_physics_baseline/alignment_formation_protection_precheck.v1.json

# --- 历史 v2 路径（station1 GO + 唯一 5 局 review；勿当作 reopen）---
# scripts/preregister_ant_ecology_same_physics_alignment_review.py
# scripts/run_ant_ecology_same_physics_alignment_review.py
# （需要 station1-v2 prereg/report/progress；当前代 v4 已设 alignment_review_authorized=false）

# --- 下游（station2 GO 前一律 NOT AUTHORIZED；仅留命令备查）---
.venv/bin/python scripts/audit_ant_ecology_mechanisms.py
.venv/bin/python scripts/run_ant_ecology_p1.py --diagnostics-only
.venv/bin/python scripts/run_ant_ecology_p1.py --seed 0 \
  --progress-dir research/ant/results/.partials/ecology_p1/seed0
.venv/bin/python scripts/run_ant_ecology_p2.py preflight --p1-report <p1-pass.json>
.venv/bin/python scripts/run_ant_ecology_p2.py shard --training-seed 0 --arm <arm> --p1-report <p1-pass.json>
.venv/bin/python scripts/run_ant_ecology_p2.py aggregate
.venv/bin/python scripts/run_ant_ecology_p2.py promote --confirmatory-report <p2-pass.json> \
  --progress-dir <p2-progress> --training-seed 0

# 可选 mastery trainer（不是 same-physics 因果链）
.venv/bin/python scripts/train_ant_ecology.py --seed 0 --device cpu
```

P1/P2 细节另见 `research/ant/06_ecology_implementation_status.md`。

### 9.3 怎么评价

评估器：`ecology_same_physics_run.evaluate_same_physics_station1`；阈值冻结在 prereg（`ecology_same_physics_baseline.py`）。

| 层 | 门 | 规则 |
|---|---|---|
| Station1 因果（4） | `control_signal` | control ≥1 pickup / physical block |
| | `pickup_noninferiority` | candidate/control pickup 比 ≥ **0.8** |
| | `no_candidate_zero_block` | control >0 的 block 上 candidate 不得零 |
| | `typed_milestone_structure` | post-pickup switch + family persistence；**8 条** structural/persistence lane 全过 |
| Alignment | food turn probe | `input_reachable ∧ action_sensitive ∧ target_aligned`；turn 阈值 **1e-4**；需 **4/4 bodies** |
| Station1→2 | v2 时代 | 因果 GO 且 alignment <4/4 → 唯一 5 局 butter-near review；仍 <4/4 → BLOCK |
| | v4（L1-C） | `food_alignment_review_authorized=false`；必须在 ep0–19 直接 **4/4**，否则 BLOCK |
| Station2 | medium | pickup ≥0.8 非劣；deliveries **严格优于** control；携食回巢 alignment + U-turn 进步 |
| P1 / P2 | station2 GO 后 | P1 多臂矩阵；P2 PE-on/off confirmatory（`preflight/shard/aggregate/promote`） |

Station1 上的 delivery 只作**描述性**读数，不是硬门。PE-off 对照只关加性 PE prior，里程碑通道两臂保持一致（它是环境事实，不是 PE readout）。

### 9.4 当前结果（两代勿混读）

| 时代 | Artifact | 判词 |
|---|---|---|
| **§4 历史 v2** | `…/ecology_same_physics_station1.seed0.20260731T052300Z.json` | **GO** — 四因果门过；pickup 47/52（ratio 0.9038）；8/8 lane；alignment **3/4** → `REVIEW_REQUIRED` |
| | `…/ecology_same_physics_alignment_review.seed0.20260731T053814Z.json` | **BLOCK** — 审前审后仍 **3/4**（需 4/4）；`next_episode_authorized=null` |
| **L1-A** | [`alignment_formation_attribution.v1.json`](../research/ant/results/ecology_recovery/same_physics_baseline/alignment_formation_attribution.v1.json) | 失败者稳定为 body **2**；H1 支持、H2 不支持、H3 inconclusive；station2 仍未授权 |
| **L1-B** | [`alignment_formation_protection_precheck.v1.json`](../research/ant/results/ecology_recovery/same_physics_baseline/alignment_formation_protection_precheck.v1.json) | `PRECHECK_PASS`（ACTIVE/DISABLED digest 相等）；只授权 L1-C prereg，**不**授权 station2/P1/P2 |
| **§13 L1-C** | [`ecology_same_physics_station1.seed0.20260731T135415Z.json`](../research/ant/results/ecology_recovery/same_physics_baseline/ecology_same_physics_station1.seed0.20260731T135415Z.json) | **BLOCK** — control/pickup/zero-block/structure 全 true，`food_alignment_4_of_4` **false**（仍 3/4）；body2 turns `−3.85e-4 / +3.87e-4`；`alignment_review_authorized=false`、`next_episode_authorized=null` |

下游终态（[`thesis prove.md`](./thesis%20prove.md) §4.2）：**station2 / Gate 4 ecology corpus / P1 / P2 = `not-authorized`**——不是「跑了然后失败」，而是按预注册 kill 纪律没有执行。可说：early station1 的局部机制、结构持久性与 early pickup 达标；不可说：medium 闭环、一般物理自主性、或形成期保护带来了要求的 learned uplift。

### 9.5 下一步

1. **本代已关闭**：禁止第二次 alignment review、换 seed、降 4/4 阈值、或在同一 packet 上加训练量（见 [`thesis prove.md`](./thesis%20prove.md) §13）。
2. **合法重开**：必须先有**新的 owner 级机制**（若是测量语义 kill，还要新 schema）+ 新 prereg + 隔离源码快照 + 空 journal——不是对 station1-v4 再探针一次。
3. **station2 GO 前禁止**：P1 全矩阵、P2 confirmatory、Gate 4 ecology corpus、以及拟用 Ecology 重开的 Gate 5/9/10/1。
4. L1-B 形成期保护：机制可实现、可预检、可回滚，但未在冻结 station1 上产生要求的 uplift；记为终局负证据，medium 层写「未测且不授权」，不伪造 PASS/FAIL。

---

## 10. 跨评测纪律（务必遵守）

- **MPS 独占**：七日 / MSC / ETA rate-distortion / gate suite 共享同一把锁 `artifacts/.companion-evidence-mps.lock`，`PYTORCH_ENABLE_MPS_FALLBACK=0`，**不得并发**；控制面外手工启动的旧进程不受锁保护，须另行确认结束。Digital Ant ecology 走 **CPU float64**，不占该锁，但仍勿与其它会改同一源码树/journal 的 formal 包并行污染。
- **冻结在先**：长任务前冻结执行根/源码树 + 预注册 SHA；读到任何 outcome 之前不改阈值/seed/判据；封包后源码漂移一律拒绝（用隔离 commit 快照复核，只重做 validation/export）。Ecology 正式跑另要求**隔离源码快照 + 新空 progress 目录**。
- **停跑与 kill 是合格终态**：不换 seed、不降阈值、不挑 metric 把结论磨成通过；停跑目录禁止原样 resume，重开必须新 prereg。Ecology 的 `not-authorized` 下游不得写成已执行的新负证据。
- **证据等级不互相冒充**：`mechanism-supported`（能跑/可回滚/可审计）≠ `causal-supported`（冻结 matched control 下达门的因果差）≠ `longitudinal-supported`（跨 session 持续）≠ `thesis-retained`。可回滚是安全证据，不是收益证据。
- **evaluation 只读**：所有金标评测不回灌 PE/credit，不反向训练 probe，不静默成为第二 owner。
- **production 晋升**：#92 终局 `production_live_promotion_authorized=false`；本文任何门通过都不自动翻转 `WiringLevel`；晋升走 SHADOW→单组件 canary→可回滚切换，并先登记 `docs/DATA_CONTRACT.md`。

## 11. 权威参考

- 框架口径：[`EVALUATION_SYSTEM.md`](./EVALUATION_SYSTEM.md)、[`specs/evaluation.md`](./specs/evaluation.md)、[`specs/evaluation-cascade.md`](./specs/evaluation-cascade.md)
- 终局判词与 Gate 台账：[`thesis prove.md`](./thesis%20prove.md)、[`specs/evidence_program.md`](./specs/evidence_program.md)
- 当前事实与剩余代码：[`currentstatus.md`](./currentstatus.md)
- 七日 × MSC 静态缺陷/仪器清单：[`moving forward/七日msctodo.md`](./moving%20forward/七日msctodo.md)
- 各评测 spec：[`specs/seven-day-companion-evidence.md`](./specs/seven-day-companion-evidence.md)、[`specs/companion-bench.md`](./specs/companion-bench.md)、[`specs/state-kv-identification-evidence.md`](./specs/state-kv-identification-evidence.md)、[`specs/character-prefix-package.md`](./specs/character-prefix-package.md)、[`specs/eta-llm-transfer-evidence.md`](./specs/eta-llm-transfer-evidence.md)、[`specs/learned-vs-heuristic-coverage.md`](./specs/learned-vs-heuristic-coverage.md)、[`specs/digital-ant-embodiment.md`](./specs/digital-ant-embodiment.md)
