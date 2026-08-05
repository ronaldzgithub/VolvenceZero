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
| 7 | ETA rate-distortion + LLM 阶梯 | 冻结 LLM 上 ETA 率失真机制是否成立（四级阶梯） | `scripts/run_eta_rate_distortion.py` | Stage3 **`kill-eta`**（operationalization-scoped）；P1 定位 free-bias bypass（96.3% recovery）；S1 v2 readout PASS，S2 additive no-bias steering 五门 FAIL | **转向"读残差+有界条件 steering+Internal RL 学何时扳"**：C2 条件写入 PASS、S3-前置非 oracle sensor PASS、S3-A 余量 PASS、**S3-E 学"何时扳" admission PASS（5/5）** → 三层闭环（读得到+扳得动+学会何时扳）；仍 SHADOW/evidence-lane、production 未提升；Stage4 关闭（见 §7.6、`research/steering-2026-08/`） |
| 8 | ETA 内部 RL / 段信用强证据 | 段级信用 vs turn 级、抽象动作复用等论文式命题 | `scripts/run_eta_segment_credit_evidence.py` / paper suite | 段信用 v13 **retain**（12 门全过） | 作为机制回归保留；不等于 rate-distortion retain |
| 9 | Digital Ant ecology（same-physics） | 无 LLM 的 2D 感觉运动床上，typed-milestone ACTIVE vs DISABLED 是否有因果净增益 | `scripts/run_ant_ecology_same_physics_station1.py` | L1-C station1-v4 **BLOCK**（food alignment 3/4）；station2/P1/P2 **not-authorized** | 新 owner 机制 + 新 prereg 才可重开；禁止二审/换 seed/降阈值 |
| 10 | RSI Forge（开发环自改进） | 从真实失败挖模式 → 有界提案 → 循环外验证/人审 → ledger 预测兑现 | `forge` CLI（`mine/propose/validate/select/apply`） | 战役一 e2e 已 apply 2 条；overlay 默认 DISABLED；无 live/GPU 晋升 | 下一轮 `mine --evidence-since-ledger` 兑现预测；扩 task-level held-out；rare-heavy 需冻结权重证据 |

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
| 1 | Gate 1 | 种子语料上 rate 仪器有效吗？ | `run_eta_rate_distortion.py`(frozen)、`run_eta_rate_axis_pilot.py`、`screen_eta_rate_axis_surrogate.py` | **PASS**（2026-08-03，v4+hard-st 权威扫）→ 进 Stage 2 |
| 2 | Gate 2 | 域续训 LLM residual 是否载子目标？ | `run_eta_stage2_corpus.py → _pretrain.py → _probe.py`（v1→v2→v3） | 三轮字面 FAIL 封存（仪器/判据/对照）；实质命题跨两 seed 四臂证实 → **用户裁定 (b) 解锁 Stage 3** |
| 3 | Gate 3 | 补课冻结 LLM 上出现率失真 gap 吗？ | `run_eta_rate_distortion.py --model-source artifacts/eta_stage2_merged_v2_20260803 --arms frozen joint` | **`kill-eta`**（36/36；双臂可分、frozen rate 轴有效但无 gap）→ 当前 operationalization REJECT |
| 4 | 待定 | 对话迁移（MSC） | 仅骨架 `research/eta/eta-stage4-dialogue-transfer-prereg-skeleton.md` | **不启动**；Gate 3 已 FAIL |

### 7.2 怎么运行

**Gate 1 权威扫（已封存 PASS；复现用）**：

```bash
# 预注册已冻结：artifacts/eta_stage1_gate1_v4_hardst_20260803_prereg.json
# 关键参数：v4 staged-plan + smooth + switch-gated + hard-st，updates=300，
# corpus_seed 20260802，64 train / 24 heldout，frozen 臂，6α×3seed=18 cells
.venv/bin/python scripts/run_eta_rate_distortion.py \
  --output-dir artifacts/eta_stage1_gate1_v4_hardst_auth_20260803 \
  --preregistration artifacts/eta_stage1_gate1_v4_hardst_20260803_prereg.json \
  --device mps --arms frozen \
  --observation-protocol partially-observable-staged-plan.v4 \
  --posterior-parameterization smooth --rate-gating switch-gated --gate-mode hard-st \
  --corpus-seed 20260802 --train-routes 64 --heldout-routes 24 \
  --updates 300 --n-z 16 --seeds 3 [--resume]
```

**Gate 2 全链（v3 终审配方；v1/v2 封存不重跑）**：

```bash
# 1) 语料（v3 用全新 seed 20260804 挡 forking paths；v4 可读 staged-plan）
.venv/bin/python scripts/run_eta_stage2_corpus.py \
  --output-dir artifacts/eta_stage2_corpus_v3_20260803 \
  --corpus-seed 20260804 --train-routes 120 --heldout-routes 60 \
  --document-protocol partially-observable-staged-plan.v4

# 2) 续训 + merge（v3 探针文本最长 507 → --max-length 512 即可；v2 用 640）
.venv/bin/python scripts/run_eta_stage2_pretrain.py \
  --corpus-file artifacts/eta_stage2_corpus_v3_20260803/corpus.jsonl \
  --merged-out artifacts/eta_stage2_merged_v3_20260803 \
  --output-dir artifacts/eta_stage2_pretrain_v3_20260803 \
  --device mps --max-steps 2000 --max-length 512

# 3) 线性 probe + Gate 2（累积轨迹前缀 + retention.v3 + train-split 选层）
.venv/bin/python scripts/run_eta_stage2_probe.py \
  --output-dir artifacts/eta_stage2_probe_v3_20260803 \
  --pretrained-model-source artifacts/eta_stage2_merged_v3_20260803 \
  --corpus-seed 20260804 --device mps \
  --probe-protocol partially-observable-staged-plan.v4 \
  --layer-selection train-split --max-length 640 \
  --gate-conditions retention.v3
```

**Gate 3 权威扫（已完成封存；复现用，不建议立即重跑）**：

```bash
# 预注册：artifacts/eta_stage3_prereg_v2_20260803/preregistration.json
# Gate-1 同款尺子 + Stage-2 v2 merged 基底 + frozen/joint 双臂 = 36 cells
.venv/bin/python scripts/run_eta_rate_distortion.py \
  --output-dir artifacts/eta_stage3_rate_distortion_20260803 \
  --preregistration artifacts/eta_stage3_prereg_v2_20260803/preregistration.json \
  --model-source artifacts/eta_stage2_merged_v2_20260803 \
  --device mps --arms frozen joint \
  --observation-protocol partially-observable-staged-plan.v4 \
  --posterior-parameterization smooth --rate-gating switch-gated --gate-mode hard-st \
  --corpus-seed 20260802 --train-routes 64 --heldout-routes 24 \
  --updates 300 --n-z 16 --seeds 3 [--resume]
```

> MPS 独占锁 `artifacts/.companion-evidence-mps.lock`（`plan_id=eta-rate-distortion-mps.v1`）+ `require_mps()`（`PYTORCH_ENABLE_MPS_FALLBACK` 必须 0）。runner fail-closed 校验 sweep 参数 / gap 阈值 / 冻结源 SHA。无 prereg → `mechanism-only-smoke`，非权威。本轮运行期间保持 `analysis_allowed=false`，仅 36/36 完成后判读；manifest 中五个冻结 source hash 与预注册一致。

### 7.3 怎么评价

- **Gate 1**（frozen 足够）：`spearman(alpha, rate) ≤ −0.8`；`rate_span ≥ 0.30`；切换存在（boundary F1 > 0、有硬切换）。权威扫另要求 `gate_mode=hard-st`（堵住连续门微幅走私）。
- **Gate 2**（线性 probe，三条件全过才字面 PASS）：`2×chance`（heldout acc ≥ 0.25）；第二条件按仪器版本分化——v1/v2=`rises-with-prefix.v1`，v3=`retention.v3`（late ≥ 2×chance 且 early−late ≤ 0.15）；`续训 > 裸基底`。字面 FAIL 不自动等于「残差装不下」——须看仪器审计与失败方向。
- **Gate 3**（需 frozen + joint 两臂）：两臂可区分（分离 ≥ max(2×pooled_std, 0.02)）；frozen 有 gap（drop_share ≥ 0.5，跨 ≤ 25% rate span）且 joint 无 gap；gap 内 boundary F1 > gap 外 → `retain-eta` / `retain-eta-on-llm`；frozen 无 gap → `kill-eta`（永久摘除）；两臂不可区分 → `instrument-invalid`；单臂 → `incomplete-sweep`。

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
| **`artifacts/eta_stage2_probe_v3_20260803`** | Gate 2 v3（新 seed 20260804 + retention 判据）：字面 **FAIL**，败因反转 | `2×chance` PASS（0.967 = 7.7×）、`retention.v3` PASS（0.995/0.918，衰减 0.077 ≤ 0.15）、`续训>基线` FAIL——**裸 Qwen 基底 0.977 反超续训臂 0.967**（基底无需续训已在天花板携带子目标） |
| **`artifacts/eta_stage3_rate_distortion_20260803`** | **`kill-eta`**（36/36） | 双臂可分 0.1264 > 0.0673；frozen rate Spearman −0.9429 / span 2.0680，但 `gap_detected=false`（最大 drop 横跨 84.16% rate span），现有 boundary F1 区内 0.000 < 区外 0.2669；joint `gap_detected=true`；manifest authoritative |

**Gate 1 = PASS（2026-08-03）**：修尺子四层根因——smooth posterior + v4 分段揭示协议（step-0 只给第一目标、各 arrival 揭示下一个）+ switch-gated KL（keep 免费/switch 付费）+ hard-st 离散门（堵住连续门每步微幅走私）+ 300 updates。frozen 臂另检出方向性近垂直 gap（drop share 0.744 / rate share 0.196），但缺 joint 臂且 gap 区内 F1（0.394）未高于区外（0.537），属 Gate 3 范畴不予主张。

**Gate 2（2026-08-03，三代仪器/判据）**：v1 全链 FAIL（0.131/0.166）后仪器审计发现计划载体是 `_context_sentence` 哈希指纹（与 Gate-1 定罪的协议 v2 缺陷同类）：非指纹信息的 heldout 贝叶斯天花板仅 0.1805，**v1 的 2×chance 条件构造性不可过**，FAIL 定罪仪器而非基底。仪器 v2（v4 staged-plan 渲染语料 + 累积轨迹前缀 probe + train-split 选层，prereg `c0a54454…` 含 ceiling 1.0 验证）重跑全链：**残差流大幅承载 active subgoal**（0.901/0.944），因果对照成立；仅 `随前缀上升` 败（regime 错配），字面 FAIL 封存。v3（用户授权）修第二条件为 `retention.v3` 并**换全新 seed 20260804 挡 forking paths**（prereg `2f3b3bf4…` 在新读数前冻结）：修正后的两条件双 PASS（0.967 = 7.7×；late 0.918 / 衰减 0.077），但因果对照反向失效——**裸 Qwen 基底 0.977 已在天花板，续训无超越余量**，字面仍 FAIL 封存。三轮合并判读：实质命题"0.5B 残差可线性承载子目标层级"跨两 seed 四臂复现（0.901/0.944/0.977/0.967）；被证伪的是"续训必要性"。三个字面 FAIL **原样封存不改判**；用户程序级裁定取处置 **(b)**——Gate-2 看门前提已实质达成，解锁 Stage 3 推进权（不是把任一 FAIL 改成 PASS）。机制级 `kill-eta`（2026-08-01）在 Stage 3 撤销前**持续有效**。

**Gate 3（2026-08-04）**：预注册权威扫完成并封存 **`kill-eta`**。rate
侧仪器和双臂可分性均通过，失败来自 frozen 臂没有近垂直 rate–distortion
gap；joint 臂反而检出 gap，现有 action-change boundary F1 也只在 joint
候选区内抬升。结论范围是当前 16 维折叠入口 + additive steering / free bias
的 operationalization，不是 ETA 理论普遍证伪。三项结构性不等价登记为
P1 解释债：入口 surface、边界真值、steering 参数化。P1 只做只读归因，
不得改变正式 verdict，也不把 evaluation 结果回灌训练。

**P1 attribution（2026-08-04）**：`artifacts/eta_stage3_equivalence_diagnostic_20260804/`
完成 6/6 matched cells，prereg sha256 `30b827b3…`，source Stage-3 report
sha256 `48a589d4…`。exact-entry 16 维 probe `0.3913` > `0.25`，入口仍可读
但只保留 Gate-2 的 41.45%；bias-only recovery `0.9632`、zero-z recovery
`0.6141`，cyclic-permuted-z penalty `−0.0068`。自动主归因是
`incentive-bypass-via-free-bias`，learned z 未显示稳定时序因果性。oracle
subgoal 与 action-change F1 的均值差 `−0.0685` 未过 `|Δ|≥0.10` 门。

### 7.5 下一步

1. **S1 frozen residual readout（权威 v2 PASS）**：新增
   `substrate_residual_readout` offline/SHADOW owner。v1 prereg `b09b68f8…`
   事后发现 heldout `7/299` 个 prefix 被 512-token 静默截断，故未被 S2
   消费；fail-loud v2 prereg `35c92904…` 固定 layer 20 / width 896 / max 768。
   真实 MPS train/heldout `551/299`，heldout accuracy `0.9833`、late `0.9720`、
   generalization gap `0.0167`，四条 admission 门全过。artifact
   `frozen-residual-readout.v1:086a8f3d…` 发布无 bias 的 8 条 class-vs-rest 轴；
   v1/v2 artifact id 相同是因为训练行未变。未安装、不回灌、不进 production DAG。
2. **S2 causal steering（已 FAIL）**：prereg `b6a427d0…` 只消费 S1 v2 轴，
   对 24 routes / 299 prefix 做有界 `+axis / −axis / noop / shuffled-axis`
   matched control；0 截断、0 substrate trainable parameter、free bias=false。
   0.50×cap 主判 plus vs noop `−0.00072`（95% CI
   `[−0.01787,0.01809]`），plus vs minus `0.02829`，plus vs shuffled
   `0.00709`，五项 admission 条件全败。线性可读轴未形成稳健动作因果接口。
3. **S3 不启动（当时结论，已被 §7.6 转向取代）**：S2 是产品域生死门，因此不以
   PE-gated segment credit / small-action Internal RL 掩盖 actuator 因果性缺失。**后续 §7.6
   的转向没有绕过这条**——它先用 C1/C2 直接补齐了 actuator 因果性（证明"可扳"且条件性有独立
   因果价值），再在此基础上启动 S3。任何 runtime owner、snapshot 或 wiring 变更仍需另开收敛包
   并先注册正式契约。
4. **Branch B faithful rewrite screen（执行中）**：P1 入口/bias 定罪与 S2
   FAIL 已同时满足触发条件。新 claim 的 prereg `c247e82e…` 固定 layer20
   full-width896 → learned 16-d projection、rank-8 no-bias
   `A·diag(tanh(Cz))·Bᵀ·e`、zero-z strict no-op、oracle boundary readout-only；
   3 alpha × 2 seed × 40 updates。screen PASS 也只准入独立预注册的权威扫，
   不改写 Stage-3 `kill-eta`、不安装 artifact、不改 production wiring。
5. Stage 4 不启动；Gate 1 / 2 / 3 封存件只在相关算法变量或证据契约改变时
   按新预注册复验，普通重构不触发昂贵全扫。

### 7.6 转向：读残差 + 有界条件 steering + Internal RL 学"何时扳"（三层闭环，S3-E admission PASS）

S2 的 additive no-bias steering FAIL 是学界已充分刻画的"可读却不可扳"失败模式（文献归因与修法见
`research/steering-2026-08/`，9 篇 2024–2026 steering/表征干预主线深读）。据此把 operationalization 从
"拿 probe 权重当方向、饱和位置单点静态加一把"换成**学习式条件干预**，逐级只读证据（全程
`substrate_trainable=0`、reader/executor 冻结、no free bias、zero-code strict no-op、SHADOW、
`production_promotion_authorized=false`，不改写任何封存 verdict）：

1. **P2c·C1 冲突映射仪器 = VALID**：目标剥离路口仪器，(view,subgoal) 残余歧义 0、基底 goal-stripped
   NLL 2.81 vs revealed 0.22 ⇒ 2.60 NLL 可 steer 余量、因果归属 subgoal（`06_*`）。
2. **P2c·C2 条件学习式写入 = PASS**：rank-8 乘性写入按 subgoal 条件化把 heldout NLL 从 2.81 关到
   **0.027**，等预算 unconditional 只到 1.36、random-condition 反伤 7.38 ⇒ **"扳得动"且条件性有独立
   因果价值**（`07_*`，owner `eta_conditional_steering_screen.py`）。
3. **P2c·S3 前置 = PASS**：把 condition 从 oracle 换成**在线非 oracle sensor**——在携带目标的上下文
   残差上 refit 冻结线性 reader 把 subgoal 读到 heldout **1.000**，驱动 C2 执行器得 `conditional-online`
   NLL 0.023 = 完全等于 oracle ⇒ **"读得到"**（`08_*`，owner `eta_read_steer_prereq.py`）。
4. **S3-A 门控余量审计 = PASS**：用诚实的过期 belief（记忆滞后）制造余量，staleness 完全可检测
   （`10_*`）。
5. **S3 本体（学"何时扳" Internal RL）**：冻结 sensor+executor，唯一在线更新的门控策略只观测 PE 代理、
   只拿每-episode 终局稀疏信用、从不给每步标签，自写 minibatch REINFORCE。**S3-D** 实质证明可学
   （4/5 seed），1 seed 探索塌缩 ⇒ 预注册 worst-seed 门 literal FAIL（历史保留）。**S3-E**（不改判据的
   multi-restart 训练侧选择稳健化，restart 数/选择规则跑前冻结进 prereg `e46b5890…`）救回塌缩 seed →
   **5/5 全过、admission PASS**：seed 平均 pe-gated 0.709（< oracle 1.09），worst-seed gain-vs-always-on
   CI 下界 +0.497，selectivity 0.494（`11_*`、`artifacts/eta_s3e_when_to_steer_rl_restart_20260805/`，
   owner `eta_when_to_steer_rl.py`）。

**结论**：三层闭环成立——**读得到 + 扳得动 + 学会何时扳**。这证明"给定稀而准的结局信用，门控策略能在
小样本内学会择时"；作用范围是代理迷宫上的 operationalization，**不复活也不改写 Stage-3 `kill-eta`**
（后者是 additive/free-bias 折叠入口那一族操作化的永久摘除），且尚未授权 production——到 companion 的
迁移需先解决"该不该扳向关系轨"缺免费客观标签、须靠情感专家/长程关系结局提供稀而准信用的问题。

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

## 10. RSI Forge（开发环自改进评测）

**Spec**：[`specs/rsi-forge.md`](./specs/rsi-forge.md)；包 README [`forge/README.md`](../forge/README.md)
**Owner**：`volvence_forge`（仓库根旁独立包 `forge/`；**不是** `vz-*` / `lifeform-*` wheel，不注册 runtime slot）
**依据**：Lilian Weng *Harness Engineering for Self-Improvement*（本地归档 `docs/external/lilian-weng-harness-engineering-2026-07-04.*`）+ R8/R10/R12/R15

### 10.1 目的

把「失败发生 → 找到机制 → 有界改动 → 验证 → 沉淀 → 下一轮证伪」做成可审计的开发环 RSI，**不**让生成提案的系统修改 evaluator、权限边界、runtime owner 或自身优化器。

它首先是速度与经济杠杆（缩短 harness 修复周期），**不能**把 harness 收益冒充 NL/ETA 机制收益，也不能绕开 `ModificationGate` 改产品 runtime。优化对象阶梯：

| 阶 | 对象 | 当前 | 晋升要求 |
|---|---|---|---|
| L1 | `.cursor/rules/*.mdc` | 开放、append-only | 人审 + held-in/out |
| L2 | `forge/prompts/**` | 开放、append-only | 人审 + schema/回归 |
| L3 | 工作流定义 | 未开放 | 独立 convergence packet |
| L4–L5 | Forge 自身 code / optimizer | **循环外** | 人工工程或独立战役 |
| L6a | companion playbook overlay（单文件） | 候选面开放；live 默认 **DISABLED** | owner validator + frozen suite + OFFLINE ALLOW + 人审；wiring 另行部署 |
| L6b | 模型参数 | 只开 DISABLED rare-heavy **请求** | substrate train/evaluate + cognition gate + loop-external READY；publish 仍独立 |

闭环：

```text
公开轨迹/证据 → 三层失败记录 → 语义聚类 → 有界 proposal
  → 循环外 validate →（runtime: OFFLINE gate）→ 人审 apply → ledger 预测
  → 下一轮 mine 发布 fulfilled / refuted / pending / inconclusive
```

### 10.2 怎么运行

Forge **不进根 workspace**，需单独安装：

```bash
python -m pip install -e 'forge[dev]'

# 1) 挖失败模式（真实 LLM 需 FORGE_LLM_API_KEY + FORGE_LLM_MODEL；无凭据时用 replay 只做契约演练）
forge mine
# 只读 applied 之后的新证据，并可挂 Companion Bench
forge mine --bench-root artifacts --evidence-since-ledger

# 2) 生成有界候选（单文件 append-only；可 --candidates-per-pattern 3）
forge propose artifacts/forge_mine_<timestamp>/failure_patterns.jsonl --candidates-per-pattern 3

# 3) 验证（不改目标文件）；任一 BLOCK → 总状态 BLOCK
forge validate artifacts/forge_propose_<timestamp>/proposals/<proposal_id>

# 4) runtime component（当前仅 companion overlay）必须先走循环外 OFFLINE 裁决
python scripts/forge_gate_adjudicator.py \
  artifacts/forge_propose_<timestamp>/proposals/<proposal_id>

# 5) population 上 Pareto 选择；无合格候选 → STOP / 非零退出
forge select artifacts/forge_propose_<timestamp>/proposals

# 6) 人审后落盘或拒绝（缺 PASS validation 或 named reviewer → fail-closed）
forge apply artifacts/forge_propose_<timestamp>/proposals/<proposal_id> \
  --validation-report artifacts/forge_propose_<timestamp>/proposals/<proposal_id>/validation.json \
  --human-approved-by '<reviewer>'
forge apply <proposal_dir> --reject --reason '<reason>' --human-approved-by '<reviewer>'

# 7) 第五阶段：只规划 DISABLED rare-heavy 请求（不训练、不发布）
forge plan-rare-heavy \
  --model-id '<model-id>' --model-weights-sha256 '<64-hex>' \
  --common-adapter-version '<version>' \
  --traces '<traces.jsonl>' --control-basis '<control-basis.json>' \
  --held-out '<held-out.jsonl>' --hook-layers '10,11,12'

# 训练/held-out/cognition gate 完成后，只读绑定裁决 READY/STOP（绝不 publish）
python scripts/forge_common_adapter_adjudicator.py \
  --request '<request.json>' --candidate '<common-adapter-candidate.json>' \
  --evaluation-report '<evaluation.json>' --gate-record '<gate.json>' \
  --held-out '<held-out.jsonl>' --output '<artifacts/.../verdict.json>'
```

契约回归（战役一退出门槛的一部分）：

```bash
ruff check forge/ tests/contracts/test_forge_boundaries.py
pytest forge/tests tests/contracts/test_forge_boundaries.py
```

### 10.3 怎么评价

**不是** F1–F6 family score，也不是 thesis gate；正式评价面是：

1. **结构 / 边界门（validate）**：schema 一致 → 单路径白名单且未命中保护面 → preimage hash → `git apply --check` → phase-1 无删除行 → 循环外 relevance judge 全肯定 →（runtime）临时文件 owner/schema + 冻结 suite 对照 → component held-in/out + 冻结静态命令 → Forge held-in tests → boundary held-out tests。检查不短路；任一 BLOCK → 总 BLOCK。
2. **Runtime OFFLINE 折算（冻结）**：`validation_delta = candidate_pass_rate - baseline_pass_rate`，最低增益 **0.05**；`capacity_cost=0.1`（文件级）；`contract_integrity` / `rollback_resilience` 仅当未触冻结 suite/阈值且 reverse patch 逐字节还原为 1.0。apply 必须 **gate ALLOW + 输入哈希一致 + named human approval** 三者同时成立。
3. **Optimizer / STOP**：`forge select` 在 `validation_delta ↑ / capacity_cost ↓ / added_lines ↓ / risk_count ↓` 上取确定性 Pareto；空 population 或无 eligible → 必须 `STOP`，不得强选。
4. **纵向证伪（ledger 预测）**：apply 冻结 `pattern_occurrence_count` baseline / 方向 / expected delta；下一轮 `mine` 只能报 `fulfilled / refuted / pending`；证据未显式晚于 applied → 必须 `inconclusive`，不得用历史样本污染。
5. **Rare-heavy**：`READY` 只表示请求 + candidate + held-out + OFFLINE ALLOW **绑定完整**；不等于 publish，也不创建 `CommonAdapterBundle`。

硬边界：保护面优先于白名单；Forge 顶层不得 import `volvence_zero.*` / `lifeform_*`；业务 wheel 不得 import `volvence_forge`；bench 与 transcript provenance **分 lane**，未映射到 OFFLINE-gated runtime owner 的 bench failure 必须 `out-of-surface`。

### 10.4 当前结果

| 项 | 状态 |
|---|---|
| Spec / 契约 | phase 3–5 contracts landed（2026-08-01）；`editable_surface.yaml` 冻结写面 |
| 战役一 e2e | `artifacts/forge_mine_rsi_e2e_20260801T032000Z/`：93 transcript + 1 verdict + 29 plan → **2** in-surface pattern；`artifacts/forge_propose_rsi_e2e_20260801T033000Z/`：**2** bundle |
| Ledger | `forge/ledger.jsonl`：**2** 条 `applied`（`forge/prompts/failure_mining.system.md`、`.cursor/rules/cursor-convergence-workflow.mdc`，reviewer=`mengfu`） |
| 预测兑现 | 该 mine 跑在 apply **之前**，`prediction_checks` 为空；**尚未**有一轮 `--evidence-since-ledger` 的纵向 fulfilled/refuted 报告 |
| Companion overlay | 生产 asset 为空；builder 默认 **DISABLED**；服务边界拒绝 ACTIVE；本阶段**无** runtime apply / ACTIVE 部署 |
| Rare-heavy | 请求/裁决契约已落地；**无**冻结模型 snapshot / GPU train / 新 held-out ALLOW → **无**新 CommonAdapterBundle |
| Live mine/propose | 依赖显式 `FORGE_LLM_*`；无凭据时只有 replay 契约演练，**不得**写成真实模型晋级证据 |

主张边界：结构 PASS ≠ 真实开发效率已提升；harness 编辑收益 ≠ substrate/ETA 机制收益；`READY` ≠ publish。

### 10.5 下一步

1. **纵向兑现**：在已 apply 的两条之后跑 `forge mine --evidence-since-ledger`（可选 `--bench-root artifacts`），发布正式 `fulfilled / refuted / pending / inconclusive`；这是最早的纵向评测面。
2. **Task-level held-out**：phase 1 尚无大规模任务级 held-out benchmark；扩写面前先建独立 split，避免只靠结构 PASS。
3. **Overlay 路径**：若要对 companion playbook 提案，走 validate → `forge_gate_adjudicator.py` → 人审 apply；`DISABLED→SHADOW→ACTIVE` 是**另一**人工部署决定，不由候选资产自授权。
4. **Rare-heavy**：凑齐冻结 weights SHA、traces、control basis、held-out 与 cognition ALLOW 后再 `plan-rare-heavy` + train + `forge_common_adapter_adjudicator.py`；`READY` 后仍走独立 substrate publish。
5. **禁止**：让 Forge 改自己的 validator/权限/LLM 配置；把 STOP 磨成 SELECT；跨 lane 把 bench 失败映射到 rules/prompts；用 judge 分数回灌 PE/credit。

回滚：删 `forge/` + 本 spec 入口即移除 Forge；已 apply 未提交用 manifesto 中的 `git apply --reverse`；已提交用独立 revert；overlay 优先 `wiring=DISABLED`。ledger **保留**负结果，不删。

---

## 11. 跨评测纪律（务必遵守）

- **MPS 独占**：七日 / MSC / ETA rate-distortion / gate suite 共享同一把锁 `artifacts/.companion-evidence-mps.lock`，`PYTORCH_ENABLE_MPS_FALLBACK=0`，**不得并发**；控制面外手工启动的旧进程不受锁保护，须另行确认结束。Digital Ant ecology 走 **CPU float64**，不占该锁，但仍勿与其它会改同一源码树/journal 的 formal 包并行污染。RSI Forge 是离线开发环工具，默认不占 MPS；若其 rare-heavy 训练调用 substrate train，则与其它 GPU/MPS evidence **串行**并另开隔离 progress。
- **冻结在先**：长任务前冻结执行根/源码树 + 预注册 SHA；读到任何 outcome 之前不改阈值/seed/判据；封包后源码漂移一律拒绝（用隔离 commit 快照复核，只重做 validation/export）。Ecology 正式跑另要求**隔离源码快照 + 新空 progress 目录**。Forge 的写面由 `editable_surface.yaml` 冻结，提案不能改白名单/保护面/验证命令。
- **停跑与 kill 是合格终态**：不换 seed、不降阈值、不挑 metric 把结论磨成通过；停跑目录禁止原样 resume，重开必须新 prereg。Ecology 的 `not-authorized` 下游不得写成已执行的新负证据。Forge 的 `STOP` / `BLOCK` / 拒绝 ledger 同为合格负证据。
- **证据等级不互相冒充**：`mechanism-supported`（能跑/可回滚/可审计）≠ `causal-supported`（冻结 matched control 下达门的因果差）≠ `longitudinal-supported`（跨 session 持续）≠ `thesis-retained`。可回滚是安全证据，不是收益证据。Forge 结构 PASS ≠ harness 效率因果；harness 收益 ≠ NL/ETA 机制收益。
- **evaluation 只读**：所有金标评测不回灌 PE/credit，不反向训练 probe，不静默成为第二 owner。Forge 的 judge / suite / ledger 同属只读 gate evidence（R12）。
- **production 晋升**：#92 终局 `production_live_promotion_authorized=false`；本文任何门通过都不自动翻转 `WiringLevel`；晋升走 SHADOW→单组件 canary→可回滚切换，并先登记 `docs/DATA_CONTRACT.md`。Forge overlay ACTIVE 与 rare-heavy publish 均需独立人工部署，绝不由候选自授权。

## 12. 权威参考

- 框架口径：[`EVALUATION_SYSTEM.md`](./EVALUATION_SYSTEM.md)、[`specs/evaluation.md`](./specs/evaluation.md)、[`specs/evaluation-cascade.md`](./specs/evaluation-cascade.md)
- 终局判词与 Gate 台账：[`thesis prove.md`](./thesis%20prove.md)、[`specs/evidence_program.md`](./specs/evidence_program.md)
- 当前事实与剩余代码：[`currentstatus.md`](./currentstatus.md)
- 七日 × MSC 静态缺陷/仪器清单：[`moving forward/七日msctodo.md`](./moving%20forward/七日msctodo.md)
- 各评测 spec：[`specs/seven-day-companion-evidence.md`](./specs/seven-day-companion-evidence.md)、[`specs/companion-bench.md`](./specs/companion-bench.md)、[`specs/state-kv-identification-evidence.md`](./specs/state-kv-identification-evidence.md)、[`specs/character-prefix-package.md`](./specs/character-prefix-package.md)、[`specs/eta-llm-transfer-evidence.md`](./specs/eta-llm-transfer-evidence.md)、[`specs/learned-vs-heuristic-coverage.md`](./specs/learned-vs-heuristic-coverage.md)、[`specs/digital-ant-embodiment.md`](./specs/digital-ant-embodiment.md)、[`specs/rsi-forge.md`](./specs/rsi-forge.md)
