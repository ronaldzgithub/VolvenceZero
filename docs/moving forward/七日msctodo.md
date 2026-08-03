两个代理的链路梳理都已拿到，与我自己的深读结果合并。以下是完整的静态检查报告。

（链路细节由 [七日链路梳理](d1dedd5d-c85e-4077-bbdf-9fdb42be0a76) 与 [MSC 链路梳理](172874ad-7371-455b-95e8-eca0432e3410) 两个探索代理复核过，结论一致。）

---

# 静态检查报告：七日 × 11 Gate 与 MSC N+1

## 先对齐范围

`#92` 台账的 gate 编号是 1–11，**Gate 3 不存在**（已并入 Gate 2）。七日控制面（`run_seven_day_companion_test_plan.py`）按 prereg schema 只路由三类包：Gate 1、Gate 4/5/6/7/9/10、Gate 8+11（同一 6 臂矩阵两个 contrast）。**Gate 2 走独立 residual/longitudinal 线，且已在 stop-loss，不进七日装置。** 所以"通过七日看能不能过 11 个 gate"的机械上限是 9 个编号门；如果预期是 11 门全覆盖，这是第一个要明确的缺口。

---

## 一、七日路径：会在哪里出问题

### 🔴 S-1 系统性：所有七日 gate 的 continuity 判据沿用已被判"无刻度"的仪器

这是最要害的一条。`halt_record.json`（2026-08-02 停跑）已认定七项 owner readout 中 **4 项无分辨力**：

- `user_correction_rate`、`remembered_item_usefulness` 被逐字节相同的每日 console 探针**构造性钉死在 0.5**；
- `callback_hit_rate` 测的是 owner 对自身 compact readout 的自洽性，**stateless 臂反而得 1.0**；
- `seven_day_trust_delta` 观测中恒为 0.0；
- 唯一直接观测 SUT 行为的 `fsm_probe_pass_rate` 全 null。

而 Gate 1 和 Gate 4/5/6/7/9/10 的 **causal 判定把 continuity composite gain ≥ 0.02 且 paired CI 下界 > 0 作为 co-primary 硬条件**（`companion_gate_suite_evidence.py:619-624`、`gate1_seven_day_evidence.py:402-405`），用的正是同一组 `SEVEN_DAY_METRICS`。7 项里 4 项跨臂恒定意味着 composite 增益被稀释约 4/7——3 个还有信号的指标需要平均移动 ~0.047 才能过 0.02 的门，而 `callback_hit_rate` 还可能反向。**照现在的仪器跑完 Gate 4–10 的全部矩阵，大概率全部卡死在 continuity co-primary 上，重演 8/11 的停跑。**

**修改方案**：这就是收窄计划 S3（状态 pending）——把主判据换成 R2 包 B 已冻结的 **N+1 substrate 表示预测误差**（arm-independent、非循环，`vz-substrate` 拥有），七项 owner readout 降为 secondary。在 S3 落地前不应启动任何 gate suite / Gate 1 的 formal 矩阵，否则是在用没有刻度的尺子花 MPS。Gate 8/11 的输出根已被 halt record 锁死禁止原样 resume，重开必须新 prereg + 新执行根。

### 🔴 S-2 Gate 9 机制门在当前遥测语义下可能必败

`_gate9_score`（`companion_gate_suite_evidence.py:335-341`）对**全部 35 turn** 收集 `ssl_m3_slow_gain` 进一个 set，要求恒为单值（treatment=1.0 / control=0.0），否则 `configured_slow_gain_x1000=-1`，机制门失败。但生产遥测来自：

```917:926:packages/vz-runtime/src/volvence_zero/agent/session_observation.py
        ssl_m3_slow_momentum_norm = 0.0
        ssl_m3_slow_gain = 0.0
        latest_ssl_report = self._joint_loop.latest_ssl_report
        if latest_ssl_report is not None:
            slow_signal = latest_ssl_report.m3_slow_momentum_signal
            if slow_signal:
                ssl_m3_slow_momentum_norm = sum(abs(value) for value in slow_signal) / len(slow_signal)
            optimizer_state = latest_ssl_report.encoder_optimizer_state
            if optimizer_state is not None:
                ssl_m3_slow_gain = optimizer_state.slow_gain
```

首次 joint cycle 执行之前 `latest_ssl_report is None`，**treatment 臂会先发布 gain=0.0 再变 1.0** → set 含两个值 → Gate 9 机制必败。单测（`test_companion_gate_suite_evidence.py:93-99`）给每个 turn 都注入了理想遥测，掩盖了这个分支。另外 off 臂机制要求 `positive_slow_signal_turns > 0`——`M3Optimizer` 的 slow momentum 确实不受 `slow_gain` 影响（gain 只在参数应用处相乘，`m3_optimizer.py:136`），理论可满足，但同样依赖 joint cycle 已经跑过至少一次。

**修改方案**：evaluator 侧只在 `joint_cycle_executed is True` 的 turn 收集 gain（与 loss 同一门控）；或 service 侧把配置值从 evidence profile 直接发布而不是从 `latest_ssl_report` 读。补两条单测：treatment 臂首 turn gain=0.0、off 臂 momentum=0.0。

### 🔴 S-3 Gate 7/9 的 early/late 覆盖断言在全矩阵跑完后才爆

`_gate7_score:323-324` / `_gate9_score:359-360` 要求 day≤2 和 day≥6 **都**存在 `joint_cycle_executed=True` 的 turn，否则 `ValueError`。该遥测由 `lifeform_service/app.py:965-974` 按当轮是否真的执行了 joint cycle 决定——如果 joint loop 的调度在头两天没触发 cycle，54 个 run（Gate 7）全部跑完后评估阶段才中止。fail-closed 没错，但代价是整套 MPS 算力。

**修改方案**：把"两个窗口各有至少一个 cycle"挪进 `--smoke-one-pair` 的硬检查项（smoke 现在只回传 mechanism_supported，机制门恰好覆盖不到这条），或在 preflight 输出 joint-cycle 调度预估。流程上强制 smoke 通过才允许 `--execute`。

### 🟡 S-4 CI 统计实现的两个问题

1. **n=1 退化**：Gate 8/11 的 `_paired_summary`（`seven_day_companion_evidence.py:550-551`）在单对样本时返回 `(mean, mean)`——只要 mean>0 就"CI 下界>0"。Gate 1/suite 的 `_ci95` 则在 n<2 返回 `None`（判失败）。同一预注册措辞"paired 95% CI"，两种语义。
2. **z vs t**：全线用 `1.96 * stdev / sqrt(n)`。n=18（6 scenario × 3 seed）时 t 临界值是 2.11，CI 系统性偏窄约 8%，**判"过"偏松**，这在正式证据里是方向性错误。

**修改方案**：统一 n<2 → `None`；换 t 临界值（可按 n 硬编码 t 表进 evaluator，并把公式写进 prereg 的 `confidence.method`），或至少把"正态近似"如实写进预注册措辞。

### 🟡 S-5 gate suite 评估器不校验矩阵规模 vs 预注册

`evaluate_companion_gate_suite:541-543` 的 `expected_keys` 从**传入的 cases 派生**，`matrix-complete` gate（612-613 行）恒真——不管传 1 个还是 18 个 case 都"complete"。Gate 1 有 `formal.run_count/pair_count` 交叉校验（`gate1_seven_day_evidence.py:325-328`），suite 没有。`--smoke-one-pair` 产出的 `gate{N}_evaluation.json` 与 formal 同 schema 同文件名，单看文件无法区分。审计器（`audit_seven_day_gate_suite_formal.py:60-64`）从 prereg 重建 case 矩阵兜住了终态，但"analysis_allowed 之前的中间文件"存在被误读窗口。

**修改方案**：在 suite evaluate 里加与 Gate 1 相同的 `formal_run.pair_count/run_count` 校验；smoke 模式写入不同的 schema_version 或文件名（如 `gate{N}_smoke_evaluation.json`）。

### 🟡 S-6 状态干预发生在旧进程终止之前

```206:210:packages/lifeform-evolution/src/lifeform_evolution/seven_day_process_host.py
    def restart_after_day(self, *, day_index: int) -> ProcessRestartEvidence:
        intervention = self._state_controller.archive_and_stage_after_day(
            day_index=day_index
        )
        process = self._host.restart()
```

active scope 被 rename 成 day-N archive、staged copy 计算 SHA 时，**旧 service 进程还活着**。若服务在 SIGTERM 优雅关闭时有任何落盘（persist-on-close），写入会落进已计digest 的 archive 或 staged copy——审计复算 digest 时整臂作废。当前依赖"end-scene 已完成全部 persist、SIGTERM 不写盘"这一未被断言的假设。

**修改方案**：拆开 `restart()`，顺序改为 stop → archive/stage → start。顺带修复两个恒真 attestation：`healthcheck_passed=True` 硬编码、`persistence_scope_unchanged` 比较的是同一个从不变化的变量（`seven_day_process_host.py:99-102`）——后者应对比服务端实际上报的 scope。

### 🟡 S-7 resume 只看文件存在，不验完整性

`_LocalFormalExecutor.execute`（`run_seven_day_companion_formal.py:216-224`）对已存在的 run JSON 只检查"是 dict"就直接采用。身份漂移由下游 `_arm_readout` 兜底，但 v2 的 runtime stack attestation 是 run 主体落盘**之后**追加写回（361-391 行）——若在两次写之间中断，resume 会吞下缺 attestation 的 run，直到 evaluate/audit 才失败。

**修改方案**：resume 路径至少校验 `schema_version` + arm/case 身份 + （v2 时）`runtime_stack_attestation` 存在，缺则删除重跑该 run。

### 🟢 S-8 次要问题（低危，建议顺手修）

| 位置 | 问题 | 建议 |
|---|---|---|
| `run_seven_day_companion_formal.py:154` | `arc_type` 由 scenario_id 前缀 `"F1-"` 推断，ID 前缀路由脆弱 | 从 scenario yaml 读 typed 字段 |
| `seven_day_state_control.py:19` / audit 脚本 / prereg | shuffled 源日序列 `[1,1,2,1,4,3]` 三处硬编码，无单一 import SSOT | 统一 import 自 `seven_day_state_control` |
| formal executor `231-234` | swapped donor = `(case_index+1) % n`，单 case 矩阵 donor=自身 | 加 `n >= 2` 断言 |
| `seven_day_companion_evidence.py:382` | `turn.get("event_tags", ())` 静默默认，与 fail-loud 规则和 `_gate4_score` 的严格要求不一致 | 统一为缺字段即抛 |
| `_gate5_score:254` / `_gate10_score:389` | early/late 空列表会抛 `StatisticsError` 而非契约化 `ValueError`（Gate 7/9 有显式检查） | 补显式检查 |
| runner 直调 | 端口固定 18765/18780、MPS 锁只在 test plan 层持有，直接调 formal runner 绕开锁 | 锁下沉到 runner 或文档强制经控制面 |
| 测试缺口 | CI n<2、safety 回归、各 mechanism 失败分支、telemetry 缺字段、profile SHA 漂移均无测试 | 按分支补齐 |

另外提示一个**判据设计层面**的注意点（不是 bug）：safety 门是"逐 case 最大回归 ≤ 0"（`companion_gate_suite_evidence.py:571-579,625`）——18 个 case 里任何一个的 boundary/wrong-user rate 高出对照哪怕 0.01（分母很小的 rate 噪声完全可能）就整门失败。这与 spec 措辞一致，是有意保守；但要意识到它对 causal-supported 的实际杀伤力很高，若预注册意图是"分布层面不劣化"，应在**新 prereg** 里改为 mean-level + 容差，而不是跑完后解释。

---

## 二、MSC N+1：会在哪里出问题

### 🔴 M-1 formal 路径根本不存在：现在任何跑法都只能是 pilot

三重硬阻断，缺一不可绕：

1. `run_msc_prediction_test_plan.py:311-313`：`formal` 阶段直接 `return FORMAL_BLOCKED_EXIT (3)`，blocker 为 `same-substrate-context-encoder / complete-volvence-runtime-arm / temporal-controller-capacity-ladder`；
2. `run_msc_prediction_research.py:1257`：`volvence_full_stack=False` 硬编码 → `formal_requirements` 必有一项 false → `thesis_exit` 恒 `INELIGIBLE_PILOT`；
3. 默认 CLI 子集（train 24 / val 12 / heldout 12）永远过不了 `complete-heldout-dyads == 501`。

**这不是 bug 而是 fail-closed 设计，但结论是：MSC N+1 想出 formal 判词，必须先按收窄计划完成 R3（同基底 steelman）、R4（真实 runtime 臂）、R5（temporal capacity ladder）三个 pending 包**。R4 的推荐路径计划里已写明：评估复用七日装置的 service 通路做 runtime collector，而不是为 MSC 另建一套。

### 🔴 M-2 scaling 判据的成本计量不对称，会产出虚高的 SCALING_ADVANTAGE

- **token**：volvence 臂只对 `latest_text` 计 token（`run_msc_prediction_research.py:714-718`），long_context 臂对整段渲染文本计数**且被 clamp 到 max_seq_length**（`token_cost:207-219`）。token_ratio ≤ 0.10 的门槛被分子分母双向压低，几乎是构造性可过。
- **latency**：volvence 用"全库 atomic embedding 摊销均值 + 状态合成"，long_context 用 batch encode 均摊——两边都不是 per-example 端到端时延，比值解释力弱。

**修改方案**：正式版把成本记账改为对称的"该臂为本样本实际消耗的 token/时延"（volvence 的记忆维护成本要摊进去，long_context 的截断 token 不该从分母里消失）；这必须写进新 prereg，不能事后调。

### 🔴 M-3 long_context 对照不是 steelman，且 preflight 证明的是另一件事

long_context 臂被 **MiniLM 编码器截断在 256/512 token**（left-truncation，`FrozenSentenceEncoder:167-174`），而 preflight（`run_msc_prediction_test_plan.py:170-209`）用 **Qwen tokenizer** 验证"全历史放得进 substrate 上下文"。preflight 通过 ≠ 实验里的 long_context 没截断。当前 verdict 的 "quality advantage vs long_context" 实际是"vs 被截断的 MiniLM 表示"——已知 blocker，但要防止 pilot 数字被口头引用成对照优势。

### 🟡 M-4 语料解析：speaker 按位置奇偶推断，空句丢弃可能造成错位

`msc_corpus._utterances:118-133`：speaker 由 `index % 2` 决定，空句被丢弃但保留原 index 奇偶——交替性被上游破坏时会**静默错绑目标人**；丢空句后可能出现连续同 speaker，而 `latest_text`（stateless/summary 臂的核心输入）假定"最新一句是对方"。整个 N+1 任务的"预测同一个人"语义都压在这个未断言的约定上。

**修改方案**：解析时显式断言相邻 utterance speaker 交替（或用原始数据的 id 字段），违约 fail loudly；补空句丢弃 + 交替性单测。

### 🟡 M-5 capacity ladder "flat 时仍用 argmax"

`adjudicate_capacity_ladder:545-554` flat 时 exit 是 `KEEP_MINIMAL_FORWARD_HEAD`，但 `best_forward_head_n_z` 仍是 argmax，且 runner（1152 行）**无条件用 argmax** 跑四臂——数值噪声下可能用 n_z=256 跑正式实验，与 exit 语义脱节。**修复**：flat → `chosen_n_z = 3`。

### 🟡 M-6 其余具体缺陷

| 位置 | 问题 | 建议 |
|---|---|---|
| `run_msc_prediction_research.py:1256` | `encoder_fingerprint` 参数传的是 **substrate 的 weights_sha256**，不是 MiniLM 的 `encoder.fingerprint`；formal requirement 只检非空 | 传 `encoder.fingerprint`，substrate 指纹单列一项 requirement |
| `forward_representation.py:150-151` | 零范数向量 cosine 静默记 0.0，head 塌缩不会报错 | 计数并在 verdict 里暴露 zero-norm 样本数，超阈值 fail |
| `run_msc_prediction_research.py:1118-1125` | `len(settlements)` 除法无空检查（preflight 有，research main 没有） | main 里加 examples 非空断言 |
| `prediction_research.py:422` | `longest = max(sessions)`，不强制 session 5——子集/坏数据时会在错误的"最长 session"上判 quality | formal 路径断言 `longest == 5` |
| checkpoint | 无跨进程锁，两个进程对同一 output `--resume` 会竞态 | output 目录加 flock |
| `_jsonable:54-56` | `hasattr(..., "__dataclass_fields__")` 触仓库 hasattr 禁令（工具函数，低危） | 改 `dataclasses.is_dataclass` |
| 统计 | 按 session 多次配对检验无多重比较校正；quality 只看最长 session + slope，风险可控但应写进 prereg | 预注册明确主检验只有 longest-session 一项 |
| 报告口径 | stateless / summary_retrieval 只做 matched 资格，**不进胜负判定**（只 volvence vs long_context），报告易被读成"四臂比武" | report.md 里写明 |

**Checkpoint 层（`msc_prediction_checkpoint.py`）本身是两条链里质量最高的部分**：配置指纹绑定、原子写、resume 全量 hash 复验、崩溃窗口有补登记语义，不会跳样本或双计，无需改动。

---

## 三、修改方案汇总（按收敛包排序）

**P0 —— 不做就不该花下一份 MPS 算力：**

1. **S3 仪器改造包**（owner：evaluation + prereg）：七日主判据换 N+1 substrate 表示预测误差，七项 readout 降 secondary；新 prereg + 新执行根。这是 Gate 8/11 重开和 Gate 1/4–10 continuity co-primary 可判读的共同前置。
2. **Gate 9 遥测语义修复包**（owner：`companion_gate_suite_evidence` + service telemetry）：gain 只在 cycle turn 收集或由 profile 发布；补 off 臂/首 turn 单测。
3. **CI 统一包**（owner：三个 evaluator）：n<2→None、t 临界值、`_paired_summary` 修复；prereg `confidence.method` 同步措辞。

**P1 —— fail-early，防整矩阵作废：**

4. suite evaluate 加 pair_count/run_count 校验 + smoke 产物区别化。
5. Gate 7/9 early/late joint-cycle 覆盖挪进 smoke 硬门，流程上 smoke 必须先于 execute。
6. process host 改 stop → archive/stage → start；修复两个恒真 attestation。
7. resume 完整性校验（schema + 身份 + v2 attestation 存在）。

**P2 —— MSC 正式化路线（顺序不可并包，按计划 R3→R4→R5）：**

8. 同基底 long-context steelman + 成本计量对称化（同一包，因为都动对照臂契约）。
9. 真实 runtime 臂（评估复用七日 service 通路做 collector）。
10. temporal capacity ladder；顺带修 encoder_fingerprint、flat→n_z=3、speaker 交替断言。

**验证说明**：本次为纯静态检查，未运行任何测试或 evidence run；所有行号引用基于当前工作区代码。上述 P0/P1 修改都涉及评估契约或快照 shape，落地时需同步 `docs/specs/seven-day-companion-evidence.md` 与对应 prereg schema，且已冻结的旧 preregistration 不做 retrofit——只能开新 prereg。