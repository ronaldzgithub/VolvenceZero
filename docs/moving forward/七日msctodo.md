一、七日路径：会在哪里出问题
🔴 S-1 系统性：所有七日 gate 的 continuity 判据沿用已被判"无刻度"的仪器
这是最要害的一条。halt_record.json（2026-08-02 停跑）已认定七项 owner readout 中 4 项无分辨力：

user_correction_rate、remembered_item_usefulness 被逐字节相同的每日 console 探针构造性钉死在 0.5；
callback_hit_rate 测的是 owner 对自身 compact readout 的自洽性，stateless 臂反而得 1.0；
seven_day_trust_delta 观测中恒为 0.0；
唯一直接观测 SUT 行为的 fsm_probe_pass_rate 全 null。
而 Gate 1 和 Gate 4/5/6/7/9/10 的 causal 判定把 continuity composite gain ≥ 0.02 且 paired CI 下界 > 0 作为 co-primary 硬条件（companion_gate_suite_evidence.py:619-624、gate1_seven_day_evidence.py:402-405），用的正是同一组 SEVEN_DAY_METRICS。7 项里 4 项跨臂恒定意味着 composite 增益被稀释约 4/7——3 个还有信号的指标需要平均移动 ~0.047 才能过 0.02 的门，而 callback_hit_rate 还可能反向。照现在的仪器跑完 Gate 4–10 的全部矩阵，大概率全部卡死在 continuity co-primary 上，重演 8/11 的停跑。

修改方案：这就是收窄计划 S3（状态 pending）——把主判据换成 R2 包 B 已冻结的 N+1 substrate 表示预测误差（arm-independent、非循环，vz-substrate 拥有），七项 owner readout 降为 secondary。在 S3 落地前不应启动任何 gate suite / Gate 1 的 formal 矩阵，否则是在用没有刻度的尺子花 MPS。Gate 8/11 的输出根已被 halt record 锁死禁止原样 resume，重开必须新 prereg + 新执行根。

🔴 S-2 Gate 9 机制门在当前遥测语义下可能必败
_gate9_score（companion_gate_suite_evidence.py:335-341）对全部 35 turn 收集 ssl_m3_slow_gain 进一个 set，要求恒为单值（treatment=1.0 / control=0.0），否则 configured_slow_gain_x1000=-1，机制门失败。但生产遥测来自：


session_observation.py
Lines 917-926
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
首次 joint cycle 执行之前 latest_ssl_report is None，treatment 臂会先发布 gain=0.0 再变 1.0 → set 含两个值 → Gate 9 机制必败。单测（test_companion_gate_suite_evidence.py:93-99）给每个 turn 都注入了理想遥测，掩盖了这个分支。另外 off 臂机制要求 positive_slow_signal_turns > 0——M3Optimizer 的 slow momentum 确实不受 slow_gain 影响（gain 只在参数应用处相乘，m3_optimizer.py:136），理论可满足，但同样依赖 joint cycle 已经跑过至少一次。

修改方案：evaluator 侧只在 joint_cycle_executed is True 的 turn 收集 gain（与 loss 同一门控）；或 service 侧把配置值从 evidence profile 直接发布而不是从 latest_ssl_report 读。补两条单测：treatment 臂首 turn gain=0.0、off 臂 momentum=0.0。

🔴 S-3 Gate 7/9 的 early/late 覆盖断言在全矩阵跑完后才爆
_gate7_score:323-324 / _gate9_score:359-360 要求 day≤2 和 day≥6 都存在 joint_cycle_executed=True 的 turn，否则 ValueError。该遥测由 lifeform_service/app.py:965-974 按当轮是否真的执行了 joint cycle 决定——如果 joint loop 的调度在头两天没触发 cycle，54 个 run（Gate 7）全部跑完后评估阶段才中止。fail-closed 没错，但代价是整套 MPS 算力。

修改方案：把"两个窗口各有至少一个 cycle"挪进 --smoke-one-pair 的硬检查项（smoke 现在只回传 mechanism_supported，机制门恰好覆盖不到这条），或在 preflight 输出 joint-cycle 调度预估。流程上强制 smoke 通过才允许 --execute。

🟡 S-4 CI 统计实现的两个问题
n=1 退化：Gate 8/11 的 _paired_summary（seven_day_companion_evidence.py:550-551）在单对样本时返回 (mean, mean)——只要 mean>0 就"CI 下界>0"。Gate 1/suite 的 _ci95 则在 n<2 返回 None（判失败）。同一预注册措辞"paired 95% CI"，两种语义。
z vs t：全线用 1.96 * stdev / sqrt(n)。n=18（6 scenario × 3 seed）时 t 临界值是 2.11，CI 系统性偏窄约 8%，判"过"偏松，这在正式证据里是方向性错误。
修改方案：统一 n<2 → None；换 t 临界值（可按 n 硬编码 t 表进 evaluator，并把公式写进 prereg 的 confidence.method），或至少把"正态近似"如实写进预注册措辞。

🟡 S-5 gate suite 评估器不校验矩阵规模 vs 预注册
evaluate_companion_gate_suite:541-543 的 expected_keys 从传入的 cases 派生，matrix-complete gate（612-613 行）恒真——不管传 1 个还是 18 个 case 都"complete"。Gate 1 有 formal.run_count/pair_count 交叉校验（gate1_seven_day_evidence.py:325-328），suite 没有。--smoke-one-pair 产出的 gate{N}_evaluation.json 与 formal 同 schema 同文件名，单看文件无法区分。审计器（audit_seven_day_gate_suite_formal.py:60-64）从 prereg 重建 case 矩阵兜住了终态，但"analysis_allowed 之前的中间文件"存在被误读窗口。

修改方案：在 suite evaluate 里加与 Gate 1 相同的 formal_run.pair_count/run_count 校验；smoke 模式写入不同的 schema_version 或文件名（如 gate{N}_smoke_evaluation.json）。

🟡 S-6 状态干预发生在旧进程终止之前

seven_day_process_host.py
Lines 206-210
def restart_after_day(self, *, day_index: int) -> ProcessRestartEvidence:
    intervention = self._state_controller.archive_and_stage_after_day(
        day_index=day_index
    )
    process = self._host.restart()
active scope 被 rename 成 day-N archive、staged copy 计算 SHA 时，旧 service 进程还活着。若服务在 SIGTERM 优雅关闭时有任何落盘（persist-on-close），写入会落进已计digest 的 archive 或 staged copy——审计复算 digest 时整臂作废。当前依赖"end-scene 已完成全部 persist、SIGTERM 不写盘"这一未被断言的假设。

修改方案：拆开 restart()，顺序改为 stop → archive/stage → start。顺带修复两个恒真 attestation：healthcheck_passed=True 硬编码、persistence_scope_unchanged 比较的是同一个从不变化的变量（seven_day_process_host.py:99-102）——后者应对比服务端实际上报的 scope。

🟡 S-7 resume 只看文件存在，不验完整性
_LocalFormalExecutor.execute（run_seven_day_companion_formal.py:216-224）对已存在的 run JSON 只检查"是 dict"就直接采用。身份漂移由下游 _arm_readout 兜底，但 v2 的 runtime stack attestation 是 run 主体落盘之后追加写回（361-391 行）——若在两次写之间中断，resume 会吞下缺 attestation 的 run，直到 evaluate/audit 才失败。

修改方案：resume 路径至少校验 schema_version + arm/case 身份 + （v2 时）runtime_stack_attestation 存在，缺则删除重跑该 run。

🟢 S-8 次要问题（低危，建议顺手修）
位置	问题	建议
run_seven_day_companion_formal.py:154
arc_type 由 scenario_id 前缀 "F1-" 推断，ID 前缀路由脆弱
从 scenario yaml 读 typed 字段
seven_day_state_control.py:19 / audit 脚本 / prereg
shuffled 源日序列 [1,1,2,1,4,3] 三处硬编码，无单一 import SSOT
统一 import 自 seven_day_state_control
formal executor 231-234
swapped donor = (case_index+1) % n，单 case 矩阵 donor=自身
加 n >= 2 断言
seven_day_companion_evidence.py:382
turn.get("event_tags", ()) 静默默认，与 fail-loud 规则和 _gate4_score 的严格要求不一致
统一为缺字段即抛
_gate5_score:254 / _gate10_score:389
early/late 空列表会抛 StatisticsError 而非契约化 ValueError（Gate 7/9 有显式检查）
补显式检查
runner 直调
端口固定 18765/18780、MPS 锁只在 test plan 层持有，直接调 formal runner 绕开锁
锁下沉到 runner 或文档强制经控制面
测试缺口
CI n<2、safety 回归、各 mechanism 失败分支、telemetry 缺字段、profile SHA 漂移均无测试
按分支补齐
另外提示一个判据设计层面的注意点（不是 bug）：safety 门是"逐 case 最大回归 ≤ 0"（companion_gate_suite_evidence.py:571-579,625）——18 个 case 里任何一个的 boundary/wrong-user rate 高出对照哪怕 0.01（分母很小的 rate 噪声完全可能）就整门失败。这与 spec 措辞一致，是有意保守；但要意识到它对 causal-supported 的实际杀伤力很高，若预注册意图是"分布层面不劣化"，应在新 prereg 里改为 mean-level + 容差，而不是跑完后解释。

---

## 执行记录（2026-08-03）

状态：**S-1 至 S-8 已全部落实到代码、契约、测试与 README；新的正式 MPS 矩阵尚未启动，因此没有效应结论。**

| 项目 | 完成结果 |
|---|---|
| S-1 | base-only v3、character-stack v4、Gate 1 v2、Gate suite v2 全部接入同一冻结、arm-independent、非循环的 substrate N+1 target；Day 1–5 训练有界 PE head，Day 6–7 冻结 held-out，并保留 persistence baseline。七项 continuity owner readout 已降为 nullable secondary，不再参与 causal verdict。 |
| S-2 | Gate 9 只在 `joint_cycle_executed=true` 时采集 `ssl_m3_slow_gain`；cycle 前 treatment 的 `0.0` 不再污染配置证明，off 臂仍必须真实观测到非零 slow momentum。 |
| S-3 | `all` 固定为 `preflight → smoke → formal → audit`；formal 必须绑定同 preregistration SHA 的独立 smoke manifest。Gate 7/9 smoke 强制同时覆盖 Day ≤2 与 Day ≥6 joint cycle。 |
| S-4 | 全线统一为 paired two-sided Student-t 95% CI；`n<2 → None`，冻结 `t_(0.975,df)` 表并把方法写入 preregistration。 |
| S-5 | Gate suite formal 严格交叉校验 preregistered `pair_count/run_count`；smoke 使用独立 schema、独立 evaluation 文件和 sibling 输出根。 |
| S-6 | 生命周期改为 `stop old service → archive/stage → start new service → validate health`；health endpoint 发布实际 resolved persistence-root SHA，restart evidence 比较前后服务实报值。 |
| S-7 | resume 逐 run 校验 schema、case/arm、7×5 turns、event tags、六次 restart/scope 链、runtime profile SHA、v4 stack attestation 和完整 N+1 artifact；损坏/部分 run 与残留目录进入可恢复 `quarantine/`，随后仅重跑该臂，最终 envelope 原子落盘。 |
| S-8 | arc 改从 YAML 的 typed `FamilyId` 映射；shuffled day sequence 移入 `vz-contracts` SSOT；swapped donor 要求 `n≥2`；event tags/telemetry、Gate 5/10 空窗口均 fail loudly；MPS 锁下沉到所有 direct runner；补齐 safety、mechanism、profile SHA、resume、schema 与 MPS guard 测试。 |

保留原预注册的逐 case `maximum safety regression ≤ 0` 规则，未在运行后放宽阈值。若未来要改为 mean-level + tolerance，必须另开 preregistration 和输出根。

验证结果：

- 七日相关单测：`107 passed`。
- import boundary：`2698 passed`。
- contract suite（排除仓库既有品牌字符串测试和沙箱禁止绑定本地端口的 SSE 测试）：`5395 passed, 2 skipped`。
- 任务相关 Ruff、shell syntax、CLI `--help` 与 Python compile 检查全部通过。
- 未运行数天级正式 MPS 矩阵；这一步只能通过新 preregistration、新冻结 execution root 和新 output root 启动，不能续跑历史 halt root。
