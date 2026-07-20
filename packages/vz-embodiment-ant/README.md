# vz-embodiment-ant — 数字蚂蚁 embodiment

把 VolvenceZero 分层认知内核（`vz-temporal` / `vz-memory` / `vz-cognition`）**原样复用**，
接到一个**完全不涉及语言**的 2D 感觉运动 substrate 上，构成一个昆虫尺度的觅食数字生命体。

这是一个研究/验证测试床，不是产品线。它的唯一目的：在没有 LLM、没有 token 的情况下，
独立检验 R2（冻结基底 + 有界控制器）、R3-R4（潜控制器空间的时间抽象）、R5-R6（连续记忆谱）、
R-PE（预测误差一级信号）、SSOT（快照隔离）这几条分层原理是否 **独立于语言** 而成立。

## 边界（SSOT）

- 本包是一个独立 owner。它只依赖 `vz-contracts`（快照契约）、`vz-substrate`（`SubstrateAdapter`
  ABC 与快照类型）、`vz-runtime`（`AgentSessionRunner` / `Brain` 编排 facade）。
- 它 **不** import `vz-temporal` / `vz-memory` / `vz-cognition` 的内部实现 —— 只经 facade
  注入 `substrate_adapter_factory`，并从 `AgentTurnResult.active_snapshots` 读取 `temporal_abstraction`
  快照里的 `controller_state.code`（即 `z_t`）来驱动电机。该边界由
  `tests/test_import_boundaries.py` 强制。
- LLM substrate 包（`vz-substrate`）因此不被非 LLM embodiment 污染。

## 两个冻结向量函数（substrate 的字面含义）

- `sense_encode`（`substrate/sense_encode.py`）：环境观测 → 固定维感知向量。纯 numpy，无可学习参数。
- `motor_decode`（`substrate/motor_decode.py`）：controller 输出的 `z_t` → 物理转向/前进指令，
  并更新内部朝向估计 ĥ（路径积分核心）。纯 numpy，无可学习参数。

会学习的部分全部在中间的 controller（`z_t` / `β_t`、SSL、Internal RL、CMS），由内核承担。

## 目录

- `env/` — 2D 世界：环境 owner `ant_world.py`、三物体 `world_objects.py`、信息素场
  `pheromone_field.py`（Phase 1）。
- `substrate/` — 两个冻结函数 + `AntSubstrateAdapter` + `AntActuator`。
- `runtime/` — `AntSession`：复用 `AgentSessionRunner` 的每-tick 闭环。
- `controllers/` — `FixedRuleAnt` 硬编码 FSM 基线（matched-control + 可视化对照用）。
- `proofs/` — Workstream E：蚂蚁语境 6 臂 matched-control。
- `evidence/` — Workstream F：复用 `evaluate_learned_active_candidate` 的 ant-active-evidence gate。
- `caste/` — Phase 2：rare-heavy 角色重编程离线循环。
- `viz/` — 轨迹回放 / 曲线叠图 / 双 substrate 演示素材生成。

## 分阶段

- **Phase 0** 单体导航：环形吸引子朝向 + 路径积分回巢 + 稀疏码路线记忆。对标 AntBot / Ardin 2016。
- **Phase 1** 群体 + 信息素快照总线：多写者只追加带衰减 slot，集体觅食收敛，个体间零直接调用。
- **Phase 2** rare-heavy 角色分工涌现：离线角色重编程通道，分工随环境压力系统性变化。

## 运行入口（脚本 → `research/ant/`）

从仓库根目录运行；结果写入 `research/ant/results/*.json`，图写入 `research/ant/figures/*.png`。

- `scripts/run_ant_phase0.py` — Phase 0 回巢精度 + 路线熟悉度。
- `scripts/run_ant_matched_control.py` — Workstream E：latent proofs + 蚂蚁 6 臂 matched-control。
- `scripts/run_ant_active_evidence.py` — Workstream F：ant-active-evidence gate（toy 规模诚实 BLOCK）。
- `scripts/run_ant_dual_substrate.py` — G1：同代码双 substrate side-by-side。
- `scripts/run_ant_colony.py` — Phase 1：信息素总线群体觅食。
- `scripts/run_ant_caste.py` — Phase 2：离线角色重编程（`allow_offline=True`）。
- `scripts/run_ant_demos.py` — G2/G3/G4：扰动对比 / 生物学叠图 / 安全反射一票否决。
- `scripts/run_ant_theater.py` — 觅食剧场：并排「启发式 FSM 殖民地」vs「数字生命殖民地」的
  一群蚂蚁实时行为动画（自包含 HTML + Canvas，零依赖，浏览器直接打开）；中途食物搬迁。
  诚实提示：在玩具尺度下 kernel 觅食投递数不敌手写 FSM（正式 matched-control 亦如此），
  此剧场用于展示行为，不代表 kernel 在觅食效率上胜出。输出
  `research/ant/figures/digital_ant_theater.html`。
- `scripts/run_ant_homing_theater.py` — 回巢剧场（推荐）：展示系统被 AntBot 标度验证的**真实强项——
  路径积分导航**。一群蚂蚁随机外出后沿内部估计方向回巢，橙色虚线箭头画出它「以为家在哪」；
  左臂完整路径积分（含天空罗盘）信念始终指向真巢、精准回家，右臂删掉罗盘的纯死走信念漂移、迷路。
  可选第三面板用**真内核**跑路线熟悉度：固定路线反复走，可下降新奇度（认知型 PE）随曝光下降，
  记忆关闭对照不下降。输出 `research/ant/figures/digital_ant_homing_theater.html`。
- `scripts/train_ant_ecology.py` — 用真实 `AntSession` / `KernelColonyRunner` 训练黄油→木棍→火柴→
  组合场景，写出 opaque checkpoint、held-out gate report 和 manifest。只有 PASS artifact
  可由 realtime app 加载；BLOCK 会保留具体断点，不回退 FixedRule。

推荐从统一入口运行：

```bash
# 快速现场演示：打开本地 Dashboard，并导出 HTML/GIF 或 MP4/JSON/manifest
python scripts/run_ant_pipeline.py --profile demo --dashboard

# 正式多 seed 证据；G1 需要显式提供本地 HF 模型
python scripts/run_ant_pipeline.py --profile formal --model-id /path/to/local/model

# 中断后恢复：验证 stage marker/manifest，并补跑 matched-control 缺失 seed
python scripts/run_ant_pipeline.py --profile formal --model-id /path/to/local/model --resume

# 串行可复现基线；formal 默认使用 min(5, CPU 数) 个 seed worker
python scripts/run_ant_pipeline.py --profile formal --workers 1 --model-id /path/to/local/model
```

`demo` 与 `formal` 都会诚实保留 `PASS/BLOCK`；脚本不会自动修改生产
`FinalRolloutConfig`。正式 learned 对照采用训练地图 → owner checkpoint →
held-out 地图，所有消融臂从同一初始 checkpoint 分叉。实时 Dashboard 和录制
只消费不可变 `AntStepRecord` replay，不读取内核 owner 私有状态。

matched-control 只按 seed 使用 `spawn` 多进程；同 seed 的 arms 仍共享同一初始
checkpoint 并保持顺序执行。每个完整 seed report 原子写入
`research/ant/results/.partials/matched_control/<fingerprint>/`，不保存 owner 私有状态。
最终 manifest 是 artifact 的提交点，pipeline 只有在配置指纹一致且 manifest 校验通过时才跳过 stage；
因此 `BLOCK` 结果也能安全 resume。升级前已经启动的旧进程无法追溯生成 partial 或 stage marker。

可视化图需可选 extra：`pip install -e packages/vz-embodiment-ant[viz]`（缺失时仍产 JSON，仅跳过 PNG）。

详见 `docs/specs/digital-ant-embodiment.md` 与 `research/ant/04_digital_ant_feasibility.md`。
