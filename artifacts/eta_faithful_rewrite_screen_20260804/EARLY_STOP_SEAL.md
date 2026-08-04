# B screen（忠实 ETA 重写 directional screen）早停封存

- 封存时间：2026-08-04
- prereg sha256：`c247e82e1eb6c9aaf32fcc5efc412caa1db622b43a3e4780fa38a3aee8a9d49b`
- prereg 路径：`artifacts/eta_faithful_rewrite_screen_prereg_20260804/preregistration.json`
- 早停时点：**3 / 6 cell**（primary α=0.30 已产出 seed-0；seed-1 及 α=3.0 未跑）
- 早停动作：停止本机 pid 21986（graceful SIGTERM），释放 stale MPS 锁 `artifacts/.companion-evidence-mps.lock`（持有者已死）。checkpoint 均已原子落盘、可 resume。

## 为什么早停：ETA 判定已被数学锁死为 FAIL

判定器 `assess_faithful_eta_screen`（packages/vz-runtime/src/volvence_zero/agent/eta_faithful_rewrite_screen.py L549-609）要求五条件**全过**才 admit。primary α=0.30 的 seed-0 一交卷即锁死两条：

1. **permuted-z-causality 永久失败**：门槛要求 primary 两 seed 的 `permuted_z_penalty` 全为正（`min_seed_positive_fraction=1.0`）。seed-0 的 `permuted_z_penalty = 0.0`（精确零）。无论 seed-1 如何，positive fraction ≤ 1/2 < 1.0 → 此条不可能通过。
2. **oracle-boundary-alignment 事实死亡**：门槛需 `oracle_boundary_f1_mean ≥ 0.2` 且 `boundary_probability_contrast_mean ≥ 0.02`。primary seed-0：F1 = 0.0、contrast ≈ −2.6e-5、`hard_switch_frequency = 0.0`。三个 cell 的 `hard_switch_frequency` 全为 0（第三次 never-switch collapse），系统性边界对齐不可能出现。

因此剩余 3 cell（约 4-5h 算力）无法改变 verdict。**本 screen 的 ETA 判定 = FAIL（锁死，非擦线）。** 不改写已封存的 Stage-3 `kill-eta` 与 S2 `causal-unsupported`。

## 正面资产：学习式写入的因果作用被证实

三个 cell 的 `zero_z_penalty ≈ 0.175`（门槛 0.02 的约 8.7 倍），且全部满足 `free_bias_present=false` + `zero_code_strict_noop=true`：

| cell | heldout_rate | heldout_distortion | zero_z_distortion | zero_z_penalty | permuted_z_penalty | oracle_F1 | boundary_contrast | hard_switch_freq |
|---|---|---|---|---|---|---|---|---|
| α0.03 s0 | 0.02676 | 0.00292 | 0.17841 | 0.17549 | 0.0 | 0.0 | −2.4e-7 | 0.0 |
| α0.03 s1 | 0.02339 | 0.00226 | 0.17841 | 0.17615 | 0.0 | 0.0 | 0.0139 | 0.0 |
| α0.30 s0 | 0.00955 | 0.00374 | 0.17841 | 0.17468 | 0.0 | 0.0 | −2.6e-5 | 0.0 |

含义：在**无 free bias、zero-code 严格 no-op** 前提下，学习式 rank-8 乘性写入 `A·diag(tanh(C·z_t))·Bᵀ·e_t` 把 heldout distortion 从 zero-z 的 0.178 打到 0.002–0.004（约 98% 降幅）。**这正是 S2 静态 probe-轴 steering 拿不到的因果作用**（S2 target-plus vs noop ≈ −0.00072，CI 跨 0），与 research/steering-2026-08 的排序结论（学习式/优化式 > 静态）一致。

## 失败机制（根因定性）

- `permuted_z_penalty = 0.0` 的成因：`hard_switch_frequency = 0` ⇒ z_t 从不切换 ⇒ 全程恒定 z ⇒ 对时间轴 cyclic-permute 是恒等变换 ⇒ penalty 精确为 0。
- 控制器找到一个**恒定低秩算子**即打满任务（train_distortion ~2e-4）。因为补课基底残差流本就线性携带子目标（Gate-2/S1 heldout 0.977/0.983 可读），恒定算子只需"路由"已存在的表征，**时间抽象没有剩余误差可吃、没有余量**。
- 结论层级：在子目标已线性表征的 LLM 基底上，ETA 要求的"切换 z_t 跨时携带子目标"是**冗余通道**——这比"ETA 又失败"更强，是转向"识别 + 有界执行器 + 学何时出手"的直接依据。

## checkpoint 清单（sha256）

- `checkpoints/points/alpha-0p03/seed-0.json` : `374c4b476dc3420398de24002176183b4f47e91fd365fc6b5dca5b6406d66406`
- `checkpoints/points/alpha-0p03/seed-1.json` : `0f20ce83e978891abd2f766f37738fcab0bf09f30836259e84043d0de5575989`
- `checkpoints/points/alpha-0p3/seed-0.json`  : `302cd14894c3c7317677b87e73f1e6ff4a80bb1bdd31a870f45201853f07d857`
