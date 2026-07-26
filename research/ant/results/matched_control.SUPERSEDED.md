# SUPERSEDED — `research/ant/results/matched_control.json`

**状态**：2026-07-27 起，该 artifact 及其 sidecar manifest、以及所有
`matched_control` stage 的 seed partials 全部失效，不得再作为 matched-control 证据引用。
文件本身按 `docs/specs`/plan 的 no-overwrite 规则**保留原样**（不删除、不覆盖），
本文件是它的失效说明。

失效的 artifact：

| 项 | 值 |
|---|---|
| artifact | `research/ant/results/matched_control.json` |
| sha256 | `4a5fd3523fa0471e36984c72bab02fdcfeeebb3a95375cb8cd85db88c68828cd` |
| manifest | `research/ant/results/matched_control.manifest.json` (`digital-ant-manifest.v2`) |
| `provenance.config_digest` | `4395f330f5bb160a97c6378244f756a6e8b63138452eb84217308395ce6b5399` |
| `provenance.git_sha` | `1c6c392f8561270ed6041017dbdfdd69f9c72c2e` (2026-07-20) |
| `provenance.working_tree_dirty` | `true` |
| `externally_retainable` | `false` |
| arms | `learned, pe_off, no_optimize, eta_off, fixed_rule, e2e_rl, random` |
| seeds / ticks / train_ticks | `0,1,2,3,4` / `60` / `200` |
| verdict | `BLOCK`（`validation_delta = 0.0`） |

## 1. 为什么失效

`docs/specs/digital-ant-embodiment.md` §3 要求 matched-control 各臂跑**同一具**冻结身体：
`heading_noise=0.01 / step_noise=0.01 / compass_gain=0.85 / compass_noise=0.007`，
并且天空罗盘读的是**动作之后**的绝对航向，因此各臂都必须先 `world.act(...)`
再 `navigator.update(..., true_heading=...)`。

该 artifact 产出时，两个臂并不满足这一条：

| 臂 | 当时的 navigator | 与 spec §3 的偏差 |
|---|---|---|
| `e2e_rl` | `heading_noise=0.0, step_noise=0.0, compass_gain=0.0`（默认），且先 `navigator.update(...)` 再 `world.act(...)` | 无本体噪声、无天空罗盘、罗盘时序反向 |
| `random` | `AntNavigator(step_size=..., seed=seed)`：`compass_gain=0.0`（默认），且先 `navigator.update(...)` 再 `world.act(...)` | 无天空罗盘、罗盘时序反向 |

也就是说，这两个臂当时是在**另一具身体**上被比较的：`e2e_rl` 拿到的是一个无本体噪声、
纯 dead-reckoning 的路径积分问题（对 PI 严格更容易），`random` 拿到的是有噪声但无罗盘的
dead-reckoning 身体。两者与 `learned / pe_off / no_optimize / eta_off / fixed_rule`
所跑的带罗盘身体不可并列。

修复：
- `e2e_rl`：2026-07-26 修好 `train()` / `evaluate()` / `step()` 三条路径；
- `random`：2026-07-27 修好（本包），`RandomAnt` 现在读取同一组冻结常数并在 `world.act`
  之后融合真值航向。

## 2. 失效范围（逐臂）

`research/ant/05_ecology_p0_p1_p2_plan.md` §5.4（P2-B）："任何代码或门槛变化都会使整批失效
并重新开始"。因此**整份 artifact 失效**，而不只是被改动的两个臂：

- **数值直接失效**：`e2e_rl`。该臂的策略输入经 `sense_encode` 消费 `navigator.state`
  （egocentric home 通道），因此 body sensor 与罗盘时序的修复会同时改变它的训练轨迹和
  evaluation 轨迹。这一行的所有数字（`food_delivered` / `mean_food_experienced` /
  `max_distance_from_nest` / `final_distance_from_nest` / `minimum_food_distance` /
  `held_out_success`）都已作废，且此前不得与其他臂并列。
- **语义失效但数值可复现**：`random`。这里必须诚实说明：`RandomAnt` 的动作只来自它自己的
  `np.random.default_rng(seed)`，从不读 `navigator.state`，而 `AntNavigator` 有独立 RNG，
  不影响世界。实测 seed 0–4、60 ticks 下修复前后的 `positions` 逐点相同，
  `food_delivered` / `food_pickups` 也相同，因此这一行的**行为数字**本身可复现。
  失效的是它的**声明**：这份 artifact 产出时 `random` 并没有跑 spec §3 要求的那具身体，
  其 docstring 所称的 "same frozen substrate" 当时不成立；任何基于该臂 navigator /
  路径积分的陈述都无效。
- **间接失效（批次规则）**：`learned`、`pe_off`、`no_optimize`、`eta_off`、`fixed_rule`
  的数字本身是在正确身体上产出的，但它们在这份 artifact 里的**意义**是"相对 baseline /
  floor 的对照"。baseline（`e2e_rl`）换了身体后，`aggregates`、
  `learned_beats_random_food`、`behavioral.verdict` 以及跨臂的任何比较都必须整批重跑。
  单独摘出某个 kernel 臂的绝对数字同样不被允许——plan 的规则是整批失效。
- 相应地，`research/ant/results/.partials/` 下 `matched_control` stage 的 seed partials
  也已失效：它们按 `ant_stage_fingerprint(stage="matched_control", config=...)` 命名，
  而下面第 3 节的 config 变更会让 fingerprint 改变，旧 partials 自然不会被复用。

## 3. resume gate 仍会复用这份失效 artifact（未修，需 owner 落地）

`scripts/run_ant_matched_control.py::_final_artifact_matches` 只比较
`payload["provenance"]["config_digest"]` 与 `stable_json_digest(config)`，而 `config` 只由
operator 旋钮构成：

```python
config = {
    "ticks": ..., "train_ticks": ..., "seeds": ..., "n_z": ...,
    "with_latent": ..., "include_e2e_rl": ...,
    "internal_rl_runtime_modulation_strength": ...,
    "internal_rl_runtime_exploration_strength": ...,
    "internal_rl_runtime_replay": "active",
}
```

这些值**都不会**因为某个臂的身体被改动而变化，所以 `--resume` 会认为上面那份 artifact
"配置一致"并直接返回它。这正是本次失效无法被自动检出的原因。

本包已在 owner 侧提供缺失的输入（`packages/vz-embodiment-ant/src/volvence_ant/proofs/matched_control.py`）：

- `arm_substrate_parameters()` —— 各臂族实际声明的四个冻结 body sensor 参数；
- `matched_control_arm_substrate_digest()` —— 其稳定摘要（复用
  `volvence_ant.evidence.provenance.stable_json_digest`）。

`scripts/run_ant_matched_control.py` **不属于本包 ownership**，仍需其 owner 落地一行改动：
在 `main()` 的 `config` dict 中加入

```python
    "arm_substrate_digest": matched_control_arm_substrate_digest(),
```

并把 `matched_control_arm_substrate_digest` 加进文件顶部已有的
`from volvence_ant.proofs import (...)` 导入块（该名字需要先由
`packages/vz-embodiment-ant/src/volvence_ant/proofs/__init__.py` 的 owner 重新导出；
在此之前可直接写
`from volvence_ant.proofs.matched_control import matched_control_arm_substrate_digest`）。

加入该 key 后：

1. `stable_json_digest(config)` 立即不同于
   `4395f330f5bb160a97c6378244f756a6e8b63138452eb84217308395ce6b5399`，
   `--resume` 会打印 `ignoring stale final artifact` 并重跑 —— 本次失效被自动兜住；
2. 今后任何一个臂的冻结 body sensor 变化都会继续移动该 digest。

**注意其边界**：该 digest 覆盖的是"各臂声明了哪组 body sensor 参数"，不覆盖
"各臂是否按 act→update 的顺序真的融合了罗盘"。顺序契约由
`packages/vz-embodiment-ant/tests/test_frozen_functions.py` 逐臂逐路径钉住
（`RandomAnt.step` / `E2ERLAnt.step` / `E2ERLAnt.evaluate` / `E2ERLAnt.train`），
不由 digest 承担。

## 4. 重跑要求

重跑前请确认：

1. 工作树干净（旧 artifact 是 `working_tree_dirty=true`、`externally_retainable=false`
   的脏树产物，本身就不可外部留存）；
2. 上面第 3 节的 resume-gate 一行改动已落地；
3. 新 artifact 使用**新文件名**（`write_ant_artifact_bundle` 拒绝就地覆盖已有 artifact，
   尤其是 `BLOCK` artifact），本文件与被它取代的 `matched_control.json` 一并保留。
