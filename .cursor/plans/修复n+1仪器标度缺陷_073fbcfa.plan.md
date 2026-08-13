---
name: 修复N+1仪器标度缺陷
overview: 主线提升方案的中枢（A0 的 N+1 表示 readout）存在可证伪的标度缺陷：55.4% 共模能量把 d=0.75 的可分辨性压成 d=0.32，导致 A1 封存了一个仪器失准产生的 null。本计划放弃续跑已死的 C3 formal，改为在 substrate owner 内部修 readout 契约、补判据分辨力预检门、并同步修正方案文档的判词范围。
todos:
  - id: disk
    content: 清理 ~/.cache/huggingface：先逐 repo 列出占用并核对 C3 只依赖本地 artifacts/eta_stage2_merged_v2_20260803，确认无关后删除，目标腾出 40G 以上
    status: completed
  - id: readout-v2
    content: 在 vz-substrate 的 forward_representation.py 新增 latest-token-selected-layer-centered-residual-l2.v2：逐层 L2 归一 + 减冻结参考均值，均值随 lineage 发布且只在 MSC train split 上拟合；v1 保留不动
    status: completed
  - id: contract-sync
    content: 同步 docs/DATA_CONTRACT.md 的 substrate_forward_representation slot 与 docs/specs/prediction-error-loop.md，登记 v2 的 owner/value_type/dependencies/wiring_level
    status: completed
  - id: discrimination-gate
    content: 新增独立只读的判据分辨力预检诊断（共模能量占比、same-vs-diff Cohen's d、1-NN 检索准确率、v1/v2 对比），严格放在 run_dialogue_steering_test_plan.py 的 SOURCE_FILES 之外
    status: completed
  - id: verify-v2
    content: 在已收集的 583 个 train 样本上跑 v1/v2 对比，确认 v2 的 Cohen's d 相对 v1 有实质提升，作为 v2 契约的落地证据
    status: completed
  - id: doc-a1-verdict
    content: 在主线提升方案 §9 补 A1 封存记录，判词限定为「v1 raw-cosine readout 下无净增益」，并修正 P4 完成定义第 2 条
    status: completed
  - id: doc-c3-scope
    content: 在方案 §4.3 补 C3 operationalization 边界（credit 实测表示对齐改善而非行为控制），§0 不变量追加第 7 条「新主判据必须先过分辨力预检」
    status: completed
isProject: false
---

# 修复 N+1 仪器标度缺陷，而非继续跑

## 1. 现状与判断

**运行态**：C3 formal（`dialogue_steering_c3_formal_fullcontext_20260806`）已死 31 小时，pid 9675 不存在，最后写入 8/11 13:46。context 收集 33/48（train 24/24 完成，validation 9/24）。两个 `flock` 建议锁随进程退出已自动释放，无需手动清理。源码树与 prereg 的 32 个哈希零漂移。死因高度指向磁盘：数据卷仅剩 5.9G（99% 满），swap 4614M/5120M。

**方案判断**：三工作流骨架、prereg 纪律、SHADOW→ACTIVE 晋升协议都成立。塌的是中枢假设——[docs/moving forward/主线提升方案_2026-08.md](docs/moving%20forward/主线提升方案_2026-08.md) §1.2 明写「A0→C1 是整个方案的中枢」，A1/C1/C3 全挂在 N+1 readout 上，而该 readout 有标度缺陷。

## 2. 证据（全部来自已收集的 583 个 train 样本，零额外 GPU）

- 共模能量占比 55.4%，任意样本对原始 cosine 均值 +0.535
- 同 dyad vs 跨 dyad 可分辨性（`swapped-user-state` 的代理问题）：
  - 原始 cosine：gap 0.053，Cohen's d = 0.315
  - 减均值：gap 0.133，d = 0.549
  - 减均值 + 去 top-1 PC：gap 0.108，d = 0.751
- 1-NN dyad 检索 24–27%，随机 4.2%（6 倍）→ 信息存在，标度压制
- 层间能量失衡：全局归一下 L20 块 norm 0.856（73% 能量），L11/12/13 仅 0.277/0.302/0.314
- C3 结构伪影：steering 在 L20，L11/12/13 在上游，steer 与 noop 之间 cos 恰为 1.000000，表观差异纯属重归一化

`sensitive_fraction` 只在 validation 上计算（[dialogue_steering_evidence.py](packages/vz-runtime/src/volvence_zero/agent/dialogue_steering_evidence.py) 第 761 行），train 侧诊断不触碰 heldout 判据。

## 3. 收敛包切分（一包一 owner 一契约）

### 包 1 · readout v2 契约（owner: vz-substrate）

唯一改动点是 [forward_representation.py](packages/vz-substrate/src/volvence_zero/substrate/forward_representation.py) 的 `_representation_from_capture`，当前对整条拼接向量做全局 L2 归一：

```python
flat = tuple(value for row in activations for value in row.activation)
norm = math.sqrt(sum(value * value for value in flat))
values = tuple(value / norm for value in flat)
```

新增 `latest-token-selected-layer-centered-residual-l2.v2`，保留 v1 不动：

- **逐层归一后再拼接**，消除 L20 独占 73% 能量的失衡
- **减去冻结参考均值**再归一
- 硬约束：参考均值必须在**冻结参考语料**（MSC train split）上拟合，随 model-bound artifact 发布并进 `SubstrateForwardRepresentationLineage`；绝不允许在评测数据上现算，否则是 heldout 泄漏
- 是否内置去 top-1 PC 作为可选 whitening 档，在包内用 d 值定夺

该文件 docstring 已明确「substrate owns both the model capture and the interpretation of its residual geometry」，改动落在 owner 内部，不破坏 SSOT。同步 [docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md) 的 `substrate_forward_representation` slot 与 [docs/specs/prediction-error-loop.md](docs/specs/prediction-error-loop.md)。

### 包 2 · 判据分辨力预检门（新诊断入口，不进 SOURCE_FILES）

方案 §2.1 换判据时漏掉了这一步，导致同一个坑踩两次。新增独立只读诊断，对任何拟作 formal 主判据的 readout 强制先算：共模能量占比、same-vs-diff cluster 的 Cohen's d、1-NN 检索准确率、以及 v1/v2 双跑对比。门槛在 prereg 冻结时定，不过门不许开 formal。

必须放在 [run_dialogue_steering_test_plan.py](scripts/run_dialogue_steering_test_plan.py) 的 `SOURCE_FILES`（第 90 行起，32 个文件）之外，否则会污染既有 prereg 哈希。

### 包 3 · 方案文档判词修正（纯文档，§9 变更纪律强制）

- 补 A1 封存记录：`passed=false`、4 个失败 gate、四项 gain（0.0058 / 5.93e-05 / 0.0074 / −0.0004）
- **判词必须限定范围**：「在 v1 raw-cosine readout 下无净增益」，而非无条件的「七日窗口无净增益」——这是本次发现的直接后果
- 修正 P4 完成定义第 2 条
- §4.3 补 C3 operationalization 边界：SHADOW 下 target 取 MSC 真实下一轮文本、与臂无关，所以 credit 实测的是**表示对齐改善**而非**行为控制**，claim 措辞需与之一致
- §0 不变量追加第 7 条：新主判据必须先过分辨力预检

### 包 4 · 环境（已获授权）

清理 `~/.cache/huggingface`（49G，hub 46G + xet 3.5G）。C3 用的是本地 `artifacts/eta_stage2_merged_v2_20260803`，不依赖 HF hub 缓存，但需先列出各 repo 占用逐项核对再删。目标腾出 40G 以上。

## 4. 明确不做

- **不续跑当前 C3 formal**：12h 采集 + ~2h target 发布，换一个已知失准仪器上的判词，且 prereg 内 `n_plus_one_primary_passed=False` 已使链条在 A1 断裂
- **不动 `max_length=32768`**：已核 [residual_backend.py](packages/vz-substrate/src/volvence_zero/substrate/residual_backend.py) 的 `_tokenize`（第 3466 行），该值仅用于截断检查、不预分配，不是每轮 42 秒的瓶颈
- **不动 `substrate_model_dtype=float32`**：8/6 变更记录的 non-finite 发现是硬约束
- **不手动清锁**：`flock` 随进程退出已释放
- **不改现有 32 个 SOURCE_FILES**：会作废 prereg 并使已采集的 33 个 dyad context 失效

## 5. 诚实边界

修标度会同时放大信号与噪声，但 Cohen's d 是无量纲的，0.315→0.751 是真实可分辨性提升，不是重新缩放的假象。即便如此，**v2 下 A1 重跑仍可能 FAIL**——那才是可以诚实封存的「七日窗口无净增益」。本计划只修复仪器，不预告判词，也不为了过门下调任何阈值。

现有 33 个 dyad 的 context 保留为诊断证据；其中 24 个 train dyad 可作为 v2 冻结参考均值的合法拟合语料（train 非 heldout）。
