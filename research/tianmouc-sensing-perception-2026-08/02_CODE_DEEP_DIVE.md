# 02 · 官方代码库拆解：TMC-SSL-Representation

> 仓库：[`Tianmouc/TMC-SSL-Representation`](https://github.com/Tianmouc/TMC-SSL-Representation)（60 MB，4205 文件，2026-08-12 完整克隆核验）。
> 这是付费墙下本篇方法部分的**唯一完整公开证据**。本文所有行号/类名可直接对照仓库复核。
> 配套算法库：[`Tianmouc/tianmoucv`](https://github.com/Tianmouc/tianmoucv)（`pip install tianmoucv`，含模拟器与一键 demo）。

## 1. 仓库地图（与论文两阶段一一对应）

```text
reconstruction/     # Bottom-Up：IGFNet 自监督训练（train_unet_mem.sh 为主模型）
task_depth/         # Top-Down：单目深度（IGF-BSNet）
task_seg_v5/        # Top-Down：视频实例分割（YOLO-CVS / EOLO 基线）
dataset_raw/        # Tianmouc-R 读取器 + IR 构建（F_HDR 在这里）
dataset_depth/      # DAM-V2-L 伪标签生成管线
dataset_vis/        # XMem + Mask2Former 半自动标注管线
tmcsim / tmcsim_gpu # 第一代模拟器（现推荐 tianmoucv.sim）
```

## 2. Bottom-Up：IGFNet = `TianmoucRecon_mem`

主模型类 `TianmoucRecon_mem`（`reconstruction/model/reconstructor.py:568`），前向由四部分组成：

```python
# reconstructor.py:638 forward(F0, TFlow_0_1, SD0, SD1) 摘要
flow_low, flow_up = self.flowComp(SD0, SD1, TFlow_1_0)   # RAFT_Mo：SD/TD -> 光流
I_1_warp = self.backWarp(F0, flow_up)                     # F_O 路：warp 前一 RGB 帧
I_1_rec  = self.reconNet(cat([F0, TFlow_0_1, SD1]))       # F_I 路：UNet 直接积分
guidance = cat([SD1, intensity, Flow_1_0])                # 传感状态 guidance
if self.training:
    I_1_rec  = self.dataAug(I_1_rec)                      # 随机 mask + 噪声（关键！）
    I_1_warp = self.dataAug(I_1_warp)
I_t_p, M8, M4, M2, M1, affinity = self.syncComp(I_1_rec, I_1_warp, guidance)
emb_loss = self.syncComp.get_mem_std_constrain()
```

三个可独立冻结/解冻的组件（`chooseStage`，stage 0–4 控制训练课程）：

| 组件 | 实现 | 职责 |
|---|---|---|
| `flowComp` | `RAFT_Mo`（改造 RAFT，输入 SD0/SD1/TD） | F_O 路光流估计 |
| `reconNet` | `UNetRecon(3+2+2, 3)` | F_I 路直接重建 |
| `syncComp` | `MergeNet_mem`（三编码器 + `FuseNet_with_Mem`） | 门控融合 + 记忆 |

README 明示训练顺序：先独立训 RAFT（F_O）与 TinyUNet（F_I），再训融合（`train_unet_mem.sh`：4 卡 torchrun，50 epoch，lr 1e-3，`--fusion_gt` 开启）。

## 3. 门控融合的最小机制（`FuseLayer`，modules.py:182）

```python
E = convE1(z_e) * h + convE2(z_e)     # F_I 路特征 FiLM 调制主干
F = convF1(z_f) * h + convF2(z_f)     # F_O 路特征 FiLM 调制主干
M = sigmoid(conv3x3(h_prompt))        # 从 guidance/记忆增强特征生成逐像素门
out = M * E + (1.0 - M) * F           # 两条通路逐像素软选择
```

- 门 M 是**单通道、逐像素、条件于传感状态**（h_prompt 含 SD/强度/光流/记忆读出）；
- 每个 `FuseBlock`（4 个尺度级联，256→16 通道）内部做两次 FiLM+门控 + 残差；
- 调试模式直接打印/可视化 `recon rate = mean(M)` 与 `flow rate = 1−mean(M)`——**门控状态天然可读出、可解释**（注释 `M ↑ O↑ I↓`）。

## 4. 记忆模块（`FuseNet_with_Mem`，unet.py:104）

```python
self.mem_dict[str(i)] = nn.Parameter(torch.randn([1, dim_i, 128]), requires_grad=False)
# 五个尺度，每尺度 128 个状态槽
def direct_mem_readout(self, z_, i):
    z = downSample[i](z_)                 # 当前特征做 query
    affinity = get_affinity(LayerNorm(z), mem)   # 注意力：query x 记忆槽
    memory = readout(affinity, mem)              # 加权读出
    return upSample[i](memory), affinity
# forward 中：new_z = z_state + memory * mem_scale   <- 残差注入，可一键关断（close_mem）
```

- 读出机制是 XMem 式 key-value affinity（`xmem_utils`），**记忆读出以残差方式加回主干**，`mem_scale=0` 即完全关断（消融开关 `_tunoff_memory`）；
- `get_mem_constrain()`：约束记忆槽方差 ≈ 1（MSE 到 1），防坍缩/防漂移，作为 `embedding_loss` 进总损失；
- 注意：`requires_grad=False` + 冻结逻辑中 `'mem_dict' not in name` 的豁免——记忆槽的更新路径与常规参数分离管理（训练脚本控制），部署时冻结为"场景状态字典"。

## 5. 自监督损失组装（`utils/train_core.py`）

```python
if args.fusion_gt:                         # 论文主设置
    F*_GT = sample['F0_HDR'/'F1_HDR']      # 伪 GT = Poisson HDR 融合（F_HDR IR）
else:                                      # 消融：Ablation_NOHDRGT_
    F*_GT = sample['F0'/'F1']              # 退化为原始 RGB 当 GT

refineLoss = MS_SSIM_L1(F1t, F1_GT) + MS_SSIM_L1(F0t, F0_GT)   # 对称双向
lpips_vgg  = LPIPS(F1t, F1_GT) + LPIPS(F0t, F0_GT)
flowLoss   = 光流空间平滑项
warpLoss   = Charbonnier(Fwarp, GT)        # F_O 分支独立监督
recnLoss   = MS_SSIM_L1(Frec, GT)          # F_I 分支独立监督
total     += 5*embedding_loss + refineLoss + lpips_vgg + 0.1*flowLoss
```

伪 GT 的构建（`dataset_raw/tianmoucData.py:391`）：

```python
def HDRRecon(self, SD, F0):
    Ix, Iy = SD 的两个方向分量            # 传感器直接测得的空间梯度
    blend_hdr = laplacian_blending(-Ix, -Iy, srcimg=F0, iteration=20,
                                   mask_rgb=True, mask_th=32)   # tianmoucv 实现
```

要点：**整条监督链没有一个人工标签、没有一个外部模型打分**——监督来自"梯度域与强度域必须物理一致"这一数据内禀约束。

## 6. 退化免疫训练（`dataAug`，reconstructor.py:611）

```python
mask = (randn([1,1,16,32]) > -0.85)        # 粗粒度随机 patch 遮挡
I *= interpolate(mask, [320,640])          # 应用到 F_I 或 F_O 的输出上
I += randn(...) * noise_amp * mask         # 可选噪声
```

对**两条输入通路分别独立**做随机遮挡——门控层若依赖单一通路则损失必然上升，唯一稳定解是学会"按传感条件分配信任"。这是论文鲁棒性声称的机制根源（不是数据增广的运气，是结构性逼迫）。

## 7. Top-Down：表示复用与蒸馏

`task_depth/depth_model/basic_models.py`（IGF-BSNet）：

```python
# MergeNet_Encoder：与 bottom-up 完全同构的三编码器 + 融合（含 MemBank）
self.freeze_layer = [self.fusion, self.encoder1, self.encoder2, self.encoder3]
def freeze_encoder(self): ...    # 加载 bottom-up 预训练权重后整体冻结
# 之上接 BSNet 深度解码器，仅训解码器
```

- 深度伪标签：DAM-V2-L 跑在 e-VGT 上（Extended Data Fig. 5 证明零微调即尺度对齐）——**基础模型的知识经由标准表示蒸馏进传感器专属小模型**；
- VIS：`task_seg_v5/` 的 YOLO-CVS 修改 YOLO 的 HDR 分支与融合层（Extended Data Fig. 8 的精度/速度权衡）；标注管线 `dataset_vis/` 用 XMem + Mask2Former 在 e-VGT 上半自动标注。

## 8. 工程可信度评注

- 代码是真实科研代码：含全部基线对照（SwinIR/UFormer/CBMNet/E2VID/LFNet/SpyNet）、消融脚本（no-mem、NOHDRGT）、分布式训练、权重发布（Google Drive + 自建网盘）——**可复现性意图明确**，非展示性开源；
- 中文注释混杂（"魔改"等）说明是一线实验代码，未过度清洗；训练配置（batch 5 × 4 卡 × accum 4）与 129 GB 数据集规模一致；
- 与论文图题的交叉验证全部对上：Extended Data Fig. 2（三 IR 构建）、Fig. 3（两步自监督 + IGFNet 结构）、Fig. 8（VIS 网络变体）在代码中均有对应实现。
