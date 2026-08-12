# 来源与下载边界（2026-08-12）

## 主论文：link-only（付费墙）

| 项 | 状态 | 说明 |
|---|---|---|
| Nature Sensors 正式版 | ❌ 付费墙 | https://doi.org/10.1038/s44460-026-00095-3 ；`.pdf` 直链 302 回 HTML 登录页（已实测） |
| arXiv / bioRxiv 预印本 | ❌ 不存在 | 多轮关键词检索无果；该团队惯例不发预印本（前作 Nature 2024 同样无预印本） |
| SharedIt（rdcu.be）免费链接 | ❌ 未找到 | 一作厦大教职页与个人主页的"[文章链接]"均指向 doi.org 原始链接（已抓 HTML 核验 href） |
| 作者自存档 | ❌ 未找到 | 个人主页 lyh983012.github.io 仅列条目无 PDF |
| 公开部分 | ✅ 已归档 | 摘要 + 图题 + 扩展数据题注 + 参考文献 + 作者贡献 → `sources/nature-sensors-article-page.txt` |

**方法证据的替代来源**：官方代码库 `Tianmouc/TMC-SSL-Representation`（完整克隆核验，见 `02_CODE_DEEP_DIVE.md`）。模型结构、损失、训练方案、IR 构建、伪标签管线均可从代码逐行核验，覆盖论文方法部分的绝大多数声称；无法核验的是正文实验数字与理论保证的形式化细节（已在 01 篇声明）。

## 已下载（`research/papers/tianmouc-sensing-2608/`，脚本 `research/download_tianmouc_sensing_2608.sh`）

| 文件 | 来源 | 价值 |
|---|---|---|
| `tianmouc-quantitative-evaluation-bvs-2504.19253.pdf` | arXiv（同团队 Taoyi Wang 等） | Tianmouc TD/SD/RGB 模态与事件相机的定量对比——理解本篇输入表示的最好公开材料 |
| `diffusion-extreme-highspeed-reconstruction-iccv2025.pdf` | CVF Open Access（同团队 Meng & Lin） | 同一传感器上的生成式重建路线，与本篇判别式 IGFNet 路线对照 |

SHA-256 校验：`research/papers/tianmouc-sensing-2608/CHECKSUMS.sha256`

## 尝试失败

| 项 | 原因 |
|---|---|
| IOP NCE 2026 去噪论文 PDF（10.1088/2634-4386/ae0a76，Open Access） | IOP 反爬，curl 返回 HTML challenge 页（换 UA/Referer 两次均失败）→ link-only |
| Code Ocean 胶囊（10.24433/co.2136222.v1） | 页面抓取超时；作为代码可用性的第三方存档记录 link-only |

## link-only 参考

- 前作（必读背景）：Yang, Wang, Lin et al. "A vision chip with complementary pathways for open-world sensing", Nature 629, 1027–1033 (2024)，https://doi.org/10.1038/s41586-024-07358-4 （付费墙；公开部分已归档 `sources/nature-2024-tianmouc-article-page.txt`）
- 数据集：https://huggingface.co/datasets/ordinarabbit/Tianmouc-R （129 GB，未下载——超出研究包需要）
- 算法生态：https://github.com/Tianmouc/tianmoucv （pip 包 + 模拟器 + demo）
- 官方代码：https://github.com/Tianmouc/TMC-SSL-Representation （60 MB，本地克隆核验后未入库，复现命令在下载脚本注释中）
