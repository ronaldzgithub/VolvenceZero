# 来源、论文与下载审计

> 核验日期：2026-08-29
>
> 来源策略：方法和数字只使用论文、官方项目页、官方技术博客与官方 benchmark 文档；二手报道不进入结论链。

## 1. 核心来源

| 等级 | 来源 | 用途 | 证据边界 |
|---|---|---|---|
| P1 | [AVO: Agentic Variation Operators for Autonomous Evolutionary Search](https://arxiv.org/abs/2603.24517) | AVO 公式、single-lineage loop、kernel setup/results、三项代码消融 | arXiv v1；无官方代码链接 |
| O1 | [NVIDIA AVO Reaches 100% on ARC-AGI-3](https://developer.nvidia.com/blog/nvidia-avo-reaches-100-on-arc-agi-3-demonstrating-a-frontier-level-general-purpose-architecture-for-long-horizon-autonomous-agents/) | 8 月 harness 图、memory/supervisor 叙述、ARC public-set 结果 | 官方博客，不是独立论文 |
| O2 | [Where Security Fits in an AI Agent Stack](https://developer.nvidia.com/blog/where-security-fits-in-an-ai-agent-stack/) | harness 与 authoritative runtime boundary 的官方 NVIDIA 立场 | 架构/安全文章，不是 AVO 实验 |
| B1 | [ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence](https://arxiv.org/abs/2603.24621) | benchmark 目标、设计与最初 frontier baseline | benchmark 论文，不证明 AVO |
| B2 | [ARC-AGI-3 Scoring Methodology](https://docs.arcprize.org/methodology) | RHAE 公式、环境 action 与内部操作的计费边界 | 官方动态文档 |
| B3 | [ARC-AGI-3 official page](https://arcprize.org/arc-agi/3) | public benchmark 定位与 100% 解释 | 官方动态页面 |

P1 是本文关于 AVO 方法的唯一论文级主锚。O1 中超出 P1 的 general-purpose / ARC 结论均单独标注为
官方工程报告。

## 2. AVO 论文元数据

- 标题：*AVO: Agentic Variation Operators for Autonomous Evolutionary Search*
- 作者：Terry Chen、Zhifan Ye、Bing Xu、Zihao Ye、Timmy Liu、Ali Hassani、Tianqi Chen 等 23 人
- 机构：NVIDIA
- arXiv：`2603.24517 [cs.LG]`
- 版本：v1，2026-03-25
- DOI：[10.48550/arXiv.2603.24517](https://doi.org/10.48550/arXiv.2603.24517)
- PDF：[arXiv PDF](https://arxiv.org/pdf/2603.24517)
- 公开状态：preprint；arXiv 页面未列 peer-reviewed venue

## 3. 直接相邻论文

| 论文 / 项目 | 链接 | 为什么相关 | 本地已有 |
|---|---|---|---|
| Evolution through Large Models | [arXiv 2206.08896](https://arxiv.org/abs/2206.08896) | LLM 作为程序 mutation operator 的早期源头 | 未为本包重复下载 |
| FunSearch | [Nature paper](https://doi.org/10.1038/s41586-023-06924-6) | 可验证程序搜索；AVO 的固定-pipeline 对照 | 仓库既有研究引用 |
| AlphaEvolve | [arXiv 2506.13131](https://arxiv.org/abs/2506.13131) | archive/sample/eval 外置、LLM 负责 Generate；AVO 主要对照 | [`research/papers/dm/...pdf`](../papers/dm/alphaevolve-coding-agent-scientific-engineering-discovery-2506.13131.pdf) |
| LoongFlow | [arXiv 2512.24077](https://arxiv.org/abs/2512.24077) | 固定 PES + MAP-Elites/Boltzmann；AVO 明确对照 | 未为本包下载 |
| Learning to Discover at Test Time | [arXiv 2601.16175](https://arxiv.org/abs/2601.16175) | test-time gradient 更新 Generate policy；与 AVO 的 in-context autonomy 正交 | 仓库前沿研究已有记录 |
| FlashAttention-4 | [arXiv 2603.05451](https://arxiv.org/abs/2603.05451) | AVO 的开源 kernel baseline 与知识来源 | 未为本包下载 |
| VISTA | [official project](https://vista-research.github.io/) | AVO ARC interface 的直接设计参照与对比 | 只有 blog/project page，无论文 |
| Tycho | [arXiv 2607.28287](https://arxiv.org/abs/2607.28287) | ARC 的 programmatic world-model 对照 | 未为本包下载 |

## 4. 开源与复现审计

截至核验日期：

- AVO arXiv 的 `Code, Data, Media` 区域没有列出官方实现；
- AVO 论文正文没有给出 NVIDIA/NVlabs GitHub repository；
- NVIDIA ARC 博客没有给出 AVO source、prompt、trajectory replay 或 memory artifact；
- 论文只把 agent 描述为内部开发、frontier LLM 驱动；kernel run 的具体模型未披露；
- 网上存在第三方“AVO framework implementation”，但不是 NVIDIA 官方代码，不用于还原实现或支持性能主张；
- VISTA 是官方项目页/blog post，不是 arXiv paper；
- ARC public-set 结果不能替代 private/semi-private result。

因此复现等级判为：

| 项 | 状态 |
|---|---|
| 方法概念与公式 | 可读 |
| benchmark configuration | 部分充分 |
| 最终 kernel source | 未公开 |
| AVO agent source/prompt/model config | 未公开 |
| 7-day attempt/commit trajectory | 未公开 |
| supervisor triggers/interventions | 未公开 |
| ARC run replay | NVIDIA 页面未提供 |
| 独立端到端复现 | 当前不可完成 |

## 5. 本地下载状态

本研究环境可通过网页检索读取 arXiv HTML 和官方页面，但命令行下载 PDF 时 DNS 解析失败：

```text
curl: (6) Could not resolve host: arxiv.org
```

因此本包采用 **link-only**，没有伪造 PDF、checksum 或“已下载”记录。需要离线归档时可在联网环境执行：

```bash
mkdir -p research/nvidia-avo-2026-08/papers
curl -L --fail \
  https://arxiv.org/pdf/2603.24517 \
  -o research/nvidia-avo-2026-08/papers/avo-agentic-variation-operators-2603.24517.pdf
curl -L --fail \
  https://arxiv.org/pdf/2603.24621 \
  -o research/nvidia-avo-2026-08/papers/arc-agi-3-benchmark-2603.24621.pdf
shasum -a 256 research/nvidia-avo-2026-08/papers/*.pdf
```

下载后应把 checksum 固定到独立 `SHA256SUMS`，并重新核对 arXiv version；不要把第三方重实现当作原始
artifact。

## 6. 仓库内部依据

本次与 Volvence 的对比以当前代码/契约为准，主要入口：

- [`docs/specs/00_INDEX.md`](../../docs/specs/00_INDEX.md)
- [`docs/appendable-readable-learnable-steerable.md`](../../docs/appendable-readable-learnable-steerable.md)
- [`docs/specs/rsi-forge.md`](../../docs/specs/rsi-forge.md)
- [`docs/specs/research-opportunity-discovery.md`](../../docs/specs/research-opportunity-discovery.md)
- [`docs/specs/research-control-plane.md`](../../docs/specs/research-control-plane.md)
- [`docs/specs/research-promotion-pipeline.md`](../../docs/specs/research-promotion-pipeline.md)
- [`forge/src/volvence_forge/research_opportunity.py`](../../forge/src/volvence_forge/research_opportunity.py)
- [`forge/src/volvence_forge/research_control.py`](../../forge/src/volvence_forge/research_control.py)
- [`forge/src/volvence_forge/research_promotion.py`](../../forge/src/volvence_forge/research_promotion.py)
- [`coding_memory_inheritance/task.yaml`](../praxist_tasks/coding_memory_inheritance/task.yaml)

当前工作区的 Research Opportunity / Control Plane / pilot task 是尚未提交的用户改动；本研究只读取并
引用其当前状态，没有修改这些文件。
