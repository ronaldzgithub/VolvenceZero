# 07｜PDF 正文证据审计

## 1. 审计目的

本文件不是另一篇综述，而是把最容易被标题、摘要或新闻稿夸大的断言重新压回论文正文。页码按归档 PDF 的印刷页/章节记录；预印本与正式版不一致时明确标注版本。

审计流程：

1. 25 个 PDF 全部通过 `%PDF`、`%%EOF`、`pdfinfo` 与 SHA-256 校验；
2. 使用 Poppler 提取全文，逐项搜索方法、样本、算力、限制和作者声明；
3. 对 Tong Test、2024 minimax 和 TongGeometry 首页做 PNG 渲染目视检查，版面/文字可读；
4. 对官网作者列表再次比对，修正“团队论文”和“朱松纯署名”标签；
5. 将正文内部不一致保留在报告中，不替作者悄悄统一数字。

## 2. 关键断言核验表

| 论文 | 正文证据位置 | 可支持的断言 | 必须携带的边界 |
|---|---|---|---|
| P06 Stochastic Grammar | 全文 §2–§6，105 页开放稿 | And/Or/terminal/parse graph 构成随机、上下文相关的图像文法 | 是视觉/对象生成表示，不是跨模块 snapshot 或持续记忆协议 |
| P17 Dark, Beyond Deep | 结论与未来方向；正文明确展开 FPICU | 功能、物理、意图、因果、效用是超越表面像素的关键常识结构 | 属综述与研究纲领；没有一个单模型完成全部 FPICU |
| P19 In-situ alignment | p.1 摘要；p.3–4 任务/实验；p.7 discussion | 三个 robot scouts 在互动中估计四类任务目标权重；135 人、三个解释条件支持双向心智模型收敛 | scouts 是软件 agent；是任务内 value ranking，不是实体机器人或全域价值对齐；短时 trust 未显著因解释组改变 |
| P21 Communicative Learning | 摘要、formalism 与 discussion | 被动、主动、教学可在递归 teacher/learner 心智模型下统一描述 | 理性与递归模型依赖假设；不是现实用户行为真值，也不自动成为合法 reward |
| P22 Tong Test | *Engineering* 34, p.12 摘要；p.13–17 DEPSI/能力/价值 | 提出 5 个 AGI 特征、DEPSI、能力+价值、5 级里程碑和组合式任务生成 | Perspective article；“infinite tasks”是生成空间主张，不是已穷尽验证的 benchmark；evaluation 不得回灌学习 |
| P25 Minimax concept induction | p.1 摘要；p.4 human evaluation；p.10 representation | RPM/MNS/O3 三任务；招募 600 人，每组问题收集 300 个有效响应；模型逐实例学习并达到/超过论文人群分布 | 使用对象中心编码、候选 filter family、渲染器与任务特定搜索空间；human-level 只限该协议 |
| P27 LEO | 摘要与 model/data sections | 统一 3D VL/VLA 接口覆盖 caption、QA、reasoning、navigation、manipulation | 依赖大规模策划/LLM 辅助数据和任务集；无跨 session 持续学习证明 |
| P32 SWM-AP | p.1 摘要；§4；p.10 limitations；appendix | 从轨迹推断 latent trait，并在 facility/team/tax 三环境改善回报和样本效率 | 实验是数值/模拟环境，含 rule-based agents、AdaSociety、AI Economist；论文未部署到真人政策，且承认可扩展性和 trait 可解释性限制 |
| P33 Physical-Social WM | 标题、摘要、roadmap | 提出物理与社会预测双向统一的 ACE 原则/路线图 | NeurIPS position paper，无完整实现比较；只能支持研究缺口 |
| P34 Absolute Zero | p.1–2 定义；p.4–5 algorithm/reward | 同一模型提出并求解代码推理题；Python executor 验证任务/答案；不使用外部人工/蒸馏训练题 | 仍依赖预训练基座、代码先验、Python 环境、格式与 RL 设计；不证明无验证器开放域自学习 |
| P35 UniFP | p.1–2 摘要/贡献；§4.2；§6 limitations | 无外部力传感器的统一力位策略；四足+人形 7 类实验；详细实验节四任务总体约 +39.5% | 摘要/§4.2 写 4 个任务，贡献列表写 3 个；高频、工作空间边缘和 sim-to-real 为明确限制 |
| P36 TongGeometry | p.1 摘要；p.3–4 search/benchmark；supplement limitation | 67 亿需辅助构造定理、41 亿对称；3/10 投稿被竞赛采用；IMO-AG-30 30/30 | 题库搜索用 10,368 CPU 核×30 天并受 196 题统计引导；32 CPU+RTX4090/38 分钟只对应求解；多点构造与 NL→DSL 仍困难 |
| P37 IAIL | Science Robotics DOI / BIGAI 正式摘要 | 共享语言意图空间支持 7 种真实机器人、30 场景的跨本体行为适配 | 当前本地无合法自动开放 PDF；只使用正式摘要，不补写未核验实验细节 |
| P38 OmniXtreme | p.1–3；方法 §III | flow-matching specialist→unified pretrain，冻结 base 后做 actuation-aware residual RL；Unitree G1 实机展示 | 2026 预印本；是运动控制两阶段训练，不满足 Volvence runtime gate/rollback 契约 |

## 3. 三个最重要的数字纠偏

### 3.1 TongGeometry 的“消费级算力”只适用于求解

- 求解：IMO-AG-30，32 CPU cores + 1 RTX 4090，最长 38 分钟。
- 题库生成：10,368 并行 CPU cores、30 天、196 道既有奥赛题作 guiding statistics。

若把二者合成“消费级机器生成 67 亿题并全部求解”，就是错误陈述。

### 3.2 UniFP 的任务数在同一 PDF 内不一致

- 摘要和 §4.2：four real-world tasks；
- contribution list：three challenging tasks；
- §4.2 实际逐项列出 wipe-blackboard、open-cabinet、close-cabinet、open-drawer-occlusion，共四项。

本包采用详细实验节的四项，同时公开记录不一致，不用新闻稿替论文修数。

### 3.3 “600 humans”不是每道题 600 个有效答案

P25 说明招募 600 名参与者，对 6 RPM、8 MNS、10 O3 题做测试，并为每组问题收集 300 个有效响应。引用时应写“招募 600 人、每组 300 个有效响应”，不能写成“每道题 600 人”。

## 4. 署名复核

官网作者字段确认下列论文均有 `Song-Chun Zhu` 署名，不能标成纯团队无朱署名：

- P23 X-VoE
- P26 Neural-Symbolic Recursive Machine
- P28 CivRealm
- P29 AdaSociety
- P30 ProAgent
- P37 Cross-Robot Intention Alignment

下列代表作在 BIGAI 官方目录中，但作者列表无朱松纯：

- P24 MEWL
- P34 Absolute Zero
- P35 UniFP
- P38 OmniXtreme

## 5. 标题/版本复核

- P22 归档正式题名为 `The Tong Test: Evaluating Artificial General Intelligence Through Dynamic Embodied Physical and Social Interactions`；2023-08-09 online，2024 年第 34 卷。
- P35 归档 CoRL 版题名为 `Learning a Unified Policy for Position and Force Control in Legged Loco-Manipulation`；BIGAI 目录使用 `Learning Unified Force and Position Control...`。
- P36 本地全文是 arXiv 开放版本；正式出版信息与最终断言以 Nature Machine Intelligence 页面为准。
- P20 文章卷期页面显示 2022，TongClass 页面标 March 2023；本包保持 `2022/2023` 双口径。

## 6. 下载审计结论

- 目标：25 个自动下载 PDF + 1 个浏览器开放/link-only 项；
- 成功并校验：25/25 PDF；
- access-limited：P20 SciOpen 自动端点返回 404/405，正式 DOI、TongClass 和浏览器 PDF 链接均已登记；
- 付费墙：P01/P02/P03/P04/P37 等只保留 DOI/正式页，不绕过；
- 校验清单：[`papers/SHA256SUMS`](papers/SHA256SUMS)；
- 状态清单：[`papers/download-summary.tsv`](papers/download-summary.tsv)。
