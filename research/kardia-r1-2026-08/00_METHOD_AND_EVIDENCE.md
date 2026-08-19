# 00｜方法、口径与证据边界

## 1. 研究问题

本包回答五个问题：

1. Kardia-R1 真实公开了什么，论文、代码、模型与数据分别到什么程度？
2. 论文的主要提升来自 KardiaBench、结构化 SFT 还是 Rubric-ERL？
3. KardiaBench 是什么数据，适合 Volvence 的哪些用途，不适合哪些用途？
4. Kardia 的离线后训练路线与 Volvence 的 Appendable / Readable / Learnable /
   Steerable 路线同构在哪里、冲突在哪里？
5. 哪些思想可以直接吸收，哪些必须改写，哪些应明确拒绝？

用户明确要求不做实验，因此本包止于研究与采纳决策，不设计训练或 benchmark run。

## 2. 来源等级

| 等级 | 来源 | 用途 |
|---|---|---|
| S | WWW 2026 论文记录、arXiv v2 正文与附录 | 方法、公式、数据统计、结果、限制与发表状态 |
| A | 作者 GitHub、Hugging Face 模型卡 / 数据集卡 | 当前公开资产、用法、许可标签、gated 状态与文件清单 |
| B | 作者主页、DBLP / 会议索引 | 发表信息交叉核验 |
| C | README 宣传语、Hub 自动元数据 | 资产发现与当前状态；不单独支撑强科学主张 |
| 排除 | 二手解读、无来源榜单、搜索摘要中的性能转述 | 不进入关键事实链 |

论文数字以 arXiv v2 的表格和正文为准；GitHub/Hugging Face 若与论文冲突，单独记录，
不自行替作者消解。

## 3. 审计快照

基准日：2026-08-19（Asia/Shanghai）。

| 对象 | 审计快照 |
|---|---|
| 论文 | arXiv `2512.01282v2`，16 页；WWW 2026，会议索引页码 9230–9240 |
| 本地 PDF | `papers/kardia-r1-arxiv-2512.01282v2.pdf`；SHA-256 `43be1231748e2cf9ba4c2160a7643f7ce4c175fd86ebb3354b5a261faab5cefa` |
| GitHub | `JhCircle/Kardia-R1`，审计 checkout `ee8d3be788c96c8811311410553414598237d1c5`（2026-03-03） |
| GitHub 实际内容 | 两份推理脚本、README、MIT LICENSE、三张图片；无训练、reward、评测、数据构造或 split 代码 |
| 模型 Hub | gated；Qwen2.5-7B-Instruct 衍生，四个 BF16 safetensors 分片约 15.23 GB；页面标注 MIT |
| 数据 Hub | gated；`train.jsonl` 约 248 MB、`test.jsonl` 约 31.7 MB；页面标注 `CC BY-NC-ND 4.0` |
| 数据访问 | 未申请、未下载；只审计公开 card、文件清单与论文附录 |

## 4. 阅读协议

论文按以下六项读取：

1. **对象**：真正解决的是短多轮 persona-grounded 情感支持，还是长期关系适应？
2. **表示**：profile、emotion、reasoning、response 分别以什么形式存在？
3. **信号**：数据筛选、SFT target、GRPO reward 和最终 evaluation 分别来自哪里？
4. **证据**：哪些数字有多 backbone、消融或人评支持？
5. **可复现性**：代码、prompt、split、seed、judge 与 checkpoint 选择是否公开？
6. **外推边界**：不能从该结果推出什么？

文中使用三类断言标签：

- **事实**：一手材料直接给出；
- **推断**：由多个事实合并得出，并明确写出推断；
- **采纳决策**：结合 Volvence 契约作出的工程判断，不冒充论文结论。

## 5. 术语收缩

| 原文用语 | 本包采用的窄口径 |
|---|---|
| `real-world profiles` | 来自公开论坛的真实、伪匿名档案文本；不等于获得主体同意的产品用户资料 |
| `multi-turn` | 平均 8.07 turn、最多 10 exchange 的合成短轨迹；不等于跨 session 长期关系 |
| `transparent reasoning` | 模型生成了可见四段文本；不等于这些文本忠实读出内部因果计算 |
| `verifiable reward` | 格式与 emotion exact-match 可机械验证；LLM rubric 只能称可解释的软评分 |
| `safety` | GPT-5-mini / 人评 rubric 的单轴均分；不等于临床安全、危机识别或长期依赖风险已通过 |
| `user-grounded` | 生成时显式给入 profile；不等于系统拥有可追加、可纠正、可撤回的用户模型 owner |

## 6. 明确限制

- 无 gated 数据样本，因此无法核验实际 JSON schema、profile 去标识质量、重复率、污染、
  split manifest、隐藏 rubric 轨迹是否随数据发布。
- 无训练代码，因此无法核验论文公式是否与实际 Ms-Swift 配置一致。
- 无模型运行；这是用户指定的范围，不是证据失败。
- 自动评测与人评原始逐样本记录未公开，无法重算置信区间、评审一致性或顺序效应。
- 许可与隐私结论是工程风险分级，不是法律意见。

