# Kardia-R1 / KardiaBench 专项研究

> 研究日期：2026-08-19  
> 范围：论文深读、公开资产审计、KardiaBench 数据价值、与 Volvence 技术路线对账  
> 不在范围：模型运行、数据下载、训练复现、实验设计、运行时接线

## 一句话裁决

Kardia-R1 是一条有效的**离线领域后训练路线**：把真实网络档案与
EmpatheticDialogues 情境组合成 GPT-4o 合成的短多轮对话，再以四段式 SFT 和
Rubric-as-Judge GRPO 提升情绪识别与 profile-conditioned 支持表达。它最强的证据是
**KardiaBench 的 SFT 数据价值**，而不是“LLM judge 已让共情变得可验证”，更不是
在线关系学习已经成立。

对 Volvence 的直接结论：

- **值得借鉴**：档案、情境、对话历史、理解、情绪标签和最终表达的分面数据组织；
  profile-disjoint 的测试意图；短多轮 persona grounding；多维 rubric 作为只读评估面。
- **可申请后评估**：KardiaBench 作为外部只读 benchmark；Kardia-R1 作为
  Companion Bench 的参考 SUT。两者都必须先通过 gated terms、许可、隐私与污染审计。
- **不应照搬**：Rubric-as-Judge 进入在线 reward、token 空间 GRPO、把显式 reasoning
  文本当真实内部状态、把 MBTI 当用户本体、把原始抓取档案写入 CMS 或产品训练集。
- **数据当前不可直接采纳**：数据集为 gated、`CC BY-NC-ND 4.0`，论文还明确写了
  research-only、禁止再分发与商业使用；这与 GitHub 代码和模型卡的 MIT 是三套不同许可。

## 为什么“数据”比“RL”更重要

论文 Table 2 的四个 backbone 上，Kardia-R1 相对 SFT-only 的平均增量只有：情绪准确率
`+0.315` 个百分点、Relevance `+0.030`、Fluency `+0.018`、Empathy `+0.046`、Persona
`+0.098`、Safety `+0.044`；而 SFT-only 相对原始 backbone 的平均增量分别约为
`+59.257` 个百分点、`+0.739`、`+0.582`、`+1.187`、`+0.947`、`+0.249`。

所以更诚实的读法是：

> KardiaBench + 四段式 SFT 完成了绝大部分提升；Rubric-ERL 在部分主观维度上继续做了
> 小幅整理，其中 Qwen2.5-7B 的 Persona 增量最明显（`+0.248`），但 Gemma-2B 的
> Relevance 和 Safety、Gemma-7B 的 Fluency 相对 SFT 反而略降。

## 研究包结构

| 文件 | 内容 |
|---|---|
| [00_METHOD_AND_EVIDENCE.md](00_METHOD_AND_EVIDENCE.md) | 研究问题、来源等级、版本快照、断言口径与审计限制 |
| [01_PAPER_DEEP_DIVE.md](01_PAPER_DEEP_DIVE.md) | 数据合成、SFT/GRPO、reward、结果、消融、复现缺口与论文主张边界 |
| [02_KARDIABENCH_DATA_AUDIT.md](02_KARDIABENCH_DATA_AUDIT.md) | 数据规模、字段、切分、短长程适用性、许可、隐私与用途分级 |
| [03_VOLVENCE_ROUTE_COMPARISON.md](03_VOLVENCE_ROUTE_COMPARISON.md) | Kardia 与 Volvence 的架构、四能力轴、owner、PE/credit 与 Relationship Lab 对账 |
| [04_ADOPTION_DECISIONS.md](04_ADOPTION_DECISIONS.md) | 采纳 / 改造后采纳 / 拒绝项，以及数据申请与隔离建议 |
| [SOURCES.md](SOURCES.md) | 一手来源、公开资产状态、版本和本地归档校验 |
| [papers/](papers/) | arXiv v2 PDF 的本地只读归档 |

## 推荐阅读顺序

- 只看结论：`README → 04`
- 关心论文是否站得住：`README → 01 → 00`
- 关心数据能否用：`README → 02 → 04`
- 关心 Volvence 路线：`README → 03 → 04`

## 诚实边界

- 本包没有访问 gated 的 train/test JSONL，因此字段级结论来自论文与公开 dataset card，
  没有把“页面可见”写成“样本已审计”。
- 本包没有运行模型，既不复现论文分数，也不判断输出实际体验。
- 本包不提供法律意见；涉及商业训练、衍生模型或对外发布前，需要权利与隐私审查。
- 本研究不授权新增 runtime owner、slot、reward 或训练路径；若后续采用任何机制，必须另立
  单一收敛包并遵守现有 `DATA_CONTRACT`、PE-first 与 SHADOW 边界。

