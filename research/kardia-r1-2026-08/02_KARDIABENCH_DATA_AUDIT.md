# 02｜KardiaBench 数据审计：有没有用、能不能用

## 1. 先给结论

KardiaBench 对 Volvence **有研究价值，但当前没有直接进入产品或训练主链的许可与证据条件**。

最合适的候选用途是：

1. 经授权后，作为短多轮 persona-conditioned empathy 的**外部只读 benchmark**；
2. 把 Kardia-R1 作为 Companion Bench 的一个**参考系统**；
3. 借鉴其数据 schema，使用我们自己有权使用的 synthetic / consented personas 生成独立数据。

不合适的用途是：

- 直接混入商业训练语料；
- 把 scraped profile 写进 `user_model` / CMS；
- 用 emotion 或 rubric 分数驱动 PE/credit；
- 用 8–10 turn 合成对话证明跨 session 关系连续性；
- 用 MBTI 当用户的稳定心理真值。

## 2. 数据事实

来源：[论文附录 A](papers/kardia-r1-arxiv-2512.01282v2.pdf) 与
[Hugging Face dataset card](https://huggingface.co/datasets/Jhcircle/KardiaBench)。

| 指标 | Train | Test | All |
|---|---:|---:|---:|
| Dialogues | 19,533 | 2,547 | 22,080 |
| Utterances / turn-level pairs | 157,790 | 20,290 | 178,080 |
| Avg. multi-turns | 8.078 | 7.966 | 8.07 |
| Avg. query length | 51.34 | 51.57 | 51.36 |
| Avg. understanding length | 41.49 | 41.28 | 41.46 |
| Avg. reasoning length | 53.21 | 52.82 | 53.17 |
| Avg. response length | 26.94 | 26.83 | 26.93 |
| Emotion labels | 32 | 32 | 32 |
| User profiles | 639 | 32 | 671 |

公开文件清单显示：

- `train.jsonl`：约 248 MB；
- `test.jsonl`：约 31.7 MB；
- gated，未登录并接受条件时只能看文件清单，不能读样本。

公开 card 声明的顶层字段：

| 字段 | 含义 | Volvence 风险/价值 |
|---|---|---|
| `person` | MBTI、About、Signature、Recent Activities 等完整 profile 字符串 | 信息丰富，但重识别与版权风险最高；禁止直接入 CMS |
| `mbti` | profile 中提取的 MBTI | 可作弱分层 metadata，不得作心理本体或隐状态真值 |
| `emotion` | 32 类目标 emotion | 可作只读分类锚；类别不平衡且不覆盖关系状态 |
| `situation` | 初始情境 | 可作 scenario seed；源自 EmpatheticDialogues 分布 |
| `anon_username` | 匿名化用户名 | 仅去标识一个字段，不保证整份 profile 不可重识别 |
| `messages` | system/user/assistant 的完整结构化对话 | 对短多轮表达与 context conditioning 有价值 |

dataset card 示例标签写成 `<understanding>...</understanding>`，论文、模型和 GitHub 实际使用
`<|understanding_begin|>...<|understanding_end|>`。因此接入前必须以真实 JSONL + tokenizer
special tokens 为准，不能照 card 示例写 parser。

## 3. 它是什么数据，不是什么数据

### 3.1 真实部分

- 671 份 PersonalityCafe 公开、伪匿名用户档案；
- EmpatheticDialogues 的 situation / emotion 分布；
- test response 的人工检查与错误修正；
- 160 个 case 的独立人类 A/B 评测（这是论文评测，不等于整个数据集都经人工逐项审核）。

### 3.2 合成部分

- 用户每一轮具体话语；
- assistant 的 understanding / reasoning / emotion / response；
- assistant 的 rubric 修订轨迹；
- 对话中的情绪演化与“问题已解决”状态。

因此“real-world users”不应改写成“真实用户对话”。更准确的定义是：

> 真实公开档案作为条件，GPT-4o 生成并筛选的短多轮情感支持语料。

## 4. 数据分布的价值与局限

### 4.1 可能是 profile-disjoint test，但尚未样本级核验

附录列出 train 639 profiles、test 32 profiles，二者相加正好为 671，强烈暗示 profile-disjoint
切分。这对测试新用户泛化很有价值。

但正文同时写 profiles “used consistently across both training and evaluation”，且没有公开 split
manifest 或分配算法。由于 gated 样本未获访问，本包只记录“统计上强烈暗示”，不把它升级为
已核验事实。

### 4.2 是短多轮，不是 longitudinal

论文的生成上限为 10 exchanges，超过 90% 的对话集中在 8–10 turn，图中 10-turn 处还有
明显堆积。这说明长度分布受到 `Tmax=10` 的截断机制强烈塑形。

适用：

- session 内 emotion tracking；
- profile + situation + recent turns 的条件化；
- 简短支持性回复；
- 相同 profile 下的多情境覆盖。

不适用：

- 跨 session 恢复；
- 用户纠正、撤回、事实漂移；
- rupture/repair 的真实后果；
- 长期依赖、信任校准或 relationship continuity。

### 4.3 emotion 分布保留原始不平衡

32 个 emotion 的频率接近 EmpatheticDialogues，`caring / hopeful / grateful` 等高频类占优。
保留自然频率有生态价值，但只报 overall accuracy 容易被常见类主导。若作为只读 benchmark，
至少应同时读 macro-F1、per-class recall、confusion matrix、calibration 和 rare-class slice；
不能只复用论文的 Emotion Accuracy。

### 4.4 profile grounding 很丰富，但可能过度 MBTI 化

档案同时包含长文本、自述、兴趣、近期活动与 MBTI，价值并不只来自 MBTI。论文 case study
却直接用 “ISFJ warmth / loyalty” 解释行为，容易把类型标签变成刻板归因。

对 Volvence，优先级应是：

```text
用户明确自述 / 可纠正事实
  > 当前关系与边界的 typed outcome
  > 有溯源的长期模式
  > MBTI 等自报弱 metadata
```

MBTI 不能覆盖用户当前明确陈述，也不能直接决定 scene、mode、route 或 action。

## 5. 用途分级

| 候选用途 | 技术价值 | 当前可用性 | 裁决 |
|---|---|---|---|
| 外部只读 empathy benchmark | 高 | gated + NC/ND + 需污染检查 | **申请后优先考虑** |
| Kardia-R1 作为 reference SUT | 中高 | 模型 gated，约 15.23 GB BF16 | **可考虑，不作核心 substrate** |
| 短多轮 expression SFT | 中 | 许可、隐私、synthetic-style、CoT 暴露 | **当前不采纳** |
| emotion classifier / readout | 中 | 32 类、英文、分布不平衡 | **只可作外部验证锚** |
| `user_model` owner 训练 | 低到中 | raw profile 风险高；无纠正/撤回轨迹 | **不直接使用** |
| relationship continuity | 低 | 最多 10 turn、无跨 session | **不支持该主张** |
| PE / credit 学习源 | 无 | label/judge 不是实际后果 | **明确禁止** |
| 危机/临床安全训练 | 低 | 无危机专项统计与 recall 证据 | **不能承担** |
| scenario 结构参考 | 高 | 可只借方法，不复制受限文本 | **立即借鉴 schema** |

## 6. 许可不是一回事

| 资产 | 页面/文件标注 | 直接含义 |
|---|---|---|
| 论文 PDF | PDF metadata：CC BY 4.0 | 可按署名条件共享论文；不覆盖数据和模型 |
| GitHub 代码 | MIT | 只覆盖公开仓库里的代码/文档；当前没有训练实现 |
| Hugging Face 模型 | MIT + gated access conditions | 需先接受实际 gated 条款；不能只看卡片标签 |
| KardiaBench | `CC BY-NC-ND 4.0` + gated research-only | 非商业；修改/衍生材料不可分发；另有访问协议 |

[CC BY-NC-ND 4.0 官方说明](https://creativecommons.org/licenses/by-nc-nd/4.0/)
明确：不得用于主要追求商业优势或金钱补偿的用途；若 remix / transform / build upon，不能分发
修改后的材料。论文还写明数据集禁止再分发与商业使用。

工程裁决：

- 不因 GitHub 是 MIT 就把数据或模型一并视为 MIT；
- 不把“可申请下载”写成 open source；更准确的词是 gated research release；
- 在商业训练、衍生权重、对外 benchmark transcript、数据转换发布前，必须取得书面许可与
  法律审查；
- 当前不要把 JSONL 放进本仓库或普通 artifact 目录。

## 7. 隐私与来源风险

论文称仅使用公开伪匿名资料，并对用户名做不可逆 hash；dataset card 还称没有敏感信息。
但论文 case study 仍展示了 gender、relationship status、occupation、兴趣、特定音乐/地点、
近期帖子与媒体活动等高度可链接信息。

这意味着：

- hash username 不能阻止基于独特文本片段的搜索重识别；
- public 可见不等于主体同意其资料被用于情感模型训练；
- rich profile 可能包含第三方、健康、性取向、关系与其他敏感线索，即使作者未将其标为
  sensitive attribute；
- 对关系产品而言，这类资料进入模型的伦理风险高于普通网页语料。

因此 raw profile 不得进入 Volvence 的 `user_model`、CMS、rare-heavy adapter 或共享训练集。
即使只做内部研究，也应使用隔离存储、访问日志、禁止搜索式重识别、禁止原文出现在报告中。

## 8. 申请数据前必须拿到的答案

1. gated access agreement 的完整文本，是否允许公司内部非商业研究；
2. 是否允许训练衍生模型、是否允许发布权重、分数或模型输出；
3. profile 的采集时间、页面许可/ToS 依据、删除与主体撤回机制；
4. train/test profile manifest、去重方法和是否真正 profile-disjoint；
5. 与 EmpatheticDialogues、公开模型预训练语料的污染 / contamination 说明；
6. test set 人工修正比例、标注者一致性、哪些字段被人工改写；
7. JSONL 是否含失败候选、rubric 分数和 revision path；公开 card 只承诺 final `messages`；
8. profile 去标识报告与重识别攻击审计；
9. 商业评测、对外报告和保存期限是否需要额外书面授权。

## 9. 获准后的安全落点

若权利审查通过，也应先把 KardiaBench 放在**受限外部数据区**，不进入 runtime owner：

```text
gated raw dataset
  → access-controlled read-only audit
  → provenance / privacy / split / contamination report
  → external benchmark adapter
  → evaluation-only artifact

禁止：evaluation result → PE / credit / controller / memory writeback
```

这是数据治理建议，不是本研究包授权的实现任务。

