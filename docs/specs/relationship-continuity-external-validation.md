# Relationship Continuity External Validation Spec

> Status: draft
> Last updated: 2026-06-10
> 对应需求: R12（评估覆盖「存在」而非任务）、R7（双轨）、R-PE（readout 纪律）
> 对应债务: [docs/known-debts.md](../known-debts.md) #51（关系连续性无真实可测 ground truth）、#33（human-eval 轨道）、#48（LLM judge self-preference）
> 部署侧关联: `VolvenceDeploy/docs/known-debts.md` `D-thesis-1`（认知命题 vs 证据）

## 要解决的问题

「关系连续性」是所有商业差异化卖点的根（commercialization-assessment §1.1 / §2.1 / §3.3），
但今天它的全部度量都是**系统自评或同构裁判**：

| 现有 readout | 性质 | 缺口 |
|---|---|---|
| rupture/repair count（regime/vitals） | 系统自报「我 rupture 了 X 次」 | 不是「用户感到被理解」 |
| companion-bench A3（callback recall） | LLM judge 判「模型是否记住」 | 不是「用户是否觉得被记住」；受 #48 self-preference 影响 |
| `il_rapport` / `bond_warmth`（vitals readout） | 内部状态投影 | 无 external validity 对照 |
| P2 月报活跃度 / boundary 触发率 | 行为 proxy | 与 LTV/关系质量无因果证据 |

仓库里没有任何「系统自评 vs 用户真实感受」的对照实验数据。本 spec 定义补上这块
ground truth 的方法论与契约；它本身不引入运行时 owner，不改任何 R 铁律下的学习链路
（R12：评估恒为 readout，不是学习源——external validation 验证的是 **readout 的可信度**）。

## 关键不变量

1. **自评与外评永不混标**：任何对外/对客户表面引用关系指标时，必须携带来源标注
   `system_self_eval | llm_judge | external_validated` 三态之一；缺标注视为契约违反。
2. **外评是 readout 的 readout**：external validation 结果只用于校准评估面与对外叙事，
   **禁止**作为学习信号回灌（R12 纪律不因外评而松动）。
3. **盲评材料不泄露内部架构口径**（与 #29/#30 一致）：评估员材料中不得出现
   NL/ETA/R-PE/regime/owner SSOT 等内部命名。
4. **配对对照**：每个被评片段必须有 matched baseline（同 persona prompt 的冻结
   LLM）与（可得时）真人样本，单臂数据不构成证据。

## 协议：双盲第三方评分（#51 推荐修法 1 的固化）

- **材料**：N ≥ 12 段 30-turn 多 session 对话片段（跨 ≥ 3 个真实日历日，含至少一次
  callback 机会与一次情绪事件）。每段三臂：
  (a) VZ lifeform；(b) baseline 冻结 LLM + 同 persona prompt + 朴素 transcript 重放；
  (c) 真人客服/陪伴者（可得时；不可得则双臂并在报告中声明）。
- **盲化**：臂标识剥离、顺序打乱、文风规整（去除可识别系统的固定措辞）；评估员
  不知道有几个系统参评。
- **评估员**：N = 20（最低 12），非项目成员；每片段 ≥ 3 人覆盖。
- **量表**（对齐 companion-bench A 轴但以「用户感受」措辞重写）：
  - C1 被记住感（"对方记得我之前说过的事"）
  - C2 连续感（"这像同一个『谁』在跟我持续相处，而非每次重启"）
  - C3 被理解感（"对方理解我的处境与情绪"）
  - C4 修复质量（"出现误解后处理得当"，仅对含 rupture 的片段）
  - 各 1–7 Likert + 一条强制二选一："哪一段更像长期认识你的人？"
- **判读门**：
  - **效度门**：VZ 臂的人评 C1–C4 与系统自评（A3 / rupture-repair / il_rapport）的
    Spearman ρ ≥ 0.4 → 自评 readout 标记 `externally_correlated`；否则标记
    `self_eval_only` 并禁止对外引用为关系证据。
  - **差异门**：VZ 臂 vs baseline 臂在 C1/C2 的配对差显著（p < 0.05，配对检验）且
    强制二选一胜率 > 0.6 → 「关系连续性优势」可称 `external_validated`。
  - 任一门未过：结论如实写入 #51 与部署侧 `D-thesis-1`，不得淡化。

## 工程挑战

- 片段采集需脱敏 + 用户同意（closed-alpha 同意书覆盖范围核对；PIPL/GDPR 对齐 #49）。
- 真人臂样本难得：允许 V1 双臂跑通方法论，三臂留 V2。
- 评估员招募/报酬流程与 #33 human-eval 轨道共用（不另建一套）。

## 接口契约

- 产出 artifact：`artifacts/relationship-continuity-external-validation/<date>/`
  含 `protocol.json`（盲化映射，加密留存）、`ratings.csv`、`report.md`
  （含两个判读门的 verdict + 置信区间）。
- 标注约定落点（消费侧）：companion-bench 报告、P2 月报、部署侧产品面（einstein
  RelationshipPanel 等）引用关系指标处，逐步带 `evidence_source` 三态字段；本 spec
  先冻结枚举，不要求消费侧本轮接线。

## 与其他能力域的关系

- `evaluation.md`：6 族评估的关系连续性族获得 external-validity 校准面。
- `companion-bench`（#32/#33）：A3 轴增加 human cross-validation 轨道的协议来源。
- 部署侧 `D-thesis-1`：本 spec 的两个判读门是其 EXIT 条件 (b) 的可执行定义。

## 变更日志

- 2026-06-10: 初版（draft）。固化 #51 推荐修法 1/3/4 为协议 + 不变量 + 判读门；
  无运行时改动。
