# Volvence Forge：有界 RSI 元层

Forge 是一个指向 Volvence 的、物理上独立的 RSI 元层。它不属于任何
`vz-*` 或 `lifeform-*` wheel，也不进入实时 Brain；它只读取已经发布的开发
轨迹、测试/评估结果、promotion artifact 和计划文档，生成受白名单约束的
编辑提案。第一阶段的目标是让“造 Volvence 的开发机器”能够从失败中提出
窄改动，而不是让 Forge 自己修改评分器、门禁或冻结模型。

## 为什么做 Forge

Lilian Weng 关于 harness engineering 的核心判断，是近期可行的 RSI 不是在线
重写基础模型权重，而是优化产生结果的部署系统：prompt、结构化上下文、工作流、
harness 代码，再到改进 harness 的 optimizer。代码是强 coding agent 已经熟悉的
搜索空间，因此改进速度可以借助现成的编辑、测试、回滚能力。这个判断比把任何
自动科研流水线都称为“AI scientist”更严格：静态、由人手写出的实验 harness 能
自动执行研究，却没有自动改进研究机器本身。

这个方向对 Volvence 的价值是速度和经济性杠杆，不是基础能力杠杆。Volvence 已经
有 prediction error、失败轨迹、owner 快照、Evaluation、ModificationGate、
`WiringLevel` 和 rollback drill；当前 rare-heavy 层的瓶颈是人要手工查看失败、
定位 owner、写提案、跑证据。Forge 把这段“失败挖掘 → 有界提案 → 验证”的前端
自动化，让长尾失败可以被系统性处理，让角色和开发路径的维护成本不随规模线性
增加。冻结基底不会因为 Forge 变聪明，关系质量也不会被短期评分直接优化；可靠的
收益必须来自硬验证或独立的纵向/人类证据。

## 为什么是元层而不是主线 wheel

Forge 物理上与 `packages/` 平级，仅读取公开 artifact 和文件接口，产出的 patch
必须经过人工审阅（第二阶段再接 `ModificationGate.OFFLINE`）。这样 evaluator、
verifier、权限控制和白名单都位于被进化循环之外，每一点收益才可归因，也不能靠
关闭测试、改变预算或改写 gate 来制造分数。Forge 不能 import Volvence runtime，
Volvence runtime 也不能 import Forge；未来若规模足够大，可按 `SPLIT.md` 的触发
条件拆成独立仓库。

## 两阶段路线

第一阶段只编辑开发环 harness：`.cursor/rules/*.mdc` 与 `forge/prompts/**`。
`packages/**`、`tests/**`、`scripts/run_gate*`、`docs/specs/**`、评价/验证代码、
Forge 自身源码和 ledger 永远是只读面。每个候选包含失败证据、三层根因（verifier
原因、agent 行为因果、暴露的 harness 机制）、预测影响、可能回归和回滚方法；
候选不会自动写入目标文件。

第二阶段才考虑产品 runtime 的 playbook、角色包或记忆整合策略，并要求接入
`ModificationGate.OFFLINE`、SHADOW 并行、独立纵向证据和人工晋升。模糊的关系质量
不能直接成为内环 reward；多样性坍缩、reward hacking、短期奖励损害长期关系、
以及“会改”与“会用”两个能力的混淆，都是必须保留在门外的风险。

## 循环与安全不变量

```text
只读轨迹/失败 artifact
        ↓
语义失败挖掘（三层因果记录）
        ↓
白名单内的提案（patch + manifesto）
        ↓
静态检查 + held-in/held-out 回归
        ↓
人工审阅 → apply → append-only ledger → 下一轮验证预测
```

1. 提案目标必须匹配 `forge/editable_surface.yaml`，扩面只能由人修改白名单。
2. Gate、evaluation、verifier、测试和 Forge 自身不能成为编辑目标。
3. 自然语言不得用关键词字典路由失败模式；结构化 LLM 输出和嵌入聚类必须带
   schema、证据引用和不确定性，无法归因时只报告、不提案。
4. apply 只接受 validate 通过且明确人工确认的提案；ledger append-only，拒绝、
   回滚和预测未兑现都保留记录。
5. Forge 失败时删除新增目录即可回滚；已应用的 rules/prompts 通过 Git revert
   回滚，Volvence runtime 状态和生产 WiringLevel 不受影响。

## 当前状态

包 0 已冻结目录、白名单、prompt/schema 资源和边界契约；mine/propose/validate/
apply 的实现按计划分包推进。Forge 不进入根 workspace 依赖，不改变 Volvence 的
实时行为，也不代表 Volvence 已获得自动生产晋升能力。
