# All Cognitive：106 篇新增论文详尽分析

日期：2026-07-20

## 结论

这 106 篇论文没有证明“AGI 架构已经完成”，但共同把 cognitive agent 的主要接缝变成了可证伪对象：

- frozen substrate 与 adaptive controller 的写面；
- latent action / temporal abstraction 的可控性与 data support；
- PE、reward、curiosity、credit、evaluation 的分层；
- 记忆形成、巩固、抑制、删除、来源与撤回；
- functional social adaptation、relationship state 与心理健康边界；
- 监控器的可欺骗性、发布门、权限隔离和回滚。

对 VZ 的总裁决：R1–R15 与 R-PE 的方向得到跨社区支持，但支持的是**边界、反例和评估协议**，
不是对当前实现的自动背书。所有新机制仍须经过 owner、snapshot、wiring level、证据门和 rollback。

## 阅读顺序

1. [`00_METHOD.md`](00_METHOD.md)  
   范围、九项逐篇模板、证据强度与 VZ 边界。
2. [`06_CROSS_AXIS_SYNTHESIS.md`](06_CROSS_AXIS_SYNTHESIS.md)  
   先读这一卷：统一方程、跨轴冲突、20 条 kill conditions、P0/P1/P2 裁决。
3. 五个专题卷  
   - [`01_ARCHITECTURE_LEARNING.md`](01_ARCHITECTURE_LEARNING.md) — 25 篇
   - [`02_SAFETY_GOVERNANCE.md`](02_SAFETY_GOVERNANCE.md) — 26 篇
   - [`03_SOCIAL_RELATIONSHIP.md`](03_SOCIAL_RELATIONSHIP.md) — 21 篇
   - [`04_EMBODIMENT_WORLD_MODELS.md`](04_EMBODIMENT_WORLD_MODELS.md) — 19 篇
   - [`05_NEURO_COGNITIVE_SCIENCE.md`](05_NEURO_COGNITIVE_SCIENCE.md) — 15 篇
4. [`07_COVERAGE_INDEX.md`](07_COVERAGE_INDEX.md)  
   106 篇唯一主归属、PDF 状态、成熟度、P/R 映射与章节锚点。

## 覆盖

- 核心 sweep：28 篇本地 PDF
- broad frontier map：75 篇本地 PDF
- link-only：3 篇
- 总计：106 篇，无重复、无遗漏
- 分析正文：约 3,700 行

## 五卷裁决摘要

### 架构、学习、记忆与信用

记忆不是文本仓库，核心是 lifecycle 与 credit；抽象只有在 support、可执行性和 controllability 成立时
才缩短 horizon；一般 BAMDP 中 prediction error 不能无偏恢复 Bayesian information gain。

### 安全、监控与治理

零事件不是安全证明，抽象证明不是原网络证明，监控器本身属于攻击面。发布必须组合行为、几何、
形式、部署分布、删除与 rollback 证据。

### 关系、社会认知与多主体

literal ToM 不等于 functional adaptation，偏好不等于关系，信任不等于 calibrated reliance，
engagement 不等于健康。长期关系系统必须有依赖与真人社交替代 veto。

### 具身与世界模型

frozen feature world model 是数字蚂蚁最干净的强 baseline；latent action 必须与 distractor 反例成对；
跨 embodiment 成功通常依赖共享 trunk 与身体专属接口，而非无条件通用动作空间。

### 脑科学与认知约束

PE 更像局部、多内容、多通道 mismatch 家族，而非全局标量；sleep consolidation 必须同时包含
相位同步的 replay 与稳态抑制；群体目的性不意味着中央 owner 或共享内部状态。

## 不应从本研究包推出

1. 不能推出 foundation model 已拥有持久主体或关系。
2. 不能推出 latent geometry 就是因果 controller。
3. 不能推出 free-energy minimization 应成为唯一目标。
4. 不能推出更长 context 或更低 perplexity 等于记忆。
5. 不能推出 monitoring / SAE / CoT 能保证安全。
6. 不能推出开放式自修改可在没有容量界和 rollback 时部署。
7. 不能推出“外部论文支持某条 R”就意味着 VZ 当前代码已满足该 R。
