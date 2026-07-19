# BOLT 专题研究执行摘要

> 更新时间：2026-07-14  
> 研究范围：BOLT 原文检索、四位主创公开论文、Boltzbit 官网论文、相邻 Bayesian online-learning transformer 文献。

## 1. 当前事实

`BOLT: Bayesian Online Learning Transformer` 仍未找到公开论文页面、arXiv 记录或 OpenReview 条目。再次检索只命中了同名异物：

- `BOLT: Bayesian Optimization in the Long Term`
- `BOLT: Bootstrap Long Chain-of-Thought`
- `BOLT: Fast Energy-based Controlled Text Generation`

因此，本专题不能把 BOLT 当作已公开论文复述，只能基于用户提供片段、主创公开论文和 Boltzbit 官网叙事进行技术推断。

## 2. 下载结果

已下载 33 篇 PDF 到 `papers/`：

- `clarke/`：4 篇。
- `jmhl/`：15 篇。
- `boltzbit/`：9 篇。
- `adjacent/`：6 篇。

两篇 OpenReview PDF 被 403 阻挡，未能二进制下载，但网页文本已可公开读取并已纳入分析：

- `Improving Continual Learning by Accurate Gradient Reconstructions of the Past`
- `Ergodic Measure Preserving Flows`

详见 `_download_summary.md`。

## 3. 主结论

BOLT 若存在，最可能不是一个普通 LLM 记忆插件，而是以下思想的交叉：

```text
Boltzmann / MCMC / HMC / Ergodic inference
    + JMHL-style amortised Bayesian inference
    + Clarke-style stable update-rule optimisation
    + Transformer latent memory / prior-to-posterior mapping
    = fixed-capacity online Bayesian updater for LLM personalization
```

它最值得我们吸收的是：

- 用一次前向更新 owner-local latent posterior。
- 在线不改全量 substrate。
- 用 uncertainty / plasticity 控制更新强度。
- 把历史压缩成固定容量状态，降低 token/context 成本。

它最不该被误读为：

- 单一 latent memory 足以替代 CMS。
- 用户 feedback 可以直接喂给黑箱 updater。
- 每次 forward update 可以替代 session/background consolidation。
- 学术 latent state 可以绕过 product runtime snapshot contract。

## 4. 对 Volvence 的判断

这批论文总体加强 Volvence，而不是证伪：

- **R1 多时间尺度**：fast latent update 被支持；慢反思/沉淀未被取代。
- **R2 冻结基底 + 自适应控制器**：强支持。
- **R3/R4 token 之上 latent 控制**：支持，但要区分 belief latent 与 control latent。
- **R-PE**：支持 prediction/evidence 驱动学习，但要求 typed evidence 更严格。
- **R5/R6 CMS**：被加强；固定容量记忆需要 replay / consolidation。
- **R7 双轨**：BOLT 公开线索偏 task workflow，未覆盖关系轨道，因此不能替代 self/relationship track。
- **R8/R11 snapshot owner**：产品化 BOLT-like 技术必须补 owner-public summary。
- **R15 rollback**：live-learning 更需要可回滚，而不是更少治理。

真正需要补强的是：Volvence 应形式化一个 `owner-local Bayesian belief update` 机制，作为 SHADOW 候选，而不是只把 belief/latent/posterior 停留在概念描述。

## 5. 建议

下一步不建议立刻实现全局 BOLT 模块。建议新增一个设计 spec：

```text
docs/specs/owner-local-belief-update.md
```

并以 SHADOW 方式在低风险 owner 中实验：

```text
OwnerPrivatePosterior_t
TypedEvidence_t
PredictionError_t
    -> OwnerPrivatePosterior_t_plus_1
    -> OwnerSnapshot_t_plus_1
```

这个 updater 只能作为 owner 内部算法；跨模块仍然只读 immutable snapshot。

## 6. 文件导览

- `01_clarke_optimization.md`：更新规则训练、二阶启发式与稳定化。
- `02_jmhl_amortised_bayes.md`：摊还推断、函数空间 posterior、不确定性、持续学习。
- `03_boltzbit_founders.md`：Yichuan / Jinli / Boltzbit 的推断算法脉络。
- `04_adjacent_landscape.md`：PFN、Distribution Transformers、latent contexts、Palimpsa 等相邻方向。
- `05_volvence_implications.md`：逐条证伪检查与工程建议。
