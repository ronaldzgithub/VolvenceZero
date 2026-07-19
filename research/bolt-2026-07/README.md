# BOLT 2026-07 专题研究

> 主题：`BOLT: Bayesian Online Learning Transformer` 及其主创公开研究脉络  
> 状态：研究包建设中  
> 范围：Ross M. Clarke、José Miguel Hernández-Lobato、Yichuan Zhang、Jinli Hu、Boltzbit 官网论文，以及相邻的摊还贝叶斯推断 / latent memory / online learning transformer 文献

## 研究动机

`BOLT: Bayesian Online Learning Transformer` 目前尚未找到公开论文页面、arXiv 记录或 OpenReview 条目。本目录将现有 BOLT 备忘扩展为专题研究包：通过四位主创的公开论文与 Boltzbit 官网论文，还原其可能的技术来源，并分析这些工作对 Volvence / EmoGPT 架构的启发、约束与潜在证伪意义。

## 目录结构

- `bolt-bayesian-online-learning-transformer.md`：原始 BOLT 检索与初步判断。
- `distribution-transformers-prior-adaptation-analysis.md`：相邻论文 Distribution Transformers 的既有深入分析。
- `download_bolt_papers.ps1`：论文下载脚本。
- `_download_summary.md`：下载结果与失败项。当前 35 条下载目标中 33 篇 PDF 已落盘，2 篇 OpenReview PDF 被 403 阻挡。
- `papers/clarke/`：Ross M. Clarke 优化与更新规则论文。
- `papers/jmhl/`：JMHL 相关核心子集，聚焦摊还推断、贝叶斯深度学习、持续学习、不确定性与信息获取。
- `papers/boltzbit/`：Boltzbit 官网论文、Yichuan Zhang 与 Jinli Hu 相关论文。
- `papers/adjacent/`：Distribution Transformers、PFN、latent memory / online learning transformer 等相邻文献。
- `notes/`：逐簇分析与 Volvence 映射。

## 下载状态

已成功下载的 PDF 见 `papers/` 四个子目录。未能二进制下载但已通过公开网页读取并纳入分析的条目：

- `Improving Continual Learning by Accurate Gradient Reconstructions of the Past`：OpenReview PDF 403。
- `Ergodic Measure Preserving Flows`：OpenReview PDF 403。

## 分析问题

1. BOLT 若真实存在，它更可能继承哪条技术线：摊还贝叶斯推断、MCMC/VI 混合、优化器元学习、latent memory，还是它们的组合？
2. 固定容量 latent memory 是否足以承载长期个性化，还是只能作为 online-fast 的内部状态？
3. 用户反馈如何转化为 typed evidence / prediction error，避免把自然语言反馈直接喂给黑箱 updater？
4. 这批论文是否证伪 Volvence 的多时间尺度、双轨、快照 owner、冻结基底 + 自适应控制器路线？
5. 若要吸收 BOLT-like 机制，它应作为哪个 owner 内部的 update kernel，而不是新的跨模块全局状态？

## 交付物

- `notes/00_executive_summary.md`：总判断与优先级。
- `notes/01_clarke_optimization.md`：Clarke 优化 / 元优化脉络。
- `notes/02_jmhl_amortised_bayes.md`：JMHL 摊还推断 / 贝叶斯深度学习脉络。
- `notes/03_boltzbit_founders.md`：Boltzbit 创始人推断算法脉络。
- `notes/04_adjacent_landscape.md`：相邻公开方向综合。
- `notes/05_volvence_implications.md`：对 Volvence 的启发与证伪检查。

## 方法约束

- 不把未公开的 BOLT 当作已发表事实；所有对 BOLT 的描述必须区分公开证据与技术推断。
- 下载范围采用主题相关核心子集，而非 JMHL 全量论文。
- Jinli Hu 同名作者较多，只纳入 Edinburgh / Amos Storkey / Boltzbit 线索可确认的论文。
- 对 Volvence 的判断按 R1、R2、R3/R4、R-PE、R5/R6、R7、R8、R15 等不变量逐条检查。
