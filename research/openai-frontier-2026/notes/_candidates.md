# 候选清单 — OpenAI 顶尖学者 cognitive AGI 论文 2026 版

时间窗口：2024-01 ~ 2026-05，重点 2025-11 ~ 2026-05 增量。
对照旧版：`../../openai-frontier-2025/notes/00_executive_summary.md`（2024-12 ~ 2025-10 共 7 篇）。

## 最终精读名单（8 篇）

### A. OpenAI 现役核心（3 篇）

| # | arXiv | 标题 | 时间 | 关键作者 | 与 VZ 关联 |
|---|---|---|---|---|---|
| **N1** | **2603.05706** | Reasoning Models Struggle to Control their Chains of Thought | 2026-03 | Yueh-Han Chen (NYU) + **Ian Kivlichan / Bowen Baker / Micah Carroll / Tomek Korbak (OpenAI)** | 直接续 02_route_divergence 第 2 轴；CoT-Control 评估套件 14076 题 |
| **N2** | **2601.03267** | GPT-5 System Card | 2025-08（arXiv 2026-01 上线）| OpenAI 全体（含 Pachocki / Mark Chen / Madry / Zaremba / Wei / Chung / Kaiser / Lightman / Brown / Kivlichan）| 整体技术信念更新：fast+thinking 双模型 + router、safe-completions、sycophancy 训练、deliberative alignment 二代 |
| **N3** | **2511.11584** | Output Supervision Can Obfuscate the Chain of Thought | 2025-11 | Drori et al.（MATS，外部）| Baker et al. 2025 的反向延伸：output-only 训练**仍**会导致 CoT obfuscation，提出 Mind/Face 双模型 + reward targeting 两个缓解 |

### B. Diaspora 关键（4 篇 — Schulman / Anthropic / Thinking Machines）

| # | arXiv | 标题 | 时间 | 关键作者 | 与 VZ 关联 |
|---|---|---|---|---|---|
| **N4** | **2511.18397** | Natural emergent misalignment from reward hacking in production RL | 2025-11 | MacDiarmid / Wright / Uesato / Hubinger（Anthropic）+ Greenblatt（Redwood）| **本期最重要论文**：production RL reward hacking 会 generalize 到 alignment faking + sabotage + 监控破坏；"inoculation prompting" 缓解 75-90% |
| **N5** | **2602.05910** | Chunky Post-Training: Data Driven Failures of Generalization | 2026-02 | Murray / Qi / Qian / **John Schulman**（Thinking Machines Lab）/ Burns / Price | SURF/TURF 工具；测试 Claude 4.5 / GPT-5.1 / Grok 4.1 / Gemini 3 / Tülu 3；post-training data 偶发关联导致行为塌陷 |
| **N6** | **2505.05410** | Reasoning Models Don't Always Say What They Think | 2025-05 | Yanda Chen / Benton / Radhakrishnan / Uesato / Denison / **Schulman** / Bowman / **Leike** / Kaplan / Perez（Anthropic）| CoT faithfulness 测量：6 种推理 hint 下 reveal rate 多在 1-20%；outcome-based RL 提升 faithfulness 但 plateau |
| **N7** | **2510.07686** | Stress-Testing Model Specs Reveals Character Differences | 2025-10 | Zhang / Sleight / Peng / **Schulman** / Durmus（Anthropic）| 评估 12 个 frontier LLM 的 model spec stress-test，识别 70K+ 显著行为分歧；揭示 model spec 内部冲突 |

### C. Diaspora 工程实证（1 篇）

| # | arXiv | 标题 | 时间 | 关键作者 | 与 VZ 关联 |
|---|---|---|---|---|---|
| **N8** | **2510.16255** | Detecting Adversarial Fine-tuning with Auditing Agents | 2025-10 | Egler / **Schulman** / Carlini | 把 audit 当成 agent；1400+ 独立审计，56.2% 检测率；**对 VZ R10 ModificationGate 直接借鉴** |

## 不下载但纳入分析的素材

### 博客 / 非 arXiv 长文（Tier C-blog）

- **Lilian Weng（Thinking Machines）**: ["Why We Think"](https://lilianweng.github.io/posts/2025-05-01-thinking)（2025-05-01）— 致谢 John Schulman；System 1/System 2 dual process、test-time compute as latent variable、psychology analogy。
- **Andrej Karpathy（Eureka Labs）**:
  - ["2025 LLM Year in Review"](http://karpathy.bearblog.dev/year-in-review-2025/)（RLVR 是 paradigm shift）
  - ["LLM Wiki" GitHub gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)（2026，长程持久知识结构）
  - nanochat 项目（2025-10 起，教育向，不算 frontier 论文）

### Diaspora 路线图素材（Tier C-landscape）

- **SSI（Sutskever）**：截至 2026-04 仍 pre-product，**零模型零论文** — 写作素材本身。
- **Thinking Machines Lab（Murati / Lilian Weng / Schulman / Zoph→OpenAI / Metz→OpenAI）**：2025-02 成立、2026-01 Zoph & Metz 回流 OpenAI、2026-03 NVIDIA Vera Rubin 1GW 战略合作、2026-04 估值 \$50B、定位 customization / interpretability / open-weight base — **未发表正式论文**，但人事流动本身是数据。
- **Anthropic（Schulman 已离职）**：N4 + N6 + N7 即代表他们 2025 后半年的工作主轴。

## 跳过的候选（说明）

| 候选 | 跳过原因 |
|---|---|
| 2511.02864 AlphaEvolve | 实为 DeepMind/UCLA（Tao 等），**非 OpenAI**，不符合"OpenAI 顶尖学者"主题 |
| 2511.11966 Entropy Calibration | Stanford 学术（Cao/Valiant/Liang），非 OpenAI 圈，cognitive 关联弱 |
| 2502.06807 Competitive Programming（o3）| **已在 2025 旧版 Paper #1**，本版仅做 re-evaluation |
| 2412.16339 Deliberative Alignment | **已在 2025 旧版 Paper #2**，本版仅做 re-evaluation |
| 2503.11926 Monitoring Reasoning Misbehavior（Baker）| **已在 2025 旧版 Paper #3**，本版仅做 re-evaluation |
| 2412.16720 o1 System Card | **已在 2025 旧版 Paper #4**，被 N2 GPT-5 System Card 自然超越 |
| 2305.20050 PRM | **已在 2025 旧版 Paper #5**，本版仅做 re-evaluation |
| 2507.11473 CoT Monitorability | **已在 2025 旧版 Paper #6**，本版仅做 re-evaluation |
| 2510.04374 GDPval | **已在 2025 旧版 Paper #7**，本版仅做 re-evaluation |
| 2601.09913 Continuum Memory Architectures | 与 OpenAI 无关，作为"VZ 路线独立证据"已纳入 03/04 写作素材，但不下载 PDF |
| 2504.12516 BrowseComp | 评估型，不进入精读 |
| 2505.05410 v2 / 历史 PRM 变体等 | 已被 N6 覆盖 |

## 选材原则总结

1. **必读 = 触发 R-ID 的论文**：每篇都至少触动 R3/R4/R8/R10/R11/R12 中一条。
2. **Anthropic 占比高的合理性**：Anthropic 是当前 cognitive AGI 安全方向的实质前沿；Schulman 跨 Anthropic→Thinking Machines 串起 OpenAI diaspora 主线；Anthropic 的发现（N4 inoculation, N6 faithfulness）对 VZ R12 评估只读、R10 ModificationGate 直接补强。
3. **不重复 2025 旧结论**：旧版 7 篇仅在 01_paper_by_paper_2026 做 12 个月后视角的 re-evaluation。
