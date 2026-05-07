# OpenAI 前沿论文 vs VolvenceZero 路线对比（2024-2025）

本目录系统调研 OpenAI 一线研究员（Pachocki / Madry / Zaremba / Baker / Chen / Tworek / Kaiser / Lightman 等）2024-2025 年与"cognitive AGI"主题最相关的 7 篇核心论文，并与本仓库的 NL+ETA 路线（R1–R15+R-PE）做正面对比。

## 目录结构

- `papers/` — 7 篇论文 PDF（已下载，约 32MB）
- `notes/` — 逐篇阅读笔记 + 综合对比报告
- `notes/00_executive_summary.md` — 给 BOSS 看的高密度摘要（先读这份）
- `notes/01_paper_by_paper.md` — 逐篇技术等级评估 + VZ 借鉴点
- `notes/02_route_divergence.md` — OpenAI 路线 vs VZ 路线分歧矩阵

## 论文清单（按 VZ 相关性排序）

| # | Paper | arXiv | 一作 / 关键作者 | 与 VZ 关联面 |
|---|---|---|---|---|
| 1 | Competitive Programming with Large Reasoning Models | 2502.06807 | Pachocki/Tworek/Kaiser/Chen | **核心**：通用 RL > 领域定制 |
| 2 | Deliberative Alignment | 2412.16339 | Guan/Wallace/Chung/Wei | spec-aware 内部审议 |
| 3 | Monitoring Reasoning Models for Misbehavior | 2503.11926 | Baker/Madry/Zaremba/Pachocki | **关键警示**：obfuscated reward hacking |
| 4 | OpenAI o1 System Card | 2412.16720 | OpenAI | 整体技术报告 |
| 5 | Let's Verify Step by Step | 2305.20050 | Lightman/Cobbe/Schulman/Sutskever | PRM 奠基 |
| 6 | CoT Monitorability | 2507.11473 | Korbak/Mark Chen/Pachocki/Madry/Zaremba/Sutskever | **跨实验室共识** |
| 7 | GDPval | 2510.04374 | Patwardhan/Tworek | 真实价值评估范式 |
