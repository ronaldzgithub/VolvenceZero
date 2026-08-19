# 来源、版本与下载边界

> 快照日期：2026-08-19（Asia/Shanghai）

## 1. 论文与发表信息

- 论文页面：<https://arxiv.org/abs/2512.01282>
- arXiv v2 PDF：<https://arxiv.org/pdf/2512.01282>
- WWW 2026 proceedings 索引（页码 9230–9240）：
  <https://researchr.org/publication/www-2026>
- DBLP persistent record：<https://dblp.org/rec/conf/www/YuanCWGZN26>
- 第一作者主页：<https://jhcircle.github.io/>

本地归档：

- `papers/kardia-r1-arxiv-2512.01282v2.pdf`
- 16 页；PDF metadata 标识 `arXivID=https://arxiv.org/abs/2512.01282v2`
- SHA-256：`43be1231748e2cf9ba4c2160a7643f7ce4c175fd86ebb3354b5a261faab5cefa`
- PDF metadata license：`CC BY 4.0`

## 2. 代码

- GitHub：<https://github.com/JhCircle/Kardia-R1>
- 审计 checkout：`ee8d3be788c96c8811311410553414598237d1c5`
- commit date：2026-03-03 01:41:28 +0800
- license：MIT

非图片文件：

```text
README.md
infer_hf.py
infer_swift.py
LICENSE
```

图片：

```text
assets/Kardia-R1-Radar.png
assets/Kardia-R1.png
assets/Kardia-R1-Logo.png
```

没有公开训练、data synthesis、reward、evaluation、split 或配置代码。

## 3. 模型

- 模型卡：<https://huggingface.co/Jhcircle/Kardia-R1>
- 文件清单：<https://huggingface.co/Jhcircle/Kardia-R1/tree/main>
- base：Qwen2.5-7B-Instruct
- card license：MIT
- gated：需要登录并接受访问条件
- 权重：4 个 BF16 safetensors 分片，页面列出的大小约
  `4.88 + 4.93 + 4.33 + 1.09 = 15.23 GB`

本包没有接受 gated conditions、下载权重或运行推理。

## 4. 数据集

- dataset card：<https://huggingface.co/datasets/Jhcircle/KardiaBench>
- README：<https://huggingface.co/datasets/Jhcircle/KardiaBench/blob/main/README.md>
- 文件清单：<https://huggingface.co/datasets/Jhcircle/KardiaBench/tree/main>
- card license：`CC BY-NC-ND 4.0`
- gated：需要提交用途、分享联系信息并等待批准
- 公开文件大小：`train.jsonl` 约 248 MB；`test.jsonl` 约 31.7 MB
- 官方 license deed：<https://creativecommons.org/licenses/by-nc-nd/4.0/>

本包没有申请、下载、复制或转换数据。论文、代码、模型与数据的许可分别判断，互不覆盖。

## 5. 版本差异与页面问题

- GitHub README 称数据集“open-sourced”，但实际为 gated + NC/ND + research-only；本包统一写
  `gated research release`。
- dataset card 的 load 示例写作 `Jhcircle/KadiaBench`，少了 `r`；正确仓库名为
  `Jhcircle/KardiaBench`。
- dataset card 的四段 tag 示例与论文/模型 special tokens 不一致；必须以真实样本和 tokenizer
  为准。
- 模型卡称 7B，Hub UI 将参数规模显示为 8B；这是 Qwen2.5-7B 约 7.6B 参数的命名/取整差异，
  不视为两套模型。
- GitHub 推理脚本中的 `{{profile}}` / `{{situation}}` 未做变量替换，不能视为完整 persona
  inference pipeline。

## 6. 引用建议

```bibtex
@inproceedings{yuan2026kardia,
  title={Kardia-R1: Unleashing LLMs to Reason toward Understanding and Empathy for Emotional Support via Rubric-as-Judge Reinforcement Learning},
  author={Yuan, Jiahao and Cui, Zhiqing and Wang, Hanqing and Gao, Yuansheng and Zhou, Yucheng and Naseem, Usman},
  booktitle={Proceedings of the ACM Web Conference 2026},
  pages={9230--9240},
  year={2026}
}
```

若只引用当前 PDF 版本，应同时标注 `arXiv:2512.01282v2`，避免把预印本版本与最终会议版
的细节默认视为完全一致。

