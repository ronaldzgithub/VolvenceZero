# Multi-Session Chat v0.1 语料准入与许可结论

> Review date: 2026-08-01
> Decision: **仅准入非商业内部研究；禁止随仓库再分发，商业训练/产品使用待单独法律确认。**

## 选择结论

主语料选择 Multi-Session Chat（MSC）v0.1。官方项目页和论文明确其为
human-human、跨多个 session 的开放域角色对话；官方 tarball 可由 ParlAI 的
`parlai/tasks/msc/build.py` 下载，归档 SHA-256 为
`e640e37cf4317cd09fc02a4cd57ef130a185f23635f4003b0cee341ffcb45e60`。

LoCoMo 不进入主证据集。其官方仓库与论文说明对话由两个 LLM agents 通过
machine-human pipeline 生成，且仓库许可证是 CC BY-NC 4.0；它可用于合成
长程 sanity check，但不满足“未来标签来自真人对话”的硬条件。

## 冻结划分

保持发布者原始身份隔离，不重新随机打散 dyad：

| Research split | Official source | Dyads | Sessions | File SHA-256 |
|---|---|---:|---:|---|
| train | `session_4/train.txt` | 1001 | 4 | `0094d6374541e3b8e690e1b5b2f56d6121d268db41257d81b7afae3d647d453c` |
| validation | `session_5/valid.txt` | 500 | 5 | `8bb7a4bd621c9dcd0692a2ad4ec5171dc62a419e168767a80731fe00923faebb` |
| heldout | `session_5/test.txt` | 501 | 5 | `8807776265cc3bde441d18b4864449753f874bb8a06aaea352a519925b054d46` |

完整 id 集哈希与下载入口冻结在
`packages/companion-bench/src/companion_bench/corpora/msc_v0_1_manifest.json`。
测试集只用于最终评估；容量、超参数、停止轮次和阈值只能由 train/validation
决定。

## 许可边界

- ParlAI 下载代码标注 MIT，但代码许可证不自动构成对语料内容的商业授权。
- 官方 MSC 压缩包没有单独附带可确认的商业数据许可；session 1 又复用了
  PersonaChat 来源。为避免把代码许可误当数据许可，本仓库采取更严格口径：
  **noncommercial research only**。
- 原始 tar、解压 JSONL、逐句缓存和可逆文本衍生物全部位于
  `data/external/msc/`，由 `.gitignore` 排除。
- 可提交的只有 loader、来源/校验和、不可逆聚合指标、模型参数和不含原文的
  prediction artifact。
- 对外论文需按 MSC 论文要求引用；任何商业训练、产品部署、原文再分发或向
  第三方服务上传原文，必须取得数据权利人许可并完成独立隐私/条款审查。

这是一项工程准入判断，不替代法律意见。

## 方法学威胁

MSC 的后续 session 是 crowdworker 扮演持续角色，论文明确指出后续 session
不一定由同一批真实个人继续完成。因此本数据可检验“身份连续的真人文本是否
可预测”，不能单独证明自然关系亲密度随时间增长，也不能支持“用户感到被
记住”的产品主张。

## 权威来源

- MSC official project: https://parl.ai/projects/msc/
- ParlAI MSC downloader and checksum: https://github.com/facebookresearch/ParlAI/blob/main/parlai/tasks/msc/build.py
- MSC paper: https://aclanthology.org/2022.acl-long.356/
- LoCoMo official repository and license: https://github.com/snap-research/locomo
- LoCoMo paper: https://aclanthology.org/2024.acl-long.747/
