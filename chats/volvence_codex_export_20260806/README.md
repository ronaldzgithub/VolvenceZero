# 导出说明

这是 2026-08-06T13:37:36.643Z 对 Codex 本地 Volvence 项目聊天所做的时间点快照，共 65 个线程。全部线程均保留可见 Markdown；包内保留 64 份原始 JSONL gzip，另有 1 份因压缩后超过 100,000,000 字节而不随包收录。

筛选条件：会话 `session_meta.payload.cwd` 规范化后必须严格等于 `/Users/mengfu/Documents/GitHub/volvence`。数据源同时覆盖 `~/.codex/sessions` 与 `~/.codex/archived_sessions`。此外，线程 ID 必须登记在 `~/.codex/session_index.jsonl` 中，以排除不出现在 Codex 聊天列表中的内部评审或派生运行。

## 目录

- `INDEX.md`：线程总索引与可读文件入口。
- `transcripts/*.md`：仅保留用户消息与 Codex 可见回复；自动心跳只保留 `NOTIFY` 的通知正文，排除触发指令及 `DONT_NOTIFY` 静默响应，同时排除系统/开发者指令、推理、工具调用和工具输出。
- `raw/active/*.jsonl.gz`：随包收录的活动线程完整原始事件快照。
- `raw/archived/*.jsonl.gz`：随包收录的已归档线程完整原始事件快照。
- `manifest.json`：来源、计数、大小及逐线程哈希。
- `SHA256SUMS`：导出包内文件的完整性哈希（不包含清单自身）。

## 完整性验证

在本目录执行：

```bash
shasum -a 256 -c SHA256SUMS
```

`manifest.json` 中的 `rawSourceSha256` 是压缩前原始 JSONL 字节的哈希；解压后可再次核对。

## 原始备份体积边界

线程 `019f8d8c-e4c7-70e1-9860-8a5fc2a359d0` 的可见 Markdown 完整保留，但其原始 gzip 为 143,058,052 字节，不随本包收录。该原始文件在生成时的 SHA-256、源文件 SHA-256、字节数和本机保留路径仍记录于 `manifest.json`；本机 `artifacts` 中的两份硬链接备份以及 Codex 原始会话源均未删除。

这项省略不影响阅读可见对话，也不影响 Volvence 代码或运行任务；它只意味着单独复制或克隆本目录时，该线程没有内置 raw 副本。

## 隐私与边界

原始 JSONL 不等于“仅聊天文本”：它还可能包含系统/开发者上下文、推理记录、工具调用、终端输出、本地文件路径以及其他敏感信息。公开分享时只使用经过人工检查的 `transcripts/`，不要直接发送 `raw/`。

图片、音频等二进制附件没有另外复制；可见对话中只记录附件数量，原始事件仍保留附件引用。活动线程在快照完成后继续产生的新消息不在本包内。
