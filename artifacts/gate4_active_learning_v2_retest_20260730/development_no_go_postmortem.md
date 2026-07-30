# Gate 4 v2 development NO-GO

- `trace-development-heldout` 上五臂、三 seed、共享 24 个 train candidates 与 12 个 evaluation episodes 已执行。
- segment-aware 相对 turn-level、random、shuffled 三个对照的平均标签节省均为 `0.0`。
- segment-aware 的最小最终 balanced-accuracy margin 为 `-0.083333`，未满足非劣门。
- typed candidate lineage、源不可变和 bounded readout isolation 全部通过；失败属于 segment-aware selector 没有产生 causal sample-efficiency 优势。
- 按预注册 GO/NO-GO 纪律停止，不消费 `trace-locked-confirmation`，不调整 selector 权重或标签门。
- 长期主张收缩：继续保留 typed feedback request / owner selector mechanism；“主动学习优势建立在 ETA segment 上”的 causal 主张不成立。
