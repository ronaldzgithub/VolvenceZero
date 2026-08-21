# relationship_transfer_v4

P1h 把已经看过 Qwen 输出的 `relationship_transfer_v3` 明确降级为
`consumer_training_only`，并把本包冻结为 `unseen_qualification_only`。本包在
`consumer_split_contract.json` 内容寻址之前没有产生任何 v4 Qwen 输出。

资格集包含 12 组、24 位合成镜像用户。每位用户有四条平衡的行动—结果经历；两个
非空动作都各成功一次、失败一次。每组两位用户收到逐字节相同的新消息，但互补的
个体策略使正确动作相反。v4 的十二种 surface family、scene/event id 和全部公开文本
均与 v3 精确隔离；跨包 loader 会 fail loudly。

P1h 不调 prompt、不训练 consumer、不运行 Qwen，也不打开 P2 或 formal hidden test。
下一包 P1i 只能在 v3 training 上最多保留三轮 consumer 候选，并按
leave-one-surface-family-out 选择后冻结；随后才能由独立包一次性运行 v4 qualification。
training 标签只可校准外部 baseline，qualification 标签及任何 evaluation 都不得进入
memory、PE、credit、reward 或 steering。
