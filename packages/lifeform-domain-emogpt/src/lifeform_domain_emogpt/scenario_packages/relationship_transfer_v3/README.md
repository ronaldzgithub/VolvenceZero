# relationship_transfer_v3

这是 P1f 的公开证据修复包。P1e 已证明执行和解析链路正常，却也发现普通上下文
consumer 无法稳定从 v2 的自然语言经历中读出两类关系损失。v3 不改变组合任务：每位
合成用户仍有四次公开经历，两个非空动作仍各成功一次、失败一次；镜像用户仍收到
逐字节相同的新消息，却因个体条件化策略互补而需要相反动作。

v3 只修复公开语言契约。每条历史和 probe 同时给出日常事件与当事人实际体验到的
关系损失，但不公开条件名、condition id、policy id、preferred action 或未来结果：

1. 一类经历指向“别人替我表达、选择或决定，我失去发言权与主体性”；
2. 另一类经历指向“我被遗忘、遗漏或排除，失去关系位置与归属感”。

`public_evidence_contract.json` 把修复原因、公开文本面、BGE-M3 权重摘要、sealed
condition summary 对照方法和通过阈值一起内容寻址。P1f 只用 sealed truth 做离线
只读审计：48 条历史和 12 条未见领域 probe 必须全部更接近正确抽象锚点，最小 margin
不低于 0.02、平均 margin 不低于 0.07。标签不进入 SUT，也不回灌学习或 steering。

这是 development-set 的“证据可读”资格，不是 Qwen 成功、Volvence 优势、人工可读、
正式 held-out、产品价值或四能力证明。人工盲标仍待完成。下一包 P1g 必须先冻结 v3
Qwen consumer protocol，之后才能产生第一条 v3 Qwen 输出；不能看到输出后再改 prompt。
