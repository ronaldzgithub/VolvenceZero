# Character Residual Adapter Package（Deprecated）

`CharacterResidualAdapterPackage` 曾用于把 reviewed live-through 轨迹编译成 target
model residual vectors。该 schema 现在仅为历史 artifact 的可读、SHADOW 审计和
回滚兼容保留，不再是可晋升载体。

废弃原因：统一 `CharacterPackageManifest` 已用 Prefix/KV（轻量档）和可选 PEFT
Character LoRA（重量档）覆盖表达层需求。继续保留独立 residual 通道会形成第四条
没有统一 fidelity/gate 链的注入路径，破坏 L2 单一 package owner。

硬约束：

- 禁止为新角色 bake 或发布新的 residual package；
- 禁止 `CharacterResidualAdapterPackage` 进入 ACTIVE；
- 禁止与统一 character manifest 同时装配；
- `character_residual_applied` 仅用于读取历史 evidence，不构成晋升依据；
- rollback 是不传该 artifact，历史 JSON/schema 暂不删除，以便旧证据可复现。

新实现与迁移流程见 [character-prefix-package.md](./character-prefix-package.md)。
