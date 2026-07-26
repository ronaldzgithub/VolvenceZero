# Volvence Three-Act Film

独立的三幕二维 Remotion 技术短片：

1. 当前聊天模型与任务型 Agent 的共同断点
2. State KV、抽象决策、预测误差、嵌套记忆与主动取证闭环
3. 共享模型、私有 Profile 与批量部署

## Commands

```bash
npm install
npm run typecheck
npm run voiceover
npm run still
npm run render
```

旁白由 `xiaozhi-esp32-server/batch_pitch_tts.py` 生成。修改
`src/data/voiceover.json` 后执行 `npm run voiceover -- --force` 可重新合成。
