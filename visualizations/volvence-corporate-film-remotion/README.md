# Volvence Corporate Film

110 秒、1920×1080、30fps 的企业宣传片。

```bash
npm install
npm run voiceover
npm run bgm
npm run typecheck
npm run render
```

`npm run voiceover` 通过 `xiaozhi-esp32-server/batch_pitch_tts.py`
调用豆包“大模型温暖男声”生成正式分段配音，并自动对齐为 110 秒总轨。可通过
`XIAOZHI_SERVER_ROOT` 和 `XIAOZHI_TTS_PYTHON` 覆盖默认路径。

远程 TTS 暂时不可用时，可执行 `npm run voiceover:preview` 生成明确标记的
macOS 本地男声预览轨。服务恢复后重新执行 `npm run voiceover` 和
`npm run render`，即可替换配音，无需修改画面。
