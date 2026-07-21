# Digital Ant Realtime Lab

启动 Python API：

```bash
pip install -e "packages/vz-embodiment-ant[app]"
digital-ant-app --host 127.0.0.1 --port 8765
```

开发前端：

```bash
cd packages/vz-embodiment-ant/web
npm install
npm run dev
```

Vite 会把 `/api` 代理到 `127.0.0.1:8765`。生产构建用 `npm run build`；
Python server 会自动托管 `web/dist`。可用
`--evidence-artifact research/ant/results/pipeline_summary.json`
只读加载正式 PASS/BLOCK；该 verdict 不进入学习链。已晋级的三物体 checkpoint 另用：

```bash
digital-ant-app \
  --ecology-checkpoint-report research/ant/results/ecology_checkpoint.v4.json
```

loader 会验证 manifest、archive sha256、`ecology-v2` 感知、latent dim、蚂蚁数和 owner
fingerprint；BLOCK candidate 或不兼容 archive 会直接拒绝启动。

App 的控制面只能 pause/resume/step/speed 或排队环境扰动，不能直接提交
`turn_command` / `step_command`。在“三物体生态”目标下，点击放置黄油/燃烧火柴，拖拽定义木棍，
选择后可移动或删除；Canvas 只渲染 `AppFrame.objects` 中环境 owner 发布的不可变快照。
