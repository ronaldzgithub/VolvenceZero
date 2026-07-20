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
只读加载正式 PASS/BLOCK；该 verdict 不进入学习链。

App 的控制面只能 pause/resume/step/speed 或排队环境扰动，不能直接提交
`turn_command` / `step_command`。
