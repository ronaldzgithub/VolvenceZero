# Volvence Core Remotion Demo

这个目录是 Volvence 核心学习机制的 Remotion 演示视频工程。它不修改任何 runtime owner、snapshot contract 或 package wheel，只把现有 spec 里的机制组织成一个可预览、可渲染的视频。

当前主 composition 是“出轨场景认知透视版”：左侧播放用户与 Volvence 的完整情感对话，右侧用一张常驻的立体认知架构同步展示任务/关系双轨状态、全景问题建模、主动探索、抽象决策、残差流控制、连续记忆、结果归因与有界更新。它的目标不是做 PPT 式概念介绍，而是让观众在一段真实问题中看见 Volvence 如何共同建模、共同探索并共同收敛。

## 内容

- `src/ContinuousLearningMechanism.tsx`: “时间抽象决策 × 嵌套式多时间尺度学习 × 主动学习”的持续学习闭环 composition
- `src/DialogueCaseMechanism.tsx`: 当前主视频 composition
- `src/data/dialogueCase.ts`: 逐句语音时间轴与五阶段认知状态
- `src/VolvenceCoreMechanism.tsx`: 第一版机制讲解 composition，保留作参考
- `src/data/storyboard.ts`: 第一版分镜数据，保留作参考
- `src/styles.css`: 视频视觉样式

## 使用

```bash
npm install
npm run dev
npm run render
npm run render:continuous
```

输出视频默认生成到：

```text
out/volvence-affair-mechanism.mp4
out/volvence-continuous-learning.mp4
```

## 叙事结构

1. 先稳住：承接冲击并确认用户当前安全。
2. 找出问题：从“我要离婚”中主动探索孩子、经济与共同生活的真实冲突。
3. 共同推演：构造可逆的阶段性选项，不替用户作最终决定。
4. 补齐事实：区分公开研究、概率判断与必须交给专业人士核验的边界。
5. 共同收敛：用户形成自己的判断，系统帮助它落到行动；真实结果进入记忆与策略候选。
