import React from 'react';
import {
  AbsoluteFill,
  Audio,
  interpolate,
  Sequence,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import {
  AUDIO_DURATION_SECONDS,
  AUDIO_START_SECONDS,
  dialogueUtterances,
  mechanismPhases,
  MechanismPhase,
  progressSteps,
} from './data/dialogueCase';

type Point = {x: number; y: number};

const clamp = (value: number, min: number, max: number) =>
  Math.max(min, Math.min(max, value));

const currentPhase = (time: number) =>
  mechanismPhases.find((phase) => time >= phase.start && time < phase.end) ??
  mechanismPhases[mechanismPhases.length - 1];

const currentUtterance = (time: number) =>
  [...dialogueUtterances].reverse().find((utterance) => time >= utterance.start) ??
  dialogueUtterances[0];

const phaseProgress = (time: number, phase: MechanismPhase) =>
  clamp((time - phase.start) / (phase.end - phase.start), 0, 1);

const getPointOnRoute = (points: Point[], progress: number): Point => {
  const segmentProgress = progress * (points.length - 1);
  const index = Math.min(Math.floor(segmentProgress), points.length - 2);
  const local = segmentProgress - index;
  return {
    x: points[index].x + (points[index + 1].x - points[index].x) * local,
    y: points[index].y + (points[index + 1].y - points[index].y) * local,
  };
};

const FlowParticle: React.FC<{
  points: Point[];
  progress: number;
  tone?: 'teal' | 'rose' | 'amber' | 'green';
  delay?: number;
}> = ({points, progress, tone = 'teal', delay = 0}) => {
  const shifted = (progress + delay + 1) % 1;
  const point = getPointOnRoute(points, shifted);
  return (
    <i
      className={`affair-flow-particle ${tone}`}
      style={{left: point.x, top: point.y}}
    />
  );
};

const ModuleCard: React.FC<{
  id: string;
  title: string;
  kicker: string;
  detail?: string;
  phase: MechanismPhase;
  className?: string;
  children?: React.ReactNode;
}> = ({id, title, kicker, detail, phase, className = '', children}) => {
  const active = phase.activeModules.includes(id);
  return (
    <section className={`affair-module ${id} ${active ? 'active' : ''} ${className}`}>
      <span>{kicker}</span>
      <strong>{title}</strong>
      {detail ? <p>{detail}</p> : null}
      {children}
    </section>
  );
};

const DialoguePanel: React.FC<{time: number}> = ({time}) => {
  const active = currentUtterance(time);
  const visible = dialogueUtterances
    .filter((utterance) => utterance.start <= time + 0.05)
    .slice(-3);

  return (
    <section className="affair-dialogue-panel">
      <header className="affair-chat-header">
        <div className="affair-avatar">V</div>
        <div>
          <strong>Volvence</strong>
          <span>和用户一起把问题解决</span>
        </div>
        <i />
      </header>

      <div className="affair-chat-body">
        <div className="affair-date">今天 · 一段真实处境</div>
        {visible.map((utterance, index) => {
          const entering = interpolate(
            time,
            [utterance.start, utterance.start + 0.45],
            [0, 1],
            {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'},
          );
          const isActive = utterance.id === active.id;
          return (
            <div
              className={`affair-bubble-row ${utterance.speaker} ${
                isActive ? 'speaking' : ''
              }`}
              key={utterance.id}
              style={{
                opacity: entering * (isActive ? 1 : 0.48 + index * 0.16),
                transform: `translateY(${(1 - entering) * 18}px)`,
              }}
            >
              <div className="affair-speaker">
                {utterance.speaker === 'user' ? '用户' : 'Volvence'}
              </div>
              <div className="affair-bubble">{utterance.text}</div>
            </div>
          );
        })}
      </div>

      <footer className="affair-chat-progress">
        {progressSteps.map((step, index) => {
          const phaseIndex = mechanismPhases.findIndex(
            (phase) => phase.id === currentPhase(time).id,
          );
          return (
            <div
              className={index === phaseIndex ? 'active' : index < phaseIndex ? 'done' : ''}
              key={step}
            >
              <i />
              <span>{step}</span>
            </div>
          );
        })}
      </footer>
    </section>
  );
};

const DualState: React.FC<{phase: MechanismPhase}> = ({phase}) => (
  <div className="affair-dual-state">
    <ModuleCard id="dual" title="任务与事实" kicker="世界状态" phase={phase}>
      <div className="affair-state-lines task">
        {phase.taskState.slice(0, 3).map((item) => (
          <b key={item}>{item}</b>
        ))}
      </div>
    </ModuleCard>
    <ModuleCard id="dual" title="关系与安全" kicker="主体状态" phase={phase}>
      <div className="affair-state-lines relation">
        {phase.relationState.slice(0, 3).map((item) => (
          <b key={item}>{item}</b>
        ))}
      </div>
    </ModuleCard>
  </div>
);

const PanoramaBoard: React.FC<{phase: MechanismPhase}> = ({phase}) => (
  <ModuleCard id="panorama" title="全景问题模型" kicker="共同建模" phase={phase}>
    <div className="affair-panorama-grid">
      {phase.panorama.map((item) => (
        <div className={`affair-panorama-item ${item.tone}`} key={item.label}>
          <span>{item.label}</span>
          <b>{item.state}</b>
        </div>
      ))}
    </div>
  </ModuleCard>
);

const ResidualCore: React.FC<{
  phase: MechanismPhase;
  frame: number;
  speaking: boolean;
}> = ({phase, frame, speaking}) => {
  const pulse = (frame % 70) / 70;
  return (
    <ModuleCard id="residual" title="基底模型" kicker="神经网络内部" phase={phase}>
      <div className="affair-residual-stack">
        {Array.from({length: 7}).map((_, index) => (
          <i
            key={index}
            style={{
              opacity: phase.activeModules.includes('residual') ? 0.48 + index * 0.07 : 0.18,
              transform: `translateX(${Math.sin(frame / 18 + index) * 3}px)`,
            }}
          />
        ))}
        <div
          className={`affair-residual-stream ${speaking ? 'speaking' : ''}`}
          style={{top: `${14 + pulse * 68}%`}}
        />
      </div>
      <div className="affair-control-directions">
        <b>理解</b>
        <b>推理</b>
        <b>表达</b>
      </div>
    </ModuleCard>
  );
};

const MemoryTower: React.FC<{phase: MechanismPhase}> = ({phase}) => (
  <ModuleCard id="memory" title="连续记忆" kicker="什么值得留下" phase={phase}>
    <div className="affair-memory-bands">
      <div><span>当前</span><b>{phase.memory[0]}</b></div>
      <div><span>会话</span><b>{phase.memory[1] ?? '等待更多证据'}</b></div>
      <div><span>长期</span><b>{phase.memory[2] ?? '跨轮验证后沉淀'}</b></div>
    </div>
  </ModuleCard>
);

const LearningLoop: React.FC<{
  phase: MechanismPhase;
  settling: boolean;
}> = ({phase, settling}) => (
  <ModuleCard
    id="learning"
    title="结果驱动学习"
    kicker="真实反馈形成归因"
    phase={{
      ...phase,
      activeModules: settling
        ? [...phase.activeModules, 'learning']
        : phase.activeModules,
    }}
  >
    <div className="affair-learning-chain">
      {['预测', '结果', '误差', '归因'].map((item, index) => (
        <React.Fragment key={item}>
          <b className={settling ? 'active' : ''}>{item}</b>
          {index < 3 ? <i>→</i> : null}
        </React.Fragment>
      ))}
    </div>
    <p>{phase.learning}</p>
  </ModuleCard>
);

const ArchitecturePanel: React.FC<{
  time: number;
  frame: number;
}> = ({time, frame}) => {
  const phase = currentPhase(time);
  const utterance = currentUtterance(time);
  const p = phaseProgress(time, phase);
  const speaking = utterance.speaker === 'assistant' && time <= utterance.end + 0.15;
  const settling =
    utterance.speaker === 'user' && time > 18 && time <= utterance.end + 0.15;
  const flow = ((frame / 92) % 1 + 1) % 1;
  const tiltX = Math.sin(frame / 240) * 0.7;
  const tiltY = Math.cos(frame / 280) * 1.15;

  const inbound: Point[] = [
    {x: 62, y: 106},
    {x: 260, y: 148},
    {x: 380, y: 330},
    {x: 594, y: 365},
  ];
  const outbound: Point[] = [
    {x: 594, y: 365},
    {x: 792, y: 372},
    {x: 1005, y: 120},
  ];
  const feedback: Point[] = [
    {x: 1005, y: 120},
    {x: 815, y: 630},
    {x: 462, y: 652},
    {x: 594, y: 365},
  ];

  return (
    <section className="affair-architecture-panel">
      <header className="affair-architecture-header">
        <div>
          <span>一页看清 Volvence 运作机制</span>
          <h1>{phase.title}</h1>
        </div>
        <strong>{phase.step}</strong>
      </header>

      <div className="affair-question-ribbon">
        <span>当前最有价值的问题</span>
        <strong>{phase.activeQuestion}</strong>
      </div>

      <div className="affair-perspective">
        <div
          className="affair-mechanism-stage"
          style={{
            transform: `rotateX(${1.6 + tiltX}deg) rotateY(${-2.6 + tiltY}deg) translateZ(0)`,
          }}
        >
          <div className="affair-depth-plane plane-a" />
          <div className="affair-depth-plane plane-b" />

          <svg
            className="affair-architecture-lines"
            viewBox="0 0 1100 760"
            preserveAspectRatio="none"
          >
            <defs>
              <linearGradient id="flow-in" x1="0" x2="1">
                <stop offset="0" stopColor="#ef7c68" stopOpacity="0.28" />
                <stop offset="1" stopColor="#54dbc0" stopOpacity="0.9" />
              </linearGradient>
              <linearGradient id="flow-out" x1="0" x2="1">
                <stop offset="0" stopColor="#f2bd58" stopOpacity="0.75" />
                <stop offset="1" stopColor="#54dbc0" stopOpacity="0.38" />
              </linearGradient>
            </defs>
            <path d="M62 106 C210 106 240 142 300 176 C350 214 340 300 380 330 C450 380 520 365 594 365" className="route inbound" />
            <path d="M594 365 C675 350 720 370 792 372 C870 350 900 180 1005 120" className="route outbound" />
            <path d="M1005 120 C980 340 920 530 815 630 C690 704 550 670 462 652 C500 545 550 450 594 365" className={`route feedback ${settling || phase.id === 'converge' ? 'active' : ''}`} />
            <path d="M465 350 C515 280 545 250 594 265" className="route control" />
            <path d="M680 112 C650 180 620 230 594 282" className={`route research ${phase.id === 'research' ? 'active' : ''}`} />
          </svg>

          <FlowParticle points={speaking ? outbound : inbound} progress={flow} tone={speaking ? 'amber' : 'rose'} />
          <FlowParticle points={speaking ? outbound : inbound} progress={flow} delay={0.42} tone="teal" />
          {settling || phase.id === 'converge' ? (
            <>
              <FlowParticle points={feedback} progress={flow} tone="green" />
              <FlowParticle points={feedback} progress={flow} delay={0.5} tone="amber" />
            </>
          ) : null}

          <ModuleCard
            id="input"
            title="原始自然交互"
            kicker="调用方只需要说话"
            detail={utterance.speaker === 'user' ? '新信息进入' : '等待真实结果'}
            phase={phase}
          />

          <DualState phase={phase} />
          <PanoramaBoard phase={phase} />

          <ModuleCard
            id="question"
            title="主动探索"
            kicker="少问，但问对"
            detail={phase.activeQuestion}
            phase={phase}
          />

          <ModuleCard
            id="research"
            title="自动研究"
            kicker="外部证据"
            detail={phase.id === 'research' ? '融资信息可查 · 法律归属转专业核验' : '需要时自动启动'}
            phase={phase}
          />

          <ModuleCard id="decision" title="决策引擎" kicker="下一步做什么" phase={phase}>
            <div className="affair-decision-core">
              <i />
              <i />
              <div>
                <span>当前抽象动作</span>
                <b>{phase.action}</b>
              </div>
            </div>
          </ModuleCard>

          <ResidualCore phase={phase} frame={frame} speaking={speaking} />

          <ModuleCard
            id="output"
            title="自然回应"
            kicker="双商同时在线"
            detail={speaking ? '正在生成' : '等待决策'}
            phase={phase}
          />

          <MemoryTower phase={phase} />
          <LearningLoop phase={phase} settling={settling} />

          <ModuleCard
            id="gate"
            title="验证与回滚"
            kicker="有界更新"
            detail={phase.id === 'converge' ? '策略候选进入验证，不直接污染线上模型' : '持续守护正式模型'}
            phase={phase}
          />

          <div
            className="affair-phase-scan"
            style={{transform: `translateX(${interpolate(p, [0, 1], [-80, 1080])}px)`}}
          />
        </div>
      </div>
    </section>
  );
};

export const DialogueCaseMechanism: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const absoluteTime = frame / fps;
  const time = clamp(absoluteTime - AUDIO_START_SECONDS, 0, AUDIO_DURATION_SECONDS);
  const intro = spring({
    frame,
    fps,
    durationInFrames: Math.round(AUDIO_START_SECONDS * fps),
    config: {damping: 32, stiffness: 92},
  });
  const introOpacity = interpolate(
    absoluteTime,
    [0, 0.5, AUDIO_START_SECONDS - 0.5, AUDIO_START_SECONDS],
    [1, 1, 1, 0],
    {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'},
  );
  const outroStart = AUDIO_START_SECONDS + AUDIO_DURATION_SECONDS;
  const outroOpacity = interpolate(
    absoluteTime,
    [outroStart, outroStart + 1.2],
    [0, 1],
    {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'},
  );

  return (
    <AbsoluteFill className="affair-video">
      <div className="affair-atmosphere" />
      <div className="affair-grid" />

      <main
        className="affair-main-layout"
        style={{
          opacity: interpolate(intro, [0, 1], [0, 1]),
          transform: `scale(${interpolate(intro, [0, 1], [0.985, 1])})`,
        }}
      >
        <DialoguePanel time={time} />
        <ArchitecturePanel time={time} frame={frame} />
      </main>

      <Sequence from={Math.round(AUDIO_START_SECONDS * fps)}>
        <Audio src={staticFile('volvence-conversation.wav')} />
      </Sequence>

      <AbsoluteFill className="affair-intro" style={{opacity: introOpacity}}>
        <div className="affair-intro-mark">
          <i />
          <i />
          <i />
          <strong>Volvence</strong>
        </div>
        <h2>一段对话，打开一个认知系统</h2>
        <p>左边，是用户听见的回应。右边，是模型真正运行的机制。</p>
      </AbsoluteFill>

      <AbsoluteFill
        className="affair-outro"
        style={{opacity: outroOpacity, pointerEvents: 'none'}}
      >
        <div>
          <span>VOLVENCE</span>
          <strong>持续学习型大语言模型</strong>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
