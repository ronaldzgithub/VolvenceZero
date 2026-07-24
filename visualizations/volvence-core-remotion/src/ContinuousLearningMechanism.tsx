import React from 'react';
import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';

export const CONTINUOUS_LEARNING_FPS = 30;
export const CONTINUOUS_LEARNING_DURATION_FRAMES = 2520;

type Scene = {
  key: string;
  start: number;
  end: number;
  eyebrow: string;
  title: string;
  subtitle: string;
};

type Point = {x: number; y: number};

const scenes: Scene[] = [
  {
    key: 'thesis',
    start: 0,
    end: 300,
    eyebrow: 'Unified continual learning',
    title: 'Volvence 把三种学习机制并成一个闭环',
    subtitle:
      '时间抽象决策负责当下行动，嵌套式多时间尺度学习负责经验沉淀，主动学习负责稀疏反馈下的高价值取样。',
  },
  {
    key: 'temporal',
    start: 300,
    end: 660,
    eyebrow: 'Temporal abstraction decision',
    title: '先判断如何行动，再调整内部推理方向',
    subtitle:
      '系统从残差流识别任务、用户与关系状态，压缩为低维策略代码，并在倾听、探索、推演、执行、关系修复之间延续或切换。',
  },
  {
    key: 'nested',
    start: 660,
    end: 1020,
    eyebrow: 'Nested multi-timescale learning',
    title: '经验以不同频率留下来',
    subtitle:
      '即时状态、会话经验、长期认知与模型能力分层更新；快速适应不重写基底，慢速沉淀不阻塞交互。',
  },
  {
    key: 'active',
    start: 1020,
    end: 1380,
    eyebrow: 'Active learning',
    title: '反馈稀疏时，系统主动选择向谁学',
    subtitle:
      '根据不确定性、信息增益和决策影响，挑出最值得询问、验证或调教的少量关键样本。',
  },
  {
    key: 'loop',
    start: 1380,
    end: 1800,
    eyebrow: 'Prediction-error closure',
    title: '执行结果通过预测误差回流到学习系统',
    subtitle:
      '模型形成预测、选择策略并干预残差流；真实结果回来后，PE 触发信用分配，再分别写入控制器、记忆与离线更新候选。',
  },
  {
    key: 'shift',
    start: 1800,
    end: 2160,
    eyebrow: 'System shift',
    title: '从静态回答工具，变成持续复盘的模型系统',
    subtitle:
      'Volvence 解决静态预训练、人工闭环、稀疏反馈、跨场景迁移和 prompt 难以稳定表达关系状态的问题。',
  },
  {
    key: 'safety',
    start: 2160,
    end: 2520,
    eyebrow: 'Bounded evolution',
    title: '在线冻结基底，进化经过边界与回滚',
    subtitle:
      '在线阶段只更新有界控制器、策略和记忆；更重的模型参数更新进入离线验证、审核与回滚路径。',
  },
];

const clamp01 = (value: number) => Math.max(0, Math.min(1, value));

const sceneProgress = (frame: number, scene: Scene) =>
  clamp01((frame - scene.start) / (scene.end - scene.start));

const activeScene = (frame: number) =>
  scenes.find((scene) => frame >= scene.start && frame < scene.end) ??
  scenes[scenes.length - 1];

const useSceneFade = (scene: Scene) => {
  const frame = useCurrentFrame();
  return interpolate(
    frame,
    [scene.start, scene.start + 26, scene.end - 28, scene.end],
    [0, 1, 1, 0],
    {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'},
  );
};

const Shell: React.FC<{scene: Scene; children: React.ReactNode}> = ({
  scene,
  children,
}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const intro = spring({frame, fps, config: {damping: 38, stiffness: 100}});
  const p = sceneProgress(frame, scene);

  return (
    <AbsoluteFill className="clm-shell">
      <div className="clm-grid" />
      <div
        className="clm-field field-a"
        style={{transform: `rotate(${p * 18}deg) scale(${1 + p * 0.03})`}}
      />
      <div
        className="clm-field field-b"
        style={{transform: `rotate(${-p * 24}deg)`}}
      />
      <header
        className="clm-header"
        style={{
          opacity: interpolate(intro, [0, 1], [0, 1]),
          transform: `translateY(${interpolate(intro, [0, 1], [-18, 0])}px)`,
        }}
      >
        <div className="clm-eyebrow">{scene.eyebrow}</div>
        <h1>{scene.title}</h1>
        <p>{scene.subtitle}</p>
      </header>
      {children}
      <footer className="clm-timeline">
        {scenes.map((item) => {
          const current = frame >= item.start && frame < item.end;
          const seen = frame >= item.end;
          return (
            <div
              className={
                current
                  ? 'clm-timeline-step active'
                  : seen
                    ? 'clm-timeline-step seen'
                    : 'clm-timeline-step'
              }
              key={item.key}
            >
              <span />
              <b>{item.eyebrow}</b>
            </div>
          );
        })}
      </footer>
    </AbsoluteFill>
  );
};

const Reveal: React.FC<{
  children: React.ReactNode;
  delay: number;
  progress: number;
}> = ({children, delay, progress}) => {
  const opacity = interpolate(progress, [delay, delay + 0.14], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const y = interpolate(progress, [delay, delay + 0.14], [28, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  return (
    <div style={{opacity, transform: `translateY(${y}px)`}}>{children}</div>
  );
};

const PillarScene: React.FC<{scene: Scene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const opacity = useSceneFade(scene);
  const p = sceneProgress(frame, scene);
  const pillars = [
    {
      label: '时间抽象决策',
      question: '当下如何行动',
      detail: '残差流 → z_t 策略代码 → 有界控制信号',
    },
    {
      label: '嵌套式多时间尺度学习',
      question: '经验如何留下',
      detail: 'online-fast / session-medium / background-slow / rare-heavy',
    },
    {
      label: '主动学习',
      question: '稀疏反馈向谁学',
      detail: '不确定性 × 信息增益 × 决策影响',
    },
  ];

  return (
    <div className="clm-scene clm-pillar-scene" style={{opacity}}>
      <div className="clm-core">
        <i />
        <i />
        <i />
        <div>
          <strong>持续学习机制</strong>
          <span>perceive · decide · learn · evolve</span>
        </div>
      </div>
      <div className="clm-pillar-grid">
        {pillars.map((pillar, index) => (
          <Reveal key={pillar.label} delay={0.12 + index * 0.13} progress={p}>
            <div className="clm-pillar-card">
              <span>{String(index + 1).padStart(2, '0')}</span>
              <strong>{pillar.label}</strong>
              <b>{pillar.question}</b>
              <p>{pillar.detail}</p>
            </div>
          </Reveal>
        ))}
      </div>
    </div>
  );
};

const TemporalDecisionScene: React.FC<{scene: Scene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const opacity = useSceneFade(scene);
  const p = sceneProgress(frame, scene);
  const local = frame - scene.start;
  const strategies = ['倾听', '探索', '推演', '执行', '关系修复'];
  const active = Math.floor(local / 58) % strategies.length;
  const zValues = Array.from({length: 18}).map((_, index) =>
    0.42 + 0.42 * Math.sin(local / 19 + index * 0.7),
  );

  return (
    <div className="clm-scene clm-temporal-scene" style={{opacity}}>
      <div className="clm-residual">
        <strong>模型残差流</strong>
        {Array.from({length: 10}).map((_, index) => (
          <i
            key={index}
            style={{
              width: `${48 + 38 * Math.abs(Math.sin(local / 24 + index))}%`,
              opacity: interpolate(p, [0.06 + index * 0.025, 0.2 + index * 0.025], [0.18, 1], {
                extrapolateLeft: 'clamp',
                extrapolateRight: 'clamp',
              }),
            }}
          />
        ))}
        <span>任务状态 · 用户状态 · 关系状态</span>
      </div>
      <div className="clm-arrow-band">
        <span>压缩</span>
        <i />
        <span>解码</span>
      </div>
      <div className="clm-code-panel">
        <div className="clm-code-title">
          <strong>z_t</strong>
          <span>低维策略代码</span>
        </div>
        <div className="clm-z-bars">
          {zValues.map((value, index) => (
            <i key={index} style={{height: `${value * 100}%`}} />
          ))}
        </div>
        <div className="clm-beta">
          <span>β_t switch gate</span>
          <b>{(0.22 + 0.64 * Math.abs(Math.sin(local / 46))).toFixed(2)}</b>
        </div>
      </div>
      <div className="clm-strategy-ring">
        {strategies.map((strategy, index) => {
          const angle = (index / strategies.length) * Math.PI * 2 - Math.PI / 2;
          const point: Point = {
            x: 250 + Math.cos(angle) * 205,
            y: 250 + Math.sin(angle) * 170,
          };
          return (
            <div
              className={
                index === active
                  ? 'clm-strategy-item active'
                  : 'clm-strategy-item'
              }
              key={strategy}
              style={{left: point.x, top: point.y}}
            >
              {strategy}
            </div>
          );
        })}
        <div className="clm-strategy-center">
          <strong>抽象动作</strong>
          <span>延续 / 切换</span>
        </div>
      </div>
      <div className="clm-control-signal">
        <strong>U_t</strong>
        <span>直接调节内部推理方向</span>
      </div>
    </div>
  );
};

const NestedLearningScene: React.FC<{scene: Scene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const opacity = useSceneFade(scene);
  const p = sceneProgress(frame, scene);
  const lanes = [
    ['online-fast', '每轮更新', '控制器 / 快速记忆'],
    ['session-medium', '每会话更新', '场景模式 / 策略偏好'],
    ['background-slow', '会话后异步', '反思、沉淀、压缩'],
    ['rare-heavy', '离线低频', 'artifact 训练与审核'],
  ] as const;

  return (
    <div className="clm-scene clm-nested-scene" style={{opacity}}>
      <div className="clm-timescale-stack">
        {lanes.map(([name, cadence, detail], index) => {
          const pulse = ((frame - scene.start + index * 24) % (34 + index * 42)) /
            (34 + index * 42);
          return (
            <Reveal key={name} delay={index * 0.1} progress={p}>
              <div className="clm-timescale-lane">
                <div>
                  <strong>{name}</strong>
                  <span>{cadence}</span>
                </div>
                <div className="clm-timescale-track">
                  {Array.from({length: 9}).map((_, marker) => (
                    <em key={marker} />
                  ))}
                  <i style={{left: `${pulse * 100}%`}} />
                </div>
                <b>{detail}</b>
              </div>
            </Reveal>
          );
        })}
      </div>
      <div className="clm-freeze-stack">
        <div className="clm-freeze-base">
          <strong>冻结基础模型</strong>
          <span>live 默认稳定</span>
        </div>
        <div className="clm-adapt-band">有界控制器</div>
        <div className="clm-adapt-band">策略与记忆</div>
        <div className="clm-adapt-band muted">离线参数候选</div>
      </div>
    </div>
  );
};

const ActiveLearningScene: React.FC<{scene: Scene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const opacity = useSceneFade(scene);
  const p = sceneProgress(frame, scene);
  const samples = [
    ['用户意图边界不清', 0.92, 0.74, 0.86, '询问'],
    ['业务结果影响大', 0.67, 0.82, 0.94, '验证'],
    ['关系状态预测分歧', 0.81, 0.76, 0.78, '询问'],
    ['重复失败路径', 0.58, 0.64, 0.72, '调教'],
    ['低影响表述偏差', 0.32, 0.25, 0.21, '跳过'],
  ] as const;

  return (
    <div className="clm-scene clm-active-scene" style={{opacity}}>
      <div className="clm-sample-board">
        {samples.map(([name, uncertainty, gain, impact, action], index) => {
          const score = (uncertainty + gain + impact) / 3;
          return (
            <div
              className={index < 3 ? 'clm-sample selected' : 'clm-sample'}
              key={name}
              style={{
                opacity: interpolate(p, [0.08 + index * 0.07, 0.23 + index * 0.07], [0, 1], {
                  extrapolateLeft: 'clamp',
                  extrapolateRight: 'clamp',
                }),
              }}
            >
              <div className="clm-sample-head">
                <strong>{name}</strong>
                <b>{action}</b>
              </div>
              <div className="clm-sample-bars">
                <span>不确定性<i style={{width: `${uncertainty * 100}%`}} /></span>
                <span>信息增益<i style={{width: `${gain * 100}%`}} /></span>
                <span>决策影响<i style={{width: `${impact * 100}%`}} /></span>
              </div>
              <em>{score.toFixed(2)}</em>
            </div>
          );
        })}
      </div>
      <div className="clm-learning-signal">
        <span>真实业务结果</span>
        <span>必要人类反馈</span>
        <strong>高价值学习信号</strong>
      </div>
    </div>
  );
};

const ClosedLoopScene: React.FC<{scene: Scene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const opacity = useSceneFade(scene);
  const p = sceneProgress(frame, scene);
  const steps = [
    '感知目标',
    '形成预测',
    '选择策略',
    '干预残差流',
    '执行观察',
    '预测误差',
    '信用分配',
    '经验写入',
  ];
  const active = Math.floor((frame - scene.start) / 34) % steps.length;

  return (
    <div className="clm-scene clm-loop-scene" style={{opacity}}>
      <div className="clm-loop">
        <svg viewBox="0 0 900 620" className="clm-loop-lines">
          <ellipse cx="450" cy="310" rx="345" ry="210" />
        </svg>
        {steps.map((step, index) => {
          const angle = (index / steps.length) * Math.PI * 2 - Math.PI / 2;
          const point: Point = {
            x: 450 + Math.cos(angle) * 345,
            y: 310 + Math.sin(angle) * 210,
          };
          return (
            <div
              className={index === active ? 'clm-loop-node active' : 'clm-loop-node'}
              key={step}
              style={{
                left: point.x,
                top: point.y,
                opacity: interpolate(p, [index * 0.055, index * 0.055 + 0.18], [0.22, 1], {
                  extrapolateLeft: 'clamp',
                  extrapolateRight: 'clamp',
                }),
              }}
            >
              {step}
            </div>
          );
        })}
        <div className="clm-loop-center">
          <strong>PE-first</strong>
          <span>自然交互 → 可审计学习</span>
        </div>
      </div>
      <div className="clm-write-targets">
        {['快速控制器', '长期记忆', '离线模型更新候选'].map((target, index) => (
          <div
            key={target}
            style={{
              opacity: interpolate(p, [0.52 + index * 0.1, 0.68 + index * 0.1], [0, 1], {
                extrapolateLeft: 'clamp',
                extrapolateRight: 'clamp',
              }),
            }}
          >
            {target}
          </div>
        ))}
      </div>
    </div>
  );
};

const SystemShiftScene: React.FC<{scene: Scene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const opacity = useSceneFade(scene);
  const p = sceneProgress(frame, scene);
  const problems = [
    ['静态预训练依赖', '持续复盘'],
    ['学习闭环靠人工', '自然交互成证据'],
    ['稀疏反馈难利用', '主动挑选关键样本'],
    ['经验难迁移', '抽象策略跨场景复用'],
    ['prompt 难稳态', '内部控制表达关系状态'],
  ] as const;

  return (
    <div className="clm-scene clm-shift-scene" style={{opacity}}>
      <div className="clm-before-after">
        <div className="clm-shift-label">现有大模型</div>
        <div className="clm-shift-label volvence">Volvence</div>
        {problems.map(([before, after], index) => (
          <React.Fragment key={before}>
            <Reveal delay={index * 0.075} progress={p}>
              <div className="clm-before">{before}</div>
            </Reveal>
            <Reveal delay={0.16 + index * 0.075} progress={p}>
              <div className="clm-after">{after}</div>
            </Reveal>
          </React.Fragment>
        ))}
      </div>
      <div className="clm-system-line">
        <strong>静态回答工具</strong>
        <i />
        <strong>持续决策 · 持续复盘 · 持续积累</strong>
      </div>
    </div>
  );
};

const SafetyScene: React.FC<{scene: Scene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const opacity = useSceneFade(scene);
  const p = sceneProgress(frame, scene);
  const gateStages = ['候选更新', '离线验证', '人工/门控审核', '回滚点', '受控导入'];

  return (
    <div className="clm-scene clm-safety-scene" style={{opacity}}>
      <div className="clm-online-boundary">
        <strong>在线阶段</strong>
        <span>基础模型保持冻结</span>
        <div>有界控制器</div>
        <div>策略状态</div>
        <div>连续记忆</div>
      </div>
      <div className="clm-offline-gate">
        {gateStages.map((stage, index) => (
          <React.Fragment key={stage}>
            <div
              className={index === 2 ? 'clm-gate-stage central' : 'clm-gate-stage'}
              style={{
                opacity: interpolate(p, [0.18 + index * 0.08, 0.33 + index * 0.08], [0.2, 1], {
                  extrapolateLeft: 'clamp',
                  extrapolateRight: 'clamp',
                }),
              }}
            >
              {stage}
            </div>
            {index < gateStages.length - 1 ? <i className="clm-gate-link" /> : null}
          </React.Fragment>
        ))}
      </div>
      <div className="clm-final-claim">
        <strong>持续进化</strong>
        <span>同时保留稳定性、安全性与可回滚证据</span>
      </div>
    </div>
  );
};

export const ContinuousLearningMechanism: React.FC = () => {
  const frame = useCurrentFrame();
  const scene = activeScene(frame);

  return (
    <Shell scene={scene}>
      <PillarScene scene={scenes[0]} />
      <TemporalDecisionScene scene={scenes[1]} />
      <NestedLearningScene scene={scenes[2]} />
      <ActiveLearningScene scene={scenes[3]} />
      <ClosedLoopScene scene={scenes[4]} />
      <SystemShiftScene scene={scenes[5]} />
      <SafetyScene scene={scenes[6]} />
    </Shell>
  );
};
