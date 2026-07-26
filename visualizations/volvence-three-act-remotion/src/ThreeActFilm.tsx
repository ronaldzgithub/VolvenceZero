import React from 'react';
import {
  AbsoluteFill,
  Audio,
  Sequence,
  interpolate,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import {
  acts,
  FILM_DURATION_FRAMES,
  FILM_FPS,
  type Act,
} from './data/film';
import {
  communicationBySequence,
  doubaoAdviceBySequence,
  doubaoDialogue,
  type DialogueUtterance,
  volvenceDialogue,
} from './data/dialogue';

export {FILM_DURATION_FRAMES, FILM_FPS};

const clamp = {
  extrapolateLeft: 'clamp' as const,
  extrapolateRight: 'clamp' as const,
};

const fadeWindow = (
  frame: number,
  start: number,
  end: number,
  fade = 24,
) =>
  interpolate(
    frame,
    [start, start + fade, end - fade, end],
    [0, 1, 1, 0],
    clamp,
  );

const appear = (frame: number, at: number, duration = 28) =>
  interpolate(frame, [at, at + duration], [0, 1], clamp);

const ActHeader: React.FC<{act: Act; localFrame: number}> = ({
  act,
  localFrame,
}) => {
  const lift = interpolate(localFrame, [0, 32], [28, 0], clamp);
  const opacity = interpolate(localFrame, [0, 24], [0, 1], clamp);
  return (
    <div
      className="act-header"
      style={{opacity, transform: `translateY(${lift}px)`}}
    >
      <div className="act-index">{act.index}</div>
      <div>
        <div className="act-eyebrow">{act.eyebrow}</div>
        <h1>{act.title}</h1>
        <p>{act.subtitle}</p>
      </div>
    </div>
  );
};

const SceneChrome: React.FC<{
  act: Act;
  localFrame: number;
  children: React.ReactNode;
}> = ({act, localFrame, children}) => {
  const outro = interpolate(
    localFrame,
    [act.end - act.start - 28, act.end - act.start],
    [1, 0],
    clamp,
  );
  return (
    <AbsoluteFill className={`film scene-${act.key}`} style={{opacity: outro}}>
      <div className="ambient ambient-a" />
      <div className="ambient ambient-b" />
      <div className="depth-grid" />
      <div className="film-brand">
        <span className="brand-mark">V</span>
        <span>VOLVENCE</span>
      </div>
      <div className="act-progress">
        {acts.map((item) => (
          <span
            key={item.key}
            className={item.key === act.key ? 'active' : ''}
          />
        ))}
      </div>
      {children}
    </AbsoluteFill>
  );
};

const Message: React.FC<{
  side: 'user' | 'ai';
  children: React.ReactNode;
  style?: React.CSSProperties;
}> = ({side, children, style}) => (
  <div className={`message ${side}`} style={style}>
    <span>{side === 'user' ? '用户' : '模型'}</span>
    {children}
  </div>
);

const CurrentModelProblem: React.FC<{frame: number}> = ({frame}) => {
  const laneSpring = spring({
    frame: frame - 110,
    fps: FILM_FPS,
    config: {damping: 15, mass: 0.8},
  });
  const chatPulse = interpolate(
    Math.sin(frame / 16),
    [-1, 1],
    [0.4, 1],
  );
  const humanOpacity = appear(frame, 580, 30);

  return (
    <div
      className="problem-world world-3d"
      style={{
        opacity: appear(frame, 80),
        transform: `translateY(${80 - laneSpring * 80}px)`,
      }}
    >
      <div className="problem-lane chat-lane">
        <div className="lane-label">
          <b>聊天</b>
          <span>追随最后一句</span>
        </div>
        <div className="chat-stack">
          <Message side="user">我要立刻做决定。</Message>
          <Message side="ai">那就果断行动。</Message>
          <Message side="user">但风险太大了。</Message>
          <Message
            side="ai"
            style={{
              boxShadow: `0 0 ${26 * chatPulse}px rgba(255,122,89,.34)`,
            }}
          >
            那还是谨慎一点。
          </Message>
        </div>
        <div className="fault-tag">判断随上下文摆动</div>
      </div>

      <div className="problem-lane agent-lane">
        <div className="lane-label">
          <b>任务型 Agent</b>
          <span>会调用，不会归因</span>
        </div>
        <div className="tool-chain">
          {['规划', '搜索', '调用', '提交'].map((item, index) => (
            <React.Fragment key={item}>
              <div
                className="tool-node"
                style={{
                  opacity: appear(frame, 210 + index * 55),
                }}
              >
                {item}
              </div>
              {index < 3 ? (
                <div
                  className={`tool-link ${index === 2 ? 'broken' : ''}`}
                  style={{opacity: appear(frame, 240 + index * 55)}}
                />
              ) : null}
            </React.Fragment>
          ))}
        </div>
        <div className="result-card bad">
          <span>结果</span>
          任务失败
        </div>
        <div className="fault-tag">不知道哪一步导致失败</div>
      </div>

      <div className="manual-bridge" style={{opacity: humanOpacity}}>
        <div className="human-orbit">
          <div className="human-core">人</div>
          {['写提示', '做标注', '改流程', '再训练'].map((item, index) => (
            <span
              key={item}
              style={{
                transform: `rotate(${index * 90}deg) translateX(116px) rotate(${
                  -index * 90
                }deg)`,
              }}
            >
              {item}
            </span>
          ))}
        </div>
        <div className="manual-caption">
          <b>闭环在模型外</b>
          <span>模型负责生成，人负责学习</span>
        </div>
      </div>
    </div>
  );
};

const ProblemAct: React.FC = () => {
  const frame = useCurrentFrame();
  const act = acts[0];
  const headerFade = fadeWindow(frame, 0, 250, 28);
  const verdict = fadeWindow(frame, 720, 1060, 28);
  return (
    <SceneChrome act={act} localFrame={frame}>
      <div style={{opacity: headerFade}}>
        <ActHeader act={act} localFrame={frame} />
      </div>
      <CurrentModelProblem frame={frame} />
      <div className="problem-verdict" style={{opacity: verdict}}>
        <span>共同根因</span>
        <strong>状态是临时上下文，结果没有进入模型自己的学习闭环</strong>
      </div>
    </SceneChrome>
  );
};

const ModelStack: React.FC<{pulse: number}> = ({pulse}) => (
  <div className="model-stack-2d">
    <div
      className="model-stack-glow"
      style={{opacity: interpolate(pulse, [-1, 1], [0.2, 0.72])}}
    />
    <div className="model-layer-2d base-layer-2d">
      <div>
        <span>共享基底模型</span>
        <b>14.5B</b>
      </div>
      <div className="capability-chips">
        <i>语言</i>
        <i>知识</i>
        <i>工具</i>
      </div>
    </div>
    <div className="model-layer-2d adapt-layer-2d">
      <div>
        <span>VOLVENCE</span>
        <b>共享自适应模型</b>
      </div>
      <small>读取残差 · 形成抽象 · 有界控制</small>
    </div>
  </div>
);

const StateHydration: React.FC<{frame: number}> = ({frame}) => {
  const opacity = fadeWindow(frame, 80, 1040, 35);
  const stream = ((frame - 100) % 130) / 130;
  const labels = ['用户', '关系', '目标', '边界', '对象', '环境'];
  return (
    <div className="state-zone" style={{opacity}}>
      <div className="state-source">
        <div className="source-title">已审计状态</div>
        <div className="profile-orbit">
          {labels.map((item, index) => {
            const angle = (Math.PI * 2 * index) / labels.length + frame / 260;
            return (
              <span
                key={item}
                style={{
                  transform: `translate(${Math.cos(angle) * 105}px, ${
                    Math.sin(angle) * 66
                  }px)`,
                }}
              >
                {item}
              </span>
            );
          })}
          <div className="profile-core">PROFILE</div>
        </div>
      </div>
      <div className="state-compiler">
        <span>状态编译器</span>
        <b>结构化状态 → 神经条件</b>
        <i style={{transform: `translateX(${stream * 100 - 50}px)`}} />
      </div>
      <div className="state-kv-gate">
        <div className="kv-plane key-plane">
          <span>KEY</span>
          何时关注
        </div>
        <div className="kv-plane value-plane">
          <span>VALUE</span>
          带入什么
        </div>
        <div className="pre-token">第一个字之前</div>
      </div>
      <div className="fact-channel">
        <b>精确事实</b>
        <span>引用 · 权限 · 血缘</span>
      </div>
    </div>
  );
};

const ResidualAbstraction: React.FC<{frame: number}> = ({frame}) => {
  const opacity = fadeWindow(frame, 620, 1740, 40);
  const local = frame - 620;
  const pulse = Math.sin(frame / 18);
  const particles = Array.from({length: 18}, (_, index) => {
    const progress = ((local * 0.9 + index * 42) % 420) / 420;
    const y = Math.sin(index * 1.7 + frame / 21) * 56;
    return (
      <i
        key={index}
        className="residual-particle"
        style={{
          left: `${progress * 100}%`,
          top: `calc(50% + ${y}px)`,
          opacity: Math.sin(progress * Math.PI),
          transform: `scale(${0.7 + progress * 0.7})`,
        }}
      />
    );
  });
  const actionLabels = ['探索', '验证', '安抚', '切换', '执行'];
  return (
    <div className="abstraction-zone" style={{opacity}}>
      <div className="model-core">
        <ModelStack pulse={pulse} />
      </div>
      <div className="residual-tunnel">
        {particles}
        <span className="tunnel-label">连续残差流</span>
      </div>
      <div className="abstraction-reactor">
        <div className="reactor-rings">
          <div className="reactor-ring ring-a" />
          <div className="reactor-ring ring-b" />
          <div className="z-core">
            <span>z</span>
            低维抽象空间
          </div>
          {actionLabels.map((item, index) => {
            const angle = (index / actionLabels.length) * Math.PI * 2 + frame / 180;
            return (
              <b
                key={item}
                style={{
                  transform: `translate(${Math.cos(angle) * 148}px, ${
                    Math.sin(angle) * 92
                  }px)`,
                }}
              >
                {item}
              </b>
            );
          })}
        </div>
        <div className="switch-control">
          <span className="keep">保持</span>
          <i
            style={{
              left: `${interpolate(
                Math.sin(frame / 28),
                [-1, 1],
                [14, 76],
              )}%`,
            }}
          />
          <span className="switch">切换</span>
        </div>
      </div>
      <div className="bounded-steering">
        <span>有界残差控制</span>
        抽象动作重新进入模型
      </div>
    </div>
  );
};

const OutcomeLoop: React.FC<{frame: number}> = ({frame}) => {
  const opacity = fadeWindow(frame, 1420, 2280, 35);
  const local = frame - 1420;
  const progress = ((local % 150) + 150) % 150 / 150;
  const steps = [
    ['行动前', '预期结果'],
    ['环境', '真实结果'],
    ['差值', '预测误差'],
    ['归因', '信用分配'],
    ['更新', '抽象策略'],
  ];
  return (
    <div className="outcome-loop" style={{opacity}}>
      <div className="loop-track" />
      {steps.map(([tag, title], index) => {
        const angle = (index / steps.length) * Math.PI * 2 - Math.PI / 2;
        return (
          <div
            key={title}
            className={`loop-node loop-node-${index}`}
            style={{
              left: `${50 + Math.cos(angle) * 39}%`,
              top: `${50 + Math.sin(angle) * 38}%`,
            }}
          >
            <span>{tag}</span>
            <b>{title}</b>
          </div>
        );
      })}
      <i
        className="loop-pulse"
        style={{
          left: `${50 + Math.cos(progress * Math.PI * 2 - Math.PI / 2) * 39}%`,
          top: `${50 + Math.sin(progress * Math.PI * 2 - Math.PI / 2) * 38}%`,
        }}
      />
      <div className="pe-core">
        <span>PREDICTION ERROR</span>
        第一学习信号
      </div>
    </div>
  );
};

const MemoryTower: React.FC<{frame: number}> = ({frame}) => {
  const opacity = fadeWindow(frame, 1980, 2760, 38);
  const pulse = interpolate(Math.sin(frame / 20), [-1, 1], [0.25, 1]);
  return (
    <div className="memory-system" style={{opacity}}>
      <div className="memory-tower">
        <div className="memory-band band-fast">
          <span>快层</span>
          <b>当前子目标</b>
          <i style={{opacity: pulse}} />
        </div>
        <div className="memory-band band-session">
          <span>中层</span>
          <b>策略组合</b>
          <i style={{opacity: 1 - pulse * 0.5}} />
        </div>
        <div className="memory-band band-slow">
          <span>慢层</span>
          <b>跨任务先验</b>
          <i style={{opacity: 0.6 + pulse * 0.3}} />
        </div>
        <div className="tower-title">嵌套记忆</div>
      </div>
      <div className="memory-reset">
        <span>慢层重建快层</span>
        <div className="reset-rail">
          <i style={{top: `${80 - pulse * 62}%`}} />
        </div>
      </div>
      <div className="memory-output output-state">
        <span>出口 1</span>
        重建下一轮状态键值
      </div>
      <div className="memory-output output-policy">
        <span>出口 2</span>
        初始化抽象控制器
      </div>
      <div className="memory-arrow arrow-state" />
      <div className="memory-arrow arrow-policy" />
      <div className="memory-thesis">
        <b>不只是记得发生过什么</b>
        <span>还学习下次应该怎么做</span>
      </div>
    </div>
  );
};

const ActiveEvidence: React.FC<{frame: number}> = ({frame}) => {
  const opacity = fadeWindow(frame, 2480, 2860, 26);
  const local = frame - 2480;
  const scan = ((local % 120) + 120) % 120 / 120;
  return (
    <div className="evidence-system" style={{opacity}}>
      <div className="decision-fork">
        <div className="fork-origin">下一步决策</div>
        <div className="fork-path path-a">继续执行</div>
        <div className="fork-path path-b">改变策略</div>
        <div className="uncertainty-cloud">
          <span>?</span>
          关键证据不足
        </div>
      </div>
      <div className="evidence-radar">
        <div
          className="radar-sweep"
          style={{transform: `rotate(${scan * 360}deg)`}}
        />
        {['不确定性', '信息价值', '决策影响', '不可逆风险'].map(
          (item, index) => (
            <span
              key={item}
              style={{
                transform: `rotate(${index * 90}deg) translateX(112px) rotate(${
                  -index * 90
                }deg)`,
              }}
            >
              {item}
            </span>
          ),
        )}
        <b>只问一个<br />关键问题</b>
      </div>
      <div className="evidence-return">
        <span>用户 · 工具 · 环境</span>
        高价值证据回到预测误差闭环
      </div>
    </div>
  );
};

type LoopPhase = 'overview' | 'state' | 'abstraction' | 'outcome' | 'memory' | 'evidence';

const loopPhaseWindows: Array<{
  key: LoopPhase;
  start: number;
  end: number;
  index: string;
  caption: string;
}> = [
  {key: 'overview', start: 0, end: 174, index: '00', caption: '一张图看清模型内部闭环'},
  {key: 'state', start: 174, end: 680, index: '01', caption: '推理前自动装载复杂状态'},
  {key: 'abstraction', start: 680, end: 1260, index: '02', caption: '从残差流自动形成抽象'},
  {key: 'outcome', start: 1260, end: 1700, index: '03', caption: '用真实结果修正抽象策略'},
  {key: 'memory', start: 1700, end: 2365, index: '04', caption: '记忆同时回流状态与抽象控制'},
  {key: 'evidence', start: 2365, end: 2880, index: '05', caption: '只获取最能改变判断的证据'},
];

const phaseAt = (frame: number): LoopPhase => {
  for (const phase of loopPhaseWindows) {
    if (frame >= phase.start && frame < phase.end) {
      return phase.key;
    }
  }
  return 'evidence';
};

const ArchitectureCard: React.FC<{
  id: string;
  eyebrow: string;
  title: string;
  active: boolean;
  visited: boolean;
  className?: string;
  children: React.ReactNode;
}> = ({id, eyebrow, title, active, visited, className = '', children}) => (
  <div
    className={`arch-card arch-${id} ${active ? 'is-active' : ''} ${
      visited ? 'is-visited' : ''
    } ${className}`}
  >
    <div className="arch-card-head">
      <span>{eyebrow}</span>
      <b>{title}</b>
    </div>
    <div className="arch-card-body">{children}</div>
  </div>
);

const FixedLoopArchitecture: React.FC<{frame: number}> = ({frame}) => {
  const phase = phaseAt(frame);
  const is = (...keys: LoopPhase[]) => keys.includes(phase);
  const hasReached = (key: LoopPhase) => {
    const target = loopPhaseWindows.find((item) => item.key === key);
    return target ? frame >= target.start : false;
  };
  const flow = ((frame % 120) + 120) % 120 / 120;
  const activeEdge = (key: string) => {
    if (phase === 'overview') return false;
    if (phase === 'state') return ['interaction-state', 'state-model'].includes(key);
    if (phase === 'abstraction') return ['state-model', 'model-abstract', 'abstract-action'].includes(key);
    if (phase === 'outcome') return ['abstract-action', 'action-outcome', 'outcome-pe'].includes(key);
    if (phase === 'memory') return ['outcome-pe', 'pe-memory', 'memory-state', 'memory-abstract'].includes(key);
    return ['abstract-evidence', 'evidence-state', 'evidence-pe'].includes(key);
  };

  return (
    <div
      className="fixed-architecture"
      style={{opacity: interpolate(frame, [90, 135], [0, 1], clamp)}}
    >
      <div className="architecture-legend">
        <span className="legend-live">当前旁白</span>
        <span className="legend-path">整体位置始终保留</span>
      </div>

      <svg
        className="architecture-edges"
        viewBox="0 0 1640 610"
        preserveAspectRatio="none"
      >
        <defs>
          <marker id="arrow-muted" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
            <path d="M0,0 L0,6 L7,3 z" fill="rgba(143,214,202,.34)" />
          </marker>
          <marker id="arrow-active" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
            <path d="M0,0 L0,6 L7,3 z" fill="#65d6c4" />
          </marker>
          <marker id="arrow-amber" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
            <path d="M0,0 L0,6 L7,3 z" fill="#ffb45e" />
          </marker>
        </defs>
        {[
          ['interaction-state', 'M270 152 H320'],
          ['state-model', 'M590 152 H640'],
          ['model-abstract', 'M910 152 H960'],
          ['abstract-action', 'M1240 152 H1300'],
          ['action-outcome', 'M1425 242 V350'],
          ['outcome-pe', 'M1320 430 H940'],
          ['pe-memory', 'M680 430 H640'],
          ['memory-state', 'M470 350 C470 300 455 282 455 242'],
          ['memory-abstract', 'M610 350 C730 282 1010 305 1080 242'],
          ['abstract-evidence', 'M1100 242 V350'],
          ['evidence-state', 'M980 430 C760 555 430 540 410 242'],
          ['evidence-pe', 'M980 452 H940'],
        ].map(([key, d]) => {
          const active = activeEdge(key);
          const amber = key.includes('memory') || key.includes('evidence');
          return (
            <path
              key={key}
              d={d}
              className={`architecture-edge ${active ? 'edge-active' : ''}`}
              markerEnd={`url(#${
                active ? (amber ? 'arrow-amber' : 'arrow-active') : 'arrow-muted'
              })`}
              style={{
                strokeDashoffset: active ? -flow * 28 : 0,
              }}
            />
          );
        })}
      </svg>

      <div className="architecture-row-label row-inference">推理链</div>
      <div className="architecture-row-label row-learning">反馈与学习链</div>

      <ArchitectureCard
        id="interaction"
        eyebrow="INPUT"
        title="自然交互"
        active={is('overview', 'state')}
        visited
      >
        <div className="interaction-wave">
          {Array.from({length: 12}, (_, index) => (
            <i
              key={index}
              style={{
                height: `${10 + Math.abs(Math.sin(frame / 11 + index)) * 30}px`,
              }}
            />
          ))}
        </div>
        <p>用户不写画像提示</p>
      </ArchitectureCard>

      <ArchitectureCard
        id="state"
        eyebrow="BEFORE TOKEN 1"
        title="状态键值"
        active={is('state')}
        visited={hasReached('state')}
      >
        <div className="mini-kv">
          <span><b>KEY</b>何时关注</span>
          <span><b>VALUE</b>带入什么</span>
        </div>
        <div className="state-facets">用户 · 关系 · 目标 · 边界 · 对象 · 环境</div>
        <small>精确事实走可审计通道</small>
      </ArchitectureCard>

      <ArchitectureCard
        id="model"
        eyebrow="SHARED WEIGHTS"
        title="共享模型"
        active={is('abstraction')}
        visited={hasReached('abstraction')}
      >
        <div className="mini-model-stack">
          <span><b>基底模型</b>语言 · 知识 · 工具</span>
          <span><b>自适应模型</b>读取连续残差流</span>
        </div>
        <div className="mini-residual">
          {Array.from({length: 8}, (_, index) => (
            <i
              key={index}
              style={{
                left: `${((flow * 100 + index * 17) % 100)}%`,
                opacity: is('abstraction') ? 1 : 0.35,
              }}
            />
          ))}
        </div>
      </ArchitectureCard>

      <ArchitectureCard
        id="abstract"
        eyebrow="LOW-DIMENSIONAL CONTROL"
        title="抽象决策"
        active={is('abstraction', 'outcome')}
        visited={hasReached('abstraction')}
      >
        <div className="mini-z">
          <b>z</b>
          <span>保持 / 切换</span>
        </div>
        <div className="abstract-actions">探索 · 验证 · 安抚 · 执行</div>
        <small>在抽象空间学习，不在词元海洋试错</small>
      </ArchitectureCard>

      <ArchitectureCard
        id="action"
        eyebrow="ACTION"
        title="对话 / 工具"
        active={is('abstraction', 'outcome')}
        visited={hasReached('abstraction')}
      >
        <div className="action-ports">
          <span>回答用户</span>
          <span>执行任务</span>
        </div>
        <small>行动前留下结果预测</small>
      </ArchitectureCard>

      <ArchitectureCard
        id="memory"
        eyebrow="MULTI-RATE MEMORY"
        title="嵌套记忆"
        active={is('memory')}
        visited={hasReached('memory')}
      >
        <div className="mini-memory-bands">
          <span><b>快层</b>当前子目标</span>
          <span><b>中层</b>会话策略组合</span>
          <span><b>慢层</b>跨任务稳定先验</span>
        </div>
        <div className="memory-two-outputs">
          <span>→ 下一轮状态键值</span>
          <span>→ 抽象控制器初始化</span>
        </div>
      </ArchitectureCard>

      <ArchitectureCard
        id="pe"
        eyebrow="LEARNING SIGNAL"
        title="预测误差与归因"
        active={is('outcome', 'memory')}
        visited={hasReached('outcome')}
      >
        <div className="pe-equation">
          <span>预期结果</span>
          <b>≠</b>
          <span>真实结果</span>
        </div>
        <div className="credit-targets">抽象动作 · 状态 · 记忆</div>
        <small>判断究竟应该修正哪里</small>
      </ArchitectureCard>

      <ArchitectureCard
        id="evidence"
        eyebrow="EVIDENCE ACQUISITION"
        title="主动取证"
        active={is('evidence')}
        visited={hasReached('evidence')}
      >
        <div className="evidence-matrix">
          {['不确定性', '信息价值', '决策影响', '不可逆风险'].map((item) => (
            <span key={item}>{item}</span>
          ))}
        </div>
        <b className="best-question">只问最关键的一问</b>
        <small>直接采集证据，不直接更新参数</small>
      </ArchitectureCard>

      <ArchitectureCard
        id="outcome"
        eyebrow="ENVIRONMENT"
        title="真实结果"
        active={is('outcome')}
        visited={hasReached('outcome')}
      >
        <div className="outcome-items">
          <span>对话反应</span>
          <span>工具结果</span>
          <span>任务成败</span>
        </div>
        <small>外部可观察，不是模型自评</small>
      </ArchitectureCard>
    </div>
  );
};

const LoopAct: React.FC = () => {
  const frame = useCurrentFrame();
  const act = acts[1];
  const titleOpacity = fadeWindow(frame, 0, 150, 22);
  return (
    <SceneChrome act={act} localFrame={frame}>
      <div style={{opacity: titleOpacity}}>
        <ActHeader act={act} localFrame={frame} />
      </div>
      <FixedLoopArchitecture frame={frame} />
      <div className="section-caption">
        {loopPhaseWindows.map(({start, end, index, caption}) => (
          <div
            key={index}
            style={{opacity: fadeWindow(frame, start, end, 16)}}
          >
            <span>{index}</span>
            <b>{caption}</b>
          </div>
        ))}
      </div>
    </SceneChrome>
  );
};

const UserProfiles: React.FC<{frame: number}> = ({frame}) => {
  const users = [
    ['A', '新客户', '#ffb45e'],
    ['B', '复购客户', '#65d6c4'],
    ['C', '高风险售后', '#ff7a73'],
    ['D', '企业采购', '#70a7ff'],
  ];
  return (
    <div className="user-profile-stack">
      {users.map(([id, label, color], index) => {
        const flow = ((frame * 0.7 + index * 74) % 340) / 340;
        return (
          <div
            className="deployment-user"
            key={id}
            style={{
              borderColor: color,
              transform: `translate(${index * 8}px, ${index * 74}px)`,
            }}
          >
            <b style={{background: color}}>{id}</b>
            <span>{label}</span>
            <i
              style={{
                background: color,
                left: `${flow * 210}px`,
                opacity: Math.sin(flow * Math.PI),
              }}
            />
          </div>
        );
      })}
      <div className="private-label">每个用户独立</div>
    </div>
  );
};

const InferencePod: React.FC<{
  index: number;
  visible: number;
  active?: boolean;
}> = ({index, visible, active}) => (
  <div
    className={`inference-pod ${active ? 'pod-active' : ''}`}
    style={{
      opacity: visible,
      transform: `translate(${index * 150}px, ${
        index * 32
      }px) scale(${0.92 + visible * 0.08})`,
      zIndex: 3 - index,
    }}
  >
    <div className="pod-cap">GPU NODE {index + 1}</div>
    <div className="pod-layer substrate-layer">
      <span>共享基底</span>
      <b>14.5B</b>
    </div>
    <div className="pod-layer adaptive-layer">
      <span>共享自适应模型</span>
      <b>约 500M</b>
    </div>
    <div className="batch-slots">
      {['A', 'B', 'C', 'D'].map((item) => (
        <span key={item}>{item}</span>
      ))}
    </div>
  </div>
);

const DeploymentAct: React.FC = () => {
  const frame = useCurrentFrame();
  const act = acts[2];
  const titleOpacity = fadeWindow(frame, 0, 180, 24);
  const topology = fadeWindow(frame, 80, 1260, 35);
  const scaleOpacity = appear(frame, 980, 35);
  const conclusion = fadeWindow(frame, 1390, 1870, 28);
  const batchProgress = ((frame - 280) % 180 + 180) % 180 / 180;

  return (
    <SceneChrome act={act} localFrame={frame}>
      <div style={{opacity: titleOpacity}}>
        <ActHeader act={act} localFrame={frame} />
      </div>
      <div
        className="deployment-world world-3d"
        style={{
          opacity: topology,
          transform: 'translateY(20px)',
        }}
      >
        <div className="deployment-floor" />
        <UserProfiles frame={frame} />

        <div className="private-store">
          <div className="store-cyl">
            <span>PRIVATE STATE</span>
          </div>
          <b>个人状态库</b>
          <small>关系 · 承诺 · 目标 · 权限</small>
        </div>

        <div className="request-compiler">
          <span>按请求加载</span>
          <b>Profile → State KV</b>
        </div>

        <div className="batch-conveyor">
          <div className="conveyor-label">
            CONTINUOUS BATCHING
            <span>共享权重，不同状态</span>
          </div>
          {['A', 'B', 'C', 'D', 'A', 'C'].map((item, index) => {
            const position = (batchProgress * 760 + index * 132) % 790;
            return (
              <i
                key={`${item}-${index}`}
                style={{left: `${position}px`}}
              >
                {item}
              </i>
            );
          })}
        </div>

        <div className="pod-cluster">
          <InferencePod index={0} visible={1} active />
          <InferencePod index={1} visible={scaleOpacity} />
          <InferencePod index={2} visible={scaleOpacity} />
          <div className="autoscale-label" style={{opacity: scaleOpacity}}>
            无状态扩容
            <span>1 → N</span>
          </div>
        </div>

        <div className="sales-agent">
          <div className="agent-head">
            <span>AI 销售 Agent</span>
            同一模型，两种行动出口
          </div>
          <div className="agent-outputs">
            <div className="output-chat">
              <span>聊天</span>
              澄清需求 · 建立信任
            </div>
            <div className="output-task">
              <span>任务</span>
              查商品 · 读 CRM · 生成方案
            </div>
          </div>
          <div className="agent-feedback">
            客户反应与工具结果
            <i />
            回到学习闭环
          </div>
        </div>

        <div className="background-learning">
          <span>后台学习</span>
          记忆分片 · 版本门控 · 不阻塞推理
        </div>
      </div>

      <div className="deployment-callouts">
        <div
          className="callout callout-shared"
          style={{opacity: fadeWindow(frame, 300, 1180, 30)}}
        >
          <span>共享</span>
          基底模型 + 自适应模型
        </div>
        <div
          className="callout callout-private"
          style={{opacity: fadeWindow(frame, 220, 920, 30)}}
        >
          <span>隔离</span>
          Profile + 动态 State KV
        </div>
      </div>

      <div className="closing-thesis" style={{opacity: conclusion}}>
        <span>VOLVENCE</span>
        <strong>共享的是能力，隔离的是个人状态</strong>
        <p>一套模型，同时服务大量聊天与任务型 Agent</p>
      </div>
    </SceneChrome>
  );
};

const PRODUCT_INTRO_FRAMES = 90;
const DOUBAO_AUDIO_FRAMES = 2435;
const PRODUCT_TRANSITION_FRAMES = 90;
const VOLVENCE_AUDIO_FRAMES = 4161;
const PRODUCT_DOUBAO_START = PRODUCT_INTRO_FRAMES;
const PRODUCT_VOLVENCE_START =
  PRODUCT_DOUBAO_START + DOUBAO_AUDIO_FRAMES + PRODUCT_TRANSITION_FRAMES;
const PRODUCT_CLOSING_START = PRODUCT_VOLVENCE_START + VOLVENCE_AUDIO_FRAMES;

const currentUtterance = (
  dialogue: DialogueUtterance[],
  time: number,
): DialogueUtterance => {
  let current = dialogue[0];
  for (const utterance of dialogue) {
    if (utterance.start <= time) {
      current = utterance;
    }
  }
  return current;
};

const DialoguePanel: React.FC<{
  title: string;
  accent: 'generic' | 'volvence';
  dialogue: DialogueUtterance[];
  time: number;
}> = ({title, accent, dialogue, time}) => {
  const current = currentUtterance(dialogue, time);
  const visible = dialogue
    .filter((utterance) => utterance.start <= time + 0.12)
    .slice(-2);
  return (
    <div className={`product-dialogue dialogue-${accent}`}>
      <div className="product-dialogue-head">
        <span className="dialogue-avatar">{accent === 'generic' ? 'AI' : 'V'}</span>
        <div>
          <b>{title}</b>
          <small>{accent === 'generic' ? '回答当前一句' : '理解完整处境'}</small>
        </div>
        <i className="dialogue-live">对话中</i>
      </div>
      <div className="product-messages">
        {visible.map((utterance) => {
          const active =
            utterance.sequence === current.sequence &&
            time <= utterance.end + 0.35;
          return (
            <div
              key={utterance.sequence}
              className={`product-message ${utterance.speaker} ${
                active ? 'message-active' : ''
              }`}
            >
              <span>{utterance.speaker === 'user' ? '用户' : title}</span>
              <p>{utterance.text}</p>
              {active ? <i className="speaking-bars"><b /><b /><b /><b /></i> : null}
            </div>
          );
        })}
      </div>
    </div>
  );
};

const GenericAdviceTracker: React.FC<{
  sequence: number;
  frame: number;
}> = ({sequence, frame}) => {
  const adviceEntries = Object.entries(doubaoAdviceBySequence)
    .map(([key, value]) => [Number(key), value] as const)
    .filter(([key]) => key <= sequence);
  const currentAdvice =
    adviceEntries.length > 0
      ? adviceEntries[adviceEntries.length - 1][1]
      : '等待用户补充';
  const options = ['准备离婚', '先谈一谈', '暂时分开', '谨慎决定', '继续等待'];
  const activeIndex = Math.max(0, options.indexOf(currentAdvice));
  const markerLeft = interpolate(activeIndex, [0, options.length - 1], [7, 93]);
  const wobble = Math.sin(frame / 13) * (sequence > 1 ? 3 : 0);
  return (
    <div className="generic-advice">
      <div className="generic-advice-title">
        <span>当前建议</span>
        <b>{currentAdvice}</b>
      </div>
      <div className="advice-rail">
        <i
          style={{
            left: `${markerLeft}%`,
            transform: `translate(-50%, -50%) rotate(${wobble}deg)`,
          }}
        />
        {options.map((option) => (
          <span
            key={option}
            className={option === currentAdvice ? 'active' : ''}
          >
            {option}
          </span>
        ))}
      </div>
      <div className="advice-history">
        {adviceEntries.slice(-4).map(([key, advice]) => (
          <span key={key}>{advice}</span>
        ))}
      </div>
    </div>
  );
};

const panoramaRows = [
  {name: '当下安全', reveal: 2, weight: '20%', scores: ['6', '5', '9', '4']},
  {name: '孩子稳定', reveal: 4, weight: '30%', scores: ['4', '7', '8', '8']},
  {name: '情绪承受', reveal: 4, weight: '30%', scores: ['7', '3', '9', '2']},
  {name: '经济与股权', reveal: 6, weight: '20%', scores: ['4', '6', '7', '7']},
];

const PanoramaBoard: React.FC<{
  sequence: number;
  audioTime: number;
}> = ({sequence, audioTime}) => {
  const options = ['离婚', '协商修复', '暂时分开', '保持现状'];
  const financeResearching = sequence === 8 && audioTime < 105.8;
  const financeResolved = sequence >= 9 || (sequence === 8 && !financeResearching);
  const totals = ['5.3', '5.2', '8.3', '5.2'];
  return (
    <div className="panorama-board">
      <div className="panorama-head">
        <div>
          <span>全景决策支持</span>
          <b>同一个框架，随事实动态修正</b>
        </div>
        <div className="panorama-gap">
          {sequence < 4
            ? '正在确认：安全与事实'
            : sequence < 7
              ? '正在补齐：孩子、收入、资产'
              : sequence < 9
                ? '正在核验：融资与股权'
                : '关键事实已形成可行动判断'}
        </div>
      </div>
      <div className="panorama-grid">
        <div className="panorama-corner">决策维度 <i>重要程度</i></div>
        {options.map((option, index) => (
          <div
            key={option}
            className={`panorama-option ${
              index === 2 && sequence >= 6 ? 'option-leading' : ''
            }`}
          >
            {option}
          </div>
        ))}
        {panoramaRows.map((row, rowIndex) => {
          const visible = sequence >= row.reveal;
          const values =
            rowIndex === 3 && !financeResolved
              ? Array.from({length: 4}, () =>
                  financeResearching ? '研究中' : '待核验',
                )
              : row.scores;
          return (
            <React.Fragment key={row.name}>
              <div
                className={`panorama-dimension ${visible ? 'dimension-clear' : ''}`}
              >
                <b>{row.name}</b>
                <span>{visible ? row.weight : '待澄清'}</span>
              </div>
              {values.map((value, optionIndex) => (
                <div
                  key={`${row.name}-${options[optionIndex]}`}
                  className={`panorama-score ${
                    visible ? 'score-clear' : ''
                  } ${
                    optionIndex === 2 && visible ? 'score-leading' : ''
                  }`}
                >
                  {visible ? value : '·'}
                </div>
              ))}
            </React.Fragment>
          );
        })}
        <div className="panorama-total">
          <b>综合判断</b>
          <span>{sequence >= 9 ? '已收敛' : '持续更新'}</span>
        </div>
        {totals.map((total, index) => (
          <div
            key={`${options[index]}-total`}
            className={`panorama-score score-total ${
              sequence >= 9 ? 'score-clear' : ''
            } ${index === 2 && sequence >= 9 ? 'score-winner' : ''}`}
          >
            {sequence >= 9 ? total : '—'}
          </div>
        ))}
      </div>
      <div
        className={`panorama-evidence ${
          sequence >= 7 ? 'evidence-visible' : ''
        }`}
      >
        <span>{financeResearching ? '自动研究中' : '证据进入决策表'}</span>
        <b>
          {financeResearching
            ? '查询公开融资与行业信息'
            : '融资不等于兑现 · 股份锁定三年 · 法律归属仍需核验'}
        </b>
      </div>
    </div>
  );
};

const CommunicationState: React.FC<{sequence: number}> = ({sequence}) => {
  const state =
    communicationBySequence[sequence] ?? communicationBySequence[1];
  return (
    <div className="communication-state">
      <div className="communication-moves">
        <span>当前沟通</span>
        {state.moves.map((move) => (
          <b key={move}>{move}</b>
        ))}
      </div>
      <div className="relationship-state">
        <span>用户状态</span>
        <b>{state.userState}</b>
      </div>
    </div>
  );
};

const ProductAct: React.FC = () => {
  const frame = useCurrentFrame();
  const act = acts[3];
  const titleOpacity = fadeWindow(frame, 0, 150, 22);
  const doubaoTime = Math.max(0, (frame - PRODUCT_DOUBAO_START) / FILM_FPS);
  const volvenceTime = Math.max(
    0,
    (frame - PRODUCT_VOLVENCE_START) / FILM_FPS,
  );
  const doubaoCurrent = currentUtterance(doubaoDialogue, doubaoTime);
  const volvenceCurrent = currentUtterance(volvenceDialogue, volvenceTime);
  const doubaoOpacity = fadeWindow(
    frame,
    PRODUCT_DOUBAO_START,
    PRODUCT_DOUBAO_START + DOUBAO_AUDIO_FRAMES,
    24,
  );
  const volvenceOpacity = fadeWindow(
    frame,
    PRODUCT_VOLVENCE_START,
    PRODUCT_CLOSING_START,
    24,
  );
  const closingOpacity = fadeWindow(
    frame,
    PRODUCT_CLOSING_START,
    act.end - act.start,
    26,
  );

  return (
    <SceneChrome act={act} localFrame={frame}>
      <div style={{opacity: titleOpacity}}>
        <ActHeader act={act} localFrame={frame} />
      </div>

      <div className="product-stage generic-stage" style={{opacity: doubaoOpacity}}>
        <div className="product-phase-label">
          <span>01</span>
          <div><b>普通对话模型</b><small>用户说什么，就顺着回答什么</small></div>
        </div>
        <DialoguePanel
          title="普通模型"
          accent="generic"
          dialogue={doubaoDialogue}
          time={doubaoTime}
        />
        <GenericAdviceTracker
          sequence={doubaoCurrent.sequence}
          frame={frame}
        />
      </div>

      <div
        className="product-stage volvence-stage"
        style={{opacity: volvenceOpacity}}
      >
        <div className="product-phase-label">
          <span>02</span>
          <div><b>Volvence</b><small>共同建模 · 共同探索 · 共同收敛</small></div>
        </div>
        <DialoguePanel
          title="Volvence"
          accent="volvence"
          dialogue={volvenceDialogue}
          time={volvenceTime}
        />
        <CommunicationState sequence={volvenceCurrent.sequence} />
        <PanoramaBoard
          sequence={volvenceCurrent.sequence}
          audioTime={volvenceTime}
        />
      </div>

      <div className="product-closing" style={{opacity: closingOpacity}}>
        <span>VOLVENCE</span>
        <strong>不是替用户做决定，而是帮用户把选择权拿回来</strong>
        <div>
          {[
            '全景决策支持',
            '高情商教练',
            '关系引导',
            '长期记忆',
            '结果驱动学习',
          ].map((item) => <b key={item}>{item}</b>)}
        </div>
      </div>
    </SceneChrome>
  );
};

export const VolvenceThreeAct: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const globalGlow = interpolate(Math.sin(frame / 34), [-1, 1], [0.55, 1]);
  return (
    <AbsoluteFill
      style={
        {
          '--global-glow': globalGlow,
          backgroundColor: '#061014',
        } as React.CSSProperties
      }
    >
      <Sequence from={acts[0].start} durationInFrames={acts[0].end - acts[0].start}>
        <ProblemAct />
      </Sequence>
      <Sequence from={acts[1].start} durationInFrames={acts[1].end - acts[1].start}>
        <LoopAct />
      </Sequence>
      <Sequence from={acts[2].start} durationInFrames={acts[2].end - acts[2].start}>
        <DeploymentAct />
      </Sequence>
      <Sequence from={acts[3].start} durationInFrames={acts[3].end - acts[3].start}>
        <ProductAct />
      </Sequence>
      <Audio
        src={staticFile('voiceover/volvence-three-act-voiceover.wav')}
        volume={1}
      />
      <Sequence
        from={acts[3].start + PRODUCT_DOUBAO_START}
        durationInFrames={DOUBAO_AUDIO_FRAMES}
      >
        <Audio src={staticFile('dialogue/doubao-conversation.wav')} volume={1} />
      </Sequence>
      <Sequence
        from={acts[3].start + PRODUCT_VOLVENCE_START}
        durationInFrames={VOLVENCE_AUDIO_FRAMES}
      >
        <Audio src={staticFile('dialogue/volvence-conversation.wav')} volume={1} />
      </Sequence>
      <div className="timecode">
        {String(Math.floor(frame / fps / 60)).padStart(2, '0')}:
        {String(Math.floor(frame / fps) % 60).padStart(2, '0')}
      </div>
    </AbsoluteFill>
  );
};
