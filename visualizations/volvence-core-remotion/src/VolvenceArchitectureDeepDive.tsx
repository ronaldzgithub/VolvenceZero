import React from 'react';
import {
  AbsoluteFill,
  Audio,
  interpolate,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import {
  ARCHITECTURE_DEEP_DIVE_DURATION_FRAMES,
  ARCHITECTURE_DEEP_DIVE_FPS,
  ArchitectureScene,
  architectureScenes,
} from './data/architectureDeepDive';
import './volvence-architecture-deep-dive.css';

export {
  ARCHITECTURE_DEEP_DIVE_DURATION_FRAMES,
  ARCHITECTURE_DEEP_DIVE_FPS,
};

const clamp01 = (value: number) => Math.max(0, Math.min(1, value));

const sceneProgress = (frame: number, scene: ArchitectureScene) =>
  clamp01((frame - scene.start) / (scene.end - scene.start));

const sceneFade = (frame: number, scene: ArchitectureScene) =>
  interpolate(
    frame,
    [scene.start, scene.start + 24, scene.end - 28, scene.end],
    [0, 1, 1, 0],
    {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'},
  );

const easeOut = (value: number) => 1 - Math.pow(1 - clamp01(value), 3);

const pulse = (frame: number, speed = 18, offset = 0) =>
  (Math.sin((frame + offset) / speed) + 1) / 2;

const statusLabel = {
  CURRENT: '当前能力',
  TARGET: '目标架构',
  HYBRID: '当前 + 目标',
} as const;

const Reveal: React.FC<{
  children: React.ReactNode;
  progress: number;
  at?: number;
  className?: string;
  distance?: number;
}> = ({children, progress, at = 0.1, className, distance = 28}) => {
  const local = easeOut((progress - at) / 0.13);
  return (
    <div
      className={className}
      style={{
        opacity: local,
        transform: `translateY(${(1 - local) * distance}px)`,
      }}
    >
      {children}
    </div>
  );
};

const FlowDots: React.FC<{
  frame: number;
  count?: number;
  vertical?: boolean;
  tone?: 'aqua' | 'amber' | 'coral' | 'lime';
  reverse?: boolean;
}> = ({
  frame,
  count = 5,
  vertical = false,
  tone = 'aqua',
  reverse = false,
}) => {
  return (
    <div
      className={`vad-flow-dots ${vertical ? 'vertical' : ''} ${tone}`}
    >
      {Array.from({length: count}).map((_, index) => {
        const raw = ((frame / 34 + index / count) % 1 + 1) % 1;
        const position = reverse ? 1 - raw : raw;
        return (
          <i
            key={index}
            style={
              vertical
                ? {top: `${position * 100}%`}
                : {left: `${position * 100}%`}
            }
          />
        );
      })}
    </div>
  );
};

const MiniVector: React.FC<{
  frame: number;
  count?: number;
  tone?: 'aqua' | 'amber' | 'coral' | 'lime';
  offset?: number;
}> = ({frame, count = 12, tone = 'aqua', offset = 0}) => (
  <div className={`vad-vector ${tone}`}>
    {Array.from({length: count}).map((_, index) => (
      <i
        key={index}
        style={{
          height: `${22 + 58 * Math.abs(Math.sin(frame / 25 + index * 0.73 + offset))}%`,
          opacity: 0.48 + 0.52 * pulse(frame, 22, index * 7 + offset),
        }}
      />
    ))}
  </div>
);

const SceneShell: React.FC<{
  scene: ArchitectureScene;
  children: React.ReactNode;
}> = ({scene, children}) => {
  const frame = useCurrentFrame();
  const p = sceneProgress(frame, scene);
  const opacity = sceneFade(frame, scene);
  const overall = frame / ARCHITECTURE_DEEP_DIVE_DURATION_FRAMES;

  return (
    <AbsoluteFill className="vad-root" style={{opacity}}>
      <div className="vad-noise" />
      <div
        className="vad-aurora vad-aurora-a"
        style={{
          transform: `translate3d(${Math.sin(frame / 110) * 42}px, ${
            Math.cos(frame / 150) * 22
          }px, 0) rotate(${frame / 90}deg)`,
        }}
      />
      <div
        className="vad-aurora vad-aurora-b"
        style={{
          transform: `translate3d(${Math.cos(frame / 130) * 36}px, ${
            Math.sin(frame / 170) * 28
          }px, 0) rotate(${-frame / 110}deg)`,
        }}
      />
      <header className="vad-header">
        <div className="vad-brand">
          <span className="vad-brand-mark">
            <i />
            <i />
            <i />
          </span>
          <strong>VOLVENCE</strong>
          <small>MODEL SYSTEM / ARCHITECTURE DEEP DIVE</small>
        </div>
        <div className={`vad-status ${scene.status?.toLowerCase()}`}>
          <i />
          {scene.status ? statusLabel[scene.status] : '架构解析'}
        </div>
      </header>

      <div className="vad-title-block">
        <div className="vad-chapter">
          <b>{scene.index}</b>
          <span>{scene.chapter}</span>
        </div>
        <h1>{scene.title}</h1>
        <p>{scene.subtitle}</p>
      </div>

      <main
        className="vad-stage"
        style={{
          transform: `translateY(${interpolate(p, [0, 1], [8, -4])}px)`,
        }}
      >
        {children}
      </main>

      <footer className="vad-footer">
        <div className="vad-progress-track">
          <i style={{width: `${overall * 100}%`}} />
        </div>
        <div className="vad-progress-labels">
          {architectureScenes.map((item) => (
            <span
              key={item.key}
              className={
                frame >= item.start && frame < item.end
                  ? 'active'
                  : frame >= item.end
                    ? 'seen'
                    : ''
              }
            >
              {item.index}
            </span>
          ))}
        </div>
      </footer>
    </AbsoluteFill>
  );
};

const OpeningScene: React.FC<{scene: ArchitectureScene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const p = sceneProgress(frame, scene);
  const local = frame - scene.start;
  const coreIn = spring({
    frame: local - 35,
    fps,
    config: {damping: 28, stiffness: 70, mass: 1.1},
  });
  const orbit = ['目标', '感知', '决策', '反馈'];

  return (
    <SceneShell scene={scene}>
      <div className="vad-opening">
        <div
          className="vad-opening-core"
          style={{
            opacity: coreIn,
            transform: `scale(${0.72 + coreIn * 0.28}) rotateX(58deg) rotateZ(${
              local / 18
            }deg)`,
          }}
        >
          <div className="vad-core-rings">
            <i />
            <i />
            <i />
            <b>V</b>
          </div>
        </div>
        <div className="vad-opening-orbit">
          {orbit.map((label, index) => {
            const angle = local / 80 + (index * Math.PI * 2) / orbit.length;
            const radiusX = 430;
            const radiusY = 145;
            return (
              <div
                className="vad-orbit-node"
                key={label}
                style={{
                  opacity: easeOut((p - 0.15 - index * 0.05) / 0.15),
                  transform: `translate(${Math.cos(angle) * radiusX}px, ${
                    Math.sin(angle) * radiusY
                  }px)`,
                }}
              >
                <i />
                <span>{label}</span>
              </div>
            );
          })}
        </div>
        <Reveal progress={p} at={0.52} className="vad-opening-claim">
          <span>PRE-TRAINED INTELLIGENCE</span>
          <b>+</b>
          <span>CONTINUAL ADAPTATION</span>
          <strong>模型自身闭合学习回路</strong>
        </Reveal>
      </div>
    </SceneShell>
  );
};

const GapScene: React.FC<{scene: ArchitectureScene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const p = sceneProgress(frame, scene);
  const local = frame - scene.start;
  const chatStep = Math.min(3, Math.floor(Math.max(0, local - 75) / 82));
  const taskStep = Math.min(3, Math.floor(Math.max(0, local - 105) / 82));
  const chatRows = [
    ['用户', '我想辞职创业。'],
    ['模型', '勇敢追求梦想，我帮你列计划。'],
    ['用户', '但我还有房贷和孩子。'],
    ['模型', '那应该谨慎，稳定可能更重要。'],
  ];
  const taskRows = [
    ['任务', '找到增长原因并执行改进'],
    ['Agent', '搜索 → 汇总 → 发邮件'],
    ['结果', '邮件发了，但问题没有改变'],
    ['缺口', '没有目标状态，也没有结果归因'],
  ];

  return (
    <SceneShell scene={scene}>
      <div className="vad-gap-grid">
        <Reveal progress={p} at={0.08}>
          <section className="vad-gap-panel">
            <div className="vad-panel-label">
              <span>CHAT</span>
              <b>随最后一句话漂移</b>
            </div>
            <div className="vad-dialogue-stack">
              {chatRows.map(([speaker, text], index) => (
                <div
                  className={`vad-dialogue-row ${speaker === '模型' ? 'ai' : ''}`}
                  key={text}
                  style={{
                    opacity: index <= chatStep ? 1 : 0.12,
                    transform: `translateX(${index <= chatStep ? 0 : -18}px)`,
                  }}
                >
                  <span>{speaker}</span>
                  <p>{text}</p>
                </div>
              ))}
            </div>
            <div className="vad-gap-diagnosis coral">
              <i />
              没有稳定的用户、关系与决策状态
            </div>
          </section>
        </Reveal>

        <div className="vad-gap-break">
          <div className="vad-broken-loop">
            {['感知', '决定', '行动', '结果'].map((item, index) => (
              <span
                key={item}
                style={{
                  transform: `rotate(${index * 90}deg) translateY(-76px) rotate(${
                    -index * 90
                  }deg)`,
                }}
              >
                {item}
              </span>
            ))}
            <b style={{transform: `rotate(${local / 9}deg)`}}>×</b>
          </div>
          <small>闭环由人完成</small>
        </div>

        <Reveal progress={p} at={0.17}>
          <section className="vad-gap-panel">
            <div className="vad-panel-label">
              <span>AGENT</span>
              <b>会执行，不会持续形成策略</b>
            </div>
            <div className="vad-task-stack">
              {taskRows.map(([label, text], index) => (
                <div
                  className={`vad-task-row ${index === 3 ? 'warning' : ''}`}
                  key={text}
                  style={{
                    opacity: index <= taskStep ? 1 : 0.12,
                    transform: `translateY(${index <= taskStep ? 0 : 12}px)`,
                  }}
                >
                  <span>{label}</span>
                  <p>{text}</p>
                  {index < taskRows.length - 1 ? <i /> : null}
                </div>
              ))}
            </div>
            <div className="vad-gap-diagnosis amber">
              <i />
              工具轨迹没有变成可复用的决策能力
            </div>
          </section>
        </Reveal>
      </div>
    </SceneShell>
  );
};

const StackScene: React.FC<{scene: ArchitectureScene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const p = sceneProgress(frame, scene);
  const local = frame - scene.start;
  const users = [
    {name: '用户 A', state: '关系修复 · 高风险', x: -460},
    {name: '用户 B', state: '销售推进 · 可逆', x: 0},
    {name: '用户 C', state: '育儿决策 · 低置信', x: 460},
  ];

  return (
    <SceneShell scene={scene}>
      <div className="vad-stack-scene">
        <Reveal progress={p} at={0.05} className="vad-stack-layer shared">
          <div className="vad-stack-side-label">
            <b>共享 / 慢变</b>
            <span>所有用户共用</span>
          </div>
          <div className="vad-stack-slab">
            <div className="vad-stack-item base">
              <span>FROZEN BASE MODEL</span>
              <strong>语言 · 知识 · 通用推理</strong>
              <small>在线冻结 / 极慢更新</small>
              <MiniVector frame={frame} count={22} />
            </div>
            <div className="vad-stack-item adapter">
              <span>SHARED ADAPTER MODEL</span>
              <strong>领域技能 · 稳定策略</strong>
              <small>rare-heavy LoRA / 稀疏专家</small>
            </div>
          </div>
        </Reveal>

        <Reveal progress={p} at={0.2} className="vad-stack-layer dynamic">
          <div className="vad-stack-side-label">
            <b>个体 / 动态</b>
            <span>每次请求加载</span>
          </div>
          <div className="vad-user-rail">
            {users.map((user, index) => {
              const enter = easeOut((p - 0.25 - index * 0.06) / 0.16);
              return (
                <div
                  className="vad-user-pod"
                  key={user.name}
                  style={{
                    left: `calc(50% + ${user.x}px)`,
                    opacity: enter,
                    transform: `translate(-50%, ${40 * (1 - enter)}px)`,
                  }}
                >
                  <div>
                    <b>{user.name}</b>
                    <span>{user.state}</span>
                  </div>
                  <small>State KV · memory refs · consent</small>
                  <i
                    style={{
                      width: `${38 + 48 * pulse(local, 22, index * 40)}%`,
                    }}
                  />
                </div>
              );
            })}
          </div>
        </Reveal>

        <Reveal progress={p} at={0.5} className="vad-stack-layer fast">
          <div className="vad-stack-side-label">
            <b>当轮 / 快变</b>
            <span>每个决策步</span>
          </div>
          <div className="vad-controller-ribbon">
            <span>残差状态 eₜ</span>
            <i>→</i>
            <span>时间代码 zₜ</span>
            <i>→</i>
            <span>βₜ 延续 / 切换</span>
            <i>→</i>
            <strong>有界控制 Uₜ</strong>
            <FlowDots frame={frame} count={7} />
          </div>
        </Reveal>

        <div className="vad-stack-equation">
          <span>不是</span>
          <del>每个用户一套 500M 权重</del>
          <b>而是</b>
          <strong>共享参数 + 每用户小型动态状态</strong>
        </div>
      </div>
    </SceneShell>
  );
};

const HydrationScene: React.FC<{scene: ArchitectureScene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const p = sceneProgress(frame, scene);
  const owners = [
    ['USER', '用户模型'],
    ['REL', '关系状态'],
    ['GOAL', '目标与价值'],
    ['BOUND', '边界与同意'],
    ['BELIEF', '信念与假设'],
    ['TASK', '计划与承诺'],
  ];
  const banks = ['Personal', 'Relationship', 'Object', 'Environment', 'World', 'Task'];

  return (
    <SceneShell scene={scene}>
      <div className="vad-hydration">
        <div className="vad-hydration-column owner-column">
          <div className="vad-column-heading">
            <span>01</span>
            <strong>语义 Owner</strong>
            <small>谁拥有，谁解释</small>
          </div>
          <div className="vad-owner-grid">
            {owners.map(([code, label], index) => (
              <Reveal
                key={code}
                progress={p}
                at={0.05 + index * 0.025}
                className="vad-owner-card"
              >
                <span>{code}</span>
                <b>{label}</b>
                <small>immutable snapshot</small>
              </Reveal>
            ))}
          </div>
        </div>

        <div className="vad-hydration-bridge">
          <FlowDots frame={frame} count={8} tone="aqua" />
          <div className="vad-contract-gate">
            <span>CONTRACT BUS</span>
            <b>版本 · 作用域 · 权限 · 血缘</b>
            <small>runtime 只传播，不重建语义</small>
          </div>
          <FlowDots frame={frame + 40} count={8} tone="lime" />
        </div>

        <div className="vad-hydration-column bank-column">
          <div className="vad-column-heading">
            <span>02</span>
            <strong>Conditioning Banks</strong>
            <small>按问题选择相关状态</small>
          </div>
          <div className="vad-bank-stack">
            {banks.map((bank, index) => {
              const active = (Math.floor((frame - scene.start) / 40) + index) % 4 !== 0;
              return (
                <Reveal
                  key={bank}
                  progress={p}
                  at={0.24 + index * 0.035}
                  className={`vad-bank-card ${active ? 'selected' : ''}`}
                >
                  <i>{String(index + 1).padStart(2, '0')}</i>
                  <b>{bank}</b>
                  <div>
                    <span>confidence</span>
                    <em
                      style={{
                        width: `${active ? 58 + index * 5 : 24}%`,
                      }}
                    />
                  </div>
                </Reveal>
              );
            })}
          </div>
        </div>

        <div className="vad-hydration-final">
          <div className="vad-state-encoder">
            <span>03</span>
            <b>共享 State Encoder</b>
            <MiniVector frame={frame} count={16} tone="lime" />
          </div>
          <FlowDots frame={frame + 80} count={5} tone="lime" />
          <div className="vad-hydration-output">
            <strong>推理前状态</strong>
            <span>首 token 前生效</span>
            <small>不是 hidden system prompt</small>
          </div>
        </div>

        <div className="vad-lineage-strip">
          <span>tenant_scope</span>
          <span>user_scope</span>
          <span>source_versions</span>
          <span>consent_version</span>
          <span>freshness</span>
          <span>fingerprint</span>
          <b>任一变化 → cache 失效</b>
        </div>
      </div>
    </SceneShell>
  );
};

const StateKvScene: React.FC<{scene: ArchitectureScene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const p = sceneProgress(frame, scene);
  const local = frame - scene.start;
  const layers = Array.from({length: 8});
  const selectedBank = Math.floor(local / 70) % 6;
  const bankNames = ['PERSONAL', 'RELATION', 'OBJECT', 'ENV', 'WORLD', 'TASK'];

  return (
    <SceneShell scene={scene}>
      <div className="vad-state-kv">
        <div className="vad-kv-left">
          <div className="vad-kv-fact-channel">
            <span>可审计事实通道</span>
            <b>金额 · 日期 · 条款 · 引用</b>
            <small>保留文本与来源，不压进潜状态</small>
          </div>
          <div className="vad-kv-bank-wheel">
            {bankNames.map((bank, index) => (
              <div
                key={bank}
                className={index === selectedBank ? 'active' : ''}
                style={{
                  transform: `rotate(${index * 60}deg) translateY(-132px) rotate(${
                    -index * 60
                  }deg)`,
                }}
              >
                <i />
                <span>{bank}</span>
              </div>
            ))}
            <strong>STATE<br />ROUTER</strong>
          </div>
        </div>

        <div className="vad-kv-generator">
          <span>Prefix Generator</span>
          <div className="vad-kv-matrix">
            <b>K</b>
            {Array.from({length: 36}).map((_, index) => (
              <i
                key={`k-${index}`}
                style={{
                  opacity: 0.18 + 0.82 * pulse(frame, 16, index * 2),
                }}
              />
            ))}
          </div>
          <div className="vad-kv-matrix value">
            <b>V</b>
            {Array.from({length: 36}).map((_, index) => (
              <i
                key={`v-${index}`}
                style={{
                  opacity: 0.18 + 0.82 * pulse(frame, 19, index * 3),
                }}
              />
            ))}
          </div>
          <small>[batch, kv_heads, prefix_slots, d_head]</small>
        </div>

        <div className="vad-transformer-cutaway">
          <div className="vad-token-entry">
            <span>本轮 token</span>
            <FlowDots frame={frame} vertical count={5} tone="amber" />
          </div>
          <div className="vad-layer-stack">
            {layers.map((_, index) => {
              const readWave = ((local / 20 - index * 0.8) % 8 + 8) % 8;
              const lit = readWave < 1.8;
              return (
                <div
                  className={`vad-transformer-layer ${lit ? 'reading' : ''}`}
                  key={index}
                  style={{
                    transform: `translate(${index * 13}px, ${-index * 4}px)`,
                  }}
                >
                  <span>L{String(index + 1).padStart(2, '0')}</span>
                  <div className="vad-attention-row">
                    <i className="state-slot" />
                    <i className="state-slot" />
                    <i className="state-slot" />
                    <i className="token-slot" />
                    <i className="token-slot" />
                    <i className="token-slot" />
                    <i className="token-slot" />
                  </div>
                  <b>{lit ? 'READ STATE' : 'ATTENTION'}</b>
                </div>
              );
            })}
          </div>
          <div className="vad-kv-injection">
            <FlowDots frame={frame} count={8} tone="lime" />
            <span>在首个 token 前预置</span>
          </div>
        </div>

        <Reveal progress={p} at={0.58} className="vad-kv-equation">
          <div>
            <span>Kₗ =</span>
            <b>[ K<span>state</span> ; K<span>text</span> ]</b>
          </div>
          <div>
            <span>Vₗ =</span>
            <b>[ V<span>state</span> ; V<span>text</span> ]</b>
          </div>
          <small>State KV：背景持续可读；Residual Control：当下动态转向</small>
        </Reveal>
      </div>
    </SceneShell>
  );
};

const EtaScene: React.FC<{scene: ArchitectureScene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const p = sceneProgress(frame, scene);
  const local = frame - scene.start;
  const switching = Math.floor(local / 115) % 3 === 2;
  const beta = switching
    ? 0.82 + 0.08 * pulse(local, 10)
    : 0.08 + 0.16 * pulse(local, 24);
  const actions = ['倾听', '探索', '共同建模', '推演', '执行', '关系修复'];
  const actionIndex = Math.floor(local / 115) % actions.length;

  return (
    <SceneShell scene={scene}>
      <div className="vad-eta">
        <div className="vad-eta-pipeline">
          <div className="vad-eta-block residual">
            <span>RESIDUAL SEQUENCE</span>
            <b>e₁ … eₜ</b>
            <div className="vad-wave-field">
              {Array.from({length: 18}).map((_, index) => (
                <i
                  key={index}
                  style={{
                    height: `${18 + 70 * Math.abs(Math.sin(local / 18 + index * 0.48))}%`,
                  }}
                />
              ))}
            </div>
            <small>任务 · 用户 · 关系 · 环境的内部表征</small>
          </div>

          <div className="vad-eta-arrow">
            <FlowDots frame={frame} count={5} tone="aqua" />
          </div>

          <div className="vad-eta-block encoder">
            <span>SEQUENCE ENCODER</span>
            <b>q(z̃ₜ | e₁:ₜ)</b>
            <MiniVector frame={frame} count={16} tone="aqua" />
            <small>高维残差 → 低维候选代码</small>
          </div>

          <div className="vad-eta-arrow">
            <FlowDots frame={frame + 30} count={5} tone="amber" />
          </div>

          <div className={`vad-beta-gate ${switching ? 'switching' : ''}`}>
            <div
              className="vad-beta-dial"
              style={{'--beta': beta} as React.CSSProperties}
            >
              <i />
              <b>{beta.toFixed(2)}</b>
            </div>
            <span>βₜ SWITCH GATE</span>
            <strong>{switching ? '切换抽象动作' : '延续当前动作'}</strong>
            <small>稀疏切换 · 对齐子目标边界</small>
          </div>

          <div className="vad-eta-arrow">
            <FlowDots frame={frame + 60} count={5} tone="lime" />
          </div>

          <div className="vad-eta-block decoder">
            <span>RESIDUAL DECODER</span>
            <b>Uₜ = Decoder(zₜ)</b>
            <MiniVector frame={frame} count={16} tone="lime" offset={12} />
            <small>有界控制，不重写基底</small>
          </div>
        </div>

        <div className="vad-eta-code-space">
          <div className="vad-eta-z">
            <span>低维控制代码 zₜ</span>
            <div>
              {Array.from({length: 18}).map((_, index) => (
                <i
                  key={index}
                  style={{
                    transform: `scaleY(${
                      0.22 + 0.78 * Math.abs(Math.sin(local / 27 + index * 0.64))
                    })`,
                  }}
                />
              ))}
            </div>
            <small>Internal RL 的动作空间</small>
          </div>
          <div className="vad-action-families">
            <span>涌现的抽象动作族</span>
            <div>
              {actions.map((action, index) => (
                <b key={action} className={index === actionIndex ? 'active' : ''}>
                  {action}
                </b>
              ))}
            </div>
            <small>reuse · create · split · merge · prune</small>
          </div>
          <div className="vad-eta-compression">
            <div>
              <b>TOKEN SPACE</b>
              <span>数万维 × 每个 token</span>
            </div>
            <i>→</i>
            <div className="compact">
              <b>z SPACE</b>
              <span>低维 × 子目标时间尺度</span>
            </div>
          </div>
        </div>

        <Reveal progress={p} at={0.62} className="vad-eta-formula">
          <span>zₜ = βₜ ⊙ z̃ₜ + (1 − βₜ) ⊙ zₜ₋₁</span>
          <span>e′ₜ,ₗ = eₜ,ₗ + λₗ · Uₜ(eₜ,ₗ)</span>
          <b>全序列 SSL 发现结构 → 因果策略接管 → Internal RL 在 z 空间强化</b>
        </Reveal>
      </div>
    </SceneShell>
  );
};

const NlScene: React.FC<{scene: ArchitectureScene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const p = sceneProgress(frame, scene);
  const local = frame - scene.start;
  const bands = [
    {
      key: 'fast',
      name: 'online-fast',
      cadence: '每轮 / 每 wave',
      keeps: '当前状态 · zₜ · 工作记忆',
      owner: '有界控制器',
      every: 18,
    },
    {
      key: 'session',
      name: 'session-medium',
      cadence: '场景 / 会话',
      keeps: '情景模式 · 开放事项 · 策略偏好',
      owner: 'CMS 中频层',
      every: 48,
    },
    {
      key: 'slow',
      name: 'background-slow',
      cadence: '会话之间',
      keeps: '跨会话认知 · 记忆与策略沉淀',
      owner: 'Reflection + CMS',
      every: 90,
    },
    {
      key: 'heavy',
      name: 'rare-heavy',
      cadence: '低频离线',
      keeps: '共享 adapter · Prefix Generator',
      owner: '训练管线 + 修改门',
      every: 150,
    },
  ];

  return (
    <SceneShell scene={scene}>
      <div className="vad-nl">
        <div className="vad-nl-tower">
          {bands.map((band, index) => {
            const beat = pulse(local, band.every / 2, index * 11);
            return (
              <Reveal
                progress={p}
                at={0.06 + index * 0.07}
                className={`vad-nl-band ${band.key}`}
                key={band.key}
              >
                <div className="vad-nl-frequency">
                  <i style={{opacity: 0.28 + beat * 0.72, transform: `scale(${0.8 + beat * 0.3})`}} />
                  <span>{band.cadence}</span>
                </div>
                <div className="vad-nl-band-body">
                  <span>{band.name}</span>
                  <strong>{band.keeps}</strong>
                  <small>{band.owner}</small>
                </div>
                <div className="vad-nl-memory-cells">
                  {Array.from({length: 10 - index}).map((_, cell) => (
                    <i
                      key={cell}
                      style={{
                        opacity:
                          ((Math.floor(local / band.every) + cell) %
                            (4 + index)) ===
                          0
                            ? 1
                            : 0.2,
                      }}
                    />
                  ))}
                </div>
              </Reveal>
            );
          })}
          <div className="vad-nl-backflow">
            <FlowDots frame={-frame} vertical reverse count={8} tone="amber" />
            <span>慢层塑造快层初始化</span>
          </div>
        </div>

        <div className="vad-nl-right">
          <div className="vad-cms-orbit">
            <div className="vad-cms-core">
              <strong>CMS</strong>
              <span>连续记忆谱</span>
            </div>
            {['瞬态', '情景', '持久', '派生'].map((label, index) => {
              const angle = local / (80 + index * 15) + index * 1.57;
              const radius = 92 + index * 42;
              return (
                <div
                  className={`vad-cms-node node-${index}`}
                  key={label}
                  style={{
                    transform: `translate(${Math.cos(angle) * radius}px, ${
                      Math.sin(angle) * radius * 0.55
                    }px)`,
                  }}
                >
                  {label}
                </div>
              );
            })}
          </div>
          <div className="vad-nl-principles">
            <div>
              <span>01</span>
              <b>不同知识不共用一个更新节奏</b>
            </div>
            <div>
              <span>02</span>
              <b>快速适应不阻塞自然交互</b>
            </div>
            <div>
              <span>03</span>
              <b>慢速沉淀不直接改写线上基底</b>
            </div>
          </div>
          <Reveal progress={p} at={0.62} className="vad-ssl-rl-loop">
            <div>
              <b>SSL</b>
              <span>从历史中压缩结构</span>
            </div>
            <i>⇄</i>
            <div>
              <b>Internal RL</b>
              <span>在压缩的 z 空间强化</span>
            </div>
            <small>先发现结构，再训练因果策略</small>
          </Reveal>
          <Reveal progress={p} at={0.72} className="vad-nl-kernels">
            <span><b>LSS / PE</b> 误差就是待记忆的变化</span>
            <span><b>M3</b> 快慢动量避免同频震荡</span>
            <span><b>ATLAS</b> 结合过去状态优化当前记忆</span>
            <span><b>Titans</b> 用惊奇度控制写入强度</span>
          </Reveal>
        </div>
      </div>
    </SceneShell>
  );
};

const ActiveLearningScene: React.FC<{scene: ArchitectureScene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const p = sceneProgress(frame, scene);
  const local = frame - scene.start;
  const candidates = [
    {q: '他到底说了什么？', score: 0.33},
    {q: '这个决定可逆吗？', score: 0.82},
    {q: '你最不能承受的结果是什么？', score: 0.94},
    {q: '你现在开心吗？', score: 0.27},
    {q: '孩子和现金流哪个约束更硬？', score: 0.88},
    {q: '要不要再想想？', score: 0.19},
  ];
  const selected = Math.floor(local / 140) % 2 === 0 ? 2 : 4;

  return (
    <SceneShell scene={scene}>
      <div className="vad-active-learning">
        <div className="vad-active-candidates">
          <div className="vad-active-heading">
            <span>CANDIDATE QUESTIONS</span>
            <b>不是多问，而是选择信息价值最高的一问</b>
          </div>
          {candidates.map((candidate, index) => {
            const isSelected = index === selected;
            return (
              <Reveal
                progress={p}
                at={0.06 + index * 0.035}
                className={`vad-question-card ${isSelected ? 'selected' : ''}`}
                key={candidate.q}
              >
                <span>Q{index + 1}</span>
                <b>{candidate.q}</b>
                <div>
                  <i style={{width: `${candidate.score * 100}%`}} />
                </div>
                <small>{isSelected ? 'REQUEST FEEDBACK' : candidate.score.toFixed(2)}</small>
              </Reveal>
            );
          })}
        </div>

        <div className="vad-active-score">
          <div className="vad-active-score-core">
            <span>QUERY VALUE</span>
            <strong>
              U <i>×</i> IG <i>×</i> IMPACT <i>×</i> RISK
            </strong>
            <small>不确定性 × 信息增益 × 决策影响 × 不可逆风险</small>
          </div>
          <FlowDots frame={frame} vertical count={8} tone="amber" />
        </div>

        <div className="vad-version-space">
          <span>RELIABLE APPRENTICESHIP</span>
          <div className="vad-space-visual">
            <i className="hypothesis h1" />
            <i className="hypothesis h2" />
            <i className="hypothesis h3" />
            <div className="agreement">
              <b>可靠区</b>
              <small>自主行动</small>
            </div>
            <div className="uncertain">
              <b>不一致区</b>
              <small>延迟 / 求证</small>
            </div>
          </div>
          <div className="vad-active-rule">
            <b>知道 optimal action</b>
            <span>→ 执行</span>
            <b>不知道 optimal action</b>
            <span>→ 向人求证</span>
          </div>
          <small>以最少标注，覆盖最有学习价值的边界</small>
          <div className="vad-active-modes">
            <div>
              <b>自然交互</b>
              <span>选择最能改变决策的一问</span>
            </div>
            <div>
              <b>可靠学徒</b>
              <span>无把握时向人求证</span>
            </div>
            <div>
              <b>Active RLHF</b>
              <span>优先标注偏好分歧最大的样本</span>
            </div>
          </div>
        </div>
      </div>
    </SceneShell>
  );
};

const PredictionErrorScene: React.FC<{scene: ArchitectureScene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const p = sceneProgress(frame, scene);
  const local = frame - scene.start;
  const expected = 0.72;
  const observed = 0.34 + 0.22 * pulse(local, 32);
  const error = observed - expected;
  const branches = [
    ['状态', '用户/关系/任务估计'],
    ['记忆', '写入、提升、衰减'],
    ['控制器', 'zₜ 策略与 βₜ'],
    ['慢训练', 'adapter / prefix 候选'],
  ];

  return (
    <SceneShell scene={scene}>
      <div className="vad-pe">
        <div className="vad-pe-loop">
          <div className="vad-pe-node predict">
            <span>01 PREDICT</span>
            <b>行动前留下预期</b>
            <small>“追问现金流会降低关键不确定性”</small>
          </div>
          <div className="vad-pe-node act">
            <span>02 ACT</span>
            <b>选择抽象动作</b>
            <small>探索 → 推演 → 建议</small>
          </div>
          <div className="vad-pe-node observe">
            <span>03 OBSERVE</span>
            <b>等待真实结果</b>
            <small>用户反应 · 工具结果 · 业务结果</small>
          </div>
          <svg className="vad-pe-svg" viewBox="0 0 700 520">
            <path d="M170,105 C440,0 650,105 590,260" />
            <path d="M590,260 C650,455 370,510 185,405" />
            <path d="M185,405 C25,330 20,165 170,105" />
            {Array.from({length: 5}).map((_, index) => {
              const angle = (local / 42 + index / 5) * Math.PI * 2;
              return (
                <circle
                  key={index}
                  cx={350 + Math.cos(angle) * 265}
                  cy={260 + Math.sin(angle) * 185}
                  r="5"
                />
              );
            })}
          </svg>
          <div className="vad-pe-core">
            <span>PREDICTION ERROR</span>
            <div>
              <b>{error > 0 ? '+' : ''}{error.toFixed(2)}</b>
              <small>observed − predicted</small>
            </div>
            <div className="vad-pe-bars">
              <i style={{height: `${expected * 100}%`}} />
              <i className="actual" style={{height: `${observed * 100}%`}} />
            </div>
          </div>
        </div>

        <div className="vad-credit">
          <div className="vad-credit-heading">
            <span>04 CREDIT ASSIGNMENT</span>
            <b>到底该改哪里？</b>
          </div>
          <div className="vad-credit-branches">
            {branches.map(([title, detail], index) => (
              <Reveal
                progress={p}
                at={0.34 + index * 0.06}
                className="vad-credit-card"
                key={title}
              >
                <span>{String(index + 1).padStart(2, '0')}</span>
                <b>{title}</b>
                <small>{detail}</small>
                <i
                  style={{
                    width: `${35 + 55 * pulse(local, 18 + index * 3, index * 30)}%`,
                  }}
                />
              </Reveal>
            ))}
          </div>
          <div className="vad-pe-primitive">
            <strong>PE 是一级原始学习信号</strong>
            <span>evaluation 只做 readout / gate，不反向冒充学习源</span>
          </div>
        </div>
      </div>
    </SceneShell>
  );
};

const RareHeavyScene: React.FC<{scene: ArchitectureScene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const p = sceneProgress(frame, scene);
  const local = frame - scene.start;
  const stages = [
    ['01', '审计经历', 'PE + credit + lineage'],
    ['02', '主动选样', '高不确定 / 高影响'],
    ['03', '离线训练', '反事实 + LoRA / Prefix'],
    ['04', '同基底消融', '效果 · 安全 · 迁移'],
    ['05', 'ModificationGate', '签名 · 灰度 · 回滚'],
    ['06', '共享版本', 'adapter-vN'],
  ];

  return (
    <SceneShell scene={scene}>
      <div className="vad-rare">
        <div className="vad-online-lane">
          <div className="vad-lane-heading">
            <span>ONLINE PATH</span>
            <b>自然交互不能等待重训练</b>
          </div>
          <div className="vad-online-base">
            <div className="vad-lock">LOCKED</div>
            <strong>Frozen Base Model</strong>
            <span>State KV + zₜ + memory 在线变化</span>
            <small>基底权重保持不变</small>
          </div>
          <FlowDots frame={frame} count={10} tone="aqua" />
          <div className="vad-online-users">
            {['A', 'B', 'C', 'D'].map((user, index) => (
              <div key={user}>
                <b>USER {user}</b>
                <i style={{width: `${30 + pulse(local, 20, index * 25) * 60}%`}} />
              </div>
            ))}
          </div>
        </div>

        <div className="vad-rare-divider">
          <span>only reviewed evidence crosses this boundary</span>
          <FlowDots frame={frame + 40} vertical count={7} tone="amber" />
        </div>

        <div className="vad-offline-lane">
          <div className="vad-lane-heading">
            <span>RARE-HEAVY PATH</span>
            <b>跨用户稳定规律，才进入共享参数</b>
          </div>
          <div className="vad-rare-pipeline">
            {stages.map(([number, title, detail], index) => {
              const active =
                Math.floor(Math.max(0, local - 60) / 72) % stages.length === index;
              return (
                <Reveal
                  progress={p}
                  at={0.06 + index * 0.045}
                  className={`vad-rare-stage ${active ? 'active' : ''}`}
                  key={number}
                >
                  <span>{number}</span>
                  <b>{title}</b>
                  <small>{detail}</small>
                  {index < stages.length - 1 ? <i>→</i> : null}
                </Reveal>
              );
            })}
          </div>

          <div className="vad-lora-cutaway">
            <div className="vad-lora-base">
              <span>W</span>
              <b>BASE WEIGHT</b>
              <small>冻结</small>
            </div>
            <b>+</b>
            <div className="vad-lora-matrices">
              <div>
                <span>A</span>
                {Array.from({length: 16}).map((_, index) => <i key={index} />)}
              </div>
              <em>×</em>
              <div>
                <span>B</span>
                {Array.from({length: 16}).map((_, index) => <i key={index} />)}
              </div>
            </div>
            <b>=</b>
            <div className="vad-lora-output">
              <span>ΔW</span>
              <strong>共享技能适配</strong>
              <small>不是个人状态</small>
            </div>
          </div>
        </div>
      </div>
    </SceneShell>
  );
};

const ChatScene: React.FC<{scene: ArchitectureScene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const p = sceneProgress(frame, scene);
  const local = frame - scene.start;
  const turns = [
    {who: 'USER', text: '合伙人突然要把分成从五五改成七三，我想直接翻脸。'},
    {who: 'V', text: '先别急着决定。你最在意的是钱、控制权，还是他临时变卦这件事？'},
    {who: 'USER', text: '其实是他最近绕过我接触客户，我怕公司最后不是我的。'},
    {who: 'V', text: '那根问题不是这次分成，是客户和决策权正在脱离你。先确认三件事：合同、客户归属、他已经做到哪一步。'},
    {who: 'USER', text: '合同没写客户归属，最大的两个客户都是他带来的。'},
    {who: 'V', text: '现在直接翻脸风险很高。先把客户、交付和现金流做成可分离方案，再谈比例，你才有真正的选择权。'},
  ];
  const visible = Math.min(turns.length, Math.floor(Math.max(0, local - 35) / 72) + 1);
  const model = [
    ['STATE', '关系信任下降 · 控制权风险上升'],
    ['ETA', visible < 3 ? '倾听 / 探索' : visible < 5 ? '共同建模' : '形成可逆行动'],
    ['ACTIVE', visible < 5 ? '寻找会改变决策的变量' : '关键信息已足够'],
    ['MEMORY', '保留客户归属与决策偏好'],
    ['PE', '等待后续谈判与业务结果'],
  ];

  return (
    <SceneShell scene={scene}>
      <div className="vad-chat">
        <div className="vad-chat-window">
          <div className="vad-chat-topbar">
            <span>VOLVENCE / DECISION CONVERSATION</span>
            <i />
          </div>
          <div className="vad-chat-messages">
            {turns.map((turn, index) => (
              <div
                className={`vad-chat-message ${turn.who === 'V' ? 'model' : 'user'}`}
                key={turn.text}
                style={{
                  opacity: index < visible ? 1 : 0,
                  transform: `translateY(${index < visible ? 0 : 16}px)`,
                }}
              >
                <span>{turn.who}</span>
                <p>{turn.text}</p>
              </div>
            ))}
          </div>
        </div>
        <div className="vad-chat-model">
          <div className="vad-chat-model-title">
            <span>INTERNAL STATE</span>
            <b>用户看见的是自然对话<br />模型内部正在共同建模</b>
          </div>
          {model.map(([label, value], index) => (
            <Reveal
              progress={p}
              at={0.14 + index * 0.08}
              className="vad-chat-model-row"
              key={label}
            >
              <span>{label}</span>
              <b>{value}</b>
              <MiniVector frame={frame} count={7} tone={index === 1 ? 'amber' : 'aqua'} offset={index * 12} />
            </Reveal>
          ))}
          <div className="vad-chat-value">
            <span>不是替用户决定</span>
            <strong>而是与用户一起发现真正的问题</strong>
          </div>
        </div>
      </div>
    </SceneShell>
  );
};

const AgentScene: React.FC<{scene: ArchitectureScene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const p = sceneProgress(frame, scene);
  const local = frame - scene.start;
  const segments = [
    {name: '诊断', tools: '数据仓库 · 搜索', color: 'aqua'},
    {name: '验证', tools: '实验 · 客户访谈', color: 'amber'},
    {name: '执行', tools: 'CRM · 邮件 · 工单', color: 'lime'},
    {name: '复盘', tools: '结果 · PE · Credit', color: 'coral'},
  ];
  const active = Math.floor(local / 130) % segments.length;
  const beta = (local % 130) / 130;

  return (
    <SceneShell scene={scene}>
      <div className="vad-agent">
        <div className="vad-agent-goal">
          <span>ROOT GOAL</span>
          <b>找出转化下降的原因，并让下周收入恢复</b>
          <small>不是“完成一次搜索”或“发送一封邮件”</small>
        </div>

        <div className="vad-agent-timeline">
          {segments.map((segment, index) => (
            <div
              className={`vad-agent-segment ${segment.color} ${index === active ? 'active' : ''}`}
              key={segment.name}
            >
              <span>β boundary {index + 1}</span>
              <strong>{segment.name}</strong>
              <small>{segment.tools}</small>
              <div>
                <i style={{width: `${index < active ? 100 : index === active ? beta * 100 : 0}%`}} />
              </div>
              <b>z{index + 1}</b>
            </div>
          ))}
        </div>

        <div className="vad-agent-coordination">
          <Reveal progress={p} at={0.26} className="vad-agent-card eta">
            <span>ETA</span>
            <b>划分子目标边界</b>
            <small>决定延续当前策略，还是切换到下一抽象动作</small>
          </Reveal>
          <Reveal progress={p} at={0.36} className="vad-agent-card nl">
            <span>NL</span>
            <b>保留跨任务经验</b>
            <small>本轮、会话、跨会话和共享技能分层沉淀</small>
          </Reveal>
          <Reveal progress={p} at={0.46} className="vad-agent-card active">
            <span>ACTIVE</span>
            <b>关键步骤请求确认</b>
            <small>高风险、不可逆或证据冲突时不盲目执行</small>
          </Reveal>
          <Reveal progress={p} at={0.56} className="vad-agent-card pe">
            <span>PE</span>
            <b>跨步骤分配信用</b>
            <small>最后收入变化回到真正产生影响的前序动作</small>
          </Reveal>
        </div>

        <div className="vad-agent-result">
          <div>
            <span>普通 Agent</span>
            <b>完成工具调用</b>
          </div>
          <i>≠</i>
          <div className="strong">
            <span>Volvence Agent</span>
            <b>持续逼近根目标</b>
          </div>
        </div>
      </div>
    </SceneShell>
  );
};

const EvidenceScene: React.FC<{scene: ArchitectureScene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const p = sceneProgress(frame, scene);
  const arms = [
    ['A', 'Frozen base'],
    ['B', 'Profile prompt'],
    ['C', 'RAG / context'],
    ['D', 'Shared LoRA'],
    ['E', 'Fixed residual'],
    ['F', 'Learned residual'],
    ['G', 'State KV only'],
    ['H', 'ETA only'],
    ['I', 'KV + ETA / PE off'],
    ['J', 'Full loop'],
  ];
  const controls = ['错用户', '错对象', '过期状态', '撤销状态', '无关 Bank', 'Cold start'];

  return (
    <SceneShell scene={scene}>
      <div className="vad-evidence">
        <div className="vad-evidence-matrix">
          <div className="vad-evidence-heading">
            <span>MATCHED ABLATION / SAME SUBSTRATE</span>
            <b>十个实验臂，只改变一个机制</b>
          </div>
          <div className="vad-arm-grid">
            {arms.map(([code, name], index) => (
              <Reveal
                progress={p}
                at={0.04 + index * 0.025}
                className={`vad-arm ${index === arms.length - 1 ? 'full' : ''}`}
                key={code}
              >
                <span>{code}</span>
                <b>{name}</b>
                <div>
                  {Array.from({length: 5}).map((_, bar) => (
                    <i
                      key={bar}
                      style={{
                        height: `${18 + ((index * 17 + bar * 23) % 72)}%`,
                        opacity: 0.35 + 0.65 * pulse(frame, 30, index * 8 + bar * 4),
                      }}
                    />
                  ))}
                </div>
              </Reveal>
            ))}
          </div>
        </div>

        <div className="vad-negative-controls">
          <div className="vad-evidence-heading">
            <span>NEGATIVE CONTROLS</span>
            <b>系统必须知道“状态错了”</b>
          </div>
          <div className="vad-control-stack">
            {controls.map((control, index) => (
              <Reveal
                progress={p}
                at={0.28 + index * 0.055}
                className="vad-control-row"
                key={control}
              >
                <span>{String(index + 1).padStart(2, '0')}</span>
                <b>{control}</b>
                <i>BLOCK / DEFER / ZERO</i>
              </Reveal>
            ))}
          </div>
          <div className="vad-evidence-gates">
            <span>因果增益</span>
            <span>安全非劣</span>
            <span>跨域迁移</span>
            <span>撤销生效</span>
            <span>回滚精确</span>
          </div>
        </div>

        <Reveal progress={p} at={0.68} className="vad-evidence-status">
          <div className="current">
            <span>CURRENT</span>
            <b>固定残差基线 · Owner/快照 · ETA/NL 主链 · 主动求证 · rare-heavy 接缝</b>
          </div>
          <div className="target">
            <span>TARGET</span>
            <b>多银行 State KV · 首感知前水合 · 全组件因果闭环 · 生产级吞吐</b>
          </div>
        </Reveal>
      </div>
    </SceneShell>
  );
};

const OutroScene: React.FC<{scene: ArchitectureScene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const local = frame - scene.start;
  const p = sceneProgress(frame, scene);
  const intro = spring({
    frame: local - 18,
    fps,
    config: {damping: 30, stiffness: 65},
  });
  const layers = [
    ['STATE', '知道现在面对谁和什么局势'],
    ['DECISION', '在低维时间代码中选择如何行动'],
    ['LEARNING', '从真实结果修正下一轮'],
  ];

  return (
    <SceneShell scene={scene}>
      <div className="vad-outro">
        <div
          className="vad-outro-system"
          style={{
            opacity: intro,
            transform: `perspective(1200px) rotateX(${58 - intro * 10}deg) rotateZ(${
              -7 + intro * 7
            }deg) scale(${0.8 + intro * 0.2})`,
          }}
        >
          {layers.map(([name, detail], index) => (
            <div className={`vad-outro-layer layer-${index}`} key={name}>
              <span>0{index + 1}</span>
              <strong>{name}</strong>
              <b>{detail}</b>
              <MiniVector
                frame={frame}
                count={14}
                tone={index === 1 ? 'amber' : index === 2 ? 'lime' : 'aqua'}
                offset={index * 18}
              />
            </div>
          ))}
          <div className="vad-outro-base">
            <span>FROZEN FOUNDATION + SHARED ADAPTERS</span>
          </div>
        </div>
        <Reveal progress={p} at={0.36} className="vad-outro-claim">
          <span>VOLVENCE</span>
          <strong>面向人类理解的持续学习型语言模型</strong>
          <b>共享能力稳定进化 · 个体状态即时加载 · 决策依据结果持续修正</b>
        </Reveal>
        <div className="vad-outro-loop">
          {['目标', '感知', '决策', '反馈'].map((label, index) => (
            <div key={label}>
              <span>{label}</span>
              {index < 3 ? <i>→</i> : <i>↺</i>}
            </div>
          ))}
        </div>
      </div>
    </SceneShell>
  );
};

const sceneComponents: Record<
  ArchitectureScene['key'],
  React.FC<{scene: ArchitectureScene}>
> = {
  opening: OpeningScene,
  gap: GapScene,
  stack: StackScene,
  hydration: HydrationScene,
  'state-kv': StateKvScene,
  eta: EtaScene,
  nl: NlScene,
  'active-learning': ActiveLearningScene,
  'prediction-error': PredictionErrorScene,
  'rare-heavy': RareHeavyScene,
  chat: ChatScene,
  agent: AgentScene,
  evidence: EvidenceScene,
  outro: OutroScene,
};

export const VolvenceArchitectureDeepDive: React.FC = () => {
  const frame = useCurrentFrame();

  return (
    <AbsoluteFill className="vad-composition">
      {architectureScenes.map((scene) => {
        if (frame < scene.start - 1 || frame > scene.end + 1) {
          return null;
        }
        const Component = sceneComponents[scene.key];
        return <Component key={scene.key} scene={scene} />;
      })}
    </AbsoluteFill>
  );
};

export const VolvenceArchitectureDeepDiveNarrated: React.FC = () => {
  return (
    <AbsoluteFill className="vad-composition">
      <VolvenceArchitectureDeepDive />
      <Audio
        src={staticFile(
          'architecture-deep-dive/volvence-architecture-voiceover.wav',
        )}
        volume={1}
      />
    </AbsoluteFill>
  );
};
