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
  CognitiveScene,
  CognitiveSceneId,
  cognitiveScenes,
} from './data/cognitiveAgi';
import './cognitive-agi.css';

const clamp = (value: number, min = 0, max = 1) =>
  Math.max(min, Math.min(max, value));

const easeOut = (value: number) => 1 - Math.pow(1 - clamp(value), 3);

const sceneProgress = (frame: number, scene: CognitiveScene) =>
  clamp(frame / Math.max(1, scene.durationInFrames - 1));

const audioProgress = (frame: number, scene: CognitiveScene, fps: number) =>
  clamp(frame / Math.max(1, scene.audioDuration * fps));

const reveal = (progress: number, at: number, span = 0.12) =>
  easeOut((progress - at) / span);

const SceneFrame: React.FC<{
  scene: CognitiveScene;
  children: React.ReactNode;
}> = ({scene, children}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const p = sceneProgress(frame, scene);
  const ap = audioProgress(frame, scene, fps);
  const activeCaption =
    scene.captions.find((caption) => ap >= caption.from && ap < caption.to) ??
    scene.captions[scene.captions.length - 1];
  const fade = interpolate(
    frame,
    [0, 18, scene.durationInFrames - 18, scene.durationInFrames],
    [0, 1, 1, 0],
    {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'},
  );
  const titleIn = spring({
    frame,
    fps,
    config: {damping: 30, stiffness: 110, mass: 0.8},
  });

  return (
    <AbsoluteFill
      className={`cagi-scene cagi-theme-${scene.id}`}
      style={{opacity: fade}}
    >
      <div className="cagi-noise" />
      <header
        className="cagi-header"
        style={{
          opacity: titleIn,
          transform: `translateY(${(1 - titleIn) * -24}px)`,
        }}
      >
        <div className="cagi-scene-index">
          <b>{scene.number}</b>
          <span>{scene.eyebrow}</span>
        </div>
        <div className="cagi-wordmark">VOLVENCE</div>
      </header>
      <div className="cagi-title-block">
        <h1>{scene.title}</h1>
      </div>
      <main className="cagi-stage">{children}</main>
      <footer className="cagi-caption">
        <div className="cagi-caption-rule">
          <i style={{width: `${p * 100}%`}} />
        </div>
        <p key={activeCaption.text}>{activeCaption.text}</p>
      </footer>
    </AbsoluteFill>
  );
};

const LoopScene: React.FC<{scene: CognitiveScene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const p = sceneProgress(frame, scene);
  const labels = ['目标', '感知', '决策', '行动', '反馈'];

  return (
    <SceneFrame scene={scene}>
      <div className="cagi-loop-composition">
        <div className="cagi-loop-row">
          {labels.map((label, index) => {
            const entered = reveal(p, 0.06 + index * 0.1, 0.1);
            return (
              <React.Fragment key={label}>
                <div
                  className={`cagi-loop-node ${index === 4 ? 'feedback' : ''}`}
                  style={{
                    opacity: entered,
                    transform: `translateY(${(1 - entered) * 34}px)`,
                  }}
                >
                  <span>0{index + 1}</span>
                  <strong>{label}</strong>
                </div>
                {index < labels.length - 1 ? (
                  <div
                    className="cagi-loop-arrow"
                    style={{opacity: reveal(p, 0.12 + index * 0.1, 0.08)}}
                  >
                    →
                  </div>
                ) : null}
              </React.Fragment>
            );
          })}
        </div>
        <div className="cagi-broken-return">
          <div style={{transform: `scaleX(${reveal(p, 0.58, 0.18)})`}} />
          <span
            style={{
              opacity: reveal(p, 0.7, 0.12),
              transform: `scale(${0.82 + reveal(p, 0.7, 0.12) * 0.18})`,
            }}
          >
            人
          </span>
          <div style={{transform: `scaleX(${reveal(p, 0.82, 0.12)})`}} />
        </div>
        <div
          className="cagi-loop-verdict"
          style={{opacity: reveal(p, 0.72, 0.16)}}
        >
          <span>AI 执行</span>
          <b>人类补完目标、反馈与修正</b>
        </div>
      </div>
      <div
        className="cagi-loop-pulse"
        style={{
          transform: `translateX(${((frame / fps) * 260) % 1800 - 120}px)`,
        }}
      />
    </SceneFrame>
  );
};

const CoverageScene: React.FC<{scene: CognitiveScene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const p = sceneProgress(frame, scene);
  const streams = [
    {label: '互联网数据', output: '预训练', tone: 'coral'},
    {label: '私有数据', output: '微调', tone: 'green'},
    {label: '场景数据', output: '对齐', tone: 'blue'},
  ];

  return (
    <SceneFrame scene={scene}>
      <div className="cagi-coverage-layout">
        <section className="cagi-predeploy">
          <div className="cagi-section-label">DEPLOYMENT BEFORE</div>
          {streams.map((stream, index) => {
            const entered = reveal(p, 0.08 + index * 0.12, 0.16);
            return (
              <div className="cagi-data-stream" key={stream.label}>
                <strong>{stream.label}</strong>
                <div className={`cagi-dot-field ${stream.tone}`}>
                  {Array.from({length: 14}).map((_, dot) => (
                    <i
                      key={dot}
                      style={{
                        opacity: entered,
                        transform: `translateX(${
                          ((frame * (1.2 + dot * 0.02) + dot * 61) % 560) - 30
                        }px)`,
                      }}
                    />
                  ))}
                </div>
                <b>{stream.output}</b>
              </div>
            );
          })}
          <div className="cagi-coverage-model">
            <span>FOUNDATION MODEL</span>
            <strong>尽可能见过整个世界</strong>
          </div>
        </section>
        <div className="cagi-deploy-divider">
          <b>DEPLOY</b>
          <i />
        </div>
        <section className="cagi-postdeploy">
          <div className="cagi-section-label">DEPLOYMENT AFTER</div>
          <div className="cagi-static-output">
            <span>服务</span>
            <span>响应</span>
            <span>执行</span>
          </div>
          <div
            className="cagi-human-loop"
            style={{opacity: reveal(p, 0.62, 0.18)}}
          >
            <b>目标</b>
            <b>反馈</b>
            <b>修正</b>
            <strong>仍由人完成</strong>
          </div>
        </section>
      </div>
      <div className="cagi-coverage-verdict">
        <strong>COVERAGE</strong>
        <span>≠</span>
        <strong>GROWTH</strong>
      </div>
    </SceneFrame>
  );
};

const HandoffScene: React.FC<{scene: CognitiveScene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const p = sceneProgress(frame, scene);
  const expansion = interpolate(p, [0.08, 0.62], [0.7, 2.2], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const bounded = reveal(p, 0.58, 0.2);

  return (
    <SceneFrame scene={scene}>
      <div className="cagi-world-space">
        <div
          className="cagi-world-grid"
          style={{
            transform: `translate(-50%, -50%) scale(${expansion}) rotate(${
              frame / 90
            }deg)`,
            opacity: 1 - bounded * 0.68,
          }}
        />
        <div className="cagi-world-counter">
          <span>PHYSICAL WORLD</span>
          <strong>
            {p < 0.28 ? '10⁶' : p < 0.48 ? '10¹²' : p < 0.66 ? '10²⁴' : '10∞'}
          </strong>
          <b>状态空间</b>
        </div>
        <div className="cagi-world-costs">
          {['状态巨大', '反馈缓慢', '错误昂贵'].map((item, index) => (
            <span
              key={item}
              style={{opacity: reveal(p, 0.22 + index * 0.1, 0.12)}}
            >
              {item}
            </span>
          ))}
        </div>
        <div
          className="cagi-bounded-world"
          style={{
            opacity: bounded,
            transform: `translate(-50%, -50%) scale(${0.75 + bounded * 0.25})`,
          }}
        >
          <div className="cagi-mini-loop">
            {['目标', '感知', '决策', '行动', '反馈'].map((item) => (
              <i key={item}>{item}</i>
            ))}
          </div>
          <span>MINIMUM COMPLETE ENVIRONMENT</span>
          <strong>Cognitive AGI</strong>
        </div>
      </div>
    </SceneFrame>
  );
};

const CognitiveScene: React.FC<{scene: CognitiveScene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const p = sceneProgress(frame, scene);
  const choices = [
    {at: 'DAY 01', label: '什么有效', y: 58},
    {at: 'DAY 09', label: '何时坚持', y: 34},
    {at: 'DAY 31', label: '何时拒绝', y: 70},
    {at: 'DAY 92', label: '如何修复', y: 42},
  ];
  const lineLength = reveal(p, 0.16, 0.58);

  return (
    <SceneFrame scene={scene}>
      <div className="cagi-cognitive-layout">
        <aside className="cagi-knowledge-column">
          <span>KNOWLEDGE</span>
          <strong>描述世界</strong>
          <div className="cagi-knowledge-stack">
            {['事实', '语言', '规则', '文档'].map((item, index) => (
              <i
                key={item}
                style={{opacity: reveal(p, 0.06 + index * 0.06, 0.12)}}
              >
                {item}
              </i>
            ))}
          </div>
        </aside>
        <section className="cagi-experience-timeline">
          <header>
            <span>EXPERIENCE</span>
            <strong>改变下一次行动</strong>
          </header>
          <div className="cagi-timeline-plot">
            <svg viewBox="0 0 1120 360" preserveAspectRatio="none">
              <path
                d="M40 250 C180 80 300 120 420 205 C540 300 650 65 785 155 C900 230 970 115 1080 95"
                style={{
                  strokeDasharray: 1500,
                  strokeDashoffset: 1500 * (1 - lineLength),
                }}
              />
            </svg>
            {choices.map((choice, index) => {
              const entered = reveal(p, 0.23 + index * 0.13, 0.14);
              return (
                <div
                  className="cagi-experience-point"
                  key={choice.at}
                  style={{
                    left: `${7 + index * 29}%`,
                    top: `${choice.y}%`,
                    opacity: entered,
                    transform: `translate(-50%, -50%) scale(${
                      0.75 + entered * 0.25
                    })`,
                  }}
                >
                  <i />
                  <span>{choice.at}</span>
                  <strong>{choice.label}</strong>
                </div>
              );
            })}
          </div>
          <div className="cagi-trajectory-label">
            <span>长期轨迹</span>
            <b>一次选择改变之后几个月</b>
          </div>
        </section>
      </div>
      <div
        className="cagi-cognitive-orbit"
        style={{transform: `rotate(${frame / 16}deg)`}}
      />
    </SceneFrame>
  );
};

const RelationshipScene: React.FC<{scene: CognitiveScene}> = ({scene}) => {
  const p = sceneProgress(useCurrentFrame(), scene);
  const nodes = ['边界', '拒绝', '冲突', '修复'];

  return (
    <SceneFrame scene={scene}>
      <div className="cagi-relationship-layout">
        <section className="cagi-satisfaction-lane">
          <header>
            <span>INSTANT SATISFACTION</span>
            <strong>即时满足</strong>
          </header>
          <div className="cagi-linear-track">
            {['用户请求', '模型满足', '会话结束'].map((item, index) => (
              <React.Fragment key={item}>
                <b style={{opacity: reveal(p, 0.05 + index * 0.09, 0.12)}}>
                  {item}
                </b>
                {index < 2 ? <i>→</i> : null}
              </React.Fragment>
            ))}
          </div>
          <p>没有可积累的关系状态</p>
        </section>
        <section className="cagi-relationship-loop">
          <div className="cagi-relationship-ring">
            {nodes.map((node, index) => {
              const angle = (Math.PI * 2 * index) / nodes.length - Math.PI / 2;
              const entered = reveal(p, 0.24 + index * 0.1, 0.14);
              return (
                <div
                  className="cagi-relation-node"
                  key={node}
                  style={{
                    left: `${50 + Math.cos(angle) * 38}%`,
                    top: `${50 + Math.sin(angle) * 38}%`,
                    opacity: entered,
                    transform: `translate(-50%, -50%) scale(${
                      0.8 + entered * 0.2
                    })`,
                  }}
                >
                  {node}
                </div>
              );
            })}
            <div className="cagi-ring-center">
              <span>LONG-TERM</span>
              <strong>关系状态</strong>
              <b>改变未来行动</b>
            </div>
          </div>
        </section>
        <div className="cagi-relation-thesis">
          <span>RELATIONSHIP IS NOT A USE CASE</span>
          <strong>关系是完整闭环的训练环境</strong>
        </div>
      </div>
    </SceneFrame>
  );
};

const SystemScene: React.FC<{scene: CognitiveScene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const p = sceneProgress(frame, scene);
  const modules = [
    {label: '自然交互', sub: 'Observe'},
    {label: '潜在状态', sub: 'Represent'},
    {label: '抽象决策', sub: 'Decide'},
    {label: '行动', sub: 'Act'},
    {label: '真实结果', sub: 'Outcome'},
    {label: '预测误差', sub: 'Learn'},
    {label: '经验沉淀', sub: 'Consolidate'},
  ];
  const pulse = ((frame / 2.2) % 1260) - 30;

  return (
    <SceneFrame scene={scene}>
      <div className="cagi-system-map">
        <div className="cagi-system-pipeline">
          {modules.map((module, index) => {
            const entered = reveal(p, 0.04 + index * 0.075, 0.11);
            const active = p > 0.1 + index * 0.075;
            return (
              <React.Fragment key={module.label}>
                <div
                  className={`cagi-system-module ${active ? 'active' : ''}`}
                  style={{
                    opacity: entered,
                    transform: `translateY(${(1 - entered) * 32}px)`,
                  }}
                >
                  <span>{module.sub}</span>
                  <strong>{module.label}</strong>
                </div>
                {index < modules.length - 1 ? (
                  <div className="cagi-system-arrow">→</div>
                ) : null}
              </React.Fragment>
            );
          })}
          <i className="cagi-system-pulse" style={{left: pulse}} />
        </div>
        <div
          className="cagi-system-return"
          style={{opacity: reveal(p, 0.54, 0.2)}}
        >
          <i />
          <span>经验重新影响目标、状态与下一次决策</span>
          <b>↩</b>
        </div>
        <div className="cagi-system-engines">
          {[
            ['Representation', '形成世界、自我与关系状态'],
            ['Decision', '在 Token 之上选择抽象动作'],
            ['Learning', '主动选择高价值反馈'],
            ['Memory', '快、中、慢、极慢沉淀经验'],
            ['Governance', '门控、比较、审计、回滚'],
          ].map(([label, detail], index) => (
            <div
              key={label}
              style={{opacity: reveal(p, 0.58 + index * 0.055, 0.12)}}
            >
              <strong>{label}</strong>
              <span>{detail}</span>
            </div>
          ))}
        </div>
        <div className="cagi-system-timescales">
          {['online-fast', 'session-medium', 'background-slow', 'rare-heavy'].map(
            (item, index) => (
              <span
                key={item}
                style={{opacity: reveal(p, 0.68 + index * 0.05, 0.1)}}
              >
                {item}
              </span>
            ),
          )}
        </div>
      </div>
    </SceneFrame>
  );
};

const CaseScene: React.FC<{scene: CognitiveScene}> = ({scene}) => {
  const p = sceneProgress(useCurrentFrame(), scene);
  const later = reveal(p, 0.5, 0.16);
  const stages = ['识别目标', '发现缺口', '选择行动', '等待结果', '完成归因'];

  return (
    <SceneFrame scene={scene}>
      <div className="cagi-case-layout">
        <section className="cagi-phone">
          <header>
            <div className="cagi-avatar">V</div>
            <div>
              <strong>Volvence</strong>
              <span>{later > 0.5 ? '七天后' : '今天'}</span>
            </div>
          </header>
          <div className="cagi-chat-body">
            <div
              className="cagi-chat-bubble user"
              style={{
                opacity: 1 - later * 0.7,
                transform: `translateY(${-later * 34}px)`,
              }}
            >
              宝宝最近每晚醒四次，我白天还要上班。我试过戒掉夜奶，但没坚持住。
            </div>
            <div
              className="cagi-chat-bubble assistant"
              style={{
                opacity: reveal(p, 0.18, 0.16) * (1 - later * 0.7),
                transform: `translateY(${-later * 22}px)`,
              }}
            >
              我们先不追求一次改完。先固定入睡流程，再选择一顿最容易减少的夜奶。
            </div>
            <div
              className="cagi-chat-bubble user result"
              style={{
                opacity: later,
                transform: `translateY(${(1 - later) * 42}px)`,
              }}
            >
              现在每晚只醒两次，我也没有那么崩溃了。
            </div>
          </div>
        </section>
        <section className="cagi-case-learning">
          <div className="cagi-case-kicker">
            <span>NATURAL INTERACTION</span>
            <strong>没有奖励按钮，没有反馈接口</strong>
          </div>
          <div className="cagi-case-stages">
            {stages.map((stage, index) => {
              const entered = reveal(p, 0.12 + index * 0.13, 0.13);
              return (
                <React.Fragment key={stage}>
                  <div
                    className={index >= 3 && later > 0.4 ? 'active' : ''}
                    style={{opacity: entered}}
                  >
                    <i>0{index + 1}</i>
                    <strong>{stage}</strong>
                  </div>
                  {index < stages.length - 1 ? <span>→</span> : null}
                </React.Fragment>
              );
            })}
          </div>
          <div className="cagi-case-outcome">
            <div>
              <span>任务结果</span>
              <strong>夜醒 4 次 → 2 次</strong>
            </div>
            <div>
              <span>关系结果</span>
              <strong>压力下降，信任增加</strong>
            </div>
            <div>
              <span>经验</span>
              <strong>分阶段方案在当前条件下有效</strong>
            </div>
          </div>
        </section>
      </div>
    </SceneFrame>
  );
};

const FlywheelScene: React.FC<{scene: CognitiveScene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const p = sceneProgress(frame, scene);
  const items = [
    {label: '解决真实问题', position: 'top'},
    {label: '形成长期使用', position: 'right'},
    {label: '获得真实结果', position: 'bottom'},
    {label: '改善决策与关系', position: 'left'},
  ];

  return (
    <SceneFrame scene={scene}>
      <div className="cagi-flywheel">
        <div
          className="cagi-flywheel-ring"
          style={{
            transform: `translate(-50%, -50%) rotate(${frame / 25}deg)`,
          }}
        >
          {Array.from({length: 16}).map((_, index) => (
            <i key={index} style={{transform: `rotate(${index * 22.5}deg)`}} />
          ))}
        </div>
        <div className="cagi-flywheel-center">
          <span>COMPOUNDING LOOP</span>
          <strong>Volvence</strong>
          <b>产品价值 × 学习数据 × 模型进化</b>
        </div>
        {items.map((item, index) => (
          <div
            className={`cagi-flywheel-item ${item.position}`}
            key={item.label}
            style={{
              opacity: reveal(p, 0.08 + index * 0.12, 0.14),
              transform: `translate(-50%, -50%) scale(${
                0.82 + reveal(p, 0.08 + index * 0.12, 0.14) * 0.18
              })`,
            }}
          >
            <span>0{index + 1}</span>
            <strong>{item.label}</strong>
          </div>
        ))}
        <div
          className="cagi-flywheel-value"
          style={{opacity: reveal(p, 0.68, 0.18)}}
        >
          更长关系 → 更可信反馈 → 更好服务 → 更高价值
        </div>
      </div>
    </SceneFrame>
  );
};

const TeamScene: React.FC<{scene: CognitiveScene}> = ({scene}) => {
  const p = sceneProgress(useCurrentFrame(), scene);
  const people = [
    {
      name: '杨柳',
      question: '为什么可学',
      role: '主动学习 · 迁移学习 · 强化学习',
    },
    {
      name: '赵江波',
      question: '学什么',
      role: '目标 · 反馈 · 产品与商业结果',
    },
    {
      name: '马志刚',
      question: '如何学习',
      role: '认知闭环 · Decision · Memory · Learning',
    },
    {
      name: '张驰',
      question: '如何规模化',
      role: '训练 · 推理 · 接口 · 部署',
    },
  ];

  return (
    <SceneFrame scene={scene}>
      <div className="cagi-team-grid">
        {people.map((person, index) => {
          const entered = reveal(p, 0.08 + index * 0.12, 0.16);
          return (
            <section
              key={person.name}
              style={{
                opacity: entered,
                transform: `translateY(${(1 - entered) * 50}px)`,
              }}
            >
              <span>0{index + 1}</span>
              <div className="cagi-team-initial">{person.name.slice(0, 1)}</div>
              <h2>{person.name}</h2>
              <strong>{person.question}</strong>
              <p>{person.role}</p>
            </section>
          );
        })}
      </div>
      <div
        className="cagi-team-chain"
        style={{opacity: reveal(p, 0.68, 0.18)}}
      >
        <span>THEORY</span>
        <i>→</i>
        <span>MODEL</span>
        <i>→</i>
        <span>ENGINEERING</span>
        <i>→</i>
        <span>REAL WORLD</span>
      </div>
    </SceneFrame>
  );
};

const OutroScene: React.FC<{scene: CognitiveScene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const p = sceneProgress(frame, scene);
  const close = reveal(p, 0.08, 0.44);

  return (
    <SceneFrame scene={scene}>
      <div className="cagi-outro">
        <div className="cagi-outro-loop">
          <svg viewBox="0 0 620 620">
            <circle
              cx="310"
              cy="310"
              r="244"
              style={{
                strokeDasharray: 1534,
                strokeDashoffset: 1534 * (1 - close),
              }}
            />
          </svg>
          {['目标', '感知', '决策', '行动', '反馈'].map((item, index) => {
            const angle = (Math.PI * 2 * index) / 5 - Math.PI / 2;
            return (
              <span
                key={item}
                style={{
                  left: `${50 + Math.cos(angle) * 40}%`,
                  top: `${50 + Math.sin(angle) * 40}%`,
                  opacity: reveal(p, 0.12 + index * 0.06, 0.12),
                }}
              >
                {item}
              </span>
            );
          })}
          <div
            className="cagi-outro-center"
            style={{
              opacity: reveal(p, 0.48, 0.16),
              transform: `translate(-50%, -50%) scale(${
                0.86 + reveal(p, 0.48, 0.16) * 0.14
              })`,
            }}
          >
            <b>V</b>
          </div>
        </div>
        <div className="cagi-outro-copy">
          <span>FOUNDATION MODELS GIVE MACHINES KNOWLEDGE</span>
          <strong>基础模型让机器拥有知识</strong>
          <i />
          <span>VOLVENCE TURNS OUTCOMES INTO EXPERIENCE</span>
          <h2>Volvence，让机器形成经验</h2>
        </div>
      </div>
      <div
        className="cagi-outro-glint"
        style={{left: `${10 + ((frame / 3) % 90)}%`}}
      />
    </SceneFrame>
  );
};

const sceneComponents: Record<
  CognitiveSceneId,
  React.FC<{scene: CognitiveScene}>
> = {
  loop: LoopScene,
  coverage: CoverageScene,
  handoff: HandoffScene,
  cognitive: CognitiveScene,
  relationship: RelationshipScene,
  system: SystemScene,
  case: CaseScene,
  flywheel: FlywheelScene,
  team: TeamScene,
  outro: OutroScene,
};

export const CognitiveAgiFilm: React.FC = () => {
  return (
    <AbsoluteFill className="cagi-film">
      {cognitiveScenes.map((scene) => {
        const SceneComponent = sceneComponents[scene.id];
        return (
          <Sequence
            from={scene.startFrame}
            durationInFrames={scene.durationInFrames}
            key={scene.id}
            premountFor={30}
          >
            <SceneComponent scene={scene} />
            <Audio
              src={staticFile(`cognitive-agi/${scene.id}.m4a`)}
              volume={0.94}
            />
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};
