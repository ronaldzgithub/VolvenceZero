import React from 'react';
import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import {
  edges,
  gateStages,
  modules,
  peAxes,
  scenes,
  StoryScene,
  timescaleLanes,
} from './data/storyboard';

type Point = {x: number; y: number};

const clamp01 = (value: number) => Math.max(0, Math.min(1, value));

const sceneProgress = (frame: number, scene: StoryScene) =>
  clamp01((frame - scene.start) / (scene.end - scene.start));

const activeScene = (frame: number) =>
  scenes.find((scene) => frame >= scene.start && frame < scene.end) ??
  scenes[scenes.length - 1];

const useSceneOpacity = (scene: StoryScene) => {
  const frame = useCurrentFrame();
  return interpolate(
    frame,
    [scene.start, scene.start + 28, scene.end - 28, scene.end],
    [0, 1, 1, 0],
    {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'},
  );
};

const Shell: React.FC<{scene: StoryScene; children: React.ReactNode}> = ({
  scene,
  children,
}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const intro = spring({frame, fps, config: {damping: 40, stiffness: 120}});
  const progress = sceneProgress(frame, scene);

  return (
    <AbsoluteFill className="video-shell">
      <div className="background-grid" />
      <div
        className="orbit orbit-a"
        style={{transform: `rotate(${progress * 36}deg)`}}
      />
      <div
        className="orbit orbit-b"
        style={{transform: `rotate(${-progress * 28}deg)`}}
      />
      <header
        className="scene-header"
        style={{
          opacity: interpolate(intro, [0, 1], [0, 1]),
          transform: `translateY(${interpolate(intro, [0, 1], [-18, 0])}px)`,
        }}
      >
        <div className="eyebrow">{scene.eyebrow}</div>
        <h1>{scene.title}</h1>
        <p>{scene.subtitle}</p>
      </header>
      {children}
      <footer className="timeline">
        {scenes.map((item) => {
          const current = frame >= item.start && frame < item.end;
          const seen = frame >= item.end;
          return (
            <div
              key={item.key}
              className={current ? 'timeline-step active' : seen ? 'timeline-step seen' : 'timeline-step'}
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

const Node: React.FC<{
  point: Point;
  label: string;
  owner: string;
  active?: boolean;
  compact?: boolean;
}> = ({point, label, owner, active = false, compact = false}) => (
  <div
    className={`node ${active ? 'active' : ''} ${compact ? 'compact' : ''}`}
    style={{left: point.x, top: point.y}}
  >
    <strong>{label}</strong>
    <span>{owner}</span>
  </div>
);

const Edge: React.FC<{from: Point; to: Point; progress: number; active?: boolean}> = ({
  from,
  to,
  progress,
  active = false,
}) => {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const length = Math.sqrt(dx * dx + dy * dy);
  const angle = Math.atan2(dy, dx) * (180 / Math.PI);
  return (
    <div
      className={`edge ${active ? 'active' : ''}`}
      style={{
        left: from.x + 118,
        top: from.y + 43,
        width: length - 210,
        transform: `rotate(${angle}deg)`,
      }}
    >
      <i style={{left: `${progress * 100}%`}} />
    </div>
  );
};

const ThesisScene: React.FC<{scene: StoryScene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const opacity = useSceneOpacity(scene);
  const p = sceneProgress(frame, scene);
  const pillars = [
    'Prediction Error first',
    'Snapshot-only exchange',
    'Stable substrate',
    'Latent temporal control',
    'World / Self dual tracks',
  ];

  return (
    <div className="scene scene-thesis" style={{opacity}}>
      <div className="core-mark">
        <div className="core-ring outer" />
        <div className="core-ring middle" />
        <div className="core-ring inner" />
        <div className="core-center">
          <b>Volvence</b>
          <span>bounded adaptive organism</span>
        </div>
      </div>
      <div className="pillar-stack">
        {pillars.map((pillar, index) => (
          <div
            className="pillar"
            key={pillar}
            style={{
              opacity: interpolate(p, [index * 0.1, index * 0.1 + 0.16], [0, 1], {
                extrapolateLeft: 'clamp',
                extrapolateRight: 'clamp',
              }),
              transform: `translateX(${interpolate(
                p,
                [index * 0.1, index * 0.1 + 0.16],
                [40, 0],
                {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'},
              )}px)`,
            }}
          >
            <span>{String(index + 1).padStart(2, '0')}</span>
            <strong>{pillar}</strong>
          </div>
        ))}
      </div>
    </div>
  );
};

const TurnScene: React.FC<{scene: StoryScene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const opacity = useSceneOpacity(scene);
  const p = sceneProgress(frame, scene);
  const moduleById = Object.fromEntries(modules.map((module) => [module.id, module]));
  const pulse = (p * 6) % 1;

  return (
    <div className="scene graph-scene" style={{opacity}}>
      <div className="graph">
        {edges.map(([fromId, toId], index) => {
          const from = moduleById[fromId];
          const to = moduleById[toId];
          const active = p > index * 0.08;
          return (
            <Edge
              key={`${fromId}-${toId}`}
              from={from}
              to={to}
              progress={pulse}
              active={active}
            />
          );
        })}
        {modules.map((module, index) => (
          <Node
            key={module.id}
            point={module}
            label={module.label}
            owner={module.owner}
            active={p > index * 0.08}
          />
        ))}
      </div>
      <div className="snapshots">
        {['substrate', 'temporal_abstraction', 'prediction_error', 'memory', 'evaluation'].map(
          (slot, index) => (
            <div
              className="snapshot"
              key={slot}
              style={{opacity: p > 0.18 + index * 0.08 ? 1 : 0.18}}
            >
              <code>{slot}</code>
              <span>frozen dataclass</span>
            </div>
          ),
        )}
      </div>
    </div>
  );
};

const PredictionErrorScene: React.FC<{scene: StoryScene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const opacity = useSceneOpacity(scene);
  const p = sceneProgress(frame, scene);
  const barProgress = interpolate(p, [0.18, 0.58], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  return (
    <div className="scene pe-scene" style={{opacity}}>
      <div className="prediction-strip">
        <div>
          <span>pre-action prediction</span>
          <strong>下一轮关系更稳定，任务推进 0.62</strong>
        </div>
        <b>compare</b>
        <div>
          <span>actual outcome</span>
          <strong>任务推进 0.48，关系张力上升</strong>
        </div>
      </div>
      <div className="axis-chart">
        {peAxes.map((axis, index) => (
          <div className="axis-row" key={axis.label}>
            <span>{axis.label}</span>
            <div className="axis-track">
              <i
                style={{
                  width: `${axis.value * barProgress * 100}%`,
                  transitionDelay: `${index * 80}ms`,
                }}
              />
            </div>
            <b>{(axis.value * barProgress).toFixed(2)}</b>
          </div>
        ))}
      </div>
      <div className="downstream-row">
        {['credit aggregation', 'memory write pressure', 'regime readout', 'temporal schedule'].map(
          (item, index) => (
            <div
              className="downstream"
              key={item}
              style={{opacity: p > 0.48 + index * 0.06 ? 1 : 0.2}}
            >
              {item}
            </div>
          ),
        )}
      </div>
    </div>
  );
};

const TimescaleScene: React.FC<{scene: StoryScene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const opacity = useSceneOpacity(scene);
  const p = sceneProgress(frame, scene);

  return (
    <div className="scene timescale-scene" style={{opacity}}>
      <div className="lanes">
        {timescaleLanes.map((lane, laneIndex) => {
          const local = (frame - scene.start + laneIndex * 20) % lane.cadence;
          const pulse = local / lane.cadence;
          return (
            <div className="lane" key={lane.label}>
              <div className="lane-label">
                <strong>{lane.label}</strong>
                <span>{lane.detail}</span>
              </div>
              <div className="lane-line">
                <i style={{left: `${pulse * 100}%`}} />
                {Array.from({length: 8}).map((_, index) => (
                  <em
                    key={index}
                    style={{opacity: p > 0.1 + index * 0.08 ? 1 : 0.15}}
                  />
                ))}
              </div>
            </div>
          );
        })}
      </div>
      <div className="substrate-lock">
        <strong>live substrate</strong>
        <span>frozen by default</span>
      </div>
    </div>
  );
};

const TemporalScene: React.FC<{scene: StoryScene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const opacity = useSceneOpacity(scene);
  const p = sceneProgress(frame, scene);
  const beta = 0.28 + 0.45 * Math.abs(Math.sin((frame - scene.start) / 42));
  const switchActive = beta > 0.55;
  const zValues = Array.from({length: 16}).map((_, index) =>
    0.5 + 0.38 * Math.sin((frame - scene.start) / 18 + index * 0.62),
  );

  return (
    <div className="scene temporal-scene" style={{opacity}}>
      <div className="residual-stack">
        {Array.from({length: 8}).map((_, index) => (
          <div
            className="residual-line"
            key={index}
            style={{transform: `scaleX(${0.62 + 0.3 * Math.sin(p * 6 + index)})`}}
          />
        ))}
      </div>
      <div className="latent-panel">
        <div className="latent-title">
          <strong>z_t controller code</strong>
          <span>low-dimensional action space</span>
        </div>
        <div className="z-grid">
          {zValues.map((value, index) => (
            <i key={index} style={{height: `${value * 100}%`}} />
          ))}
        </div>
        <div className={switchActive ? 'beta-gate active' : 'beta-gate'}>
          <span>beta_t switch gate</span>
          <b>{beta.toFixed(2)}</b>
        </div>
      </div>
      <div className="decoder-panel">
        <strong>Decoder(z_t)</strong>
        <span>residual controller U_t</span>
        <div className="decoder-wave" />
      </div>
    </div>
  );
};

const MemoryScene: React.FC<{scene: StoryScene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const opacity = useSceneOpacity(scene);
  const p = sceneProgress(frame, scene);
  const bands = [
    ['transient', 'working state', 0.86],
    ['episodic', 'session pattern', 0.64],
    ['durable', 'semantic memory', 0.48],
    ['derived', 'rebuildable index', 0.34],
  ] as const;

  return (
    <div className="scene memory-scene" style={{opacity}}>
      <div className="cms-tower">
        {bands.map(([name, detail, strength], index) => (
          <div
            className="memory-band"
            key={name}
            style={{
              width: `${520 + index * 120}px`,
              opacity: p > index * 0.1 ? 1 : 0.18,
            }}
          >
            <div>
              <strong>{name}</strong>
              <span>{detail}</span>
            </div>
            <i style={{width: `${strength * 100}%`}} />
          </div>
        ))}
      </div>
      <div className="reflection-path">
        <div>turn evidence</div>
        <span />
        <div>session-post slow loop</div>
        <span />
        <div>memory + policy consolidation</div>
      </div>
    </div>
  );
};

const GateScene: React.FC<{scene: StoryScene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const opacity = useSceneOpacity(scene);
  const p = sceneProgress(frame, scene);

  return (
    <div className="scene gate-scene" style={{opacity}}>
      <div className="gate-chain">
        {gateStages.map((stage, index) => (
          <React.Fragment key={stage}>
            <div
              className={index === 2 ? 'gate-stage central' : 'gate-stage'}
              style={{opacity: p > index * 0.12 ? 1 : 0.16}}
            >
              <strong>{stage}</strong>
            </div>
            {index < gateStages.length - 1 ? <span className="gate-arrow" /> : null}
          </React.Fragment>
        ))}
      </div>
      <div className="risk-lanes">
        <div>
          <strong>online / background</strong>
          <span>bounded owner state</span>
        </div>
        <div>
          <strong>offline / human review</strong>
          <span>adapter delta, LoRA, rare-heavy refresh</span>
        </div>
      </div>
    </div>
  );
};

const LoopScene: React.FC<{scene: StoryScene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const opacity = useSceneOpacity(scene);
  const p = sceneProgress(frame, scene);
  const orbit = p * 360;

  return (
    <div className="scene loop-scene" style={{opacity}}>
      <div className="final-loop">
        {['observe', 'predict', 'act', 'compare', 'consolidate', 'adapt'].map(
          (step, index) => {
            const angle = (index / 6) * Math.PI * 2 + (orbit * Math.PI) / 180;
            const x = 460 + Math.cos(angle) * 340;
            const y = 300 + Math.sin(angle) * 220;
            return (
              <div className="loop-node" key={step} style={{left: x, top: y}}>
                {step}
              </div>
            );
          },
        )}
        <div className="loop-center">
          <strong>world / self</strong>
          <span>dual-track continuity</span>
        </div>
      </div>
      <div className="closing-stack">
        <div>可检查 snapshots</div>
        <div>可回滚 updates</div>
        <div>可证明 learning evidence</div>
      </div>
    </div>
  );
};

export const VolvenceCoreMechanism: React.FC = () => {
  const frame = useCurrentFrame();
  const scene = activeScene(frame);

  return (
    <Shell scene={scene}>
      <ThesisScene scene={scenes[0]} />
      <TurnScene scene={scenes[1]} />
      <PredictionErrorScene scene={scenes[2]} />
      <TimescaleScene scene={scenes[3]} />
      <TemporalScene scene={scenes[4]} />
      <MemoryScene scene={scenes[5]} />
      <GateScene scene={scenes[6]} />
      <LoopScene scene={scenes[7]} />
    </Shell>
  );
};
