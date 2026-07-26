import React from 'react';
import {
  AbsoluteFill,
  Audio,
  interpolate,
  Sequence,
  staticFile,
  useCurrentFrame,
} from 'remotion';
import {
  DEPLOYMENT_AUDIO_FRAMES,
  DEPLOYMENT_DURATION_FRAMES,
  DEPLOYMENT_FPS,
  DEPLOYMENT_INTRO_FRAMES,
  deploymentNarration,
  type DeploymentNarrationLine,
} from './data/deploymentStandalone';
import './deployment-standalone.css';

export {
  DEPLOYMENT_DURATION_FRAMES,
  DEPLOYMENT_FPS,
};

const clamp = {
  extrapolateLeft: 'clamp' as const,
  extrapolateRight: 'clamp' as const,
};

const currentNarrationAt = (time: number): DeploymentNarrationLine => {
  let current = deploymentNarration[0];
  for (const line of deploymentNarration) {
    if (line.start <= time) current = line;
  }
  return current;
};

const baseCapabilities = ['语言理解与生成', '通用知识', '推理', '工具调用'];

const adapterCapabilities = [
  '通用语义本体论',
  '抽象动作族先验',
  '策略保持与切换先验',
  '预测误差解释先验',
  '安全边界先验',
  '默认关系动力学',
];

const profileCapabilities = [
  '用户模型实际取值',
  '关系状态轨迹',
  '具体记忆',
  '抽象策略有效性记录',
  '策略保持与切换偏好',
  '承诺与未完成事项',
  '授权范围与边界事实',
];

const ModelService: React.FC<{
  sequence: number;
  replica?: number;
}> = ({sequence, replica}) => {
  const isReplica = replica !== undefined;
  return (
    <div
      className={`d3-model-service ${isReplica ? 'replica' : ''} ${
        sequence === 2 ? 'focus-base' : ''
      } ${sequence === 3 ? 'focus-adapter' : ''}`}
      style={
        isReplica
          ? {
              transform: `translate(${replica * 118}px, ${replica * 34}px) scale(${
                0.9 - replica * 0.04
              })`,
              zIndex: 3 - replica,
            }
          : undefined
      }
    >
      <div className="d3-service-head">
        <span>统一模型服务</span>
        <b>共享权重</b>
      </div>
      <div className="d3-adapter-layer">
        <div>
          <span>共享 Adapter 模型</span>
          <b>Volvence 决策与关系先验</b>
        </div>
        {!isReplica ? (
          <div className="d3-adapter-caps">
            {adapterCapabilities.map((item) => <i key={item}>{item}</i>)}
          </div>
        ) : null}
      </div>
      <div className="d3-base-layer">
        <div>
          <span>Base 模型</span>
          <b>通用语言与推理底座</b>
        </div>
        {!isReplica ? (
          <div className="d3-base-caps">
            {baseCapabilities.map((item) => <i key={item}>{item}</i>)}
          </div>
        ) : null}
      </div>
      <div className="d3-request-slots">
        {['A', 'B', 'C', 'D'].map((item) => <span key={item}>{item}</span>)}
        <b>同一模型 · 不同状态</b>
      </div>
    </div>
  );
};

const ProfileStore: React.FC<{sequence: number}> = ({sequence}) => {
  const active = sequence === 4 || sequence === 5 || sequence === 6;
  return (
    <div className={`d3-profile-store ${active ? 'active' : ''}`}>
      <div className="d3-profile-head">
        <span>每用户独立</span>
        <b>用户档案存储</b>
        <i>不是模型权重 · 不是每用户微调模型</i>
      </div>
      <div className="d3-profile-users">
        {[
          ['A', '新客户'],
          ['B', '长期用户'],
          ['C', '高风险关系'],
        ].map(([id, label], index) => (
          <div
            key={id}
            className={sequence >= 5 && index === 0 ? 'loading' : ''}
          >
            <b>{id}</b>
            <span>{label}</span>
            <i>独立状态</i>
          </div>
        ))}
      </div>
      <div className="d3-profile-fields">
        {profileCapabilities.map((item, index) => (
          <span
            key={item}
            className={sequence === 4 ? 'revealed' : ''}
            style={{transitionDelay: `${index * 60}ms`}}
          >
            {item}
          </span>
        ))}
      </div>
    </div>
  );
};

const RequestFlow: React.FC<{
  sequence: number;
  frame: number;
}> = ({sequence, frame}) => {
  const active = sequence === 5 || sequence === 6;
  const flow = ((frame % 120) + 120) % 120 / 120;
  return (
    <svg
      className={`d3-flow-lines ${active ? 'active' : ''}`}
      viewBox="0 0 1760 760"
      preserveAspectRatio="none"
    >
      <defs>
        <marker id="d3-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L0,6 L7,3 z" fill="#65d6c4" />
        </marker>
        <marker id="d3-arrow-muted" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L0,6 L7,3 z" fill="rgba(137,168,165,.28)" />
        </marker>
      </defs>
      <path
        d="M535 360 C650 360 680 300 790 300"
        markerEnd={`url(#${active ? 'd3-arrow' : 'd3-arrow-muted'})`}
      />
      <path
        d="M1300 300 C1420 300 1440 360 1540 360"
        markerEnd={`url(#${active ? 'd3-arrow' : 'd3-arrow-muted'})`}
      />
      <path
        d="M1540 520 C1400 650 700 650 450 530"
        markerEnd={`url(#${active ? 'd3-arrow' : 'd3-arrow-muted'})`}
      />
      {active ? (
        <>
          <circle
            cx={535 + flow * 255}
            cy={360 - Math.sin(flow * Math.PI) * 60}
            r="7"
          />
          <circle
            cx={1300 + flow * 240}
            cy={300 + Math.sin(flow * Math.PI) * 60}
            r="7"
          />
        </>
      ) : null}
    </svg>
  );
};

const OutputPanel: React.FC<{sequence: number}> = ({sequence}) => (
  <div className={`d3-output ${sequence === 5 ? 'active' : ''}`}>
    <div className="d3-output-head">
      <span>一次请求</span>
      <b>对话 / Agent 行动</b>
    </div>
    <div className="d3-output-cards">
      <div><span>聊天</span><b>理解这个用户</b></div>
      <div><span>任务</span><b>执行具体行动</b></div>
    </div>
    <div className="d3-result-return">
      <span>结果返回</span>
      <b>只更新对应用户档案</b>
    </div>
  </div>
);

const DeploymentTopology: React.FC<{
  sequence: number;
  frame: number;
}> = ({sequence, frame}) => {
  const scaling = sequence === 6;
  return (
    <div className="d3-topology">
      <RequestFlow sequence={sequence} frame={frame} />
      <ProfileStore sequence={sequence} />
      <div className={`d3-service-cluster ${scaling ? 'scaling' : ''}`}>
        <ModelService sequence={sequence} />
        {scaling ? (
          <div className="d3-replica-stack">
            <b>无状态扩容</b>
            {['副本 01', '副本 02', '副本 N'].map((item) => (
              <span key={item}>{item}</span>
            ))}
            <i>共享同一套<br />Base + Adapter</i>
          </div>
        ) : null}
      </div>
      <OutputPanel sequence={sequence} />
      <div className="d3-deployment-legend">
        <div className={sequence === 2 ? 'active' : ''}>
          <span>Base</span>
          <b>通用能力</b>
          <i>统一部署</i>
        </div>
        <div className={sequence === 3 ? 'active' : ''}>
          <span>Adapter</span>
          <b>共享的Volvence先验</b>
          <i>统一部署</i>
        </div>
        <div className={sequence === 4 ? 'active' : ''}>
          <span>用户档案</span>
          <b>每个人的真实状态</b>
          <i>隔离存储 · 按请求加载</i>
        </div>
      </div>
    </div>
  );
};

export const VolvenceDeploymentStandalone: React.FC = () => {
  const frame = useCurrentFrame();
  const audioFrame = frame - DEPLOYMENT_INTRO_FRAMES;
  const audioTime = Math.max(0, audioFrame / DEPLOYMENT_FPS);
  const line = currentNarrationAt(audioTime);
  const introOpacity = interpolate(frame, [0, 22, 64, 90], [0, 1, 1, 0], clamp);
  const contentOpacity = interpolate(frame, [70, 105], [0, 1], clamp);
  const outroStart = DEPLOYMENT_INTRO_FRAMES + DEPLOYMENT_AUDIO_FRAMES;
  const outroOpacity = interpolate(
    frame,
    [outroStart, outroStart + 26],
    [0, 1],
    clamp,
  );

  return (
    <AbsoluteFill className="d3-film">
      <div className="d3-ambient d3-ambient-a" />
      <div className="d3-ambient d3-ambient-b" />
      <div className="d3-grid" />
      <div className="d3-brand">
        <span>V</span>
        <b>VOLVENCE</b>
        <i>部署方式</i>
      </div>
      <div className="d3-progress">
        {deploymentNarration.map((item) => (
          <span
            key={item.sequence}
            className={item.sequence === line.sequence ? 'active' : ''}
          />
        ))}
      </div>

      <div className="d3-intro" style={{opacity: introOpacity}}>
        <span>第三幕 · 独立部署说明</span>
        <strong>一套共享模型，每个人独立加载</strong>
        <b>Base + Adapter统一部署 · 用户档案隔离存储</b>
      </div>

      <div className="d3-content" style={{opacity: contentOpacity}}>
        <div className="d3-heading">
          <span>{String(line.sequence).padStart(2, '0')}</span>
          <div>
            <b>{line.caption}</b>
            <i>
              {line.sequence <= 4
                ? '先分清三者是什么'
                : line.sequence <= 6
                  ? '再看一次请求如何运行'
                  : '最后看清共享与隔离'}
            </i>
          </div>
        </div>
        <DeploymentTopology sequence={line.sequence} frame={frame} />
      </div>

      <div className="d3-outro" style={{opacity: outroOpacity}}>
        <span>VOLVENCE</span>
        <div>
          <section>
            <small>共享</small>
            <strong>语言 · 决策 · 学习能力</strong>
          </section>
          <i>≠</i>
          <section>
            <small>隔离</small>
            <strong>状态 · 关系 · 记忆</strong>
          </section>
        </div>
        <b>一套模型服务，同时服务大量不同的用户</b>
      </div>

      <Sequence
        from={DEPLOYMENT_INTRO_FRAMES}
        durationInFrames={DEPLOYMENT_AUDIO_FRAMES}
      >
        <Audio
          src={staticFile('deployment-standalone/volvence-deployment-standalone.wav')}
          volume={1}
        />
      </Sequence>
    </AbsoluteFill>
  );
};
