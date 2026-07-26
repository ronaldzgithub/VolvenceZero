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
  DIGITAL_TWIN_AUDIO_FRAMES,
  DIGITAL_TWIN_DURATION_FRAMES,
  DIGITAL_TWIN_FPS,
  DIGITAL_TWIN_INTRO_FRAMES,
  digitalTwinNarration,
  type DigitalTwinNarrationLine,
} from './data/digitalTwinStandalone';
import './digital-twin-standalone.css';

export {
  DIGITAL_TWIN_DURATION_FRAMES,
  DIGITAL_TWIN_FPS,
};

const clamp = {
  extrapolateLeft: 'clamp' as const,
  extrapolateRight: 'clamp' as const,
};

const sourceCards = [
  ['录音', '语气 · 叙事 · 自传记忆'],
  ['视频', '行为 · 表情 · 情境反应'],
  ['小说', '价值倾向 · 因果观'],
  ['对话记录', '关系模式 · 承诺 · 边界'],
  ['论文', '概念体系 · 证据标准'],
  ['工作记录', '方法 · 取舍 · 结果'],
];

const coreCards = [
  ['认知', '世界模型 · 价值排序 · 决策方式'],
  ['记忆', '人和事 · 情感意义 · 未完事项'],
  ['科学观', '概念体系 · 证据标准 · 不确定性'],
  ['经验', '案例 · 方法 · 取舍 · 真实结果'],
];

const proofCards = [
  ['记得对', '能回到具体的人、事和出处'],
  ['想得像', '面对新问题仍沿用同一套因果模型'],
  ['选得像', '在相似约束下作出稳定的价值取舍'],
  ['错得明白', '知道哪些只是推测，哪些证据不足'],
];

const verifyCards = [
  ['来源追溯', '结论可回到原文、时间和场景'],
  ['留出测试', '用训练时未见的问题检验'],
  ['历史回放', '重演本人过去的重要选择'],
  ['真人确认', '本人或获授权的同行校正'],
];

const currentNarrationAt = (time: number): DigitalTwinNarrationLine => {
  let current = digitalTwinNarration[0];
  for (const line of digitalTwinNarration) {
    if (line.start <= time) current = line;
  }
  return current;
};

const SourceField: React.FC<{active: boolean; sequence: number}> = ({
  active,
  sequence,
}) => (
  <section className={`dt-source-field ${active ? 'active' : ''}`}>
    <header>
      <span>01</span>
      <div>
        <b>人生证据</b>
        <i>不是素材堆积，是同一个人的生命轨迹</i>
      </div>
    </header>
    <div className="dt-source-grid">
      {sourceCards.map(([title, detail], index) => (
        <div
          key={title}
          className={sequence >= 2 ? 'revealed' : ''}
          style={{transitionDelay: `${index * 75}ms`}}
        >
          <span>{String(index + 1).padStart(2, '0')}</span>
          <b>{title}</b>
          <i>{detail}</i>
        </div>
      ))}
    </div>
    <div className="dt-source-rule">
      <span>小说中的角色</span>
      <b>≠</b>
      <span>作者本人的事实</span>
      <i>保留出处、角色与置信度</i>
    </div>
  </section>
);

const EvidenceCompiler: React.FC<{active: boolean; frame: number}> = ({
  active,
  frame,
}) => {
  const pulse = 0.72 + Math.sin(frame / 13) * 0.16;
  return (
    <section
      className={`dt-compiler ${active ? 'active' : ''}`}
      style={{'--dt-pulse': pulse} as React.CSSProperties}
    >
      <header>
        <span>02</span>
        <div>
          <b>证据对齐</b>
          <i>把碎片编译成可验证的个人状态</i>
        </div>
      </header>
      <div className="dt-compiler-body">
        <div className="dt-person-seal">
          <span>同一主体</span>
          <b>时间线</b>
          <i>来源链</i>
        </div>
        <div className="dt-align-list">
          <span>事实 / 观点 / 虚构分离</span>
          <span>人物 / 关系 / 事件对齐</span>
          <span>冲突保留，不强行平均</span>
          <span>每项结论标注置信度</span>
        </div>
      </div>
      <div className="dt-not-rag">
        <s>资料检索库</s>
        <b>个人模型编译</b>
      </div>
    </section>
  );
};

const PersonCore: React.FC<{active: boolean; sequence: number; frame: number}> = ({
  active,
  sequence,
  frame,
}) => {
  const rotation = (frame % 360) * 0.16;
  return (
    <section className={`dt-person-core ${active ? 'active' : ''}`}>
      <header>
        <span>03</span>
        <div>
          <b>个人内核</b>
          <i>能在新情境中运行，而不是复述资料</i>
        </div>
      </header>
      <div className="dt-core-orbit">
        <div
          className="dt-orbit-line"
          style={{transform: `rotate(${rotation}deg)`}}
        />
        <div className="dt-core-center">
          <small>可验证的</small>
          <strong>认知连续性</strong>
          <i>这个人的“为什么”</i>
        </div>
        {coreCards.map(([title, detail], index) => (
          <div
            key={title}
            className={`dt-core-card dt-core-card-${index + 1} ${
              sequence >= 4 ? 'revealed' : ''
            }`}
          >
            <b>{title}</b>
            <i>{detail}</i>
          </div>
        ))}
      </div>
      <div className="dt-shell-note">
        <span>声音 / 形象 / 口吻</span>
        <b>是表达外壳，不是人格本体</b>
      </div>
    </section>
  );
};

const ProofPanel: React.FC<{active: boolean; sequence: number}> = ({
  active,
  sequence,
}) => {
  const cards = sequence === 6 ? verifyCards : proofCards;
  return (
    <section className={`dt-proof-panel ${active ? 'active' : ''}`}>
      <header>
        <span>04</span>
        <div>
          <b>{sequence === 6 ? '一致性验证' : '为什么是他'}</b>
          <i>
            {sequence === 6
              ? '先验证，再称为数字分身'
              : '不是意识上传，是可被检验的连续性'}
          </i>
        </div>
      </header>
      <div className="dt-proof-grid">
        {cards.map(([title, detail], index) => (
          <div key={title} className={sequence >= 5 ? 'revealed' : ''}>
            <span>{index + 1}</span>
            <b>{title}</b>
            <i>{detail}</i>
          </div>
        ))}
      </div>
      <div className="dt-proof-result">
        <span>{sequence === 6 ? '留出情境' : '陌生问题'}</span>
        <i>→</i>
        <b>{sequence === 6 ? '稳定复现本人判断' : '仍保持同一套“为什么”'}</b>
      </div>
    </section>
  );
};

const DeploymentStrip: React.FC<{active: boolean; learning: boolean; frame: number}> = ({
  active,
  learning,
  frame,
}) => {
  const flow = ((frame % 110) + 110) % 110 / 110;
  return (
    <section className={`dt-deploy-strip ${active ? 'active' : ''}`}>
      <div className="dt-deploy-model">
        <span>共享能力</span>
        <b>Base + Volvence Adapter</b>
        <i>语言、推理、决策与学习机制</i>
      </div>
      <div className="dt-plus">+</div>
      <div className="dt-deploy-profile">
        <span>这个人</span>
        <b>个人状态 + 连续记忆</b>
        <i>认知、关系、科学观、经验与边界</i>
      </div>
      <div className="dt-equals">=</div>
      <div className="dt-deploy-twin">
        <span>按请求动态加载</span>
        <b>可行动的数字分身</b>
        <i>不是每个人复制一套大模型</i>
      </div>
      {learning ? (
        <div className="dt-learning-loop">
          {['新交互', '形成预期', '行动', '真实结果', '更新记忆与策略'].map(
            (item, index) => (
              <React.Fragment key={item}>
                <span className={Math.floor(flow * 5) === index ? 'active' : ''}>
                  {item}
                </span>
                {index < 4 ? <i>→</i> : null}
              </React.Fragment>
            ),
          )}
          <b>只在授权范围内 · 版本可追溯 · 随时可撤销</b>
        </div>
      ) : null}
    </section>
  );
};

const FlowLines: React.FC<{sequence: number; frame: number}> = ({
  sequence,
  frame,
}) => {
  const travel = ((frame % 130) + 130) % 130 / 130;
  return (
    <svg className="dt-flow-lines" viewBox="0 0 1760 765">
      <defs>
        <marker id="dt-arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
          <path d="M0,0 L0,6 L8,3 z" />
        </marker>
      </defs>
      <path className={sequence >= 3 ? 'active' : ''} d="M428 278 C468 278 475 278 512 278" />
      <path className={sequence >= 4 ? 'active' : ''} d="M800 278 C838 278 844 278 878 278" />
      <path className={sequence >= 5 ? 'active' : ''} d="M1247 278 C1282 278 1293 278 1322 278" />
      {sequence >= 3 ? (
        <circle cx={428 + travel * 84} cy={278} r="6" />
      ) : null}
      {sequence >= 4 ? (
        <circle cx={800 + travel * 78} cy={278} r="6" />
      ) : null}
      {sequence >= 5 ? (
        <circle cx={1247 + travel * 75} cy={278} r="6" />
      ) : null}
    </svg>
  );
};

export const VolvenceDigitalTwinStandalone: React.FC = () => {
  const frame = useCurrentFrame();
  const audioFrame = frame - DIGITAL_TWIN_INTRO_FRAMES;
  const audioTime = Math.max(0, audioFrame / DIGITAL_TWIN_FPS);
  const line = currentNarrationAt(audioTime);
  const introOpacity = interpolate(frame, [0, 22, 62, 90], [0, 1, 1, 0], clamp);
  const contentOpacity = interpolate(frame, [70, 106], [0, 1], clamp);
  const outroStart = DIGITAL_TWIN_INTRO_FRAMES + DIGITAL_TWIN_AUDIO_FRAMES;
  const outroOpacity = interpolate(
    frame,
    [outroStart, outroStart + 28],
    [0, 1],
    clamp,
  );
  const isCore = line.sequence >= 4;
  const isProof = line.sequence >= 5 && line.sequence <= 6;
  const isDeploy = line.sequence >= 7;

  return (
    <AbsoluteFill className="dt-film">
      <div className="dt-ambient dt-ambient-a" />
      <div className="dt-ambient dt-ambient-b" />
      <div className="dt-grid" />

      <div className="dt-brand">
        <span>V</span>
        <b>VOLVENCE</b>
        <i>数字分身</i>
      </div>
      <div className="dt-progress">
        {digitalTwinNarration.map((item) => (
          <span
            key={item.sequence}
            className={item.sequence === line.sequence ? 'active' : ''}
          />
        ))}
      </div>

      <div className="dt-intro" style={{opacity: introOpacity}}>
        <span>第五幕 · 数字分身</span>
        <strong>把一个人的一生，迁移成可以继续成长的个人模型</strong>
        <b>声音和形象让它像他；认知、记忆、科学观和经验，才让它是他</b>
      </div>

      <main className="dt-content" style={{opacity: contentOpacity}}>
        <div className="dt-heading">
          <span>{String(line.sequence).padStart(2, '0')}</span>
          <div>
            <b>{line.caption}</b>
            <i>
              {line.sequence <= 2
                ? '先回答：我们究竟要迁移什么'
                : line.sequence <= 4
                  ? '再回答：资料怎样变成一个可运行的人'
                  : line.sequence <= 6
                    ? '最后证明：为什么它真的是这个人'
                    : '上线运行，并在真实结果中继续成长'}
            </i>
          </div>
        </div>

        <div className={`dt-stage ${isDeploy ? 'deployment-mode' : ''}`}>
          <FlowLines sequence={line.sequence} frame={frame} />
          <SourceField active={line.sequence === 2 || line.sequence === 3} sequence={line.sequence} />
          <EvidenceCompiler active={line.sequence === 3} frame={frame} />
          <PersonCore active={isCore && !isDeploy} sequence={line.sequence} frame={frame} />
          <ProofPanel active={isProof} sequence={line.sequence} />
          <DeploymentStrip active={isDeploy} learning={line.sequence === 8} frame={frame} />
        </div>

        <div className={`dt-thesis ${line.sequence === 1 ? 'active' : ''}`}>
          <s>复刻表情、声音和口头禅</s>
          <i>≠</i>
          <b>迁移一个人的“为什么”</b>
        </div>
      </main>

      <div className="dt-outro" style={{opacity: outroOpacity}}>
        <span>VOLVENCE DIGITAL TWIN</span>
        <strong>不是保存一个人的过去</strong>
        <b>而是让他的认知、记忆、科学观与经验继续面对未来</b>
        <i>有来源 · 可验证 · 能行动 · 会继续学习</i>
      </div>

      <Sequence
        from={DIGITAL_TWIN_INTRO_FRAMES}
        durationInFrames={DIGITAL_TWIN_AUDIO_FRAMES}
      >
        <Audio
          src={staticFile('digital-twin-standalone/volvence-digital-twin-standalone.wav')}
          volume={1}
        />
      </Sequence>
    </AbsoluteFill>
  );
};
