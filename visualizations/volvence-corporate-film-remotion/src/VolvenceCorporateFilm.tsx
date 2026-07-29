import React from 'react';
import {
  AbsoluteFill,
  Audio,
  Img,
  Sequence,
  interpolate,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import {applications, captions, DURATION, FPS, partners} from './data';

export const CORPORATE_FILM_FPS = FPS;
export const CORPORATE_FILM_DURATION = DURATION;

const fade = (frame: number, duration: number) =>
  interpolate(frame, [0, 18, duration - 18, duration], [0, 1, 1, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

const rise = (frame: number, delay = 0) => {
  const value = spring({
    frame: frame - delay,
    fps: FPS,
    config: {damping: 18, stiffness: 88, mass: 0.8},
  });
  return {
    opacity: interpolate(value, [0, 1], [0, 1]),
    transform: `translateY(${interpolate(value, [0, 1], [34, 0])}px)`,
  };
};

const Brand: React.FC = () => (
  <div className="brand">
    <div className="brand-glyph">V</div>
    <span>VOLVENCE</span>
  </div>
);

const Progress: React.FC = () => {
  const frame = useCurrentFrame();
  const act = frame < 660 ? 0 : frame < 1380 ? 1 : frame < 2640 ? 2 : 3;
  return (
    <div className="progress">
      {['我们是谁', '核心产品', '产品价值', '合作伙伴'].map((label, index) => (
        <div className={`progress-item ${index === act ? 'active' : ''}`} key={label}>
          <span>{String(index + 1).padStart(2, '0')}</span>
          <i />
          <b>{label}</b>
        </div>
      ))}
    </div>
  );
};

const Caption: React.FC = () => {
  const frame = useCurrentFrame();
  const cue = captions.find((item) => frame >= item.start && frame < item.end);
  if (!cue) return null;
  const local = frame - cue.start;
  const duration = cue.end - cue.start;
  return (
    <div className="caption" style={{opacity: fade(local, duration)}}>
      <div>{cue.text}</div>
      {cue.accent ? <strong>{cue.accent}</strong> : null}
    </div>
  );
};

const SectionTitle: React.FC<{
  index: string;
  eyebrow: string;
  title: string;
  frame: number;
}> = ({index, eyebrow, title, frame}) => (
  <div className="section-title" style={rise(frame)}>
    <div className="section-index">{index}</div>
    <div>
      <div className="eyebrow">{eyebrow}</div>
      <h1>{title}</h1>
    </div>
  </div>
);

const BadgeCloud: React.FC<{frame: number}> = ({frame}) => {
  const schools = ['CMU', '北京大学', '清华大学', 'NYU'];
  const companies = ['字节跳动', '阿里巴巴', '腾讯', 'IBM'];
  return (
    <div className="badge-stage">
      <div className="badge-row schools">
        {schools.map((item, index) => (
          <div className="institution" key={item} style={rise(frame, 16 + index * 8)}>
            <span>{item.length <= 4 ? item : item.slice(0, 1)}</span>
            <b>{item}</b>
          </div>
        ))}
      </div>
      <div className="line-label">学术基因</div>
      <div className="badge-row companies">
        {companies.map((item, index) => (
          <div className="company" key={item} style={rise(frame, 52 + index * 7)}>
            {item}
          </div>
        ))}
      </div>
      <div className="line-label industry">产业履历</div>
    </div>
  );
};

const Metric: React.FC<{
  value: string;
  label: string;
  note: string;
  frame: number;
  delay: number;
}> = ({value, label, note, frame, delay}) => (
  <div className="metric" style={rise(frame, delay)}>
    <strong>{value}</strong>
    <b>{label}</b>
    <span>{note}</span>
  </div>
);

const TeamPart: React.FC = () => {
  const frame = useCurrentFrame();
  const imageOpacity = interpolate(frame, [180, 245, 625, 660], [0, 0.78, 0.58, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  return (
    <AbsoluteFill className="scene team-scene">
      <Img
        className="team-photo"
        src={staticFile('images/team-lab.png')}
        style={{
          opacity: imageOpacity,
          transform: `scale(${interpolate(frame, [180, 660], [1.04, 1.11])})`,
        }}
      />
      <div className="photo-shade" />
      <SectionTitle index="01" eyebrow="WHO WE ARE" title="一群长期主义者" frame={frame} />
      {frame < 335 ? <BadgeCloud frame={frame} /> : null}
      {frame >= 300 ? (
        <div className="team-statement" style={rise(frame, 300)}>
          <strong>20</strong>
          <div>
            <b>人复合型团队</b>
            <span>科研攻坚 · 技术落地 · 商业运营</span>
          </div>
        </div>
      ) : null}
      {frame >= 395 ? (
        <div className="metrics">
          <Metric value="40+" label="机器学习论文" note="长期学术积累" frame={frame} delay={410} />
          <Metric value="18" label="国际顶会顶刊" note="NeurIPS · ICML · AAAI · CVPR" frame={frame} delay={428} />
          <Metric value="TOP 10" label="全球主动学习" note="原创主动学徒学习方向" frame={frame} delay={446} />
        </div>
      ) : null}
    </AbsoluteFill>
  );
};

const Loop: React.FC<{compact?: boolean; frame: number}> = ({compact = false, frame}) => {
  const words = ['目标', '感知', '决策', '反馈'];
  const radius = compact ? 88 : 112;
  return (
    <div className={`loop ${compact ? 'compact' : ''}`}>
      <div className="loop-orbit" style={{transform: `rotate(${frame * 0.22}deg)`}} />
      {words.map((word, index) => {
        const angle = (Math.PI * 2 * index) / words.length - Math.PI / 2;
        return (
          <span
            key={word}
            style={{
              left: `calc(50% + ${Math.cos(angle) * radius}px)`,
              top: `calc(50% + ${Math.sin(angle) * radius}px)`,
            }}
          >
            {word}
          </span>
        );
      })}
      <b>闭环</b>
    </div>
  );
};

const Arrow: React.FC = () => (
  <div className="flow-arrow">
    <i />
    <span>›</span>
  </div>
);

const TraditionalModel: React.FC<{frame: number}> = ({frame}) => (
  <div className="compare-card traditional">
    <div className="compare-head">
      <span>传统大模型</span>
      <b>静态工具</b>
    </div>
    <div className="traditional-flow">
      <div className="model-block static-model">
        <i />
        <strong>语言模型</strong>
        <span>生成答案</span>
      </div>
      <Arrow />
      <div className="api-block">API</div>
      <Arrow />
      <Loop compact frame={frame} />
    </div>
    <div className="compare-foot">闭环在模型之外，由人完成</div>
  </div>
);

const VolvenceModel: React.FC<{frame: number}> = ({frame}) => (
  <div className="compare-card volvence-model">
    <div className="compare-head">
      <span>Volvence</span>
      <b>持续成长</b>
    </div>
    <div className="volvence-flow">
      <div className="stacked-model">
        <div className="relation-model">
          <div className="relation-label">关系与决策模型</div>
          <Loop compact frame={frame} />
        </div>
        <div className="base-model">底座模型 · 语言 / 知识 / 工具</div>
      </div>
      <Arrow />
      <div className="api-block active">API</div>
      <Arrow />
      <div className="raw-interaction">
        <div className="wave-line" />
        <strong>原始交互</strong>
        <span>自然对话 · 行动结果</span>
      </div>
    </div>
    <div className="compare-foot active">闭环进入模型，交互自然成为学习</div>
  </div>
);

const ProductPart: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill className="scene product-scene">
      <div className="neural-field">
        {Array.from({length: 28}).map((_, index) => (
          <i
            key={index}
            style={{
              left: `${8 + ((index * 31) % 86)}%`,
              top: `${9 + ((index * 47) % 78)}%`,
              opacity: 0.16 + 0.32 * Math.sin((frame + index * 9) / 30),
            }}
          />
        ))}
      </div>
      <SectionTitle index="02" eyebrow="THE MODEL" title="不是更会说，而是会成长" frame={frame} />
      {frame < 205 ? (
        <div
          className="api-hero"
          style={{
            ...rise(frame, 18),
            transform: `translateX(-50%) translateY(${
              interpolate(
                spring({
                  frame: frame - 18,
                  fps: FPS,
                  config: {damping: 18, stiffness: 88, mass: 0.8},
                }),
                [0, 1],
                [34, 0],
              )
            }px)`,
          }}
        >
          <div className="api-ring">
            <span>API</span>
          </div>
          <div>
            <h2>一个接口，接入数字生命</h2>
            <p>独立人格 · 长效记忆 · 情感感知 · 持续学习</p>
            <small>无需复杂上下文工程</small>
          </div>
        </div>
      ) : (
        <div className="compare" style={rise(frame, 205)}>
          <TraditionalModel frame={frame} />
          <VolvenceModel frame={frame} />
        </div>
      )}
      {frame > 535 ? (
        <div className="difference-strip">
          <div>
            <b>更聪明</b>
            <span>从交互与结果中找到关键因素</span>
          </div>
          <i />
          <div>
            <b>更人性</b>
            <span>拉近关系 · 建立边界 · 探索 · 推进</span>
          </div>
        </div>
      ) : null}
    </AbsoluteFill>
  );
};

const ApplicationsGrid: React.FC<{frame: number}> = ({frame}) => {
  const active = Math.min(8, Math.floor(Math.max(0, frame - 90) / 62));
  return (
    <div className="app-grid">
      {applications.map((app, index) => (
        <div
          className={`app-card ${index === active ? 'active' : ''} ${
            index < active ? 'visited' : ''
          }`}
          key={app.name}
          style={rise(frame, 36 + index * 4)}
        >
          <div className="app-mark">{app.mark}</div>
          <div>
            <b>{app.name}</b>
            <span>{app.note}</span>
          </div>
        </div>
      ))}
    </div>
  );
};

const HumanVision: React.FC<{frame: number}> = ({frame}) => {
  const statements = [
    ['每个家庭', '可靠的育儿、家庭与婚姻顾问'],
    ['每个孩子', '真正懂他的老师、朋友和家人'],
    ['每位青年', '知心朋友与契合的另一半'],
    ['每个创业者', '能创造收益的数字同事'],
  ];
  const active = Math.min(3, Math.floor(Math.max(0, frame - 680) / 115));
  return (
    <div className="human-vision">
      <Img
        className="human-photo"
        src={staticFile('images/human-life.png')}
        style={{
          transform: `scale(${interpolate(frame, [650, 1260], [1.03, 1.1])})`,
        }}
      />
      <div className="human-shade" />
      <div className="vision-copy">
        <div className="eyebrow">WHY IT MATTERS</div>
        <h2>让智能进入人的一生</h2>
        <div className="vision-statements">
          {statements.map(([lead, text], index) => (
            <div className={index === active ? 'active' : ''} key={lead}>
              <b>{lead}</b>
              <span>{text}</span>
            </div>
          ))}
        </div>
      </div>
      {frame > 1100 ? (
        <div
          className="remember-line"
          style={{
            ...rise(frame, 1100),
            transform: `translateX(-50%) translateY(${
              interpolate(
                spring({
                  frame: frame - 1100,
                  fps: FPS,
                  config: {damping: 18, stiffness: 88, mass: 0.8},
                }),
                [0, 1],
                [34, 0],
              )
            }px)`,
          }}
        >
          留存每一段人生，延续温暖与思考
        </div>
      ) : null}
    </div>
  );
};

const ValuePart: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill className="scene value-scene">
      <SectionTitle index="03" eyebrow="WHAT WE CREATE" title="新的产品形态" frame={frame} />
      {frame < 650 ? <ApplicationsGrid frame={frame} /> : <HumanVision frame={frame} />}
    </AbsoluteFill>
  );
};

const PartnerPart: React.FC = () => {
  const frame = useCurrentFrame();
  const final = frame > 410;
  return (
    <AbsoluteFill className="scene partner-scene">
      <div className="partner-glow" />
      {!final ? (
        <>
          <SectionTitle index="04" eyebrow="TRUSTED BY" title="与真实世界共同生长" frame={frame} />
          <div className="partner-orbit">
            <div className="orbit-core">
              <div className="brand-glyph large">V</div>
              <b>VOLVENCE</b>
            </div>
            {partners.map((partner, index) => {
              const angle = (Math.PI * 2 * index) / partners.length - Math.PI / 2;
              const radiusX = 530;
              const radiusY = 230;
              return (
                <div
                  className="partner-pill"
                  key={partner}
                  style={{
                    opacity: rise(frame, 40 + index * 11).opacity,
                    transform: `translate(-50%, -50%) translateY(${
                      interpolate(
                        spring({
                          frame: frame - (40 + index * 11),
                          fps: FPS,
                          config: {damping: 18, stiffness: 88, mass: 0.8},
                        }),
                        [0, 1],
                        [34, 0],
                      )
                    }px)`,
                    left: `calc(50% + ${Math.cos(angle) * radiusX}px)`,
                    top: `calc(50% + ${Math.sin(angle) * radiusY}px)`,
                  }}
                >
                  {partner}
                </div>
              );
            })}
          </div>
        </>
      ) : (
        <div className="final-lockup" style={rise(frame, 410)}>
          <div className="final-symbol">
            <div className="final-orbit one" />
            <div className="final-orbit two" />
            <div className="brand-glyph hero">V</div>
          </div>
          <h1>Volvence</h1>
          <p>模型进化，生命涌现</p>
          <i />
          <strong>每个人，都值得被记住</strong>
        </div>
      )}
    </AbsoluteFill>
  );
};

export const VolvenceCorporateFilm: React.FC = () => {
  const frame = useCurrentFrame();
  const {durationInFrames} = useVideoConfig();
  return (
    <AbsoluteFill className="film">
      <div className="grain" />
      <div className="ambient ambient-a" />
      <div className="ambient ambient-b" />
      <Sequence from={0} durationInFrames={660} premountFor={30}>
        <TeamPart />
      </Sequence>
      <Sequence from={660} durationInFrames={720} premountFor={30}>
        <ProductPart />
      </Sequence>
      <Sequence from={1380} durationInFrames={1260} premountFor={30}>
        <ValuePart />
      </Sequence>
      <Sequence from={2640} durationInFrames={660} premountFor={30}>
        <PartnerPart />
      </Sequence>
      <Audio src={staticFile('voiceover/volvence-corporate-film-voiceover.wav')} />
      <Audio
        src={staticFile('music/volvence-corporate-bed.wav')}
        volume={(currentFrame) =>
          interpolate(
            currentFrame,
            [0, 60, 1200, 2250, durationInFrames - 120, durationInFrames],
            [0, 0.14, 0.11, 0.16, 0.16, 0],
            {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'},
          )
        }
      />
      <Brand />
      <Progress />
      <Caption />
      <div className="time-line">
        <i style={{width: `${(frame / durationInFrames) * 100}%`}} />
      </div>
    </AbsoluteFill>
  );
};
